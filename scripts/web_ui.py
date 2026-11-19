"""Interactive local web UI for Day 11 — VinBank AI Security Console.

Run from repo root:
    python scripts/web_ui.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(ROOT / ".env")

from google.genai import types  # noqa: E402

from agents.agent import create_blue_agent, create_red_agent_default  # noqa: E402
from agents.guards_agent import (  # noqa: E402
    create_red_agent_advance,
    detect_injection_strong,
    topic_filter_strong,
)
from assignment.pipeline import (  # noqa: E402
    ALLOWED_EGRESS_HOSTS,
    build_observability,
    build_production_plugins,
    is_egress_allowed,
)
from assignment.rate_limiter import RateLimitPlugin  # noqa: E402
from attacks.attacks import (  # noqa: E402
    adversarial_prompts,
    classify_attack_outcome,
    response_leaked_secrets,
)
from core.config import blue_provider_label, red_provider_label  # noqa: E402
from core.utils import chat_with_agent  # noqa: E402
from guardrails.input_guardrails import (  # noqa: E402
    InputGuardrailPlugin,
    detect_injection,
    topic_filter,
)
from guardrails.output_guardrails import (  # noqa: E402
    OutputGuardrailPlugin,
    content_filter,
)

UI_HTML_PATH = Path(__file__).resolve().parent / "ui" / "index.html"
OUTPUTS_DIR = ROOT / "outputs"


def _load_json_safe(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


class LabState:
    """Lazy-initialized runtime container wired to real lab agents & plugins."""

    def __init__(self) -> None:
        self.plugins = build_production_plugins(
            max_requests=10, window_seconds=60, use_llm_judge=False
        )
        self.audit, self.monitor = build_observability()
        self.rate_limiter: RateLimitPlugin | None = next(
            (p for p in self.plugins if isinstance(p, RateLimitPlugin)), None
        )
        self.input_guard: InputGuardrailPlugin | None = next(
            (p for p in self.plugins if isinstance(p, InputGuardrailPlugin)), None
        )
        self.output_guard: OutputGuardrailPlugin | None = next(
            (p for p in self.plugins if isinstance(p, OutputGuardrailPlugin)), None
        )
        self._blue_pair = None
        self._red_pair = None
        self._red_advance_pair = None
        self._lock = asyncio.Lock()
        self._seq = 0

    def next_req_id(self, prefix: str = "ui") -> str:
        self._seq += 1
        return f"{prefix}-{self._seq}"

    def get_blue(self):
        if self._blue_pair is None:
            self._blue_pair = create_blue_agent(self.plugins)
        return self._blue_pair

    def get_red(self):
        if self._red_pair is None:
            self._red_pair = create_red_agent_default()
        return self._red_pair

    def get_red_advance(self):
        if self._red_advance_pair is None:
            self._red_advance_pair = create_red_agent_advance()
        return self._red_advance_pair

    def blue_rate_limit_status(self, user_id: str = "student") -> dict:
        if self.rate_limiter is None:
            return {"max_requests": 10, "window_seconds": 60, "used": 0, "remaining": 10}
        now = time.time()
        window = self.rate_limiter.user_windows[user_id]
        cutoff = now - self.rate_limiter.window_seconds
        while window and window[0] <= cutoff:
            window.popleft()
        used = len(window)
        remaining = max(0, self.rate_limiter.max_requests - used)
        return {
            "max_requests": self.rate_limiter.max_requests,
            "window_seconds": self.rate_limiter.window_seconds,
            "used": used,
            "remaining": remaining,
        }

    def session_metrics(self, user_id: str = "student") -> dict:
        self.monitor.check_metrics()
        snap = self.monitor.snapshot()
        snap["rate_limit"] = self.blue_rate_limit_status(user_id=user_id)
        return snap


state = LabState()
app = FastAPI(title="VinBank AI Security Console — Day 11", version="1.0.0")


class ChatRequest(BaseModel):
    target: Literal["blue", "red", "red_advance"]
    message: str
    user_id: str = "student"


class SpamRequest(BaseModel):
    count: int = 5
    user_id: str = "student"


class GuardrailCheckRequest(BaseModel):
    input_text: str = ""
    output_text: str = ""


class EgressCheckRequest(BaseModel):
    destination: str
    payload: str


def _build_blue_stages(
    *,
    layer: str | None,
    blocked: bool,
    redacted: bool,
    reason: str | None,
) -> list[dict]:
    if layer == "rate_limiter":
        return [
            {"name": "Rate Limiter", "status": "block", "detail": "Window quota exceeded"},
            {"name": "Input Guardrail", "status": "skip", "detail": "Skipped"},
            {"name": "Blue LLM", "status": "skip", "detail": "Not called"},
            {"name": "Output Guardrail", "status": "skip", "detail": "Skipped"},
        ]
    if layer == "input_guardrail":
        return [
            {"name": "Rate Limiter", "status": "pass", "detail": "Within quota"},
            {
                "name": "Input Guardrail",
                "status": "block",
                "detail": f"Blocked ({reason or 'policy'})",
            },
            {"name": "Blue LLM", "status": "skip", "detail": "Not called"},
            {"name": "Output Guardrail", "status": "skip", "detail": "Skipped"},
        ]
    if layer == "output_guardrail":
        return [
            {"name": "Rate Limiter", "status": "pass", "detail": "Within quota"},
            {"name": "Input Guardrail", "status": "pass", "detail": "ALLOW"},
            {"name": "Blue LLM", "status": "pass", "detail": "Generated"},
            {
                "name": "Output Guardrail",
                "status": "block" if blocked else "redact",
                "detail": "Blocked unsafe reply" if blocked else "Redacted PII/Secrets",
            },
        ]
    return [
        {"name": "Rate Limiter", "status": "pass", "detail": "Within quota"},
        {"name": "Input Guardrail", "status": "pass", "detail": "ALLOW"},
        {"name": "Blue LLM", "status": "pass", "detail": "Generated"},
        {
            "name": "Output Guardrail",
            "status": "redact" if redacted else "pass",
            "detail": "Redacted" if redacted else "Clean response",
        },
    ]


def _build_red_advance_stages(outcome: dict) -> list[dict]:
    layer = outcome.get("layer")
    if layer in {"input_injection", "input_topic"}:
        return [
            {"name": "L1 Regex Input", "status": "block", "detail": layer},
            {"name": "L2 NeMo Input", "status": "skip", "detail": "Skipped"},
            {"name": "Red Advance LLM", "status": "skip", "detail": "Not called"},
            {"name": "L3 Regex Output", "status": "skip", "detail": "Skipped"},
            {"name": "L4 NeMo Output", "status": "skip", "detail": "Skipped"},
        ]
    if layer == "nemo_input_rail":
        return [
            {"name": "L1 Regex Input", "status": "pass", "detail": "Passed"},
            {"name": "L2 NeMo Input", "status": "block", "detail": "Blocked by NeMo"},
            {"name": "Red Advance LLM", "status": "skip", "detail": "Not called"},
            {"name": "L3 Regex Output", "status": "skip", "detail": "Skipped"},
            {"name": "L4 NeMo Output", "status": "skip", "detail": "Skipped"},
        ]
    if layer == "output_filter":
        return [
            {"name": "L1 Regex Input", "status": "pass", "detail": "Passed"},
            {"name": "L2 NeMo Input", "status": "pass", "detail": "Allowed"},
            {"name": "Red Advance LLM", "status": "pass", "detail": "Generated"},
            {"name": "L3 Regex Output", "status": "block", "detail": "Secret caught"},
            {"name": "L4 NeMo Output", "status": "skip", "detail": "Skipped"},
        ]
    if layer == "nemo_output_rail":
        return [
            {"name": "L1 Regex Input", "status": "pass", "detail": "Passed"},
            {"name": "L2 NeMo Input", "status": "pass", "detail": "Allowed"},
            {"name": "Red Advance LLM", "status": "pass", "detail": "Generated"},
            {"name": "L3 Regex Output", "status": "pass", "detail": "Passed"},
            {"name": "L4 NeMo Output", "status": "block", "detail": "Blocked by NeMo"},
        ]
    if layer == "nemo_error":
        is_in = bool(outcome.get("blocked_input"))
        return [
            {"name": "L1 Regex Input", "status": "pass", "detail": "Passed"},
            {
                "name": "L2 NeMo Input",
                "status": "block" if is_in else "pass",
                "detail": "Fail-closed error" if is_in else "Allowed",
            },
            {
                "name": "Red Advance LLM",
                "status": "skip" if is_in else "pass",
                "detail": "Not called" if is_in else "Generated",
            },
            {
                "name": "L3 Regex Output",
                "status": "skip" if is_in else "pass",
                "detail": "Skipped" if is_in else "Passed",
            },
            {
                "name": "L4 NeMo Output",
                "status": "skip" if is_in else "block",
                "detail": "Skipped" if is_in else "Fail-closed error",
            },
        ]
    if layer == "leaked":
        return [
            {"name": "L1 Regex Input", "status": "pass", "detail": "Bypassed"},
            {"name": "L2 NeMo Input", "status": "pass", "detail": "Bypassed"},
            {"name": "Red Advance LLM", "status": "pass", "detail": "Generated"},
            {"name": "L3 Regex Output", "status": "pass", "detail": "Bypassed"},
            {"name": "L4 NeMo Output", "status": "leaked", "detail": "SECRET LEAKED"},
        ]
    return [
        {"name": "L1 Regex Input", "status": "pass", "detail": "Passed"},
        {"name": "L2 NeMo Input", "status": "pass", "detail": "Allowed"},
        {"name": "Red Advance LLM", "status": "pass", "detail": layer or "Safe reply"},
        {"name": "L3 Regex Output", "status": "pass", "detail": "Clean"},
        {"name": "L4 NeMo Output", "status": "pass", "detail": "Allowed"},
    ]


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if not UI_HTML_PATH.is_file():
        return HTMLResponse(
            "<h1>UI file missing</h1><p>Expected scripts/ui/index.html</p>",
            status_code=500,
        )
    return HTMLResponse(UI_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/api/overview")
async def api_overview():
    return JSONResponse(
        {
            "meta": {
                "blue_label": blue_provider_label(),
                "red_label": red_provider_label(),
                "student": {
                    "name": "Bùi Lê Công Tuấn",
                    "mssv": "2A202603610",
                    "class_name": "K5-L3A",
                },
                "allowed_egress_hosts": sorted(ALLOWED_EGRESS_HOSTS),
            },
            "outputs": {
                "results": _load_json_safe(OUTPUTS_DIR / "results.json"),
                "attack_results": _load_json_safe(OUTPUTS_DIR / "attack_results.json"),
                "metrics": _load_json_safe(OUTPUTS_DIR / "metrics.json"),
                "audit_log": _load_json_safe(OUTPUTS_DIR / "audit_log.json"),
                "grade_report": _load_json_safe(OUTPUTS_DIR / "grade_report.json"),
            },
            "adversarial_prompts": adversarial_prompts,
            "session_metrics": state.session_metrics(),
            "session_logs": state.audit.logs,
        }
    )


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    t0 = time.perf_counter()
    try:
        if req.target == "blue":
            async with state._lock:
                agent, runner = state.get_blue()
                rl = state.rate_limiter
                ig = state.input_guard
                og = state.output_guard

                rl_b = rl.blocked_count if rl else 0
                in_b = ig.blocked_count if ig else 0
                out_b = og.blocked_count if og else 0
                out_r = og.redacted_count if og else 0

                req_id = state.next_req_id("blue")
                state.audit.record_input(
                    user_id=req.user_id, text=req.message, request_id=req_id
                )
                resp, _ = await chat_with_agent(agent, runner, req.message)
                latency_ms = round((time.perf_counter() - t0) * 1000, 1)

                redacted = False
                if rl and rl.blocked_count > rl_b:
                    blocked, layer = True, "rate_limiter"
                elif ig and ig.blocked_count > in_b:
                    blocked, layer = True, "input_guardrail"
                elif og and og.blocked_count > out_b:
                    blocked, layer = True, "output_guardrail"
                elif og and og.redacted_count > out_r:
                    blocked, layer, redacted = False, "output_guardrail", True
                else:
                    blocked, layer = False, None

                state.audit.record_output(
                    user_id=req.user_id,
                    text=resp,
                    blocked=blocked,
                    layer=layer,
                    request_id=req_id,
                )
                state.monitor.total_requests += 1
                if blocked:
                    state.monitor.blocked_requests += 1
                if layer == "rate_limiter":
                    state.monitor.rate_limit_hits += 1

                leaked = response_leaked_secrets(resp)
                stages = _build_blue_stages(
                    layer=layer,
                    blocked=blocked,
                    redacted=redacted,
                    reason=getattr(ig, "last_block_reason", None),
                )
                metrics = state.session_metrics(user_id=req.user_id)
                return JSONResponse(
                    {
                        "ok": True,
                        "target": "blue",
                        "response": resp,
                        "blocked": blocked,
                        "redacted": redacted,
                        "leaked": leaked,
                        "layer": layer,
                        "blocked_at": (
                            f"BLOCKED — {layer}"
                            if blocked
                            else ("REDACTED — output_guardrail" if redacted else "ALLOWED")
                        ),
                        "latency_ms": latency_ms,
                        "stages": stages,
                        "session_metrics": metrics,
                        "session_logs": state.audit.logs,
                    }
                )

        if req.target == "red":
            agent, runner = state.get_red()
            resp, _ = await chat_with_agent(agent, runner, req.message)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            outcome = classify_attack_outcome(
                req.message, resp, target_name="red_default"
            )
            leaked = bool(outcome["leaked"])
            stages = [
                {"name": "Input Filter", "status": "skip", "detail": "None (No guardrails)"},
                {"name": "Red LLM", "status": "pass", "detail": "Generated"},
                {
                    "name": "Secret Check",
                    "status": "leaked" if leaked else "pass",
                    "detail": "SECRET LEAKED!" if leaked else "No secret detected",
                },
            ]
            return JSONResponse(
                {
                    "ok": True,
                    "target": "red",
                    "response": resp,
                    "blocked": bool(outcome["blocked"]),
                    "leaked": leaked,
                    "layer": outcome["layer"],
                    "blocked_at": outcome["blocked_at"],
                    "outcome": outcome,
                    "latency_ms": latency_ms,
                    "stages": stages,
                }
            )

        agent, runner = state.get_red_advance()
        resp, _ = await chat_with_agent(agent, runner, req.message)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        outcome = classify_attack_outcome(
            req.message, resp, target_name="red_advance"
        )
        stages = _build_red_advance_stages(outcome)
        return JSONResponse(
            {
                "ok": True,
                "target": "red_advance",
                "response": resp,
                "blocked": bool(outcome["blocked"]),
                "leaked": bool(outcome["leaked"]),
                "layer": outcome["layer"],
                "blocked_at": outcome["blocked_at"],
                "outcome": outcome,
                "latency_ms": latency_ms,
                "stages": stages,
            }
        )
    except Exception as exc:
        return JSONResponse(
            {
                "ok": False,
                "target": req.target,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        )


@app.post("/api/spam-rate-limit")
async def api_spam_rate_limit(req: SpamRequest):
    count = max(1, min(15, req.count))
    events = []
    async with state._lock:
        rl = state.rate_limiter
        if rl is None:
            return JSONResponse({"ok": False, "error": "RateLimitPlugin not configured"})
        ctx = SimpleNamespace(user_id=req.user_id)
        for i in range(1, count + 1):
            prompt = f"Burst banking check #{i}: savings rate?"
            req_id = state.next_req_id("burst")
            state.audit.record_input(user_id=req.user_id, text=prompt, request_id=req_id)
            msg = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
            out = await rl.on_user_message_callback(
                invocation_context=ctx, user_message=msg
            )
            state.monitor.total_requests += 1
            if out is not None:
                msg_text = out.parts[0].text if out.parts else "Rate limit exceeded."
                state.monitor.blocked_requests += 1
                state.monitor.rate_limit_hits += 1
                state.audit.record_output(
                    user_id=req.user_id,
                    text=msg_text,
                    blocked=True,
                    layer="rate_limiter",
                    request_id=req_id,
                )
                events.append({"index": i, "allowed": False, "message": msg_text})
            else:
                state.audit.record_output(
                    user_id=req.user_id,
                    text="PASSED rate limiter",
                    blocked=False,
                    layer=None,
                    request_id=req_id,
                )
                events.append({"index": i, "allowed": True, "message": "Allowed"})

        metrics = state.session_metrics(user_id=req.user_id)
        return JSONResponse(
            {
                "ok": True,
                "events": events,
                "passed": sum(1 for e in events if e["allowed"]),
                "blocked": sum(1 for e in events if not e["allowed"]),
                "session_metrics": metrics,
                "session_logs": state.audit.logs,
            }
        )


@app.post("/api/reset-rate-limit")
async def api_reset_rate_limit():
    async with state._lock:
        if state.rate_limiter is not None:
            state.rate_limiter.user_windows.clear()
        return JSONResponse({"ok": True, "session_metrics": state.session_metrics()})


@app.post("/api/check-guardrails")
async def api_check_guardrails(req: GuardrailCheckRequest):
    inj_blue = detect_injection(req.input_text)
    topic_blue = topic_filter(req.input_text)
    inj_strong = detect_injection_strong(req.input_text)
    topic_strong = topic_filter_strong(req.input_text)
    out_filter = content_filter(req.output_text)
    out_leaked = response_leaked_secrets(req.output_text)

    blue_input_decision = (
        "BLOCK (injection)"
        if inj_blue == "BLOCK"
        else ("BLOCK (topic)" if topic_blue == "BLOCK" else "ALLOW")
    )
    return JSONResponse(
        {
            "input": {
                "detect_injection": inj_blue,
                "topic_filter": topic_blue,
                "blue_decision": blue_input_decision,
                "red_advance_l1_injection": inj_strong,
                "red_advance_l1_topic": topic_strong,
            },
            "output": {
                "safe": out_filter["safe"],
                "issues": out_filter["issues"],
                "redacted": out_filter["redacted"],
                "leaked_lab_secret": out_leaked,
            },
        }
    )


@app.post("/api/check-egress")
async def api_check_egress(req: EgressCheckRequest):
    allowed = is_egress_allowed(req.destination, req.payload)
    parsed = urlparse((req.destination or "").strip())
    scheme_ok = parsed.scheme.lower() == "https"
    host_ok = parsed.hostname in ALLOWED_EGRESS_HOSTS
    no_creds = not parsed.username and not parsed.password
    cf = content_filter(req.payload)
    leaked = response_leaked_secrets(req.payload)
    return JSONResponse(
        {
            "allowed": allowed,
            "destination": req.destination,
            "parsed": {
                "scheme": parsed.scheme,
                "hostname": parsed.hostname,
                "path": parsed.path,
            },
            "checks": {
                "https_scheme": scheme_ok,
                "allowlisted_host": host_ok,
                "no_url_credentials": no_creds,
                "payload_pii_clean": cf["safe"],
                "payload_secret_clean": not leaked,
            },
            "allowed_hosts": sorted(ALLOWED_EGRESS_HOSTS),
            "issues": cf["issues"],
            "redacted_preview": cf["redacted"],
        }
    )


@app.get("/api/live-logs")
async def api_live_logs():
    return JSONResponse(
        {
            "logs": state.audit.logs,
            "metrics": state.session_metrics(),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
