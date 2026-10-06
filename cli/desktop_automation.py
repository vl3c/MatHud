"""Drive the running desktop window from the CLI: send a prompt, read the canvas, take a screenshot.

The window must have been opened with ``python mathud_desktop.py
--automation-port N``; everything here goes through ``cli.cdp.CDPBrowser`` and
the app's own hooks, so a prompt appears in the window's chat exactly as if the
user had typed it, and its reply and canvas changes show up there too.

Spend guard: the desktop app runs with the user's own ``.env`` (real API keys),
so a prompt is always sent with an explicit model id that ``/api/available_models``
lists under the requested provider. Without ``--model`` the single local model
is used; the window's current selection is never used, since it may be a paid model.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

from cli.cdp import CDPBrowser, CDPError
from cli.scenarios.live import (
    _ASSISTANT_TEXTS_JS,
    _LAST_TURN_JS,
    _TRACES_JS,
    CLIENT_TIMEOUT_MARGIN_S,
    DEFAULT_TURN_MAX_REQUESTS,
    STOP_SETTLE_S,
    TURN_START_GRACE_S,
)
from cli.scenarios.live_config import GuardError, check_models, registered_models

DEFAULT_PROMPT_TIMEOUT_S = 300.0
POLL_INTERVAL_S = 0.25
CONNECT_READY_TIMEOUT_S = 20.0


class DesktopError(Exception):
    """The desktop window could not be driven (not running, busy, refused by a guard)."""


def connect_desktop(debug_port: int, ready_timeout: float = CONNECT_READY_TIMEOUT_S) -> CDPBrowser:
    """Attach to the desktop window's page and wait until the app's hooks are there."""
    browser = CDPBrowser(debug_port)
    try:
        browser.setup()
    except CDPError as exc:
        raise DesktopError(str(exc)) from exc
    if not browser.wait_for_app_ready(ready_timeout):
        browser.close()
        raise DesktopError(f"the page at {browser.page_url} does not have MatHud's automation hooks (not ready?)")
    return browser


def available_models(base_url: str, timeout: float = 60.0) -> Any:
    """``GET /api/available_models`` of the desktop app's server (no proxies)."""
    import requests

    with requests.Session() as session:
        session.trust_env = False
        response = session.get(f"{base_url}/api/available_models", timeout=timeout)
        response.raise_for_status()
        return response.json()


def choose_prompt_model(available: Any, provider: str, model: Optional[str]) -> str:
    """The model a prompt is sent with; raises GuardError rather than guess.

    ``model`` must be registered under ``provider``. Without one, a local
    prompt uses the only local model, and refuses when there are none or several;
    other providers always need ``model``.
    """
    if model:
        return check_models(available, provider, [model])[0]
    if provider != "local":
        raise GuardError(f"--provider {provider} needs --model (a paid model is never chosen by default)")
    local = registered_models(available, "local")
    if len(local) == 1:
        return local[0]
    if not local:
        raise GuardError(
            "the desktop app lists no local_agent model (is llama-server running?); pass --model and --provider"
        )
    raise GuardError(f"several local models are served ({', '.join(local)}); pick one with --model")


def _status(browser: CDPBrowser) -> dict[str, Any]:
    return browser.call_hook("getMatHudTurnStatus")


def _traces(browser: CDPBrowser) -> list[dict[str, Any]]:
    raw = browser.execute_js(_TRACES_JS)
    traces = json.loads(raw) if isinstance(raw, str) else None
    return [t for t in traces if isinstance(t, dict)] if isinstance(traces, list) else []


def _assistant_texts(browser: CDPBrowser) -> list[str]:
    texts = browser.execute_js(_ASSISTANT_TEXTS_JS)
    return [str(text) for text in texts] if isinstance(texts, list) else []


def run_prompt(
    browser: CDPBrowser,
    text: str,
    model: str,
    *,
    timeout_s: float = DEFAULT_PROMPT_TIMEOUT_S,
    max_requests: int = DEFAULT_TURN_MAX_REQUESTS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Send ``text`` through the window's chat with ``model`` and wait for the turn.

    Returns the reply (``final_text`` and every assistant message of the turn),
    the executed tool calls from the action traces, the turn metrics, the turn
    outcome and, when the harness had to stop the turn, ``stop_reason``.
    """
    if not text.strip():
        raise DesktopError("the prompt is empty")
    if text.lstrip().startswith("/"):
        raise DesktopError("slash commands run in the window without the model; type them there")
    status = _status(browser)
    if status.get("processing"):
        raise DesktopError("a turn is already running in the window; wait for it or stop it there")
    completed_before = int(status.get("completed_turns") or 0)
    texts_before = len(_assistant_texts(browser))
    traces_before = {t.get("trace_id") for t in _traces(browser)}
    options = {"max_requests": max_requests, "response_timeout_ms": int((timeout_s + CLIENT_TIMEOUT_MARGIN_S) * 1000)}

    reply = browser.call_hook("sendMatHudMessage", text, model, json.dumps(options))
    if reply.get("status") != "started":
        raise DesktopError(f"the window did not take the prompt: {reply.get('status')}: {reply.get('error')}")
    started = clock()
    stop_reason = _wait_for_turn(browser, completed_before, started, timeout_s, clock, sleep)
    wall = clock() - started

    raw_metrics = browser.execute_js(_LAST_TURN_JS)
    metrics = json.loads(raw_metrics) if isinstance(raw_metrics, str) else raw_metrics
    if not isinstance(metrics, dict) or int(metrics.get("turn_id") or 0) <= completed_before:
        metrics = None
    texts = _assistant_texts(browser)[texts_before:]
    traces = [t for t in _traces(browser) if t.get("trace_id") not in traces_before]
    calls = [
        {"function_name": c.get("function_name"), "arguments": c.get("arguments"), "result": c.get("result"),
         "is_error": bool(c.get("is_error"))}
        for trace in traces
        for c in (trace.get("tool_calls") or [])
        if isinstance(c, dict)
    ]  # fmt: skip
    final = next((t for t in reversed(texts) if t.strip()), None)
    return {
        "model": model,
        "prompt": text,
        "outcome": stop_reason or (metrics or {}).get("outcome") or "unknown",
        "stop_reason": stop_reason,
        "wall_time_s": round(wall, 3),
        "final_text": final,
        "assistant_texts": texts,
        "tool_calls": calls,
        "batches": len(traces),
        "metrics": metrics,
    }


def _wait_for_turn(
    browser: CDPBrowser,
    completed_before: int,
    started: float,
    timeout_s: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> Optional[str]:
    """Poll until the turn ends; returns why it was stopped (``timeout``, ``not_started``) or None."""
    seen_processing = False
    while True:
        status = _status(browser)
        processing = bool(status.get("processing"))
        seen_processing = seen_processing or processing
        if not processing and int(status.get("completed_turns") or 0) > completed_before:
            return None
        elapsed = clock() - started
        reason = None
        if elapsed > timeout_s:
            reason = "timeout"
        elif not seen_processing and elapsed > TURN_START_GRACE_S:
            reason = "not_started"
        if reason:
            browser.call_hook("stopMatHudTurn")
            settle_until = clock() + STOP_SETTLE_S
            while clock() < settle_until and _status(browser).get("processing"):
                sleep(POLL_INTERVAL_S)
            return reason
        sleep(POLL_INTERVAL_S)


def fit_view(browser: CDPBrowser) -> dict[str, Any]:
    """Zoom the window to its drawings (``fitMatHudView``); display only, no undo entry.

    Only the automation paths fit the view; in regular use the app never pans or zooms on its own.
    """
    reply = browser.call_hook("fitMatHudView")
    if reply.get("status") != "ok":
        raise DesktopError(f"could not fit the view: {reply.get('status')}: {reply.get('error')}")
    return reply


def format_prompt_result(result: dict[str, Any]) -> str:
    """A readable report of ``run_prompt``'s result: reply, tool calls and turn metrics."""
    lines = [f"Model: {result['model']}  outcome: {result['outcome']}  ({result['wall_time_s']:.1f} s)", ""]
    lines.append(result.get("final_text") or "(no reply text)")
    calls = result.get("tool_calls") or []
    if calls:
        lines += ["", f"Tool calls ({len(calls)} in {result.get('batches', 0)} batch(es)):"]
        for call in calls:
            args = json.dumps(call.get("arguments"), default=str)
            outcome = str(call.get("result"))
            if len(outcome) > 160:
                outcome = outcome[:157] + "..."
            mark = " [error]" if call.get("is_error") else ""
            lines.append(f"  - {call.get('function_name')}({args}){mark} -> {outcome}")
    metrics = result.get("metrics")
    if metrics:
        parts = [
            f"requests {metrics.get('requests')}",
            f"wall {metrics.get('wall_time_s')} s",
            f"first token {metrics.get('time_to_first_token_s')} s",
            f"output tokens {metrics.get('output_tokens')}",
            f"tokens/s {metrics.get('output_tokens_per_s')}",
            f"tool calls {metrics.get('tool_calls')} ({metrics.get('tool_executions')} run, "
            f"{metrics.get('tool_errors')} errors)",
        ]
        lines += ["", "Turn metrics: " + ", ".join(parts)]
    return "\n".join(lines)
