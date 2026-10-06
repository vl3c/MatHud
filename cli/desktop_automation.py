"""Drive the running desktop window from the CLI: send a prompt, read the canvas, take a screenshot.

The window must have been opened with ``python mathud_desktop.py
--automation-port N``; everything here goes through ``cli.cdp.CDPBrowser`` and
the app's own hooks, so a prompt appears in the window's chat exactly as if the
user had typed it, and its reply and canvas changes show up there too.

Spend guard: the desktop app runs with the user's own ``.env`` (real API keys),
so a prompt is always sent with an explicit model id that ``/api/available_models``
lists under the requested provider. Without ``--model`` the single local model
is used; the window's current selection is never used, since it may be a paid model.
A paid provider also needs the app to search tools locally and OpenRouter not to
retry (``/api/automation_settings``), since those requests are not counted.
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


def validate_prompt(text: str) -> None:
    """Refuse an empty prompt or a slash command (those run in the window without the model)."""
    if not text.strip():
        raise DesktopError("the prompt is empty")
    if text.lstrip().startswith("/"):
        raise DesktopError("slash commands run in the window without the model; type them there")


def automation_settings(base_url: str, timeout: float = 30.0) -> dict[str, Any]:
    """``GET /api/automation_settings``: the server's effective tool-search mode and OpenRouter retries."""
    import requests

    with requests.Session() as session:
        session.trust_env = False
        response = session.get(f"{base_url}/api/automation_settings", timeout=timeout)
        response.raise_for_status()
        data = response.json().get("data")
    if not isinstance(data, dict):
        raise DesktopError(f"unexpected reply from {base_url}/api/automation_settings")
    return data


PAID_PROVIDER_WARNING = (
    "Warning: this uses a paid provider through the desktop app, which runs with its own .env. "
    "Every request it sends is billed; it is refused unless the app's TOOL_SEARCH_MODE is local and "
    "MATHUD_OPENROUTER_MAX_RETRIES is 0, since API tool searches and retries are requests no cap counts."
)


def check_paid_provider_settings(settings: Optional[dict[str, Any]]) -> None:
    """Refuse a paid provider unless the app searches tools locally and OpenRouter does not retry.

    An ``api`` or ``hybrid`` tool search, and SDK retries, send paid requests that
    no request cap counts. ``settings`` is ``automation_settings``'s reply, or None
    when the server could not report them (an older app), which is refused too.
    """
    if settings is None:
        raise GuardError(
            "the desktop app does not report its tool-search mode and OpenRouter retries "
            "(GET /api/automation_settings); update it before using a paid provider"
        )
    problems = []
    if settings.get("tool_search_mode") != "local":
        problems.append(f"TOOL_SEARCH_MODE is {settings.get('tool_search_mode')!r}, not 'local'")
    if settings.get("openrouter_max_retries") != 0:
        problems.append(f"MATHUD_OPENROUTER_MAX_RETRIES is {settings.get('openrouter_max_retries')!r}, not 0")
    if problems:
        raise GuardError(
            "the desktop app would send paid requests no cap counts: "
            + "; ".join(problems)
            + ". Set them in its .env and restart it"
        )


def paid_provider_guard(base_url: str, provider: str) -> Optional[dict[str, Any]]:
    """For a paid provider, read the app's settings and refuse unsafe ones; returns them (None for local)."""
    if provider == "local":
        return None
    try:
        settings: Optional[dict[str, Any]] = automation_settings(base_url)
    except Exception:
        settings = None
    check_paid_provider_settings(settings)
    return settings


_MODEL_LISTED_JS = """
var wanted = arguments[0];
var selector = document.getElementById('ai-model-selector');
return !!selector && Array.prototype.some.call(selector.options, function (o) { return o.value === wanted; });
"""
_REFRESH_MODELS_JS = "return window.refreshMatHudModels ? window.refreshMatHudModels() : false;"


def ensure_model_listed(browser: CDPBrowser, model: str) -> None:
    """Make sure the window's model dropdown offers ``model``, refreshing the list once if not.

    The page lists models when it loads; a window opened before llama-server
    started lacks the local model the server now registers.
    """
    if browser.execute_js(_MODEL_LISTED_JS, model) is True:
        return
    browser.execute_js(_REFRESH_MODELS_JS, timeout=60)
    if browser.execute_js(_MODEL_LISTED_JS, model) is not True:
        raise DesktopError(f"the window's model list does not offer {model}, even after refreshing it")


def stop_turn(browser: CDPBrowser) -> None:
    """Stop the window's running turn, if any (after Ctrl+C); errors are ignored."""
    try:
        browser.call_hook("stopMatHudTurn")
    except Exception:
        pass


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
    validate_prompt(text)
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
