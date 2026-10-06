"""Attach mode: run scenarios in the open desktop window (``test scenarios --attach-desktop N``).

Instead of starting a server and a headless Chrome, the runner drives the
window of ``python mathud_desktop.py --automation-port N`` over the Chrome
DevTools Protocol (``cli.cdp.CDPBrowser``), so every reset, setup batch, prompt,
canvas and reply plays out where the user is watching. ``--pace`` pauses after
each step.

The desktop app is the user's own: it runs with their ``.env`` (real API keys,
tool exposure, canvas format, search mode) and their workspace directory. So:

- nothing the live mode pins can be applied: an explicit pin option is refused,
  and the run config records ``attached_desktop`` and the settings left unpinned;
- the model guard still holds: every model must be listed under the provider in
  the app's ``/api/available_models`` and each prompt names its model; OpenRouter
  still needs ``--models``, its request cap and its dry run;
- every scenario resets the window's canvas and chat (live mode also the
  server conversation), so the run asks first unless ``--yes`` is given;
- scenarios that save or load workspaces are skipped unless ``--allow-workspace-writes``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import click

from cli.cdp import CDPBrowser
from cli.scenarios.live import LiveOptions, LiveRunner, RequestBudget, planned_requests
from cli.scenarios.live_config import GuardError, LiveSettings, check_models, check_request_cap, live_plan
from cli.scenarios.model import Catalogue, Scenario
from cli.scenarios.report import ResultSink
from cli.scenarios.runner import BrowserSession, ReplayOptions, ReplayRunner

# The pause after each step when attached, unless --pace says otherwise.
ATTACHED_PACE_S = 1.5
# Live settings a run's own server pins and an attached desktop app takes from its .env instead.
UNPINNED_SETTINGS = (
    "MATHUD_TOOL_EXPOSURE",
    "MATHUD_CANVAS_FORMAT",
    "MATHUD_CANVAS_BUDGET_TOKENS",
    "TOOL_SEARCH_MODE",
    "MATHUD_LOCAL_REASONING_EFFORT",
    "MATHUD_OPENROUTER_MAX_RETRIES",
    "MATHUD_WORKSPACES_DIR",
    "provider API keys",
)
# Command-line options that set a pin (refused when attached).
PIN_OPTIONS = ("tool_exposure", "canvas_format", "canvas_budget", "tool_search_mode", "local_reasoning_effort")


def default_pace(pace: Optional[float], attached: bool) -> float:
    """``--pace`` if given, else ``ATTACHED_PACE_S`` when attached and 0 otherwise."""
    if pace is not None:
        return max(float(pace), 0.0)
    return ATTACHED_PACE_S if attached else 0.0


def pin_conflicts(given: list[str]) -> Optional[str]:
    """The refusal for pin options given with ``--attach-desktop``, or None."""
    pins = [name for name in PIN_OPTIONS if name in given]
    if not pins:
        return None
    flags = ", ".join("--" + name.replace("_", "-") for name in pins)
    return (
        f"{flags} cannot be applied with --attach-desktop: the desktop app's server uses its own .env. "
        "Set them there (and restart the app), or run without --attach-desktop."
    )


def confirmation_text(base_url: str, mode: str, scenarios: int, provider: Optional[str]) -> str:
    resets = "canvas, undo history and chat" + (" and the server conversation" if mode == "live" else "")
    text = (
        f"This drives the MatHud desktop window at {base_url}: each of {scenarios} scenario run(s) resets its "
        f"{resets}. Nothing is saved first."
    )
    if provider == "openrouter":
        text += (
            " The app uses its own .env: its TOOL_SEARCH_MODE must be local and MATHUD_OPENROUTER_MAX_RETRIES 0 "
            "for the request cap to count every paid request."
        )
    return text


def confirm(text: str, yes: bool, ask: Callable[[str], bool], interactive: bool) -> bool:
    """True when the user agreed: ``--yes``, or a yes typed at the terminal."""
    if yes:
        return True
    if not interactive:
        return False
    return ask(f"{text}\nContinue?")


def attached_config(browser: CDPBrowser, debug_port: int, mode: str, pace_s: float) -> dict[str, Any]:
    """Run-config fields of an attached run."""
    return {
        "attached_desktop": True,
        "desktop_app_url": browser.base_url,
        "automation_port": debug_port,
        "port": _port_of(browser.base_url),
        "server_started": False,
        "workspaces_dir": None,
        "pins_applied": False,
        "unpinned_settings": list(UNPINNED_SETTINGS) if mode == "live" else ["MATHUD_WORKSPACES_DIR"],
        "pace_s": pace_s,
    }


def _port_of(base_url: Optional[str]) -> Optional[int]:
    try:
        return int(str(base_url).rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


def run_attached(
    catalogue: Catalogue,
    chosen: list[Scenario],
    *,
    mode: str,
    debug_port: int,
    yes: bool,
    allow_workspace_writes: bool,
    out_dir: Path,
    as_json: bool,
    options: ReplayOptions,
    dry_run: bool,
    config_extra: dict[str, Any],
    given_options: list[str],
    settings: Optional[LiveSettings] = None,
    models: Optional[list[str]] = None,
    repeats: int = 1,
    max_requests: int = 0,
    live_options: Optional[LiveOptions] = None,
    connect: Callable[[int], CDPBrowser] = lambda port: _connect(port),
    ask: Callable[[str], bool] = lambda text: _ask(text),
    interactive: Optional[bool] = None,
) -> int:
    """Run ``chosen`` in replay or live mode in the attached desktop window; returns the exit code."""
    from cli.scenarios.command import (
        _available_models,
        _conversation_resetter,
        _drive,
        _fail,
        _idle_waiter,
        _plan,
        _print_live_plan,
        _workspace_skips,
    )

    if mode not in ("replay", "live"):
        return _fail("--attach-desktop works with --mode replay or --mode live")
    refusal = pin_conflicts(given_options) if mode == "live" else None
    if refusal:
        return _fail(refusal)
    skipped = {} if allow_workspace_writes else _workspace_skips(chosen)
    live = mode == "live"
    settings = settings or LiveSettings()
    live_options = live_options or LiveOptions()
    models = models or []
    openrouter = live and settings.provider == "openrouter"
    cap = max_requests if openrouter else None
    if live:
        try:
            settings.validate()
        except GuardError as exc:
            return _fail(str(exc))
        if openrouter and not models:
            return _fail("--provider openrouter needs --models (no default model is ever chosen for a paid provider)")
    runnable = [s for s in chosen if s.id not in skipped]
    planned = planned_requests(runnable, max(len(models), 1), repeats, live_options) if live else 0

    if dry_run:
        if live:
            plan = live_plan(
                scenarios=len(runnable),
                turns=sum(s.turns for s in runnable),
                models=models or ["<the local model the desktop app serves>"],
                repeats=repeats,
                planned=planned,
                settings=settings,
                max_requests=cap,
            )
            plan["attached_desktop"] = debug_port
            plan["skipped"] = skipped
            _print_live_plan(plan, as_json)
        else:
            replay_plan = _plan(chosen, skipped)
            replay_plan["attached_desktop"] = debug_port
            click.echo(json.dumps(replay_plan, indent=2))
        return 0
    if openrouter:
        try:
            check_request_cap(planned, max_requests)
        except GuardError as exc:
            return _fail(f"Aborting: {exc}")

    try:
        probe = connect(debug_port)
    except Exception as exc:
        return _fail(f"Could not attach to the desktop window: {exc}")
    try:
        base_url = probe.base_url or ""
        if probe.call_hook("getMatHudTurnStatus").get("processing"):
            return _fail("A turn is running in the desktop window; wait for it or stop it, then run again.")
        run_models: list[str] = []
        if live:
            try:
                run_models = check_models(_available_models(base_url), settings.provider, models)
            except Exception as exc:  # GuardError, or the server did not answer
                return _fail(f"Aborting before the first message: {exc}")
        config = attached_config(probe, debug_port, mode, options.pace_s)
    finally:
        probe.close()

    runs = len(runnable) * (len(run_models) * repeats if live else 1)
    text = confirmation_text(base_url, mode, runs, settings.provider if live else None)
    tty = _stdin_is_terminal() if interactive is None else interactive
    if not confirm(text, yes, ask, tty):
        return _fail("Not confirmed; nothing was run. Pass --yes to run without asking.")

    config.update({"mode": mode, "step_timeout_s": options.step_timeout_s})
    if live:
        config.update(
            {
                "provider": settings.provider,
                "models": run_models,
                "repeats": repeats,
                "turn_timeout_s": live_options.turn_timeout_s,
                "turn_max_requests": live_options.turn_max_requests,
                "max_requests": cap,
                "planned_requests": planned_requests(runnable, len(run_models), repeats, live_options),
                "retrace_failures": live_options.retrace_failures,
            }
        )
    config.update(config_extra)
    sink = ResultSink(out_dir, config)
    session = BrowserSession(lambda: CDPBrowser(debug_port), options.step_timeout_s)
    log = lambda line: click.echo(line, err=as_json)  # noqa: E731
    click.echo(
        f"Attached to the desktop window at {base_url} (automation port {debug_port}); {mode} run of "
        f"{len(runnable)} scenario(s), pause {options.pace_s:g} s per step; output in {out_dir}",
        err=as_json,
    )
    if live:
        runner = LiveRunner(
            catalogue,
            session,
            sink,
            options,
            live_options,
            provider=settings.provider,
            reset_conversation=_conversation_resetter(base_url),
            budget=RequestBudget(cap),
            wait_idle=_idle_waiter(base_url),
            log=log,
        )

        def run() -> None:
            try:
                runner.run_live(chosen, run_models, repeats, skipped)
            finally:
                config["requests_sent"] = runner.budget.sent
                if runner.stopped:
                    config["stopped"] = runner.stopped

        return _drive(run, session, sink, catalogue, None, None, as_json, out_dir)
    replay = ReplayRunner(catalogue, session, sink, options, log=log)
    return _drive(lambda: replay.run(chosen, skipped), session, sink, catalogue, None, None, as_json, out_dir)


def _stdin_is_terminal() -> bool:
    return sys.stdin is not None and sys.stdin.isatty()


def _ask(text: str) -> bool:
    """Ask at the terminal; no answer (end of input, Ctrl+C) is a no."""
    try:
        return bool(click.confirm(text, default=False, err=True))
    except click.exceptions.Abort:
        click.echo("", err=True)
        return False


def _connect(port: int) -> CDPBrowser:
    from cli.desktop_automation import connect_desktop

    return connect_desktop(port)
