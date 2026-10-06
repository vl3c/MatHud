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
- a paid provider is refused unless the app reports (``/api/automation_settings``)
  that it searches tools locally and that OpenRouter does not retry: both would
  send requests the cap cannot count;
- every scenario resets the window's canvas and chat (live mode also the
  server conversation; replay resets the conversation once at the start, since
  it never adds to it), so the run prints what it will do and asks first unless
  ``--yes`` is given;
- scenarios that save or load workspaces are always skipped, and while the run
  drives the window its workspace tools answer with an error
  (``setMatHudAutomationGuards``), so neither a model nor a retrace can touch
  the user's workspace directory; the block is lifted when the run ends;
- Ctrl+C stops the turn running in the window before the run ends, and closing
  the window stops the run;
- with ``--fit-view`` (the default when attached) the view is zoomed to the drawings
  after each graded step, for display only (``ReplayOptions.fit_view``): grading and
  artifacts come first, the grader takes the fitted canvas as the next step's
  baseline, and scenarios that set or check the view are never fitted.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import click

from cli.cdp import CDPBrowser, CDPError, CDPUnavailable
from cli.scenarios.live import LiveOptions, LiveRunner, RequestBudget, planned_requests
from cli.scenarios.live_config import GuardError, LiveSettings, check_models, check_request_cap, live_plan
from cli.scenarios.model import Catalogue, Scenario
from cli.scenarios.report import ResultSink
from cli.scenarios.runner import BrowserSession, ReplayOptions, ReplayRunner, RunStopped

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


# The workspace-tool block is a lease: the window lifts it by itself unless the run renews
# it, so a CLI that crashes or is killed cannot leave it on. Renewed before every hook that
# can run tools and, during a long turn, by the turn's status polls every RENEW_EVERY_S.
GUARD_LEASE_S = 90.0
RENEW_EVERY_S = 20.0
BLOCK_WORKSPACE_TOOLS = json.dumps({"block_workspace_tools": True, "lease_s": GUARD_LEASE_S})
UNBLOCK_WORKSPACE_TOOLS = json.dumps({"block_workspace_tools": False})
# Hooks that can run tools: the block is re-applied before each (cheap, and idempotent).
GUARDED_HOOKS = frozenset({"resetMatHudSession", "runMatHudToolCalls", "sendMatHudMessage"})
# How long a ready page may take to finish loading its model list (window.matHudModelsLoaded).
MODELS_LOADED_TIMEOUT_S = 10.0
_MODELS_LOADED_JS = "return window.matHudModelsLoaded !== false;"


class AttachedBrowser(CDPBrowser):
    """The run's session browser: the desktop window, with the run's guards applied whenever the app is ready.

    Workspace tools are blocked each time the page is ready and before every
    hook that can run tools (a reload, also one from outside, lifts the block),
    and the block's lease is renewed while a turn is polled. A ready page is
    also given time to load its model list. A send refused for a model missing
    from the dropdown refreshes the list and is retried once (nothing was sent).
    A DevTools port that refuses connections means the window was closed: the
    run stops instead of erroring on every remaining scenario.
    """

    def __init__(self, debug_port: int, **kwargs: Any) -> None:
        super().__init__(debug_port, **kwargs)
        self._renewed_at = float("-inf")
        self._blocking = False

    def setup(self) -> None:
        try:
            super().setup()
        except CDPUnavailable as exc:
            raise RunStopped(f"the desktop window is gone ({exc})") from exc

    def wait_for_app_ready(self, timeout: float = 60.0) -> bool:
        if not super().wait_for_app_ready(timeout):
            return False
        if self._blocking:  # called back from the guard hook's own wait-and-retry
            return True
        self._wait_for_model_list()
        self._block_workspace_tools()
        return True

    def call_hook(self, name: str, *args: Any, timeout: int = 30) -> dict[str, Any]:
        # The page may have been reloaded from outside since the block was applied (a reload
        # lifts it), so it is applied again before anything that can run tools; any other
        # hook (the turn's status polls) renews the lease once it is RENEW_EVERY_S old.
        if name in GUARDED_HOOKS or time.monotonic() - self._renewed_at > RENEW_EVERY_S:
            self._block_workspace_tools()
        reply = super().call_hook(name, *args, timeout=timeout)
        if name == "sendMatHudMessage" and "Model option not found" in str(reply.get("error", "")):
            from cli.desktop_automation import ensure_model_listed

            ensure_model_listed(self, str(args[1]))  # the window refused before sending anything
            reply = super().call_hook(name, *args, timeout=timeout)
        return reply

    def _wait_for_model_list(self) -> None:
        deadline = time.monotonic() + MODELS_LOADED_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                if self.execute_js(_MODELS_LOADED_JS, timeout=5) is True:
                    return
            except CDPError:
                pass
            time.sleep(0.05)

    def _block_workspace_tools(self) -> None:
        self._blocking = True
        try:
            reply = CDPBrowser.call_hook(self, "setMatHudAutomationGuards", BLOCK_WORKSPACE_TOOLS)
        except RuntimeError as exc:  # an app without the guard hook
            raise RunStopped(f"the window cannot block its workspace tools: {exc}") from exc
        finally:
            self._blocking = False
        if reply.get("workspace_tools_blocked") is not True:
            raise RunStopped(f"the window did not block its workspace tools: {reply}")
        self._renewed_at = time.monotonic()


def attached_browser(port: int) -> CDPBrowser:
    return AttachedBrowser(port)


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
    if mode == "replay":
        text += " The server conversation is reset once first."
    if provider == "openrouter":
        text += (
            " The app uses its own .env: its TOOL_SEARCH_MODE must be local and MATHUD_OPENROUTER_MAX_RETRIES 0 "
            "for the request cap to count every paid request (checked: both are)."
        )
    return text


def confirm(text: str, yes: bool, ask: Callable[[str], bool], interactive: bool) -> bool:
    """True when the user agreed: ``--yes``, or a yes typed at the terminal."""
    if yes:
        return True
    if not interactive:
        return False
    return ask(f"{text}\nContinue?")


def attached_config(
    browser: CDPBrowser, debug_port: int, mode: str, pace_s: float, fit_view: bool = False
) -> dict[str, Any]:
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
        # The view is zoomed to the drawings after each graded step (never in a view-sensitive
        # scenario); a live model then sees the fitted view in its canvas summary.
        "fit_view": fit_view,
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
    if allow_workspace_writes:
        return _fail(
            "--allow-workspace-writes cannot be used with --attach-desktop: the desktop app's workspace "
            "directory is the user's own. Workspace scenarios are skipped when attached."
        )
    refusal = pin_conflicts(given_options) if mode == "live" else None
    if refusal:
        return _fail(refusal)
    skipped = _workspace_skips(chosen)
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

    from cli.desktop_automation import (
        DesktopError,
        automation_settings,
        ensure_model_listed,
        paid_provider_guard,
        stop_turn,
    )

    try:
        probe = connect(debug_port)
    except Exception as exc:
        return _fail(f"Could not attach to the desktop window: {exc}")
    try:
        base_url = probe.base_url or ""
        if probe.call_hook("getMatHudTurnStatus").get("processing"):
            return _fail("A turn is running in the desktop window; wait for it or stop it, then run again.")
        run_models: list[str] = []
        server_settings: Optional[dict[str, Any]] = None
        if live:
            try:
                run_models = check_models(_available_models(base_url), settings.provider, models)
                server_settings = paid_provider_guard(base_url, settings.provider)
                for model in run_models:
                    ensure_model_listed(probe, model)
            except (GuardError, DesktopError) as exc:
                return _fail(f"Aborting before the first message: {exc}")
            except Exception as exc:  # the server did not answer
                return _fail(f"Aborting before the first message: {exc}")
            if server_settings is None:
                try:
                    server_settings = automation_settings(base_url)
                except Exception:
                    server_settings = None  # an older app; only recorded for a local run
        config = attached_config(probe, debug_port, mode, options.pace_s, options.fit_view)
        if server_settings is not None:
            config["server_settings"] = server_settings
    finally:
        probe.close()

    runs = len(runnable) * (len(run_models) * repeats if live else 1)
    text = confirmation_text(base_url, mode, runs, settings.provider if live else None)
    # The text says what the run will do to the window (and, for a paid provider, what was
    # checked), so it is always shown, also with --yes.
    click.echo(text, err=True)
    tty = _stdin_is_terminal() if interactive is None else interactive
    if not confirm("", yes, lambda _text: ask("Continue?"), tty):
        return _fail("Not confirmed; nothing was run. Pass --yes to run without asking.")
    if not live:
        try:
            # Replay never adds to the server conversation; reset it once so it matches the cleared chat.
            _conversation_resetter(base_url)()
        except Exception as exc:
            return _fail(f"Could not reset the server conversation: {exc}")

    def stop_window_turn() -> None:
        """Ctrl+C: stop the turn running in the window over a connection of its own."""
        try:
            browser = connect(debug_port)
        except Exception:
            return
        try:
            stop_turn(browser)
        finally:
            browser.close()

    def unblock_workspace_tools() -> None:
        try:
            browser = connect(debug_port)
        except Exception:
            return  # the window is gone, and with it the block
        try:
            browser.call_hook("setMatHudAutomationGuards", UNBLOCK_WORKSPACE_TOOLS)
        except Exception:
            pass
        finally:
            browser.close()

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
    session = BrowserSession(lambda: attached_browser(debug_port), options.step_timeout_s)
    log = lambda line: click.echo(line, err=as_json)  # noqa: E731
    fitting = "on" if options.fit_view else "off"
    click.echo(
        f"Attached to the desktop window at {base_url} (automation port {debug_port}); {mode} run of "
        f"{len(chosen)} scenario(s) ({len(skipped)} skipped), pause {options.pace_s:g} s per step, "
        f"view fitting {fitting}; output in {out_dir}",
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

    else:
        replay = ReplayRunner(catalogue, session, sink, options, log=log)

        def run() -> None:
            try:
                replay.run(chosen, skipped)
            finally:
                if replay.stopped:
                    config["stopped"] = replay.stopped

    try:
        return _drive(run, session, sink, catalogue, None, None, as_json, out_dir, on_interrupt=stop_window_turn)
    finally:
        unblock_workspace_tools()


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
