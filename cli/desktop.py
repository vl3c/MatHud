"""Desktop app commands for the MatHud CLI.

``desktop`` on its own runs the desktop launcher in ``mathud_desktop.py`` in
this process: the Flask app is served on localhost and shown in a native
pywebview window. Its subcommands drive a window that was opened with
``--automation-port N`` (see ``cli/desktop_automation.py``): ``prompt`` sends a
prompt through the window's chat, ``fit`` zooms it to its drawings, ``state``
prints its canvas and ``screenshot`` saves a picture of it. Fitting is an
automation convenience only: in regular use the app never pans or zooms on its own.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import NoReturn, Optional

import click

from cli.config import CLI_OUTPUT_DIR, DEFAULT_AUTOMATION_PORT, PROJECT_ROOT


@click.group(invoke_without_command=True)
@click.option("--port", "-p", type=int, default=None, help="Port to serve on (default: 5100, or a free port if taken).")
@click.option("--browser", is_flag=True, help="Open MatHud in the default browser instead of a window.")
@click.option("--devtools", is_flag=True, help="Enable the WebView developer tools.")
@click.option(
    "--automation-port",
    type=int,
    default=None,
    help="Let the CLI drive the window over CDP on 127.0.0.1:N (any local program can then control it).",
)
@click.pass_context
def desktop(
    ctx: click.Context, port: Optional[int], browser: bool, devtools: bool, automation_port: Optional[int]
) -> None:
    """Open MatHud in a desktop window (server included), or drive an open one.

    Needs pywebview (pip install -r requirements-desktop.txt); use --browser
    without it. Closing the window stops the server. The prompt, state and
    screenshot subcommands drive a window opened with --automation-port.
    """
    if ctx.invoked_subcommand is not None:
        if port is not None or browser or devtools or automation_port is not None:
            _fail(
                f"desktop's own options open a window; they do not apply to `desktop {ctx.invoked_subcommand}`. "
                f"Give the automation port after the subcommand: desktop {ctx.invoked_subcommand} --port N"
            )
        return
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import mathud_desktop

    argv: list[str] = []
    if port is not None:
        argv += ["--port", str(port)]
    if browser:
        argv.append("--browser")
    if devtools:
        argv.append("--devtools")
    if automation_port is not None:
        argv += ["--automation-port", str(automation_port)]
    sys.exit(mathud_desktop.main(argv))


_PORT_HELP = f"The window's automation port (mathud_desktop.py --automation-port; default {DEFAULT_AUTOMATION_PORT})"


def _fail(message: str, code: int = 2) -> NoReturn:
    click.echo(click.style(message, fg="red"), err=True)
    raise SystemExit(code)


@desktop.command("prompt")
@click.argument("text")
@click.option("--port", "-p", default=DEFAULT_AUTOMATION_PORT, type=int, show_default=True, help=_PORT_HELP)
@click.option(
    "--model", default=None, help="Model id (default: the only local model; required for --provider openrouter)"
)
@click.option(
    "--provider",
    type=click.Choice(["local", "openrouter"]),
    default="local",
    show_default=True,
    help="Provider the model must be registered under in /api/available_models",
)
@click.option(
    "--timeout", "timeout_s", default=300.0, type=float, show_default=True, help="Seconds before the turn is stopped"
)
@click.option(
    "--max-requests",
    default=8,
    type=click.IntRange(min=1),
    show_default=True,
    help="Model requests the turn may send",
)
@click.option(
    "--fit-view/--no-fit-view",
    default=True,
    show_default=True,
    help="Zoom the window to the drawings after the turn (display only)",
)
@click.option("--json", "as_json", is_flag=True, help="Print the result as JSON")
def prompt_cmd(
    text: str,
    port: int,
    model: Optional[str],
    provider: str,
    timeout_s: float,
    max_requests: int,
    fit_view: bool,
    as_json: bool,
) -> None:
    """Send TEXT through the desktop window's chat and print the reply, tool calls and turn metrics.

    The prompt and reply appear in the window as if typed there. It is always
    sent with an explicit model id registered under --provider: without
    --model, the only local model (never the window's current selection).
    Afterwards the window is zoomed to its drawings (display only) unless
    --no-fit-view.
    """
    from cli.desktop_automation import (
        PAID_PROVIDER_WARNING,
        DesktopError,
        available_models,
        choose_prompt_model,
        connect_desktop,
        ensure_model_listed,
        format_prompt_result,
        paid_provider_guard,
        run_prompt,
        stop_turn,
        validate_prompt,
    )
    from cli.desktop_automation import fit_view as fit_window_view
    from cli.scenarios.live_config import GuardError

    try:
        validate_prompt(text)
    except DesktopError as exc:
        _fail(f"Not sent: {exc}")
    if provider != "local":
        click.echo(click.style(PAID_PROVIDER_WARNING, fg="yellow"), err=True)
    try:
        browser = connect_desktop(port)
    except DesktopError as exc:
        _fail(str(exc))
    try:
        assert browser.base_url is not None
        try:
            chosen = choose_prompt_model(available_models(browser.base_url), provider, model)
            paid_provider_guard(browser.base_url, provider)
        except GuardError as exc:
            _fail(f"Not sent: {exc}")
        except Exception as exc:
            _fail(f"Not sent: could not read {browser.base_url}/api/available_models: {exc}")
        try:
            ensure_model_listed(browser, chosen)
        except DesktopError as exc:
            _fail(f"Not sent: {exc}")
        if not as_json:
            click.echo(f"Sending to {browser.base_url} with {chosen} ...", err=True)
        try:
            result = run_prompt(browser, text, chosen, timeout_s=timeout_s, max_requests=max_requests)
        except DesktopError as exc:
            _fail(str(exc))
        except KeyboardInterrupt:
            # Do not leave the turn running in the window (it would keep sending requests).
            stop_turn(browser)
            click.echo(click.style("Interrupted; the turn was stopped.", fg="yellow"), err=True)
            raise SystemExit(130)
        if fit_view:
            try:
                result["fit_view"] = fit_window_view(browser)
            except (DesktopError, RuntimeError) as exc:
                result["fit_view"] = {"status": "error", "error": str(exc)}
    finally:
        browser.close()
    if as_json:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        click.echo(format_prompt_result(result))
    if result["outcome"] not in ("stop", "max_requests"):
        raise SystemExit(1)


@desktop.command("fit")
@click.option("--port", "-p", default=DEFAULT_AUTOMATION_PORT, type=int, show_default=True, help=_PORT_HELP)
def fit_cmd(port: int) -> None:
    """Zoom the desktop window to its drawings (display only: no undo entry).

    The app itself never pans or zooms on its own; this is an automation convenience.
    """
    from cli.desktop_automation import DesktopError, connect_desktop, fit_view

    try:
        browser = connect_desktop(port)
    except DesktopError as exc:
        _fail(str(exc))
    try:
        reply = fit_view(browser)
    except DesktopError as exc:
        _fail(str(exc), 1)
    finally:
        browser.close()
    if not reply.get("fitted"):
        click.echo("Nothing to fit; the view is unchanged.")
        return
    view = reply.get("view") or {}
    click.echo(
        f"View: x {view.get('left_bound'):.4g} to {view.get('right_bound'):.4g}, "
        f"y {view.get('bottom_bound'):.4g} to {view.get('top_bound'):.4g}"
    )


@desktop.command("state")
@click.option("--port", "-p", default=DEFAULT_AUTOMATION_PORT, type=int, show_default=True, help=_PORT_HELP)
@click.option("--inspect", is_flag=True, help="Add the inspection view (labels, colours, undo depth, ...)")
def state_cmd(port: int, inspect: bool) -> None:
    """Print the desktop window's canvas state as JSON."""
    from cli.desktop_automation import DesktopError, connect_desktop

    try:
        browser = connect_desktop(port)
    except DesktopError as exc:
        _fail(str(exc))
    try:
        snapshot = browser.get_canvas_snapshot({"inspect": True} if inspect else {})
    finally:
        browser.close()
    click.echo(json.dumps(snapshot if inspect else snapshot.get("state"), indent=2, default=str))


@desktop.command("screenshot")
@click.option("--port", "-p", default=DEFAULT_AUTOMATION_PORT, type=int, show_default=True, help=_PORT_HELP)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    help="Output file (default: cli/output/desktop_<time>.png)",
)
def screenshot_cmd(port: int, output: Optional[str]) -> None:
    """Save a PNG of the desktop window's page."""
    from cli.desktop_automation import DesktopError, connect_desktop

    path = output or str(CLI_OUTPUT_DIR / f"desktop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    try:
        browser = connect_desktop(port)
    except DesktopError as exc:
        _fail(str(exc))
    try:
        saved = browser.screenshot(path)
    finally:
        browser.close()
    if not saved:
        _fail("Could not capture the window.", 1)
    click.echo(f"Screenshot saved: {path}")
