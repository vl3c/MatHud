"""Canvas operation commands for the MatHud CLI.

Provides commands for interacting with the MatHud canvas via browser automation.
"""

from __future__ import annotations

import json
from typing import Optional

import click

from cli.browser import BrowserAutomation
from cli.config import DEFAULT_PORT
from cli.server import ServerManager


def ensure_browser_ready(port: int, headless: bool = True) -> tuple[bool, Optional[BrowserAutomation], str]:
    """Ensure server is running and browser is ready.

    Args:
        port: Server port number.
        headless: Run browser in headless mode.

    Returns:
        Tuple of (success, browser_instance, error_message).
        If success is False, browser_instance will be None.
    """
    manager = ServerManager(port=port)

    if not manager.is_server_running():
        return False, None, f"Server is not running on port {port}. Start it with 'mathud server start'."

    browser = BrowserAutomation(port=port, headless=headless)
    try:
        browser.setup()
        if not browser.navigate_to_app():
            browser.cleanup()
            return False, None, "Failed to navigate to application"

        if not browser.wait_for_app_ready():
            browser.cleanup()
            return False, None, "Application did not become ready in time"

        return True, browser, ""
    except Exception as e:
        browser.cleanup()
        return False, None, str(e)


@click.group()
def canvas() -> None:
    """Interact with the MatHud canvas."""
    pass


@canvas.command()
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
def clear(port: int) -> None:
    """Clear the canvas (remove all drawables)."""
    success, browser, error = ensure_browser_ready(port)
    if not success or browser is None:
        click.echo(click.style(f"Error: {error}", fg="red"), err=True)
        raise SystemExit(1)

    try:
        browser.call_canvas_method("clear")
        click.echo(click.style("Canvas cleared", fg="green"))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        raise SystemExit(1)
    finally:
        browser.cleanup()


@canvas.command()
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
def reset(port: int) -> None:
    """Reset canvas zoom and offset to defaults."""
    success, browser, error = ensure_browser_ready(port)
    if not success or browser is None:
        click.echo(click.style(f"Error: {error}", fg="red"), err=True)
        raise SystemExit(1)

    try:
        browser.call_canvas_method("reset_view")
        click.echo(click.style("Canvas view reset", fg="green"))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        raise SystemExit(1)
    finally:
        browser.cleanup()


@canvas.command()
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
def undo(port: int) -> None:
    """Undo the last canvas operation."""
    success, browser, error = ensure_browser_ready(port)
    if not success or browser is None:
        click.echo(click.style(f"Error: {error}", fg="red"), err=True)
        raise SystemExit(1)

    try:
        result = browser.call_canvas_method("undo")
        if result:
            click.echo(click.style("Undo successful", fg="green"))
        else:
            click.echo(click.style("Nothing to undo", fg="yellow"))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        raise SystemExit(1)
    finally:
        browser.cleanup()


@canvas.command()
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
def redo(port: int) -> None:
    """Redo the last undone canvas operation."""
    success, browser, error = ensure_browser_ready(port)
    if not success or browser is None:
        click.echo(click.style(f"Error: {error}", fg="red"), err=True)
        raise SystemExit(1)

    try:
        result = browser.call_canvas_method("redo")
        if result:
            click.echo(click.style("Redo successful", fg="green"))
        else:
            click.echo(click.style("Nothing to redo", fg="yellow"))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        raise SystemExit(1)
    finally:
        browser.cleanup()


@canvas.command()
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
@click.option("--in", "zoom_in", is_flag=True, help="Zoom in")
@click.option("--out", "zoom_out", is_flag=True, help="Zoom out")
@click.option("--factor", "-f", type=float, help="Zoom factor (e.g., 1.2 to zoom in, 0.8 to zoom out)")
def zoom(port: int, zoom_in: bool, zoom_out: bool, factor: Optional[float]) -> None:
    """Zoom the canvas view.

    Use --in to zoom in, --out to zoom out, or --factor for custom zoom.
    """
    if not zoom_in and not zoom_out and factor is None:
        click.echo("Specify --in, --out, or --factor", err=True)
        raise SystemExit(1)

    if factor is None:
        factor = 1.2 if zoom_in else 0.8

    success, browser, error = ensure_browser_ready(port)
    if not success or browser is None:
        click.echo(click.style(f"Error: {error}", fg="red"), err=True)
        raise SystemExit(1)

    try:
        browser.call_canvas_method("zoom", factor)
        direction = "in" if factor > 1 else "out"
        click.echo(click.style(f"Zoomed {direction} (factor: {factor})", fg="green"))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        raise SystemExit(1)
    finally:
        browser.cleanup()


@canvas.command()
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
@click.option("--pretty", is_flag=True, help="Pretty-print JSON output")
def state(port: int, pretty: bool) -> None:
    """Output the current canvas state as JSON."""
    success, browser, error = ensure_browser_ready(port)
    if not success or browser is None:
        click.echo(click.style(f"Error: {error}", fg="red"), err=True)
        raise SystemExit(1)

    try:
        canvas_state = browser.get_canvas_state()
        if pretty:
            click.echo(json.dumps(canvas_state, indent=2))
        else:
            click.echo(json.dumps(canvas_state))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        raise SystemExit(1)
    finally:
        browser.cleanup()


@canvas.command("exec")
@click.argument("function_name")
@click.option(
    "--args",
    "-a",
    "args_json",
    help="Function arguments as JSON object",
)
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
@click.option("--pretty", is_flag=True, help="Pretty-print JSON result")
def exec_func(function_name: str, args_json: Optional[str], port: int, pretty: bool) -> None:
    """Execute a function from the FunctionRegistry.

    FUNCTION_NAME is the name of the function to execute.

    Examples:

      mathud canvas exec create_point --args '{"x": 5, "y": 3, "name": "A"}'

      mathud canvas exec draw_function --args '{"expression": "x**2", "x_min": -5, "x_max": 5}'
    """
    # Parse arguments
    args = {}
    if args_json:
        try:
            args = json.loads(args_json)
            if not isinstance(args, dict):
                click.echo(click.style("Arguments must be a JSON object", fg="red"), err=True)
                raise SystemExit(1)
        except json.JSONDecodeError as e:
            click.echo(click.style(f"Invalid JSON: {e}", fg="red"), err=True)
            raise SystemExit(1)

    success, browser, error = ensure_browser_ready(port)
    if not success or browser is None:
        click.echo(click.style(f"Error: {error}", fg="red"), err=True)
        raise SystemExit(1)

    try:
        result = browser.call_function_registry(function_name, args)

        if result is not None:
            if pretty:
                try:
                    click.echo(json.dumps(result, indent=2))
                except (TypeError, ValueError):
                    click.echo(str(result))
            else:
                try:
                    click.echo(json.dumps(result))
                except (TypeError, ValueError):
                    click.echo(str(result))
        else:
            click.echo(click.style(f"Function '{function_name}' executed (no return value)", fg="green"))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        raise SystemExit(1)
    finally:
        browser.cleanup()
