"""Screenshot capture commands for the MatHud CLI.

Provides commands for capturing screenshots of the MatHud application.
Screenshots are saved to cli/output/ by default.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from cli.browser import BrowserAutomation
from cli.config import (
    CLI_OUTPUT_DIR,
    DEFAULT_PORT,
    DEFAULT_VIEWPORT_HEIGHT,
    DEFAULT_VIEWPORT_WIDTH,
)
from cli.server import ServerManager


def generate_default_filename(prefix: str = "screenshot") -> Path:
    """Generate a default filename with timestamp in the CLI output directory.

    Args:
        prefix: Filename prefix (default: "screenshot").

    Returns:
        Path to the output file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return CLI_OUTPUT_DIR / f"{prefix}_{timestamp}.png"


@click.group()
def screenshot() -> None:
    """Capture screenshots of the MatHud application.

    Screenshots are saved to cli/output/ by default.
    """
    pass


@screenshot.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True),
    help="Output file path (default: cli/output/screenshot_<timestamp>.png)",
)
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
@click.option(
    "--canvas-only",
    is_flag=True,
    help="Capture only the canvas area (default: full page)",
)
@click.option(
    "--width",
    "-w",
    default=DEFAULT_VIEWPORT_WIDTH,
    type=int,
    help=f"Viewport width (default: {DEFAULT_VIEWPORT_WIDTH})",
)
@click.option(
    "--height",
    "-h",
    "height",  # Rename to avoid conflict with --help
    default=DEFAULT_VIEWPORT_HEIGHT,
    type=int,
    help=f"Viewport height (default: {DEFAULT_VIEWPORT_HEIGHT})",
)
@click.option(
    "--wait",
    type=float,
    default=0.5,
    help="Seconds to wait after page load before capture (default: 0.5)",
)
@click.option(
    "--no-headless",
    is_flag=True,
    help="Show browser window (for debugging)",
)
def capture(
    output: Optional[str],
    port: int,
    canvas_only: bool,
    width: int,
    height: int,
    wait: float,
    no_headless: bool,
) -> None:
    """Capture a screenshot of the MatHud application.

    Captures the full application GUI by default (canvas, chat, controls).
    Use --canvas-only to capture just the canvas area.

    Screenshots are saved to cli/output/ by default with a timestamp filename.

    Examples:

      mathud screenshot capture

      mathud screenshot capture -o myshot.png

      mathud screenshot capture --canvas-only

      mathud screenshot capture --width 2560 --height 1440
    """
    import time

    manager = ServerManager(port=port)
    if not manager.is_server_running():
        click.echo(click.style(f"Server is not running on port {port}", fg="red"), err=True)
        raise SystemExit(1)

    # Generate default filename if not provided
    if output:
        output_path = Path(output)
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")
    else:
        prefix = "canvas" if canvas_only else "screenshot"
        output_path = generate_default_filename(prefix)

    # Create parent directories if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with BrowserAutomation(
            port=port,
            headless=not no_headless,
            viewport_width=width,
            viewport_height=height,
        ) as browser:
            click.echo(f"Navigating to http://127.0.0.1:{port}/...")

            if not browser.navigate_to_app():
                click.echo(click.style("Failed to navigate to application", fg="red"), err=True)
                raise SystemExit(1)

            click.echo("Waiting for application to be ready...")
            if not browser.wait_for_app_ready():
                click.echo(click.style("Application did not become ready in time", fg="red"), err=True)
                raise SystemExit(1)

            # Additional wait for rendering to complete
            if wait > 0:
                time.sleep(wait)

            click.echo(f"Capturing {'canvas' if canvas_only else 'full page'}...")
            if browser.capture_screenshot(str(output_path), full_page=not canvas_only):
                click.echo(click.style(f"Screenshot saved to {output_path}", fg="green"))
            else:
                click.echo(click.style("Failed to capture screenshot", fg="red"), err=True)
                raise SystemExit(1)

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        raise SystemExit(1)
