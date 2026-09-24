"""Desktop app command for the MatHud CLI.

Runs the desktop launcher in ``mathud_desktop.py`` in this process: the
Flask app is served on localhost and shown in a native pywebview window.
"""

from __future__ import annotations

import sys
from typing import Optional

import click

from cli.config import PROJECT_ROOT


@click.command()
@click.option("--port", "-p", type=int, default=None, help="Port to serve on (default: 5100, or a free port if taken).")
@click.option("--browser", is_flag=True, help="Open MatHud in the default browser instead of a window.")
@click.option("--devtools", is_flag=True, help="Enable the WebView developer tools.")
def desktop(port: Optional[int], browser: bool, devtools: bool) -> None:
    """Open MatHud in a desktop window (server included).

    Needs pywebview (pip install -r requirements-desktop.txt); use --browser
    without it. Closing the window stops the server.
    """
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
    sys.exit(mathud_desktop.main(argv))
