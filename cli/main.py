"""MatHud CLI entry point.

Provides the main command group and wires together all subcommands.
"""

from __future__ import annotations

from typing import Any

import click

from cli.console import make_stdio_safe
from cli.server import server
from cli.tests import test
from cli.canvas import canvas
from cli.workspace import workspace
from cli.chat import chat
from cli.desktop import desktop
from cli.screenshot import screenshot


class _SafeOutputGroup(click.Group):
    """The root group: makes stdout and stderr encoding-safe before any command prints."""

    def main(self, *args: Any, **kwargs: Any) -> Any:
        make_stdio_safe()
        return super().main(*args, **kwargs)


@click.group(cls=_SafeOutputGroup)
@click.version_option(package_name="mathud-cli")
def cli() -> None:
    """MatHud - Command-line interface for canvas automation.

    Manage the MatHud server, run tests, interact with the canvas,
    and capture screenshots from the command line.
    """
    pass


# Register command groups
cli.add_command(server)
cli.add_command(test)
cli.add_command(canvas)
cli.add_command(workspace)
cli.add_command(chat)
cli.add_command(screenshot)
cli.add_command(desktop)


if __name__ == "__main__":
    cli()
