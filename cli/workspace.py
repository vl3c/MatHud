"""Workspace management commands for the MatHud CLI.

Provides commands for listing, saving, loading, and deleting workspaces via HTTP API.
"""

from __future__ import annotations

import json
from typing import Optional

import click
import requests

from cli.config import DEFAULT_HOST, DEFAULT_PORT, HEALTH_CHECK_TIMEOUT
from cli.server import ServerManager


def get_api_base(host: str, port: int) -> str:
    """Get the API base URL."""
    return f"http://{host}:{port}"


def check_server(host: str, port: int) -> bool:
    """Check if server is running."""
    manager = ServerManager(host=host, port=port)
    return manager.is_server_running()


@click.group()
def workspace() -> None:
    """Manage MatHud workspaces."""
    pass


@workspace.command("list")
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_cmd(port: int, as_json: bool) -> None:
    """List all saved workspaces."""
    if not check_server(DEFAULT_HOST, port):
        click.echo(click.style(f"Server is not running on port {port}", fg="red"), err=True)
        raise SystemExit(1)

    try:
        response = requests.get(
            f"{get_api_base(DEFAULT_HOST, port)}/list_workspaces",
            timeout=HEALTH_CHECK_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "error":
            click.echo(click.style(f"Error: {data.get('message', 'Unknown error')}", fg="red"), err=True)
            raise SystemExit(1)

        workspaces = data.get("data", [])

        if as_json:
            click.echo(json.dumps(workspaces, indent=2))
        else:
            if not workspaces:
                click.echo("No workspaces found")
            else:
                click.echo(click.style("Saved workspaces:", bold=True))
                for ws in workspaces:
                    # Workspace can be a string (name) or dict with name/modified
                    if isinstance(ws, str):
                        click.echo(f"  - {ws}")
                    else:
                        name = ws.get("name", "Unknown")
                        modified = ws.get("modified", "")
                        click.echo(f"  - {name}" + (f" (modified: {modified})" if modified else ""))

    except requests.RequestException as e:
        click.echo(click.style(f"Request failed: {e}", fg="red"), err=True)
        raise SystemExit(1)


@workspace.command()
@click.argument("name")
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
@click.option(
    "--state",
    "-s",
    "state_json",
    help="Workspace state as JSON (if not provided, saves current canvas state)",
)
def save(name: str, port: int, state_json: Optional[str]) -> None:
    """Save the current workspace or provided state.

    NAME is the workspace name to save as.
    """
    if not check_server(DEFAULT_HOST, port):
        click.echo(click.style(f"Server is not running on port {port}", fg="red"), err=True)
        raise SystemExit(1)

    # Parse state if provided
    state = None
    if state_json:
        try:
            state = json.loads(state_json)
        except json.JSONDecodeError as e:
            click.echo(click.style(f"Invalid JSON state: {e}", fg="red"), err=True)
            raise SystemExit(1)

    try:
        payload = {"name": name}
        if state is not None:
            payload["state"] = state

        response = requests.post(
            f"{get_api_base(DEFAULT_HOST, port)}/save_workspace",
            json=payload,
            timeout=HEALTH_CHECK_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "error":
            click.echo(click.style(f"Error: {data.get('message', 'Unknown error')}", fg="red"), err=True)
            raise SystemExit(1)

        click.echo(click.style(f"Workspace '{name}' saved successfully", fg="green"))

    except requests.RequestException as e:
        click.echo(click.style(f"Request failed: {e}", fg="red"), err=True)
        raise SystemExit(1)


@workspace.command()
@click.argument("name")
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
@click.option("--json", "as_json", is_flag=True, help="Output state as JSON")
def load(name: str, port: int, as_json: bool) -> None:
    """Load a saved workspace.

    NAME is the workspace name to load.
    """
    if not check_server(DEFAULT_HOST, port):
        click.echo(click.style(f"Server is not running on port {port}", fg="red"), err=True)
        raise SystemExit(1)

    try:
        response = requests.get(
            f"{get_api_base(DEFAULT_HOST, port)}/load_workspace",
            params={"name": name},
            timeout=HEALTH_CHECK_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "error":
            click.echo(click.style(f"Error: {data.get('message', 'Unknown error')}", fg="red"), err=True)
            raise SystemExit(1)

        state = data.get("data", {}).get("state", {})

        if as_json:
            click.echo(json.dumps(state, indent=2))
        else:
            click.echo(click.style(f"Workspace '{name}' loaded", fg="green"))
            # Show summary of what was loaded
            drawables = state.get("drawables", {})
            total = sum(len(v) if isinstance(v, list) else 0 for v in drawables.values())
            click.echo(f"  Drawables: {total}")

    except requests.RequestException as e:
        click.echo(click.style(f"Request failed: {e}", fg="red"), err=True)
        raise SystemExit(1)


@workspace.command()
@click.argument("name")
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def delete(name: str, port: int, yes: bool) -> None:
    """Delete a saved workspace.

    NAME is the workspace name to delete.
    """
    if not check_server(DEFAULT_HOST, port):
        click.echo(click.style(f"Server is not running on port {port}", fg="red"), err=True)
        raise SystemExit(1)

    if not yes:
        if not click.confirm(f"Delete workspace '{name}'?"):
            click.echo("Cancelled")
            return

    try:
        response = requests.get(
            f"{get_api_base(DEFAULT_HOST, port)}/delete_workspace",
            params={"name": name},
            timeout=HEALTH_CHECK_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "error":
            click.echo(click.style(f"Error: {data.get('message', 'Unknown error')}", fg="red"), err=True)
            raise SystemExit(1)

        click.echo(click.style(f"Workspace '{name}' deleted", fg="green"))

    except requests.RequestException as e:
        click.echo(click.style(f"Request failed: {e}", fg="red"), err=True)
        raise SystemExit(1)
