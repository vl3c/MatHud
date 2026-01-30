"""Server management commands for the MatHud CLI.

Provides start/stop/status commands for the Flask server.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Optional

import click
import psutil
import requests

from cli.config import (
    APP_MODULE,
    DEFAULT_HOST,
    DEFAULT_PORT,
    HEALTH_CHECK_RETRIES,
    HEALTH_CHECK_TIMEOUT,
    PID_FILE,
    PROJECT_ROOT,
    get_python_path,
)


class ServerManager:
    """Manages the MatHud Flask server process."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        """Initialize the server manager.

        Args:
            host: Host address to bind the server to.
            port: Port number for the server.
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    def is_server_running(self) -> bool:
        """Check if the server is running by making a health check request.

        Returns:
            True if server responds to requests, False otherwise.
        """
        try:
            response = requests.get(
                f"{self.base_url}/auth_status",
                timeout=HEALTH_CHECK_TIMEOUT,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_pid(self) -> Optional[int]:
        """Get the PID of the running server from the PID file.

        Returns:
            The PID if available and process exists, None otherwise.
        """
        if not PID_FILE.exists():
            return None

        try:
            pid = int(PID_FILE.read_text().strip())
            # Check if process exists
            if psutil.pid_exists(pid):
                return pid
            # PID file exists but process doesn't - clean up
            PID_FILE.unlink()
            return None
        except (ValueError, OSError):
            return None

    def _save_pid(self, pid: int) -> None:
        """Save the server PID to the PID file.

        Args:
            pid: The process ID to save.
        """
        PID_FILE.write_text(str(pid))

    def _remove_pid_file(self) -> None:
        """Remove the PID file if it exists."""
        if PID_FILE.exists():
            PID_FILE.unlink()

    def start(self, wait: bool = True) -> tuple[bool, str]:
        """Start the Flask server in the background.

        Args:
            wait: If True, wait for the server to be ready before returning.

        Returns:
            Tuple of (success, message).
        """
        # Check if already running
        if self.is_server_running():
            return False, f"Server is already running at {self.base_url}"

        # Check for stale PID
        existing_pid = self.get_pid()
        if existing_pid:
            return False, f"Server process {existing_pid} exists but is not responding. Use 'mathud server stop' first."

        python_path = get_python_path()
        if not python_path.exists():
            return False, f"Python interpreter not found at {python_path}. Run from project root with venv activated."

        # Set environment to disable auth for CLI operations
        env = os.environ.copy()
        env["REQUIRE_AUTH"] = "false"

        # Start server as background process
        # Use CREATE_NEW_PROCESS_GROUP on Windows, start_new_session on Unix
        kwargs: dict = {
            "cwd": str(PROJECT_ROOT),
            "env": env,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }

        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True

        try:
            process = subprocess.Popen(
                [str(python_path), str(APP_MODULE), "--port", str(self.port)],
                **kwargs,
            )
            self._save_pid(process.pid)
        except Exception as e:
            return False, f"Failed to start server: {e}"

        if not wait:
            return True, f"Server starting on {self.base_url} (PID: {process.pid})"

        # Wait for server to be ready
        for i in range(HEALTH_CHECK_RETRIES):
            time.sleep(1)
            if self.is_server_running():
                return True, f"Server started on {self.base_url} (PID: {process.pid})"

        # Server didn't start in time
        self.stop()  # Clean up
        return False, f"Server failed to start within {HEALTH_CHECK_RETRIES} seconds"

    def stop(self) -> tuple[bool, str]:
        """Stop the running server.

        Returns:
            Tuple of (success, message).
        """
        pid = self.get_pid()

        if pid is None:
            # No PID file, but check if something is responding
            if self.is_server_running():
                return False, f"Server is running at {self.base_url} but PID is unknown. Stop it manually."
            return False, "Server is not running"

        try:
            process = psutil.Process(pid)

            # Try graceful termination first
            if sys.platform == "win32":
                process.terminate()
            else:
                process.send_signal(signal.SIGINT)

            # Wait for process to terminate
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                # Force kill if graceful shutdown failed
                process.kill()
                process.wait(timeout=5)

            self._remove_pid_file()
            return True, f"Server stopped (PID: {pid})"

        except psutil.NoSuchProcess:
            self._remove_pid_file()
            return True, "Server was not running (stale PID file removed)"
        except Exception as e:
            return False, f"Failed to stop server: {e}"

    def status(self) -> dict:
        """Get the current server status.

        Returns:
            Dictionary with status information.
        """
        pid = self.get_pid()
        running = self.is_server_running()

        return {
            "running": running,
            "url": self.base_url if running else None,
            "pid": pid,
            "port": self.port,
        }


@click.group()
def server() -> None:
    """Manage the MatHud Flask server."""
    pass


@server.command()
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Port to run the server on (default: {DEFAULT_PORT})",
)
@click.option(
    "--no-wait",
    is_flag=True,
    help="Don't wait for server to be ready before returning",
)
def start(port: int, no_wait: bool) -> None:
    """Start the Flask server in the background."""
    manager = ServerManager(port=port)
    success, message = manager.start(wait=not no_wait)

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="red"), err=True)
        raise SystemExit(1)


@server.command()
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Port the server is running on (default: {DEFAULT_PORT})",
)
def stop(port: int) -> None:
    """Stop the running server."""
    manager = ServerManager(port=port)
    success, message = manager.stop()

    if success:
        click.echo(click.style(message, fg="green"))
    else:
        click.echo(click.style(message, fg="yellow"), err=True)
        raise SystemExit(1)


@server.command()
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Port to check (default: {DEFAULT_PORT})",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output status as JSON",
)
def status(port: int, as_json: bool) -> None:
    """Check if the server is running."""
    manager = ServerManager(port=port)
    info = manager.status()

    if as_json:
        import json
        click.echo(json.dumps(info))
    else:
        if info["running"]:
            click.echo(click.style(f"Server is running at {info['url']} (PID: {info['pid']})", fg="green"))
        else:
            click.echo(click.style(f"Server is not running on port {info['port']}", fg="yellow"))
            if info["pid"]:
                click.echo(click.style(f"  (stale PID file found: {info['pid']})", fg="yellow"))
