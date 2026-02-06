"""Server management commands for the MatHud CLI.

Provides start/stop/status commands for the Flask server.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Optional

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

    def _load_pid_record(self) -> Optional[dict[str, Any]]:
        """Load PID record from disk.

        Supports both legacy plain integer format and JSON records.
        """
        if not PID_FILE.exists():
            return None

        try:
            raw = PID_FILE.read_text().strip()
            if not raw:
                return None

            # Backward compatibility with legacy format: plain PID text.
            if raw.isdigit():
                return {"pid": int(raw), "create_time": None, "port": None}

            data = json.loads(raw)
            if isinstance(data, dict) and isinstance(data.get("pid"), int):
                return data
        except (OSError, ValueError, json.JSONDecodeError):
            return None

        return None

    def _is_pid_record_current(self, pid: int, create_time: Any) -> bool:
        """Validate that PID points to the same process instance."""
        if not psutil.pid_exists(pid):
            return False

        # If create_time is unavailable (legacy records), validate process command.
        if create_time is None:
            try:
                process = psutil.Process(pid)
                cmdline = " ".join(process.cmdline()).lower()
                app_path = str(APP_MODULE).lower()
                return app_path in cmdline or "app.py" in cmdline
            except (psutil.Error, OSError):
                return False

        try:
            process = psutil.Process(pid)
            current_create_time = process.create_time()
            # Tolerate sub-second rounding differences across platforms.
            return abs(current_create_time - float(create_time)) < 1.0
        except (psutil.Error, OSError, ValueError, TypeError):
            return False

    def _is_port_available(self) -> bool:
        """Check whether host:port can be bound by a new process."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((self.host, self.port))
                return True
            except OSError:
                return False

    def _find_listener_pid_on_port(self) -> Optional[int]:
        """Best-effort lookup for PID currently listening on host:port."""
        try:
            for conn in psutil.net_connections(kind="tcp"):
                if conn.status != psutil.CONN_LISTEN:
                    continue
                laddr = conn.laddr
                if not laddr:
                    continue
                if laddr.port != self.port:
                    continue
                if laddr.ip not in (self.host, "0.0.0.0", "::", "::1"):
                    continue
                if conn.pid is not None:
                    return conn.pid
        except (psutil.Error, OSError):
            return None
        return None

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
        record = self._load_pid_record()
        if not record:
            return None

        pid = record.get("pid")
        if not isinstance(pid, int):
            self._remove_pid_file()
            return None

        # Optional port affinity check for JSON records.
        record_port = record.get("port")
        if isinstance(record_port, int) and record_port != self.port:
            return None

        if self._is_pid_record_current(pid, record.get("create_time")):
            return pid

        # PID file exists but process doesn't match anymore - clean up
        self._remove_pid_file()
        return None

    def _save_pid(self, pid: int) -> None:
        """Save the server PID to the PID file.

        Args:
            pid: The process ID to save.
        """
        create_time: Optional[float] = None
        try:
            create_time = psutil.Process(pid).create_time()
        except (psutil.Error, OSError):
            create_time = None

        PID_FILE.write_text(
            json.dumps(
                {
                    "pid": pid,
                    "create_time": create_time,
                    "port": self.port,
                }
            )
        )

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

        # Fail fast when another process already owns the port.
        if not self._is_port_available():
            owner_pid = self._find_listener_pid_on_port()
            if owner_pid is not None:
                return (
                    False,
                    f"Port {self.port} is already in use by PID {owner_pid}. "
                    f"Choose another port or stop that process.",
                )
            return (
                False,
                f"Port {self.port} is already in use. "
                f"Choose another port or stop the existing listener.",
            )

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
