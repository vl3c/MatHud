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
from pathlib import Path
from typing import Any, Optional, Sequence

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

# Seconds to wait after the stop signal before force-killing the process tree.
STOP_GRACE_SECONDS = 10
# Seconds to wait for force-killed processes to exit.
KILL_WAIT_SECONDS = 5
# Seconds to wait for the port to be released once the processes have exited.
PORT_RELEASE_SECONDS = 5
PORT_RELEASE_POLL_SECONDS = 0.2
# Werkzeug reloader variables a server must not inherit: with WERKZEUG_RUN_MAIN set,
# run_simple() treats the process as a reloader child and reuses WERKZEUG_SERVER_FD.
RELOADER_ENV_VARS = ("WERKZEUG_RUN_MAIN", "WERKZEUG_SERVER_FD")


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
        return self._is_port_available_for(self.port)

    def _is_port_available_for(self, port: int) -> bool:
        """Check whether host:port can be bound by a new process."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # On Windows SO_REUSEADDR lets a bind succeed on a port another socket
            # is listening on, so the probe would report a busy port as free.
            if sys.platform != "win32":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((self.host, port))
                return True
            except OSError:
                return False

    def _set_port(self, port: int) -> None:
        """Update active port and derived base URL."""
        self.port = port
        self.base_url = f"http://{self.host}:{port}"

    def _find_next_available_port(self, max_attempts: int = 10) -> Optional[int]:
        """Find next bindable port after current one.

        Args:
            max_attempts: Number of sequential ports to probe.

        Returns:
            Next available port, or None if none found in range.
        """
        if max_attempts <= 0:
            return None
        for candidate in range(self.port + 1, self.port + 1 + max_attempts):
            if self._is_port_available_for(candidate):
                return candidate
        return None

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

    def start(
        self,
        wait: bool = True,
        *,
        auto_increment_port: bool = False,
        max_port_tries: int = 10,
        extra_env: Optional[dict[str, str]] = None,
    ) -> tuple[bool, str]:
        """Start the Flask server in the background.

        Args:
            wait: If True, wait for the server to be ready before returning.
            auto_increment_port: If True, try next ports when requested port is occupied.
            max_port_tries: Maximum number of next ports to try when auto incrementing.
            extra_env: Environment variables to set for the server process
                (e.g. ``MATHUD_WORKSPACES_DIR`` for test runs).

        Returns:
            Tuple of (success, message).
        """
        # Check if already running
        if self.is_server_running():
            return False, f"Server is already running at {self.base_url}"

        # Check for stale PID before port probing, so tests/diagnostics remain deterministic
        # even when default ports are occupied by unrelated processes.
        existing_pid = self.get_pid()
        if existing_pid:
            return False, f"Server process {existing_pid} exists but is not responding. Use 'mathud server stop' first."

        python_path = get_python_path()
        if not python_path.exists():
            return False, f"Python interpreter not found at {python_path}. Run from project root with venv activated."

        # Fail fast when another process already owns the port.
        if not self._is_port_available():
            if auto_increment_port:
                next_port = self._find_next_available_port(max_attempts=max_port_tries)
                if next_port is not None:
                    self._set_port(next_port)
                else:
                    owner_pid = self._find_listener_pid_on_port()
                    if owner_pid is not None:
                        return (
                            False,
                            f"Port {self.port} is already in use by PID {owner_pid}. "
                            f"Choose another port or stop that process.",
                        )
                    return (
                        False,
                        f"Port {self.port} is already in use. Choose another port or stop the existing listener.",
                    )
            else:
                owner_pid = self._find_listener_pid_on_port()
                if owner_pid is not None:
                    return (
                        False,
                        f"Port {self.port} is already in use by PID {owner_pid}. "
                        f"Choose another port or stop that process.",
                    )
                return (
                    False,
                    f"Port {self.port} is already in use. Choose another port or stop the existing listener.",
                )

        # Set environment to disable auth for CLI operations
        env = os.environ.copy()
        env["REQUIRE_AUTH"] = "false"
        # Force non-debug server mode for CLI-managed processes without
        # triggering deployment-mode auth behaviors.
        env["MATHUD_NON_DEBUG"] = "1"
        for name in RELOADER_ENV_VARS:
            env.pop(name, None)
        if extra_env:
            env.update(extra_env)

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
        """Stop the running server and its child processes, then wait for the port to be free.

        A venv interpreter on Windows is a launcher that runs the base interpreter as a
        child process, so the recorded PID is not the process listening on the port.

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
            was_running = self._stop_process_tree(pid)
        except Exception as e:
            return False, f"Failed to stop server: {e}"

        self._remove_pid_file()
        port_error = self._release_port()
        if port_error:
            return False, port_error
        if not was_running:
            return True, "Server was not running (stale PID file removed)"
        return True, f"Server stopped (PID: {pid})"

    def _stop_process_tree(self, pid: int) -> bool:
        """Stop a process and all its descendants, force-killing any that outlive the grace period.

        Returns:
            False if the process no longer existed, True once the whole tree has exited.

        Raises:
            RuntimeError: If processes are still alive after being killed.
        """
        try:
            root = psutil.Process(pid)
            # Collect descendants before stopping the root: once it exits they are orphaned.
            processes = [root, *root.children(recursive=True)]
        except psutil.NoSuchProcess:
            return False

        self._signal_tree(pid, processes, kill=False)
        _, alive = psutil.wait_procs(processes, timeout=STOP_GRACE_SECONDS)
        if alive:
            self._signal_tree(pid, alive, kill=True)
            _, alive = psutil.wait_procs(alive, timeout=KILL_WAIT_SECONDS)
        if alive:
            raise RuntimeError(f"processes still running: {', '.join(str(p.pid) for p in alive)}")
        return True

    def _signal_tree(self, root_pid: int, processes: Sequence[psutil.Process], *, kill: bool) -> None:
        """Send the stop signal (SIGINT, or TerminateProcess on Windows) or a kill to each process.

        On POSIX the server leads its own session, so its process group is signalled as a
        whole, which also reaches group members that were orphaned out of the tree.
        """
        if sys.platform != "win32" and self._process_group_of(root_pid) == root_pid:
            try:
                os.killpg(root_pid, signal.SIGKILL if kill else signal.SIGINT)
            except OSError:
                pass
            processes = [p for p in processes if self._process_group_of(p.pid) != root_pid]

        for process in processes:
            try:
                if kill:
                    process.kill()
                elif sys.platform == "win32":
                    process.terminate()
                else:
                    process.send_signal(signal.SIGINT)
            except psutil.NoSuchProcess:
                pass

    @staticmethod
    def _process_group_of(pid: int) -> Optional[int]:
        """Process group ID of ``pid`` (POSIX only), or None when unavailable."""
        if sys.platform == "win32":
            return None
        try:
            return os.getpgid(pid)
        except OSError:
            return None

    def _release_port(self) -> Optional[str]:
        """Wait for the port to be free, stopping a leftover MatHud server still listening on it.

        Returns:
            None once the port is free, otherwise an error message.
        """
        if self._wait_for_port_release():
            return None

        owner_pid = self._find_listener_pid_on_port()
        if owner_pid is not None and self._is_server_process_for_port(owner_pid):
            try:
                self._stop_process_tree(owner_pid)
            except Exception as e:
                return f"Server stopped, but leftover server PID {owner_pid} on port {self.port} survived: {e}"
            if self._wait_for_port_release():
                return None

        owner = f" by PID {owner_pid}" if owner_pid is not None else ""
        return f"Server stopped, but port {self.port} is still in use{owner}"

    def _wait_for_port_release(self) -> bool:
        """Poll until the port can be bound, for up to PORT_RELEASE_SECONDS."""
        deadline = time.monotonic() + PORT_RELEASE_SECONDS
        while not self._is_port_available():
            if time.monotonic() >= deadline:
                return False
            time.sleep(PORT_RELEASE_POLL_SECONDS)
        return True

    def _is_server_process_for_port(self, pid: int) -> bool:
        """Whether ``pid`` runs ``app.py --port <self.port>``, as the CLI starts servers."""
        try:
            args = psutil.Process(pid).cmdline()
        except (psutil.Error, OSError):
            return False
        runs_app = any(Path(arg).name == APP_MODULE.name for arg in args)
        on_port = any(flag in ("--port", "-p") and value == str(self.port) for flag, value in zip(args, args[1:]))
        return runs_app and on_port

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
