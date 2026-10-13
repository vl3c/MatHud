"""Tests for cli/server.py."""

from __future__ import annotations

import json
import signal
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional, Sequence
from unittest.mock import MagicMock, patch

import psutil

from cli.server import PORT_RELEASE_SECONDS, RELOADER_ENV_VARS, ServerManager
from cli.config import DEFAULT_HOST, DEFAULT_PORT


class TestServerManagerInit:
    """Test ServerManager initialization."""

    def test_default_host_and_port(self) -> None:
        """ServerManager should use default host and port."""
        manager = ServerManager()
        assert manager.host == DEFAULT_HOST
        assert manager.port == DEFAULT_PORT

    def test_custom_host_and_port(self) -> None:
        """ServerManager should accept custom host and port."""
        manager = ServerManager(host="0.0.0.0", port=8080)
        assert manager.host == "0.0.0.0"
        assert manager.port == 8080

    def test_base_url_format(self) -> None:
        """base_url should be properly formatted."""
        manager = ServerManager(host="127.0.0.1", port=5000)
        assert manager.base_url == "http://127.0.0.1:5000"


class TestServerManagerIsRunning:
    """Test ServerManager.is_server_running method."""

    def test_server_running_returns_true(self) -> None:
        """is_server_running returns True when server responds with 200."""
        manager = ServerManager()

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("cli.server.requests.get", return_value=mock_response):
            assert manager.is_server_running() is True

    def test_server_not_running_returns_false(self) -> None:
        """is_server_running returns False when request fails."""
        import requests as req_module

        manager = ServerManager()

        with patch("cli.server.requests.get", side_effect=req_module.RequestException("Connection refused")):
            assert manager.is_server_running() is False

    def test_server_error_status_returns_false(self) -> None:
        """is_server_running returns False on non-200 status."""
        manager = ServerManager()

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("cli.server.requests.get", return_value=mock_response):
            # Returns False because status_code != 200
            assert manager.is_server_running() is False


class TestServerManagerGetPid:
    """Test ServerManager.get_pid method."""

    def test_no_pid_file_returns_none(self) -> None:
        """get_pid returns None when PID file doesn't exist."""
        manager = ServerManager()

        with patch.object(Path, "exists", return_value=False):
            with patch("cli.server.PID_FILE", Path("/fake/path")):
                assert manager.get_pid() is None

    def test_valid_pid_file_returns_pid(self) -> None:
        """get_pid returns PID when file exists and process running."""
        manager = ServerManager()

        mock_pid_file = MagicMock(spec=Path)
        mock_pid_file.exists.return_value = True
        mock_pid_file.read_text.return_value = "12345"

        mock_process = MagicMock()
        mock_process.cmdline.return_value = ["python", "app.py", "--port", "5000"]

        with patch("cli.server.PID_FILE", mock_pid_file):
            with patch("cli.server.psutil.pid_exists", return_value=True):
                with patch("cli.server.psutil.Process", return_value=mock_process):
                    assert manager.get_pid() == 12345

    def test_legacy_pid_file_non_server_process_is_cleaned_up(self) -> None:
        """Legacy PID files are removed when PID belongs to another process."""
        manager = ServerManager()

        mock_pid_file = MagicMock(spec=Path)
        mock_pid_file.exists.return_value = True
        mock_pid_file.read_text.return_value = "12345"

        mock_process = MagicMock()
        mock_process.cmdline.return_value = ["python", "some_other_script.py"]

        with patch("cli.server.PID_FILE", mock_pid_file):
            with patch("cli.server.psutil.pid_exists", return_value=True):
                with patch("cli.server.psutil.Process", return_value=mock_process):
                    result = manager.get_pid()
                    assert result is None
                    mock_pid_file.unlink.assert_called_once()

    def test_legacy_pid_file_process_cmdline_unreadable_is_cleaned_up(self) -> None:
        """Legacy PID files are removed when process cmdline cannot be inspected."""
        manager = ServerManager()

        mock_pid_file = MagicMock(spec=Path)
        mock_pid_file.exists.return_value = True
        mock_pid_file.read_text.return_value = "12345"

        import psutil

        with patch("cli.server.PID_FILE", mock_pid_file):
            with patch("cli.server.psutil.pid_exists", return_value=True):
                with patch("cli.server.psutil.Process", side_effect=psutil.AccessDenied(pid=12345)):
                    result = manager.get_pid()
                    assert result is None
                    mock_pid_file.unlink.assert_called_once()

    def test_valid_pid_file_json_with_matching_create_time_returns_pid(self) -> None:
        """JSON PID records return PID when create_time matches running process."""
        manager = ServerManager(port=5000)

        mock_pid_file = MagicMock(spec=Path)
        mock_pid_file.exists.return_value = True
        mock_pid_file.read_text.return_value = json.dumps({"pid": 12345, "create_time": 1000.0, "port": 5000})

        mock_process = MagicMock()
        mock_process.create_time.return_value = 1000.2

        with patch("cli.server.PID_FILE", mock_pid_file):
            with patch("cli.server.psutil.pid_exists", return_value=True):
                with patch("cli.server.psutil.Process", return_value=mock_process):
                    assert manager.get_pid() == 12345

    def test_stale_pid_file_returns_none_and_cleans_up(self) -> None:
        """get_pid returns None and removes stale PID file."""
        manager = ServerManager()

        mock_pid_file = MagicMock(spec=Path)
        mock_pid_file.exists.return_value = True
        mock_pid_file.read_text.return_value = "12345"

        with patch("cli.server.PID_FILE", mock_pid_file):
            with patch("cli.server.psutil.pid_exists", return_value=False):
                result = manager.get_pid()
                assert result is None
                mock_pid_file.unlink.assert_called_once()

    def test_pid_file_with_reused_pid_is_cleaned_up(self) -> None:
        """get_pid removes record when PID exists but process create_time mismatches."""
        manager = ServerManager(port=5000)

        mock_pid_file = MagicMock(spec=Path)
        mock_pid_file.exists.return_value = True
        mock_pid_file.read_text.return_value = json.dumps({"pid": 12345, "create_time": 1000.0, "port": 5000})

        mock_process = MagicMock()
        mock_process.create_time.return_value = 2000.0

        with patch("cli.server.PID_FILE", mock_pid_file):
            with patch("cli.server.psutil.pid_exists", return_value=True):
                with patch("cli.server.psutil.Process", return_value=mock_process):
                    result = manager.get_pid()
                    assert result is None
                    mock_pid_file.unlink.assert_called_once()

    def test_pid_file_wrong_port_is_ignored(self) -> None:
        """get_pid ignores records tied to another port."""
        manager = ServerManager(port=5000)

        mock_pid_file = MagicMock(spec=Path)
        mock_pid_file.exists.return_value = True
        mock_pid_file.read_text.return_value = json.dumps({"pid": 12345, "create_time": 1000.0, "port": 5001})

        with patch("cli.server.PID_FILE", mock_pid_file):
            assert manager.get_pid() is None
            mock_pid_file.unlink.assert_not_called()


class TestServerManagerStatus:
    """Test ServerManager.status method."""

    def test_status_when_running(self) -> None:
        """status returns correct info when server is running."""
        manager = ServerManager(port=5000)

        with patch.object(manager, "is_server_running", return_value=True):
            with patch.object(manager, "get_pid", return_value=12345):
                status = manager.status()

                assert status["running"] is True
                assert status["url"] == "http://127.0.0.1:5000"
                assert status["pid"] == 12345
                assert status["port"] == 5000

    def test_status_when_not_running(self) -> None:
        """status returns correct info when server is not running."""
        manager = ServerManager(port=5000)

        with patch.object(manager, "is_server_running", return_value=False):
            with patch.object(manager, "get_pid", return_value=None):
                status = manager.status()

                assert status["running"] is False
                assert status["url"] is None
                assert status["pid"] is None
                assert status["port"] == 5000


class TestServerManagerStart:
    """Test ServerManager.start method."""

    def test_start_when_already_running(self) -> None:
        """start returns error when server already running."""
        manager = ServerManager()

        with patch.object(manager, "is_server_running", return_value=True):
            success, message = manager.start()

            assert success is False
            assert "already running" in message.lower()

    def test_start_with_existing_pid(self) -> None:
        """start returns error when PID exists but server not responding."""
        manager = ServerManager()

        with patch.object(manager, "is_server_running", return_value=False):
            with patch.object(manager, "get_pid", return_value=12345):
                success, message = manager.start()

                assert success is False
                assert "12345" in message

    def test_start_missing_python(self) -> None:
        """start returns error when Python interpreter not found."""
        manager = ServerManager()

        # Create a mock Path that returns False for exists()
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False

        with patch.object(manager, "is_server_running", return_value=False):
            with patch.object(manager, "get_pid", return_value=None):
                with patch("cli.server.get_python_path", return_value=mock_path_instance):
                    success, message = manager.start()

                    assert success is False
                    assert "not found" in message.lower()

    def test_start_when_port_in_use(self) -> None:
        """start returns a clear error when target port is already occupied."""
        manager = ServerManager(port=5000)

        with patch.object(manager, "is_server_running", return_value=False):
            with patch.object(manager, "get_pid", return_value=None):
                mock_path_instance = MagicMock()
                mock_path_instance.exists.return_value = True
                with patch("cli.server.get_python_path", return_value=mock_path_instance):
                    with patch.object(manager, "_is_port_available", return_value=False):
                        with patch.object(manager, "_find_listener_pid_on_port", return_value=41268):
                            success, message = manager.start()
                            assert success is False
                            assert "port 5000 is already in use" in message.lower()
                            assert "41268" in message

    def test_start_auto_increment_uses_next_available_port(self) -> None:
        """start can auto-increment to the next free port when requested."""
        manager = ServerManager(port=5000)
        mock_process = MagicMock()
        mock_process.pid = 24680
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True

        with patch.object(manager, "is_server_running", return_value=False):
            with patch.object(manager, "get_pid", return_value=None):
                with patch("cli.server.get_python_path", return_value=mock_path_instance):
                    with patch.object(manager, "_is_port_available", return_value=False):
                        with patch.object(manager, "_find_next_available_port", return_value=5001):
                            with patch("cli.server.subprocess.Popen", return_value=mock_process) as mock_popen:
                                with patch.object(manager, "_save_pid"):
                                    success, message = manager.start(wait=False, auto_increment_port=True)
                                    assert success is True
                                    assert manager.port == 5001
                                    assert "5001" in message
                                    popen_args = mock_popen.call_args[0][0]
                                    assert "--port" in popen_args
                                    assert "5001" in popen_args

    def test_start_auto_increment_exhausted_returns_port_in_use(self) -> None:
        """start returns a clear error when no fallback port is available."""
        manager = ServerManager(port=5000)

        with patch.object(manager, "is_server_running", return_value=False):
            with patch.object(manager, "get_pid", return_value=None):
                mock_path_instance = MagicMock()
                mock_path_instance.exists.return_value = True
                with patch("cli.server.get_python_path", return_value=mock_path_instance):
                    with patch.object(manager, "_is_port_available", return_value=False):
                        with patch.object(manager, "_find_next_available_port", return_value=None):
                            with patch.object(manager, "_find_listener_pid_on_port", return_value=41268):
                                success, message = manager.start(
                                    auto_increment_port=True,
                                    max_port_tries=3,
                                )
                                assert success is False
                                assert "port 5000 is already in use" in message.lower()
                                assert "41268" in message


class TestServerManagerStop:
    """Test ServerManager.stop method."""

    def test_stop_no_pid(self) -> None:
        """stop returns error when no PID found."""
        manager = ServerManager()

        with patch.object(manager, "get_pid", return_value=None):
            with patch.object(manager, "is_server_running", return_value=False):
                success, message = manager.stop()

                assert success is False
                assert "not running" in message.lower()

    def test_stop_unknown_pid_but_running(self) -> None:
        """stop returns error when server running but PID unknown."""
        manager = ServerManager()

        with patch.object(manager, "get_pid", return_value=None):
            with patch.object(manager, "is_server_running", return_value=True):
                success, message = manager.stop()

                assert success is False
                assert "unknown" in message.lower()

    def test_stop_process_not_found(self) -> None:
        """stop handles NoSuchProcess gracefully."""
        manager = ServerManager()

        import psutil

        with patch.object(manager, "get_pid", return_value=12345):
            with patch("cli.server.psutil.Process", side_effect=psutil.NoSuchProcess(12345)):
                with (
                    patch("cli.server.PID_FILE") as mock_pid_file,
                    patch.object(manager, "_is_port_available", return_value=True),
                ):
                    mock_pid_file.exists.return_value = True

                    success, message = manager.stop()

                    assert success is True
                    assert "not running" in message.lower() or "stale" in message.lower()


# ----------------------------------------------------------------------
# Process tree shutdown (fake processes; no real servers are started)
# ----------------------------------------------------------------------

LAUNCHER_PID = 40084
INTERPRETER_PID = 43556
RELOADER_PID = 43600


class FakeProcess:
    """Stands in for psutil.Process: records signals and exits unless told to ignore them."""

    def __init__(
        self,
        pid: int,
        children: Sequence[FakeProcess] = (),
        *,
        ignores_stop: bool = False,
        survives_kill: bool = False,
        cmdline: Sequence[str] = (),
    ) -> None:
        self.pid = pid
        self._children = list(children)
        self.ignores_stop = ignores_stop
        self.survives_kill = survives_kill
        self._cmdline = list(cmdline)
        self.running = True
        self.signals: list[str] = []

    def children(self, recursive: bool = False) -> list[FakeProcess]:
        if not recursive:
            return list(self._children)
        return [d for c in self._children for d in [c, *c.children(recursive=True)]]

    def cmdline(self) -> list[str]:
        return list(self._cmdline)

    def terminate(self) -> None:
        self._stop("terminate")

    def send_signal(self, sig: int) -> None:
        self._stop(f"signal:{sig}")

    def kill(self) -> None:
        self.signals.append("kill")
        if not self.survives_kill:
            self.running = False

    def _stop(self, name: str) -> None:
        self.signals.append(name)
        if not self.ignores_stop:
            self.running = False


def _fake_wait_procs(
    procs: Sequence[FakeProcess], timeout: Optional[float] = None
) -> tuple[list[FakeProcess], list[FakeProcess]]:
    return [p for p in procs if not p.running], [p for p in procs if p.running]


@contextmanager
def _fake_processes(*roots: FakeProcess, platform: str = "win32") -> Iterator[None]:
    """Route psutil.Process lookups to the fake trees and pretend to run on ``platform``."""
    by_pid = {p.pid: p for root in roots for p in [root, *root.children(recursive=True)]}

    def lookup(pid: int) -> FakeProcess:
        process = by_pid.get(pid)
        if process is None or not process.running:
            raise psutil.NoSuchProcess(pid)
        return process

    with (
        patch("cli.server.psutil.Process", side_effect=lookup),
        patch("cli.server.psutil.wait_procs", side_effect=_fake_wait_procs),
        patch("cli.server.sys", SimpleNamespace(platform=platform)),
        patch("cli.server.PID_FILE") as pid_file,
    ):
        pid_file.exists.return_value = True
        yield


def _venv_launcher_tree(**interpreter_kwargs: Any) -> FakeProcess:
    """The Windows venv layout: Scripts launcher -> base interpreter -> its child process."""
    reloader = FakeProcess(RELOADER_PID)
    interpreter = FakeProcess(INTERPRETER_PID, [reloader], **interpreter_kwargs)
    return FakeProcess(LAUNCHER_PID, [interpreter])


class TestServerManagerStopProcessTree:
    """ServerManager.stop must stop the whole tree and leave the port free."""

    def test_stop_terminates_launcher_and_its_children(self) -> None:
        """The base interpreter under the venv launcher (the real listener) and its child are stopped."""
        manager = ServerManager(port=5131)
        launcher = _venv_launcher_tree()
        interpreter, reloader = launcher.children(recursive=True)

        with _fake_processes(launcher), patch.object(manager, "get_pid", return_value=LAUNCHER_PID):
            with patch.object(manager, "_is_port_available", return_value=True):
                success, message = manager.stop()

        assert success is True
        assert str(LAUNCHER_PID) in message
        for process in (launcher, interpreter, reloader):
            assert process.running is False
            assert process.signals == ["terminate"]

    def test_stop_kills_children_that_ignore_terminate(self) -> None:
        """Processes still alive after the grace period are force-killed."""
        manager = ServerManager(port=5131)
        launcher = _venv_launcher_tree(ignores_stop=True)
        interpreter = launcher.children()[0]

        with _fake_processes(launcher), patch.object(manager, "get_pid", return_value=LAUNCHER_PID):
            with patch.object(manager, "_is_port_available", return_value=True):
                success, _ = manager.stop()

        assert success is True
        assert interpreter.signals == ["terminate", "kill"]
        assert launcher.signals == ["terminate"]

    def test_stop_reports_processes_that_survive_kill(self) -> None:
        """stop fails, naming the PID, when a process cannot be killed; the PID file is kept."""
        manager = ServerManager(port=5131)
        launcher = _venv_launcher_tree(ignores_stop=True, survives_kill=True)

        with _fake_processes(launcher), patch.object(manager, "get_pid", return_value=LAUNCHER_PID):
            with patch.object(manager, "_remove_pid_file") as remove_pid_file:
                success, message = manager.stop()

        assert success is False
        assert str(INTERPRETER_PID) in message
        remove_pid_file.assert_not_called()

    def test_stop_on_posix_signals_the_process_group(self) -> None:
        """On POSIX the process group gets SIGINT; only processes outside it are signalled singly."""
        manager = ServerManager(port=5131)
        outsider = FakeProcess(RELOADER_PID)
        leader = FakeProcess(LAUNCHER_PID, [FakeProcess(INTERPRETER_PID), outsider])
        in_group = leader.children()[0]
        groups = {LAUNCHER_PID: LAUNCHER_PID, INTERPRETER_PID: LAUNCHER_PID, RELOADER_PID: RELOADER_PID}

        def killpg(pgid: int, sig: int) -> None:
            for process in (leader, in_group):
                process.send_signal(sig)

        with _fake_processes(leader, platform="linux"), patch.object(manager, "get_pid", return_value=LAUNCHER_PID):
            with (
                patch.object(ServerManager, "_process_group_of", side_effect=groups.get),
                patch("cli.server.os.killpg", side_effect=killpg, create=True) as mock_killpg,
                patch.object(manager, "_is_port_available", return_value=True),
            ):
                success, _ = manager.stop()

        assert success is True
        mock_killpg.assert_called_once_with(LAUNCHER_PID, signal.SIGINT)
        assert leader.signals == in_group.signals == [f"signal:{signal.SIGINT}"]
        assert outsider.signals == [f"signal:{signal.SIGINT}"]

    def test_stop_stops_leftover_server_still_listening(self) -> None:
        """An orphaned ``app.py --port N`` interpreter left on the port is found and stopped."""
        manager = ServerManager(port=5131)
        launcher = FakeProcess(LAUNCHER_PID)
        orphan = FakeProcess(
            INTERPRETER_PID, cmdline=["C:\\Python311\\python.exe", "C:\\MatHud\\app.py", "--port", "5131"]
        )

        with _fake_processes(launcher, orphan), patch.object(manager, "get_pid", return_value=LAUNCHER_PID):
            with (
                patch.object(manager, "_wait_for_port_release", side_effect=[False, True]),
                patch.object(manager, "_find_listener_pid_on_port", return_value=INTERPRETER_PID),
            ):
                success, _ = manager.stop()

        assert success is True
        assert orphan.signals == ["terminate"]

    def test_stop_leaves_unrelated_listener_alone(self) -> None:
        """A listener that is not a MatHud server on this port is reported, not killed."""
        manager = ServerManager(port=5131)
        launcher = FakeProcess(LAUNCHER_PID)
        other = FakeProcess(INTERPRETER_PID, cmdline=["nginx.exe"])

        with _fake_processes(launcher, other), patch.object(manager, "get_pid", return_value=LAUNCHER_PID):
            with (
                patch.object(manager, "_wait_for_port_release", return_value=False),
                patch.object(manager, "_find_listener_pid_on_port", return_value=INTERPRETER_PID),
            ):
                success, message = manager.stop()

        assert success is False
        assert "5131" in message and str(INTERPRETER_PID) in message
        assert other.signals == []

    def test_stop_with_stale_pid_still_checks_the_port(self) -> None:
        """A dead launcher whose interpreter still listens is cleaned up through the port check."""
        manager = ServerManager(port=5131)
        orphan = FakeProcess(INTERPRETER_PID, cmdline=["python.exe", "app.py", "-p", "5131"])

        with _fake_processes(orphan), patch.object(manager, "get_pid", return_value=LAUNCHER_PID):
            with (
                patch.object(manager, "_wait_for_port_release", side_effect=[False, True]),
                patch.object(manager, "_find_listener_pid_on_port", return_value=INTERPRETER_PID),
            ):
                success, message = manager.stop()

        assert success is True
        assert "stale" in message.lower()
        assert orphan.running is False


class TestServerManagerPortRelease:
    """Port checks used after shutdown."""

    def test_wait_for_port_release_polls_until_free(self) -> None:
        manager = ServerManager(port=5131)
        with (
            patch.object(manager, "_is_port_available", side_effect=[False, False, True]),
            patch("cli.server.time.sleep") as mock_sleep,
        ):
            assert manager._wait_for_port_release() is True
        assert mock_sleep.call_count == 2

    def test_wait_for_port_release_times_out(self) -> None:
        manager = ServerManager(port=5131)
        clock = iter([0.0, 1.0, PORT_RELEASE_SECONDS + 1.0])
        with (
            patch.object(manager, "_is_port_available", return_value=False),
            patch("cli.server.time.monotonic", side_effect=lambda: next(clock)),
            patch("cli.server.time.sleep"),
        ):
            assert manager._wait_for_port_release() is False

    def test_port_probe_skips_reuseaddr_on_windows(self) -> None:
        """SO_REUSEADDR on Windows binds over a live listener, so the probe must not set it."""
        manager = ServerManager(port=5131)
        for platform, expect_reuse in (("win32", False), ("linux", True)):
            with (
                patch("cli.server.socket.socket") as mock_socket,
                patch("cli.server.sys", SimpleNamespace(platform=platform)),
            ):
                sock = mock_socket.return_value.__enter__.return_value
                assert manager._is_port_available() is True
            assert sock.setsockopt.called is expect_reuse
            sock.bind.assert_called_once_with((manager.host, 5131))

    def test_is_server_process_for_port_matches_app_and_port(self) -> None:
        manager = ServerManager(port=5131)
        cases = {
            ("python.exe", "C:\\MatHud\\app.py", "--port", "5131"): True,
            ("python", "/srv/MatHud/app.py", "-p", "5131"): True,
            ("python.exe", "app.py", "--port", "5132"): False,
            ("python.exe", "other.py", "--port", "5131"): False,
            ("python.exe", "app.py"): False,
        }
        for cmdline, expected in cases.items():
            with _fake_processes(FakeProcess(INTERPRETER_PID, cmdline=cmdline)):
                assert manager._is_server_process_for_port(INTERPRETER_PID) is expected, cmdline


class TestServerManagerStartEnvironment:
    """The CLI-started server runs without debug mode or inherited reloader state."""

    def test_start_disables_debug_and_strips_reloader_variables(self) -> None:
        manager = ServerManager(port=5131)
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        inherited = {"WERKZEUG_RUN_MAIN": "true", "WERKZEUG_SERVER_FD": "7"}

        with (
            patch.dict("cli.server.os.environ", inherited),
            patch.object(manager, "is_server_running", return_value=False),
            patch.object(manager, "get_pid", return_value=None),
            patch("cli.server.get_python_path", return_value=mock_path),
            patch.object(manager, "_is_port_available", return_value=True),
            patch("cli.server.subprocess.Popen", return_value=MagicMock(pid=LAUNCHER_PID)) as mock_popen,
            patch.object(manager, "_save_pid"),
        ):
            success, _ = manager.start(wait=False)

        assert success is True
        env = mock_popen.call_args.kwargs["env"]
        assert env["MATHUD_NON_DEBUG"] == "1"
        assert not set(RELOADER_ENV_VARS) & set(env)
