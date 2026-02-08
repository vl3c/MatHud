"""Tests for cli/server.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


from cli.server import ServerManager
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
        mock_pid_file.read_text.return_value = json.dumps(
            {"pid": 12345, "create_time": 1000.0, "port": 5000}
        )

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
        mock_pid_file.read_text.return_value = json.dumps(
            {"pid": 12345, "create_time": 1000.0, "port": 5000}
        )

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
        mock_pid_file.read_text.return_value = json.dumps(
            {"pid": 12345, "create_time": 1000.0, "port": 5001}
        )

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
            with patch.object(manager, "_is_port_available", return_value=False):
                with patch.object(manager, "_find_listener_pid_on_port", return_value=41268):
                    success, message = manager.start()
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
                with patch("cli.server.PID_FILE") as mock_pid_file:
                    mock_pid_file.exists.return_value = True

                    success, message = manager.stop()

                    assert success is True
                    assert "not running" in message.lower() or "stale" in message.lower()
