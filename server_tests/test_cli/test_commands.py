"""Tests for CLI Click commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from cli.main import cli


class TestMainCli:
    """Test main CLI entry point."""

    def test_cli_help(self) -> None:
        """CLI should show help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "MatHud" in result.output
        assert "server" in result.output
        assert "test" in result.output
        assert "canvas" in result.output
        assert "workspace" in result.output
        assert "chat" in result.output
        assert "screenshot" in result.output


class TestServerCommands:
    """Test server subcommands."""

    def test_server_help(self) -> None:
        """server command should show help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["server", "--help"])

        assert result.exit_code == 0
        assert "start" in result.output
        assert "stop" in result.output
        assert "status" in result.output

    @patch("cli.server.ServerManager")
    def test_server_status_not_running(self, mock_manager_class: MagicMock) -> None:
        """server status shows not running."""
        mock_manager = MagicMock()
        mock_manager.status.return_value = {
            "running": False,
            "url": None,
            "pid": None,
            "port": 5000,
        }
        mock_manager_class.return_value = mock_manager

        runner = CliRunner()
        result = runner.invoke(cli, ["server", "status"])

        assert "not running" in result.output.lower()

    @patch("cli.server.ServerManager")
    def test_server_status_running(self, mock_manager_class: MagicMock) -> None:
        """server status shows running."""
        mock_manager = MagicMock()
        mock_manager.status.return_value = {
            "running": True,
            "url": "http://127.0.0.1:5000",
            "pid": 12345,
            "port": 5000,
        }
        mock_manager_class.return_value = mock_manager

        runner = CliRunner()
        result = runner.invoke(cli, ["server", "status"])

        assert result.exit_code == 0
        assert "running" in result.output.lower()
        assert "12345" in result.output

    @patch("cli.server.ServerManager")
    def test_server_status_json(self, mock_manager_class: MagicMock) -> None:
        """server status --json outputs JSON."""
        mock_manager = MagicMock()
        mock_manager.status.return_value = {
            "running": True,
            "url": "http://127.0.0.1:5000",
            "pid": 12345,
            "port": 5000,
        }
        mock_manager_class.return_value = mock_manager

        runner = CliRunner()
        result = runner.invoke(cli, ["server", "status", "--json"])

        assert result.exit_code == 0
        assert '"running": true' in result.output

    @patch("cli.server.ServerManager")
    def test_server_start_success(self, mock_manager_class: MagicMock) -> None:
        """server start succeeds."""
        mock_manager = MagicMock()
        mock_manager.start.return_value = (True, "Server started")
        mock_manager_class.return_value = mock_manager

        runner = CliRunner()
        result = runner.invoke(cli, ["server", "start"])

        assert result.exit_code == 0
        assert "started" in result.output.lower()

    @patch("cli.server.ServerManager")
    def test_server_start_failure(self, mock_manager_class: MagicMock) -> None:
        """server start fails."""
        mock_manager = MagicMock()
        mock_manager.start.return_value = (False, "Already running")
        mock_manager_class.return_value = mock_manager

        runner = CliRunner()
        result = runner.invoke(cli, ["server", "start"])

        assert result.exit_code == 1

    @patch("cli.server.ServerManager")
    def test_server_stop_success(self, mock_manager_class: MagicMock) -> None:
        """server stop succeeds."""
        mock_manager = MagicMock()
        mock_manager.stop.return_value = (True, "Server stopped")
        mock_manager_class.return_value = mock_manager

        runner = CliRunner()
        result = runner.invoke(cli, ["server", "stop"])

        assert result.exit_code == 0
        assert "stopped" in result.output.lower()


class TestTestCommands:
    """Test test subcommands."""

    def test_test_help(self) -> None:
        """test command should show help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["test", "--help"])

        assert result.exit_code == 0
        assert "server" in result.output
        assert "client" in result.output
        assert "all" in result.output

    @patch("cli.tests.run_server_tests")
    def test_test_server(self, mock_run: MagicMock) -> None:
        """test server runs pytest."""
        mock_run.return_value = 0

        runner = CliRunner()
        result = runner.invoke(cli, ["test", "server"])

        assert result.exit_code == 0
        mock_run.assert_called_once()

    @patch("cli.tests.run_client_tests")
    def test_test_client_server_not_running(self, mock_run: MagicMock) -> None:
        """test client fails when server not running."""
        mock_run.return_value = {
            "status": "error",
            "error": "Server is not running",
        }

        runner = CliRunner()
        result = runner.invoke(cli, ["test", "client"])

        assert result.exit_code == 1
        assert "error" in result.output.lower()

    @patch("cli.tests.run_client_tests")
    def test_test_client_captures_screenshot_by_default(self, mock_run: MagicMock) -> None:
        """test client captures screenshot by default."""
        mock_run.return_value = {
            "status": "complete",
            "tests_run": 10,
            "failures": 0,
            "errors": 0,
            "screenshot": "cli/output/test_results_20240101_120000.png",
        }
        runner = CliRunner()
        result = runner.invoke(cli, ["test", "client"])

        assert result.exit_code == 0
        assert "Screenshot saved to:" in result.output
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["capture_screenshot"] is True

    @patch("cli.tests.run_client_tests")
    def test_test_client_no_screenshot_flag(self, mock_run: MagicMock) -> None:
        """test client --no-screenshot disables screenshot."""
        mock_run.return_value = {"status": "complete", "tests_run": 10, "failures": 0, "errors": 0}
        runner = CliRunner()
        result = runner.invoke(cli, ["test", "client", "--no-screenshot"])

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["capture_screenshot"] is False
        assert "Screenshot saved to:" not in result.output

    @patch("cli.tests.run_client_tests")
    def test_test_client_custom_screenshot_path(self, mock_run: MagicMock) -> None:
        """test client -o uses custom screenshot path."""
        mock_run.return_value = {
            "status": "complete",
            "tests_run": 10,
            "failures": 0,
            "errors": 0,
            "screenshot": "custom_results.png",
        }
        runner = CliRunner()
        result = runner.invoke(cli, ["test", "client", "-o", "custom_results.png"])

        assert result.exit_code == 0
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["screenshot_path"] == "custom_results.png"

    @patch("cli.tests.run_client_tests")
    def test_test_client_screenshot_shown_on_failure(self, mock_run: MagicMock) -> None:
        """test client shows screenshot path even when tests fail."""
        mock_run.return_value = {
            "status": "complete",
            "tests_run": 10,
            "failures": 2,
            "errors": 0,
            "failing_tests": [{"test": "test_foo", "error": "Failed"}],
            "screenshot": "cli/output/test_results_20240101_120000.png",
        }
        runner = CliRunner()
        result = runner.invoke(cli, ["test", "client"])

        assert result.exit_code == 1
        assert "Screenshot saved to:" in result.output


class TestCanvasCommands:
    """Test canvas subcommands."""

    def test_canvas_help(self) -> None:
        """canvas command should show help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["canvas", "--help"])

        assert result.exit_code == 0
        assert "clear" in result.output
        assert "reset" in result.output
        assert "zoom" in result.output
        assert "undo" in result.output
        assert "redo" in result.output
        assert "state" in result.output
        assert "exec" in result.output


class TestWorkspaceCommands:
    """Test workspace subcommands."""

    def test_workspace_help(self) -> None:
        """workspace command should show help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["workspace", "--help"])

        assert result.exit_code == 0
        assert "list" in result.output
        assert "save" in result.output
        assert "load" in result.output
        assert "delete" in result.output

    @patch("cli.workspace.check_server")
    def test_workspace_list_server_not_running(self, mock_check: MagicMock) -> None:
        """workspace list fails when server not running."""
        mock_check.return_value = False

        runner = CliRunner()
        result = runner.invoke(cli, ["workspace", "list"])

        assert result.exit_code == 1
        assert "not running" in result.output.lower()


class TestChatCommands:
    """Test chat subcommands."""

    def test_chat_help(self) -> None:
        """chat command should show help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "--help"])

        assert result.exit_code == 0
        assert "send" in result.output
        assert "new" in result.output

    @patch("cli.chat.check_server")
    def test_chat_send_server_not_running(self, mock_check: MagicMock) -> None:
        """chat send fails when server not running."""
        mock_check.return_value = False

        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "send", "Hello"])

        assert result.exit_code == 1
        assert "not running" in result.output.lower()


class TestScreenshotCommands:
    """Test screenshot subcommands."""

    def test_screenshot_help(self) -> None:
        """screenshot command should show help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["screenshot", "--help"])

        assert result.exit_code == 0
        assert "capture" in result.output

    @patch("cli.screenshot.ServerManager")
    def test_screenshot_capture_server_not_running(self, mock_manager_class: MagicMock) -> None:
        """screenshot capture fails when server not running."""
        mock_manager = MagicMock()
        mock_manager.is_server_running.return_value = False
        mock_manager_class.return_value = mock_manager

        runner = CliRunner()
        result = runner.invoke(cli, ["screenshot", "capture"])

        assert result.exit_code == 1
        assert "not running" in result.output.lower()
