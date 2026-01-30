"""Tests for cli/config.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


class TestConfigConstants:
    """Test configuration constants."""

    def test_project_root_exists(self) -> None:
        """PROJECT_ROOT should point to an existing directory."""
        from cli.config import PROJECT_ROOT
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_project_root_contains_app_py(self) -> None:
        """PROJECT_ROOT should contain app.py."""
        from cli.config import PROJECT_ROOT
        assert (PROJECT_ROOT / "app.py").exists()

    def test_default_port_is_valid(self) -> None:
        """DEFAULT_PORT should be a valid port number."""
        from cli.config import DEFAULT_PORT
        assert isinstance(DEFAULT_PORT, int)
        assert 1 <= DEFAULT_PORT <= 65535

    def test_default_host_is_localhost(self) -> None:
        """DEFAULT_HOST should be localhost."""
        from cli.config import DEFAULT_HOST
        assert DEFAULT_HOST == "127.0.0.1"

    def test_pid_file_path(self) -> None:
        """PID_FILE should be in project root."""
        from cli.config import PID_FILE, PROJECT_ROOT
        assert PID_FILE.parent == PROJECT_ROOT
        assert PID_FILE.name == ".mathud_server.pid"

    def test_health_check_timeout_positive(self) -> None:
        """HEALTH_CHECK_TIMEOUT should be positive."""
        from cli.config import HEALTH_CHECK_TIMEOUT
        assert HEALTH_CHECK_TIMEOUT > 0

    def test_health_check_retries_positive(self) -> None:
        """HEALTH_CHECK_RETRIES should be positive."""
        from cli.config import HEALTH_CHECK_RETRIES
        assert HEALTH_CHECK_RETRIES > 0

    def test_browser_wait_timeout_positive(self) -> None:
        """BROWSER_WAIT_TIMEOUT should be positive."""
        from cli.config import BROWSER_WAIT_TIMEOUT
        assert BROWSER_WAIT_TIMEOUT > 0

    def test_app_ready_timeout_positive(self) -> None:
        """APP_READY_TIMEOUT should be positive."""
        from cli.config import APP_READY_TIMEOUT
        assert APP_READY_TIMEOUT > 0

    def test_client_test_timeout_positive(self) -> None:
        """CLIENT_TEST_TIMEOUT should be positive."""
        from cli.config import CLIENT_TEST_TIMEOUT
        assert CLIENT_TEST_TIMEOUT > 0

    def test_viewport_dimensions_positive(self) -> None:
        """Viewport dimensions should be positive."""
        from cli.config import DEFAULT_VIEWPORT_WIDTH, DEFAULT_VIEWPORT_HEIGHT
        assert DEFAULT_VIEWPORT_WIDTH > 0
        assert DEFAULT_VIEWPORT_HEIGHT > 0

    def test_cli_output_dir_exists(self) -> None:
        """CLI_OUTPUT_DIR should exist."""
        from cli.config import CLI_OUTPUT_DIR
        assert CLI_OUTPUT_DIR.exists()
        assert CLI_OUTPUT_DIR.is_dir()

    def test_cli_output_dir_is_in_cli_module(self) -> None:
        """CLI_OUTPUT_DIR should be inside cli/ directory."""
        from cli.config import CLI_OUTPUT_DIR
        assert CLI_OUTPUT_DIR.name == "output"
        assert CLI_OUTPUT_DIR.parent.name == "cli"


class TestGetPythonPath:
    """Test get_python_path function."""

    def test_returns_path_object(self) -> None:
        """get_python_path should return a Path."""
        from cli.config import get_python_path
        result = get_python_path()
        assert isinstance(result, Path)

    def test_path_ends_with_python(self) -> None:
        """Path should end with python executable."""
        from cli.config import get_python_path
        result = get_python_path()
        assert "python" in result.name.lower()

    def test_path_is_in_venv(self) -> None:
        """Path should be in venv directory."""
        from cli.config import get_python_path
        result = get_python_path()
        assert "venv" in str(result)

    def test_windows_path_structure(self) -> None:
        """On Windows, path should use Scripts directory."""
        from cli.config import get_python_path
        if os.name == "nt":
            result = get_python_path()
            assert "Scripts" in str(result)

    def test_unix_path_structure(self) -> None:
        """On Unix, path should use bin directory."""
        from cli.config import get_python_path
        if os.name != "nt":
            result = get_python_path()
            assert "bin" in str(result)


class TestAppModule:
    """Test APP_MODULE constant."""

    def test_app_module_exists(self) -> None:
        """APP_MODULE should point to existing file."""
        from cli.config import APP_MODULE
        assert APP_MODULE.exists()
        assert APP_MODULE.is_file()

    def test_app_module_is_python(self) -> None:
        """APP_MODULE should be a Python file."""
        from cli.config import APP_MODULE
        assert APP_MODULE.suffix == ".py"
        assert APP_MODULE.name == "app.py"
