"""Tests for cli/browser.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cli.browser import BrowserAutomation
from cli.config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_VIEWPORT_WIDTH,
    DEFAULT_VIEWPORT_HEIGHT,
)


class TestBrowserAutomationInit:
    """Test BrowserAutomation initialization."""

    def test_default_values(self) -> None:
        """BrowserAutomation should use default values."""
        browser = BrowserAutomation()
        assert browser.host == DEFAULT_HOST
        assert browser.port == DEFAULT_PORT
        assert browser.headless is True
        assert browser.viewport_width == DEFAULT_VIEWPORT_WIDTH
        assert browser.viewport_height == DEFAULT_VIEWPORT_HEIGHT
        assert browser.driver is None

    def test_custom_values(self) -> None:
        """BrowserAutomation should accept custom values."""
        browser = BrowserAutomation(
            host="0.0.0.0",
            port=8080,
            headless=False,
            viewport_width=1280,
            viewport_height=720,
        )
        assert browser.host == "0.0.0.0"
        assert browser.port == 8080
        assert browser.headless is False
        assert browser.viewport_width == 1280
        assert browser.viewport_height == 720

    def test_base_url_format(self) -> None:
        """base_url should be properly formatted."""
        browser = BrowserAutomation(host="127.0.0.1", port=5000)
        assert browser.base_url == "http://127.0.0.1:5000"


class TestBrowserAutomationSetup:
    """Test BrowserAutomation.setup method."""

    @patch("cli.browser.ChromeDriverManager")
    @patch("cli.browser.Service")
    @patch("cli.browser.webdriver.Chrome")
    def test_setup_creates_driver(
        self,
        mock_chrome: MagicMock,
        mock_service: MagicMock,
        mock_driver_manager: MagicMock,
    ) -> None:
        """setup should create a Chrome WebDriver."""
        mock_driver_manager.return_value.install.return_value = "/path/to/chromedriver"
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver

        browser = BrowserAutomation()
        browser.setup()

        assert browser.driver is mock_driver
        mock_driver.set_window_size.assert_called_once()

    @patch("cli.browser.ChromeDriverManager")
    @patch("cli.browser.Service")
    @patch("cli.browser.webdriver.Chrome")
    def test_setup_idempotent(
        self,
        mock_chrome: MagicMock,
        mock_service: MagicMock,
        mock_driver_manager: MagicMock,
    ) -> None:
        """setup should not recreate driver if already exists."""
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver

        browser = BrowserAutomation()
        browser.driver = mock_driver  # Pre-set driver

        browser.setup()

        mock_chrome.assert_not_called()


class TestBrowserAutomationCleanup:
    """Test BrowserAutomation.cleanup method."""

    def test_cleanup_quits_driver(self) -> None:
        """cleanup should quit the driver."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        browser.driver = mock_driver

        browser.cleanup()

        mock_driver.quit.assert_called_once()
        assert browser.driver is None

    def test_cleanup_handles_exception(self) -> None:
        """cleanup should handle exceptions gracefully."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        mock_driver.quit.side_effect = Exception("Error")
        browser.driver = mock_driver

        browser.cleanup()  # Should not raise

        assert browser.driver is None

    def test_cleanup_with_no_driver(self) -> None:
        """cleanup should handle None driver gracefully."""
        browser = BrowserAutomation()
        browser.driver = None

        browser.cleanup()  # Should not raise

        assert browser.driver is None


class TestBrowserAutomationContextManager:
    """Test BrowserAutomation context manager."""

    @patch("cli.browser.ChromeDriverManager")
    @patch("cli.browser.Service")
    @patch("cli.browser.webdriver.Chrome")
    def test_context_manager_setup_and_cleanup(
        self,
        mock_chrome: MagicMock,
        mock_service: MagicMock,
        mock_driver_manager: MagicMock,
    ) -> None:
        """Context manager should setup and cleanup."""
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver

        with BrowserAutomation() as browser:
            assert browser.driver is mock_driver

        mock_driver.quit.assert_called_once()


class TestBrowserAutomationNavigate:
    """Test BrowserAutomation.navigate_to_app method."""

    def test_navigate_success(self) -> None:
        """navigate_to_app returns True on success."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        browser.driver = mock_driver

        result = browser.navigate_to_app()

        assert result is True
        mock_driver.get.assert_called_once_with(browser.base_url)

    def test_navigate_failure(self) -> None:
        """navigate_to_app returns False on failure."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        mock_driver.get.side_effect = Exception("Navigation failed")
        browser.driver = mock_driver

        result = browser.navigate_to_app()

        assert result is False

    def test_navigate_without_driver(self) -> None:
        """navigate_to_app raises error without driver."""
        browser = BrowserAutomation()

        with pytest.raises(RuntimeError, match="not initialized"):
            browser.navigate_to_app()


class TestBrowserAutomationExecuteJs:
    """Test BrowserAutomation.execute_js method."""

    def test_execute_js_returns_result(self) -> None:
        """execute_js returns script result."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        mock_driver.execute_script.return_value = "result"
        browser.driver = mock_driver

        result = browser.execute_js("return 'result'")

        assert result == "result"
        mock_driver.execute_script.assert_called_once()

    def test_execute_js_without_driver(self) -> None:
        """execute_js raises error without driver."""
        browser = BrowserAutomation()

        with pytest.raises(RuntimeError, match="not initialized"):
            browser.execute_js("return 1")

    def test_execute_js_sets_timeout(self) -> None:
        """execute_js sets script timeout."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        browser.driver = mock_driver

        browser.execute_js("return 1", timeout=60)

        mock_driver.set_script_timeout.assert_called_with(60)


class TestBrowserAutomationGetCanvasState:
    """Test BrowserAutomation.get_canvas_state method."""

    def test_get_canvas_state_returns_dict(self) -> None:
        """get_canvas_state returns the state from getMatHudCanvasState."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        mock_driver.execute_script.return_value = '{"state": {"Points": [], "Segments": []}}'
        browser.driver = mock_driver

        result = browser.get_canvas_state()

        assert result == {"Points": [], "Segments": []}
        script = mock_driver.execute_script.call_args[0][0]
        assert "getMatHudCanvasState" in script
        assert "_canvas" not in script

    def test_get_canvas_state_empty(self) -> None:
        """get_canvas_state returns empty dict when the hook is missing."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        mock_driver.execute_script.return_value = None
        browser.driver = mock_driver

        result = browser.get_canvas_state()

        assert result == {}


class TestBrowserAutomationHooks:
    """Test the scenario-hook wrappers (runMatHudToolCalls and friends)."""

    def _browser(self, reply: str) -> tuple[BrowserAutomation, MagicMock]:
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        mock_driver.execute_script.return_value = reply
        browser.driver = mock_driver
        return browser, mock_driver

    def test_call_hook_passes_arguments(self) -> None:
        browser, driver = self._browser('{"status": "ok"}')

        assert browser.call_hook("resetMatHudSession", '{"x": 1}') == {"status": "ok"}
        args = driver.execute_script.call_args[0]
        assert "window.resetMatHudSession.apply(null, arguments)" in args[0]
        assert args[1] == '{"x": 1}'

    def test_call_hook_missing_raises(self) -> None:
        browser, _ = self._browser("")
        with pytest.raises(RuntimeError, match="not available"):
            browser.call_hook("runMatHudToolCalls", "[]")

    def test_call_hook_non_object_raises(self) -> None:
        browser, _ = self._browser("[1, 2]")
        with pytest.raises(RuntimeError, match="expected an object"):
            browser.call_hook("getMatHudTurnStatus")

    def test_run_tool_calls_sends_json(self) -> None:
        browser, driver = self._browser('{"status": "ok", "traced": []}')
        calls = [{"function_name": "create_point", "arguments": {"x": 1, "y": 2}}]

        browser.run_tool_calls(calls)

        assert (
            driver.execute_script.call_args[0][1]
            == '[{"function_name": "create_point", "arguments": {"x": 1, "y": 2}}]'
        )

    def test_call_function_registry_returns_result(self) -> None:
        browser, _ = self._browser('{"status": "ok", "traced": [{"result": "Call successful!"}]}')
        assert browser.call_function_registry("create_point", {"x": 1, "y": 2}) == "Call successful!"

    def test_call_function_registry_error_raises(self) -> None:
        browser, _ = self._browser('{"status": "error", "error": "Invalid tool calls: boom"}')
        with pytest.raises(RuntimeError, match="boom"):
            browser.call_function_registry("create_point", {})

    def test_get_canvas_snapshot_sends_options(self) -> None:
        browser, driver = self._browser('{"state": {}, "inspection": {"undo_depth": 0}}')

        payload = browser.get_canvas_snapshot({"inspect": True})

        assert payload["inspection"] == {"undo_depth": 0}
        assert driver.execute_script.call_args[0][1] == '{"inspect": true}'


class TestBrowserAutomationTestMethods:
    """Test BrowserAutomation test-related methods."""

    def test_start_tests_returns_dict(self) -> None:
        """start_tests returns parsed JSON."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        mock_driver.execute_script.return_value = '{"status": "started"}'
        browser.driver = mock_driver

        result = browser.start_tests()

        assert result == {"status": "started"}

    def test_start_tests_no_function(self) -> None:
        """start_tests returns error when function not available."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        mock_driver.execute_script.return_value = None
        browser.driver = mock_driver

        result = browser.start_tests()

        assert result["status"] == "error"

    def test_get_test_results_returns_dict(self) -> None:
        """get_test_results returns parsed JSON."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        mock_driver.execute_script.return_value = '{"tests_run": 100, "failures": 0}'
        browser.driver = mock_driver

        result = browser.get_test_results()

        assert result == {"tests_run": 100, "failures": 0}

    def test_poll_test_results_returns_on_completion(self) -> None:
        """poll_test_results returns when tests complete."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        # First call returns running, second returns complete
        mock_driver.execute_script.side_effect = [
            '{"status": "running"}',
            '{"tests_run": 100, "failures": 0}',
        ]
        browser.driver = mock_driver

        with patch("cli.browser.time.sleep"):
            result = browser.poll_test_results(timeout=10, poll_interval=1)

        assert result == {"tests_run": 100, "failures": 0}

    def test_poll_test_results_timeout(self) -> None:
        """poll_test_results returns timeout status."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        mock_driver.execute_script.return_value = '{"status": "running"}'
        browser.driver = mock_driver

        with patch("cli.browser.time.sleep"):
            with patch("cli.browser.time.time") as mock_time:
                # Simulate timeout
                mock_time.side_effect = [0, 0, 5, 10, 15]
                result = browser.poll_test_results(timeout=10, poll_interval=1)

        assert result["status"] == "timeout"


class TestBrowserAutomationScreenshot:
    """Test BrowserAutomation.capture_screenshot method."""

    def test_capture_full_page(self) -> None:
        """capture_screenshot captures full page."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        browser.driver = mock_driver

        result = browser.capture_screenshot("/path/to/screenshot.png", full_page=True)

        assert result is True
        mock_driver.save_screenshot.assert_called_once_with("/path/to/screenshot.png")

    def test_capture_canvas_only(self) -> None:
        """capture_screenshot captures element only."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        mock_element = MagicMock()
        mock_driver.find_element.return_value = mock_element
        browser.driver = mock_driver

        result = browser.capture_screenshot("/path/to/screenshot.png", full_page=False)

        assert result is True
        mock_element.screenshot.assert_called_once_with("/path/to/screenshot.png")

    def test_capture_failure(self) -> None:
        """capture_screenshot returns False on failure."""
        browser = BrowserAutomation()
        mock_driver = MagicMock()
        mock_driver.save_screenshot.side_effect = Exception("Error")
        browser.driver = mock_driver

        result = browser.capture_screenshot("/path/to/screenshot.png")

        assert result is False

    def test_capture_without_driver(self) -> None:
        """capture_screenshot raises error without driver."""
        browser = BrowserAutomation()

        with pytest.raises(RuntimeError, match="not initialized"):
            browser.capture_screenshot("/path/to/screenshot.png")
