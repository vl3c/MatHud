"""Browser automation for the MatHud CLI.

Provides headless Chrome automation using Selenium with webdriver-manager.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from cli.config import (
    APP_READY_TIMEOUT,
    BROWSER_WAIT_TIMEOUT,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_VIEWPORT_HEIGHT,
    DEFAULT_VIEWPORT_WIDTH,
)


class BrowserAutomation:
    """Headless Chrome automation for interacting with MatHud."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        headless: bool = True,
        viewport_width: int = DEFAULT_VIEWPORT_WIDTH,
        viewport_height: int = DEFAULT_VIEWPORT_HEIGHT,
    ) -> None:
        """Initialize browser automation.

        Args:
            host: Server host address.
            port: Server port number.
            headless: Whether to run browser in headless mode.
            viewport_width: Browser viewport width.
            viewport_height: Browser viewport height.
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.driver: Optional[webdriver.Chrome] = None
        self._profile_dir: Optional[Path] = None

    def setup(self) -> None:
        """Set up the Chrome WebDriver."""
        if self.driver is not None:
            return
        runtime_root = self._ensure_writable_browser_runtime_dirs()
        # Keep webdriver-manager cache inside the project tree in constrained
        # environments where $HOME is not writable.
        os.environ.setdefault("WDM_LOCAL", "1")

        options = Options()
        if self.headless:
            options.add_argument("--headless=new")

        options.add_argument(f"--window-size={self.viewport_width},{self.viewport_height}")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        # Prevent info bars
        options.add_argument("--disable-infobars")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        if os.name != "nt" and runtime_root is not None:
            # WSL/containerized environments may fail Chrome startup if
            # chromedriver relies on DevTools over HTTP and default profile dirs.
            options.add_argument("--remote-debugging-pipe")
            profiles_root = runtime_root / "profiles"
            profiles_root.mkdir(parents=True, exist_ok=True)
            self._profile_dir = Path(
                tempfile.mkdtemp(prefix="chrome-profile-", dir=str(profiles_root))
            )
            options.add_argument(f"--user-data-dir={self._profile_dir}")

        bundled = self._resolve_bundled_binaries()
        if bundled is not None:
            chrome_bin, driver_bin = bundled
            options.binary_location = str(chrome_bin)
            service = Service(executable_path=str(driver_bin))
        else:
            service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.set_window_size(self.viewport_width, self.viewport_height)
        # Set generous timeouts for long-running operations
        self.driver.set_script_timeout(300)  # 5 minutes for scripts
        self.driver.set_page_load_timeout(60)  # 1 minute for page loads

    def _ensure_writable_browser_runtime_dirs(self) -> Optional[Path]:
        """Ensure browser subprocesses use writable runtime/cache directories.

        In restricted Linux/WSL environments, Chromium can fail during startup
        when HOME or XDG paths are not writable.
        """
        if os.name == "nt":
            return None

        root = Path(__file__).resolve().parent.parent
        runtime_root = root / ".tmp" / "browser-runtime"
        home_dir = runtime_root / "home"
        cache_dir = runtime_root / "cache"
        config_dir = runtime_root / "config"
        run_dir = runtime_root / "run"
        for path in (home_dir, cache_dir, config_dir, run_dir):
            path.mkdir(parents=True, exist_ok=True)

        current_home = os.environ.get("HOME", "")
        if not current_home or not os.access(current_home, os.W_OK):
            os.environ["HOME"] = str(home_dir)

        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
        os.environ.setdefault("XDG_CONFIG_HOME", str(config_dir))
        os.environ.setdefault("XDG_RUNTIME_DIR", str(run_dir))
        return runtime_root

    def _resolve_bundled_binaries(self) -> Optional[tuple[Path, Path]]:
        """Return preferred Chrome/chromedriver paths when available.

        Resolution order:
        1) Explicit env vars
        2) System-installed binaries (Fedora/Ubuntu/etc.)
        3) Bundled project-local binaries
        """
        env_chrome = os.environ.get("MATHUD_CHROME_BIN", "").strip()
        env_driver = os.environ.get("MATHUD_CHROMEDRIVER_BIN", "").strip()
        if env_chrome and env_driver:
            chrome = Path(env_chrome)
            driver = Path(env_driver)
            if chrome.exists() and driver.exists():
                return chrome, driver

        system_driver = shutil.which("chromedriver")
        system_chrome = (
            shutil.which("chromium")
            or shutil.which("chromium-browser")
            or shutil.which("google-chrome")
            or shutil.which("google-chrome-stable")
        )
        if system_driver and system_chrome:
            return Path(system_chrome), Path(system_driver)

        root = Path(__file__).resolve().parent.parent
        bundle_root = root / ".tools" / "chrome-for-testing" / "latest"
        chrome = bundle_root / "chrome-linux64" / "chrome"
        driver = bundle_root / "chromedriver-linux64" / "chromedriver"
        if chrome.exists() and driver.exists():
            return chrome, driver
        return None

    def cleanup(self) -> None:
        """Clean up the WebDriver."""
        if self.driver is not None:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
        if self._profile_dir is not None:
            shutil.rmtree(self._profile_dir, ignore_errors=True)
            self._profile_dir = None

    def __enter__(self) -> "BrowserAutomation":
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.cleanup()

    def navigate_to_app(self) -> bool:
        """Navigate to the MatHud application.

        Returns:
            True if navigation succeeded, False otherwise.
        """
        if self.driver is None:
            raise RuntimeError("Browser not initialized. Call setup() first.")

        try:
            self.driver.get(self.base_url)
            return True
        except Exception:
            return False

    def wait_for_app_ready(self, timeout: int = APP_READY_TIMEOUT) -> bool:
        """Wait for the MatHud application to be fully loaded.

        The app is considered ready when window.startMatHudTests is defined.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if app is ready, False if timeout.
        """
        if self.driver is None:
            raise RuntimeError("Browser not initialized. Call setup() first.")

        try:
            WebDriverWait(self.driver, timeout).until(
                lambda d: d.execute_script("return typeof window.startMatHudTests === 'function'")
            )
            return True
        except Exception:
            return False

    def wait_for_element(
        self,
        selector: str,
        by: str = By.CSS_SELECTOR,
        timeout: int = BROWSER_WAIT_TIMEOUT,
    ) -> bool:
        """Wait for an element to be present on the page.

        Args:
            selector: Element selector.
            by: Selector type (CSS_SELECTOR, ID, etc.).
            timeout: Maximum seconds to wait.

        Returns:
            True if element found, False if timeout.
        """
        if self.driver is None:
            raise RuntimeError("Browser not initialized. Call setup() first.")

        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
            return True
        except Exception:
            return False

    def execute_js(self, script: str, *args: Any, timeout: int = 30) -> Any:
        """Execute JavaScript in the browser context.

        Args:
            script: JavaScript code to execute.
            *args: Arguments to pass to the script.
            timeout: Script timeout in seconds.

        Returns:
            The result of the script execution.
        """
        if self.driver is None:
            raise RuntimeError("Browser not initialized. Call setup() first.")

        # Set script timeout before execution
        self.driver.set_script_timeout(timeout)
        return self.driver.execute_script(script, *args)

    def execute_js_async(self, script: str, timeout: int = 30) -> Any:
        """Execute asynchronous JavaScript.

        Args:
            script: JavaScript code to execute. Must call the callback with result.
            timeout: Maximum seconds to wait.

        Returns:
            The result passed to the callback.
        """
        if self.driver is None:
            raise RuntimeError("Browser not initialized. Call setup() first.")

        self.driver.set_script_timeout(timeout)
        return self.driver.execute_async_script(script)

    def get_canvas_state(self) -> dict[str, Any]:
        """Get the current canvas state as a dictionary.

        Returns:
            Canvas state dictionary.
        """
        result = self.execute_js(
            "return window._canvas ? JSON.stringify(window._canvas.get_state()) : null"
        )
        if result:
            parsed: dict[str, Any] = json.loads(result)
            return parsed
        return {}

    def call_canvas_method(self, method: str, *args: Any) -> Any:
        """Call a method on the canvas object.

        Args:
            method: Method name to call.
            *args: Arguments to pass to the method.

        Returns:
            The result of the method call.
        """
        args_json = json.dumps(args)
        script = f"""
            if (window._canvas && typeof window._canvas.{method} === 'function') {{
                const args = {args_json};
                return window._canvas.{method}(...args);
            }}
            return null;
        """
        return self.execute_js(script)

    def call_function_registry(self, function_name: str, args: dict) -> Any:
        """Call a function via the FunctionRegistry.

        Args:
            function_name: Name of the function to call.
            args: Dictionary of arguments.

        Returns:
            The result of the function call.
        """
        args_json = json.dumps(args)
        script = f"""
            if (window._canvas && window._canvas.function_registry) {{
                const func = window._canvas.function_registry.get('{function_name}');
                if (func) {{
                    const args = {args_json};
                    return func(args);
                }}
            }}
            return null;
        """
        return self.execute_js(script)

    def capture_screenshot(self, output_path: str, full_page: bool = True) -> bool:
        """Capture a screenshot of the browser.

        Args:
            output_path: Path to save the screenshot.
            full_page: If True, capture full page. If False, capture just the canvas.

        Returns:
            True if screenshot captured successfully.
        """
        if self.driver is None:
            raise RuntimeError("Browser not initialized. Call setup() first.")

        try:
            if full_page:
                self.driver.save_screenshot(output_path)
            else:
                # Capture just the math container
                element = self.driver.find_element(By.CSS_SELECTOR, ".math-container")
                element.screenshot(output_path)
            return True
        except Exception:
            return False

    def start_tests(self) -> dict[str, Any]:
        """Start the MatHud test suite.

        Returns:
            Initial status response.
        """
        result = self.execute_js("return window.startMatHudTests()")
        if result:
            parsed: dict[str, Any] = json.loads(result)
            return parsed
        return {"status": "error", "error": "startMatHudTests not available"}

    def get_test_results(self) -> dict[str, Any]:
        """Get test results.

        Returns:
            Test results dictionary.
        """
        result = self.execute_js("return window.getMatHudTestResults()")
        if result:
            parsed: dict[str, Any] = json.loads(result)
            return parsed
        return {"status": "error", "error": "getMatHudTestResults not available"}

    def poll_test_results(self, timeout: int, poll_interval: int = 2) -> dict[str, Any]:
        """Poll for test results until complete or timeout.

        Args:
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between polls.

        Returns:
            Final test results.
        """
        start_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 5

        while time.time() - start_time < timeout:
            try:
                results = self.get_test_results()
                consecutive_errors = 0  # Reset on success
                status = results.get("status")

                if status not in ("running", "started"):
                    return results

            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    return {
                        "status": "error",
                        "error": f"Too many consecutive errors polling results: {e}",
                    }

            time.sleep(poll_interval)

        return {
            "status": "timeout",
            "error": f"Tests did not complete within {timeout} seconds",
        }
