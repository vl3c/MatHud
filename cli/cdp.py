"""A small Chrome DevTools Protocol client for driving the MatHud desktop window.

``python mathud_desktop.py --automation-port N`` starts the window's browser
(Edge WebView2 on Windows) with a DevTools endpoint on 127.0.0.1:N. This module
finds the app's page there (``GET /json``), connects to its websocket and runs
JavaScript (``Runtime.evaluate``) and screenshots (``Page.captureScreenshot``)
in the window the user is watching.

``CDPBrowser`` offers the same methods as ``cli.browser.BrowserAutomation``
(the headless Selenium backend), so the scenario runner and the desktop
commands drive either one through ``cli.browser_backend.AppBrowser``.
Closing a ``CDPBrowser`` only closes its connection, never the window.

Only loopback addresses are accepted: the DevTools endpoint gives full control
of the page, so the client never talks to another host.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import uuid
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

from cli.browser_backend import HookClient

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]", "::1")
DEFAULT_CDP_TIMEOUT_S = 30.0
APP_READY_TIMEOUT_S = 60.0
_READY_POLL_S = 0.25
# Set on the page (to a fresh nonce) before a reload; the reload is done once the page no
# longer carries that nonce, so a marker left by another client's reload never blocks this one.
_RELOAD_MARKER = "__mathudCdpReloadPending"
# Readiness checks talk to localhost directly; system or environment proxies must not be consulted.
_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class CDPError(Exception):
    """The DevTools endpoint could not be reached, or a command failed."""


class CDPUnavailable(CDPError):
    """Nothing listens on the DevTools port: the window was closed (or never opened with automation)."""


def is_loopback_url(url: str) -> bool:
    """True when ``url`` points at this machine's loopback interface."""
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return False
    return host in LOOPBACK_HOSTS


def list_targets(port: int, host: str = "127.0.0.1", timeout: float = 5.0) -> list[dict[str, Any]]:
    """The DevTools targets (``GET http://host:port/json``)."""
    if host not in LOOPBACK_HOSTS:
        raise CDPError(f"refusing a DevTools endpoint on {host}: only loopback addresses are allowed")
    url = f"http://{host}:{port}/json"
    try:
        with _LOCAL_OPENER.open(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        refused = isinstance(getattr(exc, "reason", exc), ConnectionRefusedError)
        raise (CDPUnavailable if refused else CDPError)(
            f"no DevTools endpoint at {url} ({exc}); start the desktop app with "
            f"`python mathud_desktop.py --automation-port {port}`"
        ) from exc
    if not isinstance(payload, list):
        raise CDPError(f"unexpected reply from {url}: {str(payload)[:120]}")
    return [target for target in payload if isinstance(target, dict)]


def find_app_target(targets: list[dict[str, Any]]) -> dict[str, Any]:
    """The MatHud page among ``targets``: a ``page`` served over http from a loopback address.

    Raises CDPError when there is none, or when several pages qualify (the
    caller would not know which window it drives).
    """
    pages = [
        target
        for target in targets
        if target.get("type") == "page"
        and str(target.get("url", "")).startswith("http")
        and is_loopback_url(str(target.get("url", "")))
        and target.get("webSocketDebuggerUrl")
    ]
    if not pages:
        seen = ", ".join(f"{t.get('type')}: {t.get('url')}" for t in targets) or "none"
        raise CDPError(f"no MatHud page among the DevTools targets ({seen})")
    if len(pages) > 1:
        raise CDPError("several pages are open: " + ", ".join(str(p.get("url")) for p in pages))
    return pages[0]


def app_base_url(page_url: str) -> str:
    """``http://host:port`` of the page's origin (the app server)."""
    parts = urlsplit(page_url)
    return f"{parts.scheme}://{parts.netloc}"


def _default_connect(url: str, timeout: float) -> Any:
    import websocket  # websocket-client, a dependency of selenium

    # No Origin header: Chromium refuses DevTools websockets from origins it was not told to allow.
    return websocket.create_connection(url, timeout=timeout, suppress_origin=True, enable_multithread=True)


class CDPConnection:
    """One websocket to a page target: numbered commands and their replies; events are ignored."""

    def __init__(
        self,
        ws_url: str,
        timeout: float = DEFAULT_CDP_TIMEOUT_S,
        connect: Callable[[str, float], Any] = _default_connect,
    ) -> None:
        if not is_loopback_url(ws_url):
            raise CDPError(f"refusing a DevTools websocket outside loopback: {ws_url}")
        self.ws_url = ws_url
        self.timeout = timeout
        try:
            self._ws = connect(ws_url, timeout)
        except Exception as exc:
            raise CDPError(f"could not connect to {ws_url}: {exc}") from exc
        self._next_id = 0
        self._lock = threading.Lock()
        self.closed = False

    def send(self, method: str, params: Optional[dict[str, Any]] = None, timeout: Optional[float] = None) -> Any:
        """Send one command and return its ``result``; raises CDPError on an error reply or a timeout."""
        if self.closed:
            raise CDPError("the DevTools connection is closed")
        wait_s = self.timeout if timeout is None else timeout
        with self._lock:
            self._next_id += 1
            message_id = self._next_id
            message: dict[str, Any] = {"id": message_id, "method": method, "params": params or {}}
            deadline = time.monotonic() + wait_s
            try:
                self._ws.send(json.dumps(message))
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise CDPError(f"{method} did not answer within {wait_s:.0f} s")
                    self._ws.settimeout(remaining)
                    raw = self._ws.recv()
                    reply = json.loads(raw) if raw else {}
                    if not isinstance(reply, dict) or reply.get("id") != message_id:
                        continue  # an event, or the reply to a command that timed out earlier
                    if "error" in reply:
                        error = reply["error"]
                        detail = error.get("message") if isinstance(error, dict) else error
                        raise CDPError(f"{method} failed: {detail}")
                    return reply.get("result") or {}
            except CDPError:
                raise
            except Exception as exc:  # socket timeout, closed socket, bad JSON
                raise CDPError(f"{method}: {type(exc).__name__}: {exc}") from exc

    def evaluate(self, expression: str, timeout: Optional[float] = None) -> Any:
        """Evaluate ``expression`` in the page, await a promise result, and return the value by value."""
        result = self.send(
            "Runtime.evaluate",
            {"expression": expression, "awaitPromise": True, "returnByValue": True, "userGesture": True},
            timeout=timeout,
        )
        details = result.get("exceptionDetails")
        if details:
            exception = details.get("exception") or {}
            text = exception.get("description") or details.get("text") or "exception"
            raise CDPError(f"JavaScript error: {text}")
        remote = result.get("result") or {}
        return remote.get("value")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self._ws.close()
        except Exception:
            pass


def wrap_script(script: str, args: tuple[Any, ...]) -> str:
    """A Selenium-style script body (``return ...``, ``arguments``) as one evaluable expression."""
    return f"(function () {{\n{script}\n}}).apply(null, {json.dumps(list(args))})"


class CDPBrowser(HookClient):
    """The desktop window's page, driven over CDP; the attach-mode counterpart of BrowserAutomation.

    ``attached`` tells the scenario runner not to navigate on open: the window
    already shows the app, and only a recovery reloads it.
    """

    attached = True

    def __init__(
        self,
        debug_port: int,
        host: str = "127.0.0.1",
        timeout: float = DEFAULT_CDP_TIMEOUT_S,
        connect: Callable[[str, float], Any] = _default_connect,
        targets: Callable[[int, str], list[dict[str, Any]]] = lambda port, host: list_targets(port, host),
    ) -> None:
        self.debug_port = debug_port
        self.host = host
        self.timeout = timeout
        self._connect = connect
        self._targets = targets
        self.connection: Optional[CDPConnection] = None
        self.page_url: Optional[str] = None
        self.base_url: Optional[str] = None
        self._reload_nonce: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------------

    def setup(self) -> None:
        """Find the app's page and connect to it."""
        if self.connection is not None and not self.connection.closed:
            return
        target = find_app_target(self._targets(self.debug_port, self.host))
        self.page_url = str(target["url"])
        self.base_url = app_base_url(self.page_url)
        self.connection = CDPConnection(str(target["webSocketDebuggerUrl"]), self.timeout, self._connect)

    def close(self) -> None:
        """Close the connection; the window stays open."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def cleanup(self) -> None:
        self.close()

    def __enter__(self) -> "CDPBrowser":
        self.setup()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _conn(self) -> CDPConnection:
        if self.connection is None:
            raise RuntimeError("Not connected. Call setup() first.")
        return self.connection

    # -- navigation --------------------------------------------------------------

    def navigate_to_app(self) -> bool:
        """Reload the app page (attach mode never navigates elsewhere)."""
        return self.reload()

    def reload(self) -> bool:
        """Reload the page; ``wait_for_app_ready`` then waits for the new document."""
        conn = self._conn()
        nonce = uuid.uuid4().hex
        try:
            conn.evaluate(f"window.{_RELOAD_MARKER} = {json.dumps(nonce)}")
        except CDPError:
            pass  # a hung or broken page: reload anyway
        conn.send("Page.reload", {"ignoreCache": False})
        self._reload_nonce = nonce
        return True

    def wait_for_app_ready(self, timeout: float = APP_READY_TIMEOUT_S) -> bool:
        """Wait until the page (the new one, after ``reload``) defines the app's hooks."""
        check = "typeof window.sendMatHudMessage === 'function'"
        if self._reload_nonce is not None:
            check = f"window.{_RELOAD_MARKER} !== {json.dumps(self._reload_nonce)} && {check}"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self._conn().evaluate(check, timeout=5) is True:
                    self._reload_nonce = None
                    return True
            except CDPError:
                pass  # the page is navigating
            time.sleep(_READY_POLL_S)
        return False

    def call_hook(self, name: str, *args: Any, timeout: int = 30) -> dict[str, Any]:
        """``HookClient.call_hook``; when the hook is missing (the page was reloaded from
        outside and is still loading), wait for the app once and retry."""
        try:
            return super().call_hook(name, *args, timeout=timeout)
        except RuntimeError as exc:
            if "is not available" not in str(exc) or not self.wait_for_app_ready(APP_READY_TIMEOUT_S):
                raise
        return super().call_hook(name, *args, timeout=timeout)

    # -- scripts and screenshots ------------------------------------------------

    def execute_js(self, script: str, *args: Any, timeout: float = 30) -> Any:
        """Run a Selenium-style script body (``return`` its value; ``arguments`` holds ``args``)."""
        return self._conn().evaluate(wrap_script(script, args), timeout=timeout)

    def screenshot_png(self, timeout: float = 30) -> bytes:
        result = self._conn().send("Page.captureScreenshot", {"format": "png"}, timeout=timeout)
        data = result.get("data")
        if not isinstance(data, str):
            raise CDPError("Page.captureScreenshot returned no image")
        return base64.b64decode(data)

    def screenshot(self, output_path: str) -> bool:
        """Save a PNG of the window's page; False on failure."""
        try:
            png = self.screenshot_png()
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(png)
            return True
        except Exception:
            return False

    def capture_screenshot(self, output_path: str, full_page: bool = True) -> bool:
        return self.screenshot(output_path)
