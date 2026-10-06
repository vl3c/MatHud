"""Tests for the Chrome DevTools Protocol client (cli/cdp.py), with a fake websocket and a local /json server.

No browser and no desktop window: the websocket is a scripted stand-in that
answers CDP commands, and target discovery talks to a tiny HTTP server on
127.0.0.1.
"""

from __future__ import annotations

import base64
import re
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import pytest

from cli.cdp import (
    CDPBrowser,
    CDPConnection,
    CDPError,
    CDPUnavailable,
    _default_connect,
    app_base_url,
    find_app_target,
    is_loopback_url,
    list_targets,
    wrap_script,
)

PAGE = {
    "id": "P1",
    "type": "page",
    "url": "http://127.0.0.1:5110/",
    "webSocketDebuggerUrl": "ws://127.0.0.1:9301/devtools/page/P1",
}


class FakeSocket:
    """A websocket that answers each command through ``responder`` (and can push events first)."""

    def __init__(self, responder: Callable[[dict[str, Any]], Any], events: Optional[list[dict[str, Any]]] = None):
        self.responder = responder
        self.events = list(events or [])
        self.sent: list[dict[str, Any]] = []
        self.inbox: list[str] = []
        self.closed = False
        self.timeouts: list[float] = []

    def send(self, raw: str) -> None:
        message = json.loads(raw)
        self.sent.append(message)
        for event in self.events:
            self.inbox.append(json.dumps(event))
        self.events = []
        reply = self.responder(message)
        if reply is None:
            return  # never answers
        if isinstance(reply, list):  # several frames, e.g. a stale reply first
            self.inbox.extend(json.dumps(item) for item in reply)
        else:
            self.inbox.append(json.dumps(reply))

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)

    def recv(self) -> str:
        if not self.inbox:
            raise TimeoutError("timed out")
        return self.inbox.pop(0)

    def close(self) -> None:
        self.closed = True


def _ok(result: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda message: {"id": message["id"], "result": result}


def _value(value: Any) -> dict[str, Any]:
    return {"result": {"type": "string", "value": value}}


def _connection(responder: Callable[[dict[str, Any]], Any], **kwargs: Any) -> tuple[CDPConnection, FakeSocket]:
    sock = FakeSocket(responder, **kwargs)
    return CDPConnection(PAGE["webSocketDebuggerUrl"], timeout=2, connect=lambda url, timeout: sock), sock


class TestLoopbackAndTargets:
    def test_is_loopback_url(self) -> None:
        assert is_loopback_url("ws://127.0.0.1:9301/devtools/page/x")
        assert is_loopback_url("http://localhost:5000/")
        assert not is_loopback_url("ws://192.168.1.5:9301/devtools/page/x")
        assert not is_loopback_url("http://example.com/")

    def test_find_app_target_picks_the_loopback_http_page(self) -> None:
        targets = [
            {
                "type": "service_worker",
                "url": "http://127.0.0.1:5110/sw.js",
                "webSocketDebuggerUrl": "ws://127.0.0.1:1/a",
            },
            {"type": "page", "url": "devtools://devtools/inspector.html", "webSocketDebuggerUrl": "ws://127.0.0.1:1/b"},
            {"type": "page", "url": "https://example.com/", "webSocketDebuggerUrl": "ws://127.0.0.1:1/c"},
            PAGE,
        ]
        assert find_app_target(targets) is PAGE

    def test_find_app_target_refuses_none_or_several(self) -> None:
        with pytest.raises(CDPError, match="no MatHud page"):
            find_app_target([{"type": "page", "url": "about:blank"}])
        with pytest.raises(CDPError, match="several pages"):
            find_app_target([PAGE, dict(PAGE, id="P2", url="http://127.0.0.1:5110/?x")])

    def test_app_base_url(self) -> None:
        assert app_base_url("http://127.0.0.1:5110/some/path?q=1") == "http://127.0.0.1:5110"

    def test_list_targets_reads_json_from_a_local_server(self, json_server: int) -> None:
        assert list_targets(json_server) == [PAGE]

    def test_list_targets_explains_a_missing_endpoint(self) -> None:
        port = _unused_port()
        with pytest.raises(CDPError, match=f"--automation-port {port}"):
            list_targets(port, timeout=1)

    def test_list_targets_refuses_other_hosts(self) -> None:
        with pytest.raises(CDPError, match="only loopback"):
            list_targets(9301, host="10.0.0.2")


def _unused_port() -> int:
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


@pytest.fixture
def json_server() -> Iterator[int]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            body = json.dumps([PAGE]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()


class TestConnection:
    def test_refuses_a_websocket_outside_loopback(self) -> None:
        with pytest.raises(CDPError, match="outside loopback"):
            CDPConnection("ws://10.1.1.1:9301/devtools/page/x", connect=lambda url, timeout: None)

    def test_send_skips_events_and_stale_replies(self) -> None:
        def responder(message: dict[str, Any]) -> list[dict[str, Any]]:
            return [{"id": message["id"] - 1, "result": {"stale": True}}, {"id": message["id"], "result": {"ok": 1}}]

        conn, sock = _connection(responder, events=[{"method": "Page.loadEventFired", "params": {}}])
        assert conn.send("Page.enable") == {"ok": 1}
        assert sock.sent[0]["method"] == "Page.enable"

    def test_error_reply_raises(self) -> None:
        conn, _ = _connection(lambda m: {"id": m["id"], "error": {"code": -32000, "message": "No target"}})
        with pytest.raises(CDPError, match="No target"):
            conn.send("Page.reload")

    def test_no_reply_times_out(self) -> None:
        conn, _ = _connection(lambda m: None)
        with pytest.raises(CDPError, match="Runtime.evaluate"):
            conn.send("Runtime.evaluate", {"expression": "1"}, timeout=0.2)

    def test_evaluate_awaits_promises_and_returns_by_value(self) -> None:
        conn, sock = _connection(_ok(_value('{"status": "ok"}')))
        assert conn.evaluate("window.x()") == '{"status": "ok"}'
        params = sock.sent[0]["params"]
        assert sock.sent[0]["method"] == "Runtime.evaluate"
        assert params["awaitPromise"] is True and params["returnByValue"] is True
        assert params["expression"] == "window.x()"

    def test_evaluate_raises_on_a_javascript_exception(self) -> None:
        conn, _ = _connection(
            _ok({"result": {}, "exceptionDetails": {"text": "Uncaught", "exception": {"description": "TypeError: x"}}})
        )
        with pytest.raises(CDPError, match="TypeError: x"):
            conn.evaluate("x()")

    def test_close_is_idempotent_and_blocks_further_commands(self) -> None:
        conn, sock = _connection(_ok({}))
        conn.close()
        conn.close()
        assert sock.closed
        with pytest.raises(CDPError, match="closed"):
            conn.send("Page.reload")


def test_wrap_script_passes_arguments_as_json() -> None:
    wrapped = wrap_script("return arguments[0] + arguments[1];", ("a", '{"b": 1}'))
    assert wrapped.startswith("(function () {")
    assert wrapped.endswith(').apply(null, ["a", "{\\"b\\": 1}"])')


class FakePage:
    """Answers the CDP commands CDPBrowser sends, with a tiny window state."""

    def __init__(self) -> None:
        self.reloads = 0
        self.marker: Optional[str] = None
        self.ready = True
        self.models_loaded = True  # window.matHudModelsLoaded
        self.png = b"\x89PNG fake"
        # Hook calls answered "missing" (as while a page reloaded from outside is loading).
        self.missing_hook_calls = 0

    def __call__(self, message: dict[str, Any]) -> dict[str, Any]:
        method, params = message["method"], message.get("params") or {}
        result: dict[str, Any] = {}
        if method == "Runtime.evaluate":
            expression = params["expression"]
            if expression.startswith("window.__mathudCdpReloadPending = "):
                self.marker = json.loads(expression.split("= ", 1)[1])
                result = _value(self.marker)
            elif "matHudModelsLoaded" in expression:
                result = _value(self.models_loaded)
            elif "typeof window.sendMatHudMessage" in expression:
                waiting = re.search(r'!== "([0-9a-f]+)"', expression)
                result = _value(self.ready and not (waiting and self.marker == waiting.group(1)))
            elif "getMatHudTurnStatus" in expression:
                if self.missing_hook_calls:
                    self.missing_hook_calls -= 1
                    result = _value(None)  # call_hook: typeof window.getMatHudTurnStatus !== 'function'
                else:
                    result = _value(json.dumps({"processing": False, "completed_turns": 3}))
            else:
                result = _value(None)
        elif method == "Page.reload":
            self.reloads += 1
            self.marker = None  # the new document has no marker
        elif method == "Page.captureScreenshot":
            result = {"data": base64.b64encode(self.png).decode()}
        return {"id": message["id"], "result": result}


def _browser(page: FakePage) -> tuple[CDPBrowser, list[FakeSocket]]:
    sockets: list[FakeSocket] = []

    def connect(url: str, timeout: float) -> FakeSocket:
        sockets.append(FakeSocket(page))
        return sockets[-1]

    browser = CDPBrowser(9301, connect=connect, targets=lambda port, host: [PAGE])
    return browser, sockets


class TestCDPBrowser:
    def test_setup_finds_the_page_and_the_app_url(self) -> None:
        browser, sockets = _browser(FakePage())
        browser.setup()
        assert browser.attached is True
        assert browser.page_url == "http://127.0.0.1:5110/"
        assert browser.base_url == "http://127.0.0.1:5110"
        assert len(sockets) == 1
        browser.setup()  # already connected
        assert len(sockets) == 1

    def test_call_hook_runs_through_execute_js(self) -> None:
        browser, sockets = _browser(FakePage())
        browser.setup()
        assert browser.call_hook("getMatHudTurnStatus") == {"processing": False, "completed_turns": 3}
        expression = sockets[0].sent[-1]["params"]["expression"]
        assert "window.getMatHudTurnStatus.apply(null, arguments)" in expression

    def test_reload_marks_the_old_page_and_waits_for_the_new_one(self) -> None:
        page = FakePage()
        browser, _ = _browser(page)
        browser.setup()
        assert browser.reload() is True
        assert page.reloads == 1
        assert browser.wait_for_app_ready(2) is True

    def test_a_reload_that_did_not_happen_is_not_mistaken_for_ready(self) -> None:
        page = FakePage()
        browser, sockets = _browser(page)
        browser.setup()
        page_reload = page.__call__

        def no_reload(message: dict[str, Any]) -> dict[str, Any]:
            if message["method"] == "Page.reload":
                return {"id": message["id"], "result": {}}  # the old page stays, marker and all
            return page_reload(message)

        sockets[0].responder = no_reload
        browser.reload()
        assert browser.wait_for_app_ready(0.3) is False

    def test_a_stale_marker_from_another_client_does_not_block(self) -> None:
        page = FakePage()
        page.marker = "0123abcd"  # left by a client that died before its reload
        browser, _ = _browser(page)
        browser.setup()
        assert browser.wait_for_app_ready(1) is True

    def test_a_missing_hook_waits_for_the_app_and_retries_once(self) -> None:
        page = FakePage()
        browser, _ = _browser(page)
        browser.setup()
        page.missing_hook_calls = 1
        assert browser.call_hook("getMatHudTurnStatus")["completed_turns"] == 3
        page.missing_hook_calls = 2
        with pytest.raises(RuntimeError, match="not available"):
            browser.call_hook("getMatHudTurnStatus")

    def test_refused_connection_is_cdp_unavailable(self) -> None:
        with pytest.raises(CDPUnavailable):
            list_targets(_unused_port(), timeout=5)  # Windows refuses only after its SYN retries (~2 s)

    def test_default_connect_suppresses_the_origin_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import websocket

        seen: dict[str, Any] = {}

        def create_connection(url: str, **kwargs: Any) -> str:
            seen.update(kwargs, url=url)
            return "ws"

        monkeypatch.setattr(websocket, "create_connection", create_connection)
        assert _default_connect("ws://127.0.0.1:9301/devtools/page/P1", 5) == "ws"
        # Chromium refuses DevTools websockets that send an Origin header (403).
        assert seen["suppress_origin"] is True
        assert "origin" not in seen

    def test_wait_for_app_ready_gives_up(self) -> None:
        page = FakePage()
        page.ready = False
        browser, _ = _browser(page)
        browser.setup()
        assert browser.wait_for_app_ready(0.3) is False

    def test_screenshot_writes_the_png(self, tmp_path: Path) -> None:
        page = FakePage()
        browser, _ = _browser(page)
        browser.setup()
        target = tmp_path / "sub" / "shot.png"
        assert browser.screenshot(str(target)) is True
        assert target.read_bytes() == page.png

    def test_close_only_disconnects(self) -> None:
        browser, sockets = _browser(FakePage())
        browser.setup()
        browser.close()
        assert sockets[0].closed
        assert browser.connection is None
        assert [m["method"] for m in sockets[0].sent] == []  # nothing like Browser.close was sent
        browser.setup()  # reconnects
        assert len(sockets) == 2

    def test_commands_before_setup_fail(self) -> None:
        browser, _ = _browser(FakePage())
        with pytest.raises(RuntimeError, match="setup"):
            browser.execute_js("return 1")
