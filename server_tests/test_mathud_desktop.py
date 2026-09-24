"""Tests for the pywebview desktop launcher (mathud_desktop.py).

No real window is opened: pywebview is replaced with a fake module and the
Flask app with a tiny WSGI app, while the server itself runs for real on
localhost.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import time
import types
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, List
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import mathud_desktop
from cli.main import cli
from mathud_desktop import (
    BackgroundServer,
    WindowStateTracker,
    choose_server_socket,
    load_window_state,
    position_is_visible,
    save_window_state,
    start_background_server,
    wait_for_server,
)

StartResponse = Callable[[str, List[Any]], Any]


def _hello_app(environ: dict[str, Any], start_response: StartResponse) -> Iterable[bytes]:
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"hello"]


def _error_app(environ: dict[str, Any], start_response: StartResponse) -> Iterable[bytes]:
    start_response("500 INTERNAL SERVER ERROR", [("Content-Type", "text/plain")])
    return [b"boom"]


@pytest.fixture
def occupied_port() -> Iterable[int]:
    """A localhost port held by another listening socket for the test's duration."""
    sock = mathud_desktop.bind_local_socket(0)
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def _unused_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


@pytest.fixture
def server() -> Iterable[BackgroundServer]:
    background = start_background_server(_hello_app, port=0)
    try:
        yield background
    finally:
        background.shutdown()


class TestPortSelection:
    def test_uses_preferred_port_when_free(self) -> None:
        preferred = _unused_port()
        with patch.object(mathud_desktop, "PREFERRED_PORT", preferred):
            sock = choose_server_socket(None)
        try:
            assert sock.getsockname()[1] == preferred
        finally:
            sock.close()

    def test_falls_back_to_free_port_when_preferred_is_taken(self, occupied_port: int) -> None:
        with patch.object(mathud_desktop, "PREFERRED_PORT", occupied_port):
            sock = choose_server_socket(None)
        try:
            port = sock.getsockname()[1]
            assert port not in (0, occupied_port)
        finally:
            sock.close()

    def test_explicit_port_in_use_raises(self, occupied_port: int) -> None:
        with pytest.raises(OSError):
            choose_server_socket(occupied_port)

    def test_binds_localhost_only(self) -> None:
        sock = choose_server_socket(0)
        try:
            assert sock.getsockname()[0] == "127.0.0.1"
        finally:
            sock.close()


class TestBackgroundServer:
    def test_serves_app_on_chosen_port(self, server: BackgroundServer) -> None:
        assert server.port > 0
        assert server.url == f"http://127.0.0.1:{server.port}/"
        assert wait_for_server(server.url, timeout=5)
        with urllib.request.urlopen(server.url, timeout=5) as response:
            assert response.read() == b"hello"

    def test_falls_back_when_preferred_port_taken(self, occupied_port: int) -> None:
        with patch.object(mathud_desktop, "PREFERRED_PORT", occupied_port):
            background = start_background_server(_hello_app)
        try:
            assert background.port != occupied_port
            assert wait_for_server(background.url, timeout=5)
        finally:
            background.shutdown()

    def test_shutdown_stops_serving_and_frees_port(self) -> None:
        background = start_background_server(_hello_app, port=0)
        assert wait_for_server(background.url, timeout=5)

        background.shutdown()

        assert not wait_for_server(background.url, timeout=0.3, interval=0.05)
        # The port can be claimed again once the server is closed.
        mathud_desktop.bind_local_socket(background.port).close()

    def test_explicit_busy_port_raises_before_starting(self, occupied_port: int) -> None:
        with pytest.raises(OSError):
            start_background_server(_hello_app, port=occupied_port)


class TestWaitForServer:
    def test_returns_false_after_timeout_when_nothing_listens(self) -> None:
        started = time.monotonic()
        assert not wait_for_server(f"http://127.0.0.1:{_unused_port()}/", timeout=0.3, interval=0.05)
        assert time.monotonic() - started < 5

    def test_http_error_status_counts_as_ready(self) -> None:
        background = start_background_server(_error_app, port=0)
        try:
            assert wait_for_server(background.url, timeout=5)
        finally:
            background.shutdown()

    def test_polls_until_server_comes_up(self) -> None:
        results = iter([OSError("refused"), OSError("refused"), MagicMock()])

        def fake_urlopen(url: str, timeout: float) -> Any:
            result = next(results)
            if isinstance(result, Exception):
                raise result
            return result

        with patch("urllib.request.urlopen", side_effect=fake_urlopen) as urlopen:
            assert wait_for_server("http://127.0.0.1:1/", timeout=5, interval=0.01)
        assert urlopen.call_count == 3


class TestWindowState:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "window.json"
        save_window_state(path, {"width": 1200, "height": 800, "x": 10, "y": 20, "maximized": False})

        assert load_window_state(path) == {"width": 1200, "height": 800, "x": 10, "y": 20}

    def test_missing_or_malformed_file_gives_defaults(self, tmp_path: Path) -> None:
        assert load_window_state(tmp_path / "absent.json") == {}
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert load_window_state(bad) == {}
        bad.write_text("[1, 2]", encoding="utf-8")
        assert load_window_state(bad) == {}

    def test_rejects_tiny_sizes_partial_positions_and_non_ints(self, tmp_path: Path) -> None:
        path = tmp_path / "window.json"
        path.write_text(json.dumps({"width": 100, "height": 50, "x": 5, "y": True, "maximized": True}))

        assert load_window_state(path) == {"maximized": True}

    def test_position_is_visible(self) -> None:
        screens = [
            types.SimpleNamespace(x=0, y=0, width=1920, height=1080),
            types.SimpleNamespace(x=1920, y=0, width=1280, height=1024),
        ]
        assert position_is_visible(100, 100, screens)
        assert position_is_visible(2000, 500, screens)
        assert not position_is_visible(-32000, -32000, screens)
        assert not position_is_visible(100, 100, [])

    def test_tracker_records_only_normal_geometry(self) -> None:
        tracker = WindowStateTracker({"width": 1000, "height": 700})

        tracker.on_resized(1100, 750)
        tracker.on_moved(40, 50)
        tracker.on_minimized()
        tracker.on_moved(-32000, -32000)
        tracker.on_restored()
        tracker.on_maximized()
        tracker.on_resized(1920, 1040)

        assert tracker.state == {"width": 1100, "height": 750, "x": 40, "y": 50, "maximized": True}

        tracker.on_restored()
        assert tracker.state["maximized"] is False


class _FakeEvent:
    def __init__(self) -> None:
        self.handlers: List[Callable[..., Any]] = []

    def __iadd__(self, handler: Callable[..., Any]) -> "_FakeEvent":
        self.handlers.append(handler)
        return self


def _fake_webview(on_start: Callable[[str], None]) -> MagicMock:
    """A stand-in for the ``webview`` module that never opens a window."""
    fake = MagicMock()
    fake.screens = [types.SimpleNamespace(x=0, y=0, width=1920, height=1080)]
    window = MagicMock()
    window.events = types.SimpleNamespace(
        **{name: _FakeEvent() for name in ("resized", "moved", "maximized", "minimized", "restored", "loaded")}
    )

    def create_window(title: str, url: str, **kwargs: Any) -> MagicMock:
        window.url = url
        return window

    def start(**kwargs: Any) -> None:
        for handler in window.events.resized.handlers:
            handler(1234, 777)
        for handler in window.events.moved.handlers:
            handler(30, 40)
        on_start(window.url)

    fake.create_window.side_effect = create_window
    fake.start.side_effect = start
    fake.window = window
    return fake


class TestRunInWindow:
    def test_shows_window_serves_app_and_stops_server_on_close(self, tmp_path: Path) -> None:
        seen: dict[str, Any] = {}

        def while_window_open(url: str) -> None:
            with urllib.request.urlopen(url, timeout=5) as response:
                seen["body"] = response.read()
            seen["url"] = url

        fake = _fake_webview(while_window_open)
        with (
            patch.dict(sys.modules, {"webview": fake}),
            patch.object(mathud_desktop, "create_flask_app", return_value=_hello_app),
            patch.object(mathud_desktop, "user_data_dir", return_value=tmp_path),
            patch("builtins.print"),
        ):
            assert mathud_desktop.run_in_window(0) == 0

        assert seen["body"] == b"hello"
        title, url = fake.create_window.call_args.args
        assert title == "MatHud"
        assert url == seen["url"]
        kwargs = fake.create_window.call_args.kwargs
        assert (kwargs["width"], kwargs["height"]) == mathud_desktop.DEFAULT_WINDOW_SIZE
        start_kwargs = fake.start.call_args.kwargs
        assert start_kwargs["private_mode"] is False
        assert start_kwargs["debug"] is False
        assert start_kwargs["storage_path"] == str(tmp_path / mathud_desktop.WEBVIEW_STORAGE_DIRNAME)
        # The server is gone once the window has closed ...
        assert not wait_for_server(url, timeout=0.3, interval=0.05)
        # ... and the last geometry was saved for the next launch.
        saved = json.loads((tmp_path / mathud_desktop.WINDOW_STATE_FILENAME).read_text(encoding="utf-8"))
        assert saved == {"width": 1234, "height": 777, "x": 30, "y": 40, "maximized": False}

    def test_restores_saved_geometry(self, tmp_path: Path) -> None:
        save_window_state(
            tmp_path / mathud_desktop.WINDOW_STATE_FILENAME, {"width": 1000, "height": 700, "x": 5, "y": 6}
        )
        fake = _fake_webview(lambda url: None)
        with (
            patch.dict(sys.modules, {"webview": fake}),
            patch.object(mathud_desktop, "create_flask_app", return_value=_hello_app),
            patch.object(mathud_desktop, "user_data_dir", return_value=tmp_path),
            patch("builtins.print"),
        ):
            mathud_desktop.run_in_window(0, devtools=True)

        kwargs = fake.create_window.call_args.kwargs
        assert (kwargs["width"], kwargs["height"], kwargs["x"], kwargs["y"]) == (1000, 700, 5, 6)
        assert fake.start.call_args.kwargs["debug"] is True

    def test_busy_explicit_port_reports_error(self, occupied_port: int, tmp_path: Path) -> None:
        fake = _fake_webview(lambda url: None)
        with (
            patch.dict(sys.modules, {"webview": fake}),
            patch.object(mathud_desktop, "create_flask_app", return_value=_hello_app),
            patch.object(mathud_desktop, "user_data_dir", return_value=tmp_path),
            patch("builtins.print") as mock_print,
        ):
            assert mathud_desktop.run_in_window(occupied_port) == 1

        fake.create_window.assert_not_called()
        assert "Could not start the MatHud server" in mock_print.call_args_list[0].args[0]


class TestMain:
    def test_browser_flag_skips_pywebview(self) -> None:
        with (
            patch.object(mathud_desktop, "run_in_browser", return_value=0) as run_in_browser,
            patch.object(mathud_desktop, "run_in_window") as run_in_window,
        ):
            assert mathud_desktop.main(["--browser", "--port", "5123"]) == 0

        run_in_browser.assert_called_once_with(5123)
        run_in_window.assert_not_called()

    def test_missing_pywebview_prints_install_hint(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch.object(mathud_desktop, "pywebview_available", return_value=False),
            patch("sys.stdin", None),
            patch.object(mathud_desktop, "run_in_browser") as run_in_browser,
        ):
            assert mathud_desktop.main([]) == 1

        output = capsys.readouterr().out
        assert "pip install -r requirements-desktop.txt" in output
        assert "--browser" in output
        run_in_browser.assert_not_called()

    def test_missing_pywebview_can_fall_back_to_browser(self) -> None:
        tty = MagicMock()
        tty.isatty.return_value = True
        with (
            patch.object(mathud_desktop, "pywebview_available", return_value=False),
            patch("sys.stdin", tty),
            patch("builtins.input", return_value="y"),
            patch("builtins.print"),
            patch.object(mathud_desktop, "run_in_browser", return_value=0) as run_in_browser,
        ):
            assert mathud_desktop.main([]) == 0

        run_in_browser.assert_called_once_with(None)

    def test_window_mode_when_pywebview_installed(self) -> None:
        with (
            patch.object(mathud_desktop, "pywebview_available", return_value=True),
            patch.object(mathud_desktop, "run_in_window", return_value=0) as run_in_window,
        ):
            assert mathud_desktop.main(["--devtools"]) == 0

        run_in_window.assert_called_once_with(None, devtools=True)


class TestRunInBrowser:
    def test_opens_browser_and_stops_server_on_ctrl_c(self) -> None:
        opened: dict[str, Any] = {}

        def fake_open(url: str) -> bool:
            with urllib.request.urlopen(url, timeout=5) as response:
                opened["body"] = response.read()
            opened["url"] = url
            return True

        with (
            patch.object(mathud_desktop, "create_flask_app", return_value=_hello_app),
            patch("webbrowser.open", side_effect=fake_open),
            patch.object(mathud_desktop, "_block_until_interrupted", side_effect=KeyboardInterrupt),
            patch("builtins.print"),
        ):
            assert mathud_desktop.run_in_browser(0) == 0

        assert opened["body"] == b"hello"
        assert not wait_for_server(opened["url"], timeout=0.3, interval=0.05)


class TestCreateFlaskApp:
    def test_port_in_dotenv_does_not_switch_to_deployed_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The README's .env example sets PORT, which marks a hosted deployment
        # (auth required, secure-only cookies) and would lock the window out.
        import dotenv

        from static.app_manager import AppManager

        env_file = tmp_path / ".env"
        env_file.write_text("PORT=5000\n", encoding="utf-8")
        real_load_dotenv = dotenv.load_dotenv
        monkeypatch.setattr("static.env_config.load_dotenv", lambda *args, **kwargs: real_load_dotenv(env_file))
        monkeypatch.chdir(Path.cwd())
        previous_sigint = signal.getsignal(signal.SIGINT)
        try:
            with (
                patch.dict(os.environ),
                patch.dict(sys.modules),
                patch.object(AppManager, "is_deployed", AppManager.__dict__["is_deployed"]),
                patch("builtins.print"),
            ):
                os.environ.pop("PORT", None)
                os.environ.pop("REQUIRE_AUTH", None)
                sys.modules.pop("app", None)
                flask_app = mathud_desktop.create_flask_app()

                assert not AppManager.is_deployed()
                assert not AppManager.requires_auth()
                assert flask_app.config.get("SESSION_COOKIE_SECURE") is not True
                response = flask_app.test_client().get("/")
                assert response.status_code == 200
        finally:
            signal.signal(signal.SIGINT, previous_sigint)


class TestDesktopCliCommand:
    def test_help(self) -> None:
        result = CliRunner().invoke(cli, ["desktop", "--help"])

        assert result.exit_code == 0
        assert "--browser" in result.output
        assert "--port" in result.output

    def test_forwards_options_to_launcher(self) -> None:
        with patch.object(mathud_desktop, "main", return_value=0) as main:
            result = CliRunner().invoke(cli, ["desktop", "--port", "5123", "--browser"])

        assert result.exit_code == 0
        main.assert_called_once_with(["--port", "5123", "--browser"])
