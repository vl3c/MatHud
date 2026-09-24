"""MatHud desktop launcher.

Starts the MatHud Flask app on a local port in a background thread and shows
it in a native window through pywebview (Edge WebView2 on Windows). Closing
the window stops the server. With ``--browser`` the app opens in the default
web browser instead, which needs no extra dependencies.

Usage:
    python mathud_desktop.py [--port PORT] [--browser] [--devtools]

pywebview is optional: ``pip install -r requirements-desktop.txt``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

from werkzeug.serving import BaseWSGIServer, make_server

if TYPE_CHECKING:
    from static.app_manager import MatHudFlask


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_HOST = "127.0.0.1"
# A stable port keeps the page origin (and its localStorage) the same across
# launches; a free port is used when it is taken.
PREFERRED_PORT = 5100
SERVER_READY_TIMEOUT_S = 30.0
WINDOW_TITLE = "MatHud"
DEFAULT_WINDOW_SIZE = (1400, 900)
MIN_WINDOW_SIZE = (800, 600)
WINDOW_STATE_FILENAME = "desktop_window.json"
WEBVIEW_STORAGE_DIRNAME = "webview"
DESKTOP_INSTALL_HINT = "pip install -r requirements-desktop.txt"
# Readiness checks talk to localhost directly; system or environment proxies
# (possibly unreachable) must not be consulted.
_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class BackgroundServer:
    """A werkzeug WSGI server running in a daemon thread that can be stopped."""

    def __init__(self, server: BaseWSGIServer) -> None:
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="mathud-server", daemon=True)

    @property
    def port(self) -> int:
        return int(self._server.port)

    @property
    def url(self) -> str:
        return f"http://{LOCAL_HOST}:{self.port}/"

    def start(self) -> None:
        self._thread.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop serving and wait for the server thread to finish."""
        if self._thread.is_alive():
            self._server.shutdown()
            self._thread.join(timeout)
        self._server.server_close()


def bind_local_socket(port: int, host: str = LOCAL_HOST) -> socket.socket:
    """Bind a listening TCP socket, failing if another process owns the port.

    Windows' SO_REUSEADDR lets a second socket share a port that is already in
    use, so the port is claimed exclusively there instead.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(128)
    except OSError:
        sock.close()
        raise
    return sock


def choose_server_socket(port: Optional[int], host: str = LOCAL_HOST) -> socket.socket:
    """Bind the requested port, or the preferred port with a free-port fallback.

    Args:
        port: Explicit port to bind (errors propagate), or None to pick one.
        host: Interface to bind; the desktop app only serves localhost.
    """
    if port is not None:
        return bind_local_socket(port, host)
    try:
        return bind_local_socket(PREFERRED_PORT, host)
    except OSError:
        return bind_local_socket(0, host)


def start_background_server(app: Any, port: Optional[int] = None, host: str = LOCAL_HOST) -> BackgroundServer:
    """Serve ``app`` on localhost in a background thread.

    Args:
        app: WSGI application to serve.
        port: Explicit port, or None for the preferred port or a free one.
        host: Interface to bind.

    Raises:
        OSError: If an explicit port cannot be bound.
    """
    sock = choose_server_socket(port, host)
    try:
        server = make_server(host, sock.getsockname()[1], app, threaded=True, fd=sock.fileno())
    finally:
        # The server works on its own duplicate of the socket.
        sock.close()
    background = BackgroundServer(server)
    background.start()
    return background


def wait_for_server(
    url: str,
    timeout: float = SERVER_READY_TIMEOUT_S,
    interval: float = 0.1,
    alive: Optional[Callable[[], bool]] = None,
) -> bool:
    """Poll ``url`` until the server answers or ``timeout`` seconds pass.

    Any HTTP response counts as ready, including error statuses such as a login
    redirect or 401, because they prove the server is accepting requests.

    Args:
        alive: Optional check (such as the server thread's ``is_alive``); once
            it returns False the wait ends early with False.
    """
    deadline = time.monotonic() + timeout
    while alive is None or alive():
        try:
            with _LOCAL_OPENER.open(url, timeout=max(interval, 1.0)):
                return True
        except urllib.error.HTTPError:
            return True
        except (urllib.error.URLError, OSError):
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)
    return False


def user_data_dir() -> Path:
    """Return the per-user directory for desktop settings and WebView storage."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / "MatHud"
        return Path.home() / "AppData" / "Local" / "MatHud"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "MatHud"
    config_home = os.environ.get("XDG_CONFIG_HOME")
    return (Path(config_home) if config_home else Path.home() / ".config") / "mathud"


def load_window_state(path: Path) -> Dict[str, Any]:
    """Load the saved window geometry, ignoring missing or malformed files."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    state: Dict[str, Any] = {}
    for key in ("width", "height", "x", "y"):
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            state[key] = value
    if state.get("width", 0) < MIN_WINDOW_SIZE[0] or state.get("height", 0) < MIN_WINDOW_SIZE[1]:
        state.pop("width", None)
        state.pop("height", None)
    if "x" not in state or "y" not in state:
        state.pop("x", None)
        state.pop("y", None)
    if raw.get("maximized") is True:
        state["maximized"] = True
    return state


def save_window_state(path: Path, state: Dict[str, Any]) -> None:
    """Write the window geometry; failures are reported but never fatal."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"Could not save window state to {path}: {e}")


def position_is_visible(x: int, y: int, screens: Sequence[Any]) -> bool:
    """Return True when the window's top-left corner lies on one of ``screens``."""
    for screen in screens:
        try:
            left, top = int(screen.x), int(screen.y)
            right, bottom = left + int(screen.width), top + int(screen.height)
        except (AttributeError, TypeError, ValueError):
            continue
        if left <= x < right and top <= y < bottom:
            return True
    return False


class WindowStateTracker:
    """Records the last normal (not minimized or maximized) window geometry.

    pywebview reports size and position through window events; reading them
    back while the window is closing is unreliable, so they are tracked live.
    """

    def __init__(self, initial: Dict[str, Any]) -> None:
        self.state: Dict[str, Any] = {key: value for key, value in initial.items() if key != "maximized"}
        self.state["maximized"] = bool(initial.get("maximized", False))
        self._minimized = False

    def attach(self, window: Any) -> None:
        window.events.resized += self.on_resized
        window.events.moved += self.on_moved
        window.events.maximized += self.on_maximized
        window.events.minimized += self.on_minimized
        window.events.restored += self.on_restored

    def on_resized(self, width: int, height: int) -> None:
        if not self._minimized and not self.state["maximized"]:
            self.state["width"], self.state["height"] = int(width), int(height)

    def on_moved(self, x: int, y: int) -> None:
        if not self._minimized and not self.state["maximized"]:
            self.state["x"], self.state["y"] = int(x), int(y)

    def on_maximized(self) -> None:
        self.state["maximized"] = True

    def on_minimized(self) -> None:
        self._minimized = True

    def on_restored(self) -> None:
        self._minimized = False
        self.state["maximized"] = False


def create_flask_app() -> "MatHudFlask":
    """Create the MatHud app exactly as ``python app.py`` does, in local mode."""
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    # PORT marks a hosted deployment (auth, secure cookies); the desktop app is local only.
    os.environ.pop("PORT", None)
    # .env files are reloaded at startup and on auth checks and may put PORT back,
    # so mark the process as local explicitly (see AppManager.is_deployed).
    os.environ["MATHUD_LOCAL_MODE"] = "1"
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    from app import app

    return app


def pywebview_available() -> bool:
    return importlib.util.find_spec("webview") is not None


def _offer_browser_fallback() -> bool:
    """Explain the missing dependency; return True if the user wants the browser."""
    print("The MatHud desktop window needs pywebview, which is not installed.")
    print(f"Install it with: {DESKTOP_INSTALL_HINT}")
    print("Or run with --browser to open MatHud in your default browser.")
    if not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        answer = input("Open MatHud in your default browser now? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def _start_server_or_report(app: Any, port: Optional[int]) -> Optional[BackgroundServer]:
    try:
        server = start_background_server(app, port)
    except OSError as e:
        print(f"Could not start the MatHud server on port {port}: {e}")
        return None
    if not wait_for_server(server.url):
        print(f"The MatHud server did not respond at {server.url}")
        server.shutdown()
        return None
    return server


def _block_until_interrupted() -> None:
    while True:
        time.sleep(1)


def run_in_browser(port: Optional[int]) -> int:
    """Serve MatHud and open it in the default browser until Ctrl+C."""
    server = _start_server_or_report(create_flask_app(), port)
    if server is None:
        return 1
    try:
        print(f"MatHud is running at {server.url}")
        print("Press Ctrl+C to stop the server")
        webbrowser.open(server.url)
        _block_until_interrupted()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


def _initial_window_geometry(saved: Dict[str, Any], webview: Any) -> Dict[str, Any]:
    geometry: Dict[str, Any] = {
        "width": saved.get("width", DEFAULT_WINDOW_SIZE[0]),
        "height": saved.get("height", DEFAULT_WINDOW_SIZE[1]),
        "maximized": bool(saved.get("maximized", False)),
    }
    if "x" in saved and "y" in saved:
        try:
            screens = webview.screens
        except Exception:
            screens = []
        if position_is_visible(saved["x"], saved["y"], screens):
            geometry["x"], geometry["y"] = saved["x"], saved["y"]
    return geometry


def run_in_window(port: Optional[int], devtools: bool = False) -> int:
    """Serve MatHud and show it in a native window; stop the server on close."""
    started = time.perf_counter()
    import webview

    server = _start_server_or_report(create_flask_app(), port)
    if server is None:
        return 1

    data_dir = user_data_dir()
    state_path = data_dir / WINDOW_STATE_FILENAME
    geometry = _initial_window_geometry(load_window_state(state_path), webview)
    tracker = WindowStateTracker(geometry)
    try:
        window = webview.create_window(
            WINDOW_TITLE,
            server.url,
            min_size=MIN_WINDOW_SIZE,
            text_select=True,
            **geometry,
        )
        if window is None:
            print("pywebview could not create the MatHud window.")
            return 1
        tracker.attach(window)

        first_load = threading.Event()

        def _report_first_load() -> None:
            if not first_load.is_set():
                first_load.set()
                print(f"MatHud window loaded in {time.perf_counter() - started:.1f}s ({server.url})")

        window.events.loaded += _report_first_load
        # Persistent (non-private) storage keeps localStorage settings between launches.
        webview.start(
            private_mode=False,
            storage_path=str(data_dir / WEBVIEW_STORAGE_DIRNAME),
            debug=devtools,
        )
    finally:
        save_window_state(state_path, tracker.state)
        server.shutdown()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open MatHud in a desktop window.")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=None,
        help=f"Port to serve on (default: {PREFERRED_PORT}, or a free port if it is taken)",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open MatHud in the default web browser instead of a desktop window",
    )
    parser.add_argument(
        "--devtools",
        action="store_true",
        help="Enable the WebView developer tools (right-click > Inspect)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.browser:
        return run_in_browser(args.port)
    if not pywebview_available():
        return run_in_browser(args.port) if _offer_browser_fallback() else 1
    return run_in_window(args.port, devtools=args.devtools)


if __name__ == "__main__":
    sys.exit(main())
