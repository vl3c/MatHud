"""The browser interface the CLI drives the app through, and the hook calls built on it.

Two backends implement ``AppBrowser``:

- ``cli.browser.BrowserAutomation``: a headless Chrome started by Selenium (the default);
- ``cli.cdp.CDPBrowser``: the desktop window, attached over the Chrome DevTools
  Protocol (``mathud_desktop.py --automation-port N``).

Everything the scenario runner and the desktop commands need is a JavaScript
call returning a string or plain value (the ``window.*MatHud*`` hooks), a
screenshot, a reload and a close, so both backends offer the same small surface.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Protocol


class AppBrowser(Protocol):
    """What the scenario runner needs from a browser on the app."""

    # True when the browser is someone's open window (attach mode): it is never
    # navigated on open, and closing only disconnects.
    attached: bool

    def setup(self) -> None: ...

    def navigate_to_app(self) -> bool: ...

    def wait_for_app_ready(self, timeout: int = ...) -> bool: ...

    def execute_js(self, script: str, *args: Any, timeout: int = ...) -> Any: ...

    def call_hook(self, name: str, *args: Any, timeout: int = ...) -> dict[str, Any]: ...

    def screenshot(self, output_path: str) -> bool: ...

    def reload(self) -> bool: ...

    def close(self) -> None: ...


class HookClient:
    """Calls to the app's JSON hooks, built on the backend's ``execute_js``."""

    def execute_js(self, script: str, *args: Any, timeout: int = 30) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def call_hook(self, name: str, *args: Any, timeout: int = 30) -> dict[str, Any]:
        """Call one of the app's ``window.*MatHud*`` JSON hooks and parse its reply.

        Args:
            name: Hook name on ``window``, e.g. ``getMatHudCanvasState``.
            *args: Arguments passed to the hook (strings, usually JSON).
            timeout: Script timeout in seconds.

        Returns:
            The parsed JSON reply.

        Raises:
            RuntimeError: If the hook is missing or does not return a JSON object.
        """
        script = f"return typeof window.{name} === 'function' ? window.{name}.apply(null, arguments) : null"
        result = self.execute_js(script, *args, timeout=timeout)
        if not result:
            raise RuntimeError(f"window.{name} is not available")
        parsed = json.loads(result)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"window.{name} returned {type(parsed).__name__}, expected an object")
        return parsed

    def get_canvas_snapshot(self, options: Optional[dict[str, Any]] = None, timeout: int = 30) -> dict[str, Any]:
        """Return ``{"state": ..., "inspection": ...?}`` from ``getMatHudCanvasState``.

        Args:
            options: Hook options, e.g. ``{"inspect": True, "samples": {"f": [1, 2]}}``.
            timeout: Script timeout in seconds.
        """
        return self.call_hook("getMatHudCanvasState", json.dumps(options or {}), timeout=timeout)

    def get_canvas_state(self) -> dict[str, Any]:
        """Get the current canvas state (``Canvas.get_canvas_state``) as a dictionary.

        Returns:
            Canvas state dictionary, or ``{}`` when the hook is unavailable.
        """
        try:
            payload = self.get_canvas_snapshot()
        except RuntimeError:
            return {}
        state = payload.get("state")
        return state if isinstance(state, dict) else {}

    def run_tool_calls(self, calls: list[dict[str, Any]], timeout: int = 60) -> dict[str, Any]:
        """Run one tool batch through ``runMatHudToolCalls``, the path a model's batch takes.

        Args:
            calls: ``[{"function_name": ..., "arguments": {...}}]`` (or ``{"tool", "args"}``).
            timeout: Script timeout in seconds.

        Returns:
            The hook reply: ``traced`` calls, ``state`` after the batch and undo depths.
        """
        return self.call_hook("runMatHudToolCalls", json.dumps(calls), timeout=timeout)

    def reset_session(self, options: Optional[dict[str, Any]] = None, timeout: int = 30) -> dict[str, Any]:
        """Reset canvas, undo history, traces, metrics and chat through ``resetMatHudSession``."""
        return self.call_hook("resetMatHudSession", json.dumps(options or {}), timeout=timeout)

    def call_function_registry(self, function_name: str, args: dict[str, Any]) -> Any:
        """Execute one AI tool as a single-call batch and return its result.

        Args:
            function_name: Name of the tool to call.
            args: Dictionary of arguments.

        Returns:
            The tool's result value (the value a model would see).

        Raises:
            RuntimeError: If the batch could not run.
        """
        reply = self.run_tool_calls([{"function_name": function_name, "arguments": args}])
        if reply.get("status") != "ok":
            raise RuntimeError(str(reply.get("error", "tool call failed")))
        traced = reply.get("traced") or []
        return traced[0].get("result") if traced else None
