"""Asks the server's SymPy engine (POST /api/cas, static/sympy_cas.py) for a CAS result.

The request is synchronous, like search_tools: the CAS tools return their result directly.
When the server has no SymPy (503) the client stops asking for the rest of the page's life
and the tools use nerdamer, the browser's symbolic engine, as before.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


class SympyClient:
    """Every method is static; _unavailable remembers a server without SymPy."""

    _unavailable = False

    @staticmethod
    def compute(operation: str, **args: Optional[str]) -> Optional[Dict[str, Any]]:
        """The server's answer, {"result": ...} or {"error": ...} or {"timeout": True}; None when unreachable."""
        if SympyClient._unavailable:
            return None
        try:
            from browser import ajax

            request = ajax.Ajax()
            request.open("POST", "/api/cas", False)  # synchronous
            request.set_header("Content-Type", "application/json")
            request.send(json.dumps({"operation": operation, "args": args}))
        except Exception:
            return None
        if request.status == 503:
            SympyClient._unavailable = True
            return None
        if request.status != 200:
            return None
        try:
            data = json.loads(request.text).get("data")
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def answer(operation: str, **args: Optional[str]) -> Optional[Dict[str, Any]]:
        """{"result": value or None} when SymPy handled the request (with "timeout": True when it
        ran out of time), else None (use nerdamer).

        A refused expression (one the server's parser does not accept, e.g. a function it
        lacks) is left to nerdamer; a computation that ran out of time counts as handled,
        with no result.
        """
        response = SympyClient.compute(operation, **args)
        if response is None or "error" in response:
            return None
        if response.get("timeout"):
            return {"result": None, "timeout": True}
        return {"result": response.get("result")}
