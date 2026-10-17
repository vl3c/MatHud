"""Asks the server's SymPy engine (POST /api/cas, static/sympy_cas.py) for a CAS result.

The request is synchronous, like search_tools: the CAS tools return their result directly.
SymPy is the app's only symbolic engine. When the server has no SymPy (503) the client stops
asking for the rest of the page's life; the symbolic tools then say so (unavailable_reason)
and the numeric ones still answer.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

_NOT_INSTALLED = (
    "the server's SymPy engine is not available (install the packages in requirements.txt and restart the app)"
)
_NOT_ANSWERING = "the server's SymPy engine did not answer (its worker failed or the server is unreachable)"


class SympyClient:
    """Every method is static; _unavailable remembers a server without SymPy."""

    _unavailable = False
    # Why the last request got no answer: _NOT_INSTALLED or _NOT_ANSWERING
    _last_failure = _NOT_INSTALLED

    @staticmethod
    def compute(operation: str, **args: Optional[str]) -> Optional[Dict[str, Any]]:
        """The server's answer, {"result": ...} or {"error": ...} or {"timeout": True}; None when
        SymPy is unavailable (unavailable_reason says why). An argument the server refuses as too
        long is an {"error": ...}."""
        if SympyClient._unavailable:
            SympyClient._last_failure = _NOT_INSTALLED
            return None
        SympyClient._last_failure = _NOT_ANSWERING
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
            SympyClient._last_failure = _NOT_INSTALLED
            return None
        if request.status == 400:
            return {"error": SympyClient._refusal(request.text)}
        if request.status != 200:
            return None
        try:
            data = json.loads(request.text).get("data")
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _refusal(text: Any) -> str:
        try:
            message = json.loads(text).get("message")
        except Exception:
            message = None
        return str(message) if message else "the server refused the request"

    @staticmethod
    def answer(operation: str, **args: Optional[str]) -> Optional[Dict[str, Any]]:
        """The outcome of a request, for the tools to word:

        - None: SymPy is unavailable (unavailable_reason says why);
        - {"error": message}: SymPy refused the input (its parser does not read it, or it is too long);
        - {"result": None, "timeout": True}: SymPy ran out of time;
        - {"result": value}: SymPy's answer, None when it has none.
        """
        response = SympyClient.compute(operation, **args)
        if response is None:
            return None
        if "error" in response:
            return {"error": str(response["error"])}
        if response.get("timeout"):
            return {"result": None, "timeout": True}
        return {"result": response.get("result")}

    @staticmethod
    def unavailable_reason() -> str:
        """Why the last request had no answer, as a clause: "the server's SymPy engine is not available (...)"."""
        return SympyClient._last_failure
