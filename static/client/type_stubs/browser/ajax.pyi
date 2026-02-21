"""Type stubs for Brython's browser.ajax module.

Provides HTTP request functionality for making AJAX calls from
the browser.  Both ``ajax()`` (lowercase function) and ``Ajax()``
(class constructor) return request objects with the same interface.
"""

from __future__ import annotations

from typing import Any, Callable

class AjaxRequest:
    """Object returned by ``ajax()`` — wraps a browser XMLHttpRequest."""

    status: int
    text: str
    response: Any
    responseType: str

    def bind(self, event: str, handler: Callable[..., Any]) -> None: ...
    def open(self, method: str, url: str, async_: bool = ...) -> None: ...
    def set_header(self, name: str, value: str) -> None: ...
    def send(self, data: str | None = ...) -> None: ...

class Ajax:
    """Alternate constructor form used as ``ajax.Ajax()``."""

    status: int
    text: str
    response: Any
    responseType: str

    def bind(self, event: str, handler: Callable[..., Any]) -> None: ...
    def open(self, method: str, url: str, async_: bool = ...) -> None: ...
    def set_header(self, name: str, value: str) -> None: ...
    def send(self, data: str | None = ...) -> None: ...

def ajax(timeout: int = ...) -> AjaxRequest: ...
def post(
    url: str,
    *,
    data: str = ...,
    headers: dict[str, str] = ...,
    oncomplete: Callable[..., Any] = ...,
    onerror: Callable[..., Any] = ...,
) -> None: ...
