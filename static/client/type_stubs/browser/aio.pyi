"""Type stubs for Brython's browser.aio module.

Provides async utilities for scheduling coroutines and sleeping
within the Brython event loop.
"""

from __future__ import annotations

from typing import Any, Awaitable, Coroutine

def run(coroutine: Coroutine[Any, Any, Any]) -> None: ...
def sleep(seconds: float) -> Awaitable[None]: ...
