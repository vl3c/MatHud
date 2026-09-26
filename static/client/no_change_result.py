"""Return value for a canvas tool call that succeeded without changing the canvas.

ResultProcessor reports the message to the model instead of the success message, and
the call adds no undo entry.
"""

from __future__ import annotations


class NoChangeResult:
    """A successful tool call that left the canvas unchanged, with a message saying why."""

    def __init__(self, message: str) -> None:
        self.message: str = message

    def __repr__(self) -> str:
        return f"NoChangeResult({self.message!r})"
