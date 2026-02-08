"""Text stream helpers used by the Brython unit-test harness."""

from __future__ import annotations

from typing import List


class BrythonTestStream:
    """Simple text stream that mirrors writes to stdout while buffering content."""

    def __init__(self) -> None:
        self.buffer: List[str] = []

    def write(self, text: str) -> int:
        """Write ``text`` to the buffer and echo it to stdout."""
        self.buffer.append(text)
        print(text, end="")  # Print without additional newline
        return len(text)

    def writeln(self, text: str = "") -> int:
        """Write ``text`` with a newline suffix to the buffer and stdout."""
        full_text = f"{text}\n"
        self.buffer.append(full_text)
        print(full_text, end="")  # Print without additional newline
        return len(full_text)

    def flush(self) -> None:
        """Flush the stream (no-op; required for compatibility)."""
        return None

    def get_output(self) -> str:
        """Return the concatenated captured output."""
        return "".join(self.buffer)
