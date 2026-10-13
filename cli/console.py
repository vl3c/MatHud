"""Console streams that never fail on a character they cannot encode.

On Windows, output that is piped or redirected uses the locale code page
(cp1252 and the like), which cannot encode many characters that model replies
and tool results contain ("≈", "π", "→"). Printing them would raise
``UnicodeEncodeError`` after the work is done. ``make_stdio_safe`` keeps each
stream's encoding and replaces what it cannot encode with "?" instead.
"""

from __future__ import annotations

import io
import sys


def make_stdio_safe() -> None:
    """Make stdout and stderr replace unencodable characters instead of raising."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if isinstance(stream, io.TextIOWrapper) and stream.errors == "strict":
            stream.reconfigure(errors="replace")
