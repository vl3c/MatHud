"""
The canvas state as sent with a prompt.

A prompt carries ``Canvas.get_canvas_state()`` plus the canvas size in CSS pixels,
so the server can tell how big the drawings are on screen and add a view note when
they are too small or outside the view (``static/canvas_state_formatter.py``).
The size is added to the prompt's copy only, never to saved workspaces or traces.

Pure Python (no ``browser`` import), tested by ``client_tests/test_prompt_canvas_state.py``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

# Mirrors static/canvas_state_formatter.CANVAS_SIZE_KEY.
CANVAS_SIZE_KEY = "canvas_size_px"


def canvas_size_px(canvas: Any) -> Optional[Dict[str, float]]:
    """Return ``{"width", "height"}`` of the canvas in CSS pixels, or None when unknown."""
    try:
        width = float(canvas.width)
        height = float(canvas.height)
    except Exception:
        return None
    if not (math.isfinite(width) and math.isfinite(height)) or width <= 0 or height <= 0:
        return None
    return {"width": round(width, 2), "height": round(height, 2)}


def with_canvas_size(state: Any, canvas: Any) -> Any:
    """Return a shallow copy of ``state`` with the canvas size added; ``state`` itself is unchanged.

    Returns ``state`` as given when it is not a dict or the size is unknown.
    """
    size = canvas_size_px(canvas)
    if size is None or not isinstance(state, dict):
        return state
    result = dict(state)
    result[CANVAS_SIZE_KEY] = size
    return result
