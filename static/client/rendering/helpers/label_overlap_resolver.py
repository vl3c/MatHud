"""Simpler label overlap resolver for greedy vertical displacement.

This module provides a lightweight alternative to the full layout solver
for resolving label overlaps with a simple greedy algorithm.

Key Features:
    - Linear overlap detection (no spatial hash)
    - Greedy vertical offset selection
    - Group-based caching for consistent positions
    - Configurable step size and max displacement
    - Padding support for visual spacing
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

Rect = Tuple[float, float, float, float]


def _coerce_float(value: Any, default: float) -> float:
    """Safely convert a value to float with fallback.

    Args:
        value: Value to convert.
        default: Fallback if conversion fails or result is NaN.

    Returns:
        Float value or default.
    """
    try:
        out = float(value)
    except Exception:
        return default
    if out != out:  # NaN check without math import
        return default
    return out


def _coerce_int(value: Any, default: int) -> int:
    """Safely convert a value to int with fallback.

    Args:
        value: Value to convert.
        default: Fallback if conversion fails.

    Returns:
        Integer value or default.
    """
    try:
        return int(value)
    except Exception:
        return default


def inflate_rect(rect: Rect, padding: float) -> Rect:
    """Expand a rectangle by padding on all sides.

    Args:
        rect: Tuple of (min_x, max_x, min_y, max_y).
        padding: Pixels to add on each side.

    Returns:
        New rectangle tuple with expanded bounds.
    """
    pad = _coerce_float(padding, 0.0)
    if pad <= 0:
        return rect
    min_x, max_x, min_y, max_y = rect
    return (min_x - pad, max_x + pad, min_y - pad, max_y + pad)


def shift_rect_y(rect: Rect, dy: float) -> Rect:
    """Shift a rectangle vertically by a given amount.

    Args:
        rect: Tuple of (min_x, max_x, min_y, max_y).
        dy: Vertical displacement in pixels.

    Returns:
        New rectangle tuple with shifted y bounds.
    """
    shift = _coerce_float(dy, 0.0)
    if shift == 0.0:
        return rect
    min_x, max_x, min_y, max_y = rect
    return (min_x, max_x, min_y + shift, max_y + shift)


def rects_intersect(a: Rect, b: Rect) -> bool:
    """Check if two rectangles overlap.

    Touching edges are treated as non-overlapping.

    Args:
        a: First rectangle (min_x, max_x, min_y, max_y).
        b: Second rectangle (min_x, max_x, min_y, max_y).

    Returns:
        True if rectangles overlap (not just touch).
    """
    # Treat touching edges as non-overlapping so labels can stack flush.
    return not (a[1] <= b[0] or a[0] >= b[1] or a[3] <= b[2] or a[2] >= b[3])


def count_overlaps(placed: List[Rect], rect: Rect) -> int:
    """Count how many placed rectangles overlap with rect.

    Args:
        placed: List of already-placed rectangles.
        rect: Rectangle to test for overlaps.

    Returns:
        Number of overlapping rectangles.
    """
    count = 0
    for other in placed:
        if rects_intersect(rect, other):
            count += 1
    return count


def pick_non_overlapping_dy(
    placed_rects: List[Rect],
    base_rect: Rect,
    *,
    step: float,
    max_steps: int,
    padding: float,
) -> float:
    """Greedy vertical offset selection.

    Tries: 0, +step, -step, +2*step, -2*step, ...
    Returns first non-overlapping dy, else returns the dy with fewest overlaps.
    """
    step_px = _coerce_float(step, 1.0)
    if step_px <= 0:
        step_px = 1.0
    steps = _coerce_int(max_steps, 0)
    if steps < 0:
        steps = 0

    padded = inflate_rect(base_rect, padding)

    best_dy = 0.0
    best_overlaps = count_overlaps(placed_rects, padded)
    if best_overlaps == 0:
        return 0.0

    for i in range(1, steps + 1):
        for dy in (i * step_px, -i * step_px):
            candidate = shift_rect_y(padded, dy)
            overlaps = count_overlaps(placed_rects, candidate)
            if overlaps == 0:
                return dy
            if overlaps < best_overlaps:
                best_overlaps = overlaps
                best_dy = dy

    return best_dy


class ScreenOffsetLabelOverlapResolver:
    """Greedy resolver for screen-offset label overlaps.

    Attributes:
        _max_steps: Maximum step multiples to try.
        _padding_px: Padding between labels in pixels.
        _placed_rects: List of placed rectangle bounds.
        _group_to_dy: Cache of resolved displacements by group.
    """

    def __init__(self, *, max_steps: int = 10, padding_px: float = 2.0) -> None:
        self._max_steps = _coerce_int(max_steps, 10)
        self._padding_px = _coerce_float(padding_px, 2.0)
        self._placed_rects: List[Rect] = []
        self._group_to_dy: Dict[Any, float] = {}

    def reset(self) -> None:
        """Clear all placed rectangles and cached displacements."""
        self._placed_rects.clear()
        self._group_to_dy.clear()

    def get_or_place_dy(self, group: Any, base_rect: Rect, *, step: float) -> float:
        """Get cached or compute new vertical displacement for a group.

        Args:
            group: Identifier for the label group.
            base_rect: Bounding rectangle at dy=0.
            step: Vertical step size for displacement search.

        Returns:
            Vertical displacement (dy) to apply to the label.
        """
        if group in self._group_to_dy:
            return self._group_to_dy[group]

        dy = pick_non_overlapping_dy(
            self._placed_rects,
            base_rect,
            step=step,
            max_steps=self._max_steps,
            padding=self._padding_px,
        )
        self._group_to_dy[group] = dy
        self._placed_rects.append(shift_rect_y(inflate_rect(base_rect, self._padding_px), dy))
        return dy


