"""Segment rendering helper for drawing line segments.

This module provides the render_segment_helper function that renders a segment
as a stroked line between two points with an optional label.
"""

from __future__ import annotations

from rendering.helpers.shape_decorator import _manages_shape
from rendering.helpers.label_renderer import render_label_helper
from rendering.primitives import StrokeStyle


@_manages_shape
def _render_segment(primitives, start, end, stroke):
    """Render a line segment between two screen coordinates."""
    primitives.stroke_line(start, end, stroke, include_width=False)


def render_segment_helper(primitives, segment, coordinate_mapper, style):
    """Render a segment drawable with its optional label.

    Args:
        primitives: The renderer primitives interface.
        segment: The Segment drawable with point1, point2, and color.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.
        style: Style dictionary with segment_color and segment_stroke_width.
    """
    try:
        start = coordinate_mapper.math_to_screen(segment.point1.x, segment.point1.y)
        end = coordinate_mapper.math_to_screen(segment.point2.x, segment.point2.y)
    except Exception:
        return
    if not start or not end:
        return
    stroke = StrokeStyle(
        color=str(getattr(segment, "color", style.get("segment_color", "#000"))),
        width=float(style.get("segment_stroke_width", 1) or 1),
    )
    _render_segment(primitives, start, end, stroke)

    try:
        if hasattr(segment, "_sync_label_position"):
            segment._sync_label_position()
    except Exception:
        pass

    embedded_label = getattr(segment, "label", None)
    if embedded_label is not None:
        render_label_helper(primitives, embedded_label, coordinate_mapper, style)
