"""Circle rendering helper for drawing circle outlines.

This module provides the render_circle_helper function that renders a circle
as a stroked outline at its center position with the specified radius.
"""

from __future__ import annotations

from rendering.helpers.shape_decorator import _manages_shape
from rendering.primitives import StrokeStyle


@_manages_shape
def _render_circle(primitives, center, radius, stroke):
    """Render a circle outline at screen coordinates."""
    primitives.stroke_circle(center, float(radius), stroke)


def render_circle_helper(primitives, circle, coordinate_mapper, style):
    """Render a circle drawable.

    Args:
        primitives: The renderer primitives interface.
        circle: The Circle drawable with center point and radius.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.
        style: Style dictionary with circle_color and circle_stroke_width.
    """
    try:
        center = coordinate_mapper.math_to_screen(circle.center.x, circle.center.y)
        radius = coordinate_mapper.scale_value(circle.radius)
    except Exception:
        return
    if not center or radius is None:
        return
    stroke = StrokeStyle(
        color=str(getattr(circle, "color", style.get("circle_color", "#000"))),
        width=float(style.get("circle_stroke_width", 1) or 1),
    )
    _render_circle(primitives, center, radius, stroke)
