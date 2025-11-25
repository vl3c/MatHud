from __future__ import annotations

from rendering.helpers.shape_decorator import _manages_shape
from rendering.renderer_primitives import StrokeStyle


@_manages_shape
def _render_circle(primitives, center, radius, stroke):
    primitives.stroke_circle(center, float(radius), stroke)


def render_circle_helper(primitives, circle, coordinate_mapper, style):
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

