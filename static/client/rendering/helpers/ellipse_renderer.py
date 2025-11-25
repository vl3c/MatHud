from __future__ import annotations

import math

from rendering.helpers.shape_decorator import _manages_shape
from rendering.renderer_primitives import StrokeStyle


@_manages_shape
def _render_ellipse(primitives, center, rx, ry, rotation_rad, stroke):
    primitives.stroke_ellipse(center, rx, ry, rotation_rad, stroke)


def render_ellipse_helper(primitives, ellipse, coordinate_mapper, style):
    try:
        center = coordinate_mapper.math_to_screen(ellipse.center.x, ellipse.center.y)
        radius_x = coordinate_mapper.scale_value(getattr(ellipse, "radius_x", None))
        radius_y = coordinate_mapper.scale_value(getattr(ellipse, "radius_y", None))
    except Exception:
        return
    if not center or radius_x is None or radius_y is None:
        return

    try:
        rx = float(radius_x)
    except Exception:
        rx = None
    try:
        ry = float(radius_y)
    except Exception:
        ry = None
    if rx is None or ry is None or rx <= 0 or ry <= 0:
        return

    color = str(getattr(ellipse, "color", style.get("ellipse_color", "#000")))
    stroke_width_raw = style.get("ellipse_stroke_width", 1) or 1
    try:
        stroke_width = float(stroke_width_raw)
    except Exception:
        stroke_width = 1.0
    stroke = StrokeStyle(color=color, width=stroke_width)

    rotation_deg = getattr(ellipse, "rotation_angle", 0) or 0
    try:
        rotation_rad = -math.radians(float(rotation_deg))
    except Exception:
        rotation_rad = 0.0

    _render_ellipse(primitives, center, rx, ry, rotation_rad, stroke)

