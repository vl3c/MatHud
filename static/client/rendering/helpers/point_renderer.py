from __future__ import annotations

import math

from constants import default_font_family
from rendering.helpers.shape_decorator import _manages_shape
from rendering.primitives import FillStyle, FontStyle, TextAlignment


@_manages_shape
def _render_point(primitives, sx, sy, radius, point, fill, style):
    primitives.fill_circle((sx, sy), radius, fill, screen_space=True)
    _render_point_label(primitives, sx, sy, radius, point, fill, style)

def _render_point_label(primitives, sx, sy, radius, point, fill, style):
    label = getattr(point, "name", "")
    if not label:
        return
    label_text = f"{label}({round(getattr(point, 'x', 0), 3)}, {round(getattr(point, 'y', 0), 3)})"
    font_size_value = style.get("point_label_font_size", 10)
    try:
        font_size_float = float(font_size_value)
    except Exception:
        font_size = font_size_value
    else:
        if math.isfinite(font_size_float) and font_size_float.is_integer():
            font_size = int(font_size_float)
        else:
            font_size = font_size_float
    font_family = style.get("point_label_font_family", style.get("font_family", default_font_family))
    font = FontStyle(family=font_family, size=font_size)
    label_metadata = {
        "point_label": {
            "math_position": (float(getattr(point, "x", 0.0)), float(getattr(point, "y", 0.0))),
            "screen_offset": (float(radius), float(-radius)),
        }
    }
    primitives.draw_text(
        label_text,
        (sx + radius, sy - radius),
        font,
        fill.color,
        TextAlignment(horizontal="left", vertical="alphabetic"),
        {
            "user-select": "none",
            "-webkit-user-select": "none",
            "-moz-user-select": "none",
            "-ms-user-select": "none",
        },
        screen_space=True,
        metadata=label_metadata,
    )


def render_point_helper(primitives, point, coordinate_mapper, style):
    try:
        sx_sy = coordinate_mapper.math_to_screen(point.x, point.y)
    except Exception:
        return
    if not sx_sy:
        return
    sx, sy = sx_sy
    radius_raw = style.get("point_radius", 0) or 0
    try:
        radius = float(radius_raw)
    except Exception:
        return
    if radius <= 0:
        return

    fill = FillStyle(color=str(getattr(point, "color", style.get("point_color", "#000"))), opacity=None)
    _render_point(primitives, sx, sy, radius, point, fill, style)

