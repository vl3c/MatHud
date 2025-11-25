from __future__ import annotations

import math

from constants import default_font_family
from rendering.function_renderable import FunctionRenderable
from rendering.helpers.shape_decorator import _manages_shape
from rendering.primitives import FontStyle, StrokeStyle, TextAlignment


@_manages_shape
def _render_function_paths(primitives, screen_paths, stroke):
    for path in screen_paths:
        if len(path) < 2:
            continue
        primitives.stroke_polyline(path, stroke)


def render_function_helper(primitives, func, coordinate_mapper, style):
    try:
        canvas = getattr(func, "canvas", None)
        cartesian = getattr(canvas, "cartesian2axis", None) if canvas is not None else None
        renderable = FunctionRenderable(func, coordinate_mapper, cartesian)
        screen_paths = renderable.build_screen_paths().paths
    except Exception:
        return
    if not screen_paths:
        return

    stroke = StrokeStyle(
        color=str(getattr(func, "color", style.get("function_color", "#000"))),
        width=float(style.get("function_stroke_width", 1) or 1),
    )
    _render_function_paths(primitives, screen_paths, stroke)

    if getattr(func, "name", "") and screen_paths and screen_paths[0]:
        font_size_value = style.get("function_label_font_size", 12)
        try:
            font_size_float = float(font_size_value)
        except Exception:
            font_size = font_size_value
        else:
            if math.isfinite(font_size_float) and font_size_float.is_integer():
                font_size = int(font_size_float)
            else:
                font_size = font_size_float
        first_point = screen_paths[0][0]
        label_offset_x = (1 + len(func.name)) * font_size / 2.0
        position = (first_point[0] - label_offset_x, max(first_point[1], font_size))
        font_family = style.get("function_label_font_family", style.get("font_family", default_font_family))
        font = FontStyle(family=font_family, size=font_size)
        primitives.draw_text(
            func.name,
            position,
            font,
            stroke.color,
            TextAlignment(horizontal="left", vertical="alphabetic"),
        )

