from __future__ import annotations

import math

from constants import default_font_family
from rendering.renderables import FunctionRenderable
from rendering.helpers.shape_decorator import _manages_shape
from rendering.primitives import FontStyle, StrokeStyle, TextAlignment


def _cull_path_to_visible(path, width, height, margin=16):
    """Cull path segments that are entirely outside the visible area."""
    if not path or width <= 0 or height <= 0:
        return path

    min_x = -margin
    max_x = width + margin
    min_y = -margin
    max_y = height + margin

    culled = []
    for sx, sy in path:
        in_bounds = min_x <= sx <= max_x and min_y <= sy <= max_y
        if in_bounds:
            culled.append((sx, sy))
        elif culled and culled[-1] is not None:
            culled.append((sx, sy))
            culled.append(None)

    result = []
    current_segment = []
    for pt in culled:
        if pt is None:
            if len(current_segment) >= 2:
                result.append(current_segment)
            current_segment = []
        else:
            current_segment.append(pt)
    if len(current_segment) >= 2:
        result.append(current_segment)

    return result if result else [path]


@_manages_shape
def _render_function_paths(primitives, screen_paths, stroke, width=0, height=0):
    for path in screen_paths:
        if len(path) < 2:
            continue
        if width > 0 and height > 0:
            culled_segments = _cull_path_to_visible(path, width, height)
            for segment in culled_segments:
                if len(segment) >= 2:
                    primitives.stroke_polyline(segment, stroke)
        else:
            primitives.stroke_polyline(path, stroke)


def _get_or_create_renderable(func, coordinate_mapper):
    renderable = getattr(func, "_renderable", None)
    if renderable is None or renderable.mapper is not coordinate_mapper:
        renderable = FunctionRenderable(func, coordinate_mapper)
        try:
            func._renderable = renderable
        except Exception:
            pass
    else:
        renderable.mapper = coordinate_mapper
    return renderable


def _build_stroke_style(func, style):
    return StrokeStyle(
        color=str(getattr(func, "color", style.get("function_color", "#000"))),
        width=float(style.get("function_stroke_width", 1) or 1),
        line_join="round",
    )


def _normalize_font_size(value):
    try:
        size_float = float(value)
    except Exception:
        return value
    if math.isfinite(size_float) and size_float.is_integer():
        return int(size_float)
    return size_float


def _render_function_label(primitives, func, screen_paths, stroke, style):
    if not getattr(func, "name", "") or not screen_paths or not screen_paths[0]:
        return
    font_size = _normalize_font_size(style.get("function_label_font_size", 12))
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


def render_function_helper(primitives, func, coordinate_mapper, style):
    try:
        renderable = _get_or_create_renderable(func, coordinate_mapper)
        screen_paths = renderable.build_screen_paths().paths
    except Exception:
        return
    if not screen_paths:
        return

    width = getattr(coordinate_mapper, "canvas_width", 0) or 0
    height = getattr(coordinate_mapper, "canvas_height", 0) or 0
    stroke = _build_stroke_style(func, style)

    _render_function_paths(primitives, screen_paths, stroke, width, height)
    _render_function_label(primitives, func, screen_paths, stroke, style)

