"""Function rendering helper for drawing mathematical function curves.

This module provides the render_function_helper function that renders
function drawables as polyline paths with optional labels.

Key Features:
    - Adaptive sampling via FunctionRenderable for smooth curves
    - Path culling to skip segments outside visible area
    - Function name label positioning at curve start
    - Renderable caching for performance
    - Stroke styling with round line joins
"""

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
def _render_function_paths(primitives, screen_paths, stroke, width=0, height=0, margin=16):
    """Render function paths as stroked polylines with culling.

    Args:
        primitives: The renderer primitives interface.
        screen_paths: List of screen coordinate paths to render.
        stroke: StrokeStyle for the function curve.
        width: Canvas width for culling (0 to disable).
        height: Canvas height for culling (0 to disable).
        margin: Pixels kept beyond each canvas edge.
    """
    for path in screen_paths:
        if len(path) < 2:
            continue
        if width > 0 and height > 0:
            culled_segments = _cull_path_to_visible(path, width, height, margin)
            for segment in culled_segments:
                if len(segment) >= 2:
                    primitives.stroke_polyline(segment, stroke)
        else:
            primitives.stroke_polyline(path, stroke)


def _unwrap_mapper(mapper):
    """Return the mapper behind a per-build caching wrapper (or the mapper itself)."""
    return getattr(mapper, "_mapper", mapper)


def _model_signature(func):
    """Serialized model state, used to drop cached paths when the function changes."""
    get_state = getattr(func, "get_state", None)
    if not callable(get_state):
        return None
    try:
        return repr(get_state())
    except Exception:
        return None


def _get_view_margin(style):
    """Fraction of the viewport sampled beyond each edge (set by renderers that reproject on pan)."""
    try:
        margin = float(style.get("function_view_margin", 0.0) or 0.0)
    except Exception:
        return 0.0
    return margin if math.isfinite(margin) and margin > 0 else 0.0


def _get_or_create_renderable(func, coordinate_mapper):
    """Get or create a FunctionRenderable for the given function.

    Caches the renderable on the function object for reuse. Plan builds wrap
    the canvas mapper in a fresh caching wrapper, so reuse is decided on the
    underlying mapper; cached paths are dropped when the model state changes.

    Args:
        func: Function drawable with expression and domain.
        coordinate_mapper: Mapper for coordinate conversion.

    Returns:
        FunctionRenderable instance for path generation.
    """
    renderable = getattr(func, "_renderable", None)
    if renderable is None or _unwrap_mapper(renderable.mapper) is not _unwrap_mapper(coordinate_mapper):
        renderable = FunctionRenderable(func, coordinate_mapper)
        try:
            func._renderable = renderable
        except Exception:
            pass
    else:
        renderable.mapper = coordinate_mapper
    signature = _model_signature(func)
    if signature is None or signature != renderable._model_signature:
        renderable.invalidate_cache()
        renderable._model_signature = signature
    return renderable


def _build_stroke_style(func, style):
    """Build the stroke style for a function curve.

    Args:
        func: Function drawable with optional color attribute.
        style: Style dictionary with function_color and function_stroke_width.

    Returns:
        StrokeStyle with function color and round line joins.
    """
    return StrokeStyle(
        color=str(getattr(func, "color", style.get("function_color", "#000"))),
        width=float(style.get("function_stroke_width", 1) or 1),
        line_join="round",
    )


def _normalize_font_size(value):
    """Normalize font size to integer if it's a whole number.

    Args:
        value: Font size value to normalize.

    Returns:
        Integer if value is a whole number, otherwise the original value.
    """
    try:
        size_float = float(value)
    except Exception:
        return value
    if math.isfinite(size_float) and size_float.is_integer():
        return int(size_float)
    return size_float


def _segment_entry_point(start, end, width, height):
    """Point where the segment start->end enters the canvas rectangle, or None if it misses it.

    Liang-Barsky clipping against [0, width] x [0, height].
    """
    x0, y0 = start
    dx = end[0] - x0
    dy = end[1] - y0
    t_enter = 0.0
    t_exit = 1.0
    for p, q in ((-dx, x0), (dx, width - x0), (-dy, y0), (dy, height - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            if t > t_exit:
                return None
            if t > t_enter:
                t_enter = t
        else:
            if t < t_enter:
                return None
            if t < t_exit:
                t_exit = t
    if t_enter == 0.0:
        return (x0, y0)
    return (x0 + t_enter * dx, y0 + t_enter * dy)


def _first_visible_point(screen_paths, width, height):
    """Where the curve first enters the canvas, or its first point when it never does (or size is unknown).

    Sampling can leave few vertices on straight stretches, so the entry point is
    found by clipping each path segment rather than by the first vertex inside.
    """
    if width > 0 and height > 0:
        for path in screen_paths:
            if len(path) == 1:
                sx, sy = path[0]
                if 0 <= sx <= width and 0 <= sy <= height:
                    return (sx, sy)
                continue
            for index in range(1, len(path)):
                try:
                    entry = _segment_entry_point(path[index - 1], path[index], width, height)
                except (ArithmeticError, TypeError, ValueError):
                    entry = None
                if entry is not None and math.isfinite(entry[0]) and math.isfinite(entry[1]):
                    return entry
    return screen_paths[0][0]


def function_label_position(name, screen_paths, font_size, width=0, height=0):
    """Screen position of a function's name label for the given (non-empty) screen paths.

    The label sits left of the first visible curve point and, when the canvas
    width is known, is kept inside the canvas. Cached plans call this again
    after reprojecting their paths so the label stays anchored to the viewport.
    """
    first_point = _first_visible_point(screen_paths, width, height)
    label_offset_x = (1 + len(name)) * font_size / 2.0
    label_x = first_point[0] - label_offset_x
    if width > 0:
        # Curves usually start at the left edge, which pushed the label off-canvas.
        text_width = len(name) * font_size * 0.6
        label_x = max(4.0, min(label_x, width - text_width))
    return (label_x, max(first_point[1], font_size))


def _render_function_label(primitives, func, screen_paths, stroke, style, width=0, height=0):
    """Render the function name label near the curve start.

    Args:
        primitives: The renderer primitives interface.
        func: Function drawable with name attribute.
        screen_paths: Rendered paths for label positioning.
        stroke: StrokeStyle containing the function color.
        style: Style dictionary with font settings.
        width: Canvas width; when positive the label is kept inside the canvas.
        height: Canvas height; with width, used to anchor the label at the first
            visible point (paths may extend beyond the canvas).
    """
    if not getattr(func, "name", "") or not screen_paths or not screen_paths[0]:
        return
    font_size = _normalize_font_size(style.get("function_label_font_size", 12))
    position = function_label_position(func.name, screen_paths, font_size, width, height)
    font_family = style.get("function_label_font_family", style.get("font_family", default_font_family))
    font = FontStyle(family=font_family, size=font_size)
    primitives.draw_text(
        func.name,
        position,
        font,
        stroke.color,
        TextAlignment(horizontal="left", vertical="alphabetic"),
        metadata={"function_label": {"font_size": font_size, "canvas_width": width, "canvas_height": height}},
    )


def render_function_helper(primitives, func, coordinate_mapper, style):
    """Render a mathematical function drawable.

    Args:
        primitives: The renderer primitives interface.
        func: Function drawable with expression, domain, name, and color.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.
        style: Style dictionary with function_color and function_stroke_width.
    """
    view_margin = _get_view_margin(style)
    try:
        renderable = _get_or_create_renderable(func, coordinate_mapper)
        if renderable.view_margin != view_margin:
            renderable.view_margin = view_margin
            renderable.invalidate_cache()
        screen_paths = renderable.build_screen_paths().paths
    except Exception as e:
        # Log the error for debugging but don't crash rendering
        print(f"[render_function_helper] Error building paths for {getattr(func, 'name', 'unknown')}: {e}")
        return
    if not screen_paths:
        return

    width = getattr(coordinate_mapper, "canvas_width", 0) or 0
    height = getattr(coordinate_mapper, "canvas_height", 0) or 0
    stroke = _build_stroke_style(func, style)

    cull_margin = max(16, view_margin * max(width, height))
    _render_function_paths(primitives, screen_paths, stroke, width, height, cull_margin)
    _render_function_label(primitives, func, screen_paths, stroke, style, width, height)
