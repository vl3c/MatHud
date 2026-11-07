"""
Parametric Function Rendering Helper for MatHud

Provides the rendering helper function for parametric curves. Handles renderable
creation/caching, path building, and polyline/label rendering using the shared
primitive interface.
"""

from __future__ import annotations

import math

from constants import default_font_family
from rendering.renderables import ParametricFunctionRenderable
from rendering.helpers.shape_decorator import _manages_shape
from rendering.primitives import FontStyle, StrokeStyle, TextAlignment
import compiled_expression


@_manages_shape
def _render_parametric_paths(primitives, screen_paths, stroke):
    """Render all path segments as polylines.

    Paths are not culled to the viewport: parametric samples do not depend on
    the view, so the Canvas2D plan is reprojected on pan instead of rebuilt and
    must keep the parts of the curve that are currently off-screen.
    """
    for path in screen_paths:
        if len(path) < 2:
            continue
        primitives.stroke_polyline(path, stroke)


def _parameter_signature(func):
    """The values of the parameters the curve reads (None when it reads none).

    Other model changes (a new expression or range) replace or invalidate the renderable
    where they happen; parameter values change without touching the curve.
    """
    if not compiled_expression.has_parameters():
        return None
    try:
        names = func.parameter_names()
    except Exception:
        return None
    if not names:
        return None
    return tuple((name, compiled_expression.parameter_value(name)) for name in names)


def _get_or_create_renderable(func, coordinate_mapper):
    """Get or create a ParametricFunctionRenderable for the given curve (parametric or polar).

    Cached paths are dropped when a parameter value the curve reads changes.
    """
    renderable = getattr(func, "_renderable", None)
    if renderable is None or renderable.mapper is not coordinate_mapper:
        renderable = ParametricFunctionRenderable(func, coordinate_mapper)
        try:
            func._renderable = renderable
        except Exception:
            pass
    else:
        renderable.mapper = coordinate_mapper
    signature = _parameter_signature(func)
    if signature != getattr(renderable, "_parameter_signature", None):
        renderable.invalidate_cache()
        renderable._parameter_signature = signature
    return renderable


def _build_stroke_style(func, style):
    """Build stroke style from function color and style settings."""
    return StrokeStyle(
        color=str(getattr(func, "color", style.get("function_color", "#000"))),
        width=float(style.get("function_stroke_width", 1) or 1),
        line_join="round",
    )


def _normalize_font_size(value):
    """Normalize font size to integer if possible."""
    try:
        size_float = float(value)
    except Exception:
        return value
    if math.isfinite(size_float) and size_float.is_integer():
        return int(size_float)
    return size_float


def _render_parametric_label(primitives, func, screen_paths, stroke, style):
    """Render the function label at the start of the first path segment."""
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


def render_parametric_function_helper(primitives, func, coordinate_mapper, style):
    """
    Main rendering helper for parametric functions.

    Creates or retrieves a cached renderable, builds screen paths, and renders
    the curve as polylines with an optional label.

    Args:
        primitives: RendererPrimitives interface for drawing
        func: ParametricFunction drawable to render
        coordinate_mapper: CoordinateMapper for math-to-screen conversion
        style: Style dictionary with rendering options
    """
    try:
        renderable = _get_or_create_renderable(func, coordinate_mapper)
        screen_paths = renderable.build_screen_paths().paths
    except Exception as e:
        print(f"[render_parametric_function_helper] Error building paths for {getattr(func, 'name', 'unknown')}: {e}")
        return

    if not screen_paths:
        return

    stroke = _build_stroke_style(func, style)

    _render_parametric_paths(primitives, screen_paths, stroke)
    _render_parametric_label(primitives, func, screen_paths, stroke, style)
