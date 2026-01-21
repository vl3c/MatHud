"""Point rendering helper for drawing point drawables.

This module provides the render_point_helper function that renders a point
as a filled circle with an optional label at its position.

Key Features:
    - Screen-space circle rendering at point location
    - Automatic label positioning with offset
    - Support for embedded Label drawables
    - Backward-compatible duck-typed point support
"""

from __future__ import annotations

from rendering.helpers.shape_decorator import _manages_shape
from rendering.helpers.label_renderer import render_label_helper
from rendering.helpers.screen_offset_label_helper import draw_point_style_label_with_coords
from rendering.primitives import FillStyle


@_manages_shape
def _render_point(primitives, sx, sy, radius, point, fill, style):
    """Render the point circle at screen coordinates."""
    primitives.fill_circle((sx, sy), radius, fill, screen_space=True)


def render_point_helper(primitives, point, coordinate_mapper, style):
    """Render a point drawable with its label.

    Args:
        primitives: The renderer primitives interface.
        point: The Point drawable with x, y, color, and optional label.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.
        style: Style dictionary with point_radius and point_color.
    """
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

    embedded_label = getattr(point, "label", None)
    if embedded_label is not None:
        render_label_helper(primitives, embedded_label, coordinate_mapper, style)
        return

    # Backward-compatible fallback: allow duck-typed points without an embedded Label.
    draw_point_style_label_with_coords(
        primitives,
        anchor_screen_x=float(sx),
        anchor_screen_y=float(sy),
        anchor_math_x=float(getattr(point, "x", 0.0)),
        anchor_math_y=float(getattr(point, "y", 0.0)),
        label=str(getattr(point, "name", "") or ""),
        radius=float(radius),
        color=str(getattr(fill, "color", "#000")),
        style=style,
    )

