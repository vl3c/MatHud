"""Colored area rendering helper for drawing filled regions.

This module provides render helpers for colored area drawables, including
function-bounded areas, segment-bounded areas, and closed shape fills.

Key Features:
    - Forward/reverse path joining for filled regions
    - Single-loop optimization for symmetric boundaries
    - Math-to-screen coordinate transformation
    - Configurable fill color and opacity
    - Point filtering for valid coordinates
"""

from __future__ import annotations

import math

from rendering.helpers.area_builders import (
    build_closed_shape_colored_area,
    build_function_segment_colored_area,
    build_functions_colored_area,
    build_segments_colored_area,
)
from rendering.helpers.shape_decorator import _manages_shape
from rendering.primitives import FillStyle


def _filter_valid_points(points):
    """Filter out invalid points from a coordinate list.

    Args:
        points: List of (x, y) tuples, possibly with None values.

    Returns:
        List of valid (x, y) float tuples.
    """
    filtered = []
    for pt in points:
        if not pt:
            continue
        x, y = pt
        if x is None or y is None:
            continue
        filtered.append((float(x), float(y)))
    return filtered


def _points_close(p1: tuple[float, float], p2: tuple[float, float], tol: float = 1e-9) -> bool:
    """Check if two points are within tolerance distance.

    Args:
        p1: First point (x, y).
        p2: Second point (x, y).
        tol: Distance tolerance for equality.

    Returns:
        True if points are within tolerance on both axes.
    """
    return abs(p1[0] - p2[0]) <= tol and abs(p1[1] - p2[1]) <= tol


def _paths_form_single_loop(
    forward: list[tuple[float, float]],
    reverse: list[tuple[float, float]],
    tol: float = 1e-9,
) -> bool:
    """Check if forward and reverse paths form a single closed loop.

    This optimization detects when the area degenerates to a simple polygon
    (e.g., when upper and lower bounds are the same reversed).

    Args:
        forward: Forward boundary points.
        reverse: Reverse boundary points.
        tol: Point comparison tolerance.

    Returns:
        True if paths are equivalent when one is reversed.
    """
    if len(forward) < 3:
        return False
    if len(forward) != len(reverse):
        return False
    reversed_reverse = list(reversed(reverse))
    return all(_points_close(f, r, tol) for f, r in zip(forward, reversed_reverse))


@_manages_shape
def _render_joined_area(primitives, forward, reverse, fill):
    """Render a joined area by connecting forward and reverse paths.

    Args:
        primitives: The renderer primitives interface.
        forward: Forward boundary screen points.
        reverse: Reverse boundary screen points.
        fill: FillStyle for the area.
    """
    primitives.fill_joined_area(forward, reverse, fill)


def render_colored_area_helper(primitives, closed_area, coordinate_mapper, style):
    """Render a colored area drawable with fill styling.

    Args:
        primitives: The renderer primitives interface.
        closed_area: ColoredArea with forward_points and reverse_points.
        coordinate_mapper: Mapper for math-to-screen conversion if needed.
        style: Style dictionary with area_fill_color and area_opacity.
    """
    if closed_area is None or not closed_area.forward_points or not closed_area.reverse_points:
        return
    forward = list(closed_area.forward_points)
    reverse = list(closed_area.reverse_points)
    if not getattr(closed_area, "is_screen", False):
        try:
            forward = [coordinate_mapper.math_to_screen(x, y) for (x, y) in forward]
            reverse = [coordinate_mapper.math_to_screen(x, y) for (x, y) in reverse]
        except Exception:
            return

    forward = _filter_valid_points(forward)
    reverse = _filter_valid_points(reverse)
    if len(forward) < 2 or len(reverse) < 1:
        return
    raw_color = getattr(closed_area, "color", None)
    if not raw_color:
        raw_color = style.get("area_fill_color", "lightblue")
    raw_opacity = getattr(closed_area, "opacity", None)
    if raw_opacity is None:
        raw_opacity = style.get("area_opacity", 0.3)
    try:
        opacity = float(raw_opacity)
    except Exception:
        opacity = 0.3
    if not math.isfinite(opacity):
        opacity = 0.3
    else:
        opacity = max(0.0, min(opacity, 1.0))
    fill = FillStyle(
        color=str(raw_color),
        opacity=opacity,
    )

    if _paths_form_single_loop(forward, reverse):
        loop_points = list(forward)
        if not _points_close(loop_points[0], loop_points[-1]):
            loop_points.append(loop_points[0])
        primitives.fill_polygon(loop_points, fill)
        return

    _render_joined_area(primitives, forward, reverse, fill)


def render_functions_bounded_area_helper(primitives, area_model, coordinate_mapper, style):
    """Render a function-bounded colored area.

    Args:
        primitives: The renderer primitives interface.
        area_model: FunctionsBoundedColoredArea with upper/lower functions.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.
        style: Style dictionary with area styling settings.
    """
    area = build_functions_colored_area(area_model, coordinate_mapper)
    render_colored_area_helper(primitives, area, coordinate_mapper, style)


def render_function_segment_area_helper(primitives, area_model, coordinate_mapper, style, *, num_points=100):
    """Render a function-segment bounded colored area.

    Args:
        primitives: The renderer primitives interface.
        area_model: FunctionSegmentBoundedColoredArea with function and segment.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.
        style: Style dictionary with area styling settings.
        num_points: Number of sample points along function curve.
    """
    area = build_function_segment_colored_area(area_model, coordinate_mapper, num_points=num_points)
    render_colored_area_helper(primitives, area, coordinate_mapper, style)


def render_segments_bounded_area_helper(primitives, area_model, coordinate_mapper, style):
    """Render a segment-bounded colored area.

    Args:
        primitives: The renderer primitives interface.
        area_model: SegmentsBoundedColoredArea with boundary segments.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.
        style: Style dictionary with area styling settings.
    """
    area = build_segments_colored_area(area_model, coordinate_mapper)
    render_colored_area_helper(primitives, area, coordinate_mapper, style)


def render_closed_shape_area_helper(primitives, area_model, coordinate_mapper, style):
    """Render a closed shape colored area.

    Args:
        primitives: The renderer primitives interface.
        area_model: ClosedShapeColoredArea with shape reference.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.
        style: Style dictionary with area styling settings.
    """
    area = build_closed_shape_colored_area(area_model, coordinate_mapper)
    if area is None:
        return
    render_colored_area_helper(primitives, area, coordinate_mapper, style)
