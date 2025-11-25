from __future__ import annotations

import math

from rendering.helpers.area_builders import (
    build_closed_shape_colored_area,
    build_function_segment_colored_area,
    build_functions_colored_area,
    build_segments_colored_area,
)
from rendering.helpers.shape_decorator import _manages_shape
from rendering.renderer_primitives import FillStyle


def _filter_valid_points(points):
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
    return abs(p1[0] - p2[0]) <= tol and abs(p1[1] - p2[1]) <= tol


def _paths_form_single_loop(
    forward: list[tuple[float, float]],
    reverse: list[tuple[float, float]],
    tol: float = 1e-9,
) -> bool:
    if len(forward) < 3:
        return False
    if len(forward) != len(reverse):
        return False
    reversed_reverse = list(reversed(reverse))
    return all(_points_close(f, r, tol) for f, r in zip(forward, reversed_reverse))


@_manages_shape
def _render_joined_area(primitives, forward, reverse, fill):
    primitives.fill_joined_area(forward, reverse, fill)


def render_colored_area_helper(primitives, closed_area, coordinate_mapper, style):
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
    area = build_functions_colored_area(area_model, coordinate_mapper)
    render_colored_area_helper(primitives, area, coordinate_mapper, style)


def render_function_segment_area_helper(primitives, area_model, coordinate_mapper, style, *, num_points=100):
    area = build_function_segment_colored_area(area_model, coordinate_mapper, num_points=num_points)
    render_colored_area_helper(primitives, area, coordinate_mapper, style)


def render_segments_bounded_area_helper(primitives, area_model, coordinate_mapper, style):
    area = build_segments_colored_area(area_model, coordinate_mapper)
    render_colored_area_helper(primitives, area, coordinate_mapper, style)


def render_closed_shape_area_helper(primitives, area_model, coordinate_mapper, style):
    area = build_closed_shape_colored_area(area_model, coordinate_mapper)
    if area is None:
        return
    render_colored_area_helper(primitives, area, coordinate_mapper, style)

