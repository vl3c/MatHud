"""Cached render plan system for efficient drawable rendering.

This module provides an optimized caching mechanism that records primitive drawing
commands and can replay them efficiently when the view changes. Instead of
recomputing all drawing operations on every frame, cached plans can be reprojected
to the new coordinate state using mathematical transformations.

Key Features:
    - Command recording: Drawing operations are captured as PrimitiveCommand objects
    - Plan caching: OptimizedPrimitivePlan stores recorded commands with metadata
    - Reprojection: When the view changes, commands are transformed to new coordinates
      with a single affine map (x' = k*x + tx, y' = k*y + ty)
    - Visibility culling: Plans track screen bounds to skip off-screen rendering

Recording is kept deliberately lean: under Brython every per-command or per-point
Python operation is expensive, so commands store only what is needed to replay
and reproject them.

Architecture:
    1. _RecordingPrimitives captures drawing calls during plan building
    2. PrimitiveCommand stores individual operations (op, args, kwargs)
    3. OptimizedPrimitivePlan manages command lists and handles reprojection
    4. Helper functions in _HELPERS map drawable types to rendering functions

Usage:
    plan = build_plan_for_drawable(drawable, mapper, style)
    plan.update_map_state(new_state)  # Reproject to new view
    plan.apply(primitives)  # Execute commands on renderer
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from constants import label_min_screen_font_px, label_vanish_threshold_px
from rendering import shared_drawable_renderers as shared
from rendering.primitives import RendererPrimitives

# Type aliases for coordinate and command data
Number = float | int
"""Numeric type accepting both integers and floats."""

MapState = Dict[str, float]
"""Coordinate mapper state with scale, offset_x/y, and origin_x/y keys."""

PrimitiveArgs = Tuple[Any, ...]
"""Positional arguments for a primitive drawing command."""

PrimitiveKwargs = Dict[str, Any]
"""Keyword arguments for a primitive drawing command."""


def _capture_map_state(mapper: Any) -> MapState:
    """Extract the current coordinate transformation state from a mapper.

    Captures scale factor, origin position, and offset values that define
    how math coordinates are transformed to screen coordinates.

    Args:
        mapper: A coordinate mapper object with scale_factor, origin, and offset.

    Returns:
        A MapState dictionary with scale, offset_x/y, and origin_x/y keys.
    """
    origin = getattr(mapper, "origin", None)
    offset = getattr(mapper, "offset", None)
    return {
        "scale": float(getattr(mapper, "scale_factor", 1.0)),
        "offset_x": float(getattr(offset, "x", 0.0)) if offset is not None else 0.0,
        "offset_y": float(getattr(offset, "y", 0.0)) if offset is not None else 0.0,
        "origin_x": float(getattr(origin, "x", 0.0)) if origin is not None else 0.0,
        "origin_y": float(getattr(origin, "y", 0.0)) if origin is not None else 0.0,
    }


def _map_state_equal(left: MapState, right: MapState, *, epsilon: float = 1e-6) -> bool:
    """Check if two map states are equivalent within a tolerance.

    Args:
        left: First map state to compare.
        right: Second map state to compare.
        epsilon: Maximum allowed difference for numeric values.

    Returns:
        True if all corresponding values differ by less than epsilon.
    """
    if left is right:
        return True
    for key in ("scale", "offset_x", "offset_y", "origin_x", "origin_y"):
        lv = float(left.get(key, 0.0))
        rv = float(right.get(key, 0.0))
        if abs(lv - rv) > epsilon:
            return False
    return True


def _math_to_screen_point(math_point: Tuple[float, float], state: MapState) -> Tuple[float, float]:
    """Convert a point from math coordinates to screen coordinates.

    Args:
        math_point: Point in mathematical space as (x, y).
        state: Coordinate mapper state with scale, origin, and offset.

    Returns:
        Point in screen pixel coordinates as (x, y).
    """
    mx, my = math_point
    sx = state["origin_x"] + mx * state["scale"] + state["offset_x"]
    sy = state["origin_y"] - my * state["scale"] + state["offset_y"]
    return (sx, sy)


AffineParams = Tuple[float, float, float]
"""Screen-to-screen reprojection as (k, tx, ty): x' = k*x + tx, y' = k*y + ty."""


def _affine_params(old: MapState, new: MapState) -> AffineParams:
    """Compute the screen-space affine map taking ``old`` screen coordinates to ``new``.

    Equivalent to converting a screen point to math space with ``old`` and back to
    screen space with ``new``. Both axes share the same scale ratio because the
    y-axis flip cancels out.

    Args:
        old: The map state the screen coordinates were calculated for.
        new: The target map state.

    Returns:
        (k, tx, ty) where k is the scale ratio and tx/ty the translations.
    """
    old_scale = old["scale"] if old["scale"] else 1.0
    k = new["scale"] / old_scale
    tx = (new["origin_x"] + new["offset_x"]) - k * (old["origin_x"] + old["offset_x"])
    ty = (new["origin_y"] + new["offset_y"]) - k * (old["origin_y"] + old["offset_y"])
    return (k, tx, ty)


def _affine_point(point: Sequence[float], xf: AffineParams) -> Tuple[float, float]:
    """Apply a screen-space affine map to a single point."""
    k, tx, ty = xf
    return (k * point[0] + tx, k * point[1] + ty)


def _affine_points(points: Sequence[Sequence[float]], xf: AffineParams) -> Tuple[Tuple[float, float], ...]:
    """Apply a screen-space affine map to a sequence of points."""
    k, tx, ty = xf
    if k == 1.0:
        return tuple([(p[0] + tx, p[1] + ty) for p in points])
    return tuple([(k * p[0] + tx, k * p[1] + ty) for p in points])


def _get_safe_scale(state: MapState, key: str = "scale") -> float:
    """Get a scale value from map state, ensuring it's positive.

    Args:
        state: The map state dictionary.
        key: The key to look up (defaults to "scale").

    Returns:
        The scale value, or 1.0 if zero, negative, or missing.
    """
    value = float(state.get(key, 1.0) or 1.0)
    return 1.0 if value <= 0 else value


def _reproject_stroke_line(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState, xf: AffineParams
) -> None:
    """Reproject a stroke_line command's start/end points in place."""
    start, end, stroke = command.args
    command.args = (_affine_point(start, xf), _affine_point(end, xf), stroke)


def _reproject_stroke_polyline(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState, xf: AffineParams
) -> None:
    """Reproject a stroke_polyline command's point list in place."""
    points, stroke = command.args
    command.args = (_affine_points(points, xf), stroke)


def _reproject_stroke_circle(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState, xf: AffineParams
) -> None:
    """Reproject a stroke_circle command (radius kept when screen_space is set)."""
    center, radius, stroke = command.args
    screen_space = bool(command.kwargs.get("screen_space"))
    new_radius = float(radius) if screen_space else float(radius) * xf[0]
    command.args = (_affine_point(center, xf), new_radius, stroke)


def _reproject_fill_circle(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState, xf: AffineParams
) -> None:
    """Reproject a fill_circle command (radius kept when screen_space is set)."""
    center, radius, fill, stroke = command.args
    screen_space = bool(command.kwargs.get("screen_space"))
    new_radius = float(radius) if screen_space else float(radius) * xf[0]
    command.args = (_affine_point(center, xf), new_radius, fill, stroke)


def _reproject_stroke_ellipse(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState, xf: AffineParams
) -> None:
    """Reproject a stroke_ellipse command's center and both radii in place."""
    center, radius_x, radius_y, rotation, stroke = command.args
    k = xf[0]
    command.args = (_affine_point(center, xf), float(radius_x) * k, float(radius_y) * k, rotation, stroke)


def _reproject_fill_joined_area(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState, xf: AffineParams
) -> None:
    """Reproject both forward and reverse point arrays used for shaded regions."""
    forward, reverse, fill = command.args
    command.args = (_affine_points(forward, xf), _affine_points(reverse, xf), fill)


def _compute_vector_arrow_points(vector_meta: Dict[str, Any], new_state: MapState) -> Tuple[Tuple[float, float], ...]:
    """Compute arrow head triangle points for a vector in the new map state.

    Vector arrows are rendered in screen space with a fixed tip size, so they
    must be recomputed from the stored math coordinates rather than transformed.

    Args:
        vector_meta: Metadata containing start_math, end_math, and tip_size.
        new_state: The target map state for screen coordinate calculation.

    Returns:
        Three screen-space points (tip, base1, base2) forming the arrow triangle.
    """
    start_math = tuple(vector_meta.get("start_math", (0.0, 0.0)))
    end_math = tuple(vector_meta.get("end_math", (0.0, 0.0)))
    tip_size = float(vector_meta.get("tip_size", 8.0))
    start_screen = _math_to_screen_point(start_math, new_state)
    end_screen = _math_to_screen_point(end_math, new_state)
    dx = end_screen[0] - start_screen[0]
    dy = end_screen[1] - start_screen[1]
    direction_length = math.hypot(dx, dy)
    if direction_length <= 1e-6:
        return (end_screen, end_screen, end_screen)
    angle = math.atan2(dy, dx)
    half_base = tip_size / 2.0
    height_sq = max(tip_size * tip_size - half_base * half_base, 0.0)
    height = min(math.sqrt(height_sq), direction_length)
    tip = end_screen
    base1 = (
        end_screen[0] - height * math.cos(angle) - half_base * math.sin(angle),
        end_screen[1] - height * math.sin(angle) + half_base * math.cos(angle),
    )
    base2 = (
        end_screen[0] - height * math.cos(angle) + half_base * math.sin(angle),
        end_screen[1] - height * math.sin(angle) - half_base * math.cos(angle),
    )
    return (tip, base1, base2)


def _reproject_fill_polygon(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState, xf: AffineParams
) -> None:
    """Reproject a fill_polygon command to a new map state.

    Handles both regular polygons and vector arrow heads specially, since
    arrow heads must be recomputed from math coordinates to maintain fixed
    screen-space tip size.
    """
    points, fill, stroke = command.args
    metadata = command.kwargs.get("metadata") or {}
    vector_meta = metadata.get("vector_arrow") if isinstance(metadata, dict) else None
    if vector_meta:
        new_points = tuple(_compute_vector_arrow_points(vector_meta, new_state))
    else:
        new_points = _affine_points(points, xf)
    command.args = (new_points, fill, stroke)


def _compute_angle_arc_radius(
    angle_meta: Dict[str, Any],
    radius: float,
    old_state: MapState,
    new_state: MapState,
) -> float:
    """Compute the arc radius for an angle drawable in the new map state.

    The radius is scaled based on zoom level but clamped to not exceed the
    minimum arm length to keep the arc inside the angle.

    Args:
        angle_meta: Metadata with arc_radius_on_screen and min_arm_length_in_math.
        radius: Original radius value.
        old_state: The map state when the angle was rendered.
        new_state: The target map state.

    Returns:
        The computed arc radius for the new map state.
    """
    base_radius_screen = float(
        angle_meta.get("clamped_arc_radius_on_screen", angle_meta.get("arc_radius_on_screen", radius))
    )
    min_arm_math = float(angle_meta.get("min_arm_length_in_math", base_radius_screen))
    base_scale = _get_safe_scale(old_state)
    new_scale = _get_safe_scale(new_state)
    radius_math = base_radius_screen / base_scale if base_scale > 0 else base_radius_screen
    new_radius = radius_math * new_scale
    max_radius = min_arm_math * new_scale
    return min(new_radius, max_radius) if max_radius > 0 else new_radius


def _reproject_arc_with_angle_meta(
    command: PrimitiveCommand,
    angle_meta: Dict[str, Any],
    radius: float,
    stroke: Any,
    old_state: MapState,
    new_state: MapState,
) -> None:
    """Reproject an angle's arc using stored math coordinates.

    Recalculates the arc from the vertex and arm positions in math space,
    applying proper scaling and sweep direction for the new map state.
    """
    vertex_math = tuple(angle_meta.get("vertex_math", (0.0, 0.0)))
    arm1_math = tuple(angle_meta.get("arm1_math", (0.0, 0.0)))
    display_degrees = float(angle_meta.get("display_degrees", 0.0))
    sweep_flag = str(angle_meta.get("final_sweep_flag", "0"))
    radius_screen = _compute_angle_arc_radius(angle_meta, radius, old_state, new_state)
    vertex_screen = _math_to_screen_point(vertex_math, new_state)
    arm1_screen = _math_to_screen_point(arm1_math, new_state)
    start_angle = math.atan2(arm1_screen[1] - vertex_screen[1], arm1_screen[0] - vertex_screen[0])
    sweep_cw = sweep_flag == "1"
    delta = math.radians(display_degrees)
    direction = 1 if sweep_cw else -1
    end_angle = start_angle + direction * delta
    command.args = (vertex_screen, radius_screen, start_angle, end_angle, sweep_cw, stroke)


def _parse_point_tuple(raw: Any, default: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    """Safely parse a point tuple from potentially malformed data.

    Args:
        raw: Data that should be a 2-element sequence of numbers.
        default: Value to return if parsing fails.

    Returns:
        A tuple of two floats, or the default if parsing failed.
    """
    try:
        return (float(raw[0]), float(raw[1]))
    except Exception:
        return default


def _compute_sweep_delta(
    center_math: Tuple[float, float],
    point1_math: Tuple[float, float],
    point2_math: Tuple[float, float],
    use_major: bool,
) -> float:
    """Compute the angular sweep between two points on a circle.

    Determines whether to use the minor or major arc based on the use_major
    flag, returning the appropriate angular distance in radians.

    Args:
        center_math: Center of the circle in math coordinates.
        point1_math: Start point on the circle.
        point2_math: End point on the circle.
        use_major: If True, return the major arc; otherwise minor arc.

    Returns:
        Angular distance in radians for the requested arc type.
    """
    start_angle_math = math.atan2(point1_math[1] - center_math[1], point1_math[0] - center_math[0])
    target_angle_math = math.atan2(point2_math[1] - center_math[1], point2_math[0] - center_math[0])
    full_turn = 2 * math.pi
    delta_ccw_math = (target_angle_math - start_angle_math) % full_turn
    delta_cw_math = (start_angle_math - target_angle_math) % full_turn
    minor_is_ccw = delta_ccw_math <= delta_cw_math
    minor_delta = delta_ccw_math if minor_is_ccw else delta_cw_math
    major_delta = delta_cw_math if minor_is_ccw else delta_ccw_math
    return major_delta if use_major else minor_delta


def _reproject_arc_with_circle_meta(
    command: PrimitiveCommand,
    circle_meta: Dict[str, Any],
    radius: float,
    stroke: Any,
    old_state: MapState,
    new_state: MapState,
) -> None:
    """Reproject a circle arc using stored math coordinates.

    Recalculates the arc from center and endpoint positions in math space,
    handling major/minor arc selection and proper radius scaling.
    """
    center_math = _parse_point_tuple(circle_meta.get("center_math", (0.0, 0.0)))
    point1_math = _parse_point_tuple(circle_meta.get("point1", (0.0, 0.0)))
    point2_math = _parse_point_tuple(circle_meta.get("point2", (0.0, 0.0)))
    try:
        radius_math = float(circle_meta.get("radius_math", radius))
    except Exception:
        radius_math = float(radius)
    use_major = bool(circle_meta.get("use_major_arc", False))
    stored_cw = bool(circle_meta.get("sweep_clockwise", False))
    new_center = _math_to_screen_point(center_math, new_state)
    point1_screen = _math_to_screen_point(point1_math, new_state)
    start_angle_screen = math.atan2(point1_screen[1] - new_center[1], point1_screen[0] - new_center[0])
    sweep_delta = _compute_sweep_delta(center_math, point1_math, point2_math, use_major)
    try:
        radius_float = float(radius)
    except Exception:
        radius_float = 0.0
    if sweep_delta <= 0.0:
        command.args = (new_center, radius_float, start_angle_screen, start_angle_screen, stored_cw, stroke)
        return
    base_scale_old = _get_safe_scale(old_state)
    base_scale_new = _get_safe_scale(new_state)
    if radius_math > 0.0 and base_scale_old > 0.0:
        style_scale = radius_float / (radius_math * base_scale_old)
        new_radius = radius_math * base_scale_new * style_scale
    elif base_scale_old > 0.0:
        new_radius = radius_float * (base_scale_new / base_scale_old)
    else:
        new_radius = radius_float
    end_angle_screen = start_angle_screen + sweep_delta if stored_cw else start_angle_screen - sweep_delta
    command.args = (new_center, new_radius, start_angle_screen, end_angle_screen, stored_cw, stroke)


def _reproject_stroke_arc(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState, xf: AffineParams
) -> None:
    """Reproject a stroke_arc command to a new map state.

    Handles three cases: angle arcs (with angle metadata), circle arcs
    (with circle_arc metadata), or plain arcs (simple reprojection).
    """
    center, radius, start_angle, end_angle, sweep_clockwise, stroke = command.args
    metadata = command.kwargs.get("metadata") or {}
    angle_meta = metadata.get("angle") if isinstance(metadata, dict) else None
    circle_meta = metadata.get("circle_arc") if isinstance(metadata, dict) else None
    if angle_meta:
        _reproject_arc_with_angle_meta(command, angle_meta, radius, stroke, old_state, new_state)
    elif circle_meta:
        _reproject_arc_with_circle_meta(command, circle_meta, radius, stroke, old_state, new_state)
    else:
        screen_space = bool(command.kwargs.get("screen_space"))
        new_radius = float(radius) if screen_space else float(radius) * xf[0]
        command.args = (_affine_point(center, xf), new_radius, start_angle, end_angle, sweep_clockwise, stroke)


def _reproject_text_with_angle_meta(
    angle_meta: Dict[str, Any],
    font: Any,
    old_state: MapState,
    new_state: MapState,
) -> Tuple[Tuple[float, float], Any]:
    """Compute position and font for angle label text in the new map state.

    Positions the label at a point along the angle bisector, scaling font
    size proportionally to the arc radius with minimum size constraints.

    Returns:
        Tuple of (new_position, new_font) for the text command.
    """
    vertex_math = tuple(angle_meta.get("vertex_math", (0.0, 0.0)))
    arm1_math = tuple(angle_meta.get("arm1_math", (0.0, 0.0)))
    base_radius_screen = float(
        angle_meta.get("clamped_arc_radius_on_screen", angle_meta.get("arc_radius_on_screen", 0.0))
    )
    style_radius_screen = float(angle_meta.get("arc_radius_on_screen", base_radius_screen))
    min_arm_math = float(angle_meta.get("min_arm_length_in_math", base_radius_screen))
    text_factor = float(angle_meta.get("text_radius_factor", 1.8))
    display_degrees = float(angle_meta.get("display_degrees", 0.0))
    sweep_flag = str(angle_meta.get("final_sweep_flag", "0"))
    base_scale = _get_safe_scale(old_state)
    new_scale = _get_safe_scale(new_state)
    radius_math = base_radius_screen / base_scale if base_scale > 0 else base_radius_screen
    new_radius = radius_math * new_scale
    max_radius = min_arm_math * new_scale
    radius_screen = min(new_radius, max_radius) if max_radius > 0 else new_radius
    vertex_screen = _math_to_screen_point(vertex_math, new_state)
    arm1_screen = _math_to_screen_point(arm1_math, new_state)
    angle_v_p1_rad = math.atan2(vertex_screen[1] - arm1_screen[1], arm1_screen[0] - vertex_screen[0])
    text_radius = radius_screen * text_factor
    text_delta = math.radians(display_degrees) / 2.0
    if sweep_flag == "0":
        text_delta = -text_delta
    text_angle = angle_v_p1_rad + text_delta
    tx = vertex_screen[0] + text_radius * math.cos(text_angle)
    ty = vertex_screen[1] + text_radius * math.sin(text_angle)
    new_position = (tx, ty)
    base_font_size = float(angle_meta.get("base_font_size", getattr(font, "size", 12.0)) or 12.0)
    if not math.isfinite(base_font_size) or base_font_size <= 0:
        base_font_size = 12.0
    min_font_size = float(angle_meta.get("min_font_size", label_min_screen_font_px) or label_min_screen_font_px)
    ratio_font = 1.0
    if style_radius_screen > 0:
        ratio_font = max(min(radius_screen / style_radius_screen, 1.0), 0.0)
    new_font_size = base_font_size * ratio_font
    if new_font_size < min_font_size:
        new_font_size = min_font_size
    new_font = shared.FontStyle(getattr(font, "family", None), new_font_size, getattr(font, "weight", None))
    return new_position, new_font


def _extract_screen_offset(raw: Any) -> Tuple[float, float]:
    """Extract a screen-space offset from raw data, defaulting to (0, 0)."""
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    return 0.0, 0.0


def _reproject_text_with_point_meta(
    point_meta: Dict[str, Any],
    new_state: MapState,
) -> Tuple[float, float]:
    """Compute new position for point label text.

    Converts the math position to screen coordinates and adds the fixed
    screen-space offset.
    """
    math_position = tuple(point_meta.get("math_position", (0.0, 0.0)))
    offset_x, offset_y = _extract_screen_offset(point_meta.get("screen_offset", (0.0, 0.0)))
    base_screen = _math_to_screen_point(math_position, new_state)
    return (base_screen[0] + offset_x, base_screen[1] + offset_y)


def _compute_label_font_scale(
    label_meta: Dict[str, Any],
    font: Any,
    new_state: MapState,
) -> Tuple[float, float]:
    """Compute base and scaled font sizes for a label based on zoom level.

    Labels scale down when zoomed out, with minimum size and vanish thresholds.

    Returns:
        Tuple of (base_font_size, new_font_size).
    """
    base_font_size = float(label_meta.get("base_font_size", getattr(font, "size", 14.0)) or 14.0)
    reference_scale = float(label_meta.get("reference_scale_factor", 1.0) or 1.0)
    min_font_size = float(label_meta.get("min_font_size", label_min_screen_font_px) or label_min_screen_font_px)
    vanish_threshold = float(
        label_meta.get("vanish_threshold_px", label_vanish_threshold_px) or label_vanish_threshold_px
    )
    current_scale = float(new_state.get("scale", 1.0) or 1.0)
    if not math.isfinite(reference_scale) or reference_scale <= 0:
        reference_scale = 1.0
    if not math.isfinite(current_scale) or current_scale <= 0:
        current_scale = 1.0
    ratio = current_scale / reference_scale if reference_scale else 1.0
    if not math.isfinite(ratio) or ratio <= 0:
        ratio = 1.0
    if ratio >= 1.0:
        new_font_size = base_font_size
    else:
        scaled = base_font_size * ratio
        if scaled <= vanish_threshold:
            new_font_size = 0.0
        else:
            new_font_size = max(scaled, min_font_size)
    return base_font_size, new_font_size


def _reproject_text_with_label_meta(
    label_meta: Dict[str, Any],
    font: Any,
    new_state: MapState,
) -> Tuple[Tuple[float, float], Any]:
    """Compute position and font for a standalone label in the new map state.

    Handles zoom-aware font scaling with offset adjustment proportional to
    font size changes.

    Returns:
        Tuple of (new_position, new_font) for the text command.
    """
    math_position = tuple(label_meta.get("math_position", (0.0, 0.0)))
    offset_x, offset_y = _extract_screen_offset(label_meta.get("screen_offset", (0.0, 0.0)))
    base_screen = _math_to_screen_point(math_position, new_state)
    base_font_size, new_font_size = _compute_label_font_scale(label_meta, font, new_state)
    if base_font_size > 0 and new_font_size > 0:
        offset_scale = new_font_size / base_font_size
    elif new_font_size <= 0:
        offset_scale = 0.0
    else:
        offset_scale = 1.0
    offset_x *= offset_scale
    offset_y *= offset_scale
    new_position = (base_screen[0] + offset_x, base_screen[1] + offset_y)
    existing_size = getattr(font, "size", None)
    try:
        existing_size_float = float(existing_size)
    except Exception:
        existing_size_float = None
    if existing_size_float is None or abs(existing_size_float - new_font_size) > 1e-6:
        font = shared.FontStyle(getattr(font, "family", None), new_font_size, getattr(font, "weight", None))
    return new_position, font


def _reproject_draw_text(command: PrimitiveCommand, old_state: MapState, new_state: MapState, xf: AffineParams) -> None:
    """Reproject a draw_text command to a new map state.

    Handles angle labels, point labels, standalone labels, and plain text
    using appropriate metadata when available.
    """
    text, position, font, color, alignment = command.args
    metadata = command.kwargs.get("metadata") or {}
    angle_meta = metadata.get("angle") if isinstance(metadata, dict) else None
    point_meta = metadata.get("point_label") if isinstance(metadata, dict) else None
    label_meta = metadata.get("label") if isinstance(metadata, dict) else None
    if angle_meta:
        new_position, font = _reproject_text_with_angle_meta(angle_meta, font, old_state, new_state)
    elif point_meta:
        new_position = _reproject_text_with_point_meta(point_meta, new_state)
    elif label_meta:
        new_position, font = _reproject_text_with_label_meta(label_meta, font, new_state)
    else:
        new_position = _affine_point(position, xf)
    command.args = (text, new_position, font, color, alignment)


# Mapping of primitive operation names to their reprojection handlers
_REPROJECT_HANDLERS: Dict[str, Any] = {
    "stroke_line": _reproject_stroke_line,
    "stroke_polyline": _reproject_stroke_polyline,
    "stroke_circle": _reproject_stroke_circle,
    "fill_circle": _reproject_fill_circle,
    "stroke_ellipse": _reproject_stroke_ellipse,
    "fill_joined_area": _reproject_fill_joined_area,
    "fill_polygon": _reproject_fill_polygon,
    "stroke_arc": _reproject_stroke_arc,
    "draw_text": _reproject_draw_text,
}


def _reproject_command(
    command: PrimitiveCommand,
    old_state: MapState,
    new_state: MapState,
    xf: Optional[AffineParams] = None,
) -> None:
    """Dispatch a command to the appropriate reprojection handler.

    Args:
        command: The primitive command to reproject in place.
        old_state: The map state the command was created for.
        new_state: The target map state to transform to.
        xf: Precomputed affine map from old to new screen coordinates.
    """
    op = command.op
    if not op:
        return
    handler = _REPROJECT_HANDLERS.get(op)
    if handler:
        if xf is None:
            xf = _affine_params(old_state, new_state)
        handler(command, old_state, new_state, xf)


def _drawable_key(drawable: Any, fallback: str) -> str:
    """Generate a unique cache key for a drawable.

    Prefers the drawable's name, then id, then falls back to a generated key
    using the Python object id.

    Args:
        drawable: The drawable object to generate a key for.
        fallback: Base string to use if drawable has no name or id.

    Returns:
        A string key uniquely identifying this drawable.
    """
    if drawable is None:
        return fallback
    name = getattr(drawable, "name", None)
    if isinstance(name, str) and name:
        return name
    identifier = getattr(drawable, "id", None)
    if isinstance(identifier, str) and identifier:
        return identifier
    return f"{fallback}:{id(drawable)}"


class _CachedCoordinateMapper:
    """Caching wrapper for coordinate mapper to avoid redundant calculations.

    During plan building, the same coordinate conversions may be requested
    multiple times. This wrapper memoizes math_to_screen and scale_value
    calls to improve performance.

    Attributes:
        _mapper: The underlying coordinate mapper.
        _point_cache: Cache of math-to-screen point conversions.
        _scale_cache: Cache of scaled values.
    """

    def __init__(self, mapper: Any) -> None:
        """Initialize the cached mapper wrapper.

        Args:
            mapper: The coordinate mapper to wrap.
        """
        self._mapper = mapper
        self._point_cache: Dict[Tuple[float, float], Tuple[float, float]] = {}
        self._scale_cache: Dict[Any, Any] = {}

    def math_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        """Convert math coordinates to screen coordinates with caching.

        Args:
            x: X coordinate in math space.
            y: Y coordinate in math space.

        Returns:
            Screen coordinates as (x, y) tuple.
        """
        key = (float(x), float(y))
        if key not in self._point_cache:
            self._point_cache[key] = self._mapper.math_to_screen(x, y)
        return self._point_cache[key]

    def scale_value(self, value: Any) -> Any:
        """Scale a value from math to screen space with caching.

        Args:
            value: The value to scale.

        Returns:
            The scaled value.
        """
        try:
            key = float(value)
        except Exception:
            key = value
        if key not in self._scale_cache:
            self._scale_cache[key] = self._mapper.scale_value(value)
        return self._scale_cache[key]

    def __getattr__(self, item: str) -> Any:
        """Delegate attribute access to the underlying mapper."""
        return getattr(self._mapper, item)


class PrimitiveCommand:
    """A recorded drawing primitive that can be replayed or reprojected.

    Commands capture a single drawing call with its arguments and keyword
    arguments. They are modified in place during reprojection to update
    coordinates.

    Attributes:
        op: The operation name (e.g., 'stroke_line', 'fill_circle').
        args: Positional arguments for the primitive call.
        kwargs: Keyword arguments for the primitive call.
        key: Optional identifier (not computed during recording).
        meta: Optional metadata (not computed during recording).
    """

    __slots__ = ("op", "args", "kwargs", "key", "meta")

    def __init__(
        self,
        op: str,
        args: PrimitiveArgs,
        kwargs: PrimitiveKwargs,
        key: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a primitive command.

        Args:
            op: The operation name.
            args: Positional arguments tuple.
            kwargs: Keyword arguments dictionary.
            key: Optional command key.
            meta: Optional metadata dictionary.
        """
        self.op = op
        self.args = args
        self.kwargs = kwargs
        self.key = key
        self.meta = meta


Bounds = Tuple[float, float, float, float]
"""Screen bounds as (min_x, max_x, min_y, max_y)."""


def _extend_bounds(points: Sequence[Sequence[float]], bounds: Bounds) -> Bounds:
    """Grow (min_x, max_x, min_y, max_y) to include every point."""
    min_x, max_x, min_y, max_y = bounds
    for p in points:
        x = p[0]
        y = p[1]
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
    return (min_x, max_x, min_y, max_y)


def _command_bounds_points(command: PrimitiveCommand) -> Sequence[Sequence[float]]:
    """Return the screen points that bound a single non-point-list command."""
    op = command.op
    args = command.args
    if op == "stroke_line":
        return (args[0], args[1])
    if op == "stroke_circle" or op == "fill_circle":
        cx, cy = args[0]
        radius = float(args[1])
        return ((cx - radius, cy - radius), (cx + radius, cy + radius))
    if op == "stroke_ellipse":
        center, rx, ry, rotation = args[:4]
        cx, cy = center
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        width = abs(rx * cos_r) + abs(ry * sin_r)
        height = abs(rx * sin_r) + abs(ry * cos_r)
        return ((cx - width, cy - height), (cx + width, cy + height))
    if op == "stroke_arc":
        center, radius, start_angle, end_angle = args[:4]
        cx, cy = center
        return (
            (cx - radius, cy - radius),
            (cx + radius, cy + radius),
            (cx + radius * math.cos(start_angle), cy + radius * math.sin(start_angle)),
            (cx + radius * math.cos(end_angle), cy + radius * math.sin(end_angle)),
        )
    if op == "draw_text":
        return (args[1],)
    return ()


class OptimizedPrimitivePlan:
    """Cached render plan for efficient drawable rendering with reprojection.

    A plan stores recorded primitive commands and updates them in place when
    the coordinate system changes (pan/zoom).

    Attributes:
        drawable: The drawable object this plan renders.
        commands: List of PrimitiveCommand objects to execute.
        plan_key: Unique identifier for this plan.
        metadata: Dictionary with class_name, map_state, screen_bounds, etc.
    """

    __slots__ = (
        "drawable",
        "commands",
        "plan_key",
        "metadata",
        "_map_state",
        "_screen_bounds",
        "_needs_apply",
        "_usage_counts",
        "_uses_screen_space",
    )

    def __init__(
        self,
        *,
        drawable: Any,
        commands: List[PrimitiveCommand],
        plan_key: str,
        metadata: Dict[str, Any],
        usage_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """Initialize an optimized primitive plan.

        Args:
            drawable: The drawable object being rendered.
            commands: Recorded primitive commands.
            plan_key: Unique plan identifier.
            metadata: Plan metadata including map_state and optional screen_bounds.
            usage_counts: Optional counts of each operation type.
        """
        self.drawable = drawable
        self.commands = commands
        self.plan_key = plan_key
        self.metadata = metadata
        self._map_state: Optional[MapState] = dict(metadata.get("map_state", {}))
        self._needs_apply: bool = True
        self._usage_counts: Dict[str, int] = dict(usage_counts or {})
        self._uses_screen_space: bool = bool(metadata.get("uses_screen_space"))
        stored_bounds = metadata.get("screen_bounds")
        self._screen_bounds: Optional[Bounds] = None
        if isinstance(stored_bounds, (list, tuple)) and len(stored_bounds) == 4:
            self._screen_bounds = (
                float(stored_bounds[0]),
                float(stored_bounds[1]),
                float(stored_bounds[2]),
                float(stored_bounds[3]),
            )
        else:
            self._recompute_bounds_from_commands()

    def update_map_state(self, new_state: MapState) -> None:
        """Reproject every command to a new coordinate mapper state.

        Args:
            new_state: The new map state to adapt to.
        """
        if not new_state:
            return
        current_state = self._map_state or {}
        state_copy = dict(new_state)
        if not current_state or _map_state_equal(current_state, new_state):
            self._map_state = state_copy
            self.metadata["map_state"] = state_copy
            return
        xf = _affine_params(current_state, new_state)
        handlers = _REPROJECT_HANDLERS
        for command in self.commands:
            handler = handlers.get(command.op)
            if handler is not None:
                handler(command, current_state, new_state, xf)
        self._map_state = state_copy
        self.metadata["map_state"] = state_copy
        self._recompute_bounds_from_commands()
        self._needs_apply = True

    def apply(self, primitives: RendererPrimitives) -> None:
        """Execute all commands in this plan on the given primitives interface.

        Args:
            primitives: The renderer primitives to draw on.
        """
        if not self.commands:
            self._needs_apply = False
            return
        primitives.begin_batch(self)
        try:
            execute = primitives.execute_optimized
            for command in self.commands:
                execute(command)
        finally:
            primitives.end_batch(self)
        self._needs_apply = False

    def _recompute_bounds_from_commands(self) -> None:
        """Recalculate screen bounds by scanning all commands."""
        inf = float("inf")
        bounds: Bounds = (inf, -inf, inf, -inf)
        for command in self.commands:
            op = command.op
            if op == "stroke_polyline" or op == "fill_polygon":
                bounds = _extend_bounds(command.args[0], bounds)
            elif op == "fill_joined_area":
                bounds = _extend_bounds(command.args[0], bounds)
                bounds = _extend_bounds(command.args[1], bounds)
            else:
                bounds = _extend_bounds(_command_bounds_points(command), bounds)
        if bounds[0] == inf or bounds[2] == inf:
            self._screen_bounds = None
        else:
            self._screen_bounds = bounds
            self.metadata["screen_bounds"] = bounds

    def is_visible(self, width: float, height: float, *, margin: float = 1.0) -> bool:
        """Check if this plan is visible within the given viewport.

        Args:
            width: Viewport width in pixels.
            height: Viewport height in pixels.
            margin: Extra margin for partially visible elements.

        Returns:
            True if the plan's bounds intersect the viewport.
        """
        if self._screen_bounds is None:
            return True
        min_x, max_x, min_y, max_y = self._screen_bounds
        if max_x < -margin:
            return False
        if max_y < -margin:
            return False
        if min_x > width + margin:
            return False
        if min_y > height + margin:
            return False
        return True

    def needs_apply(self) -> bool:
        """Check if this plan needs to be redrawn."""
        return self._needs_apply

    def mark_dirty(self) -> None:
        """Mark this plan as needing redraw."""
        self._needs_apply = True

    def get_usage_counts(self) -> Dict[str, int]:
        """Get counts of each operation type in this plan."""
        return dict(self._usage_counts)

    def supports_transform(self) -> bool:
        """Plans always reproject their commands; CSS-transform reprojection is not used."""
        return False

    def get_transform(self) -> Optional[str]:
        """Plans never carry a CSS transform; kept for renderer compatibility."""
        return None

    def uses_screen_space(self) -> bool:
        """Check if this plan uses screen-space coordinates."""
        return self._uses_screen_space


class _RecordingPrimitives(shared.RendererPrimitives):
    """Primitives implementation that records commands instead of drawing.

    Used during plan building to capture all drawing operations. Recording
    only stores the call itself; bounds are computed once by the plan.

    Attributes:
        commands: List of recorded PrimitiveCommand objects.
    """

    def __init__(self, drawable_key: str) -> None:
        """Initialize the recording primitives.

        Args:
            drawable_key: Key of the drawable being recorded.
        """
        self.commands: List[PrimitiveCommand] = []
        self._drawable_key = drawable_key
        self._usage_counts: Dict[str, int] = {}
        self._screen_space_used: bool = False

    def _record(self, op: str, args: PrimitiveArgs, kwargs: PrimitiveKwargs) -> None:
        """Record a primitive operation as a command."""
        if kwargs.get("screen_space"):
            self._screen_space_used = True
        self.commands.append(PrimitiveCommand(op, args, kwargs))
        counts = self._usage_counts
        counts[op] = counts.get(op, 0) + 1

    def get_usage_counts(self) -> Dict[str, int]:
        return dict(self._usage_counts)

    def uses_screen_space(self) -> bool:
        return self._screen_space_used

    def stroke_line(self, start, end, stroke, *, include_width=True):
        self._record("stroke_line", (start, end, stroke), {"include_width": include_width})

    def stroke_polyline(self, points, stroke):
        self._record("stroke_polyline", (tuple(points), stroke), {})

    def stroke_circle(self, center, radius, stroke):
        self._record("stroke_circle", (center, radius, stroke), {})

    def fill_circle(self, center, radius, fill, stroke=None, *, screen_space=False):
        self._record("fill_circle", (center, radius, fill, stroke), {"screen_space": screen_space})

    def stroke_ellipse(self, center, radius_x, radius_y, rotation_rad, stroke):
        self._record("stroke_ellipse", (center, radius_x, radius_y, rotation_rad, stroke), {})

    def fill_polygon(self, points, fill, stroke=None, *, screen_space=False, metadata=None):
        self._record(
            "fill_polygon",
            (tuple(points), fill, stroke),
            {"screen_space": screen_space, "metadata": metadata or {}},
        )

    def fill_joined_area(self, forward, reverse, fill):
        self._record("fill_joined_area", (tuple(forward), tuple(reverse), fill), {})

    def stroke_arc(
        self,
        center,
        radius,
        start_angle_rad,
        end_angle_rad,
        sweep_clockwise,
        stroke,
        css_class=None,
        *,
        screen_space=False,
        metadata=None,
    ):
        self._record(
            "stroke_arc",
            (center, radius, start_angle_rad, end_angle_rad, sweep_clockwise, stroke),
            {"css_class": css_class, "screen_space": screen_space, "metadata": metadata or {}},
        )

    def draw_text(
        self,
        text,
        position,
        font,
        color,
        alignment,
        style_overrides=None,
        *,
        screen_space=False,
        metadata=None,
    ):
        self._record(
            "draw_text",
            (text, position, font, color, alignment),
            {"style_overrides": style_overrides or {}, "screen_space": screen_space, "metadata": metadata or {}},
        )

    def clear_surface(self):
        return None

    def resize_surface(self, width, height):
        return None

    def begin_shape(self):
        self._record("begin_shape", (), {})

    def end_shape(self):
        self._record("end_shape", (), {})


# Registry mapping drawable class names to their rendering helper functions
_HELPERS: Dict[str, Any] = {
    "Point": shared.render_point_helper,
    "Segment": shared.render_segment_helper,
    "Circle": shared.render_circle_helper,
    "CircleArc": shared.render_circle_arc_helper,
    "Ellipse": shared.render_ellipse_helper,
    "Vector": shared.render_vector_helper,
    "Angle": shared.render_angle_helper,
    "Function": shared.render_function_helper,
    "PiecewiseFunction": shared.render_function_helper,
    "ParametricFunction": shared.render_parametric_function_helper,
    "Bar": shared.render_bar_helper,
    "FunctionsBoundedColoredArea": shared.render_functions_bounded_area_helper,
    "FunctionSegmentBoundedColoredArea": shared.render_function_segment_area_helper,
    "SegmentsBoundedColoredArea": shared.render_segments_bounded_area_helper,
    "ClosedShapeColoredArea": shared.render_closed_shape_area_helper,
    "Label": shared.render_label_helper,
}


def _finish_plan(
    drawable: Any,
    recorder: _RecordingPrimitives,
    coordinate_mapper: Any,
    plan_key: str,
    class_name: str,
) -> OptimizedPrimitivePlan:
    """Wrap recorded commands into a plan bound to the mapper's current state."""
    map_state = _capture_map_state(coordinate_mapper)
    return OptimizedPrimitivePlan(
        drawable=drawable,
        commands=recorder.commands,
        plan_key=plan_key,
        metadata={
            "class_name": class_name,
            "map_state": map_state,
            "supports_transform": False,
            "uses_screen_space": recorder.uses_screen_space(),
        },
        usage_counts=recorder.get_usage_counts(),
    )


def build_plan_for_drawable(
    drawable: Any,
    coordinate_mapper: Any,
    style: Dict[str, Any],
    *,
    supports_transform: bool = True,
) -> Optional[OptimizedPrimitivePlan]:
    """Build an optimized render plan for a drawable object.

    Records all primitive commands needed to render the drawable and returns
    a plan that can be efficiently reprojected when the view changes.

    Args:
        drawable: The drawable object to create a plan for.
        coordinate_mapper: Mapper for converting math to screen coordinates.
        style: Style dictionary with rendering options.
        supports_transform: Ignored; kept for call-site compatibility (plans always reproject).

    Returns:
        An OptimizedPrimitivePlan, or None if the drawable is not renderable
        or has no registered helper.
    """
    renderable_attr = getattr(drawable, "is_renderable", True)
    try:
        if not bool(renderable_attr):
            return None
    except Exception:
        return None
    class_name = getattr(drawable, "get_class_name", None)
    if callable(class_name):
        class_name = class_name()
    elif class_name is None:
        class_name = drawable.__class__.__name__

    helper = _HELPERS.get(class_name)
    if helper is None:
        return None

    drawable_key = _drawable_key(drawable, class_name.lower())
    recorder = _RecordingPrimitives(drawable_key)
    helper(recorder, drawable, _CachedCoordinateMapper(coordinate_mapper), style)
    return _finish_plan(drawable, recorder, coordinate_mapper, drawable_key, class_name)


def build_plan_for_cartesian(
    cartesian: Any,
    coordinate_mapper: Any,
    style: Dict[str, Any],
    *,
    supports_transform: bool = True,
) -> OptimizedPrimitivePlan:
    """Build an optimized render plan for a Cartesian coordinate system.

    Args:
        cartesian: The Cartesian2Axis grid object.
        coordinate_mapper: Mapper for coordinate conversions.
        style: Style dictionary with grid rendering options.
        supports_transform: Ignored; kept for call-site compatibility (plans always reproject).

    Returns:
        An OptimizedPrimitivePlan for rendering the grid.
    """
    key = _drawable_key(cartesian, "cartesian")
    recorder = _RecordingPrimitives(key)
    shared.render_cartesian_helper(recorder, cartesian, _CachedCoordinateMapper(coordinate_mapper), style)
    return _finish_plan(cartesian, recorder, coordinate_mapper, key, "Cartesian2Axis")


def build_plan_for_polar(
    polar_grid: Any,
    coordinate_mapper: Any,
    style: Dict[str, Any],
    *,
    supports_transform: bool = True,
) -> OptimizedPrimitivePlan:
    """Build an optimized render plan for a polar coordinate grid.

    Args:
        polar_grid: The PolarGrid object.
        coordinate_mapper: Mapper for coordinate conversions.
        style: Style dictionary with grid rendering options.
        supports_transform: Ignored; kept for call-site compatibility (plans always reproject).

    Returns:
        An OptimizedPrimitivePlan for rendering the polar grid.
    """
    key = _drawable_key(polar_grid, "polar")
    recorder = _RecordingPrimitives(key)
    shared.render_polar_helper(recorder, polar_grid, _CachedCoordinateMapper(coordinate_mapper), style)
    return _finish_plan(polar_grid, recorder, coordinate_mapper, key, "PolarGrid")
