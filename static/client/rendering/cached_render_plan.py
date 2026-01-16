from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from constants import label_min_screen_font_px, label_vanish_threshold_px
from rendering import shared_drawable_renderers as shared
from rendering.primitives import RendererPrimitives

_STYLE_CLASSES = (
    shared.StrokeStyle,
    shared.FillStyle,
    shared.FontStyle,
    shared.TextAlignment,
)

Number = float | int
MapState = Dict[str, float]


PrimitiveArgs = Tuple[Any, ...]
PrimitiveKwargs = Dict[str, Any]


def _style_signature(style: Any) -> Optional[Tuple[Any, ...]]:
    if style is None:
        return None
    attrs = []
    for attr in ("color", "width", "line_join", "line_cap", "opacity", "family", "size", "weight"):
        if hasattr(style, attr):
            attrs.append(getattr(style, attr))
    return tuple(attrs) if attrs else None


def _geometry_signature(values: Iterable[Any]) -> Tuple[Any, ...]:
    signature: List[Any] = []
    for value in values:
        if isinstance(value, (tuple, list)):
            signature.append(tuple(value))
        else:
            signature.append(value)
    return tuple(signature)


def _quantize_number(value: Any, *, decimals: int = 4) -> Any:
    if isinstance(value, float):
        if value == 0.0:
            return 0.0
        try:
            return round(value, decimals)
        except Exception:
            return value
    return value


def _quantize_geometry(value: Any, *, decimals: int = 4) -> Any:
    if isinstance(value, (tuple, list)):
        return tuple(_quantize_geometry(v, decimals=decimals) for v in value)
    return _quantize_number(value, decimals=decimals)


def _capture_map_state(mapper: Any) -> MapState:
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
    if left is right:
        return True
    for key in ("scale", "offset_x", "offset_y", "origin_x", "origin_y"):
        lv = float(left.get(key, 0.0))
        rv = float(right.get(key, 0.0))
        if abs(lv - rv) > epsilon:
            return False
    return True


def _compute_transform_params(base: MapState, target: MapState) -> Tuple[float, float, float]:
    base_scale = float(base.get("scale", 1.0) or 1.0)
    target_scale = float(target.get("scale", 1.0) or 1.0)
    if base_scale == 0.0:
        base_scale = 1.0
    scale_ratio = target_scale / base_scale
    base_sum_x = float(base.get("origin_x", 0.0)) + float(base.get("offset_x", 0.0))
    target_sum_x = float(target.get("origin_x", 0.0)) + float(target.get("offset_x", 0.0))
    tx = target_sum_x - scale_ratio * base_sum_x
    base_sum_y = float(base.get("origin_y", 0.0)) + float(base.get("offset_y", 0.0))
    target_sum_y = float(target.get("origin_y", 0.0)) + float(target.get("offset_y", 0.0))
    ty = target_sum_y - scale_ratio * base_sum_y
    return scale_ratio, tx, ty


def _transform_bounds(
    bounds: Optional[Tuple[float, float, float, float]], scale: float, tx: float, ty: float
) -> Optional[Tuple[float, float, float, float]]:
    if not bounds:
        return None
    min_x, max_x, min_y, max_y = bounds
    corners = (
        (min_x, min_y),
        (min_x, max_y),
        (max_x, min_y),
        (max_x, max_y),
    )
    transformed = [(scale * x + tx, scale * y + ty) for x, y in corners]
    xs = [point[0] for point in transformed]
    ys = [point[1] for point in transformed]
    return (min(xs), max(xs), min(ys), max(ys))


def _math_to_screen_point(math_point: Tuple[float, float], state: MapState) -> Tuple[float, float]:
    mx, my = math_point
    sx = state["origin_x"] + mx * state["scale"] + state["offset_x"]
    sy = state["origin_y"] - my * state["scale"] + state["offset_y"]
    return (sx, sy)


def _screen_to_math_point(screen_point: Tuple[float, float], state: MapState) -> Tuple[float, float]:
    sx, sy = screen_point
    scale = state["scale"] if state["scale"] else 1.0
    mx = (sx - state["offset_x"] - state["origin_x"]) / scale
    my = (state["origin_y"] + state["offset_y"] - sy) / scale
    return (mx, my)


def _reproject_points(points: Iterable[Tuple[float, float]], old: MapState, new: MapState) -> Tuple[Tuple[float, float], ...]:
    return tuple(_math_to_screen_point(_screen_to_math_point(point, old), new) for point in points)


def _reproject_radius(radius: float, old: MapState, new: MapState) -> float:
    scale_old = old["scale"] if old["scale"] else 1.0
    math_radius = radius / scale_old
    return math_radius * new["scale"]


def _get_safe_scale(state: MapState, key: str = "scale") -> float:
    value = float(state.get(key, 1.0) or 1.0)
    return 1.0 if value <= 0 else value


def _reproject_stroke_line(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState
) -> None:
    start, end, stroke = command.args
    new_start = _math_to_screen_point(_screen_to_math_point(start, old_state), new_state)
    new_end = _math_to_screen_point(_screen_to_math_point(end, old_state), new_state)
    command.args = (new_start, new_end, stroke)
    command.meta["geometry"] = _quantize_geometry((new_start, new_end))


def _reproject_stroke_polyline(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState
) -> None:
    points, stroke = command.args
    new_points = _reproject_points(points, old_state, new_state)
    command.args = (new_points, stroke)
    command.meta["geometry"] = _quantize_geometry(new_points)


def _reproject_stroke_circle(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState
) -> None:
    center, radius, stroke = command.args
    new_center = _math_to_screen_point(_screen_to_math_point(center, old_state), new_state)
    screen_space = bool(command.kwargs.get("screen_space"))
    new_radius = float(radius) if screen_space else _reproject_radius(float(radius), old_state, new_state)
    command.args = (new_center, new_radius, stroke)
    command.meta["geometry"] = _quantize_geometry((new_center, new_radius))


def _reproject_fill_circle(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState
) -> None:
    center, radius, fill, stroke = command.args
    new_center = _math_to_screen_point(_screen_to_math_point(center, old_state), new_state)
    screen_space = bool(command.kwargs.get("screen_space"))
    new_radius = float(radius) if screen_space else _reproject_radius(float(radius), old_state, new_state)
    command.args = (new_center, new_radius, fill, stroke)
    command.meta["geometry"] = _quantize_geometry((new_center, new_radius))


def _reproject_stroke_ellipse(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState
) -> None:
    center, radius_x, radius_y, rotation, stroke = command.args
    new_center = _math_to_screen_point(_screen_to_math_point(center, old_state), new_state)
    new_rx = _reproject_radius(float(radius_x), old_state, new_state)
    new_ry = _reproject_radius(float(radius_y), old_state, new_state)
    command.args = (new_center, new_rx, new_ry, rotation, stroke)
    command.meta["geometry"] = _quantize_geometry((new_center, new_rx, new_ry, rotation))


def _reproject_fill_joined_area(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState
) -> None:
    forward, reverse, fill = command.args
    new_forward = _reproject_points(forward, old_state, new_state)
    new_reverse = _reproject_points(reverse, old_state, new_state)
    command.args = (new_forward, new_reverse, fill)
    command.meta["geometry"] = _quantize_geometry(new_forward + new_reverse)


def _compute_vector_arrow_points(
    vector_meta: Dict[str, Any], new_state: MapState
) -> Tuple[Tuple[float, float], ...]:
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
    command: PrimitiveCommand, old_state: MapState, new_state: MapState
) -> None:
    points, fill, stroke = command.args
    metadata = command.kwargs.get("metadata") or {}
    vector_meta = metadata.get("vector_arrow") if isinstance(metadata, dict) else None
    if vector_meta:
        new_points = _compute_vector_arrow_points(vector_meta, new_state)
    else:
        new_points = _reproject_points(points, old_state, new_state)
    command.args = (tuple(new_points), fill, stroke)
    command.meta["geometry"] = _quantize_geometry(new_points)


def _compute_angle_arc_radius(
    angle_meta: Dict[str, Any],
    radius: float,
    old_state: MapState,
    new_state: MapState,
) -> float:
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
    command.meta["geometry"] = _quantize_geometry((vertex_screen, radius_screen))


def _parse_point_tuple(raw: Any, default: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
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
        command.meta["geometry"] = _quantize_geometry((new_center, radius_float))
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
    command.meta["geometry"] = _quantize_geometry(
        (new_center, new_radius, start_angle_screen, end_angle_screen, stored_cw)
    )


def _reproject_stroke_arc(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState
) -> None:
    center, radius, start_angle, end_angle, sweep_clockwise, stroke = command.args
    metadata = command.kwargs.get("metadata") or {}
    angle_meta = metadata.get("angle") if isinstance(metadata, dict) else None
    circle_meta = metadata.get("circle_arc") if isinstance(metadata, dict) else None
    if angle_meta:
        _reproject_arc_with_angle_meta(command, angle_meta, radius, stroke, old_state, new_state)
    elif circle_meta:
        _reproject_arc_with_circle_meta(command, circle_meta, radius, stroke, old_state, new_state)
    else:
        new_center = _math_to_screen_point(_screen_to_math_point(center, old_state), new_state)
        screen_space = bool(command.kwargs.get("screen_space"))
        new_radius = float(radius) if screen_space else _reproject_radius(float(radius), old_state, new_state)
        command.args = (new_center, new_radius, start_angle, end_angle, sweep_clockwise, stroke)
        command.meta["geometry"] = _quantize_geometry((new_center, new_radius))


def _reproject_text_with_angle_meta(
    angle_meta: Dict[str, Any],
    font: Any,
    old_state: MapState,
    new_state: MapState,
) -> Tuple[Tuple[float, float], Any]:
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
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    return 0.0, 0.0


def _reproject_text_with_point_meta(
    point_meta: Dict[str, Any],
    new_state: MapState,
) -> Tuple[float, float]:
    math_position = tuple(point_meta.get("math_position", (0.0, 0.0)))
    offset_x, offset_y = _extract_screen_offset(point_meta.get("screen_offset", (0.0, 0.0)))
    base_screen = _math_to_screen_point(math_position, new_state)
    return (base_screen[0] + offset_x, base_screen[1] + offset_y)


def _compute_label_font_scale(
    label_meta: Dict[str, Any],
    font: Any,
    new_state: MapState,
) -> Tuple[float, float]:
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


def _reproject_draw_text(
    command: PrimitiveCommand, old_state: MapState, new_state: MapState
) -> None:
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
        new_position = _math_to_screen_point(_screen_to_math_point(position, old_state), new_state)
    command.args = (text, new_position, font, color, alignment)
    command.meta["geometry"] = _quantize_geometry((new_position,))


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


def _reproject_command(command: PrimitiveCommand, old_state: MapState, new_state: MapState) -> None:
    op = command.op
    if not op:
        return
    handler = _REPROJECT_HANDLERS.get(op)
    if handler:
        handler(command, old_state, new_state)


def _drawable_key(drawable: Any, fallback: str) -> str:
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
    """Lightweight wrapper that caches math_to_screen and scale_value lookups."""

    def __init__(self, mapper: Any) -> None:
        self._mapper = mapper
        self._point_cache: Dict[Tuple[float, float], Tuple[float, float]] = {}
        self._scale_cache: Dict[Any, Any] = {}

    def math_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        key = (float(x), float(y))
        if key not in self._point_cache:
            self._point_cache[key] = self._mapper.math_to_screen(x, y)
        return self._point_cache[key]

    def scale_value(self, value: Any) -> Any:
        try:
            key = float(value)
        except Exception:
            key = value
        if key not in self._scale_cache:
            self._scale_cache[key] = self._mapper.scale_value(value)
        return self._scale_cache[key]

    def __getattr__(self, item: str) -> Any:
        return getattr(self._mapper, item)


class PrimitiveCommand:
    __slots__ = ("op", "args", "kwargs", "key", "meta")

    def __init__(
        self,
        op: str,
        args: PrimitiveArgs,
        kwargs: PrimitiveKwargs,
        key: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.op = op
        self.args = args
        self.kwargs = kwargs
        self.key = key
        self.meta = meta or {}


class OptimizedPrimitivePlan:
    __slots__ = (
        "drawable",
        "commands",
        "plan_key",
        "metadata",
        "_map_state",
        "_screen_bounds",
        "_needs_apply",
        "_usage_counts",
        "_supports_transform",
        "_base_map_state",
        "_display_map_state",
        "_current_transform",
        "_base_screen_bounds",
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
        self.drawable = drawable
        self.commands = commands
        self.plan_key = plan_key
        self.metadata = metadata
        self._map_state: Optional[MapState] = dict(metadata.get("map_state", {}))
        self._needs_apply: bool = True
        self._usage_counts: Dict[str, int] = dict(usage_counts or {})
        self._supports_transform: bool = bool(metadata.get("supports_transform"))
        self._base_map_state: MapState = dict(self._map_state or {})
        self._display_map_state: MapState = dict(self._map_state or {})
        self._current_transform: Optional[str] = None
        stored_bounds = metadata.get("screen_bounds")
        self._screen_bounds: Optional[Tuple[float, float, float, float]] = (
            tuple(stored_bounds) if isinstance(stored_bounds, (list, tuple)) and len(stored_bounds) == 4 else None
        )
        self._base_screen_bounds: Optional[Tuple[float, float, float, float]] = (
            tuple(self._screen_bounds) if self._screen_bounds is not None else None
        )
        self._uses_screen_space: bool = bool(metadata.get("uses_screen_space"))
        if not self._screen_bounds:
            self._recompute_bounds_from_commands()
        else:
            self._screen_bounds = tuple(float(v) for v in self._screen_bounds)
        if self._supports_transform:
            self._current_transform = "matrix(1 0 0 1 0 0)"
            self.metadata["transform"] = self._current_transform

    def update_map_state(self, new_state: MapState) -> None:
        if not new_state:
            return
        if self._supports_transform:
            self._display_map_state = dict(new_state)
            self.metadata["display_map_state"] = dict(new_state)
            scale_ratio, tx, ty = _compute_transform_params(self._base_map_state, new_state)
            self._current_transform = f"matrix({scale_ratio} 0 0 {scale_ratio} {tx} {ty})"
            self.metadata["transform"] = self._current_transform
            transformed_bounds = _transform_bounds(self._base_screen_bounds, scale_ratio, tx, ty)
            if transformed_bounds is not None:
                self._screen_bounds = transformed_bounds
                self.metadata["screen_bounds"] = transformed_bounds
            return
        current_state = self._map_state or {}
        if _map_state_equal(current_state, new_state):
            self._map_state = dict(new_state)
            self.metadata["map_state"] = dict(new_state)
            return
        if not current_state:
            self._map_state = dict(new_state)
            self.metadata["map_state"] = dict(new_state)
            return
        for command in self.commands:
            _reproject_command(command, current_state, new_state)
        self._map_state = dict(new_state)
        self.metadata["map_state"] = dict(new_state)
        self._recompute_bounds_from_commands()
        self._needs_apply = True

    def apply(self, primitives: RendererPrimitives) -> None:
        if not self.commands:
            self._needs_apply = False
            return
        primitives.begin_batch(self)
        try:
            for command in self.commands:
                primitives.execute_optimized(command)
        finally:
            primitives.end_batch(self)
        self._needs_apply = False

    def _recompute_bounds_from_commands(self) -> None:
        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")

        def consider_point(px: float, py: float) -> None:
            nonlocal min_x, max_x, min_y, max_y
            if px < min_x:
                min_x = px
            if px > max_x:
                max_x = px
            if py < min_y:
                min_y = py
            if py > max_y:
                max_y = py

        for command in self.commands:
            op = command.op
            if op == "stroke_line":
                start, end, _ = command.args[:3]
                consider_point(start[0], start[1])
                consider_point(end[0], end[1])
            elif op == "stroke_polyline":
                points, _ = command.args[:2]
                for x, y in points:
                    consider_point(x, y)
            elif op in {"fill_polygon", "fill_joined_area"}:
                if op == "fill_polygon":
                    points = command.args[0]
                else:
                    forward, reverse, _ = command.args[:3]
                    points = list(forward) + list(reverse)
                for x, y in points:
                    consider_point(x, y)
            elif op in {"stroke_circle", "fill_circle"}:
                center = command.args[0]
                radius = float(command.args[1])
                cx, cy = center
                consider_point(cx - radius, cy - radius)
                consider_point(cx + radius, cy + radius)
            elif op == "stroke_ellipse":
                center, rx, ry, rotation, _ = command.args[:5]
                cx, cy = center
                cos_r = math.cos(rotation)
                sin_r = math.sin(rotation)
                width = abs(rx * cos_r) + abs(ry * sin_r)
                height = abs(rx * sin_r) + abs(ry * cos_r)
                consider_point(cx - width, cy - height)
                consider_point(cx + width, cy + height)
            elif op == "stroke_arc":
                center, radius, start_angle, end_angle, _, _ = command.args[:6]
                cx, cy = center
                consider_point(cx - radius, cy - radius)
                consider_point(cx + radius, cy + radius)
                consider_point(cx + radius * math.cos(start_angle), cy + radius * math.sin(start_angle))
                consider_point(cx + radius * math.cos(end_angle), cy + radius * math.sin(end_angle))
            elif op == "draw_text":
                _, position, *_ = command.args
                consider_point(position[0], position[1])

        if min_x == float("inf") or min_y == float("inf"):
            self._screen_bounds = None
        else:
            self._screen_bounds = (min_x, max_x, min_y, max_y)
        if self._screen_bounds is not None:
            self.metadata["screen_bounds"] = self._screen_bounds
        if self._supports_transform and self._base_screen_bounds is None:
            self._base_screen_bounds = self._screen_bounds

    def is_visible(self, width: float, height: float, *, margin: float = 1.0) -> bool:
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
        return self._needs_apply

    def mark_dirty(self) -> None:
        self._needs_apply = True

    def get_usage_counts(self) -> Dict[str, int]:
        return dict(self._usage_counts)

    def supports_transform(self) -> bool:
        return self._supports_transform

    def get_transform(self) -> Optional[str]:
        return self._current_transform

    def uses_screen_space(self) -> bool:
        return self._uses_screen_space


class _RecordingPrimitives(shared.RendererPrimitives):
    def __init__(self, drawable_key: str) -> None:
        self.commands: List[PrimitiveCommand] = []
        self._drawable_key = drawable_key
        self._counter = 0
        self._style_pool: Dict[Tuple[Any, ...], Any] = {}
        self._bounds = [float("inf"), float("-inf"), float("inf"), float("-inf")]
        self._usage_counts: Dict[str, int] = {}
        self._screen_space_used: bool = False

    def _style_signature(self, style: Any) -> Optional[Tuple[Any, ...]]:
        if style is None:
            return None
        if not isinstance(style, _STYLE_CLASSES):
            return None
        signature: List[Any] = [style.__class__.__name__]
        for attr in getattr(style, "__slots__", ()):
            signature.append(getattr(style, attr, None))
        return tuple(signature)

    def _pool_style(self, style: Any) -> Any:
        signature = self._style_signature(style)
        if signature is None:
            return style
        pooled = self._style_pool.get(signature)
        if pooled is None:
            self._style_pool[signature] = style
            return style
        return pooled

    def _pool_styles(self, value: Any) -> Any:
        if isinstance(value, _STYLE_CLASSES):
            return self._pool_style(value)
        if isinstance(value, tuple):
            return tuple(self._pool_styles(item) for item in value)
        if isinstance(value, list):
            return [self._pool_styles(item) for item in value]
        if isinstance(value, dict):
            return {key: self._pool_styles(item) for key, item in value.items()}
        return value

    def _record(self, op: str, args: PrimitiveArgs, kwargs: PrimitiveKwargs, *, style: Any = None, geometry: Iterable[Any] = ()) -> None:
        command_key = f"{self._drawable_key}:{op}:{self._counter}"
        self._counter += 1
        meta: Dict[str, Any] = {}
        style_sig = _style_signature(style)
        if style_sig is not None:
            meta["style"] = style_sig
        geometry_sig = _geometry_signature(geometry)
        if geometry_sig:
            quantized = _quantize_geometry(geometry_sig)
            meta["geometry"] = quantized
            self._update_bounds_from_geometry(quantized)
        if kwargs.get("screen_space"):
            self._screen_space_used = True
        pooled_args = self._pool_styles(args)
        pooled_kwargs = self._pool_styles(kwargs)
        self.commands.append(PrimitiveCommand(op, pooled_args, pooled_kwargs, command_key, meta))
        self._usage_counts[op] = self._usage_counts.get(op, 0) + 1

    def _update_bounds_from_geometry(self, geometry: Tuple[Any, ...]) -> None:
        if not geometry:
            return
        min_x, max_x, min_y, max_y = self._bounds
        for item in geometry:
            if isinstance(item, (tuple, list)):
                if len(item) == 2 and all(isinstance(coord, (int, float)) for coord in item):
                    x, y = float(item[0]), float(item[1])
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
                else:
                    self._update_bounds_from_geometry(tuple(item))
        self._bounds = [min_x, max_x, min_y, max_y]

    def get_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        min_x, max_x, min_y, max_y = self._bounds
        if min_x == float("inf") or min_y == float("inf"):
            return None
        return (min_x, max_x, min_y, max_y)

    def get_usage_counts(self) -> Dict[str, int]:
        return dict(self._usage_counts)

    def uses_screen_space(self) -> bool:
        return self._screen_space_used

    def stroke_line(self, start, end, stroke, *, include_width=True):
        self._record("stroke_line", (start, end, stroke), {"include_width": include_width}, style=stroke, geometry=(start, end))

    def stroke_polyline(self, points, stroke):
        self._record("stroke_polyline", (tuple(points), stroke), {}, style=stroke, geometry=points)

    def stroke_circle(self, center, radius, stroke):
        self._record("stroke_circle", (center, radius, stroke), {}, style=stroke, geometry=(center, radius))

    def fill_circle(self, center, radius, fill, stroke=None, *, screen_space=False):
        self._record(
            "fill_circle",
            (center, radius, fill, stroke),
            {"screen_space": screen_space},
            style=fill,
            geometry=(center, radius),
        )

    def stroke_ellipse(self, center, radius_x, radius_y, rotation_rad, stroke):
        self._record("stroke_ellipse", (center, radius_x, radius_y, rotation_rad, stroke), {}, style=stroke, geometry=(center, radius_x, radius_y, rotation_rad))

    def fill_polygon(self, points, fill, stroke=None, *, screen_space=False, metadata=None):
        self._record(
            "fill_polygon",
            (tuple(points), fill, stroke),
            {"screen_space": screen_space, "metadata": metadata or {}},
            style=fill,
            geometry=points,
        )

    def fill_joined_area(self, forward, reverse, fill):
        self._record("fill_joined_area", (tuple(forward), tuple(reverse), fill), {}, style=fill, geometry=list(forward) + list(reverse))

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
            style=stroke,
            geometry=(center, radius, start_angle_rad, end_angle_rad, sweep_clockwise),
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
            style=font,
            geometry=(position,),
        )

    def clear_surface(self):
        return None

    def resize_surface(self, width, height):
        return None

    def begin_shape(self):
        self._record("begin_shape", (), {})

    def end_shape(self):
        self._record("end_shape", (), {})


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
    "Bar": shared.render_bar_helper,
    "FunctionsBoundedColoredArea": shared.render_functions_bounded_area_helper,
    "FunctionSegmentBoundedColoredArea": shared.render_function_segment_area_helper,
    "SegmentsBoundedColoredArea": shared.render_segments_bounded_area_helper,
    "ClosedShapeColoredArea": shared.render_closed_shape_area_helper,
    "Label": shared.render_label_helper,
}


def build_plan_for_drawable(
    drawable: Any,
    coordinate_mapper: Any,
    style: Dict[str, Any],
    *,
    supports_transform: bool = True,
) -> Optional[OptimizedPrimitivePlan]:
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
    cached_mapper = _CachedCoordinateMapper(coordinate_mapper)
    helper(recorder, drawable, cached_mapper, style)
    map_state = _capture_map_state(coordinate_mapper)
    effective_supports_transform = supports_transform and not recorder.uses_screen_space()
    plan = OptimizedPrimitivePlan(
        drawable=drawable,
        commands=list(recorder.commands),
        plan_key=drawable_key,
        metadata={
            "class_name": class_name,
            "map_state": map_state,
            "screen_bounds": recorder.get_bounds(),
            "supports_transform": effective_supports_transform,
            "uses_screen_space": recorder.uses_screen_space(),
            "display_map_state": map_state,
        },
        usage_counts=recorder.get_usage_counts(),
    )
    plan.update_map_state(map_state)
    return plan


def build_plan_for_cartesian(
    cartesian: Any,
    coordinate_mapper: Any,
    style: Dict[str, Any],
    *,
    supports_transform: bool = True,
) -> OptimizedPrimitivePlan:
    key = _drawable_key(cartesian, "cartesian")
    recorder = _RecordingPrimitives(key)
    cached_mapper = _CachedCoordinateMapper(coordinate_mapper)
    shared.render_cartesian_helper(recorder, cartesian, cached_mapper, style)
    map_state = _capture_map_state(coordinate_mapper)
    effective_supports_transform = supports_transform and not recorder.uses_screen_space()
    plan = OptimizedPrimitivePlan(
        drawable=cartesian,
        commands=list(recorder.commands),
        plan_key=key,
        metadata={
            "class_name": "Cartesian2Axis",
            "map_state": map_state,
            "screen_bounds": recorder.get_bounds(),
            "supports_transform": effective_supports_transform,
            "uses_screen_space": recorder.uses_screen_space(),
            "display_map_state": map_state,
        },
        usage_counts=recorder.get_usage_counts(),
    )
    plan.update_map_state(map_state)
    return plan


def build_plan_for_polar(
    polar_grid: Any,
    coordinate_mapper: Any,
    style: Dict[str, Any],
    *,
    supports_transform: bool = True,
) -> OptimizedPrimitivePlan:
    key = _drawable_key(polar_grid, "polar")
    recorder = _RecordingPrimitives(key)
    cached_mapper = _CachedCoordinateMapper(coordinate_mapper)
    shared.render_polar_helper(recorder, polar_grid, cached_mapper, style)
    map_state = _capture_map_state(coordinate_mapper)
    effective_supports_transform = supports_transform and not recorder.uses_screen_space()
    plan = OptimizedPrimitivePlan(
        drawable=polar_grid,
        commands=list(recorder.commands),
        plan_key=key,
        metadata={
            "class_name": "PolarGrid",
            "map_state": map_state,
            "screen_bounds": recorder.get_bounds(),
            "supports_transform": effective_supports_transform,
            "uses_screen_space": recorder.uses_screen_space(),
            "display_map_state": map_state,
        },
        usage_counts=recorder.get_usage_counts(),
    )
    plan.update_map_state(map_state)
    return plan

