from __future__ import annotations

import math
from typing import Any

from constants import default_font_family, label_min_screen_font_px, label_vanish_threshold_px
from rendering.closed_shape_area_renderable import ClosedShapeAreaRenderable
from rendering.function_renderable import FunctionRenderable
from rendering.function_segment_area_renderable import FunctionSegmentAreaRenderable
from rendering.functions_area_renderable import FunctionsBoundedAreaRenderable
from rendering.renderer_primitives import (
    FillStyle,
    FontStyle,
    RendererPrimitives,
    StrokeStyle,
    TextAlignment,
)
from rendering.segments_area_renderable import SegmentsBoundedAreaRenderable
from utils.math_utils import MathUtils

Point2D = tuple


# ----------------------------------------------------------------------------
# Font size helpers
# ----------------------------------------------------------------------------


def _coerce_font_size(candidate: Any, fallback: Any, default_value: float = 14.0) -> float:
    try:
        value = float(candidate)
    except Exception:
        value = None
    if value is not None and math.isfinite(value) and value > 0:
        return value
    if isinstance(fallback, (int, float)):
        fallback_value = float(fallback)
        if math.isfinite(fallback_value) and fallback_value > 0:
            return fallback_value
    return float(default_value)


def _compute_zoom_adjusted_font_size(base_size: float, label: Any, coordinate_mapper: Any) -> float:
    reference_scale = getattr(label, "reference_scale_factor", None)
    try:
        reference_scale_value = float(reference_scale)
    except Exception:
        reference_scale_value = 1.0
    if not math.isfinite(reference_scale_value) or reference_scale_value <= 0:
        reference_scale_value = 1.0

    current_scale = getattr(coordinate_mapper, "scale_factor", 1.0)
    try:
        current_scale_value = float(current_scale)
    except Exception:
        current_scale_value = 1.0
    if not math.isfinite(current_scale_value) or current_scale_value <= 0:
        current_scale_value = 1.0

    ratio = current_scale_value / reference_scale_value if reference_scale_value else 1.0
    if not math.isfinite(ratio) or ratio <= 0:
        ratio = 1.0

    if ratio >= 1.0:
        return base_size

    scaled = base_size * ratio
    if scaled <= label_vanish_threshold_px:
        return 0.0
    return max(scaled, label_min_screen_font_px)


# ----------------------------------------------------------------------------
# Shape management decorator
# ----------------------------------------------------------------------------


def _manages_shape(render_fn):
    """Decorator that wraps render logic with begin_shape/end_shape lifecycle calls."""
    def wrapper(primitives, *args, **kwargs):
        begin_shape = getattr(primitives, "begin_shape", None)
        end_shape = getattr(primitives, "end_shape", None)
        managing = callable(begin_shape) and callable(end_shape)
        if managing:
            begin_shape()
        try:
            return render_fn(primitives, *args, **kwargs)
        finally:
            if managing:
                end_shape()
    return wrapper


# ----------------------------------------------------------------------------
# Render helpers
# ----------------------------------------------------------------------------


@_manages_shape
def _render_point(primitives, sx, sy, radius, point, fill, style):
    primitives.fill_circle((sx, sy), radius, fill, screen_space=True)

    label = getattr(point, "name", "")
    if label:
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
    _render_point(primitives, sx, sy, radius_raw, point, fill, style)


@_manages_shape
def _render_segment(primitives, start, end, stroke):
    primitives.stroke_line(start, end, stroke, include_width=False)


def render_segment_helper(primitives, segment, coordinate_mapper, style):
    try:
        start = coordinate_mapper.math_to_screen(segment.point1.x, segment.point1.y)
        end = coordinate_mapper.math_to_screen(segment.point2.x, segment.point2.y)
    except Exception:
        return
    if not start or not end:
        return
    stroke = StrokeStyle(
        color=str(getattr(segment, "color", style.get("segment_color", "#000"))),
        width=float(style.get("segment_stroke_width", 1) or 1),
    )
    _render_segment(primitives, start, end, stroke)


@_manages_shape
def _render_ellipse(primitives, center, rx, ry, rotation_rad, stroke):
    primitives.stroke_ellipse(center, rx, ry, rotation_rad, stroke)


def render_ellipse_helper(primitives, ellipse, coordinate_mapper, style):
    try:
        center = coordinate_mapper.math_to_screen(ellipse.center.x, ellipse.center.y)
        radius_x = coordinate_mapper.scale_value(getattr(ellipse, "radius_x", None))
        radius_y = coordinate_mapper.scale_value(getattr(ellipse, "radius_y", None))
    except Exception:
        return
    if not center or radius_x is None or radius_y is None:
        return

    try:
        rx = float(radius_x)
    except Exception:
        rx = None
    try:
        ry = float(radius_y)
    except Exception:
        ry = None
    if rx is None or ry is None or rx <= 0 or ry <= 0:
        return

    color = str(getattr(ellipse, "color", style.get("ellipse_color", "#000")))
    stroke_width_raw = style.get("ellipse_stroke_width", 1) or 1
    try:
        stroke_width = float(stroke_width_raw)
    except Exception:
        stroke_width = 1.0
    stroke = StrokeStyle(color=color, width=stroke_width)

    rotation_deg = getattr(ellipse, "rotation_angle", 0) or 0
    try:
        rotation_rad = -math.radians(float(rotation_deg))
    except Exception:
        rotation_rad = 0.0

    _render_ellipse(primitives, center, rx, ry, rotation_rad, stroke)


@_manages_shape
def _render_circle(primitives, center, radius, stroke):
    primitives.stroke_circle(center, float(radius), stroke)


def render_circle_helper(primitives, circle, coordinate_mapper, style):
    try:
        center = coordinate_mapper.math_to_screen(circle.center.x, circle.center.y)
        radius = coordinate_mapper.scale_value(circle.radius)
    except Exception:
        return
    if not center or radius is None:
        return
    stroke = StrokeStyle(
        color=str(getattr(circle, "color", style.get("circle_color", "#000"))),
        width=float(style.get("circle_stroke_width", 1) or 1),
    )
    _render_circle(primitives, center, radius, stroke)


def _map_circle_arc_to_screen(circle_arc, coordinate_mapper, style):
    try:
        center_screen = coordinate_mapper.math_to_screen(circle_arc.center_x, circle_arc.center_y)
        radius_screen = coordinate_mapper.scale_value(circle_arc.radius)
        point1_screen = coordinate_mapper.math_to_screen(circle_arc.point1.x, circle_arc.point1.y)
        point2_screen = coordinate_mapper.math_to_screen(circle_arc.point2.x, circle_arc.point2.y)
    except Exception:
        return None

    if not center_screen or radius_screen is None or not point1_screen or not point2_screen:
        return None

    radius_scale = style.get("circle_arc_radius_scale", 1.0) or 1.0
    try:
        radius_on_screen = float(radius_screen) * float(radius_scale)
    except Exception:
        radius_on_screen = 0.0
    if radius_on_screen <= 0:
        return None

    return center_screen, radius_on_screen, point1_screen, point2_screen


def _compute_circle_arc_sweep(circle_arc, center_screen, point1_screen):
    cx, cy = center_screen
    p1x, p1y = point1_screen
    start_angle_math = math.atan2(circle_arc.point1.y - circle_arc.center_y, circle_arc.point1.x - circle_arc.center_x)
    target_angle_math = math.atan2(circle_arc.point2.y - circle_arc.center_y, circle_arc.point2.x - circle_arc.center_x)

    full_turn = 2 * math.pi
    delta_ccw_math = (target_angle_math - start_angle_math) % full_turn
    delta_cw_math = (start_angle_math - target_angle_math) % full_turn
    if min(delta_ccw_math, delta_cw_math) < MathUtils.EPSILON:
        return None

    minor_is_ccw = delta_ccw_math <= delta_cw_math
    minor_delta = delta_ccw_math if minor_is_ccw else delta_cw_math
    major_delta = delta_cw_math if minor_is_ccw else delta_ccw_math

    use_major = bool(getattr(circle_arc, "use_major_arc", False))
    if use_major:
        math_ccw = not minor_is_ccw
        sweep_delta = major_delta
    else:
        math_ccw = minor_is_ccw
        sweep_delta = minor_delta

    start_angle_screen = math.atan2(p1y - cy, p1x - cx)
    if math_ccw:
        sweep_clockwise = False
        end_angle_final = start_angle_screen - sweep_delta
    else:
        sweep_clockwise = True
        end_angle_final = start_angle_screen + sweep_delta

    return start_angle_screen, end_angle_final, sweep_clockwise


@_manages_shape
def _stroke_circle_arc(primitives, circle_arc, center_screen, radius_on_screen, start_angle, end_angle, sweep_clockwise, style):
    color = str(getattr(circle_arc, "color", style.get("circle_arc_color", "#000")))
    stroke = StrokeStyle(
        color=color,
        width=float(style.get("circle_arc_stroke_width", 1) or 1),
    )
    metadata = {
        "circle_arc": {
            "center_math": (float(circle_arc.center_x), float(circle_arc.center_y)),
            "radius_math": float(circle_arc.radius),
            "use_major_arc": bool(getattr(circle_arc, "use_major_arc", False)),
            "sweep_clockwise": sweep_clockwise,
            "point1": (float(circle_arc.point1.x), float(circle_arc.point1.y)),
            "point2": (float(circle_arc.point2.x), float(circle_arc.point2.y)),
        }
    }
    primitives.stroke_arc(
        center_screen,
        radius_on_screen,
        float(start_angle),
        float(end_angle),
        sweep_clockwise,
        stroke,
        screen_space=True,
        metadata=metadata,
    )


def render_circle_arc_helper(primitives, circle_arc, coordinate_mapper, style):
    if hasattr(circle_arc, "sync_with_circle") and callable(circle_arc.sync_with_circle):
        circle_arc.sync_with_circle()

    screen_data = _map_circle_arc_to_screen(circle_arc, coordinate_mapper, style)
    if not screen_data:
        return
    center_screen, radius_on_screen, point1_screen, point2_screen = screen_data

    sweep_data = _compute_circle_arc_sweep(circle_arc, center_screen, point1_screen)
    if not sweep_data:
        return

    start_angle_screen, end_angle_final, sweep_clockwise = sweep_data
    _stroke_circle_arc(
        primitives,
        circle_arc,
        center_screen,
        radius_on_screen,
        start_angle_screen,
        end_angle_final,
        sweep_clockwise,
        style,
    )


@_manages_shape
def _render_vector(primitives, start, end, seg, color, stroke, style):
    primitives.stroke_line(start, end, stroke)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    side_length = float(style.get("vector_tip_size", (style.get("point_radius", 2) or 2) * 4))
    half_base = side_length / 2
    height_sq = max(side_length * side_length - half_base * half_base, 0.0)
    height = math.sqrt(height_sq)

    tip = end
    base1 = (
        end[0] - height * math.cos(angle) - half_base * math.sin(angle),
        end[1] - height * math.sin(angle) + half_base * math.cos(angle),
    )
    base2 = (
        end[0] - height * math.cos(angle) + half_base * math.sin(angle),
        end[1] - height * math.sin(angle) - half_base * math.cos(angle),
    )
    metadata = {
        "vector_arrow": {
            "start_math": (getattr(seg.point1, "x", 0.0), getattr(seg.point1, "y", 0.0)),
            "end_math": (getattr(seg.point2, "x", 0.0), getattr(seg.point2, "y", 0.0)),
            "tip_size": side_length,
        }
    }
    primitives.fill_polygon(
        [tip, base1, base2],
        FillStyle(color=color),
        StrokeStyle(color=color, width=1),
        screen_space=True,
        metadata=metadata,
    )


def render_vector_helper(primitives, vector, coordinate_mapper, style):
    seg = getattr(vector, "segment", None)
    if seg is None:
        return
    try:
        start = coordinate_mapper.math_to_screen(seg.point1.x, seg.point1.y)
        end = coordinate_mapper.math_to_screen(seg.point2.x, seg.point2.y)
    except Exception:
        return
    if not start or not end:
        return

    color = str(getattr(vector, "color", getattr(seg, "color", style.get("vector_color", "#000"))))
    stroke = StrokeStyle(color=color, width=float(style.get("segment_stroke_width", 1) or 1))
    _render_vector(primitives, start, end, seg, color, stroke, style)


def _get_label_lines(label):
    try:
        lines = list(getattr(label, "lines", []))
    except Exception:
        lines = []
    if not lines:
        text_value = str(getattr(label, "text", ""))
        lines = text_value.split("\n") if text_value else [""]
    return lines


def _compute_label_font(label, style, coordinate_mapper):
    raw_font_size = getattr(label, "font_size", style.get("label_font_size", 14))
    fallback_font = style.get("label_font_size", 14)
    base_font_size = _coerce_font_size(raw_font_size, fallback_font)
    effective_font_size = _compute_zoom_adjusted_font_size(base_font_size, label, coordinate_mapper)

    if effective_font_size <= 0:
        return None, base_font_size, effective_font_size

    if math.isfinite(effective_font_size) and effective_font_size.is_integer():
        font_size_final = int(effective_font_size)
    else:
        font_size_final = effective_font_size

    font_family = style.get("label_font_family", style.get("font_family", default_font_family))
    font = FontStyle(family=font_family, size=font_size_final)
    return font, base_font_size, effective_font_size


def _build_label_metadata(index, position, offset_y, rotation_degrees, label, base_font_size):
    return {
        "label": {
            "line_index": index,
            "math_position": (position.x, position.y),
            "screen_offset": (0.0, float(offset_y)),
            "rotation_degrees": rotation_degrees,
            "reference_scale_factor": getattr(label, "reference_scale_factor", 1.0),
            "base_font_size": base_font_size,
            "min_font_size": label_min_screen_font_px,
            "vanish_threshold_px": label_vanish_threshold_px,
        }
    }


def render_label_helper(primitives, label, coordinate_mapper, style):
    position = getattr(label, "position", None)
    if position is None:
        return
    try:
        screen_point = coordinate_mapper.math_to_screen(position.x, position.y)
    except Exception:
        return
    if not screen_point:
        return
    screen_x, screen_y = screen_point

    font, base_font_size, effective_font_size = _compute_label_font(label, style, coordinate_mapper)
    if font is None:
        return

    color = str(getattr(label, "color", style.get("label_text_color", "#000")))
    alignment = TextAlignment(horizontal="left", vertical="alphabetic")
    lines = _get_label_lines(label)

    size_numeric = font.size if isinstance(font.size, (int, float)) else effective_font_size
    line_height = float(size_numeric) * 1.2

    try:
        rotation_degrees = float(getattr(label, "rotation_degrees", 0.0))
    except Exception:
        rotation_degrees = 0.0
    if not math.isfinite(rotation_degrees):
        rotation_degrees = 0.0

    for index, line in enumerate(lines):
        current_text = str(line)
        offset_y = index * line_height
        metadata = _build_label_metadata(index, position, offset_y, rotation_degrees, label, base_font_size)
        primitives.draw_text(
            current_text,
            (screen_x, screen_y + offset_y),
            font,
            color,
            alignment,
            metadata=metadata,
        )


def _compute_angle_arc_params(vx, vy, p1x, p1y, arc_radius, coordinate_mapper):
    try:
        arm1_length = math.hypot(p1x - vx, p1y - vy)
        arm2_length = math.hypot(p1x - vx, p1y - vy)
        min_arm_length = min(arm1_length, arm2_length)
    except Exception:
        min_arm_length = arc_radius
    clamped_radius = arc_radius if min_arm_length <= 0 else min(arc_radius, min_arm_length)

    try:
        mapper_scale = float(getattr(coordinate_mapper, "scale_factor", 1.0) or 1.0)
    except Exception:
        mapper_scale = 1.0
    if mapper_scale <= 0:
        mapper_scale = 1.0
    min_arm_length_math = min_arm_length / mapper_scale if min_arm_length > 0 else 0.0

    return clamped_radius, min_arm_length, min_arm_length_math


def _compute_angle_label_params(vx, vy, p1y, clamped_radius, arc_radius, display_degrees, params, style):
    text_radius = clamped_radius * float(style.get("angle_text_arc_radius_factor", 1.8))
    display_angle_rad = math.radians(display_degrees)
    text_delta = display_angle_rad / 2.0
    if params.get("final_sweep_flag", "0") == "0":
        text_delta = -text_delta
    text_angle = params["angle_v_p1_rad"] + text_delta
    tx = vx + text_radius * math.cos(text_angle)
    ty = vy + text_radius * math.sin(text_angle)

    font_size_value = style.get("angle_label_font_size", 12)
    try:
        base_font_size = float(font_size_value)
    except Exception:
        base_font_size = 12.0
    if not math.isfinite(base_font_size) or base_font_size <= 0:
        base_font_size = 12.0

    font_family = style.get("angle_label_font_family", style.get("font_family", default_font_family))
    ratio = clamped_radius / arc_radius if arc_radius > 0 else 1.0
    ratio = max(min(ratio, 1.0), 0.0)
    effective_font_size = base_font_size * ratio
    if effective_font_size < label_min_screen_font_px:
        effective_font_size = label_min_screen_font_px

    font = FontStyle(family=font_family, size=effective_font_size)
    should_draw_label = clamped_radius > 0 and effective_font_size > 0

    return tx, ty, font, base_font_size, should_draw_label


def _build_angle_metadata(angle_obj, arc_radius, clamped_radius, min_arm_length, min_arm_length_math,
                          display_degrees, params, style):
    return {
        "angle": {
            "vertex_math": (getattr(angle_obj.vertex_point, "x", 0.0), getattr(angle_obj.vertex_point, "y", 0.0)),
            "arm1_math": (getattr(angle_obj.arm1_point, "x", 0.0), getattr(angle_obj.arm1_point, "y", 0.0)),
            "arm2_math": (getattr(angle_obj.arm2_point, "x", 0.0), getattr(angle_obj.arm2_point, "y", 0.0)),
            "arc_radius_on_screen": arc_radius,
            "clamped_arc_radius_on_screen": clamped_radius,
            "min_arm_length_on_screen": min_arm_length,
            "min_arm_length_in_math": min_arm_length_math,
            "display_degrees": float(display_degrees),
            "final_sweep_flag": params.get("final_sweep_flag", "0"),
            "text_radius_factor": float(style.get("angle_text_arc_radius_factor", 1.8)),
            "base_font_size": style.get("angle_label_font_size", 12),
            "min_font_size": label_min_screen_font_px,
        }
    }


@_manages_shape
def _draw_angle_arc(primitives, vx, vy, clamped_radius, start_angle, end_angle, sweep_cw, stroke, css_class, metadata):
    primitives.stroke_arc(
        (vx, vy),
        clamped_radius,
        float(start_angle),
        float(end_angle),
        sweep_cw,
        stroke,
        css_class=css_class,
        screen_space=True,
        metadata=metadata,
    )


def _draw_angle_label(primitives, tx, ty, display_degrees, font, color, metadata):
    primitives.draw_text(
        f"{display_degrees:.1f}\u00b0",
        (tx, ty),
        font,
        color,
        TextAlignment(horizontal="center", vertical="middle"),
        screen_space=True,
        metadata=metadata,
    )


def render_angle_helper(primitives, angle_obj, coordinate_mapper, style):
    try:
        vx, vy = coordinate_mapper.math_to_screen(angle_obj.vertex_point.x, angle_obj.vertex_point.y)
        p1x, p1y = coordinate_mapper.math_to_screen(angle_obj.arm1_point.x, angle_obj.arm1_point.y)
        p2x, p2y = coordinate_mapper.math_to_screen(angle_obj.arm2_point.x, angle_obj.arm2_point.y)
    except Exception:
        return

    params = angle_obj._calculate_arc_parameters(
        vx, vy, p1x, p1y, p2x, p2y, arc_radius=style.get("angle_arc_radius")
    )
    if not params:
        return

    arc_radius = float(params["arc_radius_on_screen"])
    try:
        arm1_length = math.hypot(p1x - vx, p1y - vy)
        arm2_length = math.hypot(p2x - vx, p2y - vy)
        min_arm_length = min(arm1_length, arm2_length)
    except Exception:
        min_arm_length = arc_radius
    clamped_radius = arc_radius if min_arm_length <= 0 else min(arc_radius, min_arm_length)

    try:
        mapper_scale = float(getattr(coordinate_mapper, "scale_factor", 1.0) or 1.0)
    except Exception:
        mapper_scale = 1.0
    if mapper_scale <= 0:
        mapper_scale = 1.0
    min_arm_length_math = min_arm_length / mapper_scale if min_arm_length > 0 else 0.0

    display_degrees = getattr(angle_obj, "angle_degrees", None)
    if display_degrees is None:
        return

    color = str(getattr(angle_obj, "color", style.get("angle_color", "#000")))
    stroke = StrokeStyle(color=color, width=float(style.get("angle_stroke_width", 1) or 1))

    tx, ty, font, base_font_size, should_draw_label = _compute_angle_label_params(
        vx, vy, p1y, clamped_radius, arc_radius, display_degrees, params, style
    )

    start_angle = math.atan2(p1y - vy, p1x - vx)
    sweep_cw = params.get("final_sweep_flag", "0") == "1"
    delta = math.radians(display_degrees)
    direction = 1 if sweep_cw else -1
    end_angle = start_angle + direction * delta

    css_class = "angle-arc" if hasattr(primitives, "_surface") else None
    angle_metadata = _build_angle_metadata(
        angle_obj, arc_radius, clamped_radius, min_arm_length, min_arm_length_math,
        display_degrees, params, style
    )

    _draw_angle_arc(primitives, vx, vy, clamped_radius, start_angle, end_angle, sweep_cw, stroke, css_class, angle_metadata)

    if should_draw_label:
        _draw_angle_label(primitives, tx, ty, display_degrees, font, color, angle_metadata)


def _draw_cartesian_axes(primitives, ox, oy, width_px, height_px, axis_stroke):
    primitives.stroke_line((0.0, oy), (width_px, oy), axis_stroke)
    primitives.stroke_line((ox, 0.0), (ox, height_px), axis_stroke)


def _draw_cartesian_tick_x(primitives, x_pos, ox, oy, scale, tick_size, tick_font_float, font,
                           label_color, label_alignment, tick_stroke):
    primitives.stroke_line((x_pos, oy - tick_size), (x_pos, oy + tick_size), tick_stroke)
    if abs(x_pos - ox) < 1e-6:
        primitives.draw_text(
            "O",
            (x_pos + 2, oy + tick_size + tick_font_float),
            font,
            label_color,
            label_alignment,
        )
    else:
        value = (x_pos - ox) / scale
        label = MathUtils.format_number_for_cartesian(value)
        primitives.draw_text(
            label,
            (x_pos + 2, oy + tick_size + tick_font_float),
            font,
            label_color,
            label_alignment,
        )


def _draw_cartesian_tick_y(primitives, y_pos, ox, oy, scale, tick_size, font,
                           label_color, label_alignment, tick_stroke):
    primitives.stroke_line((ox - tick_size, y_pos), (ox + tick_size, y_pos), tick_stroke)
    if abs(y_pos - oy) >= 1e-6:
        value = (oy - y_pos) / scale
        label = MathUtils.format_number_for_cartesian(value)
        primitives.draw_text(
            label,
            (ox + tick_size + 2, y_pos - tick_size),
            font,
            label_color,
            label_alignment,
        )


def _draw_cartesian_mid_tick_x(primitives, x_pos, oy, mid_tick_size, tick_stroke):
    if mid_tick_size <= 0.0:
        return
    primitives.stroke_line((x_pos, oy - mid_tick_size), (x_pos, oy + mid_tick_size), tick_stroke)


def _draw_cartesian_mid_tick_y(primitives, y_pos, ox, mid_tick_size, tick_stroke):
    if mid_tick_size <= 0.0:
        return
    primitives.stroke_line((ox - mid_tick_size, y_pos), (ox + mid_tick_size, y_pos), tick_stroke)


def _draw_cartesian_grid_lines_x(primitives, ox, width_px, height_px, display_tick, grid_stroke,
                                 minor_grid_stroke):
    x = ox
    while x <= width_px:
        primitives.stroke_line((x, 0.0), (x, height_px), grid_stroke)
        mid_x = x + display_tick * 0.5
        if mid_x <= width_px and minor_grid_stroke is not None:
            primitives.stroke_line((mid_x, 0.0), (mid_x, height_px), minor_grid_stroke)
        x += display_tick

    x = ox - display_tick
    while x >= 0.0:
        primitives.stroke_line((x, 0.0), (x, height_px), grid_stroke)
        mid_x = x + display_tick * 0.5
        if mid_x >= 0.0 and minor_grid_stroke is not None:
            primitives.stroke_line((mid_x, 0.0), (mid_x, height_px), minor_grid_stroke)
        x -= display_tick


def _draw_cartesian_grid_lines_y(primitives, oy, width_px, height_px, display_tick, grid_stroke,
                                 minor_grid_stroke):
    y = oy
    while y <= height_px:
        primitives.stroke_line((0.0, y), (width_px, y), grid_stroke)
        mid_y = y + display_tick * 0.5
        if mid_y <= height_px and minor_grid_stroke is not None:
            primitives.stroke_line((0.0, mid_y), (width_px, mid_y), minor_grid_stroke)
        y += display_tick

    y = oy - display_tick
    while y >= 0.0:
        primitives.stroke_line((0.0, y), (width_px, y), grid_stroke)
        mid_y = y + display_tick * 0.5
        if mid_y >= 0.0 and minor_grid_stroke is not None:
            primitives.stroke_line((0.0, mid_y), (width_px, mid_y), minor_grid_stroke)
        y -= display_tick


def _draw_cartesian_ticks_x(primitives, ox, oy, width_px, scale, display_tick, tick_size,
                            mid_tick_size, tick_font_float, font, label_color, label_alignment,
                            tick_stroke):
    x = ox
    while x <= width_px:
        _draw_cartesian_tick_x(primitives, x, ox, oy, scale, tick_size, tick_font_float, font,
                               label_color, label_alignment, tick_stroke)
        mid_x = x + display_tick * 0.5
        if mid_x <= width_px:
            _draw_cartesian_mid_tick_x(primitives, mid_x, oy, mid_tick_size, tick_stroke)
        x += display_tick

    x = ox - display_tick
    while x >= 0.0:
        _draw_cartesian_tick_x(primitives, x, ox, oy, scale, tick_size, tick_font_float, font,
                               label_color, label_alignment, tick_stroke)
        mid_x = x + display_tick * 0.5
        if mid_x >= 0.0:
            _draw_cartesian_mid_tick_x(primitives, mid_x, oy, mid_tick_size, tick_stroke)
        x -= display_tick


def _draw_cartesian_ticks_y(primitives, ox, oy, height_px, scale, display_tick, tick_size,
                            mid_tick_size, font, label_color, label_alignment, tick_stroke):
    y = oy
    while y <= height_px:
        _draw_cartesian_tick_y(primitives, y, ox, oy, scale, tick_size, font, label_color,
                               label_alignment, tick_stroke)
        mid_y = y + display_tick * 0.5
        if mid_y <= height_px:
            _draw_cartesian_mid_tick_y(primitives, mid_y, ox, mid_tick_size, tick_stroke)
        y += display_tick

    y = oy - display_tick
    while y >= 0.0:
        _draw_cartesian_tick_y(primitives, y, ox, oy, scale, tick_size, font, label_color,
                               label_alignment, tick_stroke)
        mid_y = y + display_tick * 0.5
        if mid_y >= 0.0:
            _draw_cartesian_mid_tick_y(primitives, mid_y, ox, mid_tick_size, tick_stroke)
        y -= display_tick


@_manages_shape
def _render_cartesian_grid(
    primitives, ox, oy, width_px, height_px, scale, display_tick, tick_size, mid_tick_size,
    tick_font_float, font, label_color, label_alignment, axis_stroke, grid_stroke,
    minor_grid_stroke, tick_stroke
):
    _draw_cartesian_axes(primitives, ox, oy, width_px, height_px, axis_stroke)
    _draw_cartesian_grid_lines_x(primitives, ox, width_px, height_px, display_tick, grid_stroke,
                                 minor_grid_stroke)
    _draw_cartesian_grid_lines_y(primitives, oy, width_px, height_px, display_tick, grid_stroke,
                                 minor_grid_stroke)
    _draw_cartesian_ticks_x(primitives, ox, oy, width_px, scale, display_tick, tick_size,
                            mid_tick_size, tick_font_float, font, label_color, label_alignment,
                            tick_stroke)
    _draw_cartesian_ticks_y(primitives, ox, oy, height_px, scale, display_tick, tick_size,
                            mid_tick_size, font, label_color, label_alignment, tick_stroke)


def _get_cartesian_styles(style):
    axis_color = str(style.get("cartesian_axis_color", "#000"))
    grid_color = str(style.get("cartesian_grid_color", "lightgrey"))
    label_color = str(style.get("cartesian_label_color", "grey"))

    tick_size_raw = style.get("cartesian_tick_size", 3)
    try:
        tick_size = float(tick_size_raw)
    except Exception:
        tick_size = 3.0
    if not math.isfinite(tick_size):
        tick_size = 3.0
    tick_size = max(tick_size, 0.0)
    mid_tick_size = max(tick_size * 0.5, 0.0)

    tick_font_raw = style.get("cartesian_tick_font_size", 8)
    try:
        tick_font_float = float(tick_font_raw)
    except Exception:
        tick_font_float = 8.0
    if not math.isfinite(tick_font_float):
        tick_font_float = 8.0

    font_family = style.get("cartesian_font_family", style.get("font_family", default_font_family))
    font = FontStyle(family=font_family, size=tick_font_raw)
    label_alignment = TextAlignment(horizontal="left", vertical="alphabetic")

    axis_stroke = StrokeStyle(color=axis_color, width=1)
    grid_stroke = StrokeStyle(color=grid_color, width=1)
    tick_stroke = StrokeStyle(color=axis_color, width=1)

    minor_grid_color = str(style.get("cartesian_minor_grid_color", grid_color))
    minor_grid_width_raw = style.get("cartesian_minor_grid_width", 0.5)
    try:
        minor_grid_width = float(minor_grid_width_raw)
    except Exception:
        minor_grid_width = 0.5
    if not math.isfinite(minor_grid_width):
        minor_grid_width = 0.5
    minor_grid_width = max(minor_grid_width, 0.0)
    minor_grid_stroke = (
        StrokeStyle(color=minor_grid_color, width=minor_grid_width) if minor_grid_width > 0.0 else None
    )

    return {
        "label_color": label_color,
        "tick_size": tick_size,
        "mid_tick_size": mid_tick_size,
        "tick_font_float": tick_font_float,
        "font": font,
        "label_alignment": label_alignment,
        "axis_stroke": axis_stroke,
        "grid_stroke": grid_stroke,
        "minor_grid_stroke": minor_grid_stroke,
        "tick_stroke": tick_stroke,
    }


def _compute_cartesian_layout(cartesian, coordinate_mapper):
    width = getattr(cartesian, "width", None)
    height = getattr(cartesian, "height", None)
    if width is None or height is None:
        return None
    try:
        width_px = float(width)
        height_px = float(height)
    except Exception:
        return None
    if not math.isfinite(width_px) or not math.isfinite(height_px) or width_px <= 0 or height_px <= 0:
        return None

    try:
        ox, oy = coordinate_mapper.math_to_screen(0, 0)
    except Exception:
        return None
    try:
        ox = float(ox)
        oy = float(oy)
    except Exception:
        return None

    scale_factor = getattr(coordinate_mapper, "scale_factor", 1)
    try:
        scale = float(scale_factor)
    except Exception:
        scale = 1.0
    if not math.isfinite(scale) or scale == 0:
        scale = 1.0

    tick_spacing_raw = getattr(cartesian, "current_tick_spacing", None)
    if tick_spacing_raw is None:
        tick_spacing_raw = getattr(cartesian, "default_tick_spacing", 1)
    try:
        tick_spacing = float(tick_spacing_raw)
    except Exception:
        tick_spacing = 1.0
    if not math.isfinite(tick_spacing) or tick_spacing <= 0:
        tick_spacing = 1.0

    display_tick = tick_spacing * scale
    try:
        display_tick = float(display_tick)
    except Exception:
        display_tick = scale
    if not math.isfinite(display_tick) or display_tick <= 0:
        display_tick = scale if math.isfinite(scale) and scale > 0 else 1.0
    if not math.isfinite(display_tick) or display_tick <= 0:
        display_tick = 1.0

    return {
        "ox": ox,
        "oy": oy,
        "width_px": width_px,
        "height_px": height_px,
        "scale": scale,
        "display_tick": display_tick,
    }


def render_cartesian_helper(primitives, cartesian, coordinate_mapper, style):
    layout = _compute_cartesian_layout(cartesian, coordinate_mapper)
    if layout is None:
        return

    styles = _get_cartesian_styles(style)

    _render_cartesian_grid(
        primitives,
        layout["ox"],
        layout["oy"],
        layout["width_px"],
        layout["height_px"],
        layout["scale"],
        layout["display_tick"],
        styles["tick_size"],
        styles["mid_tick_size"],
        styles["tick_font_float"],
        styles["font"],
        styles["label_color"],
        styles["label_alignment"],
        styles["axis_stroke"],
        styles["grid_stroke"],
        styles["minor_grid_stroke"],
        styles["tick_stroke"],
    )


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


# ----------------------------------------------------------------------------
# High-level colored area helpers
# ----------------------------------------------------------------------------


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
    model_name = getattr(area_model, "name", "")
    area = build_closed_shape_colored_area(area_model, coordinate_mapper)
    if area is None:
        return
    render_colored_area_helper(primitives, area, coordinate_mapper, style)


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


# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------


def build_functions_colored_area(area_model, coordinate_mapper):
    try:
        renderable = FunctionsBoundedAreaRenderable(area_model, coordinate_mapper)
        return renderable.build_screen_area()
    except Exception:
        return None


def build_function_segment_colored_area(area_model, coordinate_mapper, *, num_points=100):
    try:
        renderable = FunctionSegmentAreaRenderable(area_model, coordinate_mapper)
        return renderable.build_screen_area(num_points=num_points)
    except Exception:
        return None


def build_segments_colored_area(area_model, coordinate_mapper):
    try:
        renderable = SegmentsBoundedAreaRenderable(area_model, coordinate_mapper)
        return renderable.build_screen_area()
    except Exception:
        return None


def build_closed_shape_colored_area(area_model, coordinate_mapper):
    try:
        renderable = ClosedShapeAreaRenderable(area_model, coordinate_mapper)
        return renderable.build_screen_area()
    except Exception:
        return None


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

