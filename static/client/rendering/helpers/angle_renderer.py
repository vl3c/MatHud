from __future__ import annotations

import math

from constants import default_font_family, label_min_screen_font_px
from rendering.helpers.shape_decorator import _manages_shape
from rendering.renderer_primitives import FontStyle, StrokeStyle, TextAlignment


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

