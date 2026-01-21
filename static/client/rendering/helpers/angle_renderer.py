"""Angle rendering helper for drawing angle arcs with degree labels.

This module provides the render_angle_helper function that renders an angle
drawable as a curved arc between two arms with an optional degree label.

Key Features:
    - Arc rendering between angle arms at vertex point
    - Automatic arc radius clamping to fit within arm lengths
    - Degree label positioning at configurable distance from vertex
    - Font size scaling based on available space
    - Metadata emission for hit testing and debugging
"""

from __future__ import annotations

import math

from constants import default_font_family, label_min_screen_font_px
from rendering.helpers.shape_decorator import _manages_shape
from rendering.primitives import FontStyle, StrokeStyle, TextAlignment


def _compute_angle_arc_params(vx, vy, p1x, p1y, arc_radius, coordinate_mapper):
    """Compute arc parameters with radius clamping.

    Args:
        vx: Vertex screen x coordinate.
        vy: Vertex screen y coordinate.
        p1x: Arm1 endpoint screen x coordinate.
        p1y: Arm1 endpoint screen y coordinate.
        arc_radius: Requested arc radius in screen pixels.
        coordinate_mapper: Mapper for scale factor extraction.

    Returns:
        Tuple of (clamped_radius, min_arm_length, min_arm_length_math).
    """
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
    """Compute label positioning and font parameters for angle text.

    Args:
        vx: Vertex screen x coordinate.
        vy: Vertex screen y coordinate.
        p1y: Arm1 endpoint screen y coordinate (unused but kept for signature).
        clamped_radius: Arc radius after clamping to arm length.
        arc_radius: Original requested arc radius.
        display_degrees: Angle measurement in degrees.
        params: Arc parameters dict with angle_v_p1_rad and final_sweep_flag.
        style: Style dictionary with font settings.

    Returns:
        Tuple of (tx, ty, font, base_font_size, should_draw_label).
    """
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
    """Build metadata dictionary for angle rendering debugging.

    Args:
        angle_obj: The Angle drawable with vertex and arm points.
        arc_radius: Original requested arc radius.
        clamped_radius: Arc radius after clamping.
        min_arm_length: Shortest arm length in screen pixels.
        min_arm_length_math: Shortest arm length in math coordinates.
        display_degrees: Angle measurement in degrees.
        params: Arc parameters dict with sweep flag.
        style: Style dictionary with rendering settings.

    Returns:
        Dict containing angle metadata for hit testing and debugging.
    """
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
    """Draw the angle arc stroke at screen coordinates.

    Args:
        primitives: The renderer primitives interface.
        vx: Vertex screen x coordinate.
        vy: Vertex screen y coordinate.
        clamped_radius: Arc radius in screen pixels.
        start_angle: Starting angle in radians.
        end_angle: Ending angle in radians.
        sweep_cw: True for clockwise sweep direction.
        stroke: StrokeStyle for the arc.
        css_class: Optional CSS class for SVG rendering.
        metadata: Metadata dict to attach to the arc.
    """
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
    """Draw the angle degree label at the computed position.

    Args:
        primitives: The renderer primitives interface.
        tx: Label screen x coordinate.
        ty: Label screen y coordinate.
        display_degrees: Angle measurement to display.
        font: FontStyle for the label text.
        color: Text color string.
        metadata: Metadata dict to attach to the label.
    """
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
    """Render an angle drawable with arc and degree label.

    Args:
        primitives: The renderer primitives interface.
        angle_obj: The Angle drawable with vertex_point, arm1_point, arm2_point.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.
        style: Style dictionary with angle_arc_radius, angle_color, etc.
    """
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

