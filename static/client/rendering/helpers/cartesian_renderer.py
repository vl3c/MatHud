from __future__ import annotations

import math

from constants import default_font_family
from rendering.helpers.shape_decorator import _manages_shape
from rendering.primitives import FontStyle, StrokeStyle, TextAlignment
from utils.math_utils import MathUtils


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

