"""Cartesian grid rendering helper for drawing axis-aligned coordinate systems.

This module provides the render_cartesian_helper function that renders a
Cartesian coordinate system with axes, grid lines, tick marks, and labels.

Key Features:
    - X and Y axis rendering with configurable colors
    - Major and minor grid lines at regular intervals
    - Tick marks with automatic precision labeling
    - Scientific notation for extreme values
    - Dynamic tick spacing based on zoom level
    - Origin marker display
"""

from __future__ import annotations

import math

from constants import default_font_family
from rendering.helpers.shape_decorator import _manages_shape
from rendering.primitives import FontStyle, StrokeStyle, TextAlignment
from utils.math_utils import MathUtils


def _calculate_tick_precision(spacing: float) -> int:
    """Calculate decimal places needed to distinguish adjacent tick labels.

    Given the spacing between ticks, determines the minimum number of decimal
    places required so that adjacent tick values format to different strings.

    Args:
        spacing: The spacing between adjacent ticks in math units

    Returns:
        Number of decimal places needed (0 for integers, positive for decimals)
    """
    if spacing <= 0 or not math.isfinite(spacing):
        return 0
    if spacing >= 1:
        return 0
    # For spacing < 1, calculate decimal places from -log10(spacing)
    # e.g., spacing 0.1 → 1 decimal place
    # e.g., spacing 0.02 → 2 decimal places
    return max(0, int(math.ceil(-math.log10(spacing))))


def _format_tick_value(value: float, precision: int) -> str:
    """Format a tick value with the specified number of decimal places.

    Uses scientific notation when precision > 4 to avoid verbose labels like
    0.000107. Shows minimal significant figures to distinguish adjacent ticks.

    Args:
        value: The numeric value to format
        precision: Number of decimal places to show

    Returns:
        Formatted string representation
    """
    if value == 0:
        return "0"
    if not math.isfinite(value):
        return str(value)
    # For very large values, use scientific notation
    if abs(value) >= 1e6:
        return f"{value:.1e}"
    # For values needing many decimal places, use scientific notation
    # This handles cases like 0.000107 → 1.1e-4 instead of verbose decimals
    if precision > 4 or (abs(value) < 0.001 and abs(value) > 0):
        # Use 2 significant figures for scientific notation
        formatted = f"{value:.1e}"
        # Clean up the exponent format (remove leading zeros)
        if "e" in formatted:
            base, exp = formatted.split("e")
            exp_sign = exp[0] if exp[0] in "+-" else "+"
            exp_num = exp.lstrip("+-").lstrip("0") or "0"
            formatted = f"{base}e{exp_sign}{exp_num}"
        return formatted
    if precision <= 0:
        return str(int(round(value)))
    formatted = f"{value:.{precision}f}"
    # Strip trailing zeros but keep at least one decimal place if precision > 0
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _draw_cartesian_axes(primitives, ox, oy, width_px, height_px, axis_stroke):
    """Draw the main X and Y axes through the origin.

    Args:
        primitives: The renderer primitives interface.
        ox: Origin x coordinate in screen pixels.
        oy: Origin y coordinate in screen pixels.
        width_px: Canvas width in pixels.
        height_px: Canvas height in pixels.
        axis_stroke: StrokeStyle for the axes.
    """
    primitives.stroke_line((0.0, oy), (width_px, oy), axis_stroke)
    primitives.stroke_line((ox, 0.0), (ox, height_px), axis_stroke)


def _draw_cartesian_tick_x(
    primitives,
    x_pos,
    ox,
    oy,
    scale,
    tick_size,
    tick_font_float,
    font,
    label_color,
    label_alignment,
    tick_stroke,
    precision=6,
):
    """Draw a single X-axis tick mark with label.

    Args:
        primitives: The renderer primitives interface.
        x_pos: Tick x coordinate in screen pixels.
        ox: Origin x coordinate in screen pixels.
        oy: Origin y coordinate in screen pixels.
        scale: Scale factor for coordinate conversion.
        tick_size: Half-height of tick mark in pixels.
        tick_font_float: Font size for positioning.
        font: FontStyle for the label.
        label_color: Color string for the label.
        label_alignment: TextAlignment for the label.
        tick_stroke: StrokeStyle for the tick mark.
        precision: Decimal places for label formatting.
    """
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
        label = _format_tick_value(value, precision)
        primitives.draw_text(
            label,
            (x_pos + 2, oy + tick_size + tick_font_float),
            font,
            label_color,
            label_alignment,
        )


def _draw_cartesian_tick_y(
    primitives, y_pos, ox, oy, scale, tick_size, font, label_color, label_alignment, tick_stroke, precision=6
):
    """Draw a single Y-axis tick mark with label.

    Args:
        primitives: The renderer primitives interface.
        y_pos: Tick y coordinate in screen pixels.
        ox: Origin x coordinate in screen pixels.
        oy: Origin y coordinate in screen pixels.
        scale: Scale factor for coordinate conversion.
        tick_size: Half-width of tick mark in pixels.
        font: FontStyle for the label.
        label_color: Color string for the label.
        label_alignment: TextAlignment for the label.
        tick_stroke: StrokeStyle for the tick mark.
        precision: Decimal places for label formatting.
    """
    primitives.stroke_line((ox - tick_size, y_pos), (ox + tick_size, y_pos), tick_stroke)
    if abs(y_pos - oy) >= 1e-6:
        value = (oy - y_pos) / scale
        label = _format_tick_value(value, precision)
        primitives.draw_text(
            label,
            (ox + tick_size + 2, y_pos - tick_size),
            font,
            label_color,
            label_alignment,
        )


def _draw_cartesian_mid_tick_x(primitives, x_pos, oy, mid_tick_size, tick_stroke):
    """Draw a smaller mid-point tick mark on the X axis.

    Args:
        primitives: The renderer primitives interface.
        x_pos: Tick x coordinate in screen pixels.
        oy: Origin y coordinate in screen pixels.
        mid_tick_size: Half-height of mid tick in pixels.
        tick_stroke: StrokeStyle for the tick mark.
    """
    if mid_tick_size <= 0.0:
        return
    primitives.stroke_line((x_pos, oy - mid_tick_size), (x_pos, oy + mid_tick_size), tick_stroke)


def _draw_cartesian_mid_tick_y(primitives, y_pos, ox, mid_tick_size, tick_stroke):
    """Draw a smaller mid-point tick mark on the Y axis.

    Args:
        primitives: The renderer primitives interface.
        y_pos: Tick y coordinate in screen pixels.
        ox: Origin x coordinate in screen pixels.
        mid_tick_size: Half-width of mid tick in pixels.
        tick_stroke: StrokeStyle for the tick mark.
    """
    if mid_tick_size <= 0.0:
        return
    primitives.stroke_line((ox - mid_tick_size, y_pos), (ox + mid_tick_size, y_pos), tick_stroke)


def _draw_cartesian_grid_lines_x(primitives, ox, width_px, height_px, display_tick, grid_stroke, minor_grid_stroke):
    """Draw vertical grid lines at regular X intervals.

    Args:
        primitives: The renderer primitives interface.
        ox: Origin x coordinate in screen pixels.
        width_px: Canvas width in pixels.
        height_px: Canvas height in pixels.
        display_tick: Spacing between major grid lines in pixels.
        grid_stroke: StrokeStyle for major grid lines.
        minor_grid_stroke: StrokeStyle for minor grid lines, or None.
    """
    if display_tick <= 0:
        return
    import math

    start_n = int(math.ceil(-ox / display_tick))
    end_n = int(math.floor((width_px - ox) / display_tick))
    for n in range(start_n, end_n + 1):
        x = ox + n * display_tick
        if 0 <= x <= width_px:
            primitives.stroke_line((x, 0.0), (x, height_px), grid_stroke)
        if minor_grid_stroke is not None:
            mid_x = x + display_tick * 0.5
            if 0 <= mid_x <= width_px:
                primitives.stroke_line((mid_x, 0.0), (mid_x, height_px), minor_grid_stroke)


def _draw_cartesian_grid_lines_y(primitives, oy, width_px, height_px, display_tick, grid_stroke, minor_grid_stroke):
    """Draw horizontal grid lines at regular Y intervals.

    Args:
        primitives: The renderer primitives interface.
        oy: Origin y coordinate in screen pixels.
        width_px: Canvas width in pixels.
        height_px: Canvas height in pixels.
        display_tick: Spacing between major grid lines in pixels.
        grid_stroke: StrokeStyle for major grid lines.
        minor_grid_stroke: StrokeStyle for minor grid lines, or None.
    """
    if display_tick <= 0:
        return
    import math

    start_n = int(math.ceil(-oy / display_tick))
    end_n = int(math.floor((height_px - oy) / display_tick))
    for n in range(start_n, end_n + 1):
        y = oy + n * display_tick
        if 0 <= y <= height_px:
            primitives.stroke_line((0.0, y), (width_px, y), grid_stroke)
        if minor_grid_stroke is not None:
            mid_y = y + display_tick * 0.5
            if 0 <= mid_y <= height_px:
                primitives.stroke_line((0.0, mid_y), (width_px, mid_y), minor_grid_stroke)


def _draw_cartesian_ticks_x(
    primitives,
    ox,
    oy,
    width_px,
    scale,
    display_tick,
    tick_size,
    mid_tick_size,
    tick_font_float,
    font,
    label_color,
    label_alignment,
    tick_stroke,
):
    """Draw all tick marks and labels along the X axis.

    Args:
        primitives: The renderer primitives interface.
        ox: Origin x coordinate in screen pixels.
        oy: Origin y coordinate in screen pixels.
        width_px: Canvas width in pixels.
        scale: Scale factor for coordinate conversion.
        display_tick: Spacing between major ticks in pixels.
        tick_size: Half-height of major tick marks in pixels.
        mid_tick_size: Half-height of minor tick marks in pixels.
        tick_font_float: Font size for positioning labels.
        font: FontStyle for labels.
        label_color: Color string for labels.
        label_alignment: TextAlignment for labels.
        tick_stroke: StrokeStyle for tick marks.
    """
    if display_tick <= 0:
        return
    import math

    # Calculate math spacing and precision needed for labels
    math_spacing = display_tick / scale if scale > 0 else display_tick
    precision = _calculate_tick_precision(math_spacing)
    start_n = int(math.ceil(-ox / display_tick))
    end_n = int(math.floor((width_px - ox) / display_tick))
    for n in range(start_n, end_n + 1):
        x = ox + n * display_tick
        if 0 <= x <= width_px:
            _draw_cartesian_tick_x(
                primitives,
                x,
                ox,
                oy,
                scale,
                tick_size,
                tick_font_float,
                font,
                label_color,
                label_alignment,
                tick_stroke,
                precision,
            )
        mid_x = x + display_tick * 0.5
        if 0 <= mid_x <= width_px:
            _draw_cartesian_mid_tick_x(primitives, mid_x, oy, mid_tick_size, tick_stroke)


def _draw_cartesian_ticks_y(
    primitives,
    ox,
    oy,
    height_px,
    scale,
    display_tick,
    tick_size,
    mid_tick_size,
    font,
    label_color,
    label_alignment,
    tick_stroke,
):
    """Draw all tick marks and labels along the Y axis.

    Args:
        primitives: The renderer primitives interface.
        ox: Origin x coordinate in screen pixels.
        oy: Origin y coordinate in screen pixels.
        height_px: Canvas height in pixels.
        scale: Scale factor for coordinate conversion.
        display_tick: Spacing between major ticks in pixels.
        tick_size: Half-width of major tick marks in pixels.
        mid_tick_size: Half-width of minor tick marks in pixels.
        font: FontStyle for labels.
        label_color: Color string for labels.
        label_alignment: TextAlignment for labels.
        tick_stroke: StrokeStyle for tick marks.
    """
    if display_tick <= 0:
        return
    import math

    # Calculate math spacing and precision needed for labels
    math_spacing = display_tick / scale if scale > 0 else display_tick
    precision = _calculate_tick_precision(math_spacing)
    start_n = int(math.ceil(-oy / display_tick))
    end_n = int(math.floor((height_px - oy) / display_tick))
    for n in range(start_n, end_n + 1):
        y = oy + n * display_tick
        if 0 <= y <= height_px:
            _draw_cartesian_tick_y(
                primitives, y, ox, oy, scale, tick_size, font, label_color, label_alignment, tick_stroke, precision
            )
        mid_y = y + display_tick * 0.5
        if 0 <= mid_y <= height_px:
            _draw_cartesian_mid_tick_y(primitives, mid_y, ox, mid_tick_size, tick_stroke)


@_manages_shape
def _render_cartesian_grid(
    primitives,
    ox,
    oy,
    width_px,
    height_px,
    scale,
    display_tick,
    tick_size,
    mid_tick_size,
    tick_font_float,
    font,
    label_color,
    label_alignment,
    axis_stroke,
    grid_stroke,
    minor_grid_stroke,
    tick_stroke,
):
    """Render the complete Cartesian grid with all components.

    Args:
        primitives: The renderer primitives interface.
        ox: Origin x coordinate in screen pixels.
        oy: Origin y coordinate in screen pixels.
        width_px: Canvas width in pixels.
        height_px: Canvas height in pixels.
        scale: Scale factor for coordinate conversion.
        display_tick: Spacing between major ticks/grid lines in pixels.
        tick_size: Half-size of major tick marks in pixels.
        mid_tick_size: Half-size of minor tick marks in pixels.
        tick_font_float: Font size for label positioning.
        font: FontStyle for labels.
        label_color: Color string for labels.
        label_alignment: TextAlignment for labels.
        axis_stroke: StrokeStyle for axes.
        grid_stroke: StrokeStyle for major grid lines.
        minor_grid_stroke: StrokeStyle for minor grid lines, or None.
        tick_stroke: StrokeStyle for tick marks.
    """
    _draw_cartesian_axes(primitives, ox, oy, width_px, height_px, axis_stroke)
    _draw_cartesian_grid_lines_x(primitives, ox, width_px, height_px, display_tick, grid_stroke, minor_grid_stroke)
    _draw_cartesian_grid_lines_y(primitives, oy, width_px, height_px, display_tick, grid_stroke, minor_grid_stroke)
    _draw_cartesian_ticks_x(
        primitives,
        ox,
        oy,
        width_px,
        scale,
        display_tick,
        tick_size,
        mid_tick_size,
        tick_font_float,
        font,
        label_color,
        label_alignment,
        tick_stroke,
    )
    _draw_cartesian_ticks_y(
        primitives,
        ox,
        oy,
        height_px,
        scale,
        display_tick,
        tick_size,
        mid_tick_size,
        font,
        label_color,
        label_alignment,
        tick_stroke,
    )


def _get_cartesian_styles(style):
    """Extract and build style objects for Cartesian grid rendering.

    Args:
        style: Style dictionary with cartesian_* settings.

    Returns:
        Dict with label_color, tick_size, mid_tick_size, tick_font_float,
        font, label_alignment, axis_stroke, grid_stroke, minor_grid_stroke,
        and tick_stroke.
    """
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
    minor_grid_stroke = StrokeStyle(color=minor_grid_color, width=minor_grid_width) if minor_grid_width > 0.0 else None

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
    """Compute layout parameters for Cartesian grid rendering.

    Args:
        cartesian: Cartesian coordinate system drawable.
        coordinate_mapper: Mapper for coordinate conversion.

    Returns:
        Dict with ox, oy, width_px, height_px, scale, display_tick,
        or None if layout cannot be computed.
    """
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
    """Render a Cartesian coordinate system drawable.

    Args:
        primitives: The renderer primitives interface.
        cartesian: Cartesian coordinate system drawable with width, height.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.
        style: Style dictionary with cartesian_* settings.
    """
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
