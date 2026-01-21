"""Polar grid rendering helper for drawing radial coordinate systems.

This module provides the render_polar_helper function that renders a
polar coordinate system with concentric circles, radial lines, and labels.

Key Features:
    - Concentric circle grid at regular radial intervals
    - Radial lines at configurable angular divisions
    - Angle labels at canvas boundary positions
    - Radius labels along the positive X axis
    - Dynamic spacing based on zoom level
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
        if 'e' in formatted:
            base, exp = formatted.split('e')
            exp_sign = exp[0] if exp[0] in '+-' else '+'
            exp_num = exp.lstrip('+-').lstrip('0') or '0'
            formatted = f"{base}e{exp_sign}{exp_num}"
        return formatted
    if precision <= 0:
        return str(int(round(value)))
    formatted = f"{value:.{precision}f}"
    # Strip trailing zeros but keep at least one decimal place if precision > 0
    if '.' in formatted:
        formatted = formatted.rstrip('0').rstrip('.')
    return formatted


def _draw_polar_axes(primitives, ox, oy, width_px, height_px, axis_stroke):
    """Draw the main x and y axes through the origin."""
    primitives.stroke_line((0.0, oy), (width_px, oy), axis_stroke)
    primitives.stroke_line((ox, 0.0), (ox, height_px), axis_stroke)


def _draw_concentric_circles(primitives, ox, oy, max_radius_screen, display_spacing, circle_stroke):
    """Draw concentric circles at regular radial intervals."""
    if display_spacing <= 0:
        return
    n = 1
    while True:
        radius = n * display_spacing
        if radius > max_radius_screen:
            break
        primitives.stroke_circle((ox, oy), radius, circle_stroke)
        n += 1


def _draw_radial_lines(primitives, ox, oy, max_radius_screen, angular_step_degrees, radial_stroke):
    """Draw radial lines from origin at regular angular intervals."""
    if angular_step_degrees <= 0:
        return
    num_lines = int(360 / angular_step_degrees)
    for i in range(num_lines):
        angle_deg = i * angular_step_degrees
        angle_rad = math.radians(angle_deg)
        end_x = ox + max_radius_screen * math.cos(angle_rad)
        end_y = oy - max_radius_screen * math.sin(angle_rad)
        primitives.stroke_line((ox, oy), (end_x, end_y), radial_stroke)


def _draw_angle_labels(primitives, ox, oy, label_radius_screen, angular_step_degrees,
                       font, label_color, label_alignment, width_px, height_px):
    """Draw angle labels at the external boundary of the visible canvas."""
    if angular_step_degrees <= 0:
        return

    padding = 15  # Padding from edge
    left_padding = 10  # Less padding for left side labels (closer to boundary)
    right_padding = 25  # Extra padding for right side labels (text extends right)
    num_labels = int(360 / angular_step_degrees)

    for i in range(num_labels):
        angle_deg = i * angular_step_degrees
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Calculate radius to edge of canvas for this angle
        # Find where the ray from origin intersects the canvas boundary
        if abs(cos_a) < 1e-10:
            # Vertical line (90° or 270°)
            if sin_a > 0:
                edge_radius = oy - padding
            else:
                edge_radius = height_px - oy - padding
        elif abs(sin_a) < 1e-10:
            # Horizontal line (0° or 180°)
            if cos_a > 0:
                edge_radius = width_px - ox - right_padding
            else:
                edge_radius = ox - left_padding
        else:
            # General case - find intersection with canvas edges
            if cos_a > 0:
                t_right = (width_px - ox - right_padding) / cos_a
            else:
                t_right = (left_padding - ox) / cos_a
            if sin_a > 0:
                t_top = (oy - padding) / sin_a
            else:
                t_top = (oy - height_px + padding) / sin_a
            edge_radius = min(t_right, t_top) if t_right > 0 and t_top > 0 else max(t_right, t_top)

        edge_radius = max(edge_radius, 30)  # Minimum radius

        label_x = ox + edge_radius * cos_a
        label_y = oy - edge_radius * sin_a
        label_text = f"{int(angle_deg)}°"
        primitives.draw_text(label_text, (label_x, label_y), font, label_color, label_alignment)


def _draw_radius_labels(primitives, ox, oy, scale, display_spacing, max_radius_screen,
                        tick_font_float, font, label_color, label_alignment):
    """Draw radius labels along the positive x-axis.

    Uses spacing-aware formatting to show minimum digits needed to distinguish
    adjacent tick labels (e.g., 0.12 and 0.14 instead of 0.125345 and 0.145546).
    """
    if display_spacing <= 0:
        return
    # Calculate math spacing and precision needed
    math_spacing = display_spacing / scale if scale > 0 else display_spacing
    precision = _calculate_tick_precision(math_spacing)
    n = 1
    while True:
        radius_screen = n * display_spacing
        if radius_screen > max_radius_screen:
            break
        radius_math = radius_screen / scale if scale > 0 else radius_screen
        label = _format_tick_value(radius_math, precision)
        label_x = ox + radius_screen + 2
        label_y = oy + tick_font_float
        primitives.draw_text(label, (label_x, label_y), font, label_color, label_alignment)
        n += 1


def _draw_origin_marker(primitives, ox, oy, tick_font_float, font, label_color, label_alignment):
    """Draw the origin marker 'O'."""
    primitives.draw_text("O", (ox + 2, oy + tick_font_float), font, label_color, label_alignment)


@_manages_shape
def _render_polar_grid(
    primitives, ox, oy, width_px, height_px, scale, display_spacing, max_radius_screen,
    angular_step_degrees, tick_font_float, font, label_color, label_alignment,
    axis_stroke, circle_stroke, radial_stroke
):
    """Render the complete polar grid with all components.

    Args:
        primitives: The renderer primitives interface.
        ox: Origin x coordinate in screen pixels.
        oy: Origin y coordinate in screen pixels.
        width_px: Canvas width in pixels.
        height_px: Canvas height in pixels.
        scale: Scale factor for coordinate conversion.
        display_spacing: Spacing between concentric circles in pixels.
        max_radius_screen: Maximum radius to render in pixels.
        angular_step_degrees: Degrees between radial lines.
        tick_font_float: Font size for label positioning.
        font: FontStyle for labels.
        label_color: Color string for labels.
        label_alignment: TextAlignment for labels.
        axis_stroke: StrokeStyle for main axes.
        circle_stroke: StrokeStyle for concentric circles.
        radial_stroke: StrokeStyle for radial lines.
    """
    _draw_polar_axes(primitives, ox, oy, width_px, height_px, axis_stroke)
    _draw_concentric_circles(primitives, ox, oy, max_radius_screen, display_spacing, circle_stroke)
    _draw_radial_lines(primitives, ox, oy, max_radius_screen, angular_step_degrees, radial_stroke)
    _draw_angle_labels(primitives, ox, oy, max_radius_screen, angular_step_degrees,
                       font, label_color, label_alignment, width_px, height_px)
    _draw_radius_labels(primitives, ox, oy, scale, display_spacing, max_radius_screen,
                        tick_font_float, font, label_color, label_alignment)
    _draw_origin_marker(primitives, ox, oy, tick_font_float, font, label_color, label_alignment)


def _get_polar_styles(style):
    """Extract and build style objects for polar grid rendering.

    Args:
        style: Style dictionary with polar_* settings.

    Returns:
        Dict with label_color, tick_font_float, font, label_alignment,
        axis_stroke, circle_stroke, and radial_stroke.
    """
    axis_color = str(style.get("polar_axis_color", "#000"))
    circle_color = str(style.get("polar_circle_color", "lightgrey"))
    radial_color = str(style.get("polar_radial_color", "lightgrey"))
    label_color = str(style.get("polar_label_color", "grey"))

    label_font_raw = style.get("polar_label_font_size", 8)
    try:
        tick_font_float = float(label_font_raw)
    except Exception:
        tick_font_float = 8.0
    if not math.isfinite(tick_font_float):
        tick_font_float = 8.0

    font_family = style.get("polar_font_family", style.get("font_family", default_font_family))
    font = FontStyle(family=font_family, size=label_font_raw)
    label_alignment = TextAlignment(horizontal="left", vertical="alphabetic")

    axis_stroke = StrokeStyle(color=axis_color, width=1)
    circle_stroke = StrokeStyle(color=circle_color, width=1)
    radial_stroke = StrokeStyle(color=radial_color, width=1)

    return {
        "label_color": label_color,
        "tick_font_float": tick_font_float,
        "font": font,
        "label_alignment": label_alignment,
        "axis_stroke": axis_stroke,
        "circle_stroke": circle_stroke,
        "radial_stroke": radial_stroke,
    }


def _compute_polar_layout(polar_grid, coordinate_mapper):
    """Compute layout parameters for polar grid rendering.

    Args:
        polar_grid: PolarGrid drawable with width, height, radial spacing.
        coordinate_mapper: Mapper for coordinate conversion.

    Returns:
        Dict with ox, oy, width_px, height_px, scale, display_spacing,
        max_radius_screen, angular_step_degrees, or None if invalid.
    """
    width = getattr(polar_grid, "width", None)
    height = getattr(polar_grid, "height", None)
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

    radial_spacing_raw = getattr(polar_grid, "_current_radial_spacing", None)
    if radial_spacing_raw is None:
        radial_spacing_raw = getattr(polar_grid, "_default_radial_spacing", 100)
    try:
        radial_spacing = float(radial_spacing_raw)
    except Exception:
        radial_spacing = 100.0
    if not math.isfinite(radial_spacing) or radial_spacing <= 0:
        radial_spacing = 100.0

    display_spacing = radial_spacing * scale
    try:
        display_spacing = float(display_spacing)
    except Exception:
        display_spacing = scale * 100
    if not math.isfinite(display_spacing) or display_spacing <= 0:
        display_spacing = scale * 100 if math.isfinite(scale) and scale > 0 else 100.0

    angular_divisions = getattr(polar_grid, "angular_divisions", 12)
    try:
        angular_divisions = int(angular_divisions)
    except Exception:
        angular_divisions = 12
    if angular_divisions <= 0:
        angular_divisions = 12
    angular_step = 360.0 / angular_divisions

    corners = [
        (0 - ox, 0 - oy),
        (width_px - ox, 0 - oy),
        (0 - ox, height_px - oy),
        (width_px - ox, height_px - oy),
    ]
    max_radius_screen = 0.0
    for dx, dy in corners:
        r = math.sqrt(dx * dx + dy * dy)
        if r > max_radius_screen:
            max_radius_screen = r
    max_radius_screen *= 1.1

    return {
        "ox": ox,
        "oy": oy,
        "width_px": width_px,
        "height_px": height_px,
        "scale": scale,
        "display_spacing": display_spacing,
        "max_radius_screen": max_radius_screen,
        "angular_step_degrees": angular_step,
    }


def render_polar_helper(primitives, polar_grid, coordinate_mapper, style):
    """Render a polar coordinate system drawable.

    Args:
        primitives: The renderer primitives interface.
        polar_grid: PolarGrid drawable with width, height, radial spacing.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.
        style: Style dictionary with polar_* settings.
    """
    layout = _compute_polar_layout(polar_grid, coordinate_mapper)
    if layout is None:
        return

    styles = _get_polar_styles(style)

    _render_polar_grid(
        primitives,
        layout["ox"],
        layout["oy"],
        layout["width_px"],
        layout["height_px"],
        layout["scale"],
        layout["display_spacing"],
        layout["max_radius_screen"],
        layout["angular_step_degrees"],
        styles["tick_font_float"],
        styles["font"],
        styles["label_color"],
        styles["label_alignment"],
        styles["axis_stroke"],
        styles["circle_stroke"],
        styles["radial_stroke"],
    )
