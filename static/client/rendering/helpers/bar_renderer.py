"""Bar rendering helper for drawing bar chart elements.

This module provides the render_bar_helper function that renders
bar drawables as filled rectangles with optional labels.

Key Features:
    - Filled rectangle rendering with configurable colors
    - Optional stroke outline on bars
    - Above and below bar labels with automatic positioning
    - Point label metadata for zoom-stable label reprojection
    - Padding configuration for label spacing
"""

from __future__ import annotations

from rendering.primitives import FillStyle, FontStyle, StrokeStyle, TextAlignment


def _resolve_label_font(style):
    """Build a FontStyle from style dictionary settings.

    Args:
        style: Style dictionary or non-dict value.

    Returns:
        FontStyle with family, size, and optional weight.
    """
    font_size = style.get("label_font_size", 12) if isinstance(style, dict) else 12
    family = style.get("label_font_family", "Arial") if isinstance(style, dict) else "Arial"
    weight = style.get("label_font_weight", None) if isinstance(style, dict) else None
    return FontStyle(family, font_size, weight)


def render_bar_helper(primitives, bar, coordinate_mapper, style):
    """Render a bar drawable with optional labels.

    Args:
        primitives: The renderer primitives interface.
        bar: Bar drawable with x_left, x_right, y_bottom, y_top bounds.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.
        style: Style dictionary with fill, stroke, and label settings.
    """
    try:
        x_left = float(getattr(bar, "x_left", 0.0))
        x_right = float(getattr(bar, "x_right", 0.0))
        y_bottom = float(getattr(bar, "y_bottom", 0.0))
        y_top = float(getattr(bar, "y_top", 0.0))
    except Exception:
        return

    if x_left == x_right:
        return
    if y_bottom == y_top:
        return

    try:
        p1 = coordinate_mapper.math_to_screen(x_left, y_bottom)
        p2 = coordinate_mapper.math_to_screen(x_right, y_bottom)
        p3 = coordinate_mapper.math_to_screen(x_right, y_top)
        p4 = coordinate_mapper.math_to_screen(x_left, y_top)
    except Exception:
        return

    if not p1 or not p2 or not p3 or not p4:
        return

    fill_color = getattr(bar, "fill_color", None)
    if fill_color is None or not str(fill_color).strip():
        fill_color = style.get("default_area_fill_color", "#88aaff") if isinstance(style, dict) else "#88aaff"
    fill_opacity = getattr(bar, "fill_opacity", None)

    fill = FillStyle(color=str(fill_color), opacity=fill_opacity)

    stroke_color = getattr(bar, "color", None)
    stroke_width = style.get("segment_width", 2) if isinstance(style, dict) else 2
    if stroke_color is None or not str(stroke_color).strip():
        stroke = None
    else:
        stroke = StrokeStyle(color=str(stroke_color), width=stroke_width)

    primitives.fill_polygon((p1, p2, p3, p4), fill, stroke, screen_space=True)

    label_above_text = getattr(bar, "label_above_text", None)
    if label_above_text is None:
        label_above_text = getattr(bar, "label_text", None)
    label_below_text = getattr(bar, "label_below_text", None)

    try:
        cx = (float(p1[0]) + float(p2[0])) / 2.0
        ys = (float(p1[1]), float(p2[1]), float(p3[1]), float(p4[1]))
        top_y = min(ys)
        bottom_y = max(ys)
    except Exception:
        return

    # Use point_label metadata so the optimized renderer can reproject labels with a
    # constant pixel offset on zoom/pan (same approach as segment AttachedLabel).
    try:
        mx = (x_left + x_right) / 2.0
        my_top = max(y_bottom, y_top)
        my_bottom = min(y_bottom, y_top)
    except Exception:
        return

    padding_px = style.get("bar_label_padding_px", 6) if isinstance(style, dict) else 6
    font = _resolve_label_font(style)
    color = style.get("label_color", "#000") if isinstance(style, dict) else "#000"

    if label_above_text is not None:
        above = str(label_above_text)
        if above.strip():
            above_meta = {
                "point_label": {
                    "math_position": (float(mx), float(my_top)),
                    "screen_offset": (0.0, float(-padding_px)),
                }
            }
            primitives.draw_text(
                above,
                (cx, top_y - float(padding_px)),
                font,
                str(color),
                TextAlignment(horizontal="center", vertical="bottom"),
                screen_space=True,
                metadata=above_meta,
            )

    if label_below_text is not None:
        below = str(label_below_text)
        if below.strip():
            below_meta = {
                "point_label": {
                    "math_position": (float(mx), float(my_bottom)),
                    "screen_offset": (0.0, float(padding_px)),
                }
            }
            primitives.draw_text(
                below,
                (cx, bottom_y + float(padding_px)),
                font,
                str(color),
                TextAlignment(horizontal="center", vertical="top"),
                screen_space=True,
                metadata=below_meta,
            )


