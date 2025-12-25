from __future__ import annotations

import math

from constants import default_font_family
from rendering.helpers.screen_offset_label_helper import draw_point_style_label_with_coords
from rendering.helpers.world_label_helper import get_label_lines, render_world_label_at_screen_point
from rendering.primitives import FontStyle, TextAlignment


def _normalize_font_size(value, default_value):
    try:
        size_float = float(value)
    except Exception:
        return default_value
    if not math.isfinite(size_float) or size_float <= 0:
        return default_value
    if size_float.is_integer():
        return int(size_float)
    return size_float


def _screen_offset_get_position(label):
    return getattr(label, "position", None)


def _screen_offset_try_math_to_screen(position, coordinate_mapper):
    try:
        return coordinate_mapper.math_to_screen(position.x, position.y)
    except Exception:
        return None


def _screen_offset_compute_offset(mode, style):
    """Return (radius, offset_x, offset_y)."""
    radius = 0.0
    if bool(getattr(mode, "offset_from_point_radius", True)):
        radius_raw = style.get("point_radius", 0) or 0
        try:
            radius = float(radius_raw)
        except Exception:
            radius = 0.0
        return radius, float(radius), float(-radius)
    return (
        radius,
        float(getattr(mode, "offset_px_x", 0.0) or 0.0),
        float(getattr(mode, "offset_px_y", 0.0) or 0.0),
    )


def _screen_offset_compute_font_size(label, mode, style):
    font_size_source = str(getattr(mode, "font_size_source", "label") or "label")
    if font_size_source == "style":
        font_size_key = str(getattr(mode, "font_size_key", "point_label_font_size") or "point_label_font_size")
        font_size_value = style.get(font_size_key, 10)
        return _normalize_font_size(font_size_value, 10)
    font_size_value = getattr(label, "font_size", style.get("label_font_size", 14))
    return _normalize_font_size(font_size_value, 14)


def _screen_offset_compute_font_family(mode, style):
    font_family_key = str(getattr(mode, "font_family_key", "label_font_family") or "label_font_family")
    return style.get(font_family_key, style.get("font_family", default_font_family))


def _screen_offset_build_font(label, mode, style):
    font_size = _screen_offset_compute_font_size(label, mode, style)
    font_family = _screen_offset_compute_font_family(mode, style)
    return font_size, FontStyle(family=font_family, size=font_size)


def _screen_offset_color(label, style):
    return str(getattr(label, "color", style.get("label_text_color", "#000")))


def _screen_offset_alignment():
    return TextAlignment(horizontal="left", vertical="alphabetic")


def _screen_offset_rotation_degrees(label):
    try:
        rotation_degrees = float(getattr(label, "rotation_degrees", 0.0))
    except Exception:
        rotation_degrees = 0.0
    if not math.isfinite(rotation_degrees):
        rotation_degrees = 0.0
    return rotation_degrees


def _screen_offset_style_overrides(mode):
    if not bool(getattr(mode, "non_selectable", False)):
        return None
    return {
        "user-select": "none",
        "-webkit-user-select": "none",
        "-moz-user-select": "none",
        "-ms-user-select": "none",
    }


def _screen_offset_text_format(mode):
    return str(getattr(mode, "text_format", "text_only") or "text_only")


def _screen_offset_draw_text_with_anchor_coords(
    primitives,
    *,
    label,
    position,
    screen_x,
    screen_y,
    radius,
    color,
    style,
    rotation_degrees,
    mode,
):
    base_text = str(getattr(label, "text", "") or "")
    if not base_text:
        return

    metadata_overrides = None
    if math.isfinite(rotation_degrees) and rotation_degrees != 0.0:
        metadata_overrides = {"label": {"rotation_degrees": rotation_degrees}}

    try:
        layout_group = id(label)
    except Exception:
        layout_group = None

    draw_point_style_label_with_coords(
        primitives,
        anchor_screen_x=float(screen_x),
        anchor_screen_y=float(screen_y),
        anchor_math_x=float(position.x),
        anchor_math_y=float(position.y),
        label=base_text,
        radius=float(radius),
        color=color,
        style=style,
        coord_precision=int(getattr(mode, "coord_precision", 3) or 3),
        non_selectable=bool(getattr(mode, "non_selectable", False)),
        layout_group=layout_group,
        metadata_overrides=metadata_overrides,
    )


def _screen_offset_draw_text_lines(
    primitives,
    *,
    label,
    position,
    screen_x,
    screen_y,
    offset_x,
    offset_y,
    font,
    font_size,
    color,
    alignment,
    style_overrides,
    rotation_degrees,
):
    text_value = str(getattr(label, "text", "") or "")
    if not text_value:
        return

    lines = get_label_lines(label)
    line_count = len(lines)
    max_line_len = 0
    for line in lines:
        try:
            line_len = len(str(line))
        except Exception:
            line_len = 0
        if line_len > max_line_len:
            max_line_len = line_len

    try:
        layout_group = id(label)
    except Exception:
        layout_group = None

    line_height = float(font_size) * 1.2
    base_draw_x = screen_x + offset_x
    base_draw_y = screen_y + offset_y

    for index, line in enumerate(lines):
        current_text = str(line)
        line_offset_y = index * line_height
        metadata = {
            "point_label": {
                "math_position": (float(position.x), float(position.y)),
                "screen_offset": (float(offset_x), float(offset_y + line_offset_y)),
                "layout_group": layout_group,
                "layout_line_index": int(index),
                "layout_line_count": int(line_count),
                "layout_max_line_len": int(max_line_len),
            },
            "label": {"rotation_degrees": rotation_degrees},
        }
        primitives.draw_text(
            current_text,
            (base_draw_x, base_draw_y + line_offset_y),
            font,
            color,
            alignment,
            style_overrides,
            screen_space=True,
            metadata=metadata,
        )


def _render_screen_offset_label(primitives, label, coordinate_mapper, style, mode):
    position = _screen_offset_get_position(label)
    if position is None:
        return

    screen_point = _screen_offset_try_math_to_screen(position, coordinate_mapper)
    if not screen_point:
        return
    screen_x, screen_y = screen_point

    radius, offset_x, offset_y = _screen_offset_compute_offset(mode, style)

    # Keep these computed even if some branches do not use them; this keeps the
    # behavior stable and makes debugging easier.
    font_size, font = _screen_offset_build_font(label, mode, style)
    color = _screen_offset_color(label, style)
    alignment = _screen_offset_alignment()
    rotation_degrees = _screen_offset_rotation_degrees(label)
    style_overrides = _screen_offset_style_overrides(mode)

    text_format = _screen_offset_text_format(mode)
    if text_format == "text_with_anchor_coords":
        _screen_offset_draw_text_with_anchor_coords(
            primitives,
            label=label,
            position=position,
            screen_x=screen_x,
            screen_y=screen_y,
            radius=radius,
            color=color,
            style=style,
            rotation_degrees=rotation_degrees,
            mode=mode,
        )
        return

    _screen_offset_draw_text_lines(
        primitives,
        label=label,
        position=position,
        screen_x=screen_x,
        screen_y=screen_y,
        offset_x=offset_x,
        offset_y=offset_y,
        font=font,
        font_size=font_size,
        color=color,
        alignment=alignment,
        style_overrides=style_overrides,
        rotation_degrees=rotation_degrees,
    )


def render_label_helper(primitives, label, coordinate_mapper, style):
    try:
        if hasattr(label, "visible") and not bool(getattr(label, "visible")):
            return
    except Exception:
        return
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

    render_mode = getattr(label, "render_mode", None)
    if getattr(render_mode, "kind", None) == "screen_offset":
        _render_screen_offset_label(primitives, label, coordinate_mapper, style, render_mode)
        return

    render_world_label_at_screen_point(
        primitives,
        label,
        coordinate_mapper,
        style,
        screen_x=screen_x,
        screen_y=screen_y,
    )

