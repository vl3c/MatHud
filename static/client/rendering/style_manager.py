from __future__ import annotations

from typing import Any, Dict, Optional

from constants import (
    default_color,
    default_point_size,
    default_font_family,
    default_label_font_size,
    point_label_font_size,
    DEFAULT_ANGLE_ARC_SCREEN_RADIUS,
    DEFAULT_ANGLE_TEXT_ARC_RADIUS_FACTOR,
    DEFAULT_CIRCLE_ARC_COLOR,
    DEFAULT_CIRCLE_ARC_STROKE_WIDTH,
    DEFAULT_CIRCLE_ARC_RADIUS_SCALE,
)


_BASE_STYLE: Dict[str, Any] = {
    "point_color": default_color,
    "point_radius": default_point_size,
    "point_label_font_size": point_label_font_size,
    "point_label_font_family": default_font_family,

    "label_text_color": default_color,
    "label_font_size": default_label_font_size,
    "label_font_family": default_font_family,

    "segment_color": default_color,
    "segment_stroke_width": 1,

    "circle_color": default_color,
    "circle_stroke_width": 1,

    "ellipse_color": default_color,
    "ellipse_stroke_width": 1,

    "vector_color": default_color,
    "vector_tip_size": default_point_size * 4,

    "angle_color": default_color,
    "angle_arc_radius": DEFAULT_ANGLE_ARC_SCREEN_RADIUS,
    "angle_label_font_size": point_label_font_size,
    "angle_text_arc_radius_factor": DEFAULT_ANGLE_TEXT_ARC_RADIUS_FACTOR,
    "angle_label_font_family": default_font_family,

    "circle_arc_color": DEFAULT_CIRCLE_ARC_COLOR,
    "circle_arc_stroke_width": DEFAULT_CIRCLE_ARC_STROKE_WIDTH,
    "circle_arc_radius_scale": DEFAULT_CIRCLE_ARC_RADIUS_SCALE,

    "function_color": default_color,
    "function_stroke_width": 1,
    "function_label_font_size": point_label_font_size,
    "function_label_font_family": default_font_family,

    "area_fill_color": "lightblue",
    "area_opacity": 0.3,

    "cartesian_axis_color": default_color,
    "cartesian_grid_color": "lightgrey",
    "cartesian_tick_size": 3,
    "cartesian_tick_font_size": 8,
    "cartesian_label_color": "grey",
    "cartesian_font_family": default_font_family,

    "fill_style": "rgba(0, 0, 0, 0)",
    "font_family": default_font_family,
}


def _clone_base_style() -> Dict[str, Any]:
    return _BASE_STYLE.copy()


def get_renderer_style(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    style = _clone_base_style()
    if overrides:
        style.update(overrides)
    return style


def get_default_style_value(key: str, default: Optional[Any] = None) -> Any:
    return _BASE_STYLE.get(key, default)

