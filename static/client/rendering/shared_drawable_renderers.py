"""Shared rendering helpers re-exported for renderer convenience.

This module consolidates rendering helper functions and style classes that are
used by all renderer implementations. It serves as a single import point for
common functionality needed when rendering drawables.

Key Exports:
    - Style classes: FillStyle, StrokeStyle, FontStyle, TextAlignment
    - Primitive interface: RendererPrimitives
    - Geometry helpers: render_point_helper, render_segment_helper, etc.
    - Area builders: build_functions_colored_area, build_segments_colored_area, etc.
    - Font utilities: _coerce_font_size, _compute_zoom_adjusted_font_size

Usage:
    Renderers import from this module to access all shared utilities through
    a single namespace without needing to know the internal organization.
"""

from __future__ import annotations

from rendering.helpers import (
    _coerce_font_size,
    _compute_zoom_adjusted_font_size,
    _filter_valid_points,
    _manages_shape,
    _paths_form_single_loop,
    _points_close,
    build_closed_shape_colored_area,
    build_function_segment_colored_area,
    build_functions_colored_area,
    build_segments_colored_area,
    render_angle_helper,
    render_cartesian_helper,
    render_polar_helper,
    render_circle_arc_helper,
    render_circle_helper,
    render_closed_shape_area_helper,
    render_colored_area_helper,
    render_ellipse_helper,
    render_function_helper,
    render_parametric_function_helper,
    render_bar_helper,
    render_function_segment_area_helper,
    render_functions_bounded_area_helper,
    render_label_helper,
    render_point_helper,
    render_segment_helper,
    render_segments_bounded_area_helper,
    render_vector_helper,
)
from rendering.primitives import (
    FillStyle,
    FontStyle,
    RendererPrimitives,
    StrokeStyle,
    TextAlignment,
)

Point2D = tuple

__all__ = [
    "Point2D",
    "FillStyle",
    "FontStyle",
    "RendererPrimitives",
    "StrokeStyle",
    "TextAlignment",
    "_coerce_font_size",
    "_compute_zoom_adjusted_font_size",
    "_filter_valid_points",
    "_manages_shape",
    "_paths_form_single_loop",
    "_points_close",
    "build_closed_shape_colored_area",
    "build_function_segment_colored_area",
    "build_functions_colored_area",
    "build_segments_colored_area",
    "render_angle_helper",
    "render_cartesian_helper",
    "render_polar_helper",
    "render_circle_arc_helper",
    "render_circle_helper",
    "render_closed_shape_area_helper",
    "render_colored_area_helper",
    "render_ellipse_helper",
    "render_function_helper",
    "render_parametric_function_helper",
    "render_bar_helper",
    "render_function_segment_area_helper",
    "render_functions_bounded_area_helper",
    "render_label_helper",
    "render_point_helper",
    "render_segment_helper",
    "render_segments_bounded_area_helper",
    "render_vector_helper",
]
