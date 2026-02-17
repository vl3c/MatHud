"""Rendering helper functions for drawable types.

This package provides specialized rendering helper functions for each drawable
type. Each helper takes a primitives interface, the drawable, a coordinate
mapper, and style dictionary to produce the visual representation.

Key Categories:
    - Geometry helpers: point, segment, circle, ellipse, vector, angle
    - Function helpers: function, parametric_function, piecewise_function
    - Area helpers: colored_area builders for shaded regions
    - Grid helpers: cartesian, polar coordinate systems
    - Font helpers: zoom-adjusted font sizing

Usage:
    Helper functions are typically called by the cached_render_plan module
    during plan building, or directly by renderers for immediate drawing.
"""

from __future__ import annotations

from rendering.helpers.font_helpers import (
    _coerce_font_size,
    _compute_zoom_adjusted_font_size,
)
from rendering.helpers.shape_decorator import _manages_shape
from rendering.helpers.point_renderer import render_point_helper
from rendering.helpers.segment_renderer import render_segment_helper
from rendering.helpers.ellipse_renderer import render_ellipse_helper
from rendering.helpers.circle_renderer import render_circle_helper
from rendering.helpers.circle_arc_renderer import render_circle_arc_helper
from rendering.helpers.vector_renderer import render_vector_helper
from rendering.helpers.label_renderer import render_label_helper
from rendering.helpers.angle_renderer import render_angle_helper
from rendering.helpers.cartesian_renderer import render_cartesian_helper
from rendering.helpers.polar_renderer import render_polar_helper
from rendering.helpers.function_renderer import render_function_helper
from rendering.helpers.parametric_function_renderer import render_parametric_function_helper
from rendering.helpers.bar_renderer import render_bar_helper
from rendering.helpers.colored_area_renderer import (
    render_colored_area_helper,
    render_functions_bounded_area_helper,
    render_function_segment_area_helper,
    render_segments_bounded_area_helper,
    render_closed_shape_area_helper,
    _points_close,
    _paths_form_single_loop,
    _filter_valid_points,
)
from rendering.helpers.area_builders import (
    build_functions_colored_area,
    build_function_segment_colored_area,
    build_segments_colored_area,
    build_closed_shape_colored_area,
)

__all__ = [
    "_coerce_font_size",
    "_compute_zoom_adjusted_font_size",
    "_manages_shape",
    "render_point_helper",
    "render_segment_helper",
    "render_ellipse_helper",
    "render_circle_helper",
    "render_circle_arc_helper",
    "render_vector_helper",
    "render_label_helper",
    "render_angle_helper",
    "render_cartesian_helper",
    "render_polar_helper",
    "render_function_helper",
    "render_parametric_function_helper",
    "render_bar_helper",
    "render_colored_area_helper",
    "render_functions_bounded_area_helper",
    "render_function_segment_area_helper",
    "render_segments_bounded_area_helper",
    "render_closed_shape_area_helper",
    "_points_close",
    "_paths_form_single_loop",
    "_filter_valid_points",
    "build_functions_colored_area",
    "build_function_segment_colored_area",
    "build_segments_colored_area",
    "build_closed_shape_colored_area",
]
