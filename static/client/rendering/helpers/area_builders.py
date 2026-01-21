"""Area builder functions for creating colored area screen coordinates.

This module provides factory functions that convert area models into
screen-space polygon data ready for rendering.

Key Features:
    - Function-bounded area construction between two curves
    - Function-segment hybrid area for mixed boundaries
    - Segment-only bounded areas for polygonal regions
    - Closed shape areas from arbitrary drawable paths
    - Safe exception handling with None fallback
"""

from __future__ import annotations

from rendering.renderables import (
    ClosedShapeAreaRenderable,
    FunctionSegmentAreaRenderable,
    FunctionsBoundedAreaRenderable,
    SegmentsBoundedAreaRenderable,
)


def build_functions_colored_area(area_model, coordinate_mapper):
    """Build screen coordinates for a function-bounded colored area.

    Args:
        area_model: FunctionsBoundedColoredArea with upper/lower functions.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.

    Returns:
        ColoredArea with forward/reverse points, or None on failure.
    """
    try:
        renderable = FunctionsBoundedAreaRenderable(area_model, coordinate_mapper)
        return renderable.build_screen_area()
    except Exception:
        return None


def build_function_segment_colored_area(area_model, coordinate_mapper, *, num_points=100):
    """Build screen coordinates for a function-segment bounded area.

    Args:
        area_model: FunctionSegmentBoundedColoredArea with function and segment.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.
        num_points: Number of sample points along the function curve.

    Returns:
        ColoredArea with forward/reverse points, or None on failure.
    """
    try:
        renderable = FunctionSegmentAreaRenderable(area_model, coordinate_mapper)
        return renderable.build_screen_area(num_points=num_points)
    except Exception:
        return None


def build_segments_colored_area(area_model, coordinate_mapper):
    """Build screen coordinates for a segment-bounded colored area.

    Args:
        area_model: SegmentsBoundedColoredArea with boundary segments.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.

    Returns:
        ColoredArea with forward/reverse points, or None on failure.
    """
    try:
        renderable = SegmentsBoundedAreaRenderable(area_model, coordinate_mapper)
        return renderable.build_screen_area()
    except Exception:
        return None


def build_closed_shape_colored_area(area_model, coordinate_mapper):
    """Build screen coordinates for a closed shape colored area.

    Args:
        area_model: ClosedShapeColoredArea with shape reference.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.

    Returns:
        ColoredArea with forward/reverse points, or None on failure.
    """
    try:
        renderable = ClosedShapeAreaRenderable(area_model, coordinate_mapper)
        return renderable.build_screen_area()
    except Exception:
        return None

