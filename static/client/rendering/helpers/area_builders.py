from __future__ import annotations

from rendering.renderables import (
    ClosedShapeAreaRenderable,
    FunctionSegmentAreaRenderable,
    FunctionsBoundedAreaRenderable,
    SegmentsBoundedAreaRenderable,
)


def build_functions_colored_area(area_model, coordinate_mapper):
    try:
        renderable = FunctionsBoundedAreaRenderable(area_model, coordinate_mapper)
        return renderable.build_screen_area()
    except Exception:
        return None


def build_function_segment_colored_area(area_model, coordinate_mapper, *, num_points=100):
    try:
        renderable = FunctionSegmentAreaRenderable(area_model, coordinate_mapper)
        return renderable.build_screen_area(num_points=num_points)
    except Exception:
        return None


def build_segments_colored_area(area_model, coordinate_mapper):
    try:
        renderable = SegmentsBoundedAreaRenderable(area_model, coordinate_mapper)
        return renderable.build_screen_area()
    except Exception:
        return None


def build_closed_shape_colored_area(area_model, coordinate_mapper):
    try:
        renderable = ClosedShapeAreaRenderable(area_model, coordinate_mapper)
        return renderable.build_screen_area()
    except Exception:
        return None

