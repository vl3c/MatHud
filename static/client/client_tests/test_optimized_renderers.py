from __future__ import annotations

import unittest

from .simple_mock import SimpleMock

from coordinate_mapper import CoordinateMapper
from rendering import shared_drawable_renderers as shared
from rendering.optimized_drawable_renderers import build_plan_for_cartesian, build_plan_for_drawable
from rendering.style_manager import get_renderer_style
from rendering.shared_drawable_renderers import RendererPrimitives
from cartesian_system_2axis import Cartesian2Axis
from drawables.point import Point
from drawables.segment import Segment
from drawables.circle import Circle


def _normalize_point(value) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))


def _serialize_stroke(stroke) -> dict[str, object]:
    return {
        "color": stroke.color,
        "width": float(stroke.width),
        "line_join": stroke.line_join,
        "line_cap": stroke.line_cap,
    }


def _serialize_fill(fill) -> dict[str, object]:
    return {
        "color": fill.color,
        "opacity": None if fill.opacity is None else float(fill.opacity),
    }


def _serialize_font(font) -> dict[str, object]:
    return {
        "family": font.family,
        "size": font.size,
        "weight": font.weight,
    }


def _serialize_alignment(alignment) -> dict[str, object]:
    return {
        "horizontal": alignment.horizontal,
        "vertical": alignment.vertical,
    }


def _serialize_style_overrides(overrides) -> dict[str, object]:
    if not overrides:
        return {}
    return dict(sorted(overrides.items()))


class RecordingPrimitives(SimpleMock, RendererPrimitives):
    def __init__(self) -> None:
        super().__init__()
        self.operations: list[tuple[str, tuple, dict]] = []

    def _record(self, op: str, *args, **kwargs) -> None:
        self.operations.append((op, args, kwargs))

    def stroke_line(self, start, end, stroke, *, include_width=True):
        self._record("stroke_line", _normalize_point(start), _normalize_point(end), _serialize_stroke(stroke), include_width=include_width)

    def stroke_polyline(self, points, stroke):
        normalized = tuple(_normalize_point(pt) for pt in points)
        self._record("stroke_polyline", normalized, _serialize_stroke(stroke))

    def stroke_circle(self, center, radius, stroke):
        self._record("stroke_circle", _normalize_point(center), float(radius), _serialize_stroke(stroke))

    def fill_circle(self, center, radius, fill, stroke=None):
        stroke_serialized = _serialize_stroke(stroke) if stroke else None
        self._record("fill_circle", _normalize_point(center), float(radius), _serialize_fill(fill), stroke_serialized)

    def stroke_ellipse(self, center, radius_x, radius_y, rotation_rad, stroke):
        self._record(
            "stroke_ellipse",
            _normalize_point(center),
            float(radius_x),
            float(radius_y),
            float(rotation_rad),
            _serialize_stroke(stroke),
        )

    def fill_polygon(self, points, fill, stroke=None):
        normalized = tuple(_normalize_point(pt) for pt in points)
        stroke_serialized = _serialize_stroke(stroke) if stroke else None
        self._record("fill_polygon", normalized, _serialize_fill(fill), stroke_serialized)

    def fill_joined_area(self, forward, reverse, fill):
        forward_norm = tuple(_normalize_point(pt) for pt in forward)
        reverse_norm = tuple(_normalize_point(pt) for pt in reverse)
        self._record("fill_joined_area", forward_norm, reverse_norm, _serialize_fill(fill))

    def stroke_arc(self, center, radius, start_angle_rad, end_angle_rad, sweep_clockwise, stroke, css_class=None):
        self._record(
            "stroke_arc",
            _normalize_point(center),
            float(radius),
            float(start_angle_rad),
            float(end_angle_rad),
            bool(sweep_clockwise),
            _serialize_stroke(stroke),
            css_class,
        )

    def draw_text(self, text, position, font, color, alignment, style_overrides=None):
        self._record(
            "draw_text",
            text,
            _normalize_point(position),
            _serialize_font(font),
            color,
            _serialize_alignment(alignment),
            _serialize_style_overrides(style_overrides),
        )

    def clear_surface(self) -> None:
        return None

    def resize_surface(self, width, height) -> None:
        return None

    def execute_optimized(self, command) -> None:
        handler = getattr(self, command.op)
        handler(*command.args, **command.kwargs)


HELPERS = {
    "Point": shared.render_point_helper,
    "Segment": shared.render_segment_helper,
    "Circle": shared.render_circle_helper,
}


class TestOptimizedRendererParity(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.style = get_renderer_style()

    def _assert_drawable_parity(self, drawable) -> None:
        class_name = drawable.get_class_name() if hasattr(drawable, "get_class_name") else drawable.__class__.__name__
        helper = HELPERS[class_name]

        legacy_primitives = RecordingPrimitives()
        helper(legacy_primitives, drawable, self.mapper, self.style)
        legacy_ops = legacy_primitives.operations

        plan = build_plan_for_drawable(drawable, self.mapper, self.style)
        self.assertIsNotNone(plan)
        optimized_primitives = RecordingPrimitives()
        plan.apply(optimized_primitives)
        optimized_ops = optimized_primitives.operations

        self.assertEqual(optimized_ops, legacy_ops)

    def test_point_matches_legacy(self) -> None:
        point = Point(2.5, -1.75, name="P")
        self._assert_drawable_parity(point)

    def test_segment_matches_legacy(self) -> None:
        p1 = Point(-5.0, 3.0, name="A")
        p2 = Point(4.0, -2.0, name="B")
        segment = Segment(p1, p2)
        self._assert_drawable_parity(segment)

    def test_circle_matches_legacy(self) -> None:
        center = Point(0.0, 0.0, name="O")
        circle = Circle(center, radius=6.5)
        self._assert_drawable_parity(circle)

    def test_plan_visibility_culling(self) -> None:
        point = Point(0.0, 0.0, name="P_cull")
        plan = build_plan_for_drawable(point, self.mapper, self.style)
        self.assertIsNotNone(plan)
        if plan is None:
            return
        viewport_width = self.mapper.canvas_width
        viewport_height = self.mapper.canvas_height
        self.assertTrue(plan.is_visible(viewport_width, viewport_height))
        shifted_mapper = CoordinateMapper(viewport_width, viewport_height)
        shifted_mapper.apply_pan(2000, 2000)
        plan.update_map_state(
            {
                "scale": shifted_mapper.scale_factor,
                "offset_x": shifted_mapper.offset.x,
                "offset_y": shifted_mapper.offset.y,
                "origin_x": shifted_mapper.origin.x,
                "origin_y": shifted_mapper.origin.y,
            }
        )
        self.assertFalse(plan.is_visible(viewport_width, viewport_height))

    def test_cartesian_plan_matches_legacy(self) -> None:
        cartesian = Cartesian2Axis(self.mapper)
        legacy_primitives = RecordingPrimitives()
        shared.render_cartesian_helper(legacy_primitives, cartesian, self.mapper, self.style)
        legacy_ops = legacy_primitives.operations

        plan = build_plan_for_cartesian(cartesian, self.mapper, self.style)
        optimized_primitives = RecordingPrimitives()
        plan.apply(optimized_primitives)
        optimized_ops = optimized_primitives.operations

        self.assertEqual(optimized_ops, legacy_ops)


if __name__ == "__main__":
    unittest.main()

