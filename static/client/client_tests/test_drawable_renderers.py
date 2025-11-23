from __future__ import annotations

import unittest
import math

from coordinate_mapper import CoordinateMapper
from drawables.point import Point
from drawables.segment import Segment
from drawables.vector import Vector
from drawables.angle import Angle
from drawables.triangle import Triangle
from drawables.rectangle import Rectangle
from drawables.ellipse import Ellipse
from drawables.label import Label
from drawables.circle_arc import CircleArc
from drawables.circle import Circle

from rendering import shared_drawable_renderers as shared

from .simple_mock import SimpleMock


class RecordingPrimitives(SimpleMock, shared.RendererPrimitives):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple] = []
        self.shapes: list[str] = []

    def _record(self, op: str, *args, **kwargs) -> None:
        self.calls.append((op, args, kwargs))

    def stroke_line(self, start, end, stroke, *, include_width=True):
        self._record("stroke_line", start, end, stroke, include_width=include_width)

    def stroke_polyline(self, points, stroke):
        self._record("stroke_polyline", points, stroke)

    def stroke_circle(self, center, radius, stroke):
        self._record("stroke_circle", center, radius, stroke)

    def fill_circle(self, center, radius, fill, stroke=None, **kwargs):
        self._record("fill_circle", center, radius, fill, stroke, **kwargs)

    def stroke_ellipse(self, center, radius_x, radius_y, rotation_rad, stroke):
        self._record("stroke_ellipse", center, radius_x, radius_y, rotation_rad, stroke)

    def fill_polygon(self, points, fill, stroke=None, **kwargs):
        self._record("fill_polygon", points, fill, stroke, **kwargs)

    def fill_joined_area(self, forward, reverse, fill):
        self._record("fill_joined_area", forward, reverse, fill)

    def stroke_arc(self, center, radius, start_angle_rad, end_angle_rad, sweep_clockwise, stroke, css_class=None, **kwargs):
        self._record("stroke_arc", center, radius, start_angle_rad, end_angle_rad, sweep_clockwise, stroke, css_class, **kwargs)

    def draw_text(self, text, position, font, color, alignment, style_overrides=None, **kwargs):
        self._record("draw_text", text, position, font, color, alignment, style_overrides, **kwargs)

    def clear_surface(self):
        self._record("clear_surface")

    def resize_surface(self, width, height):
        self._record("resize_surface", width, height)

    def begin_shape(self):
        self._record("begin_shape")
        self.shapes.append("begin")

    def end_shape(self):
        self._record("end_shape")
        self.shapes.append("end")


class TestVectorRenderer(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.primitives = RecordingPrimitives()
        self.style = {
            "segment_stroke_width": 1,
            "vector_tip_size": 8,
            "vector_color": "#0000FF",
        }

    def test_vector_draws_segment_line(self) -> None:
        p1 = Point(1, 2, name="A")
        p2 = Point(4, 6, name="B")
        vector = Vector(p1, p2, color="#FF0000")

        shared.render_vector_helper(self.primitives, vector, self.mapper, self.style)

        line_calls = [call for call in self.primitives.calls if call[0] == "stroke_line"]
        self.assertEqual(len(line_calls), 1)

    def test_vector_draws_arrowhead(self) -> None:
        p1 = Point(0, 0, name="A")
        p2 = Point(3, 0, name="B")
        vector = Vector(p1, p2, color="#00FF00")

        shared.render_vector_helper(self.primitives, vector, self.mapper, self.style)

        polygon_calls = [call for call in self.primitives.calls if call[0] == "fill_polygon"]
        self.assertEqual(len(polygon_calls), 1)
        
        _, args, _ = polygon_calls[0]
        points = args[0]
        self.assertEqual(len(points), 3)

    def test_vector_uses_shape_lifecycle(self) -> None:
        p1 = Point(1, 1, name="A")
        p2 = Point(2, 2, name="B")
        vector = Vector(p1, p2)

        shared.render_vector_helper(self.primitives, vector, self.mapper, self.style)

        self.assertIn("begin", self.primitives.shapes)
        self.assertIn("end", self.primitives.shapes)


class TestAngleRenderer(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.primitives = RecordingPrimitives()
        self.style = {
            "angle_arc_radius": 30,
            "angle_stroke_width": 1,
            "angle_color": "#FF00FF",
            "angle_label_font_size": 12,
            "angle_text_arc_radius_factor": 1.8,
        }

    def test_angle_draws_arc(self) -> None:
        p1 = Point(0, 0, name="V")
        p2 = Point(3, 0, name="A")
        p3 = Point(0, 3, name="B")
        angle = Angle(p1, p2, p3, color="#0000FF")

        shared.render_angle_helper(self.primitives, angle, self.mapper, self.style)

        arc_calls = [call for call in self.primitives.calls if call[0] == "stroke_arc"]
        self.assertEqual(len(arc_calls), 1)

    def test_angle_draws_label(self) -> None:
        p1 = Point(0, 0, name="V")
        p2 = Point(2, 0, name="A")
        p3 = Point(0, 2, name="B")
        angle = Angle(p1, p2, p3)

        shared.render_angle_helper(self.primitives, angle, self.mapper, self.style)

        text_calls = [call for call in self.primitives.calls if call[0] == "draw_text"]
        self.assertGreaterEqual(len(text_calls), 1)
        
        _, args, _ = text_calls[0]
        text = args[0]
        self.assertIn("°", text)

    def test_angle_uses_shape_lifecycle(self) -> None:
        p1 = Point(0, 0, name="V")
        p2 = Point(1, 0, name="A")
        p3 = Point(0, 1, name="B")
        angle = Angle(p1, p2, p3)

        shared.render_angle_helper(self.primitives, angle, self.mapper, self.style)

        self.assertIn("begin", self.primitives.shapes)
        self.assertIn("end", self.primitives.shapes)


class TestCircleArcRenderer(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.primitives = RecordingPrimitives()
        self.style = {
            "circle_arc_color": "#00AAFF",
            "circle_arc_stroke_width": 2,
            "circle_arc_radius_scale": 1.0,
        }

    def test_circle_arc_draws_stroke_arc(self) -> None:
        center_point = Point(0, 0, name="O")
        circle = Circle(center_point, radius=5)
        point_a = Point(5, 0, name="A")
        point_b = Point(0, 5, name="B")
        arc = CircleArc(point_a, point_b, center_x=0, center_y=0, radius=5, circle=circle)

        shared.render_circle_arc_helper(self.primitives, arc, self.mapper, self.style)
        stroke_calls = [call for call in self.primitives.calls if call[0] == "stroke_arc"]
        self.assertEqual(len(stroke_calls), 1)

    def test_circle_arc_respects_major_flag(self) -> None:
        point_a = Point(5, 0, name="A")
        point_b = Point(-5, 0, name="B")
        arc = CircleArc(point_a, point_b, center_x=0, center_y=0, radius=5, use_major_arc=True)

        shared.render_circle_arc_helper(self.primitives, arc, self.mapper, self.style)
        stroke_calls = [call for call in self.primitives.calls if call[0] == "stroke_arc"]
        self.assertEqual(len(stroke_calls), 1)

    def test_circle_arc_minor_sweep_uses_counterclockwise_path(self) -> None:
        point_a = Point(5, 0, name="A")
        point_b = Point(0, 5, name="B")
        arc = CircleArc(point_a, point_b, center_x=0, center_y=0, radius=5)

        shared.render_circle_arc_helper(self.primitives, arc, self.mapper, self.style)
        stroke_call = next(call for call in self.primitives.calls if call[0] == "stroke_arc")
        _, args, _ = stroke_call
        _, _, start_angle, end_angle, sweep_clockwise, *_ = args
        self.assertFalse(sweep_clockwise)
        delta = abs(end_angle - start_angle)
        self.assertTrue(math.isclose(delta, math.pi / 2, rel_tol=1e-6))

    def test_circle_arc_major_sweep_goes_clockwise(self) -> None:
        point_a = Point(5, 0, name="A")
        point_b = Point(0, 5, name="B")
        arc = CircleArc(point_a, point_b, center_x=0, center_y=0, radius=5, use_major_arc=True)

        shared.render_circle_arc_helper(self.primitives, arc, self.mapper, self.style)
        stroke_call = next(call for call in self.primitives.calls if call[0] == "stroke_arc")
        _, args, _ = stroke_call
        _, _, start_angle, end_angle, sweep_clockwise, *_ = args
        self.assertTrue(sweep_clockwise)
        delta = abs(end_angle - start_angle)
        self.assertTrue(math.isclose(delta, 3 * math.pi / 2, rel_tol=1e-6))
class TestEllipseRenderer(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.primitives = RecordingPrimitives()
        self.style = {
            "ellipse_stroke_width": 1,
            "ellipse_color": "#FF00FF",
        }

    def test_ellipse_draws_ellipse_primitive(self) -> None:
        center = Point(5, 5, name="O")
        ellipse = Ellipse(center, radius_x=3, radius_y=2, rotation_angle=45, color="#0000FF")

        shared.render_ellipse_helper(self.primitives, ellipse, self.mapper, self.style)

        ellipse_calls = [call for call in self.primitives.calls if call[0] == "stroke_ellipse"]
        self.assertEqual(len(ellipse_calls), 1)

    def test_ellipse_converts_rotation_to_radians(self) -> None:
        center = Point(0, 0, name="O")
        ellipse = Ellipse(center, radius_x=4, radius_y=2, rotation_angle=90)

        shared.render_ellipse_helper(self.primitives, ellipse, self.mapper, self.style)

        ellipse_calls = [call for call in self.primitives.calls if call[0] == "stroke_ellipse"]
        self.assertEqual(len(ellipse_calls), 1)
        
        _, args, _ = ellipse_calls[0]
        rotation_rad = args[3]
        
        expected_rad = -math.radians(90)
        self.assertAlmostEqual(rotation_rad, expected_rad, places=5)

    def test_ellipse_uses_shape_lifecycle(self) -> None:
        center = Point(2, 3, name="O")
        ellipse = Ellipse(center, radius_x=2, radius_y=1)

        shared.render_ellipse_helper(self.primitives, ellipse, self.mapper, self.style)

        self.assertIn("begin", self.primitives.shapes)
        self.assertIn("end", self.primitives.shapes)


class TestLabelRenderer(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.primitives = RecordingPrimitives()
        self.style = {
            "label_font_size": 14,
            "label_text_color": "#000000",
        }

    def test_label_draws_text(self) -> None:
        label = Label(5, 10, "Test Label", font_size=12, color="#FF0000")

        shared.render_label_helper(self.primitives, label, self.mapper, self.style)

        text_calls = [call for call in self.primitives.calls if call[0] == "draw_text"]
        self.assertGreaterEqual(len(text_calls), 1)
        
        _, args, _ = text_calls[0]
        text = args[0]
        self.assertEqual(text, "Test Label")

    def test_label_handles_multiline_text(self) -> None:
        label = Label(3, 4, "Line 1\nLine 2\nLine 3")

        shared.render_label_helper(self.primitives, label, self.mapper, self.style)

        text_calls = [call for call in self.primitives.calls if call[0] == "draw_text"]
        self.assertEqual(len(text_calls), 3)

    def test_label_respects_font_size(self) -> None:
        label = Label(1, 2, "Text", font_size=18)

        shared.render_label_helper(self.primitives, label, self.mapper, self.style)

        text_calls = [call for call in self.primitives.calls if call[0] == "draw_text"]
        self.assertGreaterEqual(len(text_calls), 1)
        
        _, args, _ = text_calls[0]
        font = args[2]
        self.assertEqual(font.size, 18)

    def test_label_zoom_adjusted_font_size(self) -> None:
        from drawables.position import Position
        self.mapper.apply_zoom(0.5, Position(320, 240))
        
        label = Label(2, 3, "Zoom Test", font_size=16, reference_scale_factor=1.0)

        shared.render_label_helper(self.primitives, label, self.mapper, self.style)

        text_calls = [call for call in self.primitives.calls if call[0] == "draw_text"]
        
        if len(text_calls) > 0:
            _, args, _ = text_calls[0]
            font = args[2]
            self.assertLess(font.size, 16)


class TestRendererEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.primitives = RecordingPrimitives()
        self.style = {}

    def test_segment_with_same_endpoints_does_not_crash(self) -> None:
        p = Point(5, 5, name="P")
        segment = Segment(p, p)

        try:
            shared.render_segment_helper(self.primitives, segment, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Segment with same endpoints raised exception: {e}")

    def test_zero_radius_circle_does_not_crash(self) -> None:
        from drawables.circle import Circle
        center = Point(0, 0, name="O")
        circle = Circle(center, radius=0)

        try:
            shared.render_circle_helper(self.primitives, circle, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Zero radius circle raised exception: {e}")

    def test_ellipse_with_zero_radius_does_not_crash(self) -> None:
        center = Point(1, 1, name="O")
        ellipse = Ellipse(center, radius_x=0, radius_y=5)

        try:
            shared.render_ellipse_helper(self.primitives, ellipse, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Ellipse with zero radius raised exception: {e}")

    def test_label_with_empty_text_does_not_crash(self) -> None:
        label = Label(2, 2, "")

        try:
            shared.render_label_helper(self.primitives, label, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Empty label raised exception: {e}")


__all__ = [
    "TestVectorRenderer",
    "TestAngleRenderer",
    "TestEllipseRenderer",
    "TestLabelRenderer",
    "TestRendererEdgeCases",
]

