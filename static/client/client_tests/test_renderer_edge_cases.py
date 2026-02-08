from __future__ import annotations

import unittest
from types import SimpleNamespace

from coordinate_mapper import CoordinateMapper
from drawables.point import Point
from drawables.segment import Segment
from drawables.circle import Circle
from drawables.vector import Vector
from drawables.ellipse import Ellipse
from drawables.label import Label

from rendering import shared_drawable_renderers as shared

from .simple_mock import SimpleMock


class RecordingPrimitives(SimpleMock, shared.RendererPrimitives):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple] = []
        self.errors: list[str] = []

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

    def end_shape(self):
        self._record("end_shape")


class TestPointEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.primitives = RecordingPrimitives()
        self.style = {"point_radius": 4, "point_color": "#000"}

    def test_point_with_nan_coordinates_does_not_crash(self) -> None:
        point = Point(float('nan'), 5.0, name="P")

        try:
            shared.render_point_helper(self.primitives, point, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Point with NaN coordinate raised exception: {e}")

    def test_point_with_infinity_coordinates_does_not_crash(self) -> None:
        point = Point(float('inf'), 10.0, name="P")

        try:
            shared.render_point_helper(self.primitives, point, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Point with Infinity coordinate raised exception: {e}")

    def test_point_with_negative_infinity_does_not_crash(self) -> None:
        point = Point(5.0, float('-inf'), name="P")

        try:
            shared.render_point_helper(self.primitives, point, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Point with -Infinity coordinate raised exception: {e}")

    def test_point_with_very_large_coordinates_does_not_crash(self) -> None:
        point = Point(1e100, 1e100, name="P")

        try:
            shared.render_point_helper(self.primitives, point, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Point with very large coordinates raised exception: {e}")

    def test_point_with_zero_radius_in_style_does_not_crash(self) -> None:
        point = Point(5, 5, name="P")
        style = {"point_radius": 0, "point_color": "#000"}

        try:
            shared.render_point_helper(self.primitives, point, self.mapper, style)
        except Exception as e:
            self.fail(f"Point with zero radius raised exception: {e}")

    def test_point_with_negative_radius_in_style_does_not_crash(self) -> None:
        point = Point(5, 5, name="P")
        style = {"point_radius": -5, "point_color": "#000"}

        try:
            shared.render_point_helper(self.primitives, point, self.mapper, style)
        except Exception as e:
            self.fail(f"Point with negative radius raised exception: {e}")


class TestSegmentEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.primitives = RecordingPrimitives()
        self.style = {"segment_stroke_width": 1, "segment_color": "#000"}

    def test_segment_with_nan_endpoint_does_not_crash(self) -> None:
        p1 = Point(0, 0, name="A")
        p2 = Point(float('nan'), 5, name="B")
        segment = Segment(p1, p2)

        try:
            shared.render_segment_helper(self.primitives, segment, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Segment with NaN endpoint raised exception: {e}")

    def test_segment_with_infinity_endpoint_does_not_crash(self) -> None:
        p1 = Point(0, 0, name="A")
        p2 = Point(float('inf'), float('inf'), name="B")
        segment = Segment(p1, p2)

        try:
            shared.render_segment_helper(self.primitives, segment, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Segment with Infinity endpoint raised exception: {e}")

    def test_segment_with_coincident_points_does_not_crash(self) -> None:
        p = Point(5, 5, name="P")
        segment = Segment(p, p)

        try:
            shared.render_segment_helper(self.primitives, segment, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Segment with coincident points raised exception: {e}")

    def test_segment_with_very_large_distance_does_not_crash(self) -> None:
        p1 = Point(-1e50, -1e50, name="A")
        p2 = Point(1e50, 1e50, name="B")
        segment = Segment(p1, p2)

        try:
            shared.render_segment_helper(self.primitives, segment, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Segment with very large distance raised exception: {e}")


class TestCircleEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.primitives = RecordingPrimitives()
        self.style = {"circle_stroke_width": 1, "circle_color": "#000"}

    def test_circle_with_nan_radius_does_not_crash(self) -> None:
        center = Point(5, 5, name="O")
        circle = Circle(center, radius=float('nan'))

        try:
            shared.render_circle_helper(self.primitives, circle, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Circle with NaN radius raised exception: {e}")

    def test_circle_with_infinity_radius_does_not_crash(self) -> None:
        center = Point(5, 5, name="O")
        circle = Circle(center, radius=float('inf'))

        try:
            shared.render_circle_helper(self.primitives, circle, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Circle with Infinity radius raised exception: {e}")

    def test_circle_with_zero_radius_does_not_crash(self) -> None:
        center = Point(5, 5, name="O")
        circle = Circle(center, radius=0)

        try:
            shared.render_circle_helper(self.primitives, circle, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Circle with zero radius raised exception: {e}")

    def test_circle_with_negative_radius_does_not_crash(self) -> None:
        center = Point(5, 5, name="O")
        circle = Circle(center, radius=-10)

        try:
            shared.render_circle_helper(self.primitives, circle, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Circle with negative radius raised exception: {e}")

    def test_circle_with_nan_center_does_not_crash(self) -> None:
        center = Point(float('nan'), float('nan'), name="O")
        circle = Circle(center, radius=10)

        try:
            shared.render_circle_helper(self.primitives, circle, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Circle with NaN center raised exception: {e}")


class TestVectorEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.primitives = RecordingPrimitives()
        self.style = {
            "segment_stroke_width": 1,
            "vector_tip_size": 8,
            "vector_color": "#000",
        }

    def test_zero_length_vector_does_not_crash(self) -> None:
        p = Point(5, 5, name="P")
        vector = Vector(p, p)

        try:
            shared.render_vector_helper(self.primitives, vector, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Zero-length vector raised exception: {e}")

    def test_vector_with_nan_endpoint_does_not_crash(self) -> None:
        p1 = Point(0, 0, name="A")
        p2 = Point(float('nan'), 0, name="B")
        vector = Vector(p1, p2)

        try:
            shared.render_vector_helper(self.primitives, vector, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Vector with NaN endpoint raised exception: {e}")

    def test_vector_with_infinity_does_not_crash(self) -> None:
        p1 = Point(0, 0, name="A")
        p2 = Point(float('inf'), 0, name="B")
        vector = Vector(p1, p2)

        try:
            shared.render_vector_helper(self.primitives, vector, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Vector with Infinity raised exception: {e}")


class TestEllipseEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.primitives = RecordingPrimitives()
        self.style = {"ellipse_stroke_width": 1, "ellipse_color": "#000"}

    def test_ellipse_with_zero_radius_x_does_not_crash(self) -> None:
        center = Point(5, 5, name="O")
        ellipse = Ellipse(center, radius_x=0, radius_y=10)

        try:
            shared.render_ellipse_helper(self.primitives, ellipse, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Ellipse with zero radius_x raised exception: {e}")

    def test_ellipse_with_zero_radius_y_does_not_crash(self) -> None:
        center = Point(5, 5, name="O")
        ellipse = Ellipse(center, radius_x=10, radius_y=0)

        try:
            shared.render_ellipse_helper(self.primitives, ellipse, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Ellipse with zero radius_y raised exception: {e}")

    def test_ellipse_with_nan_radii_does_not_crash(self) -> None:
        center = Point(5, 5, name="O")
        ellipse = Ellipse(center, radius_x=float('nan'), radius_y=float('nan'))

        try:
            shared.render_ellipse_helper(self.primitives, ellipse, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Ellipse with NaN radii raised exception: {e}")

    def test_ellipse_with_negative_radii_does_not_crash(self) -> None:
        center = Point(5, 5, name="O")
        ellipse = Ellipse(center, radius_x=-10, radius_y=-5)

        try:
            shared.render_ellipse_helper(self.primitives, ellipse, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Ellipse with negative radii raised exception: {e}")

    def test_ellipse_with_nan_rotation_does_not_crash(self) -> None:
        center = Point(5, 5, name="O")
        ellipse = Ellipse(center, radius_x=10, radius_y=5, rotation_angle=float('nan'))

        try:
            shared.render_ellipse_helper(self.primitives, ellipse, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Ellipse with NaN rotation raised exception: {e}")


class TestLabelEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.primitives = RecordingPrimitives()
        self.style = {"label_font_size": 14, "label_text_color": "#000"}

    def test_label_with_empty_text_does_not_crash(self) -> None:
        label = Label(5, 5, "")

        try:
            shared.render_label_helper(self.primitives, label, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Label with empty text raised exception: {e}")

    def test_label_with_none_text_does_not_crash(self) -> None:
        try:
            label = Label(5, 5, None)
            shared.render_label_helper(self.primitives, label, self.mapper, self.style)
        except Exception as exc:
            self.fail(f"Label with None text raised exception: {exc}")

    def test_label_with_very_long_text_does_not_crash(self) -> None:
        long_text = "A" * 150
        label = Label(5, 5, long_text)

        try:
            shared.render_label_helper(self.primitives, label, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Label with long text raised exception: {e}")

    def test_label_with_nan_font_size_does_not_crash(self) -> None:
        label = Label(5, 5, "Test", font_size=float('nan'))

        try:
            shared.render_label_helper(self.primitives, label, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Label with NaN font size raised exception: {e}")

    def test_label_with_zero_font_size_does_not_crash(self) -> None:
        label = Label(5, 5, "Test", font_size=0)

        try:
            shared.render_label_helper(self.primitives, label, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Label with zero font size raised exception: {e}")

    def test_label_with_negative_font_size_does_not_crash(self) -> None:
        label = Label(5, 5, "Test", font_size=-10)

        try:
            shared.render_label_helper(self.primitives, label, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Label with negative font size raised exception: {e}")

    def test_label_with_nan_position_does_not_crash(self) -> None:
        label = Label(float('nan'), float('nan'), "Test")

        try:
            shared.render_label_helper(self.primitives, label, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Label with NaN position raised exception: {e}")


class TestCartesianEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CoordinateMapper(640, 480)
        self.primitives = RecordingPrimitives()
        self.style = {
            "cartesian_axis_color": "#000",
            "cartesian_grid_color": "lightgrey",
            "cartesian_tick_size": 3,
            "cartesian_tick_font_size": 8,
        }

    def test_cartesian_with_zero_dimensions_does_not_crash(self) -> None:
        cartesian = SimpleNamespace(
            width=0,
            height=0,
            current_tick_spacing=50,
            default_tick_spacing=50,
        )

        try:
            shared.render_cartesian_helper(self.primitives, cartesian, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Cartesian with zero dimensions raised exception: {e}")

    def test_cartesian_with_nan_dimensions_does_not_crash(self) -> None:
        cartesian = SimpleNamespace(
            width=float('nan'),
            height=float('nan'),
            current_tick_spacing=50,
            default_tick_spacing=50,
        )

        try:
            shared.render_cartesian_helper(self.primitives, cartesian, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Cartesian with NaN dimensions raised exception: {e}")

    def test_cartesian_with_zero_tick_spacing_does_not_crash(self) -> None:
        cartesian = SimpleNamespace(
            width=640,
            height=480,
            current_tick_spacing=0,
            default_tick_spacing=50,
        )

        try:
            shared.render_cartesian_helper(self.primitives, cartesian, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Cartesian with zero tick spacing raised exception: {e}")

    def test_cartesian_with_negative_tick_spacing_does_not_crash(self) -> None:
        cartesian = SimpleNamespace(
            width=640,
            height=480,
            current_tick_spacing=-50,
            default_tick_spacing=50,
        )

        try:
            shared.render_cartesian_helper(self.primitives, cartesian, self.mapper, self.style)
        except Exception as e:
            self.fail(f"Cartesian with negative tick spacing raised exception: {e}")


__all__ = [
    "TestPointEdgeCases",
    "TestSegmentEdgeCases",
    "TestCircleEdgeCases",
    "TestVectorEdgeCases",
    "TestEllipseEdgeCases",
    "TestLabelEdgeCases",
    "TestCartesianEdgeCases",
]

