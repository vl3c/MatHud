from __future__ import annotations

import copy
import unittest

from drawables.point import Point
from drawables.segment import Segment
from drawables.quadrilateral import Quadrilateral


def _make_point(name: str, x: float, y: float) -> Point:
    return Point(x, y, name=name)


def _segments_from_points(points: list[Point]) -> list[Segment]:
    segments: list[Segment] = []
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        segments.append(Segment(point, next_point))
    return segments


class TestQuadrilateral(unittest.TestCase):
    def test_square_flags(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 2.0, 0.0),
            _make_point("C", 2.0, 2.0),
            _make_point("D", 0.0, 2.0),
        ]
        quad = Quadrilateral(*_segments_from_points(points))
        flags = quad.get_type_flags()
        self.assertTrue(flags["square"])
        self.assertTrue(flags["rectangle"])
        self.assertTrue(flags["rhombus"])
        self.assertFalse(flags["irregular"])
        self.assertFalse(quad.is_renderable)
        self.assertTrue(quad.is_square())
        self.assertTrue(quad.is_rectangle())
        self.assertTrue(quad.is_rhombus())
        self.assertFalse(quad.is_irregular())

    def test_rectangle_flags(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 3.0, 0.0),
            _make_point("C", 3.0, 1.0),
            _make_point("D", 0.0, 1.0),
        ]
        quad = Quadrilateral(*_segments_from_points(points))
        flags = quad.get_type_flags()
        self.assertFalse(flags["square"])
        self.assertTrue(flags["rectangle"])
        self.assertFalse(flags["rhombus"])
        self.assertFalse(flags["irregular"])
        self.assertFalse(quad.is_renderable)

    def test_rhombus_flags(self) -> None:
        import math

        h = math.sqrt(3)
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 2.0, 0.0),
            _make_point("C", 3.0, h),
            _make_point("D", 1.0, h),
        ]
        quad = Quadrilateral(*_segments_from_points(points))
        flags = quad.get_type_flags()
        self.assertFalse(flags["square"])
        self.assertFalse(flags["rectangle"])
        self.assertTrue(flags["rhombus"])
        self.assertFalse(flags["irregular"])

    def test_irregular_flags(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 2.0, 0.5),
            _make_point("C", 1.5, 1.7),
            _make_point("D", 0.3, 1.2),
        ]
        quad = Quadrilateral(*_segments_from_points(points))
        flags = quad.get_type_flags()
        self.assertFalse(flags["square"])
        self.assertFalse(flags["rectangle"])
        self.assertFalse(flags["rhombus"])
        self.assertTrue(flags["irregular"])
        self.assertFalse(quad.is_renderable)

    def test_invalid_segments_raise(self) -> None:
        p1 = _make_point("A", 0.0, 0.0)
        p2 = _make_point("B", 1.0, 0.0)
        p3 = _make_point("C", 2.0, 0.0)
        p4 = _make_point("D", 3.0, 0.0)
        segments = [
            Segment(p1, p2),
            Segment(p2, p3),
            Segment(p3, p4),
            Segment(p4, p1),
        ]
        with self.assertRaises(ValueError):
            Quadrilateral(*segments)

    def test_get_state(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 2.0, 0.0),
            _make_point("C", 2.0, 2.0),
            _make_point("D", 0.0, 2.0),
        ]
        quad = Quadrilateral(*_segments_from_points(points))
        state = quad.get_state()
        self.assertEqual(state["name"], quad.name)
        self.assertIn("args", state)
        self.assertEqual(len(state["args"]), 4)
        self.assertIn("types", state)
        self.assertIn("quadrilateral", state["types"])

    def test_get_vertices(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 2.0, 0.0),
            _make_point("C", 2.0, 2.0),
            _make_point("D", 0.0, 2.0),
        ]
        quad = Quadrilateral(*_segments_from_points(points))
        vertices = quad.get_vertices()
        self.assertEqual(len(vertices), 4)
        vertex_names = {v.name for v in vertices}
        self.assertEqual(vertex_names, {"A", "B", "C", "D"})

    def test_get_segments(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 2.0, 0.0),
            _make_point("C", 2.0, 2.0),
            _make_point("D", 0.0, 2.0),
        ]
        segments = _segments_from_points(points)
        quad = Quadrilateral(*segments)
        retrieved = quad.get_segments()
        self.assertEqual(len(retrieved), 4)
        for seg in segments:
            self.assertIn(seg, retrieved)

    def test_deepcopy(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 2.0, 0.0),
            _make_point("C", 2.0, 2.0),
            _make_point("D", 0.0, 2.0),
        ]
        quad = Quadrilateral(*_segments_from_points(points), color="red")
        quad_copy = copy.deepcopy(quad)
        self.assertIsNot(quad_copy, quad)
        self.assertIsNot(quad_copy.segment1, quad.segment1)
        self.assertEqual(quad_copy.color, quad.color)
        self.assertEqual(quad_copy.name, quad.name)

    def test_update_color(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 2.0, 0.0),
            _make_point("C", 2.0, 2.0),
            _make_point("D", 0.0, 2.0),
        ]
        quad = Quadrilateral(*_segments_from_points(points), color="blue")
        quad.update_color("green")
        self.assertEqual(quad.color, "green")
        for segment in quad.get_segments():
            self.assertEqual(segment.color, "green")

    def test_get_class_name(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 2.0, 0.0),
            _make_point("C", 2.0, 2.0),
            _make_point("D", 0.0, 2.0),
        ]
        quad = Quadrilateral(*_segments_from_points(points))
        self.assertEqual(quad.get_class_name(), "Quadrilateral")

    def test_translate(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 2.0, 0.0),
            _make_point("C", 2.0, 2.0),
            _make_point("D", 0.0, 2.0),
        ]
        quad = Quadrilateral(*_segments_from_points(points))
        quad.translate(5.0, 3.0)
        vertices = list(quad.get_vertices())
        translated_a = next(v for v in vertices if v.name == "A")
        self.assertAlmostEqual(translated_a.x, 5.0, places=6)
        self.assertAlmostEqual(translated_a.y, 3.0, places=6)

    def test_disconnected_segments_raise(self) -> None:
        p1 = _make_point("A", 0.0, 0.0)
        p2 = _make_point("B", 2.0, 0.0)
        p3 = _make_point("C", 2.0, 2.0)
        p4 = _make_point("D", 0.0, 2.0)
        disconnected = _make_point("X", 100.0, 100.0)
        segments = [
            Segment(p1, p2),
            Segment(p2, p3),
            Segment(disconnected, _make_point("Y", 101.0, 101.0)),
            Segment(p4, p1),
        ]
        with self.assertRaises(ValueError):
            Quadrilateral(*segments)

    def test_get_state_types_square(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 2.0, 0.0),
            _make_point("C", 2.0, 2.0),
            _make_point("D", 0.0, 2.0),
        ]
        quad = Quadrilateral(*_segments_from_points(points))
        state = quad.get_state()
        self.assertEqual(state["types"][0], "quadrilateral")
        self.assertIn("square", state["types"])
        self.assertIn("rectangle", state["types"])
        self.assertIn("rhombus", state["types"])

    def test_get_state_types_irregular(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 2.0, 0.5),
            _make_point("C", 1.5, 1.7),
            _make_point("D", 0.3, 1.2),
        ]
        quad = Quadrilateral(*_segments_from_points(points))
        state = quad.get_state()
        self.assertEqual(state["types"][0], "quadrilateral")
        self.assertIn("irregular", state["types"])


if __name__ == "__main__":
    unittest.main()
