from __future__ import annotations

import copy
import math
import unittest

from drawables.point import Point
from drawables.segment import Segment
from drawables.pentagon import Pentagon


def _make_regular_pentagon_points() -> list[Point]:
    points: list[Point] = []
    radius = 3.0
    for idx in range(5):
        angle = 2 * math.pi * idx / 5
        points.append(Point(radius * math.cos(angle), radius * math.sin(angle), name=f"P{idx}"))
    return points


def _segments_from_points(points: list[Point]) -> list[Segment]:
    segments: list[Segment] = []
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        segments.append(Segment(point, next_point))
    return segments


class TestPentagon(unittest.TestCase):
    def test_regular_pentagon_flags(self) -> None:
        points = _make_regular_pentagon_points()
        pentagon = Pentagon(_segments_from_points(points))
        flags = pentagon.get_type_flags()
        self.assertTrue(flags["regular"])
        self.assertFalse(flags["irregular"])
        self.assertFalse(pentagon.is_renderable)

    def test_irregular_pentagon_flags(self) -> None:
        points = _make_regular_pentagon_points()
        points[2].x *= 1.2
        pentagon = Pentagon(_segments_from_points(points))
        flags = pentagon.get_type_flags()
        self.assertFalse(flags["regular"])
        self.assertTrue(flags["irregular"])
        self.assertFalse(pentagon.is_renderable)

    def test_invalid_segment_count(self) -> None:
        points = _make_regular_pentagon_points()
        with self.assertRaises(ValueError):
            Pentagon(_segments_from_points(points)[:-1])

    def test_get_state(self) -> None:
        points = _make_regular_pentagon_points()
        pentagon = Pentagon(_segments_from_points(points))
        state = pentagon.get_state()
        self.assertEqual(state["name"], pentagon.name)
        self.assertIn("args", state)
        self.assertEqual(len(state["args"]), 5)
        self.assertIn("types", state)
        self.assertEqual(state["types"][0], "pentagon")

    def test_get_vertices(self) -> None:
        points = _make_regular_pentagon_points()
        pentagon = Pentagon(_segments_from_points(points))
        vertices = pentagon.get_vertices()
        self.assertEqual(len(vertices), 5)

    def test_get_segments(self) -> None:
        points = _make_regular_pentagon_points()
        segments = _segments_from_points(points)
        pentagon = Pentagon(segments)
        retrieved = pentagon.get_segments()
        self.assertEqual(len(retrieved), 5)

    def test_deepcopy(self) -> None:
        points = _make_regular_pentagon_points()
        pentagon = Pentagon(_segments_from_points(points), color="red")
        pentagon_copy = copy.deepcopy(pentagon)
        self.assertIsNot(pentagon_copy, pentagon)
        self.assertEqual(pentagon_copy.color, pentagon.color)
        self.assertEqual(pentagon_copy.name, pentagon.name)

    def test_update_color(self) -> None:
        points = _make_regular_pentagon_points()
        pentagon = Pentagon(_segments_from_points(points), color="blue")
        pentagon.update_color("green")
        self.assertEqual(pentagon.color, "green")
        for segment in pentagon.get_segments():
            self.assertEqual(segment.color, "green")

    def test_translate(self) -> None:
        points = _make_regular_pentagon_points()
        pentagon = Pentagon(_segments_from_points(points))
        original_x = points[0].x
        original_y = points[0].y
        pentagon.translate(5.0, 3.0)
        vertices = list(pentagon.get_vertices())
        translated = next(v for v in vertices if v.name == "P0")
        self.assertAlmostEqual(translated.x, original_x + 5.0, places=6)
        self.assertAlmostEqual(translated.y, original_y + 3.0, places=6)

    def test_disconnected_segments_raise(self) -> None:
        points = _make_regular_pentagon_points()
        segments = _segments_from_points(points)
        disconnected = Point(100.0, 100.0, name="X")
        segments[2] = Segment(disconnected, Point(101.0, 101.0, name="Y"))
        with self.assertRaises(ValueError):
            Pentagon(segments)

    def test_is_regular_helper(self) -> None:
        points = _make_regular_pentagon_points()
        pentagon = Pentagon(_segments_from_points(points))
        self.assertTrue(pentagon.is_regular())
        self.assertFalse(pentagon.is_irregular())

    def test_is_irregular_helper(self) -> None:
        points = _make_regular_pentagon_points()
        points[2].x *= 1.2
        pentagon = Pentagon(_segments_from_points(points))
        self.assertFalse(pentagon.is_regular())
        self.assertTrue(pentagon.is_irregular())


if __name__ == "__main__":
    unittest.main()

