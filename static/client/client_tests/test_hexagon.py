from __future__ import annotations

import copy
import math
import unittest

from drawables.point import Point
from drawables.segment import Segment
from drawables.hexagon import Hexagon


def _make_regular_hexagon_points() -> list[Point]:
    points: list[Point] = []
    radius = 4.0
    for idx in range(6):
        angle = 2 * math.pi * idx / 6
        points.append(Point(radius * math.cos(angle), radius * math.sin(angle), name=f"H{idx}"))
    return points


def _segments_from_points(points: list[Point]) -> list[Segment]:
    segments: list[Segment] = []
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        segments.append(Segment(point, next_point))
    return segments


class TestHexagon(unittest.TestCase):
    def test_regular_hexagon_flags(self) -> None:
        points = _make_regular_hexagon_points()
        hexagon = Hexagon(_segments_from_points(points))
        flags = hexagon.get_type_flags()
        self.assertTrue(flags["regular"])
        self.assertFalse(flags["irregular"])
        self.assertFalse(hexagon.is_renderable)

    def test_irregular_hexagon_flags(self) -> None:
        points = _make_regular_hexagon_points()
        points[2].y *= 1.1
        hexagon = Hexagon(_segments_from_points(points))
        flags = hexagon.get_type_flags()
        self.assertFalse(flags["regular"])
        self.assertTrue(flags["irregular"])
        self.assertFalse(hexagon.is_renderable)

    def test_invalid_segment_count(self) -> None:
        points = _make_regular_hexagon_points()
        with self.assertRaises(ValueError):
            Hexagon(_segments_from_points(points)[:-1])

    def test_get_state(self) -> None:
        points = _make_regular_hexagon_points()
        hexagon = Hexagon(_segments_from_points(points))
        state = hexagon.get_state()
        self.assertEqual(state["name"], hexagon.name)
        self.assertIn("args", state)
        self.assertEqual(len(state["args"]), 6)
        self.assertIn("types", state)
        self.assertEqual(state["types"][0], "hexagon")

    def test_get_vertices(self) -> None:
        points = _make_regular_hexagon_points()
        hexagon = Hexagon(_segments_from_points(points))
        vertices = hexagon.get_vertices()
        self.assertEqual(len(vertices), 6)

    def test_get_segments(self) -> None:
        points = _make_regular_hexagon_points()
        segments = _segments_from_points(points)
        hexagon = Hexagon(segments)
        retrieved = hexagon.get_segments()
        self.assertEqual(len(retrieved), 6)

    def test_deepcopy(self) -> None:
        points = _make_regular_hexagon_points()
        hexagon = Hexagon(_segments_from_points(points), color="red")
        hexagon_copy = copy.deepcopy(hexagon)
        self.assertIsNot(hexagon_copy, hexagon)
        self.assertEqual(hexagon_copy.color, hexagon.color)
        self.assertEqual(hexagon_copy.name, hexagon.name)

    def test_update_color(self) -> None:
        points = _make_regular_hexagon_points()
        hexagon = Hexagon(_segments_from_points(points), color="blue")
        hexagon.update_color("green")
        self.assertEqual(hexagon.color, "green")
        for segment in hexagon.get_segments():
            self.assertEqual(segment.color, "green")

    def test_translate(self) -> None:
        points = _make_regular_hexagon_points()
        hexagon = Hexagon(_segments_from_points(points))
        original_x = points[0].x
        original_y = points[0].y
        hexagon.translate(5.0, 3.0)
        vertices = list(hexagon.get_vertices())
        translated = next(v for v in vertices if v.name == "H0")
        self.assertAlmostEqual(translated.x, original_x + 5.0, places=6)
        self.assertAlmostEqual(translated.y, original_y + 3.0, places=6)

    def test_disconnected_segments_raise(self) -> None:
        points = _make_regular_hexagon_points()
        segments = _segments_from_points(points)
        disconnected = Point(100.0, 100.0, name="X")
        segments[2] = Segment(disconnected, Point(101.0, 101.0, name="Y"))
        with self.assertRaises(ValueError):
            Hexagon(segments)

    def test_is_regular_helper(self) -> None:
        points = _make_regular_hexagon_points()
        hexagon = Hexagon(_segments_from_points(points))
        self.assertTrue(hexagon.is_regular())
        self.assertFalse(hexagon.is_irregular())

    def test_is_irregular_helper(self) -> None:
        points = _make_regular_hexagon_points()
        points[2].y *= 1.1
        hexagon = Hexagon(_segments_from_points(points))
        self.assertFalse(hexagon.is_regular())
        self.assertTrue(hexagon.is_irregular())


if __name__ == "__main__":
    unittest.main()

