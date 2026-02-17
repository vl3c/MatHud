from __future__ import annotations

import copy
import math
import unittest

from drawables.point import Point
from drawables.segment import Segment
from drawables.heptagon import Heptagon


def _make_regular_heptagon_points() -> list[Point]:
    points: list[Point] = []
    radius = 4.0
    for idx in range(7):
        angle = 2 * math.pi * idx / 7
        points.append(Point(radius * math.cos(angle), radius * math.sin(angle), name=f"H{idx}"))
    return points


def _segments_from_points(points: list[Point]) -> list[Segment]:
    segments: list[Segment] = []
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        segments.append(Segment(point, next_point))
    return segments


class TestHeptagon(unittest.TestCase):
    def test_regular_heptagon_flags(self) -> None:
        points = _make_regular_heptagon_points()
        heptagon = Heptagon(_segments_from_points(points))
        flags = heptagon.get_type_flags()
        self.assertTrue(flags["regular"])
        self.assertFalse(flags["irregular"])
        self.assertFalse(heptagon.is_renderable)

    def test_irregular_heptagon_flags(self) -> None:
        points = _make_regular_heptagon_points()
        points[2].y *= 1.1
        heptagon = Heptagon(_segments_from_points(points))
        flags = heptagon.get_type_flags()
        self.assertFalse(flags["regular"])
        self.assertTrue(flags["irregular"])
        self.assertFalse(heptagon.is_renderable)

    def test_invalid_segment_count(self) -> None:
        points = _make_regular_heptagon_points()
        with self.assertRaises(ValueError):
            Heptagon(_segments_from_points(points)[:-1])

    def test_get_state(self) -> None:
        points = _make_regular_heptagon_points()
        heptagon = Heptagon(_segments_from_points(points))
        state = heptagon.get_state()
        self.assertEqual(state["name"], heptagon.name)
        self.assertIn("args", state)
        self.assertEqual(len(state["args"]), 7)
        self.assertIn("types", state)
        self.assertIn("heptagon", state["types"])

    def test_get_vertices(self) -> None:
        points = _make_regular_heptagon_points()
        heptagon = Heptagon(_segments_from_points(points))
        vertices = heptagon.get_vertices()
        self.assertEqual(len(vertices), 7)

    def test_get_segments(self) -> None:
        points = _make_regular_heptagon_points()
        segments = _segments_from_points(points)
        heptagon = Heptagon(segments)
        retrieved = heptagon.get_segments()
        self.assertEqual(len(retrieved), 7)

    def test_deepcopy(self) -> None:
        points = _make_regular_heptagon_points()
        heptagon = Heptagon(_segments_from_points(points), color="red")
        heptagon_copy = copy.deepcopy(heptagon)
        self.assertIsNot(heptagon_copy, heptagon)
        self.assertEqual(heptagon_copy.color, heptagon.color)
        self.assertEqual(heptagon_copy.name, heptagon.name)

    def test_update_color(self) -> None:
        points = _make_regular_heptagon_points()
        heptagon = Heptagon(_segments_from_points(points), color="blue")
        heptagon.update_color("green")
        self.assertEqual(heptagon.color, "green")
        for segment in heptagon.get_segments():
            self.assertEqual(segment.color, "green")

    def test_translate(self) -> None:
        points = _make_regular_heptagon_points()
        heptagon = Heptagon(_segments_from_points(points))
        original_x = points[0].x
        original_y = points[0].y
        heptagon.translate(5.0, 3.0)
        vertices = list(heptagon.get_vertices())
        translated = next(v for v in vertices if v.name == "H0")
        self.assertAlmostEqual(translated.x, original_x + 5.0, places=6)
        self.assertAlmostEqual(translated.y, original_y + 3.0, places=6)

    def test_disconnected_segments_raise(self) -> None:
        points = _make_regular_heptagon_points()
        segments = _segments_from_points(points)
        disconnected = Point(100.0, 100.0, name="X")
        segments[3] = Segment(disconnected, Point(101.0, 101.0, name="Y"))
        with self.assertRaises(ValueError):
            Heptagon(segments)

    def test_get_state_types(self) -> None:
        points = _make_regular_heptagon_points()
        heptagon = Heptagon(_segments_from_points(points))
        state = heptagon.get_state()
        self.assertEqual(state["types"][0], "heptagon")
        self.assertIn("regular", state["types"])

    def test_is_regular_helper(self) -> None:
        points = _make_regular_heptagon_points()
        heptagon = Heptagon(_segments_from_points(points))
        self.assertTrue(heptagon.is_regular())
        self.assertFalse(heptagon.is_irregular())

    def test_is_irregular_helper(self) -> None:
        points = _make_regular_heptagon_points()
        points[2].y *= 1.1
        heptagon = Heptagon(_segments_from_points(points))
        self.assertFalse(heptagon.is_regular())
        self.assertTrue(heptagon.is_irregular())


if __name__ == "__main__":
    unittest.main()
