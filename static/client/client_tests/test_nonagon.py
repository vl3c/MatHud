from __future__ import annotations

import copy
import math
import unittest

from drawables.point import Point
from drawables.segment import Segment
from drawables.nonagon import Nonagon


def _make_regular_nonagon_points() -> list[Point]:
    points: list[Point] = []
    radius = 4.0
    for idx in range(9):
        angle = 2 * math.pi * idx / 9
        points.append(Point(radius * math.cos(angle), radius * math.sin(angle), name=f"N{idx}"))
    return points


def _segments_from_points(points: list[Point]) -> list[Segment]:
    segments: list[Segment] = []
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        segments.append(Segment(point, next_point))
    return segments


class TestNonagon(unittest.TestCase):
    def test_regular_nonagon_flags(self) -> None:
        points = _make_regular_nonagon_points()
        nonagon = Nonagon(_segments_from_points(points))
        flags = nonagon.get_type_flags()
        self.assertTrue(flags["regular"])
        self.assertFalse(flags["irregular"])
        self.assertFalse(nonagon.is_renderable)

    def test_irregular_nonagon_flags(self) -> None:
        points = _make_regular_nonagon_points()
        points[2].y *= 1.1
        nonagon = Nonagon(_segments_from_points(points))
        flags = nonagon.get_type_flags()
        self.assertFalse(flags["regular"])
        self.assertTrue(flags["irregular"])
        self.assertFalse(nonagon.is_renderable)

    def test_invalid_segment_count(self) -> None:
        points = _make_regular_nonagon_points()
        with self.assertRaises(ValueError):
            Nonagon(_segments_from_points(points)[:-1])

    def test_get_state(self) -> None:
        points = _make_regular_nonagon_points()
        nonagon = Nonagon(_segments_from_points(points))
        state = nonagon.get_state()
        self.assertEqual(state["name"], nonagon.name)
        self.assertIn("args", state)
        self.assertEqual(len(state["args"]), 9)
        self.assertIn("types", state)
        self.assertIn("nonagon", state["types"])

    def test_get_vertices(self) -> None:
        points = _make_regular_nonagon_points()
        nonagon = Nonagon(_segments_from_points(points))
        vertices = nonagon.get_vertices()
        self.assertEqual(len(vertices), 9)

    def test_get_segments(self) -> None:
        points = _make_regular_nonagon_points()
        segments = _segments_from_points(points)
        nonagon = Nonagon(segments)
        retrieved = nonagon.get_segments()
        self.assertEqual(len(retrieved), 9)

    def test_deepcopy(self) -> None:
        points = _make_regular_nonagon_points()
        nonagon = Nonagon(_segments_from_points(points), color="red")
        nonagon_copy = copy.deepcopy(nonagon)
        self.assertIsNot(nonagon_copy, nonagon)
        self.assertEqual(nonagon_copy.color, nonagon.color)
        self.assertEqual(nonagon_copy.name, nonagon.name)

    def test_update_color(self) -> None:
        points = _make_regular_nonagon_points()
        nonagon = Nonagon(_segments_from_points(points), color="blue")
        nonagon.update_color("green")
        self.assertEqual(nonagon.color, "green")
        for segment in nonagon.get_segments():
            self.assertEqual(segment.color, "green")

    def test_translate(self) -> None:
        points = _make_regular_nonagon_points()
        nonagon = Nonagon(_segments_from_points(points))
        original_x = points[0].x
        original_y = points[0].y
        nonagon.translate(5.0, 3.0)
        vertices = list(nonagon.get_vertices())
        translated = next(v for v in vertices if v.name == "N0")
        self.assertAlmostEqual(translated.x, original_x + 5.0, places=6)
        self.assertAlmostEqual(translated.y, original_y + 3.0, places=6)

    def test_disconnected_segments_raise(self) -> None:
        points = _make_regular_nonagon_points()
        segments = _segments_from_points(points)
        disconnected = Point(100.0, 100.0, name="X")
        segments[3] = Segment(disconnected, Point(101.0, 101.0, name="Y"))
        with self.assertRaises(ValueError):
            Nonagon(segments)

    def test_get_state_types(self) -> None:
        points = _make_regular_nonagon_points()
        nonagon = Nonagon(_segments_from_points(points))
        state = nonagon.get_state()
        self.assertEqual(state["types"][0], "nonagon")
        self.assertIn("regular", state["types"])

    def test_is_regular_helper(self) -> None:
        points = _make_regular_nonagon_points()
        nonagon = Nonagon(_segments_from_points(points))
        self.assertTrue(nonagon.is_regular())
        self.assertFalse(nonagon.is_irregular())

    def test_is_irregular_helper(self) -> None:
        points = _make_regular_nonagon_points()
        points[2].y *= 1.1
        nonagon = Nonagon(_segments_from_points(points))
        self.assertFalse(nonagon.is_regular())
        self.assertTrue(nonagon.is_irregular())


if __name__ == "__main__":
    unittest.main()
