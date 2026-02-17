from __future__ import annotations

import copy
import math
import unittest

from drawables.point import Point
from drawables.segment import Segment
from drawables.octagon import Octagon


def _make_regular_octagon_points() -> list[Point]:
    points: list[Point] = []
    radius = 4.0
    for idx in range(8):
        angle = 2 * math.pi * idx / 8
        points.append(Point(radius * math.cos(angle), radius * math.sin(angle), name=f"O{idx}"))
    return points


def _segments_from_points(points: list[Point]) -> list[Segment]:
    segments: list[Segment] = []
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        segments.append(Segment(point, next_point))
    return segments


class TestOctagon(unittest.TestCase):
    def test_regular_octagon_flags(self) -> None:
        points = _make_regular_octagon_points()
        octagon = Octagon(_segments_from_points(points))
        flags = octagon.get_type_flags()
        self.assertTrue(flags["regular"])
        self.assertFalse(flags["irregular"])
        self.assertFalse(octagon.is_renderable)

    def test_irregular_octagon_flags(self) -> None:
        points = _make_regular_octagon_points()
        points[2].y *= 1.1
        octagon = Octagon(_segments_from_points(points))
        flags = octagon.get_type_flags()
        self.assertFalse(flags["regular"])
        self.assertTrue(flags["irregular"])
        self.assertFalse(octagon.is_renderable)

    def test_invalid_segment_count(self) -> None:
        points = _make_regular_octagon_points()
        with self.assertRaises(ValueError):
            Octagon(_segments_from_points(points)[:-1])

    def test_get_state(self) -> None:
        points = _make_regular_octagon_points()
        octagon = Octagon(_segments_from_points(points))
        state = octagon.get_state()
        self.assertEqual(state["name"], octagon.name)
        self.assertIn("args", state)
        self.assertEqual(len(state["args"]), 8)
        self.assertIn("types", state)
        self.assertIn("octagon", state["types"])

    def test_get_vertices(self) -> None:
        points = _make_regular_octagon_points()
        octagon = Octagon(_segments_from_points(points))
        vertices = octagon.get_vertices()
        self.assertEqual(len(vertices), 8)

    def test_get_segments(self) -> None:
        points = _make_regular_octagon_points()
        segments = _segments_from_points(points)
        octagon = Octagon(segments)
        retrieved = octagon.get_segments()
        self.assertEqual(len(retrieved), 8)

    def test_deepcopy(self) -> None:
        points = _make_regular_octagon_points()
        octagon = Octagon(_segments_from_points(points), color="red")
        octagon_copy = copy.deepcopy(octagon)
        self.assertIsNot(octagon_copy, octagon)
        self.assertEqual(octagon_copy.color, octagon.color)
        self.assertEqual(octagon_copy.name, octagon.name)

    def test_update_color(self) -> None:
        points = _make_regular_octagon_points()
        octagon = Octagon(_segments_from_points(points), color="blue")
        octagon.update_color("green")
        self.assertEqual(octagon.color, "green")
        for segment in octagon.get_segments():
            self.assertEqual(segment.color, "green")

    def test_translate(self) -> None:
        points = _make_regular_octagon_points()
        octagon = Octagon(_segments_from_points(points))
        original_x = points[0].x
        original_y = points[0].y
        octagon.translate(5.0, 3.0)
        vertices = list(octagon.get_vertices())
        translated = next(v for v in vertices if v.name == "O0")
        self.assertAlmostEqual(translated.x, original_x + 5.0, places=6)
        self.assertAlmostEqual(translated.y, original_y + 3.0, places=6)

    def test_disconnected_segments_raise(self) -> None:
        points = _make_regular_octagon_points()
        segments = _segments_from_points(points)
        disconnected = Point(100.0, 100.0, name="X")
        segments[3] = Segment(disconnected, Point(101.0, 101.0, name="Y"))
        with self.assertRaises(ValueError):
            Octagon(segments)

    def test_get_state_types(self) -> None:
        points = _make_regular_octagon_points()
        octagon = Octagon(_segments_from_points(points))
        state = octagon.get_state()
        self.assertEqual(state["types"][0], "octagon")
        self.assertIn("regular", state["types"])

    def test_is_regular_helper(self) -> None:
        points = _make_regular_octagon_points()
        octagon = Octagon(_segments_from_points(points))
        self.assertTrue(octagon.is_regular())
        self.assertFalse(octagon.is_irregular())

    def test_is_irregular_helper(self) -> None:
        points = _make_regular_octagon_points()
        points[2].y *= 1.1
        octagon = Octagon(_segments_from_points(points))
        self.assertFalse(octagon.is_regular())
        self.assertTrue(octagon.is_irregular())


if __name__ == "__main__":
    unittest.main()
