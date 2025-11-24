from __future__ import annotations

import copy
import math
import unittest

from drawables.point import Point
from drawables.segment import Segment
from drawables.decagon import Decagon


def _make_regular_decagon_points() -> list[Point]:
    points: list[Point] = []
    radius = 4.0
    for idx in range(10):
        angle = 2 * math.pi * idx / 10
        points.append(Point(radius * math.cos(angle), radius * math.sin(angle), name=f"D{idx}"))
    return points


def _segments_from_points(points: list[Point]) -> list[Segment]:
    segments: list[Segment] = []
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        segments.append(Segment(point, next_point))
    return segments


class TestDecagon(unittest.TestCase):
    def test_regular_decagon_flags(self) -> None:
        points = _make_regular_decagon_points()
        decagon = Decagon(_segments_from_points(points))
        flags = decagon.get_type_flags()
        self.assertTrue(flags["regular"])
        self.assertFalse(flags["irregular"])
        self.assertFalse(decagon.is_renderable)

    def test_irregular_decagon_flags(self) -> None:
        points = _make_regular_decagon_points()
        points[2].y *= 1.1
        decagon = Decagon(_segments_from_points(points))
        flags = decagon.get_type_flags()
        self.assertFalse(flags["regular"])
        self.assertTrue(flags["irregular"])
        self.assertFalse(decagon.is_renderable)

    def test_invalid_segment_count(self) -> None:
        points = _make_regular_decagon_points()
        with self.assertRaises(ValueError):
            Decagon(_segments_from_points(points)[:-1])

    def test_get_state(self) -> None:
        points = _make_regular_decagon_points()
        decagon = Decagon(_segments_from_points(points))
        state = decagon.get_state()
        self.assertEqual(state["name"], decagon.name)
        self.assertIn("args", state)
        self.assertEqual(len(state["args"]), 10)
        self.assertIn("types", state)
        self.assertIn("decagon", state["types"])

    def test_get_vertices(self) -> None:
        points = _make_regular_decagon_points()
        decagon = Decagon(_segments_from_points(points))
        vertices = decagon.get_vertices()
        self.assertEqual(len(vertices), 10)

    def test_get_segments(self) -> None:
        points = _make_regular_decagon_points()
        segments = _segments_from_points(points)
        decagon = Decagon(segments)
        retrieved = decagon.get_segments()
        self.assertEqual(len(retrieved), 10)

    def test_deepcopy(self) -> None:
        points = _make_regular_decagon_points()
        decagon = Decagon(_segments_from_points(points), color="red")
        decagon_copy = copy.deepcopy(decagon)
        self.assertIsNot(decagon_copy, decagon)
        self.assertEqual(decagon_copy.color, decagon.color)
        self.assertEqual(decagon_copy.name, decagon.name)

    def test_update_color(self) -> None:
        points = _make_regular_decagon_points()
        decagon = Decagon(_segments_from_points(points), color="blue")
        decagon.update_color("green")
        self.assertEqual(decagon.color, "green")
        for segment in decagon.get_segments():
            self.assertEqual(segment.color, "green")

    def test_translate(self) -> None:
        points = _make_regular_decagon_points()
        decagon = Decagon(_segments_from_points(points))
        original_x = points[0].x
        original_y = points[0].y
        decagon.translate(5.0, 3.0)
        vertices = list(decagon.get_vertices())
        translated = next(v for v in vertices if v.name == "D0")
        self.assertAlmostEqual(translated.x, original_x + 5.0, places=6)
        self.assertAlmostEqual(translated.y, original_y + 3.0, places=6)

    def test_disconnected_segments_raise(self) -> None:
        points = _make_regular_decagon_points()
        segments = _segments_from_points(points)
        disconnected = Point(100.0, 100.0, name="X")
        segments[3] = Segment(disconnected, Point(101.0, 101.0, name="Y"))
        with self.assertRaises(ValueError):
            Decagon(segments)

    def test_get_state_types(self) -> None:
        points = _make_regular_decagon_points()
        decagon = Decagon(_segments_from_points(points))
        state = decagon.get_state()
        self.assertEqual(state["types"][0], "decagon")
        self.assertIn("regular", state["types"])

    def test_is_regular_helper(self) -> None:
        points = _make_regular_decagon_points()
        decagon = Decagon(_segments_from_points(points))
        self.assertTrue(decagon.is_regular())
        self.assertFalse(decagon.is_irregular())

    def test_is_irregular_helper(self) -> None:
        points = _make_regular_decagon_points()
        points[2].y *= 1.1
        decagon = Decagon(_segments_from_points(points))
        self.assertFalse(decagon.is_regular())
        self.assertTrue(decagon.is_irregular())


if __name__ == "__main__":
    unittest.main()

