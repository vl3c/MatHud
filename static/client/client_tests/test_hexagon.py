from __future__ import annotations

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

    def test_irregular_hexagon_flags(self) -> None:
        points = _make_regular_hexagon_points()
        points[3].y *= 0.7
        hexagon = Hexagon(_segments_from_points(points))
        flags = hexagon.get_type_flags()
        self.assertFalse(flags["regular"])
        self.assertTrue(flags["irregular"])

    def test_invalid_segment_count(self) -> None:
        points = _make_regular_hexagon_points()
        with self.assertRaises(ValueError):
            Hexagon(_segments_from_points(points)[:-1])


if __name__ == "__main__":
    unittest.main()

