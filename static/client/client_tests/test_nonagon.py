from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()

