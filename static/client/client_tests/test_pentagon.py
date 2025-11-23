from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()

