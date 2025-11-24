from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()

