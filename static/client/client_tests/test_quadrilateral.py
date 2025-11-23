from __future__ import annotations

import unittest

from drawables.point import Point
from drawables.segment import Segment
from drawables.quadrilateral import Quadrilateral


def _make_point(name: str, x: float, y: float) -> Point:
    return Point(x, y, name=name)


def _segments_from_points(points: list[Point]) -> list[Segment]:
    segments: list[Segment] = []
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        segments.append(Segment(point, next_point))
    return segments


class TestQuadrilateral(unittest.TestCase):
    def test_square_flags(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 2.0, 0.0),
            _make_point("C", 2.0, 2.0),
            _make_point("D", 0.0, 2.0),
        ]
        quad = Quadrilateral(*_segments_from_points(points))
        flags = quad.get_type_flags()
        self.assertTrue(flags["square"])
        self.assertTrue(flags["rectangle"])
        self.assertTrue(flags["rhombus"])
        self.assertFalse(flags["irregular"])
        self.assertFalse(quad.is_renderable)

    def test_rectangle_flags(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 3.0, 0.0),
            _make_point("C", 3.0, 1.0),
            _make_point("D", 0.0, 1.0),
        ]
        quad = Quadrilateral(*_segments_from_points(points))
        flags = quad.get_type_flags()
        self.assertFalse(flags["square"])
        self.assertTrue(flags["rectangle"])
        self.assertFalse(flags["rhombus"])
        self.assertFalse(flags["irregular"])
        self.assertFalse(quad.is_renderable)

    def test_irregular_flags(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 2.0, 0.5),
            _make_point("C", 1.5, 1.7),
            _make_point("D", 0.3, 1.2),
        ]
        quad = Quadrilateral(*_segments_from_points(points))
        flags = quad.get_type_flags()
        self.assertFalse(flags["square"])
        self.assertFalse(flags["rectangle"])
        self.assertFalse(flags["rhombus"])
        self.assertTrue(flags["irregular"])
        self.assertFalse(quad.is_renderable)

    def test_invalid_segments_raise(self) -> None:
        p1 = _make_point("A", 0.0, 0.0)
        p2 = _make_point("B", 1.0, 0.0)
        p3 = _make_point("C", 2.0, 0.0)
        p4 = _make_point("D", 3.0, 0.0)
        segments = [
            Segment(p1, p2),
            Segment(p2, p3),
            Segment(p3, p4),
            Segment(p4, p1),
        ]
        with self.assertRaises(ValueError):
            Quadrilateral(*segments)


if __name__ == "__main__":
    unittest.main()

