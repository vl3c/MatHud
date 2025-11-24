from __future__ import annotations

import math
import unittest

from drawables.point import Point
from drawables.segment import Segment
from drawables.generic_polygon import GenericPolygon


def _make_regular_polygon_points(sides: int) -> list[Point]:
    points: list[Point] = []
    radius = 4.0
    for idx in range(sides):
        angle = 2 * math.pi * idx / sides
        points.append(Point(radius * math.cos(angle), radius * math.sin(angle), name=f"P{idx}"))
    return points


def _segments_from_points(points: list[Point]) -> list[Segment]:
    segments: list[Segment] = []
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        segments.append(Segment(point, next_point))
    return segments


class TestGenericPolygon(unittest.TestCase):
    def test_regular_11_sided_polygon_flags(self) -> None:
        points = _make_regular_polygon_points(11)
        polygon = GenericPolygon(_segments_from_points(points))
        flags = polygon.get_type_flags()
        self.assertTrue(flags["regular"])
        self.assertFalse(flags["irregular"])
        self.assertFalse(polygon.is_renderable)

    def test_regular_15_sided_polygon_flags(self) -> None:
        points = _make_regular_polygon_points(15)
        polygon = GenericPolygon(_segments_from_points(points))
        flags = polygon.get_type_flags()
        self.assertTrue(flags["regular"])
        self.assertFalse(flags["irregular"])
        self.assertFalse(polygon.is_renderable)

    def test_irregular_polygon_flags(self) -> None:
        points = _make_regular_polygon_points(12)
        points[2].y *= 1.1
        polygon = GenericPolygon(_segments_from_points(points))
        flags = polygon.get_type_flags()
        self.assertFalse(flags["regular"])
        self.assertTrue(flags["irregular"])
        self.assertFalse(polygon.is_renderable)

    def test_minimum_sides_requirement(self) -> None:
        points = _make_regular_polygon_points(10)
        with self.assertRaises(ValueError):
            GenericPolygon(_segments_from_points(points))

    def test_class_name(self) -> None:
        points = _make_regular_polygon_points(11)
        polygon = GenericPolygon(_segments_from_points(points))
        self.assertEqual(polygon.get_class_name(), "GenericPolygon")


if __name__ == "__main__":
    unittest.main()

