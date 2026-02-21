from __future__ import annotations

import copy
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

    def test_get_state(self) -> None:
        points = _make_regular_polygon_points(11)
        polygon = GenericPolygon(_segments_from_points(points))
        state = polygon.get_state()
        self.assertEqual(state["name"], polygon.name)
        self.assertIn("args", state)
        self.assertEqual(len(state["args"]), 11)
        self.assertIn("types", state)
        self.assertIn("polygon", state["types"])

    def test_get_vertices(self) -> None:
        points = _make_regular_polygon_points(11)
        polygon = GenericPolygon(_segments_from_points(points))
        vertices = polygon.get_vertices()
        self.assertEqual(len(vertices), 11)

    def test_get_segments(self) -> None:
        points = _make_regular_polygon_points(11)
        segments = _segments_from_points(points)
        polygon = GenericPolygon(segments)
        retrieved = polygon.get_segments()
        self.assertEqual(len(retrieved), 11)

    def test_deepcopy(self) -> None:
        points = _make_regular_polygon_points(11)
        polygon = GenericPolygon(_segments_from_points(points), color="red")
        polygon_copy = copy.deepcopy(polygon)
        self.assertIsNot(polygon_copy, polygon)
        self.assertEqual(polygon_copy.color, polygon.color)
        self.assertEqual(polygon_copy.name, polygon.name)

    def test_update_color(self) -> None:
        points = _make_regular_polygon_points(11)
        polygon = GenericPolygon(_segments_from_points(points), color="blue")
        polygon.update_color("green")
        self.assertEqual(polygon.color, "green")
        for segment in polygon.get_segments():
            self.assertEqual(segment.color, "green")

    def test_is_regular_helper(self) -> None:
        points = _make_regular_polygon_points(11)
        polygon = GenericPolygon(_segments_from_points(points))
        self.assertTrue(polygon.is_regular())
        self.assertFalse(polygon.is_irregular())

    def test_is_irregular_helper(self) -> None:
        points = _make_regular_polygon_points(12)
        points[2].y *= 1.1
        polygon = GenericPolygon(_segments_from_points(points))
        self.assertFalse(polygon.is_regular())
        self.assertTrue(polygon.is_irregular())

    def test_translate(self) -> None:
        points = _make_regular_polygon_points(11)
        polygon = GenericPolygon(_segments_from_points(points))
        original_x = points[0].x
        original_y = points[0].y
        polygon.translate(5.0, 3.0)
        vertices = list(polygon.get_vertices())
        translated = next(v for v in vertices if v.name == "P0")
        self.assertAlmostEqual(translated.x, original_x + 5.0, places=6)
        self.assertAlmostEqual(translated.y, original_y + 3.0, places=6)

    def test_disconnected_segments_raise(self) -> None:
        points = _make_regular_polygon_points(11)
        segments = _segments_from_points(points)
        disconnected = Point(100.0, 100.0, name="X")
        segments[3] = Segment(disconnected, Point(101.0, 101.0, name="Y"))
        with self.assertRaises(ValueError):
            GenericPolygon(segments)

    def test_get_state_types(self) -> None:
        points = _make_regular_polygon_points(11)
        polygon = GenericPolygon(_segments_from_points(points))
        state = polygon.get_state()
        self.assertEqual(state["types"][0], "polygon")
        self.assertIn("regular", state["types"])

    def test_large_polygon_20_sides(self) -> None:
        points = _make_regular_polygon_points(20)
        polygon = GenericPolygon(_segments_from_points(points))
        self.assertEqual(len(polygon.get_vertices()), 20)
        self.assertEqual(len(polygon.get_segments()), 20)
        self.assertTrue(polygon.is_regular())


if __name__ == "__main__":
    unittest.main()
