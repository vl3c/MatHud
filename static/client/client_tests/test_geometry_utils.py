from __future__ import annotations

import unittest

from utils.geometry_utils import GeometryUtils
from .simple_mock import SimpleMock


def _make_point(name: str, x: float, y: float) -> SimpleMock:
    return SimpleMock(name=name, x=float(x), y=float(y))


def _make_segment(point1: SimpleMock, point2: SimpleMock, name: str = "") -> SimpleMock:
    return SimpleMock(name=name or f"{point1.name}{point2.name}", point1=point1, point2=point2)


class TestGeometryUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.point_a = _make_point("A", 0.0, 0.0)
        self.point_b = _make_point("B", 2.0, 0.0)
        self.point_c = _make_point("C", 1.0, 2.0)
        self.point_d = _make_point("D", -1.0, 1.0)
        self.point_e = _make_point("E", 0.0, 3.0)

    # ------------------------------------------------------------------
    # segments_form_closed_loop
    # ------------------------------------------------------------------

    def test_segments_form_closed_loop_triangle(self) -> None:
        segments = [
            _make_segment(self.point_a, self.point_b),
            _make_segment(self.point_b, self.point_c),
            _make_segment(self.point_c, self.point_a),
        ]
        self.assertTrue(GeometryUtils.segments_form_closed_loop(segments))

    def test_segments_form_closed_loop_requires_degree_two(self) -> None:
        segments = [
            _make_segment(self.point_a, self.point_b),
            _make_segment(self.point_b, self.point_c),
            _make_segment(self.point_c, self.point_a),
            _make_segment(self.point_a, self.point_d),
        ]
        self.assertFalse(GeometryUtils.segments_form_closed_loop(segments))

    def test_segments_form_closed_loop_disconnected(self) -> None:
        segments = [
            _make_segment(self.point_a, self.point_b),
            _make_segment(self.point_b, self.point_c),
            _make_segment(self.point_d, self.point_e),
        ]
        self.assertFalse(GeometryUtils.segments_form_closed_loop(segments))

    def test_segments_form_closed_loop_requires_three_segments(self) -> None:
        segments = [
            _make_segment(self.point_a, self.point_b),
            _make_segment(self.point_b, self.point_c),
        ]
        self.assertFalse(GeometryUtils.segments_form_closed_loop(segments))

    # ------------------------------------------------------------------
    # order_segments_into_loop
    # ------------------------------------------------------------------

    def test_order_segments_into_loop_triangle(self) -> None:
        segments = [
            _make_segment(self.point_b, self.point_c),
            _make_segment(self.point_a, self.point_b),
            _make_segment(self.point_c, self.point_a),
        ]
        ordered_points = GeometryUtils.order_segments_into_loop(segments)
        self.assertIsNotNone(ordered_points)
        names = [point.name for point in ordered_points]
        self.assertEqual(set(names), {"A", "B", "C"})
        self.assertEqual(len(names), 3)

    def test_order_segments_into_loop_invalid_returns_none(self) -> None:
        segments = [
            _make_segment(self.point_a, self.point_b),
            _make_segment(self.point_b, self.point_c),
            _make_segment(self.point_c, self.point_d),
        ]
        self.assertIsNone(GeometryUtils.order_segments_into_loop(segments))

    # ------------------------------------------------------------------
    # polygon_math_coordinates_from_segments
    # ------------------------------------------------------------------

    def test_polygon_math_coordinates_from_segments_triangle(self) -> None:
        segments = [
            _make_segment(self.point_a, self.point_b),
            _make_segment(self.point_b, self.point_c),
            _make_segment(self.point_c, self.point_a),
        ]
        coords = GeometryUtils.polygon_math_coordinates_from_segments(segments)
        self.assertIsNotNone(coords)
        assert coords is not None
        self.assertEqual(len(coords), 3)
        self.assertIn((0.0, 0.0), coords)
        self.assertIn((2.0, 0.0), coords)
        self.assertIn((1.0, 2.0), coords)

    def test_polygon_math_coordinates_invalid_returns_none(self) -> None:
        segments = [
            _make_segment(self.point_a, self.point_b),
            _make_segment(self.point_b, self.point_c),
        ]
        self.assertIsNone(GeometryUtils.polygon_math_coordinates_from_segments(segments))


if __name__ == "__main__":
    unittest.main()

