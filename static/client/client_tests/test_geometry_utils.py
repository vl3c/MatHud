from __future__ import annotations

import math
import unittest

from utils.geometry_utils import GeometryUtils
from .simple_mock import SimpleMock


def _make_point(name: str, x: float, y: float) -> SimpleMock:
    return SimpleMock(name=name, x=float(x), y=float(y))


def _make_segment(point1: SimpleMock, point2: SimpleMock, name: str = "") -> SimpleMock:
    return SimpleMock(name=name or f"{point1.name}{point2.name}", point1=point1, point2=point2)


def _segments_from_points(points: list[SimpleMock]) -> list[SimpleMock]:
    segments: list[SimpleMock] = []
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        segments.append(_make_segment(point, next_point))
    return segments


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

    # ------------------------------------------------------------------
    # segments_form_polygon / order_segments_into_loop
    # ------------------------------------------------------------------

    def test_segments_form_polygon_positive(self) -> None:
        segments = [
            _make_segment(self.point_a, self.point_b),
            _make_segment(self.point_b, self.point_c),
            _make_segment(self.point_c, self.point_a),
        ]
        self.assertTrue(GeometryUtils.segments_form_polygon(segments))

    def test_segments_form_polygon_negative(self) -> None:
        segments = [
            _make_segment(self.point_a, self.point_b),
            _make_segment(self.point_b, self.point_c),
            _make_segment(self.point_c, self.point_d),
        ]
        self.assertFalse(GeometryUtils.segments_form_polygon(segments))

    def test_order_segments_into_loop_orders_points(self) -> None:
        segments = [
            _make_segment(self.point_b, self.point_c),
            _make_segment(self.point_c, self.point_a),
            _make_segment(self.point_a, self.point_b),
        ]
        ordered = GeometryUtils.order_segments_into_loop(segments)
        self.assertIsNotNone(ordered)
        assert ordered is not None
        self.assertEqual(len(ordered), 3)
        names = [point.name for point in ordered]
        self.assertEqual(set(names), {"A", "B", "C"})

    # ------------------------------------------------------------------
    # Triangle classification
    # ------------------------------------------------------------------

    def test_triangle_type_flags_equilateral(self) -> None:
        side = 1.0
        height = math.sqrt(3) / 2 * side
        points = [
            _make_point("A1", 0.0, 0.0),
            _make_point("B1", side, 0.0),
            _make_point("C1", side / 2, height),
        ]
        flags = GeometryUtils.triangle_type_flags(points)
        self.assertTrue(flags["equilateral"])
        self.assertTrue(flags["isosceles"])
        self.assertFalse(flags["scalene"])
        self.assertFalse(flags["right"])
        self.assertTrue(GeometryUtils.is_equilateral_triangle(points))
        self.assertTrue(GeometryUtils.is_isosceles_triangle(points))
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_equilateral_triangle_from_segments(segments))
        self.assertTrue(GeometryUtils.is_isosceles_triangle_from_segments(segments))
        triangle_flags_from_segments = GeometryUtils.triangle_type_flags_from_segments(segments)
        assert triangle_flags_from_segments is not None
        self.assertTrue(triangle_flags_from_segments["equilateral"])

    def test_triangle_type_flags_right_scalene(self) -> None:
        points = [
            _make_point("A2", 0.0, 0.0),
            _make_point("B2", 3.0, 0.0),
            _make_point("C2", 0.0, 4.0),
        ]
        flags = GeometryUtils.triangle_type_flags(points)
        self.assertFalse(flags["equilateral"])
        self.assertFalse(flags["isosceles"])
        self.assertTrue(flags["scalene"])
        self.assertTrue(flags["right"])
        self.assertTrue(GeometryUtils.is_scalene_triangle(points))
        self.assertTrue(GeometryUtils.is_right_triangle(points))
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_right_triangle_from_segments(segments))
        self.assertTrue(GeometryUtils.is_scalene_triangle_from_segments(segments))
        triangle_flags_from_segments = GeometryUtils.triangle_type_flags_from_segments(segments)
        assert triangle_flags_from_segments is not None
        self.assertTrue(triangle_flags_from_segments["scalene"])

    def test_triangle_type_flags_from_segments_invalid(self) -> None:
        segments = [
            _make_segment(self.point_a, self.point_b),
            _make_segment(self.point_b, self.point_c),
        ]
        self.assertIsNone(GeometryUtils.triangle_type_flags_from_segments(segments))

    # ------------------------------------------------------------------
    # Quadrilateral classification
    # ------------------------------------------------------------------

    def test_quadrilateral_type_flags_square(self) -> None:
        points = [
            _make_point("Q1", 0.0, 0.0),
            _make_point("Q2", 1.0, 0.0),
            _make_point("Q3", 1.0, 1.0),
            _make_point("Q4", 0.0, 1.0),
        ]
        flags = GeometryUtils.quadrilateral_type_flags(points)
        self.assertTrue(flags["square"])
        self.assertTrue(flags["rectangle"])
        self.assertTrue(flags["rhombus"])
        self.assertFalse(flags["irregular"])
        self.assertTrue(GeometryUtils.is_square(points))
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_square_from_segments(segments))
        self.assertFalse(GeometryUtils.is_irregular_quadrilateral_from_segments(segments))

    def test_quadrilateral_type_flags_rectangle(self) -> None:
        points = [
            _make_point("R1", 0.0, 0.0),
            _make_point("R2", 2.0, 0.0),
            _make_point("R3", 2.0, 1.0),
            _make_point("R4", 0.0, 1.0),
        ]
        flags = GeometryUtils.quadrilateral_type_flags(points)
        self.assertFalse(flags["square"])
        self.assertTrue(flags["rectangle"])
        self.assertFalse(flags["rhombus"])
        self.assertFalse(flags["irregular"])
        self.assertTrue(GeometryUtils.is_rectangle(points))
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_rectangle_from_segments(segments))
        self.assertFalse(GeometryUtils.is_square_from_segments(segments))

    def test_quadrilateral_type_flags_rhombus(self) -> None:
        points = [
            _make_point("H1", 0.0, 0.0),
            _make_point("H2", 2.0, 1.0),
            _make_point("H3", 3.0, 3.0),
            _make_point("H4", 1.0, 2.0),
        ]
        flags = GeometryUtils.quadrilateral_type_flags(points)
        self.assertFalse(flags["square"])
        self.assertFalse(flags["rectangle"])
        self.assertTrue(flags["rhombus"])
        self.assertFalse(flags["irregular"])
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_rhombus(points))
        self.assertTrue(GeometryUtils.is_rhombus_from_segments(segments))

    def test_quadrilateral_type_flags_irregular(self) -> None:
        points = [
            _make_point("I1", 0.0, 0.0),
            _make_point("I2", 3.0, 0.5),
            _make_point("I3", 2.0, 2.0),
            _make_point("I4", 0.5, 1.0),
        ]
        flags = GeometryUtils.quadrilateral_type_flags(points)
        self.assertFalse(flags["square"])
        self.assertFalse(flags["rectangle"])
        self.assertFalse(flags["rhombus"])
        self.assertTrue(flags["irregular"])
        self.assertTrue(GeometryUtils.is_irregular_quadrilateral(points))
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_irregular_quadrilateral_from_segments(segments))

    # ------------------------------------------------------------------
    # General polygon regularity
    # ------------------------------------------------------------------

    def test_polygon_flags_regular_pentagon(self) -> None:
        points: list[SimpleMock] = []
        radius = 3.0
        for idx in range(5):
            angle = 2 * math.pi * idx / 5
            points.append(_make_point(f"P{idx}", radius * math.cos(angle), radius * math.sin(angle)))
        flags = GeometryUtils.polygon_flags(points)
        self.assertTrue(flags["regular"])
        self.assertFalse(flags["irregular"])
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_regular_polygon_from_segments(segments))
        self.assertTrue(GeometryUtils.is_regular_polygon(points))
        polygon_flags_from_segments = GeometryUtils.polygon_flags_from_segments(segments)
        assert polygon_flags_from_segments is not None
        self.assertTrue(polygon_flags_from_segments["regular"])

    def test_polygon_flags_irregular_pentagon(self) -> None:
        points: list[SimpleMock] = []
        radius = 3.0
        for idx in range(5):
            angle = 2 * math.pi * idx / 5
            scale = 1.0 if idx != 2 else 1.2
            points.append(_make_point(f"J{idx}", radius * math.cos(angle) * scale, radius * math.sin(angle) * scale))
        flags = GeometryUtils.polygon_flags(points)
        self.assertFalse(flags["regular"])
        self.assertTrue(flags["irregular"])
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_irregular_polygon_from_segments(segments))
        self.assertTrue(GeometryUtils.is_irregular_polygon(points))
        polygon_flags_from_segments = GeometryUtils.polygon_flags_from_segments(segments)
        assert polygon_flags_from_segments is not None
        self.assertTrue(polygon_flags_from_segments["irregular"])

    def test_polygon_flags_regular_examples_three_through_ten(self) -> None:
        for sides in range(3, 11):
            radius = 2.0
            points: list[SimpleMock] = []
            for idx in range(sides):
                angle = 2 * math.pi * idx / sides
                points.append(_make_point(f"R{sides}_{idx}", radius * math.cos(angle), radius * math.sin(angle)))
            flags = GeometryUtils.polygon_flags(points)
            self.assertTrue(flags["regular"], msg=f"Expected regular flag true for {sides}-gon")
            self.assertFalse(flags["irregular"], msg=f"Expected irregular flag false for {sides}-gon")
            segments = _segments_from_points(points)
            seg_flags = GeometryUtils.polygon_flags_from_segments(segments)
            assert seg_flags is not None
            self.assertTrue(seg_flags["regular"], msg=f"Expected regular flag true for {sides}-gon segments")
            self.assertFalse(seg_flags["irregular"], msg=f"Expected irregular flag false for {sides}-gon segments")

    # ------------------------------------------------------------------
    # Polygon side counts
    # ------------------------------------------------------------------

    def test_polygon_side_count_points(self) -> None:
        points = [_make_point("S1", 0.0, 0.0), _make_point("S2", 1.0, 0.0), _make_point("S3", 1.0, 1.0)]
        self.assertEqual(GeometryUtils.polygon_side_count(points), 3)

    def test_polygon_side_count_from_segments(self) -> None:
        points = [
            _make_point("P1", 0.0, 0.0),
            _make_point("P2", 1.0, 0.0),
            _make_point("P3", 1.5, 1.0),
            _make_point("P4", 0.0, 1.0),
            _make_point("P5", -0.5, 0.5),
        ]
        segments = _segments_from_points(points)
        self.assertEqual(GeometryUtils.polygon_side_count_from_segments(segments), 5)

    def test_polygon_side_count_from_segments_invalid(self) -> None:
        segments = [
            _make_segment(self.point_a, self.point_b),
            _make_segment(self.point_b, self.point_c),
        ]
        self.assertIsNone(GeometryUtils.polygon_side_count_from_segments(segments))

    def test_is_polygon_with_sides(self) -> None:
        points = [_make_point("T1", 0.0, 0.0), _make_point("T2", 1.0, 0.0), _make_point("T3", 0.5, 1.0)]
        self.assertTrue(GeometryUtils.is_polygon_with_sides(points, 3))
        self.assertFalse(GeometryUtils.is_polygon_with_sides(points, 4))

    def test_is_polygon_with_sides_from_segments(self) -> None:
        points = [
            _make_point("Q1", 0.0, 0.0),
            _make_point("Q2", 1.0, 0.0),
            _make_point("Q3", 1.0, 1.0),
            _make_point("Q4", 0.0, 1.0),
        ]
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_polygon_with_sides_from_segments(segments, 4))
        self.assertFalse(GeometryUtils.is_polygon_with_sides_from_segments(segments, 3))

    def test_is_triangle_helpers(self) -> None:
        points = [
            _make_point("T1", 0.0, 0.0),
            _make_point("T2", 2.0, 0.0),
            _make_point("T3", 0.5, 1.5),
        ]
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_triangle(points))
        self.assertTrue(GeometryUtils.is_triangle_from_segments(segments))
        self.assertFalse(GeometryUtils.is_quadrilateral(points))

    def test_is_quadrilateral_helpers(self) -> None:
        points = [
            _make_point("Q1", 0.0, 0.0),
            _make_point("Q2", 2.0, 0.0),
            _make_point("Q3", 2.0, 1.0),
            _make_point("Q4", 0.0, 1.0),
        ]
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_quadrilateral(points))
        self.assertTrue(GeometryUtils.is_quadrilateral_from_segments(segments))
        self.assertFalse(GeometryUtils.is_triangle(points))

    def test_is_pentagon_helpers(self) -> None:
        points = [
            _make_point("A", 0.0, 0.0),
            _make_point("B", 1.0, 0.0),
            _make_point("C", 1.5, 1.0),
            _make_point("D", 0.5, 1.5),
            _make_point("E", -0.5, 0.8),
        ]
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_pentagon(points))
        self.assertTrue(GeometryUtils.is_pentagon_from_segments(segments))
        self.assertFalse(GeometryUtils.is_hexagon(points))

    def test_is_hexagon_helpers(self) -> None:
        points = []
        for idx in range(6):
            angle = 2 * math.pi * idx / 6
            points.append(_make_point(f"H{idx}", math.cos(angle), math.sin(angle)))
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_hexagon(points))
        self.assertTrue(GeometryUtils.is_hexagon_from_segments(segments))
        self.assertFalse(GeometryUtils.is_pentagon_from_segments(segments))

    def test_is_heptagon_helpers(self) -> None:
        points = []
        for idx in range(7):
            angle = 2 * math.pi * idx / 7
            points.append(_make_point(f"G{idx}", 1.0 + math.cos(angle), 1.0 + math.sin(angle)))
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_heptagon(points))
        self.assertTrue(GeometryUtils.is_heptagon_from_segments(segments))

    def test_is_octagon_helpers(self) -> None:
        points = []
        for idx in range(8):
            angle = 2 * math.pi * idx / 8
            points.append(_make_point(f"O{idx}", 2.0 * math.cos(angle), 2.0 * math.sin(angle)))
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_octagon(points))
        self.assertTrue(GeometryUtils.is_octagon_from_segments(segments))

    def test_is_nonagon_helpers(self) -> None:
        points = []
        for idx in range(9):
            angle = 2 * math.pi * idx / 9
            points.append(_make_point(f"N{idx}", 1.5 * math.cos(angle), 1.5 * math.sin(angle)))
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_nonagon(points))
        self.assertTrue(GeometryUtils.is_nonagon_from_segments(segments))

    def test_is_decagon_helpers(self) -> None:
        points = []
        for idx in range(10):
            angle = 2 * math.pi * idx / 10
            points.append(_make_point(f"D{idx}", 1.2 * math.cos(angle), 1.2 * math.sin(angle)))
        segments = _segments_from_points(points)
        self.assertTrue(GeometryUtils.is_decagon(points))
        self.assertTrue(GeometryUtils.is_decagon_from_segments(segments))


if __name__ == "__main__":
    unittest.main()

