"""Tests for RelationInspector — geometric relation verification."""

from __future__ import annotations

import math
import unittest
from typing import Dict, Any

from canvas import Canvas
from managers.polygon_type import PolygonType


class _RelationTestBase(unittest.TestCase):
    """Shared setUp for relation inspector tests."""

    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)

    def _inspect(
        self,
        operation: str,
        names: list[str],
        types: list[str],
    ) -> Dict[str, Any]:
        return self.canvas.inspect_relation(
            operation=operation,
            objects=names,
            object_types=types,
        )


# ------------------------------------------------------------------
# Parallel
# ------------------------------------------------------------------


class TestParallel(_RelationTestBase):
    def test_horizontal_parallel(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        s2 = self.canvas.create_segment(0, 2, 4, 2, name="CD")
        res = self._inspect("parallel", [s1.name, s2.name], ["segment", "segment"])
        self.assertTrue(res["result"])
        self.assertAlmostEqual(res["details"]["angle_between"], 0.0, places=4)

    def test_vertical_parallel(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 0, 5, name="AB")
        s2 = self.canvas.create_segment(3, 0, 3, 5, name="CD")
        res = self._inspect("parallel", [s1.name, s2.name], ["segment", "segment"])
        self.assertTrue(res["result"])

    def test_not_parallel(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        s2 = self.canvas.create_segment(5, 0, 5, 4, name="CD")
        res = self._inspect("parallel", [s1.name, s2.name], ["segment", "segment"])
        self.assertFalse(res["result"])

    def test_antiparallel(self) -> None:
        """Opposite direction is still parallel."""
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        s2 = self.canvas.create_segment(4, 2, 0, 2, name="CD")
        res = self._inspect("parallel", [s1.name, s2.name], ["segment", "segment"])
        self.assertTrue(res["result"])

    def test_near_threshold(self) -> None:
        """Very small angle should still be detected as not parallel."""
        s1 = self.canvas.create_segment(0, 0, 1000, 0, name="AB")
        s2 = self.canvas.create_segment(10, 10, 1010, 11, name="CD")
        res = self._inspect("parallel", [s1.name, s2.name], ["segment", "segment"])
        self.assertFalse(res["result"])

    def test_zero_length_segment_error(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        s2 = self.canvas.create_segment(2, 2, 2, 2, name="CD")
        res = self._inspect("parallel", [s1.name, s2.name], ["segment", "segment"])
        self.assertIn("error", res)

    def test_symmetry(self) -> None:
        """parallel(A, B) == parallel(B, A)."""
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        s2 = self.canvas.create_segment(0, 2, 4, 2, name="CD")
        r1 = self._inspect("parallel", [s1.name, s2.name], ["segment", "segment"])
        r2 = self._inspect("parallel", [s2.name, s1.name], ["segment", "segment"])
        self.assertEqual(r1["result"], r2["result"])


# ------------------------------------------------------------------
# Perpendicular
# ------------------------------------------------------------------


class TestPerpendicular(_RelationTestBase):
    def test_axes_cross(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        s2 = self.canvas.create_segment(5, 0, 5, 4, name="CD")
        res = self._inspect("perpendicular", [s1.name, s2.name], ["segment", "segment"])
        self.assertTrue(res["result"])
        self.assertAlmostEqual(res["details"]["angle_between"], 90.0, places=4)

    def test_45_and_135(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 1, 1, name="AB")
        s2 = self.canvas.create_segment(5, 5, 4, 6, name="CD")
        res = self._inspect("perpendicular", [s1.name, s2.name], ["segment", "segment"])
        self.assertTrue(res["result"])

    def test_not_perpendicular(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        s2 = self.canvas.create_segment(5, 0, 8, 1, name="CD")
        res = self._inspect("perpendicular", [s1.name, s2.name], ["segment", "segment"])
        self.assertFalse(res["result"])

    def test_near_threshold(self) -> None:
        """Just off 90 degrees should not pass."""
        s1 = self.canvas.create_segment(0, 0, 1000, 0, name="AB")
        s2 = self.canvas.create_segment(10, 10, 11, 1010, name="CD")
        res = self._inspect("perpendicular", [s1.name, s2.name], ["segment", "segment"])
        self.assertFalse(res["result"])

    def test_zero_length_error(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        s2 = self.canvas.create_segment(1, 1, 1, 1, name="CD")
        res = self._inspect("perpendicular", [s1.name, s2.name], ["segment", "segment"])
        self.assertIn("error", res)


# ------------------------------------------------------------------
# Collinear
# ------------------------------------------------------------------


class TestCollinear(_RelationTestBase):
    def test_three_on_x_axis(self) -> None:
        self.canvas.create_point(0, 0, name="A")
        self.canvas.create_point(3, 0, name="B")
        self.canvas.create_point(7, 0, name="C")
        res = self._inspect("collinear", ["A", "B", "C"], ["point", "point", "point"])
        self.assertTrue(res["result"])

    def test_not_collinear(self) -> None:
        self.canvas.create_point(0, 0, name="A")
        self.canvas.create_point(3, 0, name="B")
        self.canvas.create_point(1, 5, name="C")
        res = self._inspect("collinear", ["A", "B", "C"], ["point", "point", "point"])
        self.assertFalse(res["result"])

    def test_four_collinear(self) -> None:
        self.canvas.create_point(0, 0, name="A")
        self.canvas.create_point(1, 1, name="B")
        self.canvas.create_point(2, 2, name="C")
        self.canvas.create_point(5, 5, name="D")
        res = self._inspect(
            "collinear",
            ["A", "B", "C", "D"],
            ["point", "point", "point", "point"],
        )
        self.assertTrue(res["result"])

    def test_nearly_coincident_points(self) -> None:
        """Very close points should still be considered collinear."""
        self.canvas.create_point(3, 3, name="A")
        self.canvas.create_point(3.0000001, 3.0000001, name="B")
        self.canvas.create_point(3.0000002, 3.0000002, name="C")
        res = self._inspect("collinear", ["A", "B", "C"], ["point", "point", "point"])
        self.assertTrue(res["result"])

    def test_large_coordinates_nearly_collinear(self) -> None:
        """Scale-invariant collinearity: tiny angular deviation at large scale."""
        self.canvas.create_point(0, 0, name="A")
        self.canvas.create_point(1e6, 0, name="B")
        # 1e-3 offset at distance 1e6 → angle ~1e-9 rad, should be collinear
        self.canvas.create_point(1e6, 1e-3, name="C")
        res = self._inspect("collinear", ["A", "B", "C"], ["point", "point", "point"])
        self.assertTrue(res["result"])

    def test_too_few_points(self) -> None:
        self.canvas.create_point(0, 0, name="A")
        self.canvas.create_point(1, 1, name="B")
        res = self._inspect("collinear", ["A", "B"], ["point", "point"])
        self.assertIn("error", res)


# ------------------------------------------------------------------
# Concyclic
# ------------------------------------------------------------------


class TestConcyclic(_RelationTestBase):
    def test_four_on_unit_circle(self) -> None:
        r = 5.0
        self.canvas.create_point(r, 0, name="A")
        self.canvas.create_point(0, r, name="B")
        self.canvas.create_point(-r, 0, name="C")
        self.canvas.create_point(0, -r, name="D")
        res = self._inspect(
            "concyclic",
            ["A", "B", "C", "D"],
            ["point", "point", "point", "point"],
        )
        self.assertTrue(res["result"])

    def test_not_concyclic(self) -> None:
        self.canvas.create_point(5, 0, name="A")
        self.canvas.create_point(0, 5, name="B")
        self.canvas.create_point(-5, 0, name="C")
        self.canvas.create_point(1, 1, name="D")
        res = self._inspect(
            "concyclic",
            ["A", "B", "C", "D"],
            ["point", "point", "point", "point"],
        )
        self.assertFalse(res["result"])

    def test_collinear_first_three(self) -> None:
        self.canvas.create_point(0, 0, name="A")
        self.canvas.create_point(1, 0, name="B")
        self.canvas.create_point(2, 0, name="C")
        self.canvas.create_point(0, 1, name="D")
        res = self._inspect(
            "concyclic",
            ["A", "B", "C", "D"],
            ["point", "point", "point", "point"],
        )
        self.assertFalse(res["result"])


# ------------------------------------------------------------------
# Equal Length
# ------------------------------------------------------------------


class TestEqualLength(_RelationTestBase):
    def test_equal_segments(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 3, 4, name="AB")
        s2 = self.canvas.create_segment(10, 10, 13, 14, name="CD")
        res = self._inspect("equal_length", [s1.name, s2.name], ["segment", "segment"])
        self.assertTrue(res["result"])
        self.assertAlmostEqual(res["details"]["length1"], 5.0, places=5)
        self.assertAlmostEqual(res["details"]["length2"], 5.0, places=5)

    def test_unequal_segments(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 3, 0, name="AB")
        s2 = self.canvas.create_segment(5, 0, 10, 0, name="CD")
        res = self._inspect("equal_length", [s1.name, s2.name], ["segment", "segment"])
        self.assertFalse(res["result"])

    def test_near_threshold(self) -> None:
        """Tiny difference should count as not equal."""
        s1 = self.canvas.create_segment(0, 0, 10, 0, name="AB")
        s2 = self.canvas.create_segment(20, 0, 30.001, 0, name="CD")
        res = self._inspect("equal_length", [s1.name, s2.name], ["segment", "segment"])
        self.assertFalse(res["result"])


# ------------------------------------------------------------------
# Similar Triangles
# ------------------------------------------------------------------


class TestSimilarTriangles(_RelationTestBase):
    def test_scaled_copy(self) -> None:
        t1 = self.canvas.create_polygon(
            [(0, 0), (3, 0), (0, 4)],
            polygon_type=PolygonType.TRIANGLE,
            name="ABC",
        )
        t2 = self.canvas.create_polygon(
            [(10, 10), (16, 10), (10, 18)],
            polygon_type=PolygonType.TRIANGLE,
            name="DEF",
        )
        res = self._inspect("similar", [t1.name, t2.name], ["triangle", "triangle"])
        self.assertTrue(res["result"])

    def test_not_similar(self) -> None:
        t1 = self.canvas.create_polygon(
            [(0, 0), (3, 0), (0, 4)],
            polygon_type=PolygonType.TRIANGLE,
            name="ABC",
        )
        t2 = self.canvas.create_polygon(
            [(10, 10), (20, 10), (10, 11)],
            polygon_type=PolygonType.TRIANGLE,
            name="DEF",
        )
        res = self._inspect("similar", [t1.name, t2.name], ["triangle", "triangle"])
        self.assertFalse(res["result"])


# ------------------------------------------------------------------
# Congruent Triangles
# ------------------------------------------------------------------


class TestCongruentTriangles(_RelationTestBase):
    def test_same_shape(self) -> None:
        t1 = self.canvas.create_polygon(
            [(0, 0), (3, 0), (0, 4)],
            polygon_type=PolygonType.TRIANGLE,
            name="ABC",
        )
        t2 = self.canvas.create_polygon(
            [(10, 10), (13, 10), (10, 14)],
            polygon_type=PolygonType.TRIANGLE,
            name="DEF",
        )
        res = self._inspect("congruent", [t1.name, t2.name], ["triangle", "triangle"])
        self.assertTrue(res["result"])

    def test_similar_not_congruent(self) -> None:
        t1 = self.canvas.create_polygon(
            [(0, 0), (3, 0), (0, 4)],
            polygon_type=PolygonType.TRIANGLE,
            name="ABC",
        )
        t2 = self.canvas.create_polygon(
            [(10, 10), (16, 10), (10, 18)],
            polygon_type=PolygonType.TRIANGLE,
            name="DEF",
        )
        res = self._inspect("congruent", [t1.name, t2.name], ["triangle", "triangle"])
        self.assertFalse(res["result"])


# ------------------------------------------------------------------
# Tangent
# ------------------------------------------------------------------


class TestTangent(_RelationTestBase):
    def test_segment_tangent_to_circle(self) -> None:
        c1 = self.canvas.create_circle(0, 0, 5)
        # Horizontal line y = 5 is tangent to circle centered at origin with radius 5
        s1 = self.canvas.create_segment(-10, 5, 10, 5, name="AB")
        res = self._inspect("tangent", [s1.name, c1.name], ["segment", "circle"])
        self.assertTrue(res["result"])

    def test_segment_not_tangent(self) -> None:
        c1 = self.canvas.create_circle(0, 0, 5)
        s1 = self.canvas.create_segment(-10, 3, 10, 3, name="AB")
        res = self._inspect("tangent", [s1.name, c1.name], ["segment", "circle"])
        self.assertFalse(res["result"])

    def test_circles_externally_tangent(self) -> None:
        c1 = self.canvas.create_circle(0, 0, 3)
        c2 = self.canvas.create_circle(7, 0, 4)
        res = self._inspect("tangent", [c1.name, c2.name], ["circle", "circle"])
        self.assertTrue(res["result"])
        self.assertTrue(res["details"]["externally_tangent"])

    def test_circles_internally_tangent(self) -> None:
        c1 = self.canvas.create_circle(0, 0, 5)
        c2 = self.canvas.create_circle(2, 0, 3)
        res = self._inspect("tangent", [c1.name, c2.name], ["circle", "circle"])
        self.assertTrue(res["result"])
        self.assertTrue(res["details"]["internally_tangent"])

    def test_circles_not_tangent(self) -> None:
        c1 = self.canvas.create_circle(0, 0, 3)
        c2 = self.canvas.create_circle(10, 0, 3)
        res = self._inspect("tangent", [c1.name, c2.name], ["circle", "circle"])
        self.assertFalse(res["result"])

    def test_reversed_order(self) -> None:
        """circle first, segment second — should still work."""
        c1 = self.canvas.create_circle(0, 0, 5)
        s1 = self.canvas.create_segment(-10, 5, 10, 5, name="AB")
        res = self._inspect("tangent", [c1.name, s1.name], ["circle", "segment"])
        self.assertTrue(res["result"])

    def test_segment_far_from_tangent_point(self) -> None:
        """Extended line is tangent but the segment itself is far away."""
        c1 = self.canvas.create_circle(0, 0, 5)
        # y = 5 line is tangent, but this segment is at x=[100, 200]
        s1 = self.canvas.create_segment(100, 5, 200, 5, name="AB")
        res = self._inspect("tangent", [s1.name, c1.name], ["segment", "circle"])
        self.assertFalse(res["result"])
        self.assertIn("does not lie on the segment", res["explanation"])


# ------------------------------------------------------------------
# Concurrent
# ------------------------------------------------------------------


class TestConcurrent(_RelationTestBase):
    def test_three_lines_through_origin(self) -> None:
        s1 = self.canvas.create_segment(-5, 0, 5, 0, name="AB")
        s2 = self.canvas.create_segment(0, -5, 0, 5, name="CD")
        s3 = self.canvas.create_segment(-5, -5, 5, 5, name="EF")
        res = self._inspect(
            "concurrent",
            [s1.name, s2.name, s3.name],
            ["segment", "segment", "segment"],
        )
        self.assertTrue(res["result"])
        ix, iy = res["details"]["intersection"]
        self.assertAlmostEqual(ix, 0.0, places=4)
        self.assertAlmostEqual(iy, 0.0, places=4)

    def test_non_concurrent(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        s2 = self.canvas.create_segment(5, 0, 5, 4, name="CD")
        s3 = self.canvas.create_segment(1, 1, 5, 2, name="EF")
        res = self._inspect(
            "concurrent",
            [s1.name, s2.name, s3.name],
            ["segment", "segment", "segment"],
        )
        self.assertFalse(res["result"])

    def test_parallel_pair_among_three(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        s2 = self.canvas.create_segment(0, 2, 4, 2, name="CD")
        s3 = self.canvas.create_segment(5, 0, 5, 4, name="EF")
        res = self._inspect(
            "concurrent",
            [s1.name, s2.name, s3.name],
            ["segment", "segment", "segment"],
        )
        self.assertFalse(res["result"])


# ------------------------------------------------------------------
# Point on Line
# ------------------------------------------------------------------


class TestPointOnLine(_RelationTestBase):
    def test_on_extended_line(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 2, 2, name="AB")
        self.canvas.create_point(5, 5, name="P")
        res = self._inspect("point_on_line", ["P", s1.name], ["point", "segment"])
        self.assertTrue(res["result"])

    def test_not_on_line(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        self.canvas.create_point(2, 3, name="P")
        res = self._inspect("point_on_line", ["P", s1.name], ["point", "segment"])
        self.assertFalse(res["result"])

    def test_on_endpoint(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        # Use point A which is the first endpoint of the segment
        res = self._inspect("point_on_line", ["A", s1.name], ["point", "segment"])
        self.assertTrue(res["result"])

    def test_reversed_order(self) -> None:
        """segment first, point second."""
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        self.canvas.create_point(2, 0, name="P")
        res = self._inspect("point_on_line", [s1.name, "P"], ["segment", "point"])
        self.assertTrue(res["result"])


# ------------------------------------------------------------------
# Point on Circle
# ------------------------------------------------------------------


class TestPointOnCircle(_RelationTestBase):
    def test_on_circle(self) -> None:
        c1 = self.canvas.create_circle(0, 0, 5)
        self.canvas.create_point(3, 4, name="P")
        res = self._inspect("point_on_circle", ["P", c1.name], ["point", "circle"])
        self.assertTrue(res["result"])

    def test_not_on_circle(self) -> None:
        c1 = self.canvas.create_circle(0, 0, 5)
        self.canvas.create_point(1, 1, name="P")
        res = self._inspect("point_on_circle", ["P", c1.name], ["point", "circle"])
        self.assertFalse(res["result"])

    def test_reversed_order(self) -> None:
        c1 = self.canvas.create_circle(0, 0, 5)
        self.canvas.create_point(5, 0, name="P")
        res = self._inspect("point_on_circle", [c1.name, "P"], ["circle", "point"])
        self.assertTrue(res["result"])


# ------------------------------------------------------------------
# Auto Inspect
# ------------------------------------------------------------------


class TestAutoInspect(_RelationTestBase):
    def test_two_parallel_equal_segments(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        s2 = self.canvas.create_segment(0, 3, 4, 3, name="CD")
        res = self._inspect("auto", [s1.name, s2.name], ["segment", "segment"])
        self.assertEqual(res["operation"], "auto")
        checks = res["details"]["checks_run"]
        self.assertIn("parallel", checks)
        self.assertIn("perpendicular", checks)
        self.assertIn("equal_length", checks)
        # Both parallel and equal length should be true
        true_ops = [r["operation"] for r in res["details"]["results"] if r.get("result") is True]
        self.assertIn("parallel", true_ops)
        self.assertIn("equal_length", true_ops)

    def test_symmetry_ab_ba(self) -> None:
        """Auto results for (s1, s2) and (s2, s1) agree on truth values."""
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        s2 = self.canvas.create_segment(0, 3, 4, 3, name="CD")
        r1 = self._inspect("auto", [s1.name, s2.name], ["segment", "segment"])
        r2 = self._inspect("auto", [s2.name, s1.name], ["segment", "segment"])
        truth1 = {r["operation"]: r["result"] for r in r1["details"]["results"]}
        truth2 = {r["operation"]: r["result"] for r in r2["details"]["results"]}
        self.assertEqual(truth1, truth2)


# ------------------------------------------------------------------
# Error cases
# ------------------------------------------------------------------


class TestInspectErrors(_RelationTestBase):
    def test_wrong_object_count(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        res = self._inspect("parallel", [s1.name], ["segment"])
        self.assertIn("error", res)

    def test_nonexistent_name(self) -> None:
        res = self._inspect("parallel", ["nope1", "nope2"], ["segment", "segment"])
        self.assertIn("error", res)

    def test_mismatched_lengths(self) -> None:
        res = self._inspect("parallel", ["AB", "CD"], ["segment"])
        self.assertIn("error", res)

    def test_unsupported_operation(self) -> None:
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        s2 = self.canvas.create_segment(0, 2, 4, 2, name="CD")
        res = self._inspect("unknown_op", [s1.name, s2.name], ["segment", "segment"])
        self.assertIn("error", res)

    def test_tolerance_in_result(self) -> None:
        """Every successful result should include tolerance_used."""
        s1 = self.canvas.create_segment(0, 0, 4, 0, name="AB")
        s2 = self.canvas.create_segment(0, 2, 4, 2, name="CD")
        res = self._inspect("parallel", [s1.name, s2.name], ["segment", "segment"])
        self.assertIn("tolerance_used", res)
        self.assertIsInstance(res["tolerance_used"], float)
