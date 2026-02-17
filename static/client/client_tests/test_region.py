from __future__ import annotations

import math
import unittest

from geometry import (
    LineSegment,
    CompositePath,
    Region,
)
from utils.geometry_utils import GeometryUtils


class TestRegion(unittest.TestCase):
    def test_region_from_square_points(self) -> None:
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        region = Region.from_points(points)
        self.assertIsNotNone(region)
        self.assertEqual(len(region.holes), 0)

    def test_region_from_triangle_points(self) -> None:
        points = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)]
        region = Region.from_points(points)
        self.assertIsNotNone(region)

    def test_region_from_circle(self) -> None:
        region = Region.from_circle((0.0, 0.0), 1.0)
        self.assertIsNotNone(region)
        expected_area = math.pi * 1.0 * 1.0
        self.assertAlmostEqual(region.area(), expected_area, places=1)

    def test_region_from_ellipse(self) -> None:
        region = Region.from_ellipse((0.0, 0.0), 2.0, 1.0)
        self.assertIsNotNone(region)
        expected_area = math.pi * 2.0 * 1.0
        self.assertAlmostEqual(region.area(), expected_area, places=1)

    def test_region_requires_closed_path(self) -> None:
        seg = LineSegment((0.0, 0.0), (1.0, 0.0))
        path = CompositePath([seg])
        with self.assertRaises(ValueError):
            Region(path)

    def test_square_area(self) -> None:
        points = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        region = Region.from_points(points)
        self.assertAlmostEqual(region.area(), 4.0, places=5)

    def test_triangle_area(self) -> None:
        points = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]
        region = Region.from_points(points)
        self.assertAlmostEqual(region.area(), 6.0, places=5)

    def test_signed_area_ccw(self) -> None:
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        region = Region.from_points(points)
        self.assertGreater(region.signed_area(), 0)

    def test_signed_area_cw(self) -> None:
        points = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        region = Region.from_points(points)
        self.assertLess(region.signed_area(), 0)

    def test_area_with_hole(self) -> None:
        outer = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
        outer_path = CompositePath.from_points(outer + [outer[0]])

        hole = [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]
        hole_path = CompositePath.from_points(hole + [hole[0]])

        region = Region(outer_path, [hole_path])
        outer_area = 16.0
        hole_area = 4.0
        expected = outer_area - hole_area
        self.assertAlmostEqual(region.area(), expected, places=5)

    def test_add_hole(self) -> None:
        outer = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
        outer_path = CompositePath.from_points(outer + [outer[0]])
        region = Region(outer_path)
        self.assertEqual(len(region.holes), 0)

        hole = [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]
        hole_path = CompositePath.from_points(hole + [hole[0]])
        region.add_hole(hole_path)
        self.assertEqual(len(region.holes), 1)

    def test_add_hole_requires_closed(self) -> None:
        outer = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        outer_path = CompositePath.from_points(outer + [outer[0]])
        region = Region(outer_path)

        open_path = CompositePath([LineSegment((0.5, 0.5), (0.6, 0.6))])
        with self.assertRaises(ValueError):
            region.add_hole(open_path)

    def test_contains_point_inside_square(self) -> None:
        points = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        region = Region.from_points(points)
        self.assertTrue(region.contains_point(1.0, 1.0))

    def test_contains_point_outside_square(self) -> None:
        points = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        region = Region.from_points(points)
        self.assertFalse(region.contains_point(5.0, 5.0))

    def test_contains_point_in_hole(self) -> None:
        outer = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
        outer_path = CompositePath.from_points(outer + [outer[0]])

        hole = [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]
        hole_path = CompositePath.from_points(hole + [hole[0]])

        region = Region(outer_path, [hole_path])
        self.assertTrue(region.contains_point(0.5, 0.5))
        self.assertFalse(region.contains_point(2.0, 2.0))

    def test_contains_point_circle(self) -> None:
        region = Region.from_circle((0.0, 0.0), 1.0)
        self.assertTrue(region.contains_point(0.0, 0.0))
        self.assertTrue(region.contains_point(0.5, 0.0))
        self.assertFalse(region.contains_point(2.0, 0.0))

    def test_intersection_overlapping_squares(self) -> None:
        sq1 = Region.from_points([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
        sq2 = Region.from_points([(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])

        result = sq1.intersection(sq2)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.area(), 1.0, places=1)

    def test_intersection_non_overlapping(self) -> None:
        sq1 = Region.from_points([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        sq2 = Region.from_points([(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)])

        result = sq1.intersection(sq2)
        self.assertIsNone(result)

    def test_union_overlapping_returns_correct_area(self) -> None:
        sq1 = Region.from_points([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
        sq2 = Region.from_points([(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])

        result = sq1.union(sq2)
        # Two 2x2 squares with 1x1 overlap: 4 + 4 - 1 = 7
        self.assertAlmostEqual(result.area(), 7.0, places=1)

    def test_union_disjoint_returns_correct_area(self) -> None:
        sq1 = Region.from_points([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        sq2 = Region.from_points([(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)])

        result = sq1.union(sq2)
        # Two disjoint 1x1 squares: 1 + 1 = 2
        self.assertAlmostEqual(result.area(), 2.0, places=1)

    def test_difference_overlapping(self) -> None:
        sq1 = Region.from_points([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
        sq2 = Region.from_points([(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])

        result = sq1.difference(sq2)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.holes), 1)

    def test_difference_non_overlapping(self) -> None:
        sq1 = Region.from_points([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        sq2 = Region.from_points([(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)])

        result = sq1.difference(sq2)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.holes), 0)

    def test_repr(self) -> None:
        region = Region.from_points([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
        repr_str = repr(region)
        self.assertIn("Region", repr_str)
        self.assertIn("boundary_elements", repr_str)

    def test_circle_area_exact(self) -> None:
        for radius in [0.5, 1.0, 2.0, 5.0]:
            region = Region.from_circle((0.0, 0.0), radius)
            expected = math.pi * radius * radius
            self.assertAlmostEqual(region.area(), expected, places=1)

    def test_ellipse_area_exact(self) -> None:
        region = Region.from_ellipse((0.0, 0.0), 3.0, 2.0)
        expected = math.pi * 3.0 * 2.0
        self.assertAlmostEqual(region.area(), expected, places=1)


class TestAreaUtilities(unittest.TestCase):
    def test_polygon_area_square(self) -> None:
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        area = GeometryUtils.polygon_area(points)
        self.assertAlmostEqual(abs(area), 1.0, places=5)

    def test_polygon_area_triangle(self) -> None:
        points = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)]
        area = GeometryUtils.polygon_area(points)
        self.assertAlmostEqual(abs(area), 2.0, places=5)

    def test_polygon_area_signed(self) -> None:
        ccw = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        cw = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]

        area_ccw = GeometryUtils.polygon_area(ccw)
        area_cw = GeometryUtils.polygon_area(cw)

        self.assertGreater(area_ccw, 0)
        self.assertLess(area_cw, 0)
        self.assertAlmostEqual(abs(area_ccw), abs(area_cw), places=5)

    def test_circular_sector_area(self) -> None:
        area = GeometryUtils.circular_sector_area(1.0, math.pi / 2)
        expected = 0.5 * 1.0 * (math.pi / 2)
        self.assertAlmostEqual(area, expected, places=5)

    def test_line_segment_area_contribution(self) -> None:
        area = GeometryUtils.line_segment_area_contribution((0.0, 0.0), (1.0, 0.0))
        self.assertAlmostEqual(area, 0.0, places=5)

        area = GeometryUtils.line_segment_area_contribution((0.0, 0.0), (0.0, 1.0))
        self.assertAlmostEqual(area, 0.0, places=5)

        area = GeometryUtils.line_segment_area_contribution((0.0, 0.0), (1.0, 1.0))
        self.assertAlmostEqual(area, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
