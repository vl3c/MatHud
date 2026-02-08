from __future__ import annotations

import math
import unittest

from geometry import (
    LineSegment,
    CircularArc,
    EllipticalArc,
    CompositePath,
    line_line_intersection,
    line_circle_intersection,
    line_ellipse_intersection,
    circle_circle_intersection,
    circle_ellipse_intersection,
    ellipse_ellipse_intersection,
    element_element_intersection,
    path_path_intersections,
)


class TestIntersections(unittest.TestCase):

    def _assert_point_near(
        self,
        actual: tuple[float, float],
        expected: tuple[float, float],
        places: int = 5
    ) -> None:
        self.assertAlmostEqual(actual[0], expected[0], places=places)
        self.assertAlmostEqual(actual[1], expected[1], places=places)

    def test_line_line_crossing(self) -> None:
        seg1 = LineSegment((0.0, 0.0), (2.0, 2.0))
        seg2 = LineSegment((0.0, 2.0), (2.0, 0.0))
        result = line_line_intersection(seg1, seg2)
        self.assertEqual(len(result), 1)
        self._assert_point_near(result[0], (1.0, 1.0))

    def test_line_line_parallel(self) -> None:
        seg1 = LineSegment((0.0, 0.0), (2.0, 0.0))
        seg2 = LineSegment((0.0, 1.0), (2.0, 1.0))
        result = line_line_intersection(seg1, seg2)
        self.assertEqual(len(result), 0)

    def test_line_line_no_intersection(self) -> None:
        seg1 = LineSegment((0.0, 0.0), (1.0, 1.0))
        seg2 = LineSegment((2.0, 0.0), (3.0, 1.0))
        result = line_line_intersection(seg1, seg2)
        self.assertEqual(len(result), 0)

    def test_line_line_t_junction(self) -> None:
        seg1 = LineSegment((0.0, 0.0), (2.0, 0.0))
        seg2 = LineSegment((1.0, -1.0), (1.0, 1.0))
        result = line_line_intersection(seg1, seg2)
        self.assertEqual(len(result), 1)
        self._assert_point_near(result[0], (1.0, 0.0))

    def test_line_circle_two_intersections(self) -> None:
        seg = LineSegment((-2.0, 0.0), (2.0, 0.0))
        arc = CircularArc((0.0, 0.0), 1.0, 0.0, 2 * math.pi)
        result = line_circle_intersection(seg, arc)
        self.assertEqual(len(result), 2)
        xs = sorted([p[0] for p in result])
        self.assertAlmostEqual(xs[0], -1.0)
        self.assertAlmostEqual(xs[1], 1.0)

    def test_line_circle_tangent(self) -> None:
        seg = LineSegment((-2.0, 1.0), (2.0, 1.0))
        arc = CircularArc((0.0, 0.0), 1.0, 0.0, 2 * math.pi)
        result = line_circle_intersection(seg, arc)
        self.assertEqual(len(result), 1)
        self._assert_point_near(result[0], (0.0, 1.0))

    def test_line_circle_no_intersection(self) -> None:
        seg = LineSegment((-2.0, 2.0), (2.0, 2.0))
        arc = CircularArc((0.0, 0.0), 1.0, 0.0, 2 * math.pi)
        result = line_circle_intersection(seg, arc)
        self.assertEqual(len(result), 0)

    def test_line_circle_partial_arc(self) -> None:
        seg = LineSegment((-2.0, 0.0), (2.0, 0.0))
        arc = CircularArc((0.0, 0.0), 1.0, 0.0, math.pi / 2)
        result = line_circle_intersection(seg, arc)
        self.assertEqual(len(result), 1)
        self._assert_point_near(result[0], (1.0, 0.0))

    def test_line_ellipse_two_intersections(self) -> None:
        seg = LineSegment((-3.0, 0.0), (3.0, 0.0))
        arc = EllipticalArc((0.0, 0.0), 2.0, 1.0, 0.0, 2 * math.pi)
        result = line_ellipse_intersection(seg, arc)
        self.assertEqual(len(result), 2)
        xs = sorted([p[0] for p in result])
        self.assertAlmostEqual(xs[0], -2.0)
        self.assertAlmostEqual(xs[1], 2.0)

    def test_line_ellipse_no_intersection(self) -> None:
        seg = LineSegment((-3.0, 2.0), (3.0, 2.0))
        arc = EllipticalArc((0.0, 0.0), 2.0, 1.0, 0.0, 2 * math.pi)
        result = line_ellipse_intersection(seg, arc)
        self.assertEqual(len(result), 0)

    def test_line_ellipse_rotated(self) -> None:
        seg = LineSegment((0.0, -3.0), (0.0, 3.0))
        arc = EllipticalArc((0.0, 0.0), 2.0, 1.0, 0.0, 2 * math.pi, rotation=math.pi / 2)
        result = line_ellipse_intersection(seg, arc)
        self.assertEqual(len(result), 2)
        ys = sorted([p[1] for p in result])
        self.assertAlmostEqual(ys[0], -2.0)
        self.assertAlmostEqual(ys[1], 2.0)

    def test_circle_circle_two_intersections(self) -> None:
        arc1 = CircularArc((0.0, 0.0), 1.0, 0.0, 2 * math.pi)
        arc2 = CircularArc((1.0, 0.0), 1.0, 0.0, 2 * math.pi)
        result = circle_circle_intersection(arc1, arc2)
        self.assertEqual(len(result), 2)
        for point in result:
            self.assertAlmostEqual(point[0], 0.5)

    def test_circle_circle_tangent(self) -> None:
        arc1 = CircularArc((0.0, 0.0), 1.0, 0.0, 2 * math.pi)
        arc2 = CircularArc((2.0, 0.0), 1.0, 0.0, 2 * math.pi)
        result = circle_circle_intersection(arc1, arc2)
        self.assertEqual(len(result), 1)
        self._assert_point_near(result[0], (1.0, 0.0))

    def test_circle_circle_no_intersection(self) -> None:
        arc1 = CircularArc((0.0, 0.0), 1.0, 0.0, 2 * math.pi)
        arc2 = CircularArc((3.0, 0.0), 1.0, 0.0, 2 * math.pi)
        result = circle_circle_intersection(arc1, arc2)
        self.assertEqual(len(result), 0)

    def test_circle_circle_contained(self) -> None:
        arc1 = CircularArc((0.0, 0.0), 2.0, 0.0, 2 * math.pi)
        arc2 = CircularArc((0.0, 0.0), 1.0, 0.0, 2 * math.pi)
        result = circle_circle_intersection(arc1, arc2)
        self.assertEqual(len(result), 0)

    def test_circle_circle_partial_arcs(self) -> None:
        arc1 = CircularArc((0.0, 0.0), 1.0, 0.0, math.pi / 2)
        arc2 = CircularArc((1.0, 0.0), 1.0, math.pi / 2, math.pi)
        result = circle_circle_intersection(arc1, arc2)
        self.assertEqual(len(result), 1)

    def test_circle_ellipse_intersection(self) -> None:
        circle = CircularArc((0.0, 0.0), 1.5, 0.0, 2 * math.pi)
        ellipse = EllipticalArc((0.0, 0.0), 2.0, 1.0, 0.0, 2 * math.pi)
        result = circle_ellipse_intersection(circle, ellipse)
        self.assertGreaterEqual(len(result), 2)
        for point in result:
            dist_to_circle = math.sqrt(point[0]**2 + point[1]**2)
            self.assertAlmostEqual(dist_to_circle, 1.5, places=1)

    def test_ellipse_ellipse_intersection(self) -> None:
        ellipse1 = EllipticalArc((0.0, 0.0), 2.0, 1.0, 0.0, 2 * math.pi)
        ellipse2 = EllipticalArc((1.0, 0.0), 2.0, 1.0, 0.0, 2 * math.pi)
        result = ellipse_ellipse_intersection(ellipse1, ellipse2)
        self.assertGreaterEqual(len(result), 2)

    def test_element_element_dispatch(self) -> None:
        seg1 = LineSegment((0.0, 0.0), (2.0, 2.0))
        seg2 = LineSegment((0.0, 2.0), (2.0, 0.0))
        result = element_element_intersection(seg1, seg2)
        self.assertEqual(len(result), 1)

        seg = LineSegment((-2.0, 0.0), (2.0, 0.0))
        arc = CircularArc((0.0, 0.0), 1.0, 0.0, 2 * math.pi)
        result = element_element_intersection(seg, arc)
        self.assertEqual(len(result), 2)

        result = element_element_intersection(arc, seg)
        self.assertEqual(len(result), 2)

    def test_path_path_intersections(self) -> None:
        square = CompositePath.from_points([
            (0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)
        ])
        diagonal = CompositePath([
            LineSegment((-1.0, 1.0), (3.0, 1.0))
        ])
        result = path_path_intersections(square, diagonal)
        self.assertEqual(len(result), 2)

    def test_path_path_no_intersections(self) -> None:
        square1 = CompositePath.from_points([
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)
        ])
        square2 = CompositePath.from_points([
            (5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0), (5.0, 5.0)
        ])
        result = path_path_intersections(square1, square2)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()

