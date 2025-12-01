from __future__ import annotations

import math
import unittest

from geometry import (
    LineSegment,
    CircularArc,
    EllipticalArc,
    CompositePath,
)
from .simple_mock import SimpleMock


class TestPathElements(unittest.TestCase):

    def test_line_segment_creation(self) -> None:
        seg = LineSegment((0.0, 0.0), (3.0, 4.0))
        self.assertEqual(seg.start_point(), (0.0, 0.0))
        self.assertEqual(seg.end_point(), (3.0, 4.0))

    def test_line_segment_length(self) -> None:
        seg = LineSegment((0.0, 0.0), (3.0, 4.0))
        self.assertAlmostEqual(seg.length(), 5.0)

    def test_line_segment_sample_returns_endpoints(self) -> None:
        seg = LineSegment((1.0, 2.0), (5.0, 6.0))
        points = seg.sample(100)
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0], (1.0, 2.0))
        self.assertEqual(points[1], (5.0, 6.0))

    def test_line_segment_reversed(self) -> None:
        seg = LineSegment((0.0, 0.0), (1.0, 1.0))
        rev = seg.reversed()
        self.assertEqual(rev.start_point(), (1.0, 1.0))
        self.assertEqual(rev.end_point(), (0.0, 0.0))

    def test_line_segment_from_segment_drawable(self) -> None:
        point1 = SimpleMock(x=1.0, y=2.0)
        point2 = SimpleMock(x=3.0, y=4.0)
        segment = SimpleMock(point1=point1, point2=point2)
        line = LineSegment.from_segment(segment)
        self.assertEqual(line.start_point(), (1.0, 2.0))
        self.assertEqual(line.end_point(), (3.0, 4.0))

    def test_line_segment_equality(self) -> None:
        seg1 = LineSegment((0.0, 0.0), (1.0, 1.0))
        seg2 = LineSegment((0.0, 0.0), (1.0, 1.0))
        seg3 = LineSegment((0.0, 0.0), (2.0, 2.0))
        self.assertEqual(seg1, seg2)
        self.assertNotEqual(seg1, seg3)

    def test_line_segment_connects_to(self) -> None:
        seg1 = LineSegment((0.0, 0.0), (1.0, 1.0))
        seg2 = LineSegment((1.0, 1.0), (2.0, 0.0))
        seg3 = LineSegment((5.0, 5.0), (6.0, 6.0))
        self.assertTrue(seg1.connects_to(seg2))
        self.assertFalse(seg1.connects_to(seg3))

    def test_circular_arc_creation(self) -> None:
        arc = CircularArc((0.0, 0.0), 1.0, 0.0, math.pi / 2)
        self.assertEqual(arc.center, (0.0, 0.0))
        self.assertEqual(arc.radius, 1.0)
        self.assertEqual(arc.start_angle, 0.0)
        self.assertEqual(arc.end_angle, math.pi / 2)
        self.assertFalse(arc.clockwise)

    def test_circular_arc_invalid_radius_raises(self) -> None:
        with self.assertRaises(ValueError):
            CircularArc((0.0, 0.0), 0.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            CircularArc((0.0, 0.0), -1.0, 0.0, 1.0)

    def test_circular_arc_start_end_points(self) -> None:
        arc = CircularArc((0.0, 0.0), 1.0, 0.0, math.pi / 2)
        start = arc.start_point()
        end = arc.end_point()
        self.assertAlmostEqual(start[0], 1.0)
        self.assertAlmostEqual(start[1], 0.0)
        self.assertAlmostEqual(end[0], 0.0)
        self.assertAlmostEqual(end[1], 1.0)

    def test_circular_arc_sample_generates_points(self) -> None:
        arc = CircularArc((0.0, 0.0), 1.0, 0.0, math.pi / 2)
        points = arc.sample(10)
        self.assertGreaterEqual(len(points), 2)
        self.assertAlmostEqual(points[0][0], 1.0, places=5)
        self.assertAlmostEqual(points[0][1], 0.0, places=5)
        self.assertAlmostEqual(points[-1][0], 0.0, places=5)
        self.assertAlmostEqual(points[-1][1], 1.0, places=5)

    def test_circular_arc_length_quarter_circle(self) -> None:
        arc = CircularArc((0.0, 0.0), 1.0, 0.0, math.pi / 2)
        expected = (2 * math.pi * 1.0) / 4
        self.assertAlmostEqual(arc.length(), expected, places=5)

    def test_circular_arc_reversed(self) -> None:
        arc = CircularArc((0.0, 0.0), 1.0, 0.0, math.pi / 2)
        rev = arc.reversed()
        self.assertEqual(rev.start_angle, math.pi / 2)
        self.assertEqual(rev.end_angle, 0.0)
        self.assertTrue(rev.clockwise)
        self.assertEqual(rev.start_point(), arc.end_point())
        self.assertEqual(rev.end_point(), arc.start_point())

    def test_circular_arc_clockwise(self) -> None:
        arc = CircularArc((0.0, 0.0), 1.0, 0.0, math.pi / 2, clockwise=True)
        self.assertTrue(arc.clockwise)
        length = arc.length()
        expected = (2 * math.pi * 1.0) * (3 / 4)
        self.assertAlmostEqual(length, expected, places=5)

    def test_circular_arc_from_circle_arc_minor(self) -> None:
        point1 = SimpleMock(x=1.0, y=0.0)
        point2 = SimpleMock(x=0.0, y=1.0)
        circle_arc = SimpleMock(
            center_x=0.0,
            center_y=0.0,
            radius=1.0,
            point1=point1,
            point2=point2,
            use_major_arc=False,
        )
        arc = CircularArc.from_circle_arc(circle_arc)
        self.assertEqual(arc.center, (0.0, 0.0))
        self.assertEqual(arc.radius, 1.0)
        start = arc.start_point()
        end = arc.end_point()
        self.assertAlmostEqual(start[0], 1.0, places=5)
        self.assertAlmostEqual(start[1], 0.0, places=5)
        self.assertAlmostEqual(end[0], 0.0, places=5)
        self.assertAlmostEqual(end[1], 1.0, places=5)

    def test_circular_arc_from_circle_arc_major(self) -> None:
        point1 = SimpleMock(x=1.0, y=0.0)
        point2 = SimpleMock(x=0.0, y=1.0)
        circle_arc = SimpleMock(
            center_x=0.0,
            center_y=0.0,
            radius=1.0,
            point1=point1,
            point2=point2,
            use_major_arc=True,
        )
        arc = CircularArc.from_circle_arc(circle_arc)
        major_length = (3 * math.pi / 2) * 1.0
        self.assertAlmostEqual(arc.length(), major_length, places=5)

    def test_circular_arc_from_circle(self) -> None:
        center = SimpleMock(x=1.0, y=2.0)
        circle = SimpleMock(center=center, radius=3.0)
        arc = CircularArc.from_circle(circle)
        self.assertEqual(arc.center, (1.0, 2.0))
        self.assertEqual(arc.radius, 3.0)
        self.assertAlmostEqual(arc.length(), 2 * math.pi * 3.0, places=5)

    def test_elliptical_arc_creation(self) -> None:
        arc = EllipticalArc((0.0, 0.0), 2.0, 1.0, 0.0, math.pi / 2)
        self.assertEqual(arc.center, (0.0, 0.0))
        self.assertEqual(arc.radius_x, 2.0)
        self.assertEqual(arc.radius_y, 1.0)
        self.assertEqual(arc.rotation, 0.0)
        self.assertFalse(arc.clockwise)

    def test_elliptical_arc_invalid_radii_raises(self) -> None:
        with self.assertRaises(ValueError):
            EllipticalArc((0.0, 0.0), 0.0, 1.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            EllipticalArc((0.0, 0.0), 1.0, -1.0, 0.0, 1.0)

    def test_elliptical_arc_start_end_points_no_rotation(self) -> None:
        arc = EllipticalArc((0.0, 0.0), 2.0, 1.0, 0.0, math.pi / 2)
        start = arc.start_point()
        end = arc.end_point()
        self.assertAlmostEqual(start[0], 2.0)
        self.assertAlmostEqual(start[1], 0.0)
        self.assertAlmostEqual(end[0], 0.0)
        self.assertAlmostEqual(end[1], 1.0)

    def test_elliptical_arc_sample_generates_points(self) -> None:
        arc = EllipticalArc((0.0, 0.0), 2.0, 1.0, 0.0, math.pi)
        points = arc.sample(20)
        self.assertGreaterEqual(len(points), 2)
        self.assertAlmostEqual(points[0][0], 2.0, places=5)
        self.assertAlmostEqual(points[0][1], 0.0, places=5)
        self.assertAlmostEqual(points[-1][0], -2.0, places=5)
        self.assertAlmostEqual(points[-1][1], 0.0, places=5)

    def test_elliptical_arc_reversed(self) -> None:
        arc = EllipticalArc((0.0, 0.0), 2.0, 1.0, 0.0, math.pi / 2)
        rev = arc.reversed()
        self.assertEqual(rev.start_angle, math.pi / 2)
        self.assertEqual(rev.end_angle, 0.0)
        self.assertTrue(rev.clockwise)

    def test_elliptical_arc_rotation(self) -> None:
        arc = EllipticalArc((0.0, 0.0), 2.0, 1.0, 0.0, 0.0, rotation=math.pi / 2)
        start = arc.start_point()
        self.assertAlmostEqual(start[0], 0.0, places=5)
        self.assertAlmostEqual(start[1], 2.0, places=5)

    def test_elliptical_arc_from_ellipse(self) -> None:
        center = SimpleMock(x=0.0, y=0.0)
        ellipse = SimpleMock(
            center=center,
            radius_x=2.0,
            radius_y=1.0,
            rotation_angle=0.0,
        )
        arc = EllipticalArc.from_ellipse(ellipse)
        self.assertEqual(arc.center, (0.0, 0.0))
        self.assertEqual(arc.radius_x, 2.0)
        self.assertEqual(arc.radius_y, 1.0)
        self.assertEqual(arc.rotation, 0.0)

    def test_elliptical_arc_from_ellipse_rotated(self) -> None:
        center = SimpleMock(x=0.0, y=0.0)
        ellipse = SimpleMock(
            center=center,
            radius_x=2.0,
            radius_y=1.0,
            rotation_angle=45.0,
        )
        arc = EllipticalArc.from_ellipse(ellipse)
        self.assertAlmostEqual(arc.rotation, math.radians(45.0), places=5)

    def test_composite_path_empty(self) -> None:
        path = CompositePath()
        self.assertTrue(path.is_empty())
        self.assertFalse(path.is_closed())
        self.assertIsNone(path.start_point())
        self.assertIsNone(path.end_point())
        self.assertEqual(path.sample(), [])

    def test_composite_path_single_segment(self) -> None:
        seg = LineSegment((0.0, 0.0), (1.0, 1.0))
        path = CompositePath([seg])
        self.assertEqual(len(path), 1)
        self.assertFalse(path.is_closed())
        self.assertEqual(path.start_point(), (0.0, 0.0))
        self.assertEqual(path.end_point(), (1.0, 1.0))

    def test_composite_path_connected_segments(self) -> None:
        seg1 = LineSegment((0.0, 0.0), (1.0, 0.0))
        seg2 = LineSegment((1.0, 0.0), (1.0, 1.0))
        seg3 = LineSegment((1.0, 1.0), (0.0, 1.0))
        path = CompositePath([seg1, seg2, seg3])
        self.assertEqual(len(path), 3)
        self.assertFalse(path.is_closed())
        self.assertEqual(path.start_point(), (0.0, 0.0))
        self.assertEqual(path.end_point(), (0.0, 1.0))

    def test_composite_path_closed(self) -> None:
        seg1 = LineSegment((0.0, 0.0), (1.0, 0.0))
        seg2 = LineSegment((1.0, 0.0), (1.0, 1.0))
        seg3 = LineSegment((1.0, 1.0), (0.0, 1.0))
        seg4 = LineSegment((0.0, 1.0), (0.0, 0.0))
        path = CompositePath([seg1, seg2, seg3, seg4])
        self.assertTrue(path.is_closed())

    def test_composite_path_disconnected_raises(self) -> None:
        seg1 = LineSegment((0.0, 0.0), (1.0, 0.0))
        seg2 = LineSegment((5.0, 5.0), (6.0, 6.0))
        with self.assertRaises(ValueError):
            CompositePath([seg1, seg2])

    def test_composite_path_append(self) -> None:
        path = CompositePath()
        path.append(LineSegment((0.0, 0.0), (1.0, 0.0)))
        path.append(LineSegment((1.0, 0.0), (2.0, 0.0)))
        self.assertEqual(len(path), 2)

    def test_composite_path_append_disconnected_raises(self) -> None:
        path = CompositePath([LineSegment((0.0, 0.0), (1.0, 0.0))])
        with self.assertRaises(ValueError):
            path.append(LineSegment((5.0, 5.0), (6.0, 6.0)))

    def test_composite_path_prepend(self) -> None:
        path = CompositePath([LineSegment((1.0, 0.0), (2.0, 0.0))])
        path.prepend(LineSegment((0.0, 0.0), (1.0, 0.0)))
        self.assertEqual(len(path), 2)
        self.assertEqual(path.start_point(), (0.0, 0.0))

    def test_composite_path_sample_concatenates(self) -> None:
        seg1 = LineSegment((0.0, 0.0), (1.0, 0.0))
        seg2 = LineSegment((1.0, 0.0), (2.0, 0.0))
        path = CompositePath([seg1, seg2])
        points = path.sample()
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0], (0.0, 0.0))
        self.assertEqual(points[1], (1.0, 0.0))
        self.assertEqual(points[2], (2.0, 0.0))

    def test_composite_path_sample_avoids_duplicates(self) -> None:
        seg1 = LineSegment((0.0, 0.0), (1.0, 0.0))
        seg2 = LineSegment((1.0, 0.0), (2.0, 0.0))
        seg3 = LineSegment((2.0, 0.0), (3.0, 0.0))
        path = CompositePath([seg1, seg2, seg3])
        points = path.sample()
        self.assertEqual(len(points), 4)

    def test_composite_path_reversed(self) -> None:
        seg1 = LineSegment((0.0, 0.0), (1.0, 0.0))
        seg2 = LineSegment((1.0, 0.0), (1.0, 1.0))
        path = CompositePath([seg1, seg2])
        rev = path.reversed()
        self.assertEqual(rev.start_point(), (1.0, 1.0))
        self.assertEqual(rev.end_point(), (0.0, 0.0))

    def test_composite_path_length(self) -> None:
        seg1 = LineSegment((0.0, 0.0), (3.0, 0.0))
        seg2 = LineSegment((3.0, 0.0), (3.0, 4.0))
        path = CompositePath([seg1, seg2])
        self.assertAlmostEqual(path.length(), 7.0)

    def test_composite_path_from_points(self) -> None:
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        path = CompositePath.from_points(points)
        self.assertEqual(len(path), 3)
        self.assertEqual(path.start_point(), (0.0, 0.0))
        self.assertEqual(path.end_point(), (0.0, 1.0))

    def test_composite_path_from_points_closed(self) -> None:
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        path = CompositePath.from_points(points)
        self.assertEqual(len(path), 4)
        self.assertTrue(path.is_closed())

    def test_composite_path_iteration(self) -> None:
        seg1 = LineSegment((0.0, 0.0), (1.0, 0.0))
        seg2 = LineSegment((1.0, 0.0), (2.0, 0.0))
        path = CompositePath([seg1, seg2])
        elements = list(path)
        self.assertEqual(len(elements), 2)
        self.assertEqual(elements[0], seg1)
        self.assertEqual(elements[1], seg2)

    def test_composite_path_from_polygon(self) -> None:
        p1 = SimpleMock(x=0.0, y=0.0)
        p2 = SimpleMock(x=1.0, y=0.0)
        p3 = SimpleMock(x=1.0, y=1.0)
        seg1 = SimpleMock(point1=p1, point2=p2)
        seg2 = SimpleMock(point1=p2, point2=p3)
        seg3 = SimpleMock(point1=p3, point2=p1)
        polygon = SimpleMock(get_segments=lambda: [seg1, seg2, seg3])
        path = CompositePath.from_polygon(polygon)
        self.assertEqual(len(path), 3)
        self.assertTrue(path.is_closed())

    def test_composite_path_from_polygon_not_closed_raises(self) -> None:
        p1 = SimpleMock(x=0.0, y=0.0)
        p2 = SimpleMock(x=1.0, y=0.0)
        p3 = SimpleMock(x=1.0, y=1.0)
        seg1 = SimpleMock(point1=p1, point2=p2)
        seg2 = SimpleMock(point1=p2, point2=p3)
        bad_polygon = SimpleMock(get_segments=lambda: [seg1, seg2])
        with self.assertRaises(ValueError):
            CompositePath.from_polygon(bad_polygon)

    def test_mixed_path_segment_then_arc(self) -> None:
        seg = LineSegment((0.0, 0.0), (1.0, 0.0))
        arc = CircularArc((1.0, 1.0), 1.0, -math.pi / 2, 0.0)
        path = CompositePath([seg, arc])
        self.assertEqual(len(path), 2)
        self.assertEqual(path.start_point(), (0.0, 0.0))
        end = path.end_point()
        self.assertAlmostEqual(end[0], 2.0, places=5)
        self.assertAlmostEqual(end[1], 1.0, places=5)

    def test_mixed_path_arc_then_segment(self) -> None:
        arc = CircularArc((0.0, 0.0), 1.0, 0.0, math.pi / 2)
        seg = LineSegment((0.0, 1.0), (0.0, 2.0))
        path = CompositePath([arc, seg])
        self.assertEqual(len(path), 2)
        start = path.start_point()
        self.assertAlmostEqual(start[0], 1.0, places=5)
        self.assertAlmostEqual(start[1], 0.0, places=5)
        self.assertEqual(path.end_point(), (0.0, 2.0))

    def test_mixed_path_closed(self) -> None:
        seg1 = LineSegment((1.0, 0.0), (-1.0, 0.0))
        arc = CircularArc((0.0, 0.0), 1.0, math.pi, 0.0)
        path = CompositePath([seg1, arc])
        self.assertTrue(path.is_closed())

    def test_mixed_path_sample(self) -> None:
        seg = LineSegment((0.0, 0.0), (1.0, 0.0))
        arc = CircularArc((1.0, 1.0), 1.0, -math.pi / 2, 0.0)
        path = CompositePath([seg, arc])
        points = path.sample(10)
        self.assertGreater(len(points), 3)
        self.assertEqual(points[0], (0.0, 0.0))
        self.assertEqual(points[1], (1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
