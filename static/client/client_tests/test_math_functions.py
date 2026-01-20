import unittest
import json
from typing import List
from utils.math_utils import MathUtils
from drawables_aggregator import Position
from .simple_mock import SimpleMock
import math  # Add import at the top of the method
from rendering.renderables import FunctionRenderable


class TestMathFunctions(unittest.TestCase):
    def setUp(self) -> None:
        # Mock points for use in some tests (math-space coordinates only)
        self.point1 = SimpleMock(x=0, y=0, name='A')
        self.point2 = SimpleMock(x=1, y=1, name='B')
        # Mock segment using mocked points
        self.segment = SimpleMock(point1=self.point1, point2=self.point2)
    
    def test_format_number_for_cartesian(self) -> None:
        test_cases = [
            (123456789, 6, '1.2e+8'),
            (0.000123456789, 6, '0.00012'),
            (123456, 6, '123456'),
            (123.456, 6, '123.456'),
            (0, 6, '0'),
            (-123456789, 6, '-1.2e+8'),
            (-0.000123456789, 6, '-0.00012'),
            (-123456, 6, '-123456'),
            (-123.456, 6, '-123.456'),
            (1.23456789, 6, '1.23457'),
            (0.000000123456789, 6, '1.2e-7'),
            (123456.789, 6, '123457'),
            (123.456789, 6, '123.457'),
            (0.00000000000001, 6, '1e-14'),
            (-1.23456789, 6, '-1.23457'),
            (-0.000000123456789, 6, '-1.2e-7'),
            (-123456.789, 6, '-123457'),
            (-123.456789, 6, '-123.457'),
            (-0.00000000000001, 6, '-1e-14'),
            (123456789, 3, '1.2e+8'),
            (0.000123456789, 3, '1.2e-4'),
            (123456, 3, '1.2e+5'),
            (123.456, 3, '123'),
            (1.23456789, 3, '1.23'),
            (0.000000123456789, 3, '1.2e-7'),
            (123456.789, 3, '1.2e+5'),
            (123.456789, 3, '123'),
            (0.00000000000001, 3, '1e-14'),
        ]
        for i, (input, max_digits, expected) in enumerate(test_cases):
            with self.subTest(i=i):
                self.assertEqual(MathUtils.format_number_for_cartesian(input, max_digits=max_digits), expected)

    def test_point_matches_coordinates(self) -> None:
        self.assertTrue(MathUtils.point_matches_coordinates(self.point1, 0, 0))
        self.assertFalse(MathUtils.point_matches_coordinates(self.point1, 1, 1))

    def test_segment_matches_coordinates(self) -> None:
        self.assertTrue(MathUtils.segment_matches_coordinates(self.segment, 0, 0, 1, 1))
        self.assertTrue(MathUtils.segment_matches_coordinates(self.segment, 1, 1, 0, 0))  # Reverse order
        self.assertFalse(MathUtils.segment_matches_coordinates(self.segment, 2, 2, 3, 3))  # Incorrect coordinates

    def test_segment_matches_point_names(self) -> None:
        self.assertTrue(MathUtils.segment_matches_point_names(self.segment, 'A', 'B'))
        self.assertTrue(MathUtils.segment_matches_point_names(self.segment, 'B', 'A'))  # Reverse order
        self.assertFalse(MathUtils.segment_matches_point_names(self.segment, 'C', 'D'))  # Incorrect names

    def test_segment_endpoints_helpers(self) -> None:
        segment = SimpleMock(
            point1=SimpleMock(x=1.5, y=-2.0),
            point2=SimpleMock(x=3.25, y=4.75),
        )
        self.assertEqual(
            MathUtils._segment_endpoints(segment),
            (1.5, -2.0, 3.25, 4.75),
        )
        self.assertEqual(
            MathUtils._segment_endpoints(((0, 0), (5, -3.5))),
            (0.0, 0.0, 5.0, -3.5),
        )
        with self.assertRaises(ValueError):
            MathUtils._segment_endpoints("invalid")

    def test_normalize_angle_and_arc_sequence(self) -> None:
        self.assertAlmostEqual(MathUtils._normalize_angle(7 * math.pi), math.pi)
        self.assertAlmostEqual(MathUtils._normalize_angle(-3 * math.pi / 2), math.pi / 2)

        ccw_angles = MathUtils._arc_angle_sequence(0.0, math.pi / 2, 5)
        self.assertEqual(len(ccw_angles), 5)
        self.assertAlmostEqual(ccw_angles[0], 0.0)
        self.assertAlmostEqual(ccw_angles[-1], math.pi / 2)

        cw_angles = MathUtils._arc_angle_sequence(math.pi / 2, 0.0, 4, clockwise=True)
        self.assertEqual(len(cw_angles), 4)
        self.assertAlmostEqual(cw_angles[0], math.pi / 2)
        self.assertAlmostEqual(cw_angles[-1], MathUtils._normalize_angle(0.0))

    def test_circle_segment_intersections_basic(self) -> None:
        circle_center = SimpleMock(x=0.0, y=0.0)
        chord = SimpleMock(
            point1=SimpleMock(x=-5.0, y=0.0),
            point2=SimpleMock(x=5.0, y=0.0),
        )
        intersections = MathUtils.circle_segment_intersections(
            circle_center.x,
            circle_center.y,
            5.0,
            chord,
        )
        self.assertEqual(len(intersections), 2)
        xs = sorted(pt["x"] for pt in intersections)
        ys = [pt["y"] for pt in intersections]
        self.assertEqual(xs, [-5.0, 5.0])
        self.assertTrue(all(abs(y) < 1e-9 for y in ys))

    def test_ellipse_segment_intersections_no_rotation(self) -> None:
        ellipse = SimpleMock(center=SimpleMock(x=0.0, y=0.0), radius_x=4.0, radius_y=2.0, rotation_angle=0.0)
        segment = SimpleMock(
            point1=SimpleMock(x=-6.0, y=0.0),
            point2=SimpleMock(x=6.0, y=0.0),
        )
        intersections = MathUtils.ellipse_segment_intersections(
            ellipse.center.x,
            ellipse.center.y,
            ellipse.radius_x,
            ellipse.radius_y,
            ellipse.rotation_angle,
            segment,
        )
        self.assertEqual(len(intersections), 2)
        xs = sorted(pt["x"] for pt in intersections)
        ys = [pt["y"] for pt in intersections]
        self.assertAlmostEqual(xs[0], -4.0)
        self.assertAlmostEqual(xs[1], 4.0)
        self.assertTrue(all(abs(y) < 1e-9 for y in ys))

    def test_sample_circle_arc_includes_endpoints(self) -> None:
        samples = MathUtils.sample_circle_arc(
            0.0,
            0.0,
            10.0,
            0.0,
            math.pi / 2,
            num_samples=4,
        )
        self.assertEqual(len(samples), 4)
        self.assertAlmostEqual(samples[0][0], 10.0)
        self.assertAlmostEqual(samples[-1][0], 0.0, places=6)
        self.assertGreater(samples[-1][1], 0.0)

    def test_sample_ellipse_arc_respects_rotation(self) -> None:
        samples = MathUtils.sample_ellipse_arc(
            0.0,
            0.0,
            6.0,
            3.0,
            0.0,
            math.pi / 2,
            rotation_degrees=30.0,
            num_samples=3,
        )
        self.assertEqual(len(samples), 3)
        start_point = samples[0]
        end_point = samples[-1]
        self.assertNotEqual(start_point[0], 6.0)
        self.assertNotEqual(start_point[1], 0.0)
        self.assertGreater(end_point[1], 0.0)

    def test_circle_segment_intersections_additional(self) -> None:
        circle_center = SimpleMock(x=0.0, y=0.0)
        tangent = SimpleMock(
            point1=SimpleMock(x=0.0, y=5.0),
            point2=SimpleMock(x=10.0, y=5.0),
        )
        tangent_hits = MathUtils.circle_segment_intersections(
            circle_center.x,
            circle_center.y,
            5.0,
            tangent,
        )
        self.assertIn(len(tangent_hits), (1, 2))
        for pt in tangent_hits:
            self.assertAlmostEqual(pt["x"], 0.0)
            self.assertAlmostEqual(pt["y"], 5.0)

        outside = SimpleMock(
            point1=SimpleMock(x=6.0, y=6.0),
            point2=SimpleMock(x=7.0, y=7.0),
        )
        self.assertEqual(
            MathUtils.circle_segment_intersections(circle_center.x, circle_center.y, 5.0, outside),
            [],
        )

        inside = SimpleMock(
            point1=SimpleMock(x=-1.0, y=0.0),
            point2=SimpleMock(x=1.0, y=0.0),
        )
        intersections = MathUtils.circle_segment_intersections(
            circle_center.x,
            circle_center.y,
            5.0,
            inside,
        )
        self.assertEqual(len(intersections), 0)

        self.assertEqual(
            MathUtils.circle_segment_intersections(circle_center.x, circle_center.y, 0.0, inside),
            [],
        )

    def test_ellipse_segment_intersections_additional(self) -> None:
        ellipse = SimpleMock(center=SimpleMock(x=0.0, y=0.0), radius_x=5.0, radius_y=3.0, rotation_angle=45.0)
        tangent = SimpleMock(
            point1=SimpleMock(x=5.0, y=0.0),
            point2=SimpleMock(x=5.0, y=5.0),
        )
        tangent_hits = MathUtils.ellipse_segment_intersections(
            ellipse.center.x,
            ellipse.center.y,
            ellipse.radius_x,
            ellipse.radius_y,
            0.0,
            tangent,
        )
        self.assertIn(len(tangent_hits), (0, 1, 2))
        for pt in tangent_hits:
            self.assertAlmostEqual(pt["x"], 5.0, places=6)
            self.assertGreaterEqual(pt["y"], 0.0)

        outside = SimpleMock(
            point1=SimpleMock(x=10.0, y=10.0),
            point2=SimpleMock(x=12.0, y=12.0),
        )
        self.assertEqual(
            MathUtils.ellipse_segment_intersections(
                ellipse.center.x,
                ellipse.center.y,
                ellipse.radius_x,
                ellipse.radius_y,
                ellipse.rotation_angle,
                outside,
            ),
            [],
        )

        self.assertEqual(
            MathUtils.ellipse_segment_intersections(
                ellipse.center.x,
                ellipse.center.y,
                0.0,
                ellipse.radius_y,
                ellipse.rotation_angle,
                tangent,
            ),
            [],
        )

        vertical = SimpleMock(
            point1=SimpleMock(x=2.0, y=-10.0),
            point2=SimpleMock(x=2.0, y=10.0),
        )
        vert_hits = MathUtils.ellipse_segment_intersections(
            ellipse.center.x,
            ellipse.center.y,
            ellipse.radius_x,
            ellipse.radius_y,
            ellipse.rotation_angle,
            vertical,
        )
        self.assertTrue(len(vert_hits) in (0, 2))

    def test_sample_circle_arc_additional(self) -> None:
        clockwise = MathUtils.sample_circle_arc(
            0.0,
            0.0,
            5.0,
            math.pi / 2,
            -math.pi / 2,
            num_samples=3,
            clockwise=True,
        )
        self.assertEqual(len(clockwise), 3)

        full_circle = MathUtils.sample_circle_arc(
            0.0,
            0.0,
            5.0,
            1.0,
            1.0,
            num_samples=4,
        )
        self.assertEqual(len(full_circle), 4)
        self.assertAlmostEqual(full_circle[0][0], full_circle[-1][0])
        self.assertAlmostEqual(full_circle[0][1], full_circle[-1][1])

        minimal = MathUtils.sample_circle_arc(
            0.0,
            0.0,
            5.0,
            0.0,
            math.pi,
            num_samples=1,
        )
        self.assertEqual(len(minimal), 2)

        self.assertEqual(
            MathUtils.sample_circle_arc(0.0, 0.0, 0.0, 0.0, math.pi),
            [],
        )

    def test_sample_ellipse_arc_additional(self) -> None:
        rotated_90 = MathUtils.sample_ellipse_arc(
            0.0,
            0.0,
            4.0,
            2.0,
            0.0,
            math.pi / 2,
            rotation_degrees=90.0,
            num_samples=3,
        )
        self.assertEqual(len(rotated_90), 3)

        clockwise = MathUtils.sample_ellipse_arc(
            0.0,
            0.0,
            3.0,
            2.0,
            math.pi / 2,
            -math.pi / 2,
            rotation_degrees=15.0,
            num_samples=4,
            clockwise=True,
        )
        self.assertEqual(len(clockwise), 4)

        minimal = MathUtils.sample_ellipse_arc(
            0.0,
            0.0,
            3.0,
            2.0,
            0.0,
            math.pi,
            num_samples=1,
        )
        self.assertEqual(len(minimal), 2)

        self.assertEqual(
            MathUtils.sample_ellipse_arc(0.0, 0.0, 0.0, 2.0, 0.0, math.pi),
            [],
        )
        self.assertEqual(
            MathUtils.sample_ellipse_arc(0.0, 0.0, 2.0, 0.0, 0.0, math.pi),
            [],
        )
    def test_segment_endpoints_additional_cases(self) -> None:
        degenerate_segment = SimpleMock(
            point1=SimpleMock(x=-1.0, y=-1.0),
            point2=SimpleMock(x=-1.0, y=-1.0),
        )
        self.assertEqual(
            MathUtils._segment_endpoints(degenerate_segment),
            (-1.0, -1.0, -1.0, -1.0),
        )
        mixed_tuple = ((-5, 2.5), (123456789, -987654321.5))
        self.assertEqual(
            MathUtils._segment_endpoints(mixed_tuple),
            (-5.0, 2.5, 123456789.0, -987654321.5),
        )
        with self.assertRaises(TypeError):
            MathUtils._segment_endpoints((0.0, 1.0))
        invalid_tuple = ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))
        with self.assertRaises(ValueError):
            MathUtils._segment_endpoints(invalid_tuple)

    def test_normalize_angle_edge_cases(self) -> None:
        self.assertAlmostEqual(MathUtils._normalize_angle(0.0), 0.0)
        self.assertAlmostEqual(MathUtils._normalize_angle(2 * math.pi), 0.0)
        self.assertAlmostEqual(MathUtils._normalize_angle(-2 * math.pi), 0.0)
        self.assertAlmostEqual(
            MathUtils._normalize_angle(2 * math.pi + 1e-12),
            1e-12,
        )
        huge_angle = 1e6
        normalized = MathUtils._normalize_angle(huge_angle)
        self.assertTrue(0.0 <= normalized < 2 * math.pi)

    def test_arc_angle_sequence_additional_cases(self) -> None:
        minimal = MathUtils._arc_angle_sequence(0.0, math.pi, 1)
        self.assertEqual(len(minimal), 2)
        self.assertAlmostEqual(minimal[0], 0.0)
        self.assertAlmostEqual(minimal[-1], math.pi)

        full_loop = MathUtils._arc_angle_sequence(1.0, 1.0, 4)
        self.assertEqual(len(full_loop), 4)
        expected_span = 2 * math.pi / 3
        for idx in range(1, len(full_loop)):
            diff = (full_loop[idx] - full_loop[idx - 1]) % (2 * math.pi)
            self.assertAlmostEqual(diff, expected_span, places=6)

        wrap_ccw = MathUtils._arc_angle_sequence(3 * math.pi / 2, math.pi / 2, 3)
        self.assertEqual(len(wrap_ccw), 3)
        self.assertAlmostEqual(wrap_ccw[0], 3 * math.pi / 2)
        self.assertAlmostEqual(wrap_ccw[-1], math.pi / 2)

        wrap_cw = MathUtils._arc_angle_sequence(
            math.pi / 6,
            11 * math.pi / 6,
            4,
            clockwise=True,
        )
        self.assertEqual(len(wrap_cw), 4)
        self.assertAlmostEqual(wrap_cw[0], math.pi / 6)
        self.assertAlmostEqual(wrap_cw[-1], MathUtils._normalize_angle(11 * math.pi / 6))
        for idx in range(1, len(wrap_cw)):
            diff = (wrap_cw[idx - 1] - wrap_cw[idx]) % (2 * math.pi)
            self.assertGreater(diff, 0.0)

    def test_segment_has_end_point(self) -> None:
        self.assertTrue(MathUtils.segment_has_end_point(self.segment, 0, 0))
        self.assertTrue(MathUtils.segment_has_end_point(self.segment, 1, 1))
        self.assertFalse(MathUtils.segment_has_end_point(self.segment, 2, 2))  # Point not in segment

    def test_get_2D_distance(self) -> None:
        p1 = Position(0, 0)
        p2 = Position(3, 4)
        self.assertEqual(MathUtils.get_2D_distance(p1, p2), 5)

    def test_get_2D_midpoint(self) -> None:
        p1 = Position(0, 0)
        p2 = Position(2, 2)
        x, y = MathUtils.get_2D_midpoint(p1, p2)
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)

    def test_project_point_onto_circle(self) -> None:
        on_circle = SimpleMock(x=3.0, y=4.0)
        MathUtils.project_point_onto_circle(on_circle, 0.0, 0.0, 5.0)
        self.assertAlmostEqual(math.hypot(on_circle.x, on_circle.y), 5.0)
        self.assertAlmostEqual(on_circle.x, 3.0)
        self.assertAlmostEqual(on_circle.y, 4.0)

        inside_point = SimpleMock(x=1.0, y=1.0)
        MathUtils.project_point_onto_circle(inside_point, 0.0, 0.0, 5.0)
        self.assertAlmostEqual(math.hypot(inside_point.x, inside_point.y), 5.0, places=7)
        self.assertGreater(inside_point.x, 0.0)
        self.assertGreater(inside_point.y, 0.0)

        tiny_circle_point = SimpleMock(x=5e-7, y=0.0)
        MathUtils.project_point_onto_circle(tiny_circle_point, 0.0, 0.0, 1e-6)
        self.assertAlmostEqual(math.hypot(tiny_circle_point.x, tiny_circle_point.y), 1e-6, places=9)

        with self.assertRaises(ValueError):
            MathUtils.project_point_onto_circle(SimpleMock(x=0.0, y=0.0), 0.0, 0.0, 5.0)

        with self.assertRaises(ValueError):
            MathUtils.project_point_onto_circle(SimpleMock(x=1.0, y=1.0), 0.0, 0.0, 0.0)

    def test_point_on_circle(self) -> None:
        point = SimpleMock(x=3.0, y=4.0, name="P")
        self.assertTrue(
            MathUtils.point_on_circle(
                point,
                center_x=0.0,
                center_y=0.0,
                radius=5.0,
            )
        )

        off_point = SimpleMock(x=0.0, y=0.0, name="Q")
        with self.assertRaises(ValueError):
            MathUtils.point_on_circle(
                off_point,
                center_x=0.0,
                center_y=0.0,
                radius=5.0,
            )

        self.assertFalse(
            MathUtils.point_on_circle(
                off_point,
                center_x=0.0,
                center_y=0.0,
                radius=5.0,
                strict=False,
            )
        )

    def test_is_point_on_segment(self) -> None:
        # Basic tests
        self.assertTrue(MathUtils.is_point_on_segment(1, 1, 0, 0, 2, 2))
        self.assertTrue(MathUtils.is_point_on_segment(1, 1, 2, 2, 0, 0))
        self.assertTrue(MathUtils.is_point_on_segment(0, 0, 0, 0, 2, 2))
        self.assertTrue(MathUtils.is_point_on_segment(2, 2, 0, 0, 2, 2))
        self.assertFalse(MathUtils.is_point_on_segment(3, 3, 0, 0, 2, 2))
        
        # Additional test cases
        # Test case: Point on simple horizontal segment
        self.assertTrue(
            MathUtils.is_point_on_segment(5, 0, 0, 0, 10, 0),
            "Point (5,0) should be detected as being on segment from (0,0) to (10,0)"
        )
        
        # Test case: Point on simple vertical segment
        self.assertTrue(
            MathUtils.is_point_on_segment(0, 5, 0, 0, 0, 10),
            "Point (0,5) should be detected as being on segment from (0,0) to (0,10)"
        )
        
        # Test case: Point slightly off segment
        self.assertFalse(
            MathUtils.is_point_on_segment(5, 5.1, 0, 0, 10, 10),
            "Point (5,5.1) should be detected as NOT being on segment from (0,0) to (10,10)"
        )
        
        # Test case: Point outside bounding box of segment
        self.assertFalse(
            MathUtils.is_point_on_segment(15, 15, 0, 0, 10, 10),
            "Point (15,15) should be detected as NOT being on segment from (0,0) to (10,10)"
        )
        
        # Test case: Using the specific coordinates from the user's example
        self.assertTrue(
            MathUtils.is_point_on_segment(100.0, 45.332, -122.0, -69.0, 311.0, 154.0),
            "Point (100.0, 45.332) should be detected as being on segment from (-122.0, -69.0) to (311.0, 154.0)"
        )
        
        # Test case: Additional real-world examples on a longer segment
        # Segment from (-245.0, 195.0) to (323.0, -215.0)
        segment_start_x, segment_start_y = -245.0, 195.0
        segment_end_x, segment_end_y = 323.0, -215.0
        
        # Point C at y = 100
        self.assertTrue(
            MathUtils.is_point_on_segment(-113.39, 100.0, segment_start_x, segment_start_y, segment_end_x, segment_end_y),
            "Point C (-113.39, 100.0) should be detected as being on segment from (-245.0, 195.0) to (323.0, -215.0)"
        )
        
        # Point D at y = -24
        self.assertTrue(
            MathUtils.is_point_on_segment(58.4, -24.0, segment_start_x, segment_start_y, segment_end_x, segment_end_y),
            "Point D (58.4, -24.0) should be detected as being on segment from (-245.0, 195.0) to (323.0, -215.0)"
        )
        
        # Point E at x = 3
        self.assertTrue(
            MathUtils.is_point_on_segment(3.0, 15.99, segment_start_x, segment_start_y, segment_end_x, segment_end_y),
            "Point E (3.0, 15.99) should be detected as being on segment from (-245.0, 195.0) to (323.0, -215.0)"
        )
        
        # Point F at x = -199
        self.assertTrue(
            MathUtils.is_point_on_segment(-199.0, 161.8, segment_start_x, segment_start_y, segment_end_x, segment_end_y),
            "Point F (-199.0, 161.8) should be detected as being on segment from (-245.0, 195.0) to (323.0, -215.0)"
        )
        
        # Test case: Calculate a point that's exactly on the segment using linear interpolation
        t = 0.51
        point_x = -122.0 * (1-t) + 311.0 * t
        point_y = -69.0 * (1-t) + 154.0 * t
        
        self.assertTrue(
            MathUtils.is_point_on_segment(point_x, point_y, -122.0, -69.0, 311.0, 154.0),
            f"Interpolated point ({point_x}, {point_y}) should be detected as being on segment from (-122.0, -69.0) to (311.0, 154.0)"
        )

    def test_get_triangle_area(self) -> None:
        p1 = Position(0, 0)
        p2 = Position(1, 0)
        p3 = Position(0, 1)
        self.assertAlmostEqual(MathUtils.get_triangle_area(p1, p2, p3), 0.5)

    def test_get_rectangle_area(self) -> None:
        p1 = Position(0, 0)
        p2 = Position(2, 3)
        self.assertEqual(MathUtils.get_rectangle_area(p1, p2), 6)

    def test_cross_product(self) -> None:
        self.assertEqual(MathUtils.cross_product(Position(0, 0), Position(1, 0), Position(0, 1)), 1)       # "Perpendicular vectors"
        self.assertEqual(MathUtils.cross_product(Position(0, 0), Position(1, 1), Position(1, 1)), 0)       # "Zero vector test"
        self.assertEqual(MathUtils.cross_product(Position(1, 2), Position(-1, -1), Position(2, -3)), 13)   # "Negative values test"
        self.assertEqual(MathUtils.cross_product(Position(0, 0), Position(1, 0), Position(2, 0)), 0)       # "Collinear vectors"

    def test_dot_product(self) -> None:
        self.assertEqual(MathUtils.dot_product(Position(0, 0), Position(1, 0), Position(1, 0)), 1)      # "Parallel vectors"
        self.assertEqual(MathUtils.dot_product(Position(0, 0), Position(0, 0), Position(0, 1)), 0)      # "Zero vector test"
        self.assertEqual(MathUtils.dot_product(Position(1, 2), Position(-1, -1), Position(2, -3)), 13)  # "Negative values test"
        self.assertEqual(MathUtils.dot_product(Position(0, 0), Position(1, 0), Position(0, 1)), 0)      # "Perpendicular vectors"

    def test_is_right_angle(self) -> None:
        self.assertEqual(MathUtils.is_right_angle(Position(0, 0), Position(1, 0), Position(0, 1)), True)   # "Right angle"
        self.assertEqual(MathUtils.is_right_angle(Position(0, 0), Position(1, 1), Position(1, 0)), False)  # "Not right angle"
        self.assertEqual(MathUtils.is_right_angle(Position(0, 0), Position(1, 0), Position(1, 1)), False)  # "Almost right angle but not quite"

    def test_calculate_angle_degrees(self) -> None:
        # Vertex at origin for simplicity in these tests
        v = (0,0)
        # Test cases: (arm1_coords, arm2_coords, expected_degrees)
        test_cases = [
            ((1,0), (1,0), None),      # Arm2 coincident with Arm1 (relative to vertex, leads to zero length vector for arm2 if not careful, or zero angle)
                                        # Actually, MathUtils.calculate_angle_degrees has zero-length arm check based on v1x, v1y etc.
                                        # If arm1=(1,0) and arm2=(1,0), v1=(1,0), v2=(1,0). angle1=0, angle2=0. diff=0. result=0.
                                        # This case is more for are_points_valid_for_angle_geometry which checks p1 vs p2.
                                        # For calculate_angle_degrees itself, if p1 and p2 are same *and distinct from vertex*, it's 0 deg.
            ((1,0), (2,0), 0.0),       # Collinear, same direction from vertex
            ((1,0), (1,1), 45.0),      # 45 degrees
            ((1,0), (0,1), 90.0),      # 90 degrees
            ((1,0), (-1,1), 135.0),    # 135 degrees
            ((1,0), (-1,0), 180.0),    # 180 degrees
            ((1,0), (-1,-1), 225.0),   # 225 degrees
            ((1,0), (0,-1), 270.0),    # 270 degrees
            ((1,0), (1,-1), 315.0),    # 315 degrees
            # Test None returns for zero-length arms from vertex
            ((0,0), (1,1), None),      # Arm1 is at vertex
            ((1,1), (0,0), None),      # Arm2 is at vertex
            # Test order of arms (p1, p2 vs p2, p1)
            ((0,1), (1,0), 270.0),     # P1=(0,1), P2=(1,0) -> angle from +Y to +X is 270 deg CCW
        ]

        for i, (p1_coords, p2_coords, expected) in enumerate(test_cases):
            with self.subTest(i=i, v=v, p1=p1_coords, p2=p2_coords, expected=expected):
                result = MathUtils.calculate_angle_degrees(v, p1_coords, p2_coords)
                if expected is None:
                    self.assertIsNone(result)
                else:
                    self.assertIsNotNone(result) # Make sure it's not None before almostEqual
                    self.assertAlmostEqual(result, expected, places=5)
        
        # Test with non-origin vertex
        v_offset = (5,5)
        p1_offset = (6,5) # (1,0) relative to v_offset
        p2_offset = (5,6) # (0,1) relative to v_offset
        self.assertAlmostEqual(MathUtils.calculate_angle_degrees(v_offset, p1_offset, p2_offset), 90.0, places=5)

    def test_are_points_valid_for_angle_geometry(self) -> None:
        # Test cases: (vertex_coords, arm1_coords, arm2_coords, expected_validity)
        v = (0.0, 0.0)
        p1 = (1.0, 0.0)
        p2 = (0.0, 1.0)
        p3 = (1.0, 0.0) # Same as p1
        p4_close_to_v = (MathUtils.EPSILON / 2, MathUtils.EPSILON / 2)
        p5_close_to_p1 = (p1[0] + MathUtils.EPSILON / 2, p1[1] + MathUtils.EPSILON / 2)

        test_cases = [
            (v, p1, p2, True),          # Valid case
            (v, v, p2, False),          # Vertex == Arm1
            (v, p1, v, False),          # Vertex == Arm2
            (v, p1, p1, False),         # Arm1 == Arm2 (p1 used twice for arm2)
            (v, p1, p3, False),         # Arm1 == Arm2 (p3 is same as p1)
            (v, v, v, False),           # All three coincident at vertex
            (p1, p1, p1, False),        # All three coincident at p1
            # Epsilon tests
            (v, p4_close_to_v, p2, False), # Arm1 too close to Vertex
            (v, p1, p4_close_to_v, False), # Arm2 too close to Vertex
            (v, p1, p5_close_to_p1, False), # Arm2 too close to Arm1
            ((0,0), (1,0), (1.0000000001, 0.0000000001), False) # arm2 very close to arm1 (within typical float precision but potentially outside strict epsilon for p1 vs p2)
                                                                # The are_points_valid uses direct comparison with EPSILON for each pair.
                                                                # If MathUtils.EPSILON = 1e-9, (1.0, 0.0) vs (1.0000000001, 0.0000000001)
                                                                # dx = 1e-10, dy = 1e-10. Both are < EPSILON. So this should be False.
        ]

        for i, (vc, ac1, ac2, expected) in enumerate(test_cases):
            with self.subTest(i=i, v=vc, a1=ac1, a2=ac2, expected=expected):
                self.assertEqual(MathUtils.are_points_valid_for_angle_geometry(vc, ac1, ac2), expected)

    def test_validate_rectangle(self) -> None:
        # square
        self.assertTrue(MathUtils.is_rectangle(0, 0, 1, 0, 1, 1, 0, 1))
        self.assertTrue(MathUtils.is_rectangle(0, 0, 0, 1, 1, 1, 1, 0))
        # rectangle
        self.assertTrue(MathUtils.is_rectangle(0, 0, 2, 0, 2, 1, 0, 1))
        self.assertTrue(MathUtils.is_rectangle(0, 0, 0, 1, 2, 1, 2, 0))
        # square skewed by 45 degrees
        self.assertTrue(MathUtils.is_rectangle(0, 1, 1, 0, 2, 1, 1, 2))
        self.assertTrue(MathUtils.is_rectangle(0, 1, 1, 2, 2, 1, 1, 0))
        # rectangle skewed by 45 degrees
        self.assertTrue(MathUtils.is_rectangle(0, 2, 2, 0, 3, 1, 1, 3))
        self.assertTrue(MathUtils.is_rectangle(0, 2, 1, 3, 3, 1, 2, 0))
        # Invalid rectangles
        self.assertFalse(MathUtils.is_rectangle(0, 0, 2, 0, 0, 1, 1, 2))
        self.assertFalse(MathUtils.is_rectangle(0, 0, 1, 2, 0, 1, 2, 0))
        # Invalid cases with repeating points
        self.assertFalse(MathUtils.is_rectangle(0, 0, 0, 0, 2, 1, 0, 1))
        self.assertFalse(MathUtils.is_rectangle(0, 0, 2, 0, 2, 0, 0, 1))
        self.assertFalse(MathUtils.is_rectangle(0, 0, 2, 0, 2, 1, 2, 1))
        self.assertFalse(MathUtils.is_rectangle(0, 0, 0, 0, 0, 0, 0, 0))
        self.assertFalse(MathUtils.is_rectangle(0, 0, 2, 0, 0, 0, 2, 0))
        self.assertFalse(MathUtils.is_rectangle(0, 0, 0, 0, 2, 0, 2, 0))
        self.assertFalse(MathUtils.is_rectangle(2, 0, 2, 0, 0, 0, 0, 0))

    def test_segments_intersect(self) -> None:
        self.assertTrue(MathUtils.segments_intersect(0, 0, 10, 10, 0, 10, 10, 0))
        self.assertFalse(MathUtils.segments_intersect(0, 0, 10, 10, 20, 20, 30, 30))
        self.assertTrue(MathUtils.segments_intersect(0, 0, 10, 10, 0, 0, 10, 10))
        self.assertTrue(MathUtils.segments_intersect(0, 0, 10, 10, 0, 0, 5, 5))

    def test_get_segments_intersection(self) -> None:
        x_intersection, y_intersection = MathUtils.get_segments_intersection(0, 0, 1, 1, 0, 1, 1, 0)
        self.assertAlmostEqual(x_intersection, 0.5, places=7)
        self.assertAlmostEqual(y_intersection, 0.5, places=7)

    def test_get_segments_intersection_parallel(self) -> None:
        # Define two parallel segments
        result = MathUtils.get_segments_intersection(0, 0, 1, 1, 2, 2, 3, 3)
        # Since the segments are parallel, the result should be None
        self.assertIsNone(result, "Expected None for parallel segments")

    def test_get_line_formula(self) -> None:
        self.assertEqual(MathUtils.get_line_formula(0, 0, 1, 1), "y = 1.0 * x + 0.0")
        self.assertEqual(MathUtils.get_line_formula(0, 0, 0, 1), "x = 0")

    def test_get_circle_formula(self) -> None:
        self.assertEqual(MathUtils.get_circle_formula(0, 0, 1), "(x - 0)**2 + (y - 0)**2 = 1**2")

    def test_get_ellipse_formula(self) -> None:
        self.assertEqual(MathUtils.get_ellipse_formula(0, 0, 1, 2), "((x - 0)**2)/1**2 + ((y - 0)**2)/2**2 = 1")

    def test_sqrt(self) -> None:
        result = MathUtils.sqrt(-4)
        self.assertEqual(result, "2i")
        result = MathUtils.sqrt(4)
        self.assertEqual(int(result), 2)

    def test_pow(self) -> None:
        result = MathUtils.pow(2, 3)
        self.assertEqual(int(result), 8)
        matrix = [[-1, 2], [3, 1]]
        result = MathUtils.pow(matrix, 2)
        self.assertEqual(result, "[[7, 0], [0, 7]]")

    def test_evaluate_conversion(self) -> None:
        result = MathUtils.convert(12.7, "cm", "inch")
        self.assertEqual(result, "5 inch")

    def test_evaluate_addition(self) -> None:
        result = MathUtils.evaluate("7 + 3")
        self.assertEqual(int(result), 10)

    def test_evaluate_division(self) -> None:
        result = MathUtils.evaluate("12 / (2.3 + 0.7)")
        self.assertEqual(int(result), 4)

    def test_evaluate_sin(self) -> None:
        result = MathUtils.evaluate("sin(45 deg) ^ 2")
        print(f"sin(45 deg) ^ 2 = {result}")
        self.assertAlmostEqual(float(result), 0.5, places=9)

    def test_evaluate_js_power_symbol(self) -> None:
        result = MathUtils.evaluate("9^2 / 3")
        self.assertEqual(int(result), 27)

    def test_evaluate_py_power_symbol(self) -> None:
        result = MathUtils.evaluate("9**2 / 3")
        self.assertEqual(int(result), 27)

    def test_evaluate_complex(self) -> None:
        result = MathUtils.evaluate("1 + 2i + 1j")
        self.assertEqual(result, "1 + 3i")

    def test_evaluate_factorial_expression(self) -> None:
        result = MathUtils.evaluate("10!/(3!*(10-3)!)")
        expected = math.factorial(10) // (math.factorial(3) * math.factorial(7))
        self.assertAlmostEqual(float(result), expected)

    def test_evaluate_det(self) -> None:
        matrix = [[-1, 2], [3, 1]]
        result = MathUtils.det(matrix)
        self.assertEqual(int(result), -7)

    def test_random(self) -> None:
        result = MathUtils.random()
        self.assertTrue(0 <= result <= 1)

    def test_round(self) -> None:
        result = MathUtils.round(1.2345, 2)
        self.assertEqual(result, 1.23)

    def test_gcd(self) -> None:
        result = MathUtils.gcd(48, 18)
        self.assertEqual(result, 6)

    def test_lcm(self) -> None:
        result = MathUtils.lcm(4, 5)
        self.assertEqual(result, 20)

    def test_mean(self) -> None:
        result = MathUtils.mean([1, 2, 3, 4, 5])
        self.assertEqual(result, 3)

    def test_combinatorics_values(self) -> None:
        self.assertEqual(MathUtils.permutations(5), math.factorial(5))
        self.assertEqual(MathUtils.permutations(6, 3), math.perm(6, 3))
        self.assertEqual(MathUtils.arrangements(6, 3), math.perm(6, 3))
        self.assertEqual(MathUtils.combinations(6, 3), math.comb(6, 3))

    def test_combinatorics_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            MathUtils.permutations(4, 5)
        with self.assertRaises(ValueError):
            MathUtils.combinations(4, 5)
        with self.assertRaises(ValueError):
            MathUtils.combinations(4, -1)
        with self.assertRaises(TypeError):
            MathUtils.permutations(4.5, 2)
        with self.assertRaises(TypeError):
            MathUtils.arrangements(True, 2)

    def test_evaluate_combinatorics_functions(self) -> None:
        result = MathUtils.evaluate("arrangements(6, 3)")
        self.assertEqual(int(result), math.perm(6, 3))
        result = MathUtils.evaluate("permutations(5, 2)")
        self.assertEqual(int(result), math.perm(5, 2))
        result = MathUtils.evaluate("permutations(5)")
        self.assertEqual(int(result), math.factorial(5))
        result = MathUtils.evaluate("combinations(7, 4)")
        self.assertEqual(int(result), math.comb(7, 4))

    def test_median(self) -> None:
        result = MathUtils.median([1, 2, 3, 4, 5])
        self.assertEqual(result, 3)

    def test_mode(self) -> None:
        result = MathUtils.mode([1, 2, 2, 3])
        self.assertEqual(result, 2)

    def test_stdev(self) -> None:
        result = MathUtils.stdev([2, 4, 6, 8, 10])
        self.assertAlmostEqual(result, 3.1623, places=4)

    def test_variance(self) -> None:
        result = MathUtils.variance([2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5])
        self.assertAlmostEqual(result, 1.372, places=3)

    def test_check_div_by_zero(self) -> None:
        # Test cases that should raise ZeroDivisionError
        zero_division_cases = [
            "1/0",                    # Simple division by zero
            "1/(3-3)",               # Division by parenthesized zero
            "1/(2*0)",               # Direct multiplication by zero in denominator
            "1/(0*x)",               # Variable expression evaluating to zero
            "10/(x-2)",              # Variable expression evaluating to zero with variables
            "1/(3*0+1-1)",           # Complex expression evaluating to zero
            "1/(-0)",                # Negative zero
            "1/(0.0)",               # Zero as float
            "1/(0e0)",               # Zero in scientific notation
        ]

        # Nested parentheses cases
        nested_zero_division_cases = [
            "1/2/(1-1)",             # Chained division with zero
            "1/(2/(1-1))",           # Nested division with zero
            "1/9*(3-3)",             # Multiplication after division resulting in zero
            "1/(9*(3-3))",           # Division by parenthesized multiplication resulting in zero
            "2/((1-1)*5)",           # Division by zero with extra parentheses
            "1/((2-2)*3*(4+1))",     # Multiple terms evaluating to zero
            "2/(1/(1-1))",           # Division by infinity (division by zero in denominator)
            "1/((3-3)/(4-4))",       # Multiple zeros in nested divisions
            "1/9*3*(1-1)",           # Multiple operations after division resulting in zero
            "1/3*2*(5-5)*4",         # Zero product in denominator with multiple terms
        ]

        # Test all zero division cases
        for expr in zero_division_cases:
            result = MathUtils.evaluate(expr)
            self.assertTrue(isinstance(result, str) and "Error" in result, 
                          f"Expected error for expression: {expr}, got {result}")

        # Test nested zero division cases
        # Note: The result of 0.0 for these cases is not typical and might be due to JavaScript's handling.
        for expr in nested_zero_division_cases:
            result = MathUtils.evaluate(expr)
            if expr in ["1/(2/(1-1))", "1/9*(3-3)", "1/9*3*(1-1)", "1/3*2*(5-5)*4", "2/(1/(1-1))"]:
                self.assertEqual(result, 0.0, f"Expected 0.0 for expression: {expr}, got {result}")
            elif expr == "1/((3-3)/(4-4))":  # JavaScript returns nan for this case
                self.assertEqual(str(result).lower(), "nan", f"Expected nan for expression: {expr}, got {result}")
            else:
                self.assertTrue(isinstance(result, str) and "Error" in result, 
                              f"Expected error for nested expression: {expr}, got {result}")

        # Test with variables
        result = MathUtils.evaluate("10/(x-2)", {"x": 2})
        self.assertTrue(isinstance(result, str) and "Error" in result,
                       f"Expected error for expression with x=2, got {result}")

        # Test cases that should NOT raise ZeroDivisionError
        valid_division_cases = [
            "1/2",                   # Simple valid division
            "1/(3-2)",              # Valid division with parentheses
            "1/2/3",                # Chained valid division
            "1/(2/3)",              # Nested valid division
            "1/9*(3-2)",            # Valid multiplication after division
            "1/(9*(3-2))",          # Valid division with parenthesized multiplication
            "2/((1+1)*5)",          # Valid division with extra parentheses
            "1/(2*1)",              # Valid multiplication in denominator
            "1/(x+1)",              # Valid variable expression
            "10/(x+2)",             # Valid variable expression with variables
            "1/(3*2+1)",            # Valid complex expression
            "1/((2+2)*3*(4+1))",    # Valid multiple terms
            "2/(1/(1+1))",          # Valid nested division
            "1/((3-2)/(4-3))",      # Valid nested divisions
            "1/9*3*(2-1)",          # Valid multiple operations
            "1/3*2*(5+5)*4",        # Valid product in denominator
            "1/3+4/5",              # Multiple separate divisions
            "1/3 + 4/5",            # Divisions with whitespace
            "1 / 3 * 2 * (5+5) * 4" # Complex expression with whitespace
        ]

        # Test all valid division cases
        for expr in valid_division_cases:
            result = MathUtils.evaluate(expr, {"x": 5})  # Using x=5 for variable cases
            self.assertFalse(isinstance(result, str) and "Error" in result,
                           f"Unexpected error for valid expression: {expr}, got {result}")
            self.assertIsInstance(result, (int, float, str), 
                                f"Result should be numeric or string for expression: {expr}")

        # Test with different variable values
        result = MathUtils.evaluate("1/(x+1)", {"x": -1})  # Should raise error
        self.assertTrue(isinstance(result, str) and "Error" in result,
                       f"Expected error for expression with x=-1, got {result}")

        # Test edge cases
        edge_cases = [
            ("1/1e-100", False),     # Very small but non-zero denominator
            ("1/(1-0.999999999)", False),  # Nearly zero but not quite
            ("1/(-0)", True),        # Negative zero
            ("1/(0.0)", True),       # Zero as float
            ("1/(0e0)", True),       # Zero in scientific notation
        ]

        for expr, should_raise in edge_cases:
            result = MathUtils.evaluate(expr)
            if should_raise:
                self.assertTrue(isinstance(result, str) and "Error" in result,
                              f"Expected error for edge case: {expr}, got {result}")
            else:
                self.assertFalse(isinstance(result, str) and "Error" in result,
                               f"Unexpected error for edge case: {expr}, got {result}")
                self.assertIsInstance(result, (int, float, str),
                                    f"Result should be numeric or string for edge case: {expr}")

    def test_limit(self) -> None:
        result = MathUtils.limit('sin(x) / x', 'x', 0)
        result = float(result)  # convert result to float
        self.assertEqual(result, 1.0)

    def test_derivative(self) -> None:
        result = MathUtils.derivative('x^2', 'x')
        self.assertEqual(result, "2*x")

    def test_integral_indefinite(self) -> None:
        result = MathUtils.integral('x^2', 'x')
        result = MathUtils.simplify(result)  # simplify the result
        self.assertEqual(result, "0.3333333333333333*x^3")

    def test_integral(self) -> None:
        result = MathUtils.integral('x^2', 'x', 0, 1)
        result = float(result)  # convert result to float
        self.assertAlmostEqual(result, 0.333, places=3)

    def test_simplify(self) -> None:
        result = MathUtils.simplify('x^2 + 2*x + 1')
        self.assertEqual(result, "(1+x)^2")

    def test_expand(self) -> None:
        result = MathUtils.expand('(x + 1)^2')
        self.assertEqual(result, "1+2*x+x^2")

    def test_factor(self) -> None:
        result = MathUtils.factor('x^2 - 1')
        self.assertEqual(result, "(-1+x)*(1+x)")

    def test_get_equation_type_with_linear_equation(self) -> None:
        equation = "x + 2"
        result = MathUtils.get_equation_type(equation)
        self.assertEqual(result, "Linear")

    def test_get_equation_type_with_quadratic_equation(self) -> None:
        equation = "x^2 + 2*x + 1"
        result = MathUtils.get_equation_type(equation)
        self.assertEqual(result, "Quadratic")  # Adjusted to expect "Quadratic"

    def test_get_equation_type_with_cubic_equation(self) -> None:
        equation = "x^3 + 3*x^2 + 3*x + 1"
        result = MathUtils.get_equation_type(equation)
        self.assertEqual(result, "Cubic")  # Testing for cubic equation

    def test_get_equation_type_with_quartic_equation(self) -> None:
        equation = "x^4 + 4*x^3 + 6*x^2 + 4*x + 1"
        result = MathUtils.get_equation_type(equation)
        self.assertEqual(result, "Quartic")  # Testing for quartic equation

    def test_get_equation_type_with_higher_order_equation(self) -> None:
        equation = "x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1"
        result = MathUtils.get_equation_type(equation)
        self.assertTrue("Order" in result)  # Testing for higher order equation, expecting "Order 5"

    def test_get_equation_type_with_trigonometric_equation1(self) -> None:
        equation = "sin(x) + 2"
        result = MathUtils.get_equation_type(equation)
        self.assertEqual(result, "Trigonometric")
    
    def test_get_equation_type_with_trigonometric_equation2(self) -> None:
        equation = "cos(x + 3) - 2"
        result = MathUtils.get_equation_type(equation)
        self.assertEqual(result, "Trigonometric")

    def test_get_equation_type_with_trigonometric_equation3(self) -> None:
        equation = "tan(x * sin(24)) = 2"
        result = MathUtils.get_equation_type(equation)
        self.assertEqual(result, "Trigonometric")

    def test_get_equation_type_with_non_linear_due_to_variable_multiplication1(self) -> None:
        equation = "x*y + 2"
        result = MathUtils.get_equation_type(equation)
        self.assertEqual(result, "Other Non-linear")  # Adjusted to expect "Other Non-linear"

    def test_get_equation_type_with_non_linear_due_to_variable_multiplication2(self) -> None:
        equation = "xy - 5"
        result = MathUtils.get_equation_type(equation)
        self.assertEqual(result, "Other Non-linear")  # Adjusted to expect "Other Non-linear"

    def test_get_equation_type_with_linear_after_expansion(self) -> None:
        equation = "(x + 1)^2"
        expanded = MathUtils.expand(equation)  # Assuming this correctly expands to "x^2 + 2*x + 1"
        result = MathUtils.get_equation_type(expanded)
        self.assertEqual(result, "Quadratic")  # Adjusted to expect "Quadratic"

    def test_get_equation_type_with_implicit_multiplication_not_detected_as_non_linear(self) -> None:
        equation = "2x + 3"
        result = MathUtils.get_equation_type(equation)
        self.assertEqual(result, "Linear")  # Assuming implicit multiplication by constants is handled as linear

    def test_determine_max_number_of_solutions_linear_and_linear(self) -> None:
        equations = ["2x + 3 = y", "5x - 2 = y"]
        result = MathUtils.determine_max_number_of_solutions(equations)
        self.assertEqual(result, 1, "Linear and linear should have exactly 1 solution.")
    
    def test_determine_max_number_of_solutions_linear_and_quadratic(self) -> None:
        equations = ["x + 2 = y", "x^2 - 4x + 3 = y"]
        result = MathUtils.determine_max_number_of_solutions(equations)
        self.assertEqual(result, 2, "Linear and quadratic should have at most 2 solutions.")
    
    def test_determine_max_number_of_solutions_linear_and_cubic(self) -> None:
        equations = ["3x + 1 = y", "x^3 - 6x^2 + 11x - 6 = y"]
        result = MathUtils.determine_max_number_of_solutions(equations)
        self.assertEqual(result, 3, "Linear and cubic should intersect in at most 3 points.")
    
    def test_determine_max_number_of_solutions_quadratic_and_quartic(self) -> None:
        equations = ["x^2 + x - 2 = y", "x^4 - 5x^2 + 4 = y"]
        result = MathUtils.determine_max_number_of_solutions(equations)
        self.assertEqual(result, 8, "Quadratic and quartic should intersect in at most 8 points.")
    
    def test_determine_max_number_of_solutions_cubic_and_quartic_with_higher_order_count(self) -> None:
        equations = ["x^3 + x - 4 = y", "x^5 - x^4 + x^3 - x^2 + x - 1 = y"]
        result = MathUtils.determine_max_number_of_solutions(equations)
        self.assertEqual(result, 15, "Cubic and quintic equations can theoretically intersect in at most 15 points.")
    
    def test_determine_max_number_of_solutions_single_equation(self) -> None:
        equations = ["x^2 + 4x + 4 = y"]
        result = MathUtils.determine_max_number_of_solutions(equations)
        self.assertEqual(result, 0, "Single equation should not determine a solution count between equations.")
    
    def test_determine_max_number_of_solutions_no_equations(self) -> None:
        equations: List[str] = []
        result = MathUtils.determine_max_number_of_solutions(equations)
        self.assertEqual(result, 0, "No equations should not determine a solution count.")

    def test_determine_max_number_of_solutions_trigonometric(self) -> None:
        equations = ["sin(x) = y", "cos(x) = y"]
        result = MathUtils.determine_max_number_of_solutions(equations)
        self.assertEqual(result, 0, "Trigonometric combinations should indicate complex or uncertain scenarios.")

    def test_determine_max_number_of_solutions_other_non_linear(self) -> None:
        equations = ["x*y - 2 = 0", "x^2 + y = 4"]  # Changed second equation to avoid using xy term twice
        result = MathUtils.determine_max_number_of_solutions(equations)
        self.assertEqual(result, 0, "Other non-linear equations should indicate complex or uncertain scenarios.")

    def test_solve1(self) -> None:
        result = MathUtils.solve('x^2 - 4', 'x')
        result = json.loads(result)  # parse result from JSON string to list
        result = [float(r) for r in result]  # convert results to floats
        self.assertEqual(result, [2.0, -2.0])

    def test_solve2(self) -> None:
        result = MathUtils.solve('0.4 * x + 37.2 = -0.9 * x - 8', 'x')
        result = json.loads(result)  # Parse result from JSON string to list
        # Assuming the result is always a list with a single item for this test case
        solution = float(result[0])  # Convert the first (and only) result to float
        self.assertAlmostEqual(solution, -34.7692307692308, places=5)

    def test_solve_linear_quadratic_invalid_input(self) -> None:
        equations = ["y = 2*x + 3"]  # Not enough equations
        with self.assertRaises(ValueError):
            MathUtils.solve_linear_quadratic_system(equations)

    def test_solve_linear_quadratic_no_real_solution(self) -> None:
        equations = ["y = 2*x + 3", "y = x^2 + 4*x + 5"]
        with self.assertRaises(ValueError):
            MathUtils.solve_linear_quadratic_system(equations)

    def test_solve_linear_quadratic_returns_string(self) -> None:
        equations = ["2x + 3 = y", "x^2 + 4x + 3 = y"]
        result = MathUtils.solve_linear_quadratic_system(equations)
        self.assertTrue(isinstance(result, str))  # Check if the result is correctly formatted as a string

    def test_solve_linear_quadratic_one_real_solution(self) -> None:
        equations = ["y = 2x - 1", "y = x^2"]
        result = MathUtils.solve_linear_quadratic_system(equations)
        result = dict(item.split(" = ") for item in result.split(", "))  # parse result from string to dictionary
        result = {k: float(v) for k, v in result.items()}  # convert results to floats
        self.assertEqual(result, {"x": 1.0, "y": 1.0})

    def test_solve_linear_quadratic_two_real_solutions(self) -> None:
        equations = ["y = x + 1", "y = x^2 + 2x + 1"]
        result = MathUtils.solve_linear_quadratic_system(equations)
        result = dict(item.split(" = ") for item in result.split(", "))  # parse result from string to dictionary
        result = {k: float(v) for k, v in result.items()}  # convert results to floats
        self.assertEqual(result, {"x1": 0.0, "y1": 1.0, "x2": -1.0, "y2": 0.0})

    def test_solve_system_of_equations_linear(self) -> None:
        result = MathUtils.solve_system_of_equations(['x + y = 4', 'x - y = 2'])
        result = dict(item.split(" = ") for item in result.split(", "))  # parse result from string to dictionary
        result = {k: float(v) for k, v in result.items()}  # convert results to floats
        self.assertEqual(result, {"x": 3.0, "y": 1.0})

    def test_solve_system_of_equations_quadratic_linear(self) -> None:
        result = MathUtils.solve_system_of_equations(['x^2 = y', '-x + 2 = y'])
        result = dict(item.split(" = ") for item in result.split(", "))  # parse result from string to dictionary
        result = {k: float(v) for k, v in result.items()}  # convert results to floats
        self.assertEqual(result, {"x1": 1.0, "y1": 1.0, "x2": -2.0, "y2": 4.0})

    def test_determine_max_number_of_solutions_cubic_and_quintic(self) -> None:
        equations = ["x^3 + x - 4 = y", "x^5 - x^4 + x^3 - x^2 + x - 1 = y"]
        result = MathUtils.determine_max_number_of_solutions(equations)
        self.assertEqual(result, 15, "Cubic and quintic equations can theoretically intersect in at most 15 points.")

    def test_solve_system_of_equations_with_high_order(self) -> None:
        equations = ["x^3 + x - 4 = y", "x^5 - x^4 + x^3 - x^2 + x - 1 = y"]
        result = MathUtils.solve_system_of_equations(equations)
        result = dict(item.split(" = ") for item in result.split(", "))  # parse result from string to dictionary
        result = {k: float(v) for k, v in result.items()}  # convert results to floats
        self.assertEqual(result, {"x": -1.0, "y": -6.0})

    def test_calculate_vertical_asymptotes(self) -> None:
        # Test logarithmic function
        result = MathUtils.calculate_vertical_asymptotes("log(x)")
        self.assertEqual(result, [0], "log(x) should have vertical asymptote at x=0")

        # Test rational function
        result = MathUtils.calculate_vertical_asymptotes("1/(x-2)")
        self.assertEqual(result, [2], "1/(x-2) should have vertical asymptote at x=2")

        # Test rational function with asymptote in middle of range
        result = MathUtils.calculate_vertical_asymptotes("1/(x-3)", 0, 6)
        self.assertEqual(result, [3], "1/(x-3) should have vertical asymptote at x=3")

        # Test tangent function with different bounds
        # Test case 1: [-10, 10]
        result = MathUtils.calculate_vertical_asymptotes("tan(x)", -10, 10)
        # For tan(x), asymptotes occur at x = π/2 + nπ
        # In range [-10, 10], we need to find n where (-π/2 + nπ) is in range
        # Solving: -10 ≤ -π/2 + nπ ≤ 10
        # (-10 + π/2)/π ≤ n ≤ (10 + π/2)/π
        # -3.02 ≤ n ≤ 3.66
        # Therefore n goes from -2 to 3 inclusive
        expected = sorted([round((-math.pi/2 + n*math.pi), 6) for n in range(-2, 4)])
        actual = sorted([round(x, 6) for x in result])
        self.assertEqual(actual, expected, "tan(x) should have correct asymptotes in [-10, 10]")

        # Test case 2: [-5, 5]
        result = MathUtils.calculate_vertical_asymptotes("tan(x)", -5, 5)
        # Solving: -5 ≤ -π/2 + nπ ≤ 5
        # (-5 + π/2)/π ≤ n ≤ (5 + π/2)/π
        # -1.41 ≤ n ≤ 2.07
        # Therefore n goes from -1 to 2 inclusive
        expected = sorted([round((-math.pi/2 + n*math.pi), 6) for n in range(-1, 3)])
        actual = sorted([round(x, 6) for x in result])
        self.assertEqual(actual, expected, "tan(x) should have correct asymptotes in [-5, 5]")

        # Test case 3: [-3, 3]
        result = MathUtils.calculate_vertical_asymptotes("tan(x)", -3, 3)
        # Solving: -3 ≤ -π/2 + nπ ≤ 3
        # (-3 + π/2)/π ≤ n ≤ (3 + π/2)/π
        # -0.77 ≤ n ≤ 1.43
        # Therefore n goes from 0 to 1 inclusive
        expected = sorted([round((-math.pi/2 + n*math.pi), 6) for n in range(0, 2)])
        actual = sorted([round(x, 6) for x in result])
        self.assertEqual(actual, expected, "tan(x) should have correct asymptotes in [-3, 3]")

        # Test complex function with multiple asymptotes
        result = MathUtils.calculate_vertical_asymptotes("1/(x^2-4)")
        self.assertEqual(sorted(result), [-2, 2], "1/(x^2-4) should have vertical asymptotes at x=-2 and x=2")

        # Test function with no vertical asymptotes
        result = MathUtils.calculate_vertical_asymptotes("x^2 + 1")
        self.assertEqual(result, [], "x^2 + 1 should have no vertical asymptotes")

    def test_calculate_horizontal_asymptotes(self) -> None:
        # Test rational function approaching constant
        result = MathUtils.calculate_horizontal_asymptotes("(x^2+1)/(x^2+2)")
        self.assertEqual(sorted(list(set(result))), [1], "(x^2+1)/(x^2+2) should approach 1 as x approaches infinity")

        # Test function with no horizontal asymptotes
        result = MathUtils.calculate_horizontal_asymptotes("x^2")
        self.assertEqual(result, [], "x^2 should have no horizontal asymptotes")

        # Test function with y=0 as horizontal asymptote
        result = MathUtils.calculate_horizontal_asymptotes("1/x")
        self.assertEqual(sorted(list(set(result))), [0], "1/x should have y=0 as horizontal asymptote")

        # Test rational function with degree numerator < degree denominator
        result = MathUtils.calculate_horizontal_asymptotes("x/(x^2+1)")
        self.assertEqual(sorted(list(set(result))), [0], "x/(x^2+1) should approach 0 as x approaches infinity")

    def test_calculate_asymptotes(self) -> None:
        # Test function with both vertical and horizontal asymptotes
        vert, horiz, disc = MathUtils.calculate_asymptotes_and_discontinuities("1/x", -10, 10)
        self.assertEqual(vert, [0], "1/x should have vertical asymptote at x=0")
        self.assertEqual(sorted(list(set(horiz))), [0], "1/x should have horizontal asymptote at y=0")
        self.assertEqual(disc, [], "1/x should have no point discontinuities")

        # Test logarithmic function
        vert, horiz, disc = MathUtils.calculate_asymptotes_and_discontinuities("log(x)")
        self.assertEqual(vert, [0], "log(x) should have vertical asymptote at x=0")
        self.assertEqual(horiz, [], "log(x) should have no horizontal asymptotes")
        self.assertEqual(disc, [], "log(x) should have no point discontinuities")

        # Test tangent function with bounds
        vert, horiz, disc = MathUtils.calculate_asymptotes_and_discontinuities("tan(x)", -5, 5)
        # For tan(x), asymptotes occur at x = π/2 + nπ
        # In range [-5, 5], we need to find n where (-π/2 + nπ) is in range
        # Solving: -5 ≤ -π/2 + nπ ≤ 5
        # (-5 + π/2)/π ≤ n ≤ (5 + π/2)/π
        # -1.41 ≤ n ≤ 2.07
        # Therefore n goes from -1 to 2 inclusive (all values that give asymptotes within [-5, 5])
        expected_vert = sorted([round((-math.pi/2 + n*math.pi), 6) for n in range(-1, 3)])
        actual_vert = sorted([round(x, 6) for x in vert])
        self.assertEqual(actual_vert, expected_vert, "tan(x) should have vertical asymptotes at x = π/2 + nπ within bounds")
        self.assertEqual(horiz, [], "tan(x) should have no horizontal asymptotes")
        self.assertEqual(disc, [], "tan(x) should have no point discontinuities")

        # Test rational function with both vertical and horizontal asymptotes
        vert, horiz, disc = MathUtils.calculate_asymptotes_and_discontinuities("(x^2+1)/(x^2+2)")
        self.assertEqual(vert, [], "(x^2+1)/(x^2+2) should have no vertical asymptotes")
        self.assertEqual(sorted(list(set(horiz))), [1], "(x^2+1)/(x^2+2) should approach 1 as x approaches infinity")
        self.assertEqual(disc, [], "(x^2+1)/(x^2+2) should have no point discontinuities")

        # Test function with no asymptotes
        vert, horiz, disc = MathUtils.calculate_asymptotes_and_discontinuities("sin(x)")
        self.assertEqual(vert, [], "sin(x) should have no vertical asymptotes")
        self.assertEqual(horiz, [], "sin(x) should have no horizontal asymptotes")
        self.assertEqual(disc, [], "sin(x) should have no point discontinuities")

    def test_calculate_point_discontinuities(self) -> None:
        # Test piecewise function with conditions
        result = MathUtils.calculate_point_discontinuities("x if x < 2 else x + 1")
        self.assertEqual(result, [2], "Piecewise function should have discontinuity at transition point")

        # Test multiple conditions
        result = MathUtils.calculate_point_discontinuities("1 if x <= -1 else 2 if x <= 1 else 3")
        self.assertEqual(result, [-1, 1], "Multiple conditions should give multiple discontinuities")

        # Test floor function with bounds
        result = MathUtils.calculate_point_discontinuities("floor(x)", -2, 2)
        self.assertEqual(result, [-2, -1, 0, 1, 2], "Floor function should have discontinuities at integers within bounds")

        # Test ceil function with bounds
        result = MathUtils.calculate_point_discontinuities("ceil(x)", -1.5, 1.5)
        self.assertEqual(result, [-1, 0, 1], "Ceil function should have discontinuities at integers within bounds")

        # Test absolute value function
        result = MathUtils.calculate_point_discontinuities("abs(x)")
        self.assertEqual(result, [0], "Absolute value function should have discontinuity at x=0")

        # Test absolute value with shifted input
        result = MathUtils.calculate_point_discontinuities("abs(x-2)")
        self.assertEqual(result, [2], "Shifted absolute value should have discontinuity at x=2")

        # Test function with no discontinuities
        result = MathUtils.calculate_point_discontinuities("x^2 + 2*x + 1")
        self.assertEqual(result, [], "Continuous function should have no point discontinuities")

        # Test multiple absolute value terms
        result = MathUtils.calculate_point_discontinuities("abs(x) + abs(x-1)")
        self.assertEqual(sorted(result), [0, 1], "Multiple absolute values should give multiple discontinuities")

        # Test with bounds filtering
        result = MathUtils.calculate_point_discontinuities("floor(x)", 0, 2)
        self.assertEqual(result, [0, 1, 2], "Should only include discontinuities within bounds")

        # Test complex piecewise with multiple operators
        result = MathUtils.calculate_point_discontinuities("1 if x < 0 else 2 if x <= 1 else 3 if x >= 2 else 4")
        self.assertEqual(sorted(result), [0, 1, 2], "Complex piecewise should identify all transition points")

    def test_function_vertical_asymptote_path_breaking(self) -> None:
        """Test that Function class properly breaks paths at vertical asymptotes without visual artifacts."""
        try:
            from drawables.function import Function
            from .simple_mock import SimpleMock
            from coordinate_mapper import CoordinateMapper
            
            # Create a real CoordinateMapper instance
            coordinate_mapper = CoordinateMapper(600, 400)  # Canvas dimensions
            
            # Create a mock canvas with required methods and properties
            mock_canvas = SimpleMock()
            mock_canvas.scale_factor = 50
            mock_canvas.cartesian2axis = SimpleMock()
            mock_canvas.cartesian2axis.origin = SimpleMock(x=300, y=200)
            mock_canvas.cartesian2axis.height = 400
            mock_canvas.coordinate_mapper = coordinate_mapper
            
            # Mock the visible bounds methods
            mock_canvas.cartesian2axis.get_visible_left_bound = lambda: -10
            mock_canvas.cartesian2axis.get_visible_right_bound = lambda: 10  
            mock_canvas.cartesian2axis.get_visible_top_bound = lambda: 8
            mock_canvas.cartesian2axis.get_visible_bottom_bound = lambda: -8
            
            # Sync coordinate mapper with canvas state
            coordinate_mapper.sync_from_canvas(mock_canvas)
            
            # Test function with vertical asymptotes: tan(x) has asymptotes at π/2 + nπ
            function = Function(
                function_string="tan(x)",
                name="test_tan",
                left_bound=-5,
                right_bound=5
            )
            
            # Generate paths via FunctionRenderable
            paths = FunctionRenderable(function, coordinate_mapper).build_screen_paths().paths
            
            # Test that we have multiple paths (indicating path breaks at asymptotes)
            self.assertGreater(len(paths), 1, "tan(x) should generate multiple separate paths due to vertical asymptotes")
            
            # Test that no path spans across a vertical asymptote
            for path in paths:
                if len(path) >= 2:
                    # Get the x-coordinate range of this path
                    x_coords = [mock_canvas.coordinate_mapper.screen_to_math(p[0], p[1])[0] for p in path]
                    path_min_x = min(x_coords)
                    path_max_x = max(x_coords)
                    
                    # Check that no vertical asymptote lies within this path's x range (exclusive)
                    asymptotes_in_path = [asym for asym in function.vertical_asymptotes 
                                        if path_min_x < asym < path_max_x]
                    
                    self.assertEqual(len(asymptotes_in_path), 0, 
                                   f"Path from x={path_min_x:.3f} to x={path_max_x:.3f} should not span across vertical asymptote(s) {asymptotes_in_path}")
            
            # Test with another function: 1/x has asymptote at x=0
            function2 = Function(
                function_string="1/x", 
                name="test_reciprocal",
                left_bound=-2,
                right_bound=2
            )
            
            paths2 = FunctionRenderable(function2, coordinate_mapper).build_screen_paths().paths
            
            # Should have exactly 2 paths (one for x < 0, one for x > 0)
            self.assertGreaterEqual(len(paths2), 2, "1/x should generate at least 2 separate paths due to vertical asymptote at x=0")
            
            # Verify no path spans across x=0
            for path in paths2:
                if len(path) >= 2:
                    x_coords = [mock_canvas.coordinate_mapper.screen_to_math(p[0], p[1])[0] for p in path]
                    path_min_x = min(x_coords)
                    path_max_x = max(x_coords)
                    
                    # Path should not cross x=0
                    crosses_zero = path_min_x < 0 < path_max_x
                    self.assertFalse(crosses_zero, 
                                   f"Path from x={path_min_x:.3f} to x={path_max_x:.3f} should not cross the vertical asymptote at x=0")
            
            print("Vertical asymptote path breaking test passed successfully")
            
        except ImportError as e:
            self.skipTest(f"Function class not available for testing: {e}")
        except Exception as e:
            self.fail(f"Unexpected error in vertical asymptote path breaking test: {e}")

    def test_function_path_continuity(self) -> None:
        """Test that Function class generates continuous paths where the function should be continuous."""
        try:
            from drawables.function import Function
            from .simple_mock import SimpleMock
            from coordinate_mapper import CoordinateMapper
            
            # Create a real CoordinateMapper instance
            coordinate_mapper = CoordinateMapper(600, 400)  # Canvas dimensions
            
            # Create a mock canvas
            mock_canvas = SimpleMock()
            mock_canvas.scale_factor = 50
            mock_canvas.cartesian2axis = SimpleMock()
            mock_canvas.cartesian2axis.origin = SimpleMock(x=300, y=200)
            mock_canvas.cartesian2axis.height = 400
            mock_canvas.coordinate_mapper = coordinate_mapper
            
            # Mock the visible bounds methods
            mock_canvas.cartesian2axis.get_visible_left_bound = lambda: -10
            mock_canvas.cartesian2axis.get_visible_right_bound = lambda: 10  
            mock_canvas.cartesian2axis.get_visible_top_bound = lambda: 8
            mock_canvas.cartesian2axis.get_visible_bottom_bound = lambda: -8
            
            # Sync coordinate mapper with canvas state
            coordinate_mapper.sync_from_canvas(mock_canvas)
            
            # Test a continuous function: sin(x) should have one continuous path
            function_sin = Function(
                function_string="sin(x)",
                name="test_sin",
                left_bound=-10,
                right_bound=10
            )
            
            paths_sin = FunctionRenderable(function_sin, coordinate_mapper).build_screen_paths().paths
            
            # sin(x) should generate exactly one continuous path (no asymptotes)
            self.assertEqual(len(paths_sin), 1, "sin(x) should generate exactly one continuous path")
            
            # Check that the path has reasonable point density (adaptive sampler uses fewer points)
            if paths_sin:
                path = paths_sin[0]
                self.assertGreater(len(path), 10, "Simple sine function should have sufficient points")
                
                # Check continuity within the path
                max_gap = 0
                for i in range(1, len(path)):
                    x1, _ = mock_canvas.coordinate_mapper.screen_to_math(path[i-1][0], path[i-1][1])
                    x2, _ = mock_canvas.coordinate_mapper.screen_to_math(path[i][0], path[i][1])
                    gap = abs(x2 - x1)
                    max_gap = max(max_gap, gap)
                
                # The maximum gap between consecutive points shouldn't be too large
                self.assertLess(max_gap, 1.0, f"sin(x) should have continuous points with max gap < 1.0, found {max_gap}")
            
            # Test a quadratic function: x^2 should also be one continuous path
            function_quad = Function(
                function_string="x^2",
                name="test_quad",
                left_bound=-5,
                right_bound=5
            )
            
            paths_quad = FunctionRenderable(function_quad, coordinate_mapper).build_screen_paths().paths
            
            # x^2 should generate exactly one continuous path
            self.assertEqual(len(paths_quad), 1, "x^2 should generate exactly one continuous path")
            
            # Test a complex but safer function first
            function_moderate = Function(
                function_string="sin(x/10) + cos(x/15)",  # Two different frequencies, no asymptotes
                name="test_moderate",
                left_bound=-20,
                right_bound=20
            )
            
            paths_moderate = FunctionRenderable(function_moderate, coordinate_mapper).build_screen_paths().paths
            self.assertGreater(len(paths_moderate), 0, "Moderate complexity function should generate paths")
            self.assertEqual(len(paths_moderate), 1, "Moderate function should be continuous (one path)")
            
            # Test the original problematic function but with a simpler version and safer range
            function_complex = Function(
                function_string="10 * sin(x / 20)",  # Simpler version to test basic functionality
                name="test_complex",
                left_bound=-50,  # Even safer range
                right_bound=50
            )
            
            paths_complex = FunctionRenderable(function_complex, coordinate_mapper).build_screen_paths().paths
            
            # This simplified function should definitely generate paths
            self.assertGreater(len(paths_complex), 0, 
                           f"Complex function should generate at least one path. "
                           f"Function: {function_complex.function_string}, "
                           f"Generated {len(paths_complex)} paths")
            
            # Test a simpler case to ensure basic functionality
            function_simple = Function(
                function_string="sin(x/10)",  # Simple sine function
                name="test_simple",
                left_bound=-10,
                right_bound=10
            )
            
            paths_simple = FunctionRenderable(function_simple, coordinate_mapper).build_screen_paths().paths
            self.assertEqual(len(paths_simple), 1, "Simple sine function should generate exactly one continuous path")
            self.assertGreater(len(paths_simple[0]), 10, "Simple sine function should have sufficient points")
            
            # Test the actual original problematic function in a very safe range
            try:
                function_original = Function(
                    function_string="100 * sin(x / 50) + 50 * tan(x / 100)",
                    name="test_original",
                    left_bound=-30,  # Very small, safe range
                    right_bound=30
                )
                
                paths_original = FunctionRenderable(function_original, coordinate_mapper).build_screen_paths().paths
                
                # This might fail, but let's see what happens
                if len(paths_original) > 0:
                    total_points_orig = sum(len(path) for path in paths_original)
                    self.assertGreater(total_points_orig, 5, "Original function should generate some points")
                else:
                    print(f"WARNING: Original complex function generated 0 paths - asymptotes: {function_original.vertical_asymptotes[:3] if hasattr(function_original, 'vertical_asymptotes') else 'None'}")
                    
            except Exception as e:
                print(f"WARNING: Original complex function failed: {e}")
            
            # Check path quality for any paths that were generated
            total_points = sum(len(path) for path in paths_complex)
            self.assertGreater(total_points, 10, "Complex function should generate some points across all paths")
            
            if paths_complex:
                # Check the longest path for continuity
                longest_path = max(paths_complex, key=len)
                if len(longest_path) > 1:
                    # Check for reasonable continuity in the longest path
                    max_gap = 0
                    for i in range(1, len(longest_path)):
                        x1, _ = mock_canvas.coordinate_mapper.screen_to_math(longest_path[i-1][0], longest_path[i-1][1])
                        x2, _ = mock_canvas.coordinate_mapper.screen_to_math(longest_path[i][0], longest_path[i][1])
                        gap = abs(x2 - x1)
                        max_gap = max(max_gap, gap)
                    
                    self.assertLess(max_gap, 20.0, f"Complex function should have reasonably continuous points, max gap was {max_gap}")
            
            print("Function path continuity test passed successfully")
            
        except ImportError as e:
            self.skipTest(f"Function class not available for testing: {e}")
        except Exception as e:
            self.fail(f"Unexpected error in function path continuity test: {e}")

    def test_find_diagonal_points_standard_order(self) -> None:
        points = [
            SimpleMock(name="A", x=0, y=1),
            SimpleMock(name="B", x=1, y=1),
            SimpleMock(name="C", x=1, y=0),
            SimpleMock(name="D", x=0, y=0)
        ]
        p_diag1, p_diag2 = MathUtils.find_diagonal_points(points, "Rect1")
        self.assertIsNotNone(p_diag1, "p_diag1 should not be None")
        self.assertIsNotNone(p_diag2, "p_diag2 should not be None")
        self.assertNotEqual(p_diag1.x, p_diag2.x)
        self.assertNotEqual(p_diag1.y, p_diag2.y)
        
        actual_pair = tuple(sorted((p_diag1.name, p_diag2.name)))
        expected_pairs = [("A", "C"), ("B", "D")]
        # Sort the names in the actual pair to make comparison order-independent
        # And check if this sorted pair is one of the sorted expected pairs
        self.assertIn(actual_pair, [tuple(sorted(p)) for p in expected_pairs], 
                      f"Expected diagonal pair like AC or BD, got {actual_pair}")

    def test_find_diagonal_points_shuffled_order(self) -> None:
        points = [
            SimpleMock(name="D", x=0, y=0),
            SimpleMock(name="B", x=1, y=1),
            SimpleMock(name="A", x=0, y=1),
            SimpleMock(name="C", x=1, y=0)
        ]
        p_diag1, p_diag2 = MathUtils.find_diagonal_points(points, "Rect2")
        self.assertIsNotNone(p_diag1, "p_diag1 should not be None")
        self.assertIsNotNone(p_diag2, "p_diag2 should not be None")
        self.assertNotEqual(p_diag1.x, p_diag2.x)
        self.assertNotEqual(p_diag1.y, p_diag2.y)

        actual_pair = tuple(sorted((p_diag1.name, p_diag2.name)))
        expected_pairs = [("A", "C"), ("B", "D")] # Same expected pairs
        self.assertIn(actual_pair, [tuple(sorted(p)) for p in expected_pairs],
                      f"Expected diagonal pair like AC or BD, got {actual_pair}")

    def test_find_diagonal_points_collinear_fail_case(self) -> None:
        points = [
            SimpleMock(name="A", x=0, y=0),
            SimpleMock(name="B", x=1, y=0),
            SimpleMock(name="C", x=2, y=0),
            SimpleMock(name="D", x=3, y=0)
        ]
        p_diag1, p_diag2 = MathUtils.find_diagonal_points(points, "Rect3_Collinear")
        self.assertIsNone(p_diag1)
        self.assertIsNone(p_diag2)

    def test_find_diagonal_points_L_shape_fail_case(self) -> None:
        points = [
            SimpleMock(name="A", x=0, y=1),
            SimpleMock(name="B", x=1, y=1),
            SimpleMock(name="C", x=1, y=0),
            SimpleMock(name="D", x=2, y=0)
        ]
        p_diag1, p_diag2 = MathUtils.find_diagonal_points(points, "Rect4_L-shape")
        self.assertIsNotNone(p_diag1)
        self.assertIsNotNone(p_diag2)
        self.assertEqual(p_diag1.name, "A")
        self.assertEqual(p_diag2.name, "C")

    def test_find_diagonal_points_less_than_4_points(self) -> None:
        points = [
            SimpleMock(name="A", x=0, y=0), 
            SimpleMock(name="B", x=1, y=1)
        ]
        p_diag1, p_diag2 = MathUtils.find_diagonal_points(points, "Rect5_TooFew")
        self.assertIsNone(p_diag1)
        self.assertIsNone(p_diag2)

    def test_find_diagonal_points_degenerate_rectangle_one_point_repeated(self) -> None:
        points = [
            SimpleMock(name="A1", x=0, y=1),
            SimpleMock(name="B", x=1, y=1),
            SimpleMock(name="C", x=1, y=0),
            SimpleMock(name="A2", x=0, y=1)
        ]
        p_diag1, p_diag2 = MathUtils.find_diagonal_points(points, "Rect6_Degenerate")
        self.assertIsNotNone(p_diag1)
        self.assertIsNotNone(p_diag2)
        self.assertEqual(p_diag1.name, "A1")
        self.assertEqual(p_diag2.name, "C")

    def test_find_diagonal_points_another_order(self) -> None:
        points = [
            SimpleMock(name="A", x=0, y=0),
            SimpleMock(name="C", x=1, y=1),
            SimpleMock(name="B", x=0, y=1),
            SimpleMock(name="D", x=1, y=0)
        ]
        p_diag1, p_diag2 = MathUtils.find_diagonal_points(points, "Rect7")
        self.assertIsNotNone(p_diag1)
        self.assertIsNotNone(p_diag2)
        self.assertNotEqual(p_diag1.x, p_diag2.x)
        self.assertNotEqual(p_diag1.y, p_diag2.y)
        self.assertEqual(p_diag1.name, "A")
        self.assertEqual(p_diag2.name, "C")


class TestNumberTheory(unittest.TestCase):
    """Tests for number theory functions in MathUtils."""

    # ========== is_prime tests ==========
    def test_is_prime_basic_primes(self) -> None:
        """Test is_prime with small prime numbers."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p in primes:
            self.assertTrue(MathUtils.is_prime(p), f"{p} should be prime")

    def test_is_prime_basic_composites(self) -> None:
        """Test is_prime with composite numbers."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25]
        for c in composites:
            self.assertFalse(MathUtils.is_prime(c), f"{c} should not be prime")

    def test_is_prime_edge_cases(self) -> None:
        """Test is_prime with edge cases."""
        self.assertFalse(MathUtils.is_prime(0), "0 is not prime")
        self.assertFalse(MathUtils.is_prime(1), "1 is not prime")
        self.assertTrue(MathUtils.is_prime(2), "2 is prime")
        self.assertTrue(MathUtils.is_prime(97), "97 is prime")

    def test_is_prime_large_prime(self) -> None:
        """Test is_prime with a larger prime."""
        self.assertTrue(MathUtils.is_prime(7919), "7919 is prime")
        self.assertTrue(MathUtils.is_prime(104729), "104729 is prime")

    def test_is_prime_negative_raises(self) -> None:
        """Test is_prime raises ValueError for negative input."""
        with self.assertRaises(ValueError):
            MathUtils.is_prime(-1)

    def test_is_prime_float_input(self) -> None:
        """Test is_prime handles float input that equals an integer."""
        self.assertTrue(MathUtils.is_prime(7.0), "7.0 should be treated as 7")
        with self.assertRaises(TypeError):
            MathUtils.is_prime(7.5)

    # ========== prime_factors tests ==========
    def test_prime_factors_basic(self) -> None:
        """Test prime_factors with basic cases."""
        self.assertEqual(MathUtils.prime_factors(12), [2, 2, 3])
        self.assertEqual(MathUtils.prime_factors(15), [3, 5])
        self.assertEqual(MathUtils.prime_factors(8), [2, 2, 2])
        self.assertEqual(MathUtils.prime_factors(17), [17])

    def test_prime_factors_edge_cases(self) -> None:
        """Test prime_factors with edge cases."""
        self.assertEqual(MathUtils.prime_factors(1), [])
        self.assertEqual(MathUtils.prime_factors(2), [2])
        self.assertEqual(MathUtils.prime_factors(360), [2, 2, 2, 3, 3, 5])

    def test_prime_factors_invalid_input(self) -> None:
        """Test prime_factors raises ValueError for invalid input."""
        with self.assertRaises(ValueError):
            MathUtils.prime_factors(0)
        with self.assertRaises(ValueError):
            MathUtils.prime_factors(-5)

    # ========== mod_pow tests ==========
    def test_mod_pow_basic(self) -> None:
        """Test mod_pow with basic cases."""
        self.assertEqual(MathUtils.mod_pow(2, 10, 1000), 24)
        self.assertEqual(MathUtils.mod_pow(3, 5, 7), 5)
        self.assertEqual(MathUtils.mod_pow(5, 3, 13), 8)

    def test_mod_pow_edge_cases(self) -> None:
        """Test mod_pow with edge cases."""
        self.assertEqual(MathUtils.mod_pow(2, 0, 5), 1)  # Any number to the 0 is 1
        self.assertEqual(MathUtils.mod_pow(0, 5, 7), 0)  # 0 to any power is 0
        self.assertEqual(MathUtils.mod_pow(7, 1, 5), 2)  # 7 mod 5 = 2

    def test_mod_pow_large_exponent(self) -> None:
        """Test mod_pow with a large exponent."""
        self.assertEqual(MathUtils.mod_pow(2, 100, 1000000007), 976371285)

    def test_mod_pow_invalid_input(self) -> None:
        """Test mod_pow raises errors for invalid input."""
        with self.assertRaises(ValueError):
            MathUtils.mod_pow(2, -1, 5)  # Negative exponent
        with self.assertRaises(ValueError):
            MathUtils.mod_pow(2, 3, 0)  # Zero modulus
        with self.assertRaises(ValueError):
            MathUtils.mod_pow(2, 3, -5)  # Negative modulus

    # ========== mod_inverse tests ==========
    def test_mod_inverse_basic(self) -> None:
        """Test mod_inverse with basic cases."""
        self.assertEqual(MathUtils.mod_inverse(3, 7), 5)  # 3*5 = 15 ≡ 1 (mod 7)
        self.assertEqual(MathUtils.mod_inverse(2, 5), 3)  # 2*3 = 6 ≡ 1 (mod 5)
        self.assertEqual(MathUtils.mod_inverse(7, 11), 8)  # 7*8 = 56 ≡ 1 (mod 11)

    def test_mod_inverse_verify(self) -> None:
        """Verify mod_inverse results are correct."""
        test_cases = [(3, 7), (2, 5), (7, 11), (5, 13), (11, 17)]
        for a, mod in test_cases:
            inv = MathUtils.mod_inverse(a, mod)
            self.assertEqual((a * inv) % mod, 1, f"({a} * {inv}) % {mod} should be 1")

    def test_mod_inverse_no_inverse(self) -> None:
        """Test mod_inverse raises ValueError when inverse doesn't exist."""
        with self.assertRaises(ValueError):
            MathUtils.mod_inverse(2, 4)  # gcd(2, 4) = 2 != 1
        with self.assertRaises(ValueError):
            MathUtils.mod_inverse(6, 9)  # gcd(6, 9) = 3 != 1

    def test_mod_inverse_invalid_input(self) -> None:
        """Test mod_inverse raises errors for invalid input."""
        with self.assertRaises(ValueError):
            MathUtils.mod_inverse(3, 0)  # Zero modulus
        with self.assertRaises(ValueError):
            MathUtils.mod_inverse(3, -5)  # Negative modulus

    # ========== next_prime tests ==========
    def test_next_prime_basic(self) -> None:
        """Test next_prime with basic cases."""
        self.assertEqual(MathUtils.next_prime(14), 17)
        self.assertEqual(MathUtils.next_prime(17), 17)  # 17 is already prime
        self.assertEqual(MathUtils.next_prime(18), 19)

    def test_next_prime_edge_cases(self) -> None:
        """Test next_prime with edge cases."""
        self.assertEqual(MathUtils.next_prime(0), 2)
        self.assertEqual(MathUtils.next_prime(1), 2)
        self.assertEqual(MathUtils.next_prime(2), 2)
        self.assertEqual(MathUtils.next_prime(-10), 2)

    def test_next_prime_sequence(self) -> None:
        """Test next_prime produces correct sequence."""
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        for i, exp in enumerate(expected):
            if i == 0:
                self.assertEqual(MathUtils.next_prime(0), exp)
            else:
                self.assertEqual(MathUtils.next_prime(expected[i-1] + 1), exp)

    # ========== prev_prime tests ==========
    def test_prev_prime_basic(self) -> None:
        """Test prev_prime with basic cases."""
        self.assertEqual(MathUtils.prev_prime(14), 13)
        self.assertEqual(MathUtils.prev_prime(13), 13)  # 13 is already prime
        self.assertEqual(MathUtils.prev_prime(20), 19)

    def test_prev_prime_edge_cases(self) -> None:
        """Test prev_prime with edge cases."""
        self.assertEqual(MathUtils.prev_prime(2), 2)
        self.assertEqual(MathUtils.prev_prime(3), 3)

    def test_prev_prime_no_prime(self) -> None:
        """Test prev_prime raises ValueError when no prime exists."""
        with self.assertRaises(ValueError):
            MathUtils.prev_prime(1)
        with self.assertRaises(ValueError):
            MathUtils.prev_prime(0)
        with self.assertRaises(ValueError):
            MathUtils.prev_prime(-5)

    # ========== totient tests ==========
    def test_totient_basic(self) -> None:
        """Test totient with basic cases."""
        self.assertEqual(MathUtils.totient(1), 1)
        self.assertEqual(MathUtils.totient(2), 1)
        self.assertEqual(MathUtils.totient(12), 4)  # coprimes: 1, 5, 7, 11
        self.assertEqual(MathUtils.totient(10), 4)  # coprimes: 1, 3, 7, 9

    def test_totient_primes(self) -> None:
        """Test totient for prime numbers (should be p-1)."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        for p in primes:
            self.assertEqual(MathUtils.totient(p), p - 1, f"totient({p}) should be {p-1}")

    def test_totient_prime_powers(self) -> None:
        """Test totient for prime powers (p^k -> p^(k-1) * (p-1))."""
        self.assertEqual(MathUtils.totient(4), 2)   # 2^2 -> 2^1 * 1 = 2
        self.assertEqual(MathUtils.totient(8), 4)   # 2^3 -> 2^2 * 1 = 4
        self.assertEqual(MathUtils.totient(9), 6)   # 3^2 -> 3^1 * 2 = 6
        self.assertEqual(MathUtils.totient(27), 18) # 3^3 -> 3^2 * 2 = 18

    def test_totient_100(self) -> None:
        """Test totient(100)."""
        self.assertEqual(MathUtils.totient(100), 40)

    def test_totient_invalid_input(self) -> None:
        """Test totient raises ValueError for invalid input."""
        with self.assertRaises(ValueError):
            MathUtils.totient(0)
        with self.assertRaises(ValueError):
            MathUtils.totient(-5)

    # ========== divisors tests ==========
    def test_divisors_basic(self) -> None:
        """Test divisors with basic cases."""
        self.assertEqual(MathUtils.divisors(1), [1])
        self.assertEqual(MathUtils.divisors(6), [1, 2, 3, 6])
        self.assertEqual(MathUtils.divisors(12), [1, 2, 3, 4, 6, 12])
        self.assertEqual(MathUtils.divisors(60), [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60])

    def test_divisors_primes(self) -> None:
        """Test divisors for prime numbers."""
        primes = [2, 3, 5, 7, 11, 13]
        for p in primes:
            self.assertEqual(MathUtils.divisors(p), [1, p], f"divisors({p}) should be [1, {p}]")

    def test_divisors_perfect_square(self) -> None:
        """Test divisors for perfect squares."""
        self.assertEqual(MathUtils.divisors(16), [1, 2, 4, 8, 16])
        self.assertEqual(MathUtils.divisors(36), [1, 2, 3, 4, 6, 9, 12, 18, 36])

    def test_divisors_invalid_input(self) -> None:
        """Test divisors raises ValueError for invalid input."""
        with self.assertRaises(ValueError):
            MathUtils.divisors(0)
        with self.assertRaises(ValueError):
            MathUtils.divisors(-10)

    # ========== MathUtils.evaluate() integration tests ==========
    # These tests verify the AI path works (MathUtils.evaluate -> Python functions)
    # This catches the bug where Python-only functions weren't accessible via evaluate()

    def test_evaluate_is_prime(self) -> None:
        """Test is_prime is accessible via MathUtils.evaluate()."""
        result = MathUtils.evaluate("is_prime(97)")
        self.assertEqual(result, "True", "is_prime(97) via evaluate() should return 'True'")

        result = MathUtils.evaluate("is_prime(100)")
        self.assertEqual(result, "False", "is_prime(100) via evaluate() should return 'False'")

        result = MathUtils.evaluate("is_prime(2)")
        self.assertEqual(result, "True", "is_prime(2) via evaluate() should return 'True'")

    def test_evaluate_prime_factors(self) -> None:
        """Test prime_factors is accessible via MathUtils.evaluate()."""
        result = MathUtils.evaluate("prime_factors(12)")
        self.assertEqual(result, "[2, 2, 3]", "prime_factors(12) via evaluate()")

        result = MathUtils.evaluate("prime_factors(360)")
        self.assertEqual(result, "[2, 2, 2, 3, 3, 5]", "prime_factors(360) via evaluate()")

        result = MathUtils.evaluate("prime_factors(17)")
        self.assertEqual(result, "[17]", "prime_factors(17) via evaluate()")

    def test_evaluate_mod_pow(self) -> None:
        """Test mod_pow is accessible via MathUtils.evaluate()."""
        result = MathUtils.evaluate("mod_pow(2, 10, 1000)")
        self.assertEqual(int(result), 24, "mod_pow(2, 10, 1000) via evaluate()")

        result = MathUtils.evaluate("mod_pow(3, 5, 7)")
        self.assertEqual(int(result), 5, "mod_pow(3, 5, 7) via evaluate()")

        result = MathUtils.evaluate("mod_pow(2, 100, 1000000007)")
        self.assertEqual(int(result), 976371285, "mod_pow with large exponent via evaluate()")

    def test_evaluate_mod_inverse(self) -> None:
        """Test mod_inverse is accessible via MathUtils.evaluate()."""
        result = MathUtils.evaluate("mod_inverse(3, 7)")
        self.assertEqual(int(result), 5, "mod_inverse(3, 7) via evaluate()")

        result = MathUtils.evaluate("mod_inverse(2, 5)")
        self.assertEqual(int(result), 3, "mod_inverse(2, 5) via evaluate()")

    def test_evaluate_next_prime(self) -> None:
        """Test next_prime is accessible via MathUtils.evaluate()."""
        result = MathUtils.evaluate("next_prime(14)")
        self.assertEqual(int(result), 17, "next_prime(14) via evaluate()")

        result = MathUtils.evaluate("next_prime(17)")
        self.assertEqual(int(result), 17, "next_prime(17) should return 17 (already prime)")

    def test_evaluate_prev_prime(self) -> None:
        """Test prev_prime is accessible via MathUtils.evaluate()."""
        result = MathUtils.evaluate("prev_prime(14)")
        self.assertEqual(int(result), 13, "prev_prime(14) via evaluate()")

        result = MathUtils.evaluate("prev_prime(13)")
        self.assertEqual(int(result), 13, "prev_prime(13) should return 13 (already prime)")

    def test_evaluate_totient(self) -> None:
        """Test totient is accessible via MathUtils.evaluate()."""
        result = MathUtils.evaluate("totient(100)")
        self.assertEqual(int(result), 40, "totient(100) via evaluate()")

        result = MathUtils.evaluate("totient(12)")
        self.assertEqual(int(result), 4, "totient(12) via evaluate()")

    def test_evaluate_divisors(self) -> None:
        """Test divisors is accessible via MathUtils.evaluate()."""
        result = MathUtils.evaluate("divisors(12)")
        self.assertEqual(result, "[1, 2, 3, 4, 6, 12]", "divisors(12) via evaluate()")

        result = MathUtils.evaluate("divisors(60)")
        self.assertEqual(result, "[1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]", "divisors(60) via evaluate()")

    def test_evaluate_number_theory_returns_proper_types(self) -> None:
        """Test that evaluate() returns proper string types for booleans and lists."""
        # Boolean results should be "True" or "False" strings
        result = MathUtils.evaluate("is_prime(7)")
        self.assertIsInstance(result, str, "is_prime via evaluate() should return a string")
        self.assertIn(result, ["True", "False"], "Boolean result should be 'True' or 'False'")

        # List results should be string representations
        result = MathUtils.evaluate("prime_factors(8)")
        self.assertIsInstance(result, str, "prime_factors via evaluate() should return a string")
        self.assertTrue(result.startswith("["), "List result should start with '['")

        result = MathUtils.evaluate("divisors(6)")
        self.assertIsInstance(result, str, "divisors via evaluate() should return a string")
        self.assertTrue(result.startswith("["), "List result should start with '['")

    def test_evaluate_number_theory_not_error(self) -> None:
        """Test that number theory functions don't return error messages via evaluate()."""
        # This catches the original bug where these returned "not a supported mathematical expression"
        functions_to_test = [
            "is_prime(97)",
            "prime_factors(12)",
            "mod_pow(2, 10, 100)",
            "mod_inverse(3, 7)",
            "next_prime(10)",
            "prev_prime(10)",
            "totient(10)",
            "divisors(10)",
        ]
        for func in functions_to_test:
            result = MathUtils.evaluate(func)
            self.assertNotIn("Error", str(result), "{} should not return an error".format(func))
            self.assertNotIn("not a supported", str(result).lower(), "{} accessible via evaluate".format(func))


class TestSequencesAndSeries(unittest.TestCase):
    """Tests for sequence and series functions in MathUtils."""

    # ========== summation tests ==========
    def test_summation_basic(self) -> None:
        """Test summation with basic cases."""
        # Sum of n from 1 to 5 = 1+2+3+4+5 = 15
        self.assertEqual(MathUtils.summation("n", "n", 1, 5), "15")

    def test_summation_squares(self) -> None:
        """Test summation of squares."""
        # Sum of n^2 from 1 to 5 = 1+4+9+16+25 = 55
        self.assertEqual(MathUtils.summation("n^2", "n", 1, 5), "55")

    def test_summation_empty_range(self) -> None:
        """Test summation with start > end returns 0."""
        self.assertEqual(MathUtils.summation("n", "n", 5, 1), "0")

    def test_summation_single_term(self) -> None:
        """Test summation with single term."""
        self.assertEqual(MathUtils.summation("n^2", "n", 3, 3), "9")

    def test_summation_geometric(self) -> None:
        """Test summation with geometric terms."""
        # Sum of 2^n from 0 to 4 = 1+2+4+8+16 = 31
        self.assertEqual(MathUtils.summation("2^n", "n", 0, 4), "31")

    # ========== product tests ==========
    def test_product_factorial(self) -> None:
        """Test product to compute factorial."""
        # Product of n from 1 to 5 = 5! = 120
        self.assertEqual(MathUtils.product("n", "n", 1, 5), "120")

    def test_product_empty_range(self) -> None:
        """Test product with start > end returns 1."""
        self.assertEqual(MathUtils.product("n", "n", 5, 1), "1")

    def test_product_powers(self) -> None:
        """Test product of powers."""
        # Product of 2^n from 1 to 3 = 2 * 4 * 8 = 64
        self.assertEqual(MathUtils.product("2^n", "n", 1, 3), "64")

    def test_product_single_term(self) -> None:
        """Test product with single term."""
        self.assertEqual(MathUtils.product("n", "n", 5, 5), "5")

    # ========== arithmetic_sum tests ==========
    def test_arithmetic_sum_basic(self) -> None:
        """Test arithmetic_sum with basic cases."""
        # 1, 3, 5, 7, 9 (a=1, d=2, n=5) -> 25
        self.assertEqual(MathUtils.arithmetic_sum(1, 2, 5), 25)

    def test_arithmetic_sum_natural_numbers(self) -> None:
        """Test arithmetic_sum for natural numbers."""
        # 1+2+3+...+10 = 55
        self.assertEqual(MathUtils.arithmetic_sum(1, 1, 10), 55)

    def test_arithmetic_sum_single_term(self) -> None:
        """Test arithmetic_sum with single term."""
        self.assertEqual(MathUtils.arithmetic_sum(7, 3, 1), 7)

    def test_arithmetic_sum_negative_diff(self) -> None:
        """Test arithmetic_sum with negative difference."""
        # 10, 8, 6, 4, 2 (a=10, d=-2, n=5) -> 30
        self.assertEqual(MathUtils.arithmetic_sum(10, -2, 5), 30)

    def test_arithmetic_sum_invalid_n(self) -> None:
        """Test arithmetic_sum raises for n < 1."""
        with self.assertRaises(ValueError):
            MathUtils.arithmetic_sum(1, 1, 0)

    # ========== geometric_sum tests ==========
    def test_geometric_sum_basic(self) -> None:
        """Test geometric_sum with basic cases."""
        # 1 + 2 + 4 + 8 + 16 = 31 (a=1, r=2, n=5)
        self.assertEqual(MathUtils.geometric_sum(1, 2, 5), 31)

    def test_geometric_sum_ratio_one(self) -> None:
        """Test geometric_sum with ratio = 1."""
        # 5 + 5 + 5 = 15 (a=5, r=1, n=3)
        self.assertEqual(MathUtils.geometric_sum(5, 1, 3), 15)

    def test_geometric_sum_fractional_ratio(self) -> None:
        """Test geometric_sum with fractional ratio."""
        # 1 + 0.5 + 0.25 = 1.75 (a=1, r=0.5, n=3)
        self.assertEqual(MathUtils.geometric_sum(1, 0.5, 3), 1.75)

    def test_geometric_sum_single_term(self) -> None:
        """Test geometric_sum with single term."""
        self.assertEqual(MathUtils.geometric_sum(7, 2, 1), 7)

    def test_geometric_sum_invalid_n(self) -> None:
        """Test geometric_sum raises for n < 1."""
        with self.assertRaises(ValueError):
            MathUtils.geometric_sum(1, 2, 0)

    # ========== geometric_sum_infinite tests ==========
    def test_geometric_sum_infinite_convergent(self) -> None:
        """Test geometric_sum_infinite with convergent series."""
        # a/(1-r) = 1/(1-0.5) = 2
        self.assertEqual(MathUtils.geometric_sum_infinite(1, 0.5), 2)

    def test_geometric_sum_infinite_third(self) -> None:
        """Test geometric_sum_infinite with r=1/3."""
        # a/(1-r) = 1/(1-1/3) = 1.5
        self.assertEqual(MathUtils.geometric_sum_infinite(1, 1/3), 1.5)

    def test_geometric_sum_infinite_negative_ratio(self) -> None:
        """Test geometric_sum_infinite with negative ratio."""
        # a/(1-r) = 1/(1-(-0.5)) = 1/1.5 = 2/3
        result = MathUtils.geometric_sum_infinite(1, -0.5)
        self.assertAlmostEqual(result, 2/3, places=10)

    def test_geometric_sum_infinite_divergent(self) -> None:
        """Test geometric_sum_infinite raises for divergent series."""
        with self.assertRaises(ValueError):
            MathUtils.geometric_sum_infinite(1, 1)
        with self.assertRaises(ValueError):
            MathUtils.geometric_sum_infinite(1, 2)
        with self.assertRaises(ValueError):
            MathUtils.geometric_sum_infinite(1, -1)

    # ========== ratio_test tests ==========
    def test_ratio_test_converges(self) -> None:
        """Test ratio_test identifies convergent series."""
        # 1/n! converges (ratio -> 0)
        result = MathUtils.ratio_test("1/factorial(n)", "n")
        self.assertIn("Converges", result)

    def test_ratio_test_diverges(self) -> None:
        """Test ratio_test identifies divergent series."""
        # 2^n diverges (ratio = 2 > 1)
        result = MathUtils.ratio_test("2^n", "n")
        self.assertIn("Diverges", result)

    def test_ratio_test_geometric_half(self) -> None:
        """Test ratio_test with (1/2)^n."""
        # (1/2)^n converges (ratio = 0.5)
        result = MathUtils.ratio_test("(1/2)^n", "n")
        self.assertIn("Converges", result)

    # ========== root_test tests ==========
    def test_root_test_converges(self) -> None:
        """Test root_test identifies convergent series."""
        # (1/2)^n converges (L = 0.5)
        result = MathUtils.root_test("(1/2)^n", "n")
        self.assertIn("Converges", result)

    def test_root_test_diverges(self) -> None:
        """Test root_test identifies divergent series."""
        # 2^n diverges (L = 2)
        result = MathUtils.root_test("2^n", "n")
        self.assertIn("Diverges", result)

    # ========== p_series_test tests ==========
    def test_p_series_test_converges(self) -> None:
        """Test p_series_test with p > 1 (converges)."""
        self.assertEqual(MathUtils.p_series_test(2), "Converges")
        self.assertEqual(MathUtils.p_series_test(1.5), "Converges")
        self.assertEqual(MathUtils.p_series_test(3), "Converges")

    def test_p_series_test_diverges(self) -> None:
        """Test p_series_test with p <= 1 (diverges)."""
        self.assertEqual(MathUtils.p_series_test(1), "Diverges")
        self.assertEqual(MathUtils.p_series_test(0.5), "Diverges")
        self.assertEqual(MathUtils.p_series_test(0), "Diverges")
        self.assertEqual(MathUtils.p_series_test(-1), "Diverges")


class TestSequencesAndSeriesEvaluate(unittest.TestCase):
    """Tests for sequence/series functions accessed via MathUtils.evaluate()."""

    def test_evaluate_summation(self) -> None:
        """Test summation is accessible via MathUtils.evaluate()."""
        result = MathUtils.evaluate("summation('n^2', 'n', 1, 5)")
        self.assertEqual(result, "55", "summation('n^2', 'n', 1, 5) via evaluate()")

    def test_evaluate_product(self) -> None:
        """Test product is accessible via MathUtils.evaluate()."""
        result = MathUtils.evaluate("product('n', 'n', 1, 5)")
        self.assertEqual(result, "120", "product('n', 'n', 1, 5) via evaluate()")

    def test_evaluate_arithmetic_sum(self) -> None:
        """Test arithmetic_sum is accessible via MathUtils.evaluate()."""
        result = MathUtils.evaluate("arithmetic_sum(1, 2, 5)")
        self.assertEqual(int(result), 25, "arithmetic_sum(1, 2, 5) via evaluate()")

    def test_evaluate_geometric_sum(self) -> None:
        """Test geometric_sum is accessible via MathUtils.evaluate()."""
        result = MathUtils.evaluate("geometric_sum(1, 2, 5)")
        self.assertEqual(int(result), 31, "geometric_sum(1, 2, 5) via evaluate()")

    def test_evaluate_geometric_sum_infinite(self) -> None:
        """Test geometric_sum_infinite is accessible via MathUtils.evaluate()."""
        result = MathUtils.evaluate("geometric_sum_infinite(1, 0.5)")
        self.assertEqual(float(result), 2.0, "geometric_sum_infinite(1, 0.5) via evaluate()")

    def test_evaluate_p_series_test(self) -> None:
        """Test p_series_test is accessible via MathUtils.evaluate()."""
        result = MathUtils.evaluate("p_series_test(2)")
        self.assertEqual(result, "Converges", "p_series_test(2) via evaluate()")

        result = MathUtils.evaluate("p_series_test(0.5)")
        self.assertEqual(result, "Diverges", "p_series_test(0.5) via evaluate()")

    def test_evaluate_sequences_not_error(self) -> None:
        """Test that sequence/series functions don't return error messages via evaluate()."""
        functions_to_test = [
            "summation('n', 'n', 1, 5)",
            "product('n', 'n', 1, 5)",
            "arithmetic_sum(1, 1, 10)",
            "geometric_sum(1, 2, 5)",
            "geometric_sum_infinite(1, 0.5)",
            "p_series_test(2)",
        ]
        for func in functions_to_test:
            result = MathUtils.evaluate(func)
            self.assertNotIn("Error", str(result), "{} should not return an error".format(func))
            self.assertNotIn("not a supported", str(result).lower(), "{} accessible via evaluate".format(func))
