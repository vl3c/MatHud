from __future__ import annotations

import unittest

from canvas import Canvas
from managers.polygon_type import PolygonType
from utils.polygon_canonicalizer import canonicalize_rectangle
from drawables.drawable import Drawable
from geometry import Position
from client_tests.simple_mock import SimpleMock


class TestCustomDrawableNames(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.mock_cartesian2axis = SimpleMock(draw=SimpleMock(return_value=None), reset=SimpleMock(return_value=None),
                                              get_state=SimpleMock(return_value={'Cartesian_System_Visibility': 'cartesian_state'}),
                                              origin=Position(0, 0))
        self.canvas.cartesian2axis = self.mock_cartesian2axis

    def tearDown(self) -> None:
        # Clear the canvas
        self.canvas.clear()

    def test_point_basic_naming(self) -> None:
        # Test creating a point with a custom name
        point = self.canvas.create_point(10, 10, "Custom")
        self.assertEqual(point.name, "C")
        # Test creating another point with the same name - should use next available letter
        point2 = self.canvas.create_point(20, 20, "Custom")
        self.assertEqual(point2.name, "U")
        # Test with a name that has no available letters - should fall back to default naming
        point3 = self.canvas.create_point(30, 30, "CU")
        self.assertEqual(point3.name, "C'")

    def test_point_apostrophe_naming(self) -> None:
        # Test point with apostrophes in name
        point = self.canvas.create_point(40, 40, "TR'IA'NG'LE")
        self.assertEqual(point.name, "T")
        point2 = self.canvas.create_point(50, 50, "TR'IA'NG'LE")
        self.assertEqual(point2.name, "R'")

        # Test point with multiple consecutive apostrophes
        point3 = self.canvas.create_point(60, 60, "A''B'''C")
        self.assertEqual(point3.name, "A''")
        point4 = self.canvas.create_point(70, 70, "A''B'''C")
        self.assertEqual(point4.name, "B'''")

    def test_segment_basic_naming(self) -> None:
        # Test creating a segment with a custom name
        segment = self.canvas.create_segment(10, 10, 20, 20, "Segment")
        self.assertEqual(segment.point1.name, "S")
        self.assertEqual(segment.point2.name, "E")
        # Test creating another segment with same name - should use next available letters
        segment2 = self.canvas.create_segment(30, 30, 40, 40, "Segment")
        self.assertEqual(segment2.point1.name, "G")
        self.assertEqual(segment2.point2.name, "M")

    def test_segment_apostrophe_naming(self) -> None:
        # Test segment with apostrophes in name
        segment = self.canvas.create_segment(50, 50, 60, 60, "A'B''")
        self.assertEqual(segment.point1.name, "A'")
        self.assertEqual(segment.point2.name, "B''")
        # Test segment with mixed apostrophes
        segment2 = self.canvas.create_segment(70, 70, 80, 80, "X'Y")
        self.assertEqual(segment2.point1.name, "X'")
        self.assertEqual(segment2.point2.name, "Y")

    def test_triangle_basic_naming(self) -> None:
        # Test creating a triangle with a custom name
        triangle = self.canvas.create_polygon(
            [(10, 10), (20, 20), (30, 30)],
            polygon_type=PolygonType.TRIANGLE,
            name="Triangle",
        )
        self.assertEqual(triangle.segment1.point1.name, "T")
        self.assertEqual(triangle.segment1.point2.name, "R")
        self.assertEqual(triangle.segment2.point2.name, "I")
        # Test creating another triangle with same name - should use next available letters
        triangle2 = self.canvas.create_polygon(
            [(40, 40), (50, 50), (60, 60)],
            polygon_type=PolygonType.TRIANGLE,
            name="Triangle",
        )
        self.assertEqual(triangle2.segment1.point1.name, "A")
        self.assertEqual(triangle2.segment1.point2.name, "N")
        self.assertEqual(triangle2.segment2.point2.name, "G")

    def test_triangle_apostrophe_naming(self) -> None:
        # Test triangle with apostrophes in name
        triangle = self.canvas.create_polygon(
            [(70, 70), (80, 80), (90, 90)],
            polygon_type=PolygonType.TRIANGLE,
            name="A'B''C'''",
        )
        self.assertEqual(triangle.segment1.point1.name, "A'")
        self.assertEqual(triangle.segment1.point2.name, "B''")
        self.assertEqual(triangle.segment2.point2.name, "C'''")

    def test_rectangle_basic_naming(self) -> None:
        # Test creating a rectangle with a custom name
        rectangle = self.canvas.create_polygon(
            canonicalize_rectangle([(10, 10), (30, 30)], construction_mode="diagonal"),
            polygon_type=PolygonType.RECTANGLE,
            name="Rectangle",
        )
        points = [
            rectangle.segment1.point1.name,
            rectangle.segment1.point2.name,
            rectangle.segment2.point2.name,
            rectangle.segment3.point2.name
        ]
        # Check that the points use the first four letters of "Rectangle"
        self.assertIn("R", points)
        self.assertIn("E", points)
        self.assertIn("C", points)
        self.assertIn("T", points)
        # Test creating another rectangle with same name - should use next available letters
        rectangle2 = self.canvas.create_polygon(
            canonicalize_rectangle([(40, 40), (60, 60)], construction_mode="diagonal"),
            polygon_type=PolygonType.RECTANGLE,
            name="Rectangle",
        )
        points2 = [
            rectangle2.segment1.point1.name,
            rectangle2.segment1.point2.name,
            rectangle2.segment2.point2.name,
            rectangle2.segment3.point2.name
        ]
        # Check that the points use the next available letters
        self.assertIn("A", points2)
        self.assertIn("N", points2)
        self.assertIn("G", points2)
        self.assertIn("L", points2)

    def test_rectangle_apostrophe_naming(self) -> None:
        # Test rectangle with apostrophes in name
        rectangle = self.canvas.create_polygon(
            canonicalize_rectangle([(70, 70), (90, 90)], construction_mode="diagonal"),
            polygon_type=PolygonType.RECTANGLE,
            name="W'X''Y'''Z",
        )
        points = [
            rectangle.segment1.point1.name,
            rectangle.segment1.point2.name,
            rectangle.segment2.point2.name,
            rectangle.segment3.point2.name
        ]
        # Check that the points use the letters with their apostrophes
        self.assertIn("W'", points)
        self.assertIn("X''", points)
        self.assertIn("Y'''", points)
        self.assertIn("Z", points)

    def test_circle_basic_naming(self) -> None:
        # Test creating a circle with a custom name
        circle = self.canvas.create_circle(10, 10, 5, "Circle")
        self.assertEqual(circle.center.name, "C")
        # Test creating another circle with same name - should use next available letter
        circle2 = self.canvas.create_circle(20, 20, 5, "Circle")
        self.assertEqual(circle2.center.name, "I")

    def test_circle_apostrophe_naming(self) -> None:
        # Test circle with apostrophes in name
        circle = self.canvas.create_circle(30, 30, 5, "A'B''")
        self.assertEqual(circle.center.name, "A'")
        # Test circle with multiple consecutive apostrophes
        circle2 = self.canvas.create_circle(40, 40, 5, "X''Y'''Z")
        self.assertEqual(circle2.center.name, "X''")

    def test_ellipse_basic_naming(self) -> None:
        # Test creating an ellipse with a custom name
        ellipse = self.canvas.create_ellipse(10, 10, 5, 3, name="Ellipse")
        self.assertEqual(ellipse.center.name, "E")
        # Test creating another ellipse with same name - should use next available letter
        ellipse2 = self.canvas.create_ellipse(20, 20, 5, 3, name="Ellipse")
        self.assertEqual(ellipse2.center.name, "L")

    def test_ellipse_apostrophe_naming(self) -> None:
        # Test ellipse with apostrophes in name
        ellipse = self.canvas.create_ellipse(30, 30, 5, 3, name="A'B''")
        self.assertEqual(ellipse.center.name, "A'")
        # Test ellipse with multiple consecutive apostrophes
        ellipse2 = self.canvas.create_ellipse(40, 40, 5, 3, name="X''Y'''Z")
        self.assertEqual(ellipse2.center.name, "X''")

    def test_vector_basic_naming(self) -> None:
        # Test creating a vector with a custom name
        vector = self.canvas.create_vector(10, 10, 20, 20, "Vector")
        self.assertEqual(vector.segment.point1.name, "V")  # Origin point
        self.assertEqual(vector.segment.point2.name, "E")  # Tip point
        # Test creating another vector with same name - should use next available letters
        vector2 = self.canvas.create_vector(30, 30, 40, 40, "Vector")
        self.assertEqual(vector2.segment.point1.name, "C")  # Origin point
        self.assertEqual(vector2.segment.point2.name, "T")  # Tip point

    def test_vector_apostrophe_naming(self) -> None:
        # Test vector with apostrophes in name
        vector = self.canvas.create_vector(50, 50, 60, 60, "A'B''")
        self.assertEqual(vector.segment.point1.name, "A'")  # Origin point
        self.assertEqual(vector.segment.point2.name, "B''")  # Tip point
        # Test vector with mixed apostrophes
        vector2 = self.canvas.create_vector(70, 70, 80, 80, "X'Y")
        self.assertEqual(vector2.segment.point1.name, "X'")  # Origin point
        self.assertEqual(vector2.segment.point2.name, "Y")  # Tip point

    def test_name_fallback_sequence(self) -> None:
        # Create 26 points to use up all letters
        for i in range(26):
            point = self.canvas.create_point(i*10, i*10)
            expected_letter = chr(ord('A') + i)  # A, B, C, ...
            self.assertEqual(point.name, expected_letter)
        # Check we have all letters A-Z
        expected_names = [letter for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        actual_names = sorted(self.canvas.name_generator.get_drawable_names('Point'))
        self.assertEqual(actual_names, expected_names)

        # Try to create a point with custom name - should use first letter with apostrophe
        point = self.canvas.create_point(300, 300, "Custom")
        self.assertEqual(point.name, "C'")
        expected_names.append("C'")
        expected_names.sort()
        actual_names = sorted(self.canvas.name_generator.get_drawable_names('Point'))
        self.assertEqual(actual_names, expected_names)

        # Create 25 more points without names - should get A'-Z' (except C' which is already used)
        for i in range(25):  # 25 because C' is already used
            point = self.canvas.create_point(i*10 + 400, i*10 + 400)

        # Check we have all letters A'-Z'
        expected_names = expected_names + [letter + "'" for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if letter != 'C']
        expected_names.sort()
        actual_names = sorted(self.canvas.name_generator.get_drawable_names('Point'))
        self.assertEqual(actual_names, expected_names)

        # Try to create another point - should use first letter with two apostrophes
        point2 = self.canvas.create_point(310, 310)  # No custom name
        self.assertEqual(point2.name, "A''")
        expected_names.append("A''")
        self.assertEqual(sorted(self.canvas.name_generator.get_drawable_names('Point')), sorted(expected_names)) 