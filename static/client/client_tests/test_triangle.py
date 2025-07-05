import unittest
import copy
from geometry import Point, Position, Segment, Triangle
from .simple_mock import SimpleMock


class TestTriangle(unittest.TestCase):
    def setUp(self):
        self.canvas = SimpleMock(scale_factor=1, cartesian2axis=SimpleMock(origin=Position(0, 0)),
                                 is_point_within_canvas_visible_area=SimpleMock(return_value=True),
                                 any_segment_part_visible_in_canvas_area=SimpleMock(return_value=True))
        # Setup points for the triangle
        self.p1 = Point(0, 0, self.canvas, name="P1", color="red")
        self.p2 = Point(4, 0, self.canvas, name="P2", color="green")
        self.p3 = Point(0, 3, self.canvas, name="P3", color="blue")
        # Setup segments for the triangle
        self.segment1 = Segment(self.p1, self.p2, self.canvas, "red")
        self.segment2 = Segment(self.p2, self.p3, self.canvas, "green")
        self.segment3 = Segment(self.p3, self.p1, self.canvas, "blue")
        # Setup the triangle
        self.triangle = Triangle(self.segment1, self.segment2, self.segment3, self.canvas, color="yellow")

    def test_initialize(self):
        self.triangle._initialize()
        # Test that the triangle's segments have been initialized correctly
        self.assertEqual(self.triangle.segment1.point1.x, 0)
        self.assertEqual(self.triangle.segment1.point1.y, 0)
        self.assertEqual(self.triangle.segment1.point2.x, 4)
        self.assertEqual(self.triangle.segment1.point2.y, 0)
        self.assertEqual(self.triangle.segment2.point1.x, 4)
        self.assertEqual(self.triangle.segment2.point1.y, 0)
        self.assertEqual(self.triangle.segment2.point2.x, 0)
        self.assertEqual(self.triangle.segment2.point2.y, -3) # Assuming coordinate system adjustments with 0,0 at top-left
        self.assertEqual(self.triangle.segment3.point1.x, 0)
        self.assertEqual(self.triangle.segment3.point1.y, -3) # Assuming coordinate system adjustments with 0,0 at top-left
        self.assertEqual(self.triangle.segment3.point2.x, 0)
        self.assertEqual(self.triangle.segment3.point2.y, 0)

    def test_init(self):
        # Test the initial properties of the triangle
        self.assertEqual(self.triangle.segment1, self.segment1)
        self.assertEqual(self.triangle.segment2, self.segment2)
        self.assertEqual(self.triangle.segment3, self.segment3)
        self.assertEqual(self.triangle.color, "yellow")

    def test_segments_form_triangle(self):
        # Given the corrected logic, ensure your segments are connected as per the new requirements
        self.assertTrue(self.triangle._segments_form_triangle(self.segment1, self.segment2, self.segment3))

    def test_segments_do_not_form_triangle(self):
        # Ensure this test reflects an incorrect connection based on the new logic
        incorrect_segment = Segment(self.p1, Point(2, 2, self.canvas, name="D"), self.canvas, "red")
        self.assertFalse(self.triangle._segments_form_triangle(self.segment1, self.segment2, incorrect_segment))

    def test_identical_points_not_forming_triangle(self):
        # Adjust if necessary to ensure the segments do not meet the new criteria for forming a triangle
        p4 = Point(1, 1, self.canvas, name="D", color="black")
        segment4 = Segment(self.p2, p4, self.canvas, "orange")
        segment5 = Segment(p4, self.p3, self.canvas, "purple")
        self.assertFalse(self.triangle._segments_form_triangle(self.segment1, segment4, segment5))

    def test_get_class_name(self):
        self.assertEqual(self.triangle.get_class_name(), 'Triangle')

    def test_get_state(self):
        state = self.triangle.get_state()
        expected_state = {"name": self.triangle.name, "args": {"p1": "P1", "p2": "P2", "p3": "P3"}}
        self.assertEqual(state, expected_state)

    def test_deepcopy(self):
        triangle_copy = copy.deepcopy(self.triangle)
        self.assertIsNot(triangle_copy, self.triangle)
        self.assertIsNot(triangle_copy.segment1, self.triangle.segment1)
        self.assertEqual(triangle_copy.color, self.triangle.color)

    def test_draw(self):
        # This test would check if draw calls create_svg_element with expected arguments
        # Might require a more complex setup or mocking to verify SVG output
        pass

