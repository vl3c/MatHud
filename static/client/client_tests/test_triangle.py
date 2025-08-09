import unittest
import copy
from geometry import Point, Position, Segment, Triangle
from .simple_mock import SimpleMock
from coordinate_mapper import CoordinateMapper


class TestTriangle(unittest.TestCase):
    def setUp(self):
        # Create a real CoordinateMapper instance
        self.coordinate_mapper = CoordinateMapper(500, 500)  # 500x500 canvas
        
        # Create canvas mock with all properties that CoordinateMapper needs
        self.canvas = SimpleMock(
            width=500,  # Required by sync_from_canvas
            height=500,  # Required by sync_from_canvas
            scale_factor=1, 
            center=Position(250, 250),  # Canvas center
            cartesian2axis=SimpleMock(origin=Position(250, 250)),  # Coordinate system origin
            coordinate_mapper=self.coordinate_mapper,
            is_point_within_canvas_visible_area=SimpleMock(return_value=True),
            any_segment_part_visible_in_canvas_area=SimpleMock(return_value=True),
            zoom_point=Position(1, 1), 
            zoom_direction=1, 
            zoom_step=0.1, 
            offset=Position(0, 0)  # Set to (0,0) for simpler tests
        )
        
        # Sync canvas state with coordinate mapper
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        
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
        # Test that the triangle's segments have been initialized correctly with real coordinate transformations
        # P1 (0,0) in math space -> (250,250) in screen space
        self.assertEqual(self.triangle.segment1.point1.screen_x, 250)
        self.assertEqual(self.triangle.segment1.point1.screen_y, 250)
        # P2 (4,0) in math space -> (254,250) in screen space
        self.assertEqual(self.triangle.segment1.point2.screen_x, 254)
        self.assertEqual(self.triangle.segment1.point2.screen_y, 250)
        # P2 (4,0) in math space -> (254,250) in screen space
        self.assertEqual(self.triangle.segment2.point1.screen_x, 254)
        self.assertEqual(self.triangle.segment2.point1.screen_y, 250)
        # P3 (0,3) in math space -> (250,247) in screen space
        self.assertEqual(self.triangle.segment2.point2.screen_x, 250)
        self.assertEqual(self.triangle.segment2.point2.screen_y, 247)
        # P3 (0,3) in math space -> (250,247) in screen space
        self.assertEqual(self.triangle.segment3.point1.screen_x, 250)
        self.assertEqual(self.triangle.segment3.point1.screen_y, 247)
        # P1 (0,0) in math space -> (250,250) in screen space
        self.assertEqual(self.triangle.segment3.point2.screen_x, 250)
        self.assertEqual(self.triangle.segment3.point2.screen_y, 250)

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

    def test_translate_triangle_in_math_space(self):
        """Test triangle translation in mathematical space"""
        # Translate by (1, 2) in mathematical coordinates
        self.triangle.translate(1, 2)
        
        # P1 should move from (0,0) to (1,2) in math space -> (251,248) in screen space
        self.assertEqual(self.triangle.segment1.point1.screen_x, 251)
        self.assertEqual(self.triangle.segment1.point1.screen_y, 248)
        # P2 should move from (4,0) to (5,2) in math space -> (255,248) in screen space
        self.assertEqual(self.triangle.segment1.point2.screen_x, 255)
        self.assertEqual(self.triangle.segment1.point2.screen_y, 248)
        # P3 should move from (0,3) to (1,5) in math space -> (251,245) in screen space
        self.assertEqual(self.triangle.segment2.point2.screen_x, 251)
        self.assertEqual(self.triangle.segment2.point2.screen_y, 245)

