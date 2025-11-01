import unittest
import copy
from geometry import Point, Position, Segment, Triangle
from .simple_mock import SimpleMock
from coordinate_mapper import CoordinateMapper


class TestTriangle(unittest.TestCase):
    def setUp(self) -> None:
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
        self.p1 = Point(0, 0, name="P1", color="red")
        self.p2 = Point(4, 0, name="P2", color="green")
        self.p3 = Point(0, 3, name="P3", color="blue")
        # Setup segments for the triangle
        self.segment1 = Segment(self.p1, self.p2, "red")
        self.segment2 = Segment(self.p2, self.p3, "green")
        self.segment3 = Segment(self.p3, self.p1, "blue")
        # Setup the triangle
        self.triangle = Triangle(self.segment1, self.segment2, self.segment3, color="yellow")

    def test_initialize(self) -> None:
        # Validate screen-space via CoordinateMapper
        m = self.coordinate_mapper
        s1p1x, s1p1y = m.math_to_screen(self.triangle.segment1.point1.x, self.triangle.segment1.point1.y)
        s1p2x, s1p2y = m.math_to_screen(self.triangle.segment1.point2.x, self.triangle.segment1.point2.y)
        s2p1x, s2p1y = m.math_to_screen(self.triangle.segment2.point1.x, self.triangle.segment2.point1.y)
        s2p2x, s2p2y = m.math_to_screen(self.triangle.segment2.point2.x, self.triangle.segment2.point2.y)
        s3p1x, s3p1y = m.math_to_screen(self.triangle.segment3.point1.x, self.triangle.segment3.point1.y)
        s3p2x, s3p2y = m.math_to_screen(self.triangle.segment3.point2.x, self.triangle.segment3.point2.y)
        self.assertEqual((s1p1x, s1p1y), (250, 250))
        self.assertEqual((s1p2x, s1p2y), (254, 250))
        self.assertEqual((s2p1x, s2p1y), (254, 250))
        self.assertEqual((s2p2x, s2p2y), (250, 247))
        self.assertEqual((s3p1x, s3p1y), (250, 247))
        self.assertEqual((s3p2x, s3p2y), (250, 250))

    def test_init(self) -> None:
        # Test the initial properties of the triangle
        self.assertEqual(self.triangle.segment1, self.segment1)
        self.assertEqual(self.triangle.segment2, self.segment2)
        self.assertEqual(self.triangle.segment3, self.segment3)
        self.assertEqual(self.triangle.color, "yellow")

    def test_segments_form_triangle(self) -> None:
        # Given the corrected logic, ensure your segments are connected as per the new requirements
        self.assertTrue(self.triangle._segments_form_triangle(self.segment1, self.segment2, self.segment3))

    def test_segments_do_not_form_triangle(self) -> None:
        # Ensure this test reflects an incorrect connection based on the new logic
        incorrect_segment = Segment(self.p1, Point(2, 2, name="D"), "red")
        self.assertFalse(self.triangle._segments_form_triangle(self.segment1, self.segment2, incorrect_segment))

    def test_identical_points_not_forming_triangle(self) -> None:
        # Adjust if necessary to ensure the segments do not meet the new criteria for forming a triangle
        p4 = Point(1, 1, name="D", color="black")
        segment4 = Segment(self.p2, p4, "orange")
        segment5 = Segment(p4, self.p3, "purple")
        self.assertFalse(self.triangle._segments_form_triangle(self.segment1, segment4, segment5))

    def test_get_class_name(self) -> None:
        self.assertEqual(self.triangle.get_class_name(), 'Triangle')

    def test_get_state(self) -> None:
        state = self.triangle.get_state()
        expected_state = {"name": self.triangle.name, "args": {"p1": "P1", "p2": "P2", "p3": "P3"}}
        self.assertEqual(state, expected_state)

    def test_deepcopy(self) -> None:
        triangle_copy = copy.deepcopy(self.triangle)
        self.assertIsNot(triangle_copy, self.triangle)
        self.assertIsNot(triangle_copy.segment1, self.triangle.segment1)
        self.assertEqual(triangle_copy.color, self.triangle.color)

    def test_draw(self) -> None:
        # This test would check if draw calls create_svg_element with expected arguments
        # Might require a more complex setup or mocking to verify SVG output
        pass

    def test_translate_triangle_in_math_space(self) -> None:
        """Test triangle translation in mathematical space"""
        # Translate by (1, 2) in mathematical coordinates
        self.triangle.translate(1, 2)
        
        # Validate via CoordinateMapper
        m = self.coordinate_mapper
        s1p1x, s1p1y = m.math_to_screen(self.triangle.segment1.point1.x, self.triangle.segment1.point1.y)
        s1p2x, s1p2y = m.math_to_screen(self.triangle.segment1.point2.x, self.triangle.segment1.point2.y)
        s2p2x, s2p2y = m.math_to_screen(self.triangle.segment2.point2.x, self.triangle.segment2.point2.y)
        self.assertEqual((s1p1x, s1p1y), (251, 248))
        self.assertEqual((s1p2x, s1p2y), (255, 248))
        self.assertEqual((s2p2x, s2p2y), (251, 245))

