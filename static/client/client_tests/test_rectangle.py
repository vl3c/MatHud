import unittest
import copy
from geometry import Point, Position, Segment, Rectangle
from .simple_mock import SimpleMock
from coordinate_mapper import CoordinateMapper


class TestRectangle(unittest.TestCase):
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
        
        # Setup points for the rectangle
        self.p1 = Point(0, 0, self.canvas, name="P1", color="red")
        self.p2 = Point(4, 0, self.canvas, name="P2", color="green")
        self.p3 = Point(4, 3, self.canvas, name="P3", color="blue")
        self.p4 = Point(0, 3, self.canvas, name="P4", color="yellow")
        # Setup segments for the rectangle
        self.segment1 = Segment(self.p1, self.p2, self.canvas, "red")
        self.segment2 = Segment(self.p2, self.p3, self.canvas, "green")
        self.segment3 = Segment(self.p3, self.p4, self.canvas, "blue")
        self.segment4 = Segment(self.p4, self.p1, self.canvas, "yellow")
        # Setup the rectangle
        self.rectangle = Rectangle(self.segment1, self.segment2, self.segment3, self.segment4, self.canvas, color="orange")

    def test_initialize(self):
        self.rectangle._initialize()
        # Test with real coordinate transformations
        # P1 (0,0) in math space -> (250,250) in screen space
        self.assertEqual(self.rectangle.segment1.point1.screen_x, 250)
        self.assertEqual(self.rectangle.segment1.point1.screen_y, 250)
        # P2 (4,0) in math space -> (254,250) in screen space
        self.assertEqual(self.rectangle.segment1.point2.screen_x, 254)
        self.assertEqual(self.rectangle.segment1.point2.screen_y, 250)
        # P2 (4,0) in math space -> (254,250) in screen space
        self.assertEqual(self.rectangle.segment2.point1.screen_x, 254)
        self.assertEqual(self.rectangle.segment2.point1.screen_y, 250)
        # P3 (4,3) in math space -> (254,247) in screen space
        self.assertEqual(self.rectangle.segment2.point2.screen_x, 254)
        self.assertEqual(self.rectangle.segment2.point2.screen_y, 247)
        # P3 (4,3) in math space -> (254,247) in screen space
        self.assertEqual(self.rectangle.segment3.point1.screen_x, 254)
        self.assertEqual(self.rectangle.segment3.point1.screen_y, 247)

    def test_init(self):
        # Test the initial properties of the rectangle
        self.assertEqual(self.rectangle.segment1, self.segment1)
        self.assertEqual(self.rectangle.segment2, self.segment2)
        self.assertEqual(self.rectangle.segment3, self.segment3)
        self.assertEqual(self.rectangle.segment4, self.segment4)
        self.assertEqual(self.rectangle.color, "orange")

    def test_get_class_name(self):
        self.assertEqual(self.rectangle.get_class_name(), 'Rectangle')

    def test_get_state(self):
        state = self.rectangle.get_state()
        # Expected state needs to account for the names of the points, which should be sorted and unique
        expected_state = {"name": self.rectangle.name, "args": {"p1": "P1", "p2": "P2", "p3": "P3", "p4": "P4"}}
        self.assertEqual(state, expected_state)

    def test_deepcopy(self):
        rectangle_copy = copy.deepcopy(self.rectangle)
        self.assertIsNot(rectangle_copy, self.rectangle)
        self.assertIsNot(rectangle_copy.segment1, self.rectangle.segment1)
        self.assertEqual(rectangle_copy.color, self.rectangle.color)

    def test_draw(self):
        # This test would check if draw calls create_svg_element with expected arguments
        # Might require a more complex setup or mocking to verify SVG output
        pass

    def test_translate_rectangle_in_math_space(self):
        """Test rectangle translation in mathematical space"""
        # Translate by (2, 1) in mathematical coordinates
        self.rectangle.translate(2, 1)
        
        # P1 should move from (0,0) to (2,1) in math space -> (252,249) in screen space
        self.assertEqual(self.rectangle.segment1.point1.screen_x, 252)
        self.assertEqual(self.rectangle.segment1.point1.screen_y, 249)
        # P2 should move from (4,0) to (6,1) in math space -> (256,249) in screen space
        self.assertEqual(self.rectangle.segment1.point2.screen_x, 256)
        self.assertEqual(self.rectangle.segment1.point2.screen_y, 249)
        # P3 should move from (4,3) to (6,4) in math space -> (256,246) in screen space
        self.assertEqual(self.rectangle.segment2.point2.screen_x, 256)
        self.assertEqual(self.rectangle.segment2.point2.screen_y, 246)

