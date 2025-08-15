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
        # Validate via CoordinateMapper
        m = self.coordinate_mapper
        s1p1x, s1p1y = m.math_to_screen(self.rectangle.segment1.point1.x, self.rectangle.segment1.point1.y)
        s1p2x, s1p2y = m.math_to_screen(self.rectangle.segment1.point2.x, self.rectangle.segment1.point2.y)
        s2p1x, s2p1y = m.math_to_screen(self.rectangle.segment2.point1.x, self.rectangle.segment2.point1.y)
        s2p2x, s2p2y = m.math_to_screen(self.rectangle.segment2.point2.x, self.rectangle.segment2.point2.y)
        s3p1x, s3p1y = m.math_to_screen(self.rectangle.segment3.point1.x, self.rectangle.segment3.point1.y)
        self.assertEqual((s1p1x, s1p1y), (250, 250))
        self.assertEqual((s1p2x, s1p2y), (254, 250))
        self.assertEqual((s2p1x, s2p1y), (254, 250))
        self.assertEqual((s2p2x, s2p2y), (254, 247))
        self.assertEqual((s3p1x, s3p1y), (254, 247))

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
        
        # Validate via CoordinateMapper after translation
        m = self.coordinate_mapper
        s1p1x, s1p1y = m.math_to_screen(self.rectangle.segment1.point1.x, self.rectangle.segment1.point1.y)
        s1p2x, s1p2y = m.math_to_screen(self.rectangle.segment1.point2.x, self.rectangle.segment1.point2.y)
        s2p2x, s2p2y = m.math_to_screen(self.rectangle.segment2.point2.x, self.rectangle.segment2.point2.y)
        self.assertEqual((s1p1x, s1p1y), (252, 249))
        self.assertEqual((s1p2x, s1p2y), (256, 249))
        self.assertEqual((s2p2x, s2p2y), (256, 246))

