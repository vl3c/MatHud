import unittest
import copy
from geometry import Point, Position, Vector
from .simple_mock import SimpleMock
from coordinate_mapper import CoordinateMapper


class TestVector(unittest.TestCase):
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
        
        self.origin = Point(0, 0, self.canvas, name="O", color="black")
        self.tip = Point(3, 4, self.canvas, name="T", color="black")
        self.vector = Vector(self.origin, self.tip, self.canvas, color="green")

    def test_initialize(self):
        self.vector._initialize()
        # Validate via CoordinateMapper
        m = self.coordinate_mapper
        ox, oy = m.math_to_screen(self.vector.segment.point1.original_position.x, self.vector.segment.point1.original_position.y)
        tx, ty = m.math_to_screen(self.vector.segment.point2.original_position.x, self.vector.segment.point2.original_position.y)
        self.assertEqual((ox, oy), (250, 250))
        self.assertEqual((tx, ty), (253, 246))

    def test_init(self):
        # Test the initial properties of the vector
        self.assertEqual(self.vector.segment.point1, self.origin)
        self.assertEqual(self.vector.segment.point2, self.tip)
        self.assertEqual(self.vector.color, "green")

    def test_get_class_name(self):
        self.assertEqual(self.vector.get_class_name(), 'Vector')

    def test_get_state(self):
        state = self.vector.get_state()
        expected_state = {"name": "OT", "args": {"origin": "O", "tip": "T", "line_formula": self.vector.segment.line_formula}}
        self.assertEqual(state, expected_state)

    def test_deepcopy(self):
        vector_copy = copy.deepcopy(self.vector)
        self.assertIsNot(vector_copy, self.vector)
        self.assertIsNot(vector_copy.segment, self.vector.segment)
        self.assertEqual(vector_copy.color, self.vector.color)

    def test_draw(self):
        # This test would check if draw calls create_svg_element with expected arguments
        # Might require a more complex setup or mocking to verify SVG output
        pass

    def test_translate_vector_in_math_space(self):
        """Test vector translation in mathematical space"""
        # Translate by (2, 3) in mathematical coordinates
        self.vector.translate(2, 3)
        
        # Validate via CoordinateMapper after translation
        m = self.coordinate_mapper
        ox, oy = m.math_to_screen(self.vector.origin.original_position.x, self.vector.origin.original_position.y)
        tx, ty = m.math_to_screen(self.vector.tip.original_position.x, self.vector.tip.original_position.y)
        self.assertEqual((ox, oy), (252, 247))
        self.assertEqual((tx, ty), (255, 243))

