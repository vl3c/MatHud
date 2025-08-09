import unittest
import copy
from geometry import Point, Position, Circle
from .simple_mock import SimpleMock
from coordinate_mapper import CoordinateMapper


class TestCircle(unittest.TestCase):
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
            zoom_point=Position(1, 1), 
            zoom_direction=1, 
            zoom_step=0.1, 
            offset=Position(0, 0)  # Set to (0,0) for simpler tests
        )
        
        # Sync canvas state with coordinate mapper
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        
        self.center = Point(1, 1, self.canvas, name="Center", color="black")
        self.radius = 5
        self.circle = Circle(self.center, self.radius, self.canvas, color="blue")

    def test_init(self):
        self.assertEqual(self.circle.center, self.center)
        self.assertEqual(self.circle.radius, self.radius)
        self.assertEqual(self.circle.color, "blue")

    def test_get_class_name(self):
        self.assertEqual(self.circle.get_class_name(), 'Circle')

    def test_calculate_circle_algebraic_formula(self):
        formula = self.circle._calculate_circle_algebraic_formula()
        self.assertIsNotNone(formula)

    def test_get_state(self):
        state = self.circle.get_state()
        expected_state = {"name": self.circle.name, "args": {"center": self.center.name, "radius": self.radius, "circle_formula": self.circle.circle_formula}}
        self.assertEqual(state, expected_state)

    def test_deepcopy(self):
        circle_copy = copy.deepcopy(self.circle)
        self.assertIsNot(circle_copy, self.circle)
        self.assertIsNot(circle_copy.center, self.circle.center)
        self.assertEqual(circle_copy.radius, self.circle.radius)
        self.assertEqual(circle_copy.color, self.circle.color)

    def test_draw(self):
        # This test would check if draw calls create_svg_element with expected arguments
        # Might require a more complex setup or mocking to verify SVG output
        pass

    def test_translate_circle_in_math_space(self):
        """Test circle translation in mathematical space"""
        # Test center point coordinates before translation
        # Center at (1,1) in math space -> (251,249) in screen space
        self.assertEqual(self.circle.center.screen_x, 251)
        self.assertEqual(self.circle.center.screen_y, 249)
        
        # Translate by (2, 3) in mathematical coordinates
        self.circle.translate(2, 3)
        
        # Center should move from (1,1) to (3,4) in math space -> (253,246) in screen space
        self.assertEqual(self.circle.center.screen_x, 253)
        self.assertEqual(self.circle.center.screen_y, 246)
