import unittest
import copy
from drawables_aggregator import Point, Position, Ellipse
from .simple_mock import SimpleMock
from coordinate_mapper import CoordinateMapper


class TestEllipse(unittest.TestCase):
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
            zoom_point=Position(1, 1), 
            zoom_direction=1, 
            zoom_step=0.1, 
            offset=Position(0, 0)  # Set to (0,0) for simpler tests
        )
        
        # Sync canvas state with coordinate mapper
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        
        self.center = Point(2, 2, name="Center", color="black")
        self.radius_x = 5
        self.radius_y = 3
        self.rotation_angle = 45
        self.ellipse = Ellipse(self.center, self.radius_x, self.radius_y, color="red", rotation_angle=self.rotation_angle)

    def test_init(self) -> None:
        self.assertEqual(self.ellipse.center, self.center)
        self.assertEqual(self.ellipse.radius_x, self.radius_x)
        self.assertEqual(self.ellipse.radius_y, self.radius_y)
        self.assertEqual(self.ellipse.rotation_angle, self.rotation_angle)
        self.assertEqual(self.ellipse.color, "red")

    def test_get_class_name(self) -> None:
        self.assertEqual(self.ellipse.get_class_name(), 'Ellipse')

    def test_calculate_ellipse_algebraic_formula(self) -> None:
        formula = self.ellipse._calculate_ellipse_algebraic_formula()
        self.assertIsNotNone(formula)

    def test_get_state(self) -> None:
        state = self.ellipse.get_state()
        expected_state = {
            "name": self.ellipse.name, 
            "args": {
                "center": self.center.name, 
                "radius_x": self.radius_x, 
                "radius_y": self.radius_y,
                "rotation_angle": self.rotation_angle,
                "ellipse_formula": self.ellipse.ellipse_formula
            }
        }
        self.assertEqual(state, expected_state)

    def test_deepcopy(self) -> None:
        ellipse_copy = copy.deepcopy(self.ellipse)
        self.assertIsNot(ellipse_copy, self.ellipse)
        self.assertIsNot(ellipse_copy.center, self.ellipse.center)
        self.assertEqual(ellipse_copy.radius_x, self.ellipse.radius_x)
        self.assertEqual(ellipse_copy.radius_y, self.ellipse.radius_y)
        self.assertEqual(ellipse_copy.color, self.ellipse.color)

    def test_draw(self) -> None:
        # This test would check if draw calls create_svg_element with expected arguments
        # Might require a more complex setup or mocking to verify SVG output
        pass

    def test_rotate(self) -> None:
        initial_angle = self.ellipse.rotation_angle
        rotation = 30
        self.ellipse.rotate(rotation)
        expected_angle = (initial_angle + rotation) % 360
        self.assertEqual(self.ellipse.rotation_angle, expected_angle)

    def test_translate_ellipse_in_math_space(self) -> None:
        """Test ellipse translation in mathematical space"""
        # Test center point coordinates before translation
        # Center at (2,2) in math space -> (252,248) in screen space
        x, y = self.coordinate_mapper.math_to_screen(self.ellipse.center.x, self.ellipse.center.y)
        self.assertEqual((x, y), (252, 248))
        
        original_formula = self.ellipse.ellipse_formula
        # Translate by (3, 1) in mathematical coordinates
        self.ellipse.translate(3, 1)
        
        # Center should move from (2,2) to (5,3) in math space -> (255,247) in screen space
        x, y = self.coordinate_mapper.math_to_screen(self.ellipse.center.x, self.ellipse.center.y)
        self.assertEqual((x, y), (255, 247))
        self.assertNotEqual(original_formula, self.ellipse.ellipse_formula)

