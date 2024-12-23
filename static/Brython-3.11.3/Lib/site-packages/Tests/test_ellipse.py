import unittest
import copy
from geometry import Point, Position, Ellipse
from .simple_mock import SimpleMock


class TestEllipse(unittest.TestCase):
    def setUp(self):
        self.canvas = SimpleMock(scale_factor=1, cartesian2axis=SimpleMock(origin=Position(0, 0)),
                                 is_point_within_canvas_visible_area=SimpleMock(return_value=True))
        self.center = Point(2, 2, self.canvas, name="Center", color="black")
        self.radius_x = 5
        self.radius_y = 3
        self.rotation_angle = 45
        self.ellipse = Ellipse(self.center, self.radius_x, self.radius_y, self.canvas, color="red", rotation_angle=self.rotation_angle)

    def test_initialize(self):
        self.ellipse._initialize()
        self.assertEqual(self.ellipse.drawn_radius_x, self.radius_x * self.canvas.scale_factor)
        self.assertEqual(self.ellipse.drawn_radius_y, self.radius_y * self.canvas.scale_factor)
        self.assertEqual(self.ellipse.center, self.center)

    def test_init(self):
        self.assertEqual(self.ellipse.center, self.center)
        self.assertEqual(self.ellipse.radius_x, self.radius_x)
        self.assertEqual(self.ellipse.radius_y, self.radius_y)
        self.assertEqual(self.ellipse.rotation_angle, self.rotation_angle)
        self.assertEqual(self.ellipse.color, "red")

    def test_get_class_name(self):
        self.assertEqual(self.ellipse.get_class_name(), 'Ellipse')

    def test_calculate_ellipse_algebraic_formula(self):
        formula = self.ellipse._calculate_ellipse_algebraic_formula()
        self.assertIsNotNone(formula)

    def test_zoom(self):
        new_scale_factor = 2
        self.canvas.scale_factor = new_scale_factor
        self.ellipse.zoom()
        self.assertEqual(self.ellipse.drawn_radius_x, self.radius_x * new_scale_factor)
        self.assertEqual(self.ellipse.drawn_radius_y, self.radius_y * new_scale_factor)

    def test_get_state(self):
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

    def test_deepcopy(self):
        ellipse_copy = copy.deepcopy(self.ellipse)
        self.assertIsNot(ellipse_copy, self.ellipse)
        self.assertIsNot(ellipse_copy.center, self.ellipse.center)
        self.assertEqual(ellipse_copy.radius_x, self.ellipse.radius_x)
        self.assertEqual(ellipse_copy.radius_y, self.ellipse.radius_y)
        self.assertEqual(ellipse_copy.color, self.ellipse.color)

    def test_draw(self):
        # This test would check if draw calls create_svg_element with expected arguments
        # Might require a more complex setup or mocking to verify SVG output
        pass

    def test_rotate(self):
        initial_angle = self.ellipse.rotation_angle
        rotation = 30
        self.ellipse.rotate(rotation)
        expected_angle = (initial_angle + rotation) % 360
        self.assertEqual(self.ellipse.rotation_angle, expected_angle)

