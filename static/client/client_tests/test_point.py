import unittest
import copy
from geometry import Point, Position
from coordinate_mapper import CoordinateMapper
from .simple_mock import SimpleMock


class TestPoint(unittest.TestCase):
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
        
        self.point = Point(1, 2, name="p1", color="red")

    def test_initialize(self):
        # Screen-space assertions use CoordinateMapper
        x, y = self.coordinate_mapper.math_to_screen(self.point.x, self.point.y)
        self.assertEqual((x, y), (251, 248))

    def test_init(self):
        self.assertEqual(self.point.x, 1)
        self.assertEqual(self.point.y, 2)
        self.assertEqual(self.point.x, 1)
        self.assertEqual(self.point.y, 2)
        self.assertEqual(self.point.name, "p1")
        self.assertEqual(self.point.color, "red")

    def test_get_class_name(self):
        self.assertEqual(self.point.get_class_name(), 'Point')

    def test_str(self):
        self.assertEqual(str(self.point), '1,2')

    def test_get_state(self):
        expected_state = {"name": "p1", "args": {"position": {"x": 1, "y": 2}}}
        self.assertEqual(self.point.get_state(), expected_state)

    def test_deepcopy(self):
        point_copy = copy.deepcopy(self.point)
        self.assertEqual(point_copy.x, self.point.x)
        self.assertEqual(point_copy.y, self.point.y)
        self.assertEqual(point_copy.name, self.point.name)
        self.assertEqual(point_copy.color, self.point.color)
        self.assertIsNot(point_copy, self.point)
        # Compatibility view removed; comparing direct coords instead

    def test_translate(self):
        initial_x, initial_y = self.coordinate_mapper.math_to_screen(self.point.x, self.point.y)
        # Translate in math-space so that screen shifts by (+1, +1) with scale 1
        self.point.translate(1, -1)
        x, y = self.coordinate_mapper.math_to_screen(self.point.x, self.point.y)
        self.assertEqual(x, initial_x + 1)
        self.assertEqual(y, initial_y + 1)

    def test_translate_point_in_math_space(self):
        # Test translating the point in mathematical coordinate space
        original_math_x = self.point.x
        original_math_y = self.point.y
        
        self.point.translate(2, 3)  # Translate by (2, 3) in math space
        
        # Check that original position was updated
        self.assertEqual(self.point.x, original_math_x + 2)
        self.assertEqual(self.point.y, original_math_y + 3)
        
        # Check that screen coordinates were recalculated
        # New math coords (3, 5) -> screen (253, 245)
        x, y = self.coordinate_mapper.math_to_screen(self.point.x, self.point.y)
        self.assertEqual((x, y), (253, 245))

    def test_draw(self):
        # This test would check if draw calls create_svg_element with expected arguments
        # Might require a more complex setup or mocking to verify SVG output
        pass

