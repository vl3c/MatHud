import unittest
import copy
from geometry import Point, Position, Segment
from coordinate_mapper import CoordinateMapper
from .simple_mock import SimpleMock


class TestSegment(unittest.TestCase):
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
        
        # Create test points with proper coordinate transformation
        self.p1 = Point(0, 0, self.canvas, name="A", color="red")
        self.p2 = Point(3, 4, self.canvas, name="B", color="red")
        self.segment = Segment(self.p1, self.p2, self.canvas, color="blue")

    def test_initialize(self):
        self.segment._initialize()
        # Real coordinate transformation: origin at (250, 250)
        # Point 1: (0,0) -> screen (250, 250)
        # Point 2: (3,4) -> screen (253, 246)
        self.assertEqual(self.segment.point1.x, 250)
        self.assertEqual(self.segment.point1.y, 250)
        self.assertEqual(self.segment.point2.x, 253)
        self.assertEqual(self.segment.point2.y, 246)

    def test_init(self):
        self.assertEqual(self.segment.point1, self.p1)
        self.assertEqual(self.segment.point2, self.p2)
        self.assertEqual(self.segment.color, "blue")

    def test_get_class_name(self):
        self.assertEqual(self.segment.get_class_name(), 'Segment')

    def test_calculate_line_algebraic_formula(self):
        line_formula = self.segment._calculate_line_algebraic_formula()
        # Expected formula depends on MathUtil.get_line_formula implementation
        self.assertIsNotNone(line_formula)  # Assert based on expected output

    def test_any_segment_part_visible_in_canvas_area(self):
        visible = self.segment._any_segment_part_visible_in_canvas_area()
        self.assertTrue(visible)

    def test_is_visible(self):
        self.assertTrue(self.segment.is_visible(), "Segment visibility should now depend only on canvas area checks.")

    def test_get_state(self):
        state = self.segment.get_state()
        expected_state = {"name": "AB", "args": {"p1": "A", "p2": "B", "line_formula": self.segment.line_formula}}
        self.assertEqual(state, expected_state)

    def test_deepcopy(self):
        segment_copy = copy.deepcopy(self.segment)
        self.assertIsNot(segment_copy, self.segment)
        self.assertIsNot(segment_copy.point1, self.segment.point1)
        self.assertIsNot(segment_copy.point2, self.segment.point2)
        self.assertEqual(segment_copy.color, self.segment.color)

    def test_translate_segment_in_math_space(self):
        # Test translating the segment in mathematical coordinate space
        original_p1_x = self.segment.point1.original_position.x
        original_p1_y = self.segment.point1.original_position.y
        original_p2_x = self.segment.point2.original_position.x
        original_p2_y = self.segment.point2.original_position.y
        
        self.segment.translate(2, 3)  # Translate by (2, 3) in math space
        
        # Check that original positions were updated
        self.assertEqual(self.segment.point1.original_position.x, original_p1_x + 2)  # 0 + 2 = 2
        self.assertEqual(self.segment.point1.original_position.y, original_p1_y + 3)  # 0 + 3 = 3
        self.assertEqual(self.segment.point2.original_position.x, original_p2_x + 2)  # 3 + 2 = 5
        self.assertEqual(self.segment.point2.original_position.y, original_p2_y + 3)  # 4 + 3 = 7
        
        # Check that screen coordinates were recalculated
        # New math coords: p1(2, 3) -> screen (252, 247), p2(5, 7) -> screen (255, 243)
        self.assertEqual(self.segment.point1.x, 252)
        self.assertEqual(self.segment.point1.y, 247)
        self.assertEqual(self.segment.point2.x, 255)
        self.assertEqual(self.segment.point2.y, 243)

    def test_draw(self):
        # This test would check if draw calls create_svg_element with expected arguments
        # Might require a more complex setup or mocking to verify SVG output
        pass

