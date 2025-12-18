import unittest
import copy
from drawables_aggregator import Point, Position, Segment
from coordinate_mapper import CoordinateMapper
from .simple_mock import SimpleMock


class TestSegment(unittest.TestCase):
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
        
        # Create test points with proper coordinate transformation
        self.p1 = Point(0, 0, name="A", color="red")
        self.p2 = Point(3, 4, name="B", color="red")
        self.segment = Segment(self.p1, self.p2, color="blue")

    def test_initialize(self) -> None:
        # Real coordinate transformation: origin at (250, 250)
        # Point 1: (0,0) -> screen (250, 250)
        # Point 2: (3,4) -> screen (253, 246)
        x1, y1 = self.coordinate_mapper.math_to_screen(self.segment.point1.x, self.segment.point1.y)
        x2, y2 = self.coordinate_mapper.math_to_screen(self.segment.point2.x, self.segment.point2.y)
        self.assertEqual(x1, 250)
        self.assertEqual(y1, 250)
        self.assertEqual(x2, 253)
        self.assertEqual(y2, 246)

    def test_init(self) -> None:
        self.assertEqual(self.segment.point1, self.p1)
        self.assertEqual(self.segment.point2, self.p2)
        self.assertEqual(self.segment.color, "blue")

    def test_get_class_name(self) -> None:
        self.assertEqual(self.segment.get_class_name(), 'Segment')

    def test_calculate_line_algebraic_formula(self) -> None:
        line_formula = self.segment._calculate_line_algebraic_formula()
        # Expected formula depends on MathUtil.get_line_formula implementation
        self.assertIsNotNone(line_formula)  # Assert based on expected output

    def test_visibility_via_canvas(self) -> None:
        from canvas import Canvas  # not used directly; we mimic Canvas._is_drawable_visible logic
        # Compute visibility using canvas-level predicate
        # Endpoint-in-viewport or intersects viewport
        x1, y1 = self.coordinate_mapper.math_to_screen(self.segment.point1.x, self.segment.point1.y)
        x2, y2 = self.coordinate_mapper.math_to_screen(self.segment.point2.x, self.segment.point2.y)
        in_view = self.canvas.is_point_within_canvas_visible_area(x1, y1) or \
                  self.canvas.is_point_within_canvas_visible_area(x2, y2) or \
                  self.canvas.any_segment_part_visible_in_canvas_area(x1, y1, x2, y2)
        self.assertTrue(in_view)

    def test_get_state(self) -> None:
        state = self.segment.get_state()
        expected_state = {
            "name": "AB",
            "args": {
                "p1": "A",
                "p2": "B",
                "label": {
                    "text": "",
                    "visible": False,
                    "font_size": self.segment.label.font_size,
                },
            },
            "_p1_coords": [self.p1.x, self.p1.y],
            "_p2_coords": [self.p2.x, self.p2.y],
        }
        self.assertEqual(state, expected_state)

    def test_deepcopy(self) -> None:
        segment_copy = copy.deepcopy(self.segment)
        self.assertIsNot(segment_copy, self.segment)
        self.assertIsNot(segment_copy.point1, self.segment.point1)
        self.assertIsNot(segment_copy.point2, self.segment.point2)
        self.assertEqual(segment_copy.color, self.segment.color)

    def test_translate_segment_in_math_space(self) -> None:
        # Test translating the segment in mathematical coordinate space
        original_p1_x = self.segment.point1.x
        original_p1_y = self.segment.point1.y
        original_p2_x = self.segment.point2.x
        original_p2_y = self.segment.point2.y
        
        self.segment.translate(2, 3)  # Translate by (2, 3) in math space
        
        # Check that original positions were updated
        self.assertEqual(self.segment.point1.x, original_p1_x + 2)
        self.assertEqual(self.segment.point1.y, original_p1_y + 3)
        self.assertEqual(self.segment.point2.x, original_p2_x + 2)
        self.assertEqual(self.segment.point2.y, original_p2_y + 3)
        
        # Check that screen coordinates were recalculated
        # New math coords: p1(2, 3) -> screen (252, 247), p2(5, 7) -> screen (255, 243)
        x1, y1 = self.coordinate_mapper.math_to_screen(self.segment.point1.x, self.segment.point1.y)
        x2, y2 = self.coordinate_mapper.math_to_screen(self.segment.point2.x, self.segment.point2.y)
        self.assertEqual(x1, 252)
        self.assertEqual(y1, 247)
        self.assertEqual(x2, 255)
        self.assertEqual(y2, 243)

    def test_segment_line_formula_updates_after_point_translation(self) -> None:
        # Move only the first endpoint as a polygon translation would.
        self.segment.point1.translate(1, 2)

        # Recalculate line formula to reflect moved point.
        self.segment.line_formula = self.segment._calculate_line_algebraic_formula()

        self.assertEqual(self.segment.line_formula, "y = 1.0 * x + 1.0")

    def test_draw(self) -> None:
        # This test would check if draw calls create_svg_element with expected arguments
        # Might require a more complex setup or mocking to verify SVG output
        pass

    def test_get_state_includes_coordinates_for_cache_invalidation(self) -> None:
        """Verify get_state includes point coordinates so render cache invalidates on move."""
        state = self.segment.get_state()
        
        # Must include coordinate lists for render signature
        self.assertIn("_p1_coords", state)
        self.assertIn("_p2_coords", state)
        self.assertEqual(state["_p1_coords"], [self.p1.x, self.p1.y])
        self.assertEqual(state["_p2_coords"], [self.p2.x, self.p2.y])

    def test_get_state_changes_when_points_move(self) -> None:
        """Verify get_state signature changes when point coordinates change."""
        state_before = self.segment.get_state()
        
        # Move a point
        self.p1.x = 100.0
        self.p1.y = 200.0
        
        state_after = self.segment.get_state()
        
        # Coordinates in state should reflect new position
        self.assertEqual(state_after["_p1_coords"], [100.0, 200.0])
        # And be different from before
        self.assertNotEqual(state_before["_p1_coords"], state_after["_p1_coords"])

    def test_same_name_different_coords_produces_different_state(self) -> None:
        """Two segments with same name but different coords must have different states."""
        # Original segment AB at (0,0) to (3,4)
        state1 = self.segment.get_state()
        
        # Create another segment with same point names but different coordinates
        other_p1 = Point(500, 600, name="A", color="red")
        other_p2 = Point(700, 800, name="B", color="red")
        other_segment = Segment(other_p1, other_p2, color="blue")
        state2 = other_segment.get_state()
        
        # Names are the same
        self.assertEqual(state1["name"], state2["name"])
        self.assertEqual(state1["args"]["p1"], state2["args"]["p1"])
        self.assertEqual(state1["args"]["p2"], state2["args"]["p2"])
        
        # But coordinates differ, so render cache should invalidate
        self.assertNotEqual(state1["_p1_coords"], state2["_p1_coords"])
        self.assertNotEqual(state1["_p2_coords"], state2["_p2_coords"])

