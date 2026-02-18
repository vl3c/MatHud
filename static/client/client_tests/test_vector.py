import unittest
import copy
from drawables_aggregator import Point, Position, Vector
from .simple_mock import SimpleMock
from coordinate_mapper import CoordinateMapper


class TestVector(unittest.TestCase):
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
            offset=Position(0, 0),  # Set to (0,0) for simpler tests
        )

        # Sync canvas state with coordinate mapper
        self.coordinate_mapper.sync_from_canvas(self.canvas)

        self.origin = Point(0, 0, name="O", color="black")
        self.tip = Point(3, 4, name="T", color="black")
        self.vector = Vector(self.origin, self.tip, color="green")

    def test_initialize(self) -> None:
        # Validate via CoordinateMapper
        m = self.coordinate_mapper
        ox, oy = m.math_to_screen(self.vector.origin.x, self.vector.origin.y)
        tx, ty = m.math_to_screen(self.vector.tip.x, self.vector.tip.y)
        self.assertEqual((ox, oy), (250, 250))
        self.assertEqual((tx, ty), (253, 246))

    def test_init(self) -> None:
        # Test the initial properties of the vector
        self.assertEqual(self.vector.segment.point1, self.origin)
        self.assertEqual(self.vector.segment.point2, self.tip)
        self.assertEqual(self.vector.color, "green")

    def test_get_class_name(self) -> None:
        self.assertEqual(self.vector.get_class_name(), "Vector")

    def test_get_state(self) -> None:
        state = self.vector.get_state()
        expected_state = {
            "name": "OT",
            "args": {
                "origin": "O",
                "tip": "T",
            },
            "_origin_coords": [self.origin.x, self.origin.y],
            "_tip_coords": [self.tip.x, self.tip.y],
        }
        self.assertEqual(state, expected_state)

    def test_deepcopy(self) -> None:
        vector_copy = copy.deepcopy(self.vector)
        self.assertIsNot(vector_copy, self.vector)
        self.assertIsNot(vector_copy.segment, self.vector.segment)
        self.assertEqual(vector_copy.color, self.vector.color)

    def test_draw(self) -> None:
        # This test would check if draw calls create_svg_element with expected arguments
        # Might require a more complex setup or mocking to verify SVG output
        pass

    def test_translate_vector_in_math_space(self) -> None:
        """Test vector translation in mathematical space"""
        # Translate by (2, 3) in mathematical coordinates
        self.vector.translate(2, 3)

        # Validate via CoordinateMapper after translation
        m = self.coordinate_mapper
        ox, oy = m.math_to_screen(self.vector.origin.x, self.vector.origin.y)
        tx, ty = m.math_to_screen(self.vector.tip.x, self.vector.tip.y)
        self.assertEqual((ox, oy), (252, 247))
        self.assertEqual((tx, ty), (255, 243))

    def test_get_state_includes_coordinates_for_cache_invalidation(self) -> None:
        """Verify get_state includes point coordinates so render cache invalidates on move."""
        state = self.vector.get_state()

        # Must include coordinate lists for render signature
        self.assertIn("_origin_coords", state)
        self.assertIn("_tip_coords", state)
        self.assertEqual(state["_origin_coords"], [self.origin.x, self.origin.y])
        self.assertEqual(state["_tip_coords"], [self.tip.x, self.tip.y])

    def test_get_state_changes_when_points_move(self) -> None:
        """Verify get_state signature changes when point coordinates change."""
        state_before = self.vector.get_state()

        # Move a point
        self.origin.x = 100.0
        self.origin.y = 200.0

        state_after = self.vector.get_state()

        # Coordinates in state should reflect new position
        self.assertEqual(state_after["_origin_coords"], [100.0, 200.0])
        # And be different from before
        self.assertNotEqual(state_before["_origin_coords"], state_after["_origin_coords"])

    def test_same_name_different_coords_produces_different_state(self) -> None:
        """Two vectors with same name but different coords must have different states."""
        state1 = self.vector.get_state()

        # Create another vector with same point names but different coordinates
        other_origin = Point(500, 600, name="O", color="red")
        other_tip = Point(700, 800, name="T", color="red")
        other_vector = Vector(other_origin, other_tip, color="blue")
        state2 = other_vector.get_state()

        # Names are the same
        self.assertEqual(state1["name"], state2["name"])
        self.assertEqual(state1["args"]["origin"], state2["args"]["origin"])
        self.assertEqual(state1["args"]["tip"], state2["args"]["tip"])

        # But coordinates differ, so render cache should invalidate
        self.assertNotEqual(state1["_origin_coords"], state2["_origin_coords"])
        self.assertNotEqual(state1["_tip_coords"], state2["_tip_coords"])
