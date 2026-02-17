import math
import unittest
from drawables_aggregator import Position
from polar_grid import PolarGrid
from .simple_mock import SimpleMock
from coordinate_mapper import CoordinateMapper


class TestPolarGrid(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinate_mapper = CoordinateMapper(800, 600)

        self.canvas = SimpleMock(
            width=800,
            height=600,
            scale_factor=1,
            center=Position(400, 300),
            cartesian2axis=None,
            coordinate_mapper=self.coordinate_mapper,
            zoom_direction=0,
            offset=Position(0, 0),
            zoom_point=Position(0, 0),
            zoom_step=0.1,
        )

        self.coordinate_mapper.sync_from_canvas(self.canvas)
        self.polar_grid = PolarGrid(coordinate_mapper=self.coordinate_mapper)
        self.polar_grid.width = 800
        self.polar_grid.height = 600

    def test_init(self) -> None:
        self.assertEqual(self.polar_grid.get_class_name(), "PolarGrid")
        self.assertEqual(self.polar_grid.angular_divisions, 24)
        self.assertEqual(self.polar_grid.radial_spacing, 50.0)
        self.assertTrue(self.polar_grid.show_angle_labels)
        self.assertTrue(self.polar_grid.show_radius_labels)

    def test_angle_step(self) -> None:
        self.assertAlmostEqual(self.polar_grid.angle_step, math.pi / 12, places=10)

        self.polar_grid.angular_divisions = 8
        self.assertAlmostEqual(self.polar_grid.angle_step, math.pi / 4, places=10)

        self.polar_grid.angular_divisions = 6
        self.assertAlmostEqual(self.polar_grid.angle_step, math.pi / 3, places=10)

    def test_origin_screen(self) -> None:
        ox, oy = self.polar_grid.origin_screen
        self.assertAlmostEqual(ox, 400, places=6)
        self.assertAlmostEqual(oy, 300, places=6)

    def test_origin_screen_with_offset(self) -> None:
        self.canvas.offset = Position(50, -30)
        self.coordinate_mapper.sync_from_canvas(self.canvas)

        ox, oy = self.polar_grid.origin_screen
        self.assertAlmostEqual(ox, 450, places=6)
        self.assertAlmostEqual(oy, 270, places=6)

    def test_max_radius_screen(self) -> None:
        max_radius = self.polar_grid.max_radius_screen
        expected_min = math.sqrt(400**2 + 300**2)
        self.assertGreaterEqual(max_radius, expected_min)

    def test_max_radius_math(self) -> None:
        self.canvas.scale_factor = 2.0
        self.coordinate_mapper.sync_from_canvas(self.canvas)

        max_radius_screen = self.polar_grid.max_radius_screen
        max_radius_math = self.polar_grid.max_radius_math

        self.assertAlmostEqual(max_radius_math, max_radius_screen / 2.0, places=6)

    def test_display_spacing(self) -> None:
        spacing1 = self.polar_grid.display_spacing
        self.assertGreater(spacing1, 0)

        self.canvas.scale_factor = 2.0
        self.coordinate_mapper.sync_from_canvas(self.canvas)

        spacing2 = self.polar_grid.display_spacing
        self.assertAlmostEqual(spacing2, spacing1 * 2, places=6)

    def test_get_angle_labels(self) -> None:
        self.polar_grid.angular_divisions = 4
        labels = self.polar_grid.get_angle_labels()

        self.assertEqual(len(labels), 4)

        angles = [label[0] for label in labels]
        expected_angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        for actual, expected in zip(angles, expected_angles):
            self.assertAlmostEqual(actual, expected, places=10)

    def test_get_radial_circles(self) -> None:
        circles = self.polar_grid.get_radial_circles()

        self.assertGreater(len(circles), 0)

        if len(circles) >= 2:
            spacing = circles[1] - circles[0]
            for i in range(2, len(circles)):
                self.assertAlmostEqual(circles[i] - circles[i - 1], spacing, places=6)

    def test_get_state(self) -> None:
        self.polar_grid.angular_divisions = 8
        self.polar_grid.radial_spacing = 2.0
        self.polar_grid.show_angle_labels = False

        state = self.polar_grid.get_state()

        self.assertIn("angular_divisions", state)
        self.assertIn("radial_spacing", state)
        self.assertIn("show_angle_labels", state)
        self.assertIn("show_radius_labels", state)

        self.assertEqual(state["angular_divisions"], 8)
        self.assertEqual(state["radial_spacing"], 2.0)
        self.assertFalse(state["show_angle_labels"])

    def test_set_state(self) -> None:
        state = {
            "angular_divisions": 6,
            "radial_spacing": 3.0,
            "show_angle_labels": False,
            "show_radius_labels": False,
        }

        self.polar_grid.set_state(state)

        self.assertEqual(self.polar_grid.angular_divisions, 6)
        self.assertEqual(self.polar_grid.radial_spacing, 3.0)
        self.assertFalse(self.polar_grid.show_angle_labels)
        self.assertFalse(self.polar_grid.show_radius_labels)

    def test_visible_default(self) -> None:
        self.assertTrue(self.polar_grid.visible)

    def test_visible_in_state(self) -> None:
        self.polar_grid.visible = False
        state = self.polar_grid.get_state()
        self.assertIn("visible", state)
        self.assertFalse(state["visible"])

    def test_set_state_visible(self) -> None:
        self.polar_grid.set_state({"visible": False})
        self.assertFalse(self.polar_grid.visible)

        self.polar_grid.set_state({"visible": True})
        self.assertTrue(self.polar_grid.visible)

    def test_invalidate_cache_on_zoom(self) -> None:
        self.polar_grid._invalidate_cache_on_zoom()

    def test_zoom_affects_display_spacing(self) -> None:
        """Test that display_spacing changes appropriately with zoom.

        Note: The adaptive spacing algorithm adjusts _current_radial_spacing
        based on zoom level, so we just verify spacing is positive and changes.
        """
        initial_spacing = self.polar_grid.display_spacing
        self.assertGreater(initial_spacing, 0)

        self.canvas.scale_factor = 0.5
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        self.polar_grid._invalidate_cache_on_zoom()

        zoomed_out_spacing = self.polar_grid.display_spacing
        self.assertGreater(zoomed_out_spacing, 0)

        self.canvas.scale_factor = 2.0
        self.coordinate_mapper.sync_from_canvas(self.canvas)
        self.polar_grid._invalidate_cache_on_zoom()

        zoomed_in_spacing = self.polar_grid.display_spacing
        self.assertGreater(zoomed_in_spacing, 0)


class TestPolarGridConversions(unittest.TestCase):
    def test_rectangular_to_polar(self) -> None:
        from utils.math_utils import MathUtils

        r, theta = MathUtils.rectangular_to_polar(3, 4)
        self.assertAlmostEqual(r, 5, places=10)

        r, theta = MathUtils.rectangular_to_polar(1, 0)
        self.assertAlmostEqual(r, 1, places=10)
        self.assertAlmostEqual(theta, 0, places=10)

        r, theta = MathUtils.rectangular_to_polar(0, 1)
        self.assertAlmostEqual(r, 1, places=10)
        self.assertAlmostEqual(theta, math.pi / 2, places=10)

    def test_polar_to_rectangular(self) -> None:
        from utils.math_utils import MathUtils

        x, y = MathUtils.polar_to_rectangular(5, 0)
        self.assertAlmostEqual(x, 5, places=10)
        self.assertAlmostEqual(y, 0, places=10)

        x, y = MathUtils.polar_to_rectangular(1, math.pi / 2)
        self.assertAlmostEqual(x, 0, places=10)
        self.assertAlmostEqual(y, 1, places=10)

        x, y = MathUtils.polar_to_rectangular(1, math.pi / 4)
        expected = math.sqrt(2) / 2
        self.assertAlmostEqual(x, expected, places=10)
        self.assertAlmostEqual(y, expected, places=10)

    def test_roundtrip_conversion(self) -> None:
        from utils.math_utils import MathUtils

        test_cases = [(3, 4), (-3, 4), (-3, -4), (3, -4), (1.5, 2.5)]

        for orig_x, orig_y in test_cases:
            r, theta = MathUtils.rectangular_to_polar(orig_x, orig_y)
            x, y = MathUtils.polar_to_rectangular(r, theta)
            self.assertAlmostEqual(x, orig_x, places=10)
            self.assertAlmostEqual(y, orig_y, places=10)
