import unittest
from drawables_aggregator import Position
from managers.coordinate_system_manager import CoordinateSystemManager
from cartesian_system_2axis import Cartesian2Axis
from polar_grid import PolarGrid
from .simple_mock import SimpleMock
from coordinate_mapper import CoordinateMapper


class TestCoordinateSystemManager(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinate_mapper = CoordinateMapper(800, 600)

        self.cartesian_grid = Cartesian2Axis(coordinate_mapper=self.coordinate_mapper)

        draw_called = [False]

        def mock_draw():
            draw_called[0] = True

        self.canvas = SimpleMock(
            width=800,
            height=600,
            scale_factor=1,
            center=Position(400, 300),
            cartesian2axis=self.cartesian_grid,
            coordinate_mapper=self.coordinate_mapper,
            zoom_direction=0,
            offset=Position(0, 0),
            zoom_point=Position(0, 0),
            zoom_step=0.1,
            draw=mock_draw,
            _draw_called=draw_called,
        )

        self.coordinate_mapper.sync_from_canvas(self.canvas)
        self.manager = CoordinateSystemManager(self.canvas)

    def test_default_mode_is_cartesian(self) -> None:
        self.assertEqual(self.manager.mode, "cartesian")

    def test_cartesian_grid_reference(self) -> None:
        self.assertEqual(self.manager.cartesian_grid, self.cartesian_grid)

    def test_polar_grid_created(self) -> None:
        self.assertIsNotNone(self.manager.polar_grid)
        self.assertEqual(self.manager.polar_grid.get_class_name(), "PolarGrid")

    def test_set_mode_to_polar(self) -> None:
        self.manager.set_mode("polar")
        self.assertEqual(self.manager.mode, "polar")

    def test_set_mode_to_cartesian(self) -> None:
        self.manager.set_mode("polar")
        self.manager.set_mode("cartesian")
        self.assertEqual(self.manager.mode, "cartesian")

    def test_set_mode_triggers_redraw(self) -> None:
        self.canvas._draw_called[0] = False
        self.manager.set_mode("polar")
        self.assertTrue(self.canvas._draw_called[0])

    def test_set_mode_invalid_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.set_mode("invalid_mode")

    def test_get_active_grid_cartesian(self) -> None:
        active_grid = self.manager.get_active_grid()
        self.assertEqual(active_grid, self.manager.cartesian_grid)

    def test_get_active_grid_polar(self) -> None:
        self.manager.set_mode("polar")
        active_grid = self.manager.get_active_grid()
        self.assertEqual(active_grid, self.manager.polar_grid)

    def test_active_grid_changes_with_mode(self) -> None:
        grid1 = self.manager.get_active_grid()
        self.assertEqual(grid1.get_class_name(), "Cartesian2Axis")

        self.manager.set_mode("polar")
        grid2 = self.manager.get_active_grid()
        self.assertEqual(grid2.get_class_name(), "PolarGrid")

        self.manager.set_mode("cartesian")
        grid3 = self.manager.get_active_grid()
        self.assertEqual(grid3.get_class_name(), "Cartesian2Axis")

    def test_get_state_cartesian(self) -> None:
        state = self.manager.get_state()
        self.assertIn("mode", state)
        self.assertEqual(state["mode"], "cartesian")

    def test_get_state_polar(self) -> None:
        self.manager.set_mode("polar")
        state = self.manager.get_state()
        self.assertEqual(state["mode"], "polar")

    def test_set_state_cartesian(self) -> None:
        self.manager.set_mode("polar")
        self.manager.set_state({"mode": "cartesian"})
        self.assertEqual(self.manager.mode, "cartesian")

    def test_set_state_polar(self) -> None:
        self.manager.set_state({"mode": "polar"})
        self.assertEqual(self.manager.mode, "polar")

    def test_set_state_empty_defaults_to_cartesian(self) -> None:
        self.manager.set_mode("polar")
        self.manager.set_state({})
        self.assertEqual(self.manager.mode, "cartesian")

    def test_state_roundtrip(self) -> None:
        self.manager.set_mode("polar")
        state = self.manager.get_state()

        new_manager = CoordinateSystemManager(self.canvas)
        new_manager.set_state(state)

        self.assertEqual(self.manager.mode, new_manager.mode)

    def test_mode_switch_preserves_grid_instances(self) -> None:
        cartesian_id = id(self.manager.cartesian_grid)
        polar_id = id(self.manager.polar_grid)

        self.manager.set_mode("polar")
        self.manager.set_mode("cartesian")
        self.manager.set_mode("polar")

        self.assertEqual(id(self.manager.cartesian_grid), cartesian_id)
        self.assertEqual(id(self.manager.polar_grid), polar_id)

    def test_polar_grid_uses_canvas_mapper(self) -> None:
        self.assertEqual(self.manager.polar_grid.coordinate_mapper, self.coordinate_mapper)

    def test_is_grid_visible_default(self) -> None:
        self.assertTrue(self.manager.is_grid_visible())

    def test_set_grid_visible_false(self) -> None:
        self.manager.set_grid_visible(False)
        self.assertFalse(self.manager.is_grid_visible())

    def test_set_grid_visible_true(self) -> None:
        self.manager.set_grid_visible(False)
        self.manager.set_grid_visible(True)
        self.assertTrue(self.manager.is_grid_visible())

    def test_visibility_separate_per_mode(self) -> None:
        self.manager.set_grid_visible(False)
        self.assertFalse(self.manager.cartesian_grid.visible)
        self.assertTrue(self.manager.polar_grid.visible)

        self.manager.set_mode("polar")
        self.assertTrue(self.manager.is_grid_visible())

        self.manager.set_grid_visible(False)
        self.assertFalse(self.manager.polar_grid.visible)
        self.assertFalse(self.manager.cartesian_grid.visible)

    def test_set_grid_visible_triggers_redraw(self) -> None:
        self.canvas._draw_called[0] = False
        self.manager.set_grid_visible(False)
        self.assertTrue(self.canvas._draw_called[0])


class TestCoordinateSystemManagerCanvasIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinate_mapper = CoordinateMapper(800, 600)

        self.cartesian_grid = Cartesian2Axis(coordinate_mapper=self.coordinate_mapper)

        draw_called = [False]

        def mock_draw():
            draw_called[0] = True

        self.canvas = SimpleMock(
            width=800,
            height=600,
            scale_factor=1,
            center=Position(400, 300),
            cartesian2axis=self.cartesian_grid,
            coordinate_mapper=self.coordinate_mapper,
            zoom_direction=0,
            offset=Position(0, 0),
            zoom_point=Position(0, 0),
            zoom_step=0.1,
            draw=mock_draw,
            _draw_called=draw_called,
        )

        self.coordinate_mapper.sync_from_canvas(self.canvas)
        self.manager = CoordinateSystemManager(self.canvas)

    def test_zoom_affects_both_grids(self) -> None:
        self.canvas.scale_factor = 2.0
        self.coordinate_mapper.sync_from_canvas(self.canvas)

        self.manager.cartesian_grid._invalidate_cache_on_zoom()
        self.manager.polar_grid._invalidate_cache_on_zoom()

    def test_pan_affects_origin(self) -> None:
        self.canvas.offset = Position(100, -50)
        self.coordinate_mapper.sync_from_canvas(self.canvas)

        ox, oy = self.manager.polar_grid.origin_screen
        self.assertAlmostEqual(ox, 500, places=6)
        self.assertAlmostEqual(oy, 250, places=6)
