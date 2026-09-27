"""Regression tests for view undo (K4), the polar grid reset (K25) and truthful no-op updates (K26).

Every tool call runs through ``ProcessFunctionCalls.get_results_traced``, the same path a
model tool batch takes, against a real canvas.
"""

from __future__ import annotations

from typing import Dict

from constants import nothing_to_undo_message, successful_call_message

from .test_tool_batch_results import _ToolBatchTestCase

BOUND_TOLERANCE = 1e-9


class _ViewTestCase(_ToolBatchTestCase):
    """Helpers to read the view, the coordinate mode and the grids."""

    def bounds(self) -> Dict[str, float]:
        return dict(self.canvas.coordinate_mapper.get_visible_bounds())

    def assert_bounds(self, expected: Dict[str, float]) -> None:
        actual = self.bounds()
        for key in ("left", "right", "top", "bottom"):
            self.assertAlmostEqual(actual[key], expected[key], delta=BOUND_TOLERANCE, msg=key)

    def polar_spacing(self) -> float:
        return float(self.canvas.coordinate_system_manager.polar_grid.current_radial_spacing)

    def zoom_and_adapt_polar_grid(self, range_val: float) -> None:
        """Zoom as the model does; the polar grid adapts its spacing on the next frame, so adapt it here."""
        self.run_single("zoom", center_x=0, center_y=0, range_val=range_val, range_axis="x")
        self.canvas.coordinate_system_manager.polar_grid._invalidate_cache_on_zoom()

    def circle_name(self) -> str:
        return str(self.canvas.drawable_manager.drawables.Circles[0].name)


class TestViewUndo(_ViewTestCase):
    """K4: an undo entry captures and restores the view, the coordinate mode and grid visibility."""

    def test_undo_restores_the_view_after_a_zoom(self) -> None:
        before = self.bounds()

        self.run_single("zoom", center_x=0, center_y=0, range_val=2, range_axis="x")
        self.assertAlmostEqual(self.bounds()["left"], -2, delta=BOUND_TOLERANCE)
        self.assertEqual(self.undo_depth(), 1)

        self.assertEqual(self.run_single("undo"), successful_call_message)
        self.assert_bounds(before)

    def test_redo_reapplies_the_zoom(self) -> None:
        self.run_single("zoom", center_x=1, center_y=1, range_val=3, range_axis="y")
        zoomed = self.bounds()
        self.run_single("undo")

        self.run_single("redo")

        self.assert_bounds(zoomed)

    def test_undo_after_clear_restores_the_objects_and_the_view(self) -> None:
        self.build_triangle()
        self.run_single("zoom", center_x=0, center_y=0, range_val=10, range_axis="x")
        before_clear = self.snapshot()

        self.run_single("clear_canvas")
        self.assertEqual(self.snapshot(), {})

        self.run_single("undo")
        self.assertEqual(self.snapshot(), before_clear)
        self.assertAlmostEqual(self.bounds()["left"], -10, delta=BOUND_TOLERANCE)
        self.assertAlmostEqual(self.bounds()["right"], 10, delta=BOUND_TOLERANCE)

    def test_undo_restores_the_zoom_adapted_grid_spacing(self) -> None:
        cartesian = self.canvas.cartesian2axis
        default_tick = cartesian.current_tick_spacing
        default_polar = self.polar_spacing()
        self.zoom_and_adapt_polar_grid(0.5)
        self.assertNotEqual(cartesian.current_tick_spacing, default_tick)
        self.assertNotEqual(self.polar_spacing(), default_polar)

        self.run_single("undo")

        self.assertEqual(cartesian.current_tick_spacing, default_tick)
        self.assertEqual(self.polar_spacing(), default_polar)

    def test_coordinate_mode_and_grid_changes_are_one_undo_step(self) -> None:
        self.run_batch(
            ("set_coordinate_system", {"mode": "polar"}),
            ("set_grid_visible", {"visible": False}),
        )
        self.assertEqual(self.canvas.get_coordinate_system(), "polar")
        self.assertFalse(self.canvas.is_grid_visible())
        self.assertEqual(self.undo_depth(), 1)

        self.run_single("undo")

        self.assertEqual(self.canvas.get_coordinate_system(), "cartesian")
        self.assertTrue(self.canvas.coordinate_system_manager.polar_grid.visible)
        self.assertTrue(self.canvas.is_grid_visible())

        self.run_single("redo")
        self.assertEqual(self.canvas.get_coordinate_system(), "polar")
        self.assertFalse(self.canvas.is_grid_visible())

    def test_undo_of_a_batch_that_switched_mode_restores_the_mode(self) -> None:
        self.run_batch(("set_coordinate_system", {"mode": "polar"}), ("create_point", {"x": 0, "y": 2, "name": "P"}))

        self.run_single("undo")

        self.assertEqual(self.snapshot(), {})
        self.assertEqual(self.canvas.get_coordinate_system(), "cartesian")

    def test_mouse_pan_and_zoom_add_no_undo_entry(self) -> None:
        mapper = self.canvas.coordinate_mapper
        mapper.apply_pan(40, -25)
        mapper.apply_zoom_step(-1)

        self.assertEqual(self.undo_depth(), 0)
        self.assertEqual(self.run_single("undo"), nothing_to_undo_message)

    def test_undoing_a_step_that_kept_the_view_keeps_a_later_mouse_pan(self) -> None:
        self.run_single("create_point", x=1, y=1, name="A")
        self.canvas.coordinate_mapper.apply_pan(40, -25)
        panned = self.bounds()

        self.run_single("undo")

        self.assertEqual(self.snapshot(), {})
        self.assert_bounds(panned)

        self.run_single("redo")
        self.assertEqual(self.snapshot(), {"Point": ["A(1.0, 1.0)"]})
        self.assert_bounds(panned)

    def test_undoing_a_zoom_after_a_mouse_pan_restores_the_view_before_the_zoom(self) -> None:
        before = self.bounds()
        self.run_single("zoom", center_x=5, center_y=5, range_val=2, range_axis="x")
        self.canvas.coordinate_mapper.apply_pan(40, -25)

        self.run_single("undo")

        self.assert_bounds(before)

    def test_batch_that_only_zooms_is_judged_changed(self) -> None:
        self.canvas.begin_undo_batch()
        try:
            self.assertFalse(self.canvas.state_differs_from_undo_batch_baseline())
            self.canvas.zoom(0, 0, 3, "x")
            self.assertTrue(self.canvas.state_differs_from_undo_batch_baseline())
        finally:
            self.canvas.end_undo_batch()

    def test_undo_entries_without_a_view_still_restore(self) -> None:
        """Entries pushed by older code paths (no "view" key) restore the drawables only."""
        self.run_single("create_point", x=1, y=1, name="A")
        self.canvas.undo_redo_manager.undo_stack[-1].pop("view", None)
        self.run_single("zoom", center_x=0, center_y=0, range_val=2, range_axis="x")
        self.canvas.undo_redo_manager.undo_stack.pop()
        zoomed = self.bounds()

        self.run_single("undo")

        self.assertEqual(self.snapshot(), {})
        self.assert_bounds(zoomed)


class TestPolarGridReset(_ViewTestCase):
    """K25: clearing or resetting the canvas also resets the polar grid's zoom-adapted spacing."""

    def test_clear_after_zoom_resets_the_polar_spacing(self) -> None:
        default_spacing = self.polar_spacing()
        self.zoom_and_adapt_polar_grid(2)
        self.assertLess(self.polar_spacing(), default_spacing)

        self.run_batch(("clear_canvas", {}), ("set_coordinate_system", {"mode": "polar"}))

        self.assertEqual(self.polar_spacing(), default_spacing)

    def test_reset_canvas_resets_the_polar_spacing(self) -> None:
        default_spacing = self.polar_spacing()
        self.zoom_and_adapt_polar_grid(2)
        self.assertLess(self.polar_spacing(), default_spacing)

        self.run_single("reset_canvas")

        self.assertEqual(self.polar_spacing(), default_spacing)
