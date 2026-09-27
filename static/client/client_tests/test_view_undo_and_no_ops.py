"""Regression tests for view undo (K4), the polar grid reset (K25) and truthful no-op updates (K26).

Every tool call runs through ``ProcessFunctionCalls.get_results_traced``, the same path a
model tool batch takes, against a real canvas.
"""

from __future__ import annotations

from typing import Any, Dict

from constants import nothing_to_undo_message, successful_call_message

from .test_tool_batch_results import TRIANGLE_VERTICES, _ToolBatchTestCase

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

    def test_undoing_a_zoom_restores_zoom_and_pan_as_one_unit(self) -> None:
        """A pan after the zoom is undone with it: the view before the zoom comes back whole."""
        before = self.bounds()
        self.run_single("zoom", center_x=5, center_y=5, range_val=2, range_axis="x")
        self.canvas.coordinate_mapper.apply_pan(40, -25)

        self.run_single("undo")

        self.assert_bounds(before)

    def test_undoing_a_zoom_keeps_a_later_slash_command_mode_switch(self) -> None:
        before = self.bounds()
        self.run_single("zoom", center_x=5, center_y=5, range_val=2, range_axis="x")
        self.canvas.set_coordinate_system("polar")  # what /polar does: no undo entry

        self.run_single("undo")

        self.assert_bounds(before)
        self.assertEqual(self.canvas.get_coordinate_system(), "polar")

    def test_undoing_a_mode_switch_keeps_a_later_slash_command_grid_toggle(self) -> None:
        self.run_single("set_coordinate_system", mode="polar")
        self.canvas.set_grid_visible(False)  # what /grid does: hides the polar grid, no undo entry

        self.run_single("undo")

        manager = self.canvas.coordinate_system_manager
        self.assertEqual(self.canvas.get_coordinate_system(), "cartesian")
        self.assertFalse(manager.polar_grid.visible)
        self.assertTrue(manager.cartesian_grid.visible)

        self.run_single("redo")
        self.assertEqual(self.canvas.get_coordinate_system(), "polar")
        self.assertFalse(self.canvas.is_grid_visible())

    def test_undoing_a_zoom_keeps_the_objects_restore_and_a_later_grid_toggle(self) -> None:
        self.run_batch(
            ("create_point", {"x": 1, "y": 1, "name": "A"}),
            ("zoom", {"center_x": 0, "center_y": 0, "range_val": 3, "range_axis": "x"}),
        )
        self.canvas.set_grid_visible(False)

        self.run_single("undo")

        self.assertEqual(self.snapshot(), {})
        self.assertFalse(self.canvas.is_grid_visible())

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


class TestNoOpUpdates(_ViewTestCase):
    """K26: a change already in effect says so and adds no undo entry."""

    def assert_no_op(self, tool: str, args: Dict[str, Any], *words: str) -> str:
        depth = self.undo_depth()
        _, traced = self.run_batch((tool, args))
        result = traced[0]["result"]
        self.assertNotEqual(result, successful_call_message)
        self.assertFalse(traced[0]["is_error"], str(result))
        self.assertIn("nothing changed", str(result))
        for word in words:
            self.assertIn(word, str(result))
        self.assertEqual(self.undo_depth(), depth)
        return str(result)

    def assert_changed(self, tool: str, args: Dict[str, Any]) -> None:
        depth = self.undo_depth()
        result = self.run_single(tool, **args)
        self.assertEqual(result, successful_call_message)
        self.assertEqual(self.undo_depth(), depth + 1)

    def test_update_circle_to_its_own_colour_is_a_no_op(self) -> None:
        self.run_single("create_circle", center_x=0, center_y=0, radius=2, color="red")
        name = self.circle_name()

        result = self.assert_no_op(
            "update_circle", {"name": name, "new_color": "red", "new_center_x": None, "new_center_y": None}
        )

        self.assertEqual(result, f"Circle '{name}' already has color red; nothing changed.")

    def test_colour_comparison_ignores_case_and_spaces(self) -> None:
        self.run_single("create_circle", center_x=0, center_y=0, radius=2, color="red")

        self.assert_no_op("update_circle", {"name": self.circle_name(), "new_color": " Red "}, "already")

    def test_update_circle_to_a_new_colour_still_changes_it(self) -> None:
        self.run_single("create_circle", center_x=0, center_y=0, radius=2, color="red")

        self.assert_changed("update_circle", {"name": self.circle_name(), "new_color": "blue"})

        self.assertEqual(self.canvas.drawable_manager.drawables.Circles[0].color, "blue")

    def test_same_colour_with_a_new_center_is_a_change(self) -> None:
        self.run_single("create_circle", center_x=0, center_y=0, radius=2, color="red")

        self.assert_changed(
            "update_circle", {"name": self.circle_name(), "new_color": "red", "new_center_x": 1, "new_center_y": 1}
        )

    def test_update_circle_to_its_own_center_is_a_no_op(self) -> None:
        self.run_single("create_circle", center_x=1, center_y=2, radius=2)

        self.assert_no_op(
            "update_circle", {"name": self.circle_name(), "new_center_x": 1, "new_center_y": 2}, "center (1, 2)"
        )

    def test_update_point_with_its_own_values_is_a_no_op(self) -> None:
        self.run_single("create_point", x=1, y=2, name="A", color="green")

        result = self.assert_no_op(
            "update_point", {"point_name": "A", "new_name": "A", "new_x": 1, "new_y": 2, "new_color": "green"}
        )

        self.assertEqual(result, "Point 'A' already has name A, position (1, 2) and color green; nothing changed.")

    def test_update_point_to_a_new_position_still_moves_it(self) -> None:
        self.run_single("create_point", x=1, y=2, name="A")

        self.assert_changed("update_point", {"point_name": "A", "new_x": 3, "new_y": 2})

    def test_update_segment_vector_and_angle_colour_no_ops(self) -> None:
        self.run_single("create_segment", x1=0, y1=0, x2=4, y2=0, name="AB", color="blue")
        self.assert_no_op("update_segment", {"name": "AB", "new_color": "blue"}, "Segment 'AB'")

        self.run_single("create_vector", origin_x=0, origin_y=1, tip_x=3, tip_y=1, name="CD", color="orange")
        self.assert_no_op("update_vector", {"name": "CD", "new_color": "orange"}, "Vector 'CD'")

        self.run_single("create_angle", vx=0, vy=0, p1x=4, p1y=0, p2x=0, p2y=4, color="purple")
        angle_name = self.snapshot()["Angle"][0]
        self.assert_no_op("update_angle", {"name": angle_name, "new_color": "purple"}, "Angle")

    def test_polygon_colour_is_a_no_op_only_when_every_edge_has_it(self) -> None:
        self.run_single("create_polygon", vertices=TRIANGLE_VERTICES, polygon_type="triangle", name="ABC", color="red")
        self.assert_no_op(
            "update_polygon", {"polygon_name": "ABC", "polygon_type": "triangle", "new_color": "red"}, "Polygon 'ABC'"
        )

        self.run_single("update_segment", name="AB", new_color="blue")

        self.assert_changed("update_polygon", {"polygon_name": "ABC", "new_color": "red"})
        colors = {segment.color for segment in self.canvas.drawable_manager.drawables.Segments}
        self.assertEqual(colors, {"red"})

    def test_function_and_label_colour_no_ops(self) -> None:
        self.run_single("draw_function", function_string="x^2", name="f", color="red")
        self.assert_no_op("update_function", {"name": "f", "new_color": "red"}, "Function 'f'")

        self.run_single("create_label", x=1, y=1, text="hi", name="L1", color="black")
        self.assert_no_op("update_label", {"name": "L1", "new_color": "black"}, "Label 'L1'")

    def test_fields_without_a_no_op_check_still_run_the_update(self) -> None:
        self.run_single("draw_function", function_string="x^2", name="f", color="red")

        self.assert_changed("update_function", {"name": "f", "new_color": "red", "new_left_bound": -1})

    def test_update_of_a_missing_object_is_still_an_error(self) -> None:
        _, traced = self.run_batch(("update_circle", {"name": "nope", "new_color": "red"}))

        self.assertTrue(traced[0]["is_error"])

    def test_zoom_to_the_current_view_is_a_no_op(self) -> None:
        self.run_single("zoom", center_x=0, center_y=0, range_val=2, range_axis="x")

        result = self.assert_no_op("zoom", {"center_x": 0, "center_y": 0, "range_val": 2, "range_axis": "x"})

        self.assertIn("already", result)

    def test_repeated_zoom_far_from_the_origin_is_a_no_op(self) -> None:
        args = {"center_x": 1e6, "center_y": 1e6, "range_val": 1e-3, "range_axis": "x"}
        self.run_single("zoom", **args)

        result = self.assert_no_op("zoom", args)

        self.assertIn("already", result)

    def test_reset_or_clear_of_a_canvas_already_reset_adds_no_entry(self) -> None:
        self.assert_no_op_batch(("reset_canvas", {}))
        self.assert_no_op_batch(("clear_canvas", {}))

        self.run_single("zoom", center_x=0, center_y=0, range_val=2, range_axis="x")
        depth = self.undo_depth()
        self.run_single("reset_canvas")
        self.run_single("reset_canvas")
        self.assertEqual(self.undo_depth(), depth + 1)

    def test_batch_that_changes_and_reverts_adds_no_entry(self) -> None:
        self.run_single("zoom", center_x=0, center_y=0, range_val=2, range_axis="x")

        self.assert_no_op_batch(
            ("zoom", {"center_x": 3, "center_y": 3, "range_val": 5, "range_axis": "x"}),
            ("zoom", {"center_x": 0, "center_y": 0, "range_val": 2, "range_axis": "x"}),
        )
        self.assert_no_op_batch(
            ("set_coordinate_system", {"mode": "polar"}),
            ("set_coordinate_system", {"mode": "cartesian"}),
        )
        self.assert_no_op_batch(
            ("create_point", {"x": 4, "y": 4, "name": "Q"}),
            ("delete_point", {"x": 4, "y": 4}),
        )

    def assert_no_op_batch(self, *calls: Any) -> None:
        """The batch adds no undo entry and keeps a pending redo."""
        self.run_single("create_point", x=9, y=9, name="R")
        self.run_single("undo")
        depth, redo_depth = self.undo_depth(), len(self.canvas.undo_redo_manager.redo_stack)

        self.run_batch(*calls)

        self.assertEqual(self.undo_depth(), depth)
        self.assertEqual(len(self.canvas.undo_redo_manager.redo_stack), redo_depth)

    def test_zoom_to_a_new_view_still_adds_an_entry(self) -> None:
        self.run_single("zoom", center_x=0, center_y=0, range_val=2, range_axis="x")

        self.assert_changed("zoom", {"center_x": 0, "center_y": 0, "range_val": 3, "range_axis": "x"})

    def test_setting_the_current_mode_or_grid_visibility_is_a_no_op(self) -> None:
        self.assert_no_op("set_coordinate_system", {"mode": "cartesian"}, "already")
        self.assert_no_op("set_grid_visible", {"visible": True}, "already")

    def test_colour_no_op_after_a_real_change_in_the_same_batch_keeps_the_entry(self) -> None:
        self.run_single("create_circle", center_x=0, center_y=0, radius=2, color="red")
        name = self.circle_name()
        depth = self.undo_depth()

        self.run_batch(
            ("update_circle", {"name": name, "new_color": "blue"}),
            ("update_circle", {"name": name, "new_color": "blue"}),
        )

        self.assertEqual(self.undo_depth(), depth + 1)
        self.run_single("undo")
        self.assertEqual(self.canvas.drawable_manager.drawables.Circles[0].color, "red")
