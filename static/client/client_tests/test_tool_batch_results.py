"""Regression tests for model tool batches: undo granularity and truthful results.

Every test runs tool calls through ``ProcessFunctionCalls.get_results_traced``, the
same path a model tool batch takes, against a real canvas.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Tuple

from canvas import Canvas
from constants import successful_call_message
from function_registry import FunctionRegistry
from process_function_calls import ProcessFunctionCalls
from tool_call_log_manager import ToolCallLogManager
from turn_metrics import aggregate_turn
from workspace_manager import WorkspaceManager

TRIANGLE_VERTICES: List[Dict[str, float]] = [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}]
TRIANGLE_VERTICES_ISOSCELES: List[Dict[str, float]] = [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 2, "y": 3}]


class _ToolBatchTestCase(unittest.TestCase):
    """Shared setup: a real canvas with the model's function registry."""

    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.workspace_manager = WorkspaceManager(self.canvas)
        self.available_functions: Dict[str, Any] = FunctionRegistry.get_available_functions(
            self.canvas, self.workspace_manager
        )
        self.undoable_functions: Tuple[str, ...] = FunctionRegistry.get_undoable_functions()

    def run_batch(self, *calls: Tuple[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run the calls as one model tool batch; return (results, traced calls)."""
        batch = [{"function_name": name, "arguments": dict(args)} for name, args in calls]
        return ProcessFunctionCalls.get_results_traced(
            batch, self.available_functions, self.undoable_functions, self.canvas
        )

    def run_single(self, tool_name: str, **args: Any) -> Any:
        """Run one call as its own batch and return its result value."""
        _, traced = self.run_batch((tool_name, args))
        return traced[0]["result"]

    def undo_depth(self) -> int:
        return len(self.canvas.undo_redo_manager.undo_stack)

    def snapshot(self) -> Dict[str, List[str]]:
        """Names of every drawable, by bucket, plus point coordinates."""
        buckets: Dict[str, List[str]] = {}
        for bucket, drawables in self.canvas.drawable_manager.drawables._drawables.items():
            if not drawables:
                continue
            entries = []
            for drawable in drawables:
                entry = str(getattr(drawable, "name", ""))
                if bucket == "Point":
                    entry += f"({float(drawable.x)}, {float(drawable.y)})"
                entries.append(entry)
            buckets[bucket] = sorted(entries)
        return buckets

    def build_triangle(self) -> None:
        self.run_single("create_polygon", vertices=TRIANGLE_VERTICES, polygon_type="triangle", name="ABC")


class TestToolBatchUndo(_ToolBatchTestCase):
    """K1: one model tool batch is one undo step."""

    def test_batch_of_three_segments_adds_one_undo_entry(self) -> None:
        self.run_batch(
            ("create_segment", {"x1": 0, "y1": 0, "x2": 4, "y2": 0}),
            ("create_segment", {"x1": 4, "y1": 0, "x2": 0, "y2": 3}),
            ("create_segment", {"x1": 0, "y1": 3, "x2": 0, "y2": 0}),
        )

        self.assertEqual(self.undo_depth(), 1)
        self.assertEqual(len(self.snapshot().get("Segment", [])), 3)

    def test_one_undo_restores_the_state_before_the_batch(self) -> None:
        self.run_batch(
            ("create_segment", {"x1": 0, "y1": 0, "x2": 4, "y2": 0}),
            ("create_segment", {"x1": 4, "y1": 0, "x2": 0, "y2": 3}),
        )

        self.run_single("undo")

        self.assertEqual(self.snapshot(), {})

    def test_redo_restores_the_whole_batch(self) -> None:
        self.run_batch(
            ("create_polygon", {"vertices": TRIANGLE_VERTICES, "polygon_type": "triangle", "name": "ABC"}),
            ("construct_circumcircle", {"triangle_name": "ABC"}),
        )
        after_batch = self.snapshot()

        self.run_single("undo")
        self.assertEqual(self.snapshot(), {})

        self.run_single("redo")
        self.assertEqual(self.snapshot(), after_batch)
        self.assertEqual(self.undo_depth(), 1)

    def test_undo_after_point_splitting_a_segment_is_one_step(self) -> None:
        self.run_batch(
            ("create_segment", {"x1": 0, "y1": 0, "x2": 4, "y2": 0}),
            ("create_segment", {"x1": 4, "y1": 0, "x2": 0, "y2": 3}),
            ("create_segment", {"x1": 0, "y1": 3, "x2": 0, "y2": 0}),
        )
        before = self.snapshot()

        self.run_single("create_point", x=2, y=0, name="M")
        self.assertNotEqual(self.snapshot(), before)

        self.run_single("undo")
        self.assertEqual(self.snapshot(), before)

    def test_undo_after_cascading_delete_restores_everything(self) -> None:
        self.build_triangle()
        self.run_single("create_segment", x1=4, y1=0, x2=6, y2=2)
        before = self.snapshot()

        self.run_single("delete_point", x=4, y=0)
        self.assertNotIn("Triangle", self.snapshot())

        self.run_single("undo")
        self.assertEqual(self.snapshot(), before)

    def test_batch_with_only_a_no_op_adds_no_undo_entry(self) -> None:
        self.run_single("create_circle", center_x=0, center_y=0, radius=2)
        depth = self.undo_depth()

        self.run_single("delete_circle", name="nope")

        self.assertEqual(self.undo_depth(), depth)

    def test_undo_then_create_in_one_batch_undoes_the_previous_batch(self) -> None:
        self.run_single("create_point", x=1, y=1, name="A")

        self.run_batch(("undo", {}), ("create_point", {"x": 2, "y": 2, "name": "B"}))
        self.assertEqual(self.snapshot(), {"Point": ["B(2.0, 2.0)"]})

        self.run_single("undo")
        self.assertEqual(self.snapshot(), {})

    def test_create_then_undo_in_one_batch_reverts_the_create(self) -> None:
        self.run_batch(("create_point", {"x": 1, "y": 1, "name": "A"}), ("undo", {}))
        self.assertEqual(self.snapshot(), {})

        self.run_single("redo")
        self.assertEqual(self.snapshot(), {"Point": ["A(1.0, 1.0)"]})

    def test_interactive_operations_still_archive_each_change(self) -> None:
        self.canvas.create_point(1, 1, name="A")
        self.canvas.create_point(2, 2, name="B")

        self.assertEqual(self.undo_depth(), 2)
        self.canvas.undo()
        self.assertEqual(self.snapshot(), {"Point": ["A(1.0, 1.0)"]})

    def test_failing_call_does_not_leave_archiving_suspended(self) -> None:
        def explode(**_: Any) -> None:
            raise ValueError("boom")

        self.available_functions["explode"] = explode
        self.undoable_functions = self.undoable_functions + ("explode",)

        results, traced = self.run_batch(("create_point", {"x": 1, "y": 1}), ("explode", {}))

        self.assertTrue(traced[1]["is_error"])
        self.assertEqual(self.undo_depth(), 1)
        self.canvas.create_point(5, 5)
        self.assertEqual(self.undo_depth(), 2)

    def test_workspace_restore_is_one_undo_step(self) -> None:
        self.build_triangle()
        saved = json.loads(json.dumps(self.workspace_manager._snapshot_persistable_canvas_state()))
        self.run_batch(("clear_canvas", {}), ("create_point", {"x": 9, "y": 9, "name": "Z"}))
        before_load = self.snapshot()
        depth = self.undo_depth()

        self.workspace_manager._restore_workspace_state(saved)

        self.assertIn("Triangle", self.snapshot())
        self.assertEqual(self.undo_depth(), depth + 1)
        self.canvas.undo()
        self.assertEqual(self.snapshot(), before_load)

    def redo_depth(self) -> int:
        return len(self.canvas.undo_redo_manager.redo_stack)

    def create_angle_and_pending_redo(self) -> str:
        """Create an angle, then undo a later point so a redo is pending; return the angle's name."""
        self.run_single("create_angle", vx=0, vy=0, p1x=4, p1y=0, p2x=0, p2y=4)
        angle_name = self.snapshot()["Angle"][0]
        self.run_single("create_point", x=9, y=9, name="Q")
        self.run_single("undo")
        self.assertEqual(self.redo_depth(), 1)
        return angle_name

    def test_missing_angle_delete_adds_no_entry_and_keeps_redo(self) -> None:
        self.create_angle_and_pending_redo()
        depth = self.undo_depth()

        result = self.run_single("delete_angle", name="nope")

        self.assertTrue(str(result).startswith("Error:"))
        self.assertEqual(self.undo_depth(), depth)
        self.assertEqual(self.redo_depth(), 1)

    def test_failing_translate_adds_no_entry_and_keeps_redo(self) -> None:
        angle_name = self.create_angle_and_pending_redo()
        depth = self.undo_depth()
        before = self.snapshot()

        _, traced = self.run_batch(("translate_object", {"name": angle_name, "x_offset": 1, "y_offset": 0}))

        self.assertTrue(traced[0]["is_error"])
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.undo_depth(), depth)
        self.assertEqual(self.redo_depth(), 1)
        self.run_single("redo")
        self.assertIn("Q(9.0, 9.0)", self.snapshot()["Point"])

    def assert_partial_change_is_one_undo_step(
        self, before: Dict[str, List[str]], traced: List[Dict[str, Any]]
    ) -> None:
        """The failed call left a change behind; it is one undo entry that undo and redo move across."""
        self.assertTrue(traced[-1]["is_error"])
        partial = self.snapshot()
        self.assertNotEqual(partial, before)
        self.assertEqual(self.undo_depth(), 1)

        self.run_single("undo")
        self.assertEqual(self.snapshot(), before)
        self.run_single("redo")
        self.assertEqual(self.snapshot(), partial)

    def assert_no_op_batch_after_transform_adds_no_entry(
        self, polygon_type: str, vertices: List[Dict[str, float]], transform: Tuple[str, Dict[str, Any]]
    ) -> None:
        """A transform that changes a polygon's classification, then a batch of one no-op call.

        The live polygon keeps its creation-time types while a deep copy recomputes them, so
        the no-op batch must be compared against the live objects, not the copied baseline.
        No redo can be pending here: an undo or redo replaces the live objects with copies.
        """
        self.run_single("create_polygon", vertices=vertices, polygon_type=polygon_type)
        before_transform = self.snapshot()
        name = before_transform[polygon_type.capitalize()][0]
        transform_name, transform_args = transform
        self.assertEqual(self.run_single(transform_name, name=name, **transform_args), successful_call_message)
        after_transform = self.snapshot()
        depth = self.undo_depth()
        redo_depth = self.redo_depth()

        self.run_single("delete_angle", name="nope")

        self.assertEqual(self.undo_depth(), depth)
        self.assertEqual(self.redo_depth(), redo_depth)
        self.assertEqual(self.snapshot(), after_transform)
        self.run_single("undo")
        self.assertEqual(self.snapshot(), before_transform)

    def test_no_op_batch_after_scaling_a_square_adds_no_entry(self) -> None:
        self.assert_no_op_batch_after_transform_adds_no_entry(
            "rectangle",
            [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 4}, {"x": 0, "y": 4}],
            ("scale_object", {"sx": 2, "sy": 1, "cx": 0, "cy": 0}),
        )

    def test_no_op_batch_after_shearing_an_isosceles_triangle_adds_no_entry(self) -> None:
        self.assert_no_op_batch_after_transform_adds_no_entry(
            "triangle",
            TRIANGLE_VERTICES_ISOSCELES,
            ("shear_object", {"axis": "horizontal", "factor": 1, "cx": 0, "cy": 0}),
        )

    def test_polygon_that_fails_after_creating_parts_stays_undoable(self) -> None:
        self.canvas.create_point(9, 9, name="Z")
        self.canvas.undo_redo_manager.clear()
        before = self.snapshot()

        _, traced = self.run_batch(
            ("create_polygon", {"vertices": [[0, 0], [0, 0], [2, 3]], "polygon_type": "triangle"})
        )

        self.assert_partial_change_is_one_undo_step(before, traced)

    def test_distribution_that_fails_after_drawing_its_curve_stays_undoable(self) -> None:
        before = self.snapshot()

        _, traced = self.run_batch(
            (
                "plot_distribution",
                {
                    "name": "P",
                    "representation": "continuous",
                    "distribution_type": "normal",
                    "distribution_params": {"mean": 0.0, "sigma": 1.0},
                    "plot_bounds": {"left_bound": -4.0, "right_bound": 4.0},
                    "shade_bounds": {"left_bound": 5.0, "right_bound": 6.0},
                    "curve_color": None,
                    "fill_color": None,
                    "fill_opacity": None,
                    "bar_count": None,
                },
            )
        )

        self.assert_partial_change_is_one_undo_step(before, traced)

    def test_workspace_restore_that_fails_partway_stays_undoable(self) -> None:
        self.build_triangle()
        saved = json.loads(json.dumps(self.workspace_manager._snapshot_persistable_canvas_state()))
        self.run_batch(("clear_canvas", {}), ("create_point", {"x": 9, "y": 9, "name": "Z"}))
        before_load = self.snapshot()
        depth = self.undo_depth()

        def fail_late(_: Dict[str, Any]) -> None:
            raise ValueError("restore failed")

        self.workspace_manager._restore_computations = fail_late  # type: ignore[method-assign]
        self.available_functions["load_broken"] = lambda: self.workspace_manager._restore_workspace_state(saved)

        _, traced = self.run_batch(("delete_angle", {"name": "nope"}), ("load_broken", {}))

        self.assertTrue(traced[1]["is_error"])
        self.assertIn("Triangle", self.snapshot())
        self.assertEqual(self.undo_depth(), depth + 1)
        self.run_single("undo")
        self.assertEqual(self.snapshot(), before_load)

    def serve_triangle_workspace(self) -> None:
        """Make the load_workspace tool load a triangle without any workspace file on the server.

        The load still builds, sends and finalizes a real synchronous Ajax request (where a
        load used to run its handler twice), but to the read-only /list_workspaces route, and
        the saved triangle state stands in for the response's state. Nothing is written to
        the server's workspaces folder. The canvas is left holding only point Z.
        """
        self.build_triangle()
        saved = json.loads(json.dumps(self.workspace_manager._snapshot_persistable_canvas_state()))
        send_real_request = WorkspaceManager._open_and_send_sync_request
        state_from_response = WorkspaceManager._workspace_state_from_response

        def send_read_only_request(manager: WorkspaceManager, req: Any, method: str, url: str) -> None:
            send_real_request(manager, req, "GET", "/list_workspaces")

        def served_state(manager: WorkspaceManager, response: Dict[str, Any]) -> Dict[str, Any]:
            return saved

        # Patched on the class (restored after the test): Brython resolves self-method calls
        # inside WorkspaceManager on the class, so instance attributes would not take effect.
        WorkspaceManager._open_and_send_sync_request = send_read_only_request  # type: ignore[method-assign]
        WorkspaceManager._workspace_state_from_response = served_state  # type: ignore[method-assign,assignment]
        self.addCleanup(setattr, WorkspaceManager, "_open_and_send_sync_request", send_real_request)
        self.addCleanup(setattr, WorkspaceManager, "_workspace_state_from_response", state_from_response)
        self.run_batch(("clear_canvas", {}), ("create_point", {"x": 9, "y": 9, "name": "Z"}))

    def load_triangle(self, *calls: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run the batch ending in load_workspace and check that the load succeeded."""
        _, traced = self.run_batch(*calls, ("load_workspace", {"name": "served"}))
        load = traced[-1]
        self.assertIn("loaded successfully", str(load["result"]))
        return load

    def test_sync_workspace_request_runs_its_handler_once(self) -> None:
        calls: List[Any] = []

        def on_complete(req: Any) -> str:
            calls.append(req)
            return "done"

        result = self.workspace_manager._execute_sync_request(
            method="GET", url="/list_workspaces", on_complete=on_complete, error_prefix="Error listing workspaces"
        )

        self.assertEqual(result, "done")
        self.assertEqual(len(calls), 1)

    def test_tool_load_workspace_is_one_undo_step(self) -> None:
        self.serve_triangle_workspace()
        before_load = self.snapshot()
        depth = self.undo_depth()

        load = self.load_triangle()

        self.assertIn("loaded successfully", str(load["result"]))
        self.assertIn("Triangle", self.snapshot())
        self.assertEqual(self.undo_depth(), depth + 1)
        self.run_single("undo")
        self.assertEqual(self.snapshot(), before_load)

    def test_tool_load_workspace_in_an_undoable_batch_is_one_undo_step(self) -> None:
        self.serve_triangle_workspace()
        before_batch = self.snapshot()
        depth = self.undo_depth()

        self.load_triangle(("delete_angle", {"name": "nope"}))

        self.assertIn("Triangle", self.snapshot())
        self.assertEqual(self.undo_depth(), depth + 1)
        self.run_single("undo")
        self.assertEqual(self.snapshot(), before_batch)

    def test_fit_regression_is_one_undo_step(self) -> None:
        self.run_single("create_point", x=9, y=9, name="Z")
        before = self.snapshot()
        depth = self.undo_depth()

        _, traced = self.run_batch(
            (
                "fit_regression",
                {
                    "name": "fit1",
                    "x_data": [1, 2, 3, 4],
                    "y_data": [3, 5, 7, 9],
                    "model_type": "linear",
                    "degree": None,
                    "plot_bounds": None,
                    "curve_color": None,
                    "show_points": True,
                    "point_color": None,
                },
            )
        )

        self.assertFalse(traced[0]["is_error"])
        self.assertIn("r_squared", traced[0]["result"])
        self.assertEqual(self.undo_depth(), depth + 1)
        self.run_single("undo")
        self.assertEqual(self.snapshot(), before)

    def test_failed_call_keeps_the_entry_of_an_earlier_change_in_the_batch(self) -> None:
        self.run_batch(("create_point", {"x": 1, "y": 1, "name": "A"}), ("delete_angle", {"name": "nope"}))

        self.assertEqual(self.undo_depth(), 1)
        self.run_single("undo")
        self.assertEqual(self.snapshot(), {})

    def test_change_after_a_failed_call_in_the_batch_still_adds_an_entry(self) -> None:
        self.run_batch(("delete_angle", {"name": "nope"}), ("create_point", {"x": 1, "y": 1, "name": "A"}))

        self.assertEqual(self.undo_depth(), 1)
        self.run_single("undo")
        self.assertEqual(self.snapshot(), {})


class TestToolNoOpResults(_ToolBatchTestCase):
    """K2: tool results do not claim success when nothing happened."""

    def test_deleting_a_missing_circle_is_not_reported_as_success(self) -> None:
        self.run_single("create_circle", center_x=0, center_y=0, radius=2)

        _, traced = self.run_batch(("delete_circle", {"name": "nope"}))

        result = traced[0]["result"]
        self.assertNotEqual(result, successful_call_message)
        self.assertTrue(str(result).startswith("Error:"))
        self.assertTrue(traced[0]["is_error"])
        self.assertEqual(len(self.snapshot().get("Circle", [])), 1)

    def test_deleting_a_point_at_an_empty_spot_is_not_reported_as_success(self) -> None:
        result = self.run_single("delete_point", x=7, y=7)

        self.assertTrue(str(result).startswith("Error:"))

    def test_deleting_an_existing_circle_still_reports_success(self) -> None:
        circle = self.canvas.create_circle(0, 0, 2)

        result = self.run_single("delete_circle", name=circle.name)

        self.assertEqual(result, successful_call_message)
        self.assertNotIn("Circle", self.snapshot())

    def test_undo_with_empty_history_says_nothing_was_undone(self) -> None:
        _, traced = self.run_batch(("undo", {}))

        result = traced[0]["result"]
        self.assertNotEqual(result, successful_call_message)
        self.assertIn("Nothing to undo", str(result))

    def test_redo_with_empty_history_says_nothing_was_redone(self) -> None:
        result = self.run_single("redo")

        self.assertNotEqual(result, successful_call_message)
        self.assertIn("Nothing to redo", str(result))

    def test_create_point_on_an_occupied_spot_names_the_existing_point(self) -> None:
        self.run_single("create_point", x=1, y=1, name="A")
        depth = self.undo_depth()

        _, traced = self.run_batch(("create_point", {"x": 1, "y": 1, "name": "B", "color": None}))

        self.assertEqual(
            traced[0]["result"],
            "Point 'A' already exists at (1, 1); no new point was created. The requested name 'B' was not applied.",
        )
        self.assertFalse(traced[0]["is_error"])
        self.assertEqual(self.undo_depth(), depth)
        self.assertEqual(self.snapshot(), {"Point": ["A(1.0, 1.0)"]})

    def test_create_point_with_the_existing_name_does_not_mention_the_name(self) -> None:
        self.run_single("create_point", x=2.5, y=-1, name="A")

        result = self.run_single("create_point", x=2.5, y=-1, name="A")

        self.assertEqual(result, "Point 'A' already exists at (2.5, -1); no new point was created.")

    def test_create_point_on_a_free_spot_still_reports_success(self) -> None:
        result = self.run_single("create_point", x=1, y=1, name="A")

        self.assertEqual(result, successful_call_message)
        self.assertEqual(self.undo_depth(), 1)

    def test_undo_and_redo_with_history_still_report_success(self) -> None:
        self.run_single("create_point", x=1, y=1)

        self.assertEqual(self.run_single("undo"), successful_call_message)
        self.assertEqual(self.run_single("redo"), successful_call_message)


class TestToolErrorResults(_ToolBatchTestCase):
    """K21: results shaped {"error": ...} are flagged as tool errors."""

    def test_error_dict_from_a_tool_is_flagged(self) -> None:
        _, traced = self.run_batch(("analyze_graph", {"graph_name": "missing", "operation": "shortest_path"}))

        self.assertIsInstance(traced[0]["result"], dict)
        self.assertTrue(traced[0]["is_error"])

    def test_dict_with_empty_error_field_is_not_flagged(self) -> None:
        self.available_functions["lookup"] = lambda: {"tools": [], "error": None}

        _, traced = self.run_batch(("lookup", {}))

        self.assertFalse(traced[0]["is_error"])

    def test_typed_error_payload_is_flagged(self) -> None:
        self.available_functions["invert"] = lambda: {"type": "error", "value": "Error: singular matrix"}

        _, traced = self.run_batch(("invert", {}))

        self.assertTrue(traced[0]["is_error"])

    def test_json_error_string_from_solve_numeric_is_flagged(self) -> None:
        _, traced = self.run_batch(("solve_numeric", {"equations": []}))

        self.assertIsInstance(traced[0]["result"], str)
        self.assertIn('"error"', traced[0]["result"])
        self.assertTrue(traced[0]["is_error"])

    def test_json_string_without_error_is_not_flagged(self) -> None:
        _, traced = self.run_batch(("solve_numeric", {"equations": ["x - 2"]}))

        self.assertFalse(traced[0]["is_error"])

    def test_turn_metrics_count_json_error_strings(self) -> None:
        tool_results = [
            {"function_name": "solve_numeric", "result": '{"solutions": [], "error": "No equations provided."}'},
            {"function_name": "solve_numeric", "result": '{"solutions": [2.0], "method": "newton_raphson"}'},
        ]

        turn = aggregate_turn([], tool_results, None, "stop")

        self.assertEqual(turn["tool_errors"], 1)

    def test_error_string_is_still_flagged(self) -> None:
        _, traced = self.run_batch(("no_such_tool", {}))

        self.assertTrue(traced[0]["is_error"])

    def test_tool_call_log_marks_error_dicts_as_failed(self) -> None:
        calls = [{"function_name": "analyze_graph", "arguments": {"graph_name": "g"}}]
        results = {"analyze_graph(graph_name:g)": {"error": "Graph not found or spec missing"}}
        log = ToolCallLogManager()

        log.add_entries(calls, results)

        self.assertTrue(log.entries[0]["is_error"])
        self.assertIn("Graph not found", log.entries[0]["error_message"])

    def test_turn_metrics_count_error_dicts(self) -> None:
        tool_results = [
            {"function_name": "analyze_graph", "result": {"error": "Graph not found"}, "is_error": False},
            {"function_name": "search_tools", "result": {"tools": [], "error": None}, "is_error": False},
        ]

        turn = aggregate_turn([], tool_results, None, "stop")

        self.assertEqual(turn["tool_errors"], 1)


if __name__ == "__main__":
    unittest.main()
