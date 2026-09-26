"""Regression tests for model tool batches: undo granularity.

Every test runs tool calls through ``ProcessFunctionCalls.get_results_traced``, the
same path a model tool batch takes, against a real canvas.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Tuple

from canvas import Canvas
from function_registry import FunctionRegistry
from process_function_calls import ProcessFunctionCalls
from workspace_manager import WorkspaceManager

TRIANGLE_VERTICES: List[Dict[str, float]] = [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}]


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


if __name__ == "__main__":
    unittest.main()
