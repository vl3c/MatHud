"""Regression tests for tool arguments sent as null and for truthful results (fix batch 7).

- ``update_circle_arc`` always failed: the canvas passed endpoint arguments the arc manager
  does not take.
- Strict-schema models send ``null`` for every optional argument; each create and update
  tool must treat it as "use the default" or "keep the current value" (the K10 pattern).
- An update, translation or label change that changes nothing says so (the K26 family).

Every tool call runs through ``ProcessFunctionCalls.get_results_traced``, the same path a
model tool batch takes, against a real canvas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from constants import successful_call_message
from expression_validator import ExpressionValidator

from .test_tool_batch_results import _ToolBatchTestCase

NullCall = Tuple[str, Dict[str, Any]]


class _NullToolTestCase(_ToolBatchTestCase):
    """Helpers: run a call, and read the newest drawable of a bucket."""

    def call(self, tool: str, **args: Any) -> Dict[str, Any]:
        _, traced = self.run_batch((tool, args))
        return traced[0]

    def newest(self, bucket: str) -> Any:
        drawables = getattr(self.canvas.drawable_manager.drawables, bucket)
        self.assertTrue(drawables, f"no {bucket}")
        return drawables[-1]

    def newest_name(self, bucket: str) -> str:
        return str(self.newest(bucket).name)


class TestCircleArcUpdate(_NullToolTestCase):
    """update_circle_arc passed endpoint arguments the arc manager does not accept, so it always failed."""

    def make_arc(self, use_major_arc: bool = False) -> Any:
        traced = self.call(
            "create_circle_arc",
            point1_x=5,
            point1_y=0,
            point2_x=0,
            point2_y=5,
            point1_name=None,
            point2_name=None,
            point3_x=None,
            point3_y=None,
            point3_name=None,
            center_point_choice=None,
            circle_name=None,
            center_x=0,
            center_y=0,
            radius=5,
            use_major_arc=use_major_arc,
            arc_name=None,
            color=None,
        )
        self.assertFalse(traced["is_error"], traced["result"])
        return self.newest("CircleArcs")

    def test_update_color(self) -> None:
        arc = self.make_arc()
        depth = self.undo_depth()

        traced = self.call("update_circle_arc", name=arc.name, new_color="red", use_major_arc=None)

        self.assertFalse(traced["is_error"], traced["result"])
        self.assertEqual(arc.color, "red")
        self.assertEqual(self.undo_depth(), depth + 1)

    def test_switch_to_the_major_arc_and_undo(self) -> None:
        arc = self.make_arc()

        traced = self.call("update_circle_arc", name=arc.name, new_color=None, use_major_arc=True)

        self.assertFalse(traced["is_error"], traced["result"])
        self.assertTrue(arc.use_major_arc)
        self.run_single("undo")
        self.assertFalse(self.newest("CircleArcs").use_major_arc)

    def test_the_sweep_already_selected_is_a_no_op(self) -> None:
        arc = self.make_arc()
        depth = self.undo_depth()

        traced = self.call("update_circle_arc", name=arc.name, new_color=None, use_major_arc=False)

        self.assertFalse(traced["is_error"], traced["result"])
        self.assertEqual(
            traced["result"], f"Circle arc '{arc.name}' already has the minor arc selected; nothing changed."
        )
        self.assertEqual(self.undo_depth(), depth)

    def test_update_with_nothing_to_change_is_an_error(self) -> None:
        arc = self.make_arc()

        traced = self.call("update_circle_arc", name=arc.name, new_color=None, use_major_arc=None)

        self.assertTrue(traced["is_error"], traced["result"])
        self.assertIn("at least one property", str(traced["result"]))

    def test_endpoint_arguments_are_not_accepted(self) -> None:
        """The schema offers no endpoint arguments; the canvas no longer takes them either."""
        arc = self.make_arc()

        with self.assertRaises(TypeError):
            self.canvas.update_circle_arc(arc.name, point1_x=1, point1_y=2)  # type: ignore[call-arg]
