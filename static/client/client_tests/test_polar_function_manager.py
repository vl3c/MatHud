"""Tests for polar curves r(θ) on a real canvas: the tools, undo, names, tangents, intersections,
transforms and workspace restore.

The drawable itself (parsing, refusals, range, state) is covered by pytest in
server_tests/test_polar_function.py.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict

from constants import successful_call_message

from .test_tool_batch_results import _ToolBatchTestCase

_DRAW_DEFAULTS: Dict[str, Any] = {"name": None, "theta_min": None, "theta_max": None, "color": None}


class TestPolarFunctionTools(_ToolBatchTestCase):
    def draw(self, r_expression: str, **args: Any) -> Dict[str, Any]:
        call = dict(_DRAW_DEFAULTS)
        call.update(args)
        call["r_expression"] = r_expression
        _, traced = self.run_batch(("draw_polar_function", call))
        return traced[0]

    def curves(self) -> Any:
        return self.canvas.drawable_manager.drawables.PolarFunctions

    def test_draw_with_every_nullable_field_null(self) -> None:
        traced = self.draw("1 + cos(θ)")
        self.assertFalse(traced.get("is_error"), traced)
        curve = self.curves()[0]
        self.assertEqual(curve.name, "f_polar")
        self.assertEqual((curve.theta_min, curve.theta_max), (0.0, 2 * math.pi))
        self.assertEqual(curve.r_expression, "1 + cos(θ)")

    def test_default_names_and_requested_names(self) -> None:
        self.draw("1")
        self.draw("2")
        self.draw("θ", name="spiral")
        self.assertEqual([curve.name for curve in self.curves()], ["f_polar", "g_polar", "spiral"])

    def test_only_theta_min_gives_one_turn_from_it(self) -> None:
        self.draw("θ", theta_min=10)
        curve = self.curves()[0]
        self.assertAlmostEqual(curve.theta_min, 10.0)
        self.assertAlmostEqual(curve.theta_max, 10.0 + 2 * math.pi)

    def test_one_undo_step_per_draw_and_none_for_a_refusal(self) -> None:
        depth = self.undo_depth()
        self.draw("2*sin(3*θ)")
        self.assertEqual(self.undo_depth(), depth + 1)
        traced = self.draw("x + θ")
        self.assertTrue(traced.get("is_error"), traced)
        self.assertIn("may use only θ", str(traced.get("result")))
        self.assertEqual(self.undo_depth(), depth + 1)
        self.assertEqual(len(self.curves()), 1)
        self.canvas.undo()
        self.assertEqual(len(self.curves()), 0)

    def test_update_and_delete(self) -> None:
        self.draw("θ", name="s")
        _, traced = self.run_batch(
            (
                "update_polar_function",
                {"name": "s", "new_color": "red", "new_theta_min": None, "new_theta_max": 4 * math.pi},
            )
        )
        self.assertFalse(traced[0].get("is_error"), traced)
        curve = self.curves()[0]
        self.assertEqual(curve.color, "red")
        self.assertAlmostEqual(curve.theta_max, 4 * math.pi)
        _, traced = self.run_batch(
            ("update_polar_function", {"name": "s", "new_color": None, "new_theta_min": 0, "new_theta_max": 400})
        )
        self.assertTrue(traced[0].get("is_error"), traced)
        self.assertAlmostEqual(curve.theta_max, 4 * math.pi)
        _, traced = self.run_batch(("delete_polar_function", {"name": "s"}))
        self.assertFalse(traced[0].get("is_error"), traced)
        self.assertEqual(len(self.curves()), 0)

    def test_update_to_the_current_color_changes_nothing(self) -> None:
        self.draw("1", name="c", color="blue")
        depth = self.undo_depth()
        result = self.run_single(
            "update_polar_function", name="c", new_color="blue", new_theta_min=None, new_theta_max=None
        )
        self.assertNotEqual(result, successful_call_message)
        self.assertEqual(self.undo_depth(), depth)

    def test_translate_is_refused_and_changes_nothing(self) -> None:
        self.draw("1 + cos(θ)", name="card")
        depth = self.undo_depth()
        _, traced = self.run_batch(("translate_object", {"name": "card", "x_offset": 2, "y_offset": 0}))
        self.assertTrue(traced[0].get("is_error"), traced)
        self.assertIn("PolarFunction", str(traced[0].get("result")))
        self.assertEqual(self.undo_depth(), depth)
        self.assertEqual(self.curves()[0].evaluate(0.0), (2.0, 0.0))

    def test_tangent_at_theta(self) -> None:
        # The unit circle r = 1: at θ = π/2 the tangent is horizontal through (0, 1)
        self.draw("1", name="u")
        segment = self.canvas.create_tangent_line("u", math.pi / 2)
        self.assertAlmostEqual(segment.point1.y, 1.0, places=5)
        self.assertAlmostEqual(segment.point2.y, 1.0, places=5)

    def test_tangent_where_r_is_not_real_names_theta(self) -> None:
        self.draw("sqrt(cos(2θ))", name="lem")
        with self.assertRaises(ValueError) as caught:
            self.canvas.create_tangent_line("lem", math.pi / 2)
        self.assertIn("θ=", str(caught.exception))

    def test_intersections_report_theta(self) -> None:
        self.draw("1", name="u")
        self.run_single("draw_function", function_string="0.5", name="h", left_bound=None, right_bound=None)
        result = self.run_single("find_intersections", object_names=["u", "h"], extend_lines=None, place_points=None)
        self.assertEqual(result["count"], 2)
        thetas = sorted(point["params"]["u"]["theta"] for point in result["intersections"])
        self.assertAlmostEqual(thetas[0], math.pi / 6, places=6)
        self.assertAlmostEqual(thetas[1], 5 * math.pi / 6, places=6)

    def test_workspace_round_trip(self) -> None:
        self.draw("2*sin(3*θ)", name="rose", theta_min=0.5, theta_max=3, color="green")
        saved = json.loads(json.dumps(self.workspace_manager._snapshot_persistable_canvas_state()))
        self.assertEqual(
            saved["PolarFunctions"],
            [
                {
                    "name": "rose",
                    "args": {"r_expression": "2*sin(3*θ)", "theta_min": 0.5, "theta_max": 3.0, "color": "green"},
                }
            ],
        )
        self.canvas.clear()
        self.workspace_manager._restore_workspace_state(saved)
        curve = self.curves()[0]
        self.assertEqual(curve.get_state(), saved["PolarFunctions"][0])

    def test_canvas_state_lists_the_curve(self) -> None:
        self.draw("1 + cos(θ)", name="card")
        state = self.canvas.get_canvas_state()
        self.assertEqual(state["PolarFunctions"][0]["args"]["r_expression"], "1 + cos(θ)")
