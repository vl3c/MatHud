"""Tests for the arc_length tool on a real canvas: curves by expression and plotted curves by
name, through the model's function registry, and its speed under Brython.

The algorithm and the tool's refusals are covered by pytest in server_tests/test_arc_length.py.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict

from .test_tool_batch_results import _ToolBatchTestCase

_ARC_DEFAULTS: Dict[str, Any] = {
    "curve_name": None,
    "function_string": None,
    "x_expression": None,
    "y_expression": None,
    "r_expression": None,
    "lower_bound": None,
    "upper_bound": None,
}

# Length of y = x^2 from 0 to 1, and of the cardioid r = 1 + cos(θ)
_PARABOLA = math.sqrt(5) / 2 + math.asinh(2) / 4
_CARDIOID = 8.0


class TestArcLengthTool(_ToolBatchTestCase):
    def measure(self, **args: Any) -> Dict[str, Any]:
        call = dict(_ARC_DEFAULTS)
        call.update(args)
        _, traced = self.run_batch(("arc_length", call))
        return traced[0]

    def length_of(self, **args: Any) -> float:
        traced = self.measure(**args)
        self.assertFalse(traced.get("is_error"), traced)
        return float(traced["result"]["length"])

    def test_function_by_expression(self) -> None:
        self.assertAlmostEqual(self.length_of(function_string="x^2", lower_bound=0, upper_bound=1), _PARABOLA, places=9)

    def test_parametric_and_polar_by_expression(self) -> None:
        astroid = self.length_of(x_expression="cos(t)^3", y_expression="sin(t)^3", lower_bound=0, upper_bound="2*pi")
        self.assertAlmostEqual(astroid, 6.0, places=8)
        self.assertAlmostEqual(self.length_of(r_expression="1 + cos(θ)"), _CARDIOID, places=8)

    def test_a_refusal_is_an_error_result(self) -> None:
        traced = self.measure(function_string="1/x", lower_bound=-1, upper_bound=1)
        self.assertTrue(traced.get("is_error"), traced)
        self.assertIn("x = 0", str(traced.get("result")))

    def test_measuring_adds_no_undo_step(self) -> None:
        depth = self.undo_depth()
        self.length_of(function_string="sin(x)", lower_bound=0, upper_bound="pi")
        self.assertEqual(self.undo_depth(), depth)

    def test_plotted_function_by_name_uses_its_bounds(self) -> None:
        self.run_single(
            "draw_function",
            function_string="x^2",
            name="f",
            left_bound=0,
            right_bound=1,
            color=None,
            undefined_at=None,
        )
        traced = self.measure(curve_name="f")
        self.assertFalse(traced.get("is_error"), traced)
        self.assertAlmostEqual(float(traced["result"]["length"]), _PARABOLA, places=9)
        self.assertEqual(traced["result"]["curve_type"], "function")

    def test_plotted_piecewise_function_by_name(self) -> None:
        piece = {"left_inclusive": True, "right_inclusive": False, "undefined_at": None}
        self.run_single(
            "draw_piecewise_function",
            pieces=[
                dict(piece, expression="-x", left=-1, right=0),
                dict(piece, expression="x", left=0, right=1),
            ],
            name="v",
            color=None,
        )
        self.assertAlmostEqual(
            self.length_of(curve_name="v", lower_bound=-1, upper_bound=1), 2 * math.sqrt(2), places=9
        )

    def test_plotted_parametric_polar_and_circle_by_name(self) -> None:
        self.run_single(
            "draw_parametric_function",
            x_expression="cos(t)",
            y_expression="sin(t)",
            name="half",
            t_min=0,
            t_max=math.pi,
            color=None,
        )
        self.assertAlmostEqual(self.length_of(curve_name="half"), math.pi, places=9)
        self.run_single(
            "draw_polar_function", r_expression="1 + cos(θ)", name="heart", theta_min=None, theta_max=None, color=None
        )
        self.assertAlmostEqual(self.length_of(curve_name="heart"), _CARDIOID, places=8)
        self.run_single("create_circle", center_x=1, center_y=2, radius=3, color=None, name=None)
        circle = self.canvas.drawable_manager.drawables.Circles[0]
        self.assertAlmostEqual(self.length_of(curve_name=circle.name), 6 * math.pi, places=9)

    def test_twice_traced_rose_reports_one_trace(self) -> None:
        traced = self.measure(r_expression="cos(θ)")
        self.assertFalse(traced.get("is_error"), traced)
        result = traced["result"]
        self.assertEqual(result["traced_times"], 2)
        self.assertAlmostEqual(float(result["length_of_one_trace"]), math.pi, places=8)

    def test_unknown_name_is_an_error(self) -> None:
        traced = self.measure(curve_name="nothing")
        self.assertTrue(traced.get("is_error"), traced)
        self.assertIn("nothing", str(traced.get("result")))

    def test_oscillating_curve_stays_within_the_time_limit(self) -> None:
        """sin(10x) on [0, 10] is about the slowest common input: it must answer, not time out."""
        started = time.time()
        length = self.length_of(function_string="sin(10*x)", lower_bound=0, upper_bound=10)
        elapsed = time.time() - started
        self.assertAlmostEqual(length, 64.8301227949, places=6)
        self.assertLess(elapsed, 6.0)
