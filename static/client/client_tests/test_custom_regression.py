"""Custom-model fits through the canvas: parameters, the curve that follows them, undo, saving, refusals.

The engine and its notes are covered by pytest (server_tests/test_custom_regression_pure.py,
test_custom_regression_manager.py); these tests run the tool in Brython on a real canvas.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List

import compiled_expression
from json_safe import loads_json
from utils.statistics.custom_model import read_model
from utils.statistics.custom_regression import fit_custom_model

from .test_parameters import _ParameterTestCase

_FIT = {
    "name": None,
    "degree": None,
    "plot_bounds": None,
    "curve_color": None,
    "show_points": False,
    "point_color": None,
    "model": None,
    "initial_values": None,
    "update_parameters": None,
}


def _grid(low: float, high: float, count: int) -> List[float]:
    return [low + (high - low) * i / (count - 1) for i in range(count)]


XS = _grid(0.0, 10.0, 40)
YS = [3 * math.exp(-0.4 * x) * math.cos(2.5 * x) + 1 + 0.02 * math.sin(7.3 * x) for x in XS]


class TestCustomRegression(_ParameterTestCase):
    def fit(self, **args: Any) -> Dict[str, Any]:
        call = dict(_FIT)
        call.update({"x_data": XS, "y_data": YS, "model_type": "custom", "model": "a*exp(-b*x)*cos(c*x) + d"})
        call.update(args)
        _, traced = self.run_batch(("fit_regression", call))
        return traced[0]

    def curve(self, name: str = "custom_fit") -> Any:
        return [f for f in self.canvas.drawable_manager.drawables.Functions if f.name == name][0]

    def test_fit_creates_parameters_and_a_curve_that_follows_them(self) -> None:
        traced = self.fit()
        self.assert_ok(traced)
        result = traced["result"]
        self.assertEqual(result["created_parameters"], ["a", "b", "c", "d"])
        self.assertGreater(result["r_squared"], 0.999)
        self.assertLess(abs(result["coefficients"]["c"] - 2.5), 0.01)
        sse = sum((y - self.curve().function(x)) ** 2 for x, y in zip(XS, YS))
        self.assertLess(sse, 40 * 0.02**2)
        self.assertAlmostEqual(compiled_expression.parameter_value("b"), result["coefficients"]["b"])
        before = self.curve().function(1.0)
        self.set("d", value=result["coefficients"]["d"] + 2)
        self.assertAlmostEqual(self.curve().function(1.0), before + 2, places=9)

    def test_one_undo_step_removes_everything(self) -> None:
        depth = self.undo_depth()
        self.fit()
        self.assertEqual(self.undo_depth(), depth + 1)
        self.canvas.undo_redo_manager.undo()
        self.assertEqual(self.manager.parameters, [])
        self.assertEqual(list(self.canvas.drawable_manager.drawables.Functions), [])
        self.assertIsNone(compiled_expression.parameter_value("a"))

    def test_refusals_change_nothing(self) -> None:
        depth = self.undo_depth()
        self.assert_error(self.fit(model="ax + b"), "a*x")
        self.assert_error(self.fit(model="a*x", initial_values=[{"name": "q", "value": 1}]), "constants are a")
        self.assert_error(self.fit(model_type="linear", model="a*x"), "only read with model_type 'custom'")
        self.point(1, 2, "A")
        self.assert_error(self.fit(model="A*exp(-k*x)"), "rename the constant")
        self.assertEqual(self.manager.parameters, [])
        self.assertEqual(self.undo_depth(), depth + 1)  # the point only

    def test_existing_parameters_read_by_other_objects_need_consent(self) -> None:
        self.set("d", value=0)
        self.draw_function("d + x", name="g")
        self.assert_error(self.fit(), "update_parameters: true")
        traced = self.fit(update_parameters=True)
        self.assert_ok(traced)
        self.assertEqual(traced["result"]["moved_objects"], ["g"])

    def test_large_constants_survive_a_workspace_round_trip(self) -> None:
        xs = _grid(2000.0, 2020.0, 21)
        ys = [1000 * math.exp(-0.05 * (x - 2000)) for x in xs]
        self.assert_ok(self.fit(model="A*exp(-k*x)", x_data=xs, y_data=ys))
        value = self.manager.get_parameter("A").value
        self.assertGreater(value, 1e40)
        text = json.dumps(self.workspace_manager._snapshot_persistable_canvas_state())
        self.assertIn("e+", text)
        state = loads_json(text)
        self.canvas.clear()
        self.workspace_manager._restore_workspace_state(state)
        self.assertEqual(self.manager.get_parameter("A").value, value)

    def test_fixed_types_report_rmse_and_residuals(self) -> None:
        traced = self.fit(model_type="linear", model=None, x_data=[1, 2, 3, 4], y_data=[2, 4, 5, 8])
        self.assert_ok(traced)
        self.assertEqual(len(traced["result"]["residuals"]), 4)
        self.assertGreater(traced["result"]["rmse"], 0)

    def test_loads_json_reads_plus_exponents(self) -> None:
        self.assertEqual(loads_json('{"a": 1.1e+46, "s": "2e+3"}'), {"a": 1.1e46, "s": "2e+3"})

    def test_a_typical_fit_is_quick(self) -> None:
        started = time.time()
        fit = fit_custom_model(XS, YS, read_model("a*exp(-b*x)*cos(c*x) + d"))
        elapsed = time.time() - started
        self.assertGreater(fit.r_squared, 0.999)
        self.assertLess(elapsed, 15.0)
