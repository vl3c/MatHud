"""fit_regression with model_type "custom" through StatisticsManager (CPython, stub function manager).

Parameters are real (ParameterManager over the container); functions are drawn by the
harness's stub, which records the calls.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict

from server_tests import client_renderer  # noqa: F401  (browser stub, so MathUtils resolves)
from drawables.function import Function
from drawables.point import Point
from json_safe import _STRING_OR_PLUS_EXPONENT, _drop_exponent_plus, loads_json
from drawables.parameter import range_around
from managers.parameter_manager import ParameterManager
from server_tests.test_statistics_manager_dispatch import StatisticsManagerHarness
from utils.statistics.regression import residual_report


class _UndoManager:
    def __init__(self) -> None:
        self.archives = 0
        self.batches = 0

    def begin_batch(self) -> None:
        self.batches += 1

    def end_batch(self) -> None:
        return None

    def archive(self) -> None:
        self.archives += 1


XS = [0.5 * i for i in range(20)]
YS = [3 * math.exp(-0.4 * x) * math.cos(2.5 * x) + 1 for x in XS]


class CustomFitHarness(StatisticsManagerHarness):
    def setUp(self) -> None:
        super().setUp()
        self.undo = _UndoManager()
        self.canvas.undo_redo_manager = self.undo
        self.parameters = ParameterManager(self.canvas, self.drawables)  # type: ignore[arg-type]
        self.canvas.drawable_manager = SimpleNamespace(parameter_manager=self.parameters, point_manager=None)
        # Function.parameter_names reads the shared namespace, which only its owner writes
        previous = self.parameters.claim_namespace()
        self.addCleanup(self.parameters.release_namespace, previous)

    def fit(self, **overrides: Any) -> Dict[str, Any]:
        args: Dict[str, Any] = {
            "name": None,
            "x_data": XS,
            "y_data": YS,
            "model_type": "custom",
            "degree": None,
            "plot_bounds": None,
            "curve_color": None,
            "show_points": False,
            "point_color": None,
            "model": "y = a*exp(-b*x)*cos(c*x) + d",
            "initial_values": None,
            "update_parameters": None,
        }
        args.update(overrides)
        return dict(self.stats.fit_regression(**args))

    def parameter_values(self) -> Dict[str, float]:
        return {p.name: p.value for p in self.parameters.parameters}

    def assertRefused(self, hint: str, **overrides: Any) -> None:
        before = (self.parameter_values(), list(self.functions.drawn))
        with self.assertRaises(ValueError) as caught:
            self.fit(**overrides)
        self.assertIn(hint, str(caught.exception))
        self.assertEqual((self.parameter_values(), list(self.functions.drawn)), before)


class TestCustomFit(CustomFitHarness):
    def test_constants_become_parameters_and_the_model_is_drawn(self) -> None:
        result = self.fit()
        self.assertEqual(result["model_type"], "custom")
        self.assertEqual(result["model"], "a*exp(-b*x)*cos(c*x) + d")
        self.assertEqual(result["expression"], "3*exp(-0.4*x)*cos(2.5*x) + 1")
        self.assertEqual(result["parameters"], ["a", "b", "c", "d"])
        self.assertEqual(result["created_parameters"], ["a", "b", "c", "d"])
        self.assertGreater(result["r_squared"], 1 - 1e-12)
        self.assertLess(result["rmse"], 1e-9)
        self.assertEqual(len(result["residuals"]), len(XS))
        self.assertEqual(result["function_name"], "custom_fit")
        self.assertEqual(self.functions.drawn[0][:2], ("a*exp(-b*x)*cos(c*x) + d", "custom_fit"))
        values = self.parameter_values()
        for name, value in {"a": 3.0, "b": 0.4, "c": 2.5, "d": 1.0}.items():
            self.assertAlmostEqual(values[name], value, places=8)
            self.assertAlmostEqual(result["coefficients"][name], value, places=8)
        b = self.parameters.get_parameter("b")
        assert b is not None
        self.assertEqual((b.min, b.max, b.step), (0.0, 0.8, 0.01))

    def test_an_unused_parameter_is_updated_from_its_value(self) -> None:
        self.parameters.set_parameter("d", 5.0)
        result = self.fit()
        self.assertEqual(result["created_parameters"], ["a", "b", "c"])
        self.assertAlmostEqual(self.parameter_values()["d"], 1.0, places=8)

    def test_parameters_other_objects_read_need_update_parameters(self) -> None:
        self.parameters.set_parameter("d", 5.0)
        self.drawables.add(Function("d*x", name="g"))
        self.assertRefused("update_parameters: true")
        result = self.fit(update_parameters=True)
        self.assertEqual(result["moved_objects"], ["g"])
        self.assertTrue(any("g read the updated parameters" in note for note in result["notes"]))

    def test_a_function_drawing_the_model_is_reused(self) -> None:
        for name in "abcd":
            self.parameters.set_parameter(name, 1.0)
        self.drawables.add(Function("a*exp(-b*x)*cos(c*x)+d", name="f"))
        result = self.fit()
        self.assertEqual(result["function_name"], "f")
        self.assertEqual(self.functions.drawn, [])
        self.assertNotIn("bounds", result)
        self.assertNotIn("moved_objects", result)
        self.assertAlmostEqual(self.parameter_values()["c"], 2.5, places=8)

    def test_a_constant_named_like_another_object_is_refused(self) -> None:
        self.drawables.add(Point(0.0, 0.0, name="A"))
        self.assertRefused("rename the constant", model="A*exp(-k*x)")

    def test_refusals_leave_the_canvas_unchanged(self) -> None:
        self.assertRefused("a*x", model="ax + b")
        self.assertRefused("not a constant of the model", initial_values=[{"name": "q", "value": 1}])
        self.assertRefused("twice", initial_values=[{"name": "a", "value": 1}, {"name": "a", "value": 2}])
        self.assertRefused("finite number", initial_values=[{"name": "a", "value": float("inf")}])
        self.assertRefused("left_bound must be less", plot_bounds={"left_bound": 5, "right_bound": 1})
        self.assertRefused("only read with model_type 'custom'", model_type="linear")
        self.assertRefused(
            "only read with model_type 'custom'", model_type="linear", model=None, update_parameters=True
        )

    def test_one_undo_batch(self) -> None:
        self.fit()
        self.assertEqual(self.undo.batches, 1)

    def test_fixed_types_report_rmse_and_residuals(self) -> None:
        result = self.fit(model_type="linear", model=None, x_data=[1, 2, 3, 4], y_data=[2, 4, 5, 8])
        self.assertAlmostEqual(result["rmse"], math.sqrt(sum(r * r for r in result["residuals"]) / 4), places=5)
        self.assertEqual(len(result["residuals"]), 4)


class TestHelpers(CustomFitHarness):
    def test_range_around(self) -> None:
        self.assertEqual(range_around(0.0), (-1.0, 1.0, 0.02))
        self.assertEqual(range_around(2.5), (0.0, 5.0, 0.05))
        self.assertEqual(range_around(-3.7), (-7.4, 0.0, 0.1))
        self.assertEqual(range_around(1.1e46), (0.0, 2.2e46, 5e44))

    def test_residual_report_lists_at_most_100(self) -> None:
        short = residual_report([1.0, 2.0], [1.5, 1.0])
        self.assertEqual(short["residuals"], [-0.5, 1.0])
        long_report = residual_report([1.0] * 101, [0.0] * 100 + [3.0])
        self.assertNotIn("residuals", long_report)
        self.assertEqual(long_report["max_abs_residual"], 2.0)

    def test_loads_json_drops_exponent_plus_outside_strings(self) -> None:
        text = '{"a": 1.1e+46, "s": "1e+5 \\" 2E+3", "b": [1E+2]}'
        repaired = _STRING_OR_PLUS_EXPONENT.sub(_drop_exponent_plus, text)
        self.assertEqual(repaired, '{"a": 1.1e46, "s": "1e+5 \\" 2E+3", "b": [1E2]}')
        self.assertEqual(loads_json(text), {"a": 1.1e46, "s": '1e+5 " 2E+3', "b": [100.0]})
        with self.assertRaises(ValueError):
            loads_json("{bad")
