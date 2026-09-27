"""Regression tests for function analysis kept in step with the function (K11-K14, K24).

Tool calls run through ``ProcessFunctionCalls.get_results_traced``, the path a model
tool batch takes, against a real canvas.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, List, Optional, Tuple

from canvas import Canvas
from function_registry import FunctionRegistry
from process_function_calls import ProcessFunctionCalls
from utils.math_utils import MathUtils
from workspace_manager import WorkspaceManager


def _rounded(values: Optional[List[float]]) -> List[float]:
    return sorted(round(float(value), 6) for value in values or [])


class _FunctionToolTestCase(unittest.TestCase):
    """Shared setup: a real canvas with the model's function registry."""

    def setUp(self) -> None:
        self.canvas = Canvas(500, 500, draw_enabled=False)
        self.workspace_manager = WorkspaceManager(self.canvas)
        self.available_functions: Dict[str, Any] = FunctionRegistry.get_available_functions(
            self.canvas, self.workspace_manager
        )
        self.undoable_functions: Tuple[str, ...] = FunctionRegistry.get_undoable_functions()

    def run_call(self, tool_name: str, **args: Any) -> Dict[str, Any]:
        """Run one call as its own batch and return its traced record."""
        batch = [{"function_name": tool_name, "arguments": dict(args)}]
        _, traced = ProcessFunctionCalls.get_results_traced(
            batch, self.available_functions, self.undoable_functions, self.canvas
        )
        return traced[0]

    def draw(self, function_string: str, name: str, left: Optional[float], right: Optional[float], **extra: Any) -> Any:
        args: Dict[str, Any] = {
            "function_string": function_string,
            "name": name,
            "left_bound": left,
            "right_bound": right,
            "color": None,
            "undefined_at": None,
        }
        args.update(extra)
        traced = self.run_call("draw_function", **args)
        self.assertFalse(traced["is_error"], traced["result"])
        return traced["result"]

    def function(self, name: str) -> Any:
        function = self.canvas.drawable_manager.get_function(name)
        self.assertIsNotNone(function)
        return function

    def function_args(self, name: str) -> Dict[str, Any]:
        state: Dict[str, Any] = self.function(name).get_state()["args"]
        return state

    def undo_depth(self) -> int:
        return len(self.canvas.undo_redo_manager.undo_stack)

    def assert_all_equal(self, values: Optional[List[float]], expected: float) -> None:
        """Every listed value (at least one) equals expected."""
        self.assertTrue(values, f"expected values equal to {expected}, got {values}")
        for value in values or []:
            self.assertAlmostEqual(float(value), expected)


class TestFunctionTranslationAnalysis(_FunctionToolTestCase):
    """K11: translating a function moves its asymptotes and discontinuities."""

    def test_asymptotes_follow_the_translation(self) -> None:
        self.draw("1/x", "f", -10, 10)

        traced = self.run_call("translate_object", name="f", x_offset=2, y_offset=3)

        self.assertFalse(traced["is_error"], traced["result"])
        args = self.function_args("f")
        self.assertEqual(_rounded(args.get("vertical_asymptotes")), [2.0])
        self.assert_all_equal(args.get("horizontal_asymptotes"), 3.0)
        self.assertEqual((args["left_bound"], args["right_bound"]), (-8, 12))
        self.assertAlmostEqual(self.function("f").function(3), 4.0)

    def test_tangent_asymptotes_follow_the_translation(self) -> None:
        self.draw("tan(x)", "t", -2, 2)

        self.run_call("translate_object", name="t", x_offset=1, y_offset=0)

        expected = [round(-math.pi / 2 + 1, 6), round(math.pi / 2 + 1, 6)]
        self.assertEqual(_rounded(self.function_args("t").get("vertical_asymptotes")), expected)
        function = self.function("t")
        self.assertIsNotNone(function.get_vertical_asymptote_between_x(math.pi / 2 + 0.9, math.pi / 2 + 1.1))
        self.assertIsNone(function.get_vertical_asymptote_between_x(math.pi / 2 - 0.1, math.pi / 2 + 0.1))

    def test_point_discontinuities_follow_the_translation(self) -> None:
        self.draw("abs(x)", "a", -5, 5)
        self.draw("x^2", "h", -5, 5, undefined_at=[1])

        self.run_call("translate_object", name="a", x_offset=2, y_offset=0)
        self.run_call("translate_object", name="h", x_offset=2, y_offset=0)

        self.assertEqual(_rounded(self.function_args("a").get("point_discontinuities")), [2.0])
        h_args = self.function_args("h")
        self.assertEqual(_rounded(h_args.get("point_discontinuities")), [3.0])
        self.assertEqual(_rounded(h_args.get("undefined_at")), [3.0])

    def test_undo_restores_the_asymptotes(self) -> None:
        self.draw("1/x", "f", -10, 10)
        self.run_call("translate_object", name="f", x_offset=2, y_offset=3)

        self.run_call("undo")

        args = self.function_args("f")
        self.assertEqual(_rounded(args.get("vertical_asymptotes")), [0.0])
        self.assert_all_equal(args.get("horizontal_asymptotes"), 0.0)


# (expression, left, right, vertical asymptotes, removable discontinuities)
_SINGULARITY_CASES: List[Tuple[str, Optional[float], Optional[float], List[float], List[float]]] = [
    ("(x^2-1)/(x-1)", None, None, [], [1.0]),
    ("(x-1)/(x^2-1)", None, None, [-1.0], [1.0]),
    ("sin(x)/x", None, None, [], [0.0]),
    ("1/x", None, None, [0.0], []),
    ("1/x^2", None, None, [0.0], []),
    ("1/(x^2-1)", None, None, [-1.0, 1.0], []),
    ("sqrt(x)/x", None, None, [0.0], []),
    ("exp(1/x)", None, None, [0.0], []),
    ("log(x)", None, None, [0.0], []),
    ("tan(x)", -2, 2, [round(-math.pi / 2, 6), round(math.pi / 2, 6)], []),
    ("x^2 + 1", None, None, [], []),
]


class TestRemovableDiscontinuities(_FunctionToolTestCase):
    """K24: a zero of a denominator where f stays bounded is a hole, not an asymptote."""

    def test_singularities_are_classified(self) -> None:
        for expression, left, right, asymptotes, holes in _SINGULARITY_CASES:
            with self.subTest(expression=expression):
                vertical, _, discontinuities = MathUtils.calculate_asymptotes_and_discontinuities(
                    expression, left, right
                )
                self.assertEqual(_rounded(vertical), asymptotes)
                self.assertEqual(_rounded(discontinuities), holes)

    def test_holes_outside_the_bounds_are_not_listed(self) -> None:
        _, _, discontinuities = MathUtils.calculate_asymptotes_and_discontinuities("(x^2-1)/(x-1)", 2, 5)

        self.assertEqual(discontinuities, [])

    def test_drawn_function_with_a_hole_lists_no_asymptote(self) -> None:
        self.draw("(x^2 - 1)/(x - 1)", "h", None, None, undefined_at=[1])

        args = self.function_args("h")
        self.assertNotIn("vertical_asymptotes", args)
        self.assertEqual(_rounded(args.get("point_discontinuities")), [1.0])
        self.assertAlmostEqual(self.function("h").function(2), 3.0)
