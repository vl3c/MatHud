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


class TestFunctionRedefinition(_FunctionToolTestCase):
    """K12: redrawing under an existing name re-analyses; reversed bounds are swapped."""

    def test_redraw_recomputes_asymptotes(self) -> None:
        self.draw("1/x", "f", -10, 10)

        self.draw("x^2", "f", -10, 10)

        self.assertEqual(len(self.canvas.drawable_manager.drawables.Functions), 1)
        args = self.function_args("f")
        self.assertNotIn("vertical_asymptotes", args)
        self.assertNotIn("horizontal_asymptotes", args)
        self.assertAlmostEqual(self.function("f").function(2), 4.0)

    def test_redraw_finds_the_new_asymptotes(self) -> None:
        self.draw("x^2", "f", -10, 10)

        self.draw("1/(x-1)", "f", -10, 10)

        self.assertEqual(_rounded(self.function_args("f").get("vertical_asymptotes")), [1.0])

    def test_redraw_without_holes_clears_the_old_holes(self) -> None:
        self.draw("x^2", "f", -10, 10, undefined_at=[1])

        self.draw("x^3", "f", -10, 10)

        args = self.function_args("f")
        self.assertNotIn("undefined_at", args)
        self.assertNotIn("point_discontinuities", args)
        self.assertAlmostEqual(self.function("f").function(1), 1.0)

    def test_redraw_with_a_bad_expression_changes_nothing(self) -> None:
        self.draw("1/x", "f", -10, 10)

        traced = self.run_call(
            "draw_function",
            function_string="1/(",
            name="f",
            left_bound=-5,
            right_bound=5,
            color=None,
            undefined_at=None,
        )

        self.assertTrue(traced["is_error"])
        args = self.function_args("f")
        self.assertEqual(args["function_string"], "1/x")
        self.assertEqual((args["left_bound"], args["right_bound"]), (-10, 10))
        self.assertEqual(_rounded(args.get("vertical_asymptotes")), [0.0])

    def test_reversed_bounds_are_swapped_and_reported(self) -> None:
        result = self.draw("x", "g", 5, -5)

        args = self.function_args("g")
        self.assertEqual((args["left_bound"], args["right_bound"]), (-5, 5))
        self.assertIsInstance(result, str)
        self.assertIn("swapped", result)
        self.assertIn("[-5, 5]", result)

    def test_equal_bounds_are_rejected(self) -> None:
        depth = self.undo_depth()

        traced = self.run_call(
            "draw_function",
            function_string="x",
            name="g",
            left_bound=2,
            right_bound=2,
            color=None,
            undefined_at=None,
        )

        self.assertTrue(traced["is_error"])
        self.assertIn("left_bound must be less than right_bound", str(traced["result"]))
        self.assertEqual(len(self.canvas.drawable_manager.drawables.Functions), 0)
        self.assertEqual(self.undo_depth(), depth)

    def test_update_bounds_recomputes_bound_dependent_asymptotes(self) -> None:
        self.draw("tan(x)", "t", -2, 2)

        traced = self.run_call("update_function", name="t", new_color=None, new_left_bound=-5, new_right_bound=5)

        self.assertFalse(traced["is_error"], traced["result"])
        expected = [round(-math.pi / 2 + n * math.pi, 6) for n in range(-1, 3)]
        self.assertEqual(_rounded(self.function_args("t").get("vertical_asymptotes")), expected)


class TestFunctionTangentGuard(_FunctionToolTestCase):
    """K13: no tangent or normal where the function or its derivative is not finite."""

    def assert_refused(self, tool_name: str, curve_name: str, parameter: float) -> None:
        depth = self.undo_depth()

        traced = self.run_call(
            tool_name, curve_name=curve_name, parameter=parameter, name=None, length=None, color=None
        )

        self.assertTrue(traced["is_error"], traced["result"])
        self.assertEqual(len(self.canvas.drawable_manager.drawables.Segments), 0)
        self.assertEqual(len(self.canvas.drawable_manager.drawables.Points), 0)
        self.assertEqual(self.undo_depth(), depth)

    def test_tangent_at_a_tan_asymptote_is_refused(self) -> None:
        self.draw("tan(x)", "t", -4, 4)

        self.assert_refused("draw_tangent_line", "t", math.pi / 2)

    def test_normal_at_a_tan_asymptote_is_refused(self) -> None:
        self.draw("tan(x)", "t", -4, 4)

        self.assert_refused("draw_normal_line", "t", math.pi / 2)

    def test_tangent_just_beside_an_asymptote_is_refused(self) -> None:
        self.draw("tan(x)", "t", -4, 4)

        self.assert_refused("draw_tangent_line", "t", math.pi / 2 + 1e-8)

    def test_tangent_where_the_function_is_undefined_is_refused(self) -> None:
        self.draw("1/(x-1)", "r", -4, 4)

        self.assert_refused("draw_tangent_line", "r", 1)

    def test_refusal_names_the_asymptote(self) -> None:
        self.draw("tan(x)", "t", -4, 4)

        traced = self.run_call(
            "draw_tangent_line", curve_name="t", parameter=math.pi / 2, name=None, length=None, color=None
        )

        self.assertIn("vertical asymptote", str(traced["result"]))

    def test_large_or_steep_tangents_follow_the_magnitude_rule(self) -> None:
        # (expression, right bound, x, expected error fragment or None when the tangent is drawn)
        cases: List[Tuple[str, float, float, Optional[str]]] = [
            ("exp(x)", 30, 28, "too large to draw here"),
            ("x^3", 20000, 10001, "too large to draw here"),
            ("tan(x)", 4, math.pi / 2 - 1e-6, "too steep to draw here"),
            ("1/x", 5, 1e-6, "too steep to draw here"),
            ("exp(x)", 30, 27, None),
            ("10^13*x", 5, 0.01, None),
            ("tan(x)", 4, math.pi / 2 - 1e-4, None),
        ]
        for expression, right, x, fragment in cases:
            with self.subTest(expression=expression, x=x):
                self.setUp()
                self.draw(expression, "f", -4, right)
                depth = self.undo_depth()

                traced = self.run_call(
                    "draw_tangent_line", curve_name="f", parameter=x, name=None, length=None, color=None
                )

                segments = self.canvas.drawable_manager.drawables.Segments
                if fragment is None:
                    self.assertFalse(traced["is_error"], traced["result"])
                    self.assertEqual(len(segments), 1)
                else:
                    self.assertTrue(traced["is_error"], traced["result"])
                    self.assertIn(fragment, str(traced["result"]))
                    self.assertNotIn("no tangent", str(traced["result"]))
                    self.assertEqual(len(segments), 0)
                    self.assertEqual(self.undo_depth(), depth)

    def test_tangent_away_from_the_asymptote_still_works(self) -> None:
        self.draw("tan(x)", "t", -4, 4)

        traced = self.run_call("draw_tangent_line", curve_name="t", parameter=1, name=None, length=None, color=None)

        self.assertFalse(traced["is_error"], traced["result"])
        segments = self.canvas.drawable_manager.drawables.Segments
        self.assertEqual(len(segments), 1)
        segment = segments[0]
        slope = (segment.point2.y - segment.point1.y) / (segment.point2.x - segment.point1.x)
        self.assertAlmostEqual(slope, 1 / math.cos(1) ** 2, places=4)


class TestFunctionAreaTranslation(_FunctionToolTestCase):
    """K14: an area shaded under a function follows the function when it is translated."""

    def shade(self, drawable1: str, drawable2: Optional[str], left: Optional[float], right: Optional[float]) -> Any:
        traced = self.run_call(
            "create_colored_area",
            drawable1_name=drawable1,
            drawable2_name=drawable2,
            left_bound=left,
            right_bound=right,
            color="orange",
            opacity=0.4,
        )
        self.assertFalse(traced["is_error"], traced["result"])
        areas = self.canvas.drawable_manager.drawables.FunctionsBoundedColoredAreas
        self.assertEqual(len(areas), 1)
        return areas[0]

    def test_area_under_a_function_moves_with_it(self) -> None:
        self.draw("x^2", "f", -3, 3)
        area = self.shade("f", None, 0, 2)

        self.run_call("translate_object", name="f", x_offset=3, y_offset=0)

        self.assertEqual((area.left_bound, area.right_bound), (3, 5))
        self.assertEqual(area._get_bounds(), (3, 5))

    def test_area_between_two_functions_keeps_its_bounds_when_one_moves(self) -> None:
        self.draw("x^2", "f", -5, 5)
        self.draw("x", "g", -5, 5)
        area = self.shade("f", "g", 0, 1)

        self.run_call("translate_object", name="f", x_offset=3, y_offset=0)

        self.assertEqual((area.left_bound, area.right_bound), (0, 1))

    def test_area_without_explicit_bounds_follows_the_function_bounds(self) -> None:
        self.draw("x^2", "f", -3, 3)
        area = self.shade("f", None, None, None)

        self.run_call("translate_object", name="f", x_offset=3, y_offset=0)

        self.assertEqual((area.left_bound, area.right_bound), (None, None))
        self.assertEqual(area._get_bounds(), (0, 6))

    def test_undo_restores_the_area_bounds(self) -> None:
        self.draw("x^2", "f", -3, 3)
        self.shade("f", None, 0, 2)
        self.run_call("translate_object", name="f", x_offset=3, y_offset=0)

        self.run_call("undo")

        area = self.canvas.drawable_manager.drawables.FunctionsBoundedColoredAreas[0]
        self.assertEqual((area.left_bound, area.right_bound), (0, 2))


# (expression, left, right, vertical asymptotes, point discontinuities)
# abs(...) corners are always listed as point discontinuities, hence some 0s in the last list.
_SINGULARITY_CASES: List[Tuple[str, Optional[float], Optional[float], List[float], List[float]]] = [
    # Holes
    ("(x^2-1)/(x-1)", None, None, [], [1.0]),
    ("(x-1)/(x^2-1)", None, None, [-1.0], [1.0]),
    ("sin(x)/x", None, None, [], [0.0]),
    ("(x^2-10^6)/(x-1000)", None, None, [], [1000.0]),
    # Holes whose samples cancel badly or are large
    ("(e^x-1)/x", None, None, [], [0.0]),
    ("(1-cos(x))/x^2", None, None, [], [0.0]),
    ("(exp(x)-1-x-x^2/2)/x^3", None, None, [], [0.0]),
    ("10^15*(x^2-1)/(x-1)", None, None, [], [1.0]),
    # Holes within 0.2% of another pole: sampling starts closer than the pole
    ("(x^2-1)/((x-1)*(x-1.001))", None, None, [1.001], [1.0]),
    ("(x-1)/((x-1)*(x-1.002))", None, None, [1.002], [1.0]),
    # A slow limit (|x|^0.25 -> 0) is not an asymptote
    ("x/abs(x)^0.75", None, None, [], [0.0]),
    # Bounded without a limit: a point discontinuity, not an asymptote
    ("sin(1/x)", None, None, [], [0.0]),
    # Asymptotes
    ("1/x", None, None, [0.0], []),
    ("1/x^2", None, None, [0.0], []),
    ("1/(x^2-1)", None, None, [-1.0, 1.0], []),
    ("sqrt(x)/x", None, None, [0.0], []),
    ("exp(1/x)", None, None, [0.0], []),
    ("1/sqrt(abs(x))", None, None, [0.0], [0.0]),
    ("sin(1/x)/x", None, None, [0.0], []),
    ("log(x)", None, None, [0.0], []),
    ("tan(x)", -2, 2, [round(-math.pi / 2, 6), round(math.pi / 2, 6)], []),
    # Slowly growing asymptotes; pure power growth is caught for any exponent (1/x^0.004 too)
    ("log(1/x)", None, None, [0.0], []),
    ("log(1/x^2)", None, None, [0.0], []),
    ("ln(1/abs(x))", None, None, [0.0], [0.0]),
    ("1/x^0.1", None, None, [0.0], []),
    ("1/x^(1/8)", None, None, [0.0], []),
    ("1/x^0.05", None, None, [0.0], []),
    ("1/x^0.004", None, None, [0.0], []),
    ("log(log(1/x))", None, None, [0.0, 1.0], []),  # log(1/x) = 0 at x = 1 too
    # Adding a constant changes nothing
    ("ln(1/x)+500", None, None, [0.0], []),
    ("ln(1/abs(x))+1000", None, None, [0.0], [0.0]),
    ("1/x^0.1+100", None, None, [0.0], []),
    ("1/sqrt(abs(x))+20000", None, None, [0.0], [0.0]),
    ("1/x+10^7", None, None, [0.0], []),
    # Neither
    ("x^2 + 1", None, None, [], []),
]


class TestRemovableDiscontinuities(_FunctionToolTestCase):
    """K24: a zero of a denominator is an asymptote only where f grows; otherwise a point discontinuity."""

    def test_singularities_are_classified(self) -> None:
        for expression, left, right, asymptotes, holes in _SINGULARITY_CASES:
            with self.subTest(expression=expression):
                vertical, _, discontinuities = MathUtils.calculate_asymptotes_and_discontinuities(
                    expression, left, right
                )
                self.assertEqual(_rounded(vertical), asymptotes)
                self.assertEqual(_rounded(discontinuities), holes)

    def test_hole_and_asymptotes_of_the_same_denominator_are_told_apart(self) -> None:
        vertical, _, discontinuities = MathUtils.calculate_asymptotes_and_discontinuities("x/sin(x)", -7, 7)

        self.assertEqual(_rounded(discontinuities), [0.0])
        self.assertNotIn(0.0, _rounded(vertical))
        self.assertTrue(vertical, "the zeros of sin(x) other than 0 are asymptotes")
        for asymptote in vertical:
            self.assertAlmostEqual(math.sin(asymptote), 0.0, places=6)

    def test_holes_outside_the_bounds_are_not_listed(self) -> None:
        _, _, discontinuities = MathUtils.calculate_asymptotes_and_discontinuities("(x^2-1)/(x-1)", 2, 5)

        self.assertEqual(discontinuities, [])

    def test_drawn_function_with_a_hole_lists_no_asymptote(self) -> None:
        self.draw("(x^2 - 1)/(x - 1)", "h", None, None, undefined_at=[1])

        args = self.function_args("h")
        self.assertNotIn("vertical_asymptotes", args)
        self.assertEqual(_rounded(args.get("point_discontinuities")), [1.0])
        self.assertAlmostEqual(self.function("h").function(2), 3.0)
