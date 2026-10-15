"""Known-answer tests for the CAS tools (derive, integrate, limit, solve, simplify, systems).

Each case is one nerdamer got wrong before its results were checked numerically
(utils.symbolic_checks): a misread function name, a wrong antiderivative, a wrong limit, a
lost or spurious root. Answers are compared by value, not by spelling.
"""

from __future__ import annotations

import json
import math
import unittest
from typing import Dict, List, Optional

from browser import window

from utils.math_utils import MathUtils


def _value(text: str, scope: Optional[Dict[str, float]] = None) -> float:
    return float(window.math.evaluate(str(text).replace("**", "^"), scope or {}))


def _real_roots(result: str) -> List[float]:
    return sorted(float(item) for item in json.loads(result.split(" (")[0]))


class TestCasKnownAnswers(unittest.TestCase):
    def assert_same_function(self, actual: str, expected: str, points: List[float]) -> None:
        self.assertFalse(str(actual).startswith("Error"), actual)
        for point in points:
            self.assertAlmostEqual(_value(actual, {"x": point}), _value(expected, {"x": point}), places=9, msg=actual)

    def assert_number(self, result: str, expected: float, places: int = 6) -> None:
        self.assertFalse(str(result).startswith("Error"), result)
        self.assertAlmostEqual(float(str(result).split(" ")[0]), expected, places=places, msg=result)

    # ----- derivatives ------------------------------------------------------------------

    def test_derivative_of_ln_and_other_logs(self) -> None:
        points = [0.5, 2.0, 3.7]
        self.assert_same_function(MathUtils.derivative("ln(x)", "x"), "1/x", points)
        self.assert_same_function(MathUtils.derivative("log2(x)", "x"), "1/(x*log(2))", points)
        self.assert_same_function(MathUtils.derivative("log(x, 2)", "x"), "1/(x*log(2))", points)
        self.assert_same_function(MathUtils.derivative("ln(sin(x))", "x"), "cos(x)/sin(x)", [0.5, 1.2])
        self.assert_same_function(MathUtils.derivative("sgn(x)*x^2", "x"), "2*abs(x)", [-1.5, 0.7, 2.0])

    def test_unknown_function_is_refused(self) -> None:
        result = MathUtils.derivative("gamma(x)", "x")
        self.assertTrue(result.startswith("Error"), result)
        self.assertIn("gamma()", result)

    def test_wrong_derivative_is_refused(self) -> None:
        # nerdamer answers 0
        result = MathUtils.derivative("atan2(y,x)", "x")
        self.assertTrue(result.startswith("Error"), result)

    # ----- integrals --------------------------------------------------------------------

    def test_antiderivatives_of_logs(self) -> None:
        points = [0.5, 2.0, 3.0]
        result = MathUtils.integral("ln(x)", "x")
        for a, b in ((0.5, 2.0), (2.0, 3.0)):
            self.assertAlmostEqual(
                _value(result, {"x": b}) - _value(result, {"x": a}),
                b * math.log(b) - b - (a * math.log(a) - a),
                places=9,
            )
        self.assert_same_function(MathUtils.integral("x^3*ln(x)", "x"), "x^4*log(x)/4 - x^4/16", points)

    def test_wrong_antiderivatives_are_refused(self) -> None:
        for expression in ("1/(x^3+1)", "1/(x^2+2*x+5)", "sin(x^2)"):
            result = MathUtils.integral(expression, "x")
            self.assertTrue(result.startswith("Error"), f"{expression}: {result}")

    def test_definite_integrals(self) -> None:
        self.assert_number(MathUtils.integral("sqrt(1-x^2)", "x", -1, 1), math.pi / 2, places=9)
        self.assert_number(MathUtils.integral("ln(x)", "x", 1, 2), 2 * math.log(2) - 1, places=9)
        self.assert_number(MathUtils.integral("sin(x^2)", "x", 0, 1), 0.3102683017233811, places=9)
        self.assert_number(MathUtils.integral("1/(x^3+1)", "x", 0, 1), 0.8356488482647211, places=9)
        self.assert_number(MathUtils.integral("abs(sin(x))", "x", 0, "2*pi"), 4.0, places=9)
        self.assert_number(MathUtils.integral("1/(x^2-1)", "x", -0.5, 0.5), -math.log(3), places=9)

    def test_definite_integral_needs_both_bounds(self) -> None:
        result = MathUtils.integral("x", "x", 0, None)
        self.assertTrue(result.startswith("Error"), result)

    # ----- limits -----------------------------------------------------------------------

    def test_limits_with_a_value(self) -> None:
        self.assert_number(MathUtils.limit("ln(x)/x", "x", "inf"), 0.0)
        self.assert_number(MathUtils.limit("ln(1+x)/x", "x", "0"), 1.0)
        self.assert_number(MathUtils.limit("sqrt(x^2+x)-x", "x", "inf"), 0.5)
        self.assert_number(MathUtils.limit("(x-sin(x))/x^3", "x", "0"), 1 / 6)
        self.assert_number(MathUtils.limit("(1-x)^(1/x)", "x", "0"), math.exp(-1))
        self.assert_number(MathUtils.limit("x*e^x", "x", "-inf"), 0.0)
        self.assert_number(MathUtils.limit("atan(x)", "x", "inf"), math.pi / 2)

    def test_one_sided_limits(self) -> None:
        result = MathUtils.limit("ln(x)", "x", "0")
        self.assertTrue(result.startswith("-Infinity"), result)
        self.assertIn("only on the right", result)
        self.assertTrue(MathUtils.limit("x*ln(x)", "x", "0").startswith("0"))

    def test_limits_that_do_not_exist(self) -> None:
        for expression, target in (
            ("1/x", "0"),
            ("abs(x)/x", "0"),
            ("sin(1/x)", "0"),
            ("tan(x)", "pi/2"),
            ("floor(x)", "1"),
            ("e^(1/x)", "0"),
            ("sin(x)", "inf"),
        ):
            result = MathUtils.limit(expression, "x", target)
            self.assertTrue(result.startswith("The limit does not exist"), f"{expression} -> {target}: {result}")

    # ----- equations --------------------------------------------------------------------

    def test_spurious_roots_are_dropped(self) -> None:
        self.assertEqual(_real_roots(MathUtils.solve("x/(x-1)=2", "x")), [2.0])
        self.assertEqual(_real_roots(MathUtils.solve("(x^2-1)/(x-1)=0", "x")), [-1.0])
        roots = _real_roots(MathUtils.solve("log(x)+log(x-1)=log(6)", "x"))
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 3.0, places=9)

    def test_duplicate_roots_are_merged(self) -> None:
        roots = _real_roots(MathUtils.solve("e^(2*x)-3*e^x+2=0", "x"))
        self.assertEqual(len(roots), 2)
        self.assertAlmostEqual(roots[1], math.log(2), places=9)

    def test_cubic_roots_are_all_found(self) -> None:
        roots = _real_roots(MathUtils.solve("x^3-3*x+1=0", "x"))
        expected = sorted(2 * math.cos(2 * math.pi * k / 9) for k in (1, 2, 4))
        self.assertEqual(len(roots), 3)
        for root, value in zip(roots, expected):
            self.assertAlmostEqual(root, value, places=9)
        result = MathUtils.solve("x^3+x+1=0", "x")
        self.assertEqual(result.count(",") + 1, 3, result)
        self.assertIn("-0.682327803828019", result)

    def test_ln_equation(self) -> None:
        roots = _real_roots(MathUtils.solve("ln(x)=2", "x"))
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], math.exp(2), places=9)

    def test_identity_and_contradiction(self) -> None:
        self.assertIn("identity", MathUtils.solve("x=x", "x"))
        self.assertEqual(MathUtils.solve("x=x+1", "x"), "[]")
        self.assertEqual(MathUtils.solve("1/x=0", "x"), "[]")

    def test_periodic_roots_are_listed_in_one_window(self) -> None:
        result = MathUtils.solve("sin(x)=0", "x")
        self.assertIn("more outside it", result)
        roots = _real_roots(result)
        self.assertEqual(len(roots), 5)
        self.assertAlmostEqual(roots[-1], 2 * math.pi, places=9)

    # ----- simplify and systems ---------------------------------------------------------

    def test_simplify_with_ln(self) -> None:
        self.assert_number(MathUtils.simplify("ln(e^2)"), 2.0)

    def test_degenerate_linear_systems(self) -> None:
        dependent = MathUtils.solve_linear_system(["x+y=2", "2*x+2*y=4"])
        self.assertTrue(dependent.startswith("Infinitely many solutions"), dependent)
        inconsistent = MathUtils.solve_linear_system(["x+y=2", "x+y=3"])
        self.assertTrue(inconsistent.startswith("No solution"), inconsistent)
