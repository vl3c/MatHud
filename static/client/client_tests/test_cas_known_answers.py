"""Known-answer tests for the CAS tools (derive, integrate, limit, solve, simplify, systems).

TestCasKnownAnswers runs with whichever engine is available (the server's SymPy, else
nerdamer); TestCasKnownAnswersNerdamer runs the same cases on the nerdamer path, whose
results utils.symbolic_checks verifies numerically. Each case is one nerdamer got wrong
before: a misread function name, a wrong antiderivative or limit, a lost or spurious root.
Answers are compared by value, not by spelling.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, List, Optional

from browser import window

from utils.math_utils import MathUtils
from utils.sympy_client import SympyClient

from .cas_engines import real_roots, roots_of, value_of


def _value(text: str, scope: Optional[Dict[str, float]] = None) -> float:
    return float(window.math.evaluate(str(text).replace("**", "^"), scope or {}))


class TestCasKnownAnswers(unittest.TestCase):
    def assert_same_function(self, actual: str, expected: str, points: List[float]) -> None:
        self.assertFalse(str(actual).startswith("Error"), actual)
        for point in points:
            self.assertAlmostEqual(_value(actual, {"x": point}), _value(expected, {"x": point}), places=9, msg=actual)

    def assert_number(self, result: Any, expected: float, places: int = 6) -> None:
        self.assertFalse(str(result).startswith("Error"), result)
        self.assertAlmostEqual(value_of(result), expected, places=places, msg=result)

    def assert_antiderivative(self, result: str, integrand: str, points: List[float]) -> None:
        """F(b) - F(a) equals the integral of the integrand between neighbouring points."""
        self.assertFalse(str(result).startswith("Error"), result)
        for a, b in zip(points, points[1:]):
            exact = value_of(MathUtils.integral(integrand, "x", a, b))
            self.assertAlmostEqual(_value(result, {"x": b}) - _value(result, {"x": a}), exact, places=6, msg=result)

    # ----- derivatives ------------------------------------------------------------------

    def test_derivative_of_ln_and_other_logs(self) -> None:
        points = [0.5, 2.0, 3.7]
        self.assert_same_function(MathUtils.derivative("ln(x)", "x"), "1/x", points)
        self.assert_same_function(MathUtils.derivative("2ln(x)", "x"), "2/x", points)
        self.assert_same_function(MathUtils.derivative("log2(x)", "x"), "1/(x*log(2))", points)
        self.assert_same_function(MathUtils.derivative("log(x, 2)", "x"), "1/(x*log(2))", points)
        self.assert_same_function(MathUtils.derivative("ln(sin(x))", "x"), "cos(x)/sin(x)", [0.5, 1.2])
        self.assert_same_function(MathUtils.derivative("sgn(x)*x^2", "x"), "2*abs(x)", [-1.5, 0.7, 2.0])

    def test_unknown_function_is_refused(self) -> None:
        result = MathUtils.derivative("foo(x)", "x")
        self.assertTrue(result.startswith("Error"), result)
        self.assertIn("foo()", result)

    # ----- integrals --------------------------------------------------------------------

    def test_antiderivatives(self) -> None:
        points = [0.5, 2.0, 3.0]
        self.assert_same_function(MathUtils.integral("x^3*ln(x)", "x"), "x^4*log(x)/4 - x^4/16", points)
        self.assert_antiderivative(MathUtils.integral("ln(x)", "x"), "ln(x)", points)
        self.assert_antiderivative(MathUtils.integral("2ln(x)", "x"), "2ln(x)", points)

    def test_expanded_polynomials_pass_the_check(self) -> None:
        # Rounding in an expanded polynomial near its root must not fail the numeric check
        self.assert_antiderivative(MathUtils.integral("(x-2)^10", "x"), "(x-2)^10", [1.5, 2.2, 3.1])
        self.assertFalse(MathUtils.expand("(x-3)^15").startswith("Error"))

    def test_definite_integrals(self) -> None:
        self.assert_number(MathUtils.integral("sqrt(1-x^2)", "x", -1, 1), math.pi / 2, places=9)
        self.assert_number(MathUtils.integral("ln(x)", "x", 1, 2), 2 * math.log(2) - 1, places=9)
        self.assert_number(MathUtils.integral("1/(x^3+1)", "x", 0, 1), 0.8356488482647211, places=9)
        self.assert_number(MathUtils.integral("abs(sin(x))", "x", 0, "2*pi"), 4.0, places=9)
        self.assert_number(MathUtils.integral("1/(x^2-1)", "x", -0.5, 0.5), -math.log(3), places=9)

    def test_definite_integrals_without_elementary_values(self) -> None:
        # SymPy gives Si, fresnels or Ei forms with the numeric value; nerdamer the numeric value
        for expression, lower, upper, expected in (
            ("sin(x^2)", 0, 1, 0.3102683017233811),
            ("(e^x-1)/x", -1, 1, 2.1145017507435094),
            ("sin(x)/x", -1, 1, 1.8921661407343662),
        ):
            result = MathUtils.integral(expression, "x", lower, upper)
            self.assertFalse(result.startswith("Error"), result)
            number = result.split("about ")[1].split(" ")[0] if "about " in result else result.split(" ")[0]
            self.assertAlmostEqual(float(number), expected, places=9, msg=result)

    def test_definite_integral_needs_both_bounds(self) -> None:
        result = MathUtils.integral("x", "x", 0, None)
        self.assertTrue(result.startswith("Error"), result)

    def test_symbolic_bound(self) -> None:
        self.assert_same_function(MathUtils.integral("x^2", "x", 0, "x"), "x^3/3", [0.5, 2.0])

    # ----- limits -----------------------------------------------------------------------

    def test_limits_with_a_value(self) -> None:
        self.assert_number(MathUtils.limit("ln(x)/x", "x", "inf"), 0.0)
        self.assert_number(MathUtils.limit("ln(1+x)/x", "x", "0"), 1.0)
        self.assert_number(MathUtils.limit("sqrt(x^2+x)-x", "x", "inf"), 0.5)
        self.assert_number(MathUtils.limit("(x-sin(x))/x^3", "x", "0"), 1 / 6)
        self.assert_number(MathUtils.limit("(1-x)^(1/x)", "x", "0"), math.exp(-1))
        self.assert_number(MathUtils.limit("x*e^x", "x", "-inf"), 0.0)
        self.assert_number(MathUtils.limit("atan(x)", "x", "inf"), math.pi / 2)
        self.assert_number(MathUtils.limit("(e^x-1-x)/x^2", "x", "0"), 0.5)

    def test_slow_limits_are_not_infinite(self) -> None:
        # Monotone but slow: the samples alone must not call these infinite
        self.assert_number(MathUtils.limit("1/log(x)", "x", "inf"), 0.0)
        self.assert_number(MathUtils.limit("x^(1/20)", "x", "0"), 0.0)
        self.assert_number(MathUtils.limit("1/log(abs(x))", "x", "0"), 0.0)

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

    def test_cancellation_noise_is_not_oscillation(self) -> None:
        result = MathUtils.limit("(e^x-e^(-x)-2*x)/x^3", "x", "0")
        self.assertNotIn("does not exist", result)

    # ----- equations --------------------------------------------------------------------

    def test_spurious_roots_are_dropped(self) -> None:
        self.assertEqual(real_roots(MathUtils.solve("x/(x-1)=2", "x")), [2.0])
        self.assertEqual(real_roots(MathUtils.solve("(x^2-1)/(x-1)=0", "x")), [-1.0])
        self.assertEqual(real_roots(MathUtils.solve("log(x)+log(x-1)=log(6)", "x")), [3.0])

    def test_duplicate_roots_are_merged(self) -> None:
        self.assertEqual(real_roots(MathUtils.solve("e^(2*x)-3*e^x+2=0", "x")), [0.0, round(math.log(2), 9)])

    def test_repeated_polynomial_roots(self) -> None:
        for equation, root in (("(x-1)^5=0", 1.0), ("(x-3)^8=0", 3.0), ("(x-2)^6=0", 2.0)):
            result = MathUtils.solve(equation, "x")
            self.assertEqual([complex(round(z.real, 9), round(z.imag, 9)) for z in roots_of(result)], [root], result)

    def test_cubic_roots_are_all_found(self) -> None:
        roots = real_roots(MathUtils.solve("x^3-3*x+1=0", "x"))
        expected = sorted(round(2 * math.cos(2 * math.pi * k / 9), 9) for k in (1, 2, 4))
        self.assertEqual(roots, expected)
        result = MathUtils.solve("x^3+x+1=0", "x")
        self.assertEqual(len(roots_of(result)), 3, result)
        self.assertEqual(real_roots(result), [-0.682327804])

    def test_ln_equation(self) -> None:
        self.assertEqual(real_roots(MathUtils.solve("ln(x)=2", "x")), [round(math.exp(2), 9)])

    def test_identity_and_contradiction(self) -> None:
        self.assertIn("identity", MathUtils.solve("x=x", "x"))
        self.assertEqual(MathUtils.solve("x=x+1", "x"), "[]")
        self.assertEqual(MathUtils.solve("1/x=0", "x"), "[]")

    def test_periodic_roots_are_listed_in_one_window(self) -> None:
        result = MathUtils.solve("sin(x)=0", "x")
        self.assertIn("[-2*pi, 2*pi]", result)
        roots = real_roots(result)
        self.assertEqual(len(roots), 5)
        self.assertAlmostEqual(roots[-1], 2 * math.pi, places=9)

    # ----- simplify and systems ---------------------------------------------------------

    def test_simplify_with_ln(self) -> None:
        self.assert_number(MathUtils.simplify("ln(e^2)"), 2.0)

    def test_linear_systems_without_a_distinct_solution(self) -> None:
        dependent = MathUtils.solve_linear_system(["x+y=2", "2*x+2*y=4"])
        self.assertTrue(dependent.startswith("Infinitely many solutions"), dependent)
        inconsistent = MathUtils.solve_linear_system(["x+y=2", "x+y=3"])
        self.assertTrue(inconsistent.startswith("No solution"), inconsistent)
        overdetermined = MathUtils.solve_linear_system(["x+y=1", "x+y=1", "x-y=0"])
        self.assertEqual(overdetermined, "x = 0.5, y = 0.5")


class TestCasKnownAnswersNerdamer(TestCasKnownAnswers):
    """The same cases with SymPy unavailable, plus the nerdamer results the checks refuse."""

    def setUp(self) -> None:
        self._saved = SympyClient._unavailable
        SympyClient._unavailable = True

    def tearDown(self) -> None:
        SympyClient._unavailable = self._saved

    def test_cancellation_noise_is_not_oscillation(self) -> None:
        # nerdamer has no answer and the samples cannot settle it, but they do not claim oscillation
        result = MathUtils.limit("(e^x-e^(-x)-2*x)/x^3", "x", "0")
        self.assertNotIn("does not exist", result)

    def test_unknown_function_is_refused(self) -> None:
        result = MathUtils.derivative("gamma(x)", "x")
        self.assertTrue(result.startswith("Error"), result)
        self.assertIn("gamma()", result)

    def test_wrong_derivative_is_refused(self) -> None:
        # nerdamer answers 0
        result = MathUtils.derivative("atan2(y,x)", "x")
        self.assertTrue(result.startswith("Error"), result)

    def test_wrong_antiderivatives_are_refused(self) -> None:
        for expression in ("1/(x^3+1)", "1/(x^2+2*x+5)", "sin(x^2)"):
            result = MathUtils.integral(expression, "x")
            self.assertTrue(result.startswith("Error"), f"{expression}: {result}")

    def test_definite_integrals_without_elementary_values(self) -> None:
        for expression, lower, upper, expected in (
            ("sin(x^2)", 0, 1, 0.3102683017233811),
            ("(e^x-1)/x", -1, 1, 2.1145017507435094),
        ):
            result = MathUtils.integral(expression, "x", lower, upper)
            self.assertIn("computed numerically", result)
            self.assertAlmostEqual(value_of(result), expected, places=9, msg=result)


class TestCasSympyAnswers(unittest.TestCase):
    """Answers only the server's SymPy engine gives exactly (skipped when it is unavailable)."""

    def setUp(self) -> None:
        if SympyClient.compute("simplify", expression="x") is None:
            self.skipTest("the server has no SymPy")

    def test_exact_results(self) -> None:
        self.assertEqual(MathUtils.limit("sqrt(x^2+x)-x", "x", "inf"), "1/2")
        self.assertEqual(MathUtils.limit("(e^x-e^(-x)-2*x)/x^3", "x", "0"), "1/3")
        self.assertEqual(MathUtils.integral("1/(x^2+2*x+5)", "x"), "atan(x/2+1/2)/2")
        self.assertEqual(MathUtils.integral("sqrt(1-x^2)", "x", -1, 1), "pi/2")
        self.assertEqual(MathUtils.simplify("(x^3-8)/(x-2)"), "x^2+2*x+4")
        self.assertEqual(MathUtils.solve("ln(x)=2", "x"), "[exp(2)]")
        self.assertEqual(MathUtils.derivative("atan2(y,x)", "x"), "-y/(x^2+y^2)")

    def test_general_solution_of_a_periodic_equation(self) -> None:
        result = MathUtils.solve("tan(x)=1", "x")
        self.assertIn("all solutions: x = pi*n+pi/4, for any integer n", result)
