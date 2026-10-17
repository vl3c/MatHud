"""Known-answer tests for the CAS tools (derive, integrate, limit, solve, simplify, systems).

TestCasKnownAnswers runs on the server's SymPy engine, each result checked numerically by
utils.symbolic_checks. Many cases are ones the earlier nerdamer engine got wrong: a misread
function name, a wrong antiderivative or limit, a lost or spurious root. Answers are compared
by value, not by spelling. TestCasWithoutSympy pins a server without SymPy: the symbolic
tools give an error, the numeric ones still answer.
"""

from __future__ import annotations

import json
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

    def test_checks_read_ln_and_sgn(self) -> None:
        # math.js has no ln or sgn; the numeric checks of every tool must still read them
        from utils.symbolic_checks import SymbolicChecks

        self.assertIs(SymbolicChecks.equivalent("x*ln(x)", "x*log(x)"), True)
        self.assertIs(SymbolicChecks.equivalent("sgn(x)*x", "abs(x)"), True)
        self.assertIs(SymbolicChecks.equivalent("ln(x)", "log(x, 10)"), False)

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
        # SymPy gives Si, fresnels or Ei forms with the numeric value
        for expression, lower, upper, expected in (
            ("sin(x^2)", 0, 1, 0.3102683017233811),
            ("(e^x-1)/x", -1, 1, 2.1145017507435094),
            ("sin(x)/x", -1, 1, 1.8921661407343662),
        ):
            result = MathUtils.integral(expression, "x", lower, upper)
            self.assertFalse(result.startswith("Error"), result)
            number = result.split("about ")[1].split(" ")[0] if "about " in result else result.split(" ")[0]
            self.assertAlmostEqual(float(number), expected, places=9, msg=result)

    def test_piecewise_antiderivatives(self) -> None:
        for expression in ("abs(x)", "sign(x)"):
            result = MathUtils.integral(expression, "x")
            self.assertFalse(result.startswith("Error"), f"{expression}: {result}")

    def test_divergent_integral_has_no_complex_value(self) -> None:
        result = MathUtils.integral("1/x", "x", "-Infinity", -1)
        self.assertNotIn("pi", result)
        self.assertNotIn("i*", result)

    def test_unevaluated_derivative_is_not_returned(self) -> None:
        self.assertNotIn("Derivative", MathUtils.derivative("floor(x)", "x"))

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

    def test_repeated_roots_with_decimal_coefficients(self) -> None:
        for equation, root in (("(x-0.3)^3=0", 0.3), ("(x-0.1)^5=0", 0.1)):
            result = MathUtils.solve(equation, "x")
            self.assertEqual([complex(round(z.real, 9), round(z.imag, 9)) for z in roots_of(result)], [root], result)

    def test_complex_roots_of_a_rational_equation(self) -> None:
        result = MathUtils.solve("1/x=-x", "x")
        self.assertEqual(sorted((z.real, z.imag) for z in roots_of(result)), [(0.0, -1.0), (0.0, 1.0)], result)

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

    def test_systems_without_a_distinct_solution(self) -> None:
        dependent = MathUtils.solve_system_of_equations(["x+y=2", "2*x+2*y=4"])
        self.assertTrue(dependent.startswith("Infinitely many solutions"), dependent)
        inconsistent = MathUtils.solve_system_of_equations(["x+y=2", "x+y=3"])
        self.assertTrue(inconsistent.startswith("No solution"), inconsistent)
        overdetermined = MathUtils.solve_system_of_equations(["x+y=1", "x+y=1", "x-y=0"])
        self.assertEqual(overdetermined, "x = 1/2, y = 1/2")


class TestCasWithoutSympy(unittest.TestCase):
    """A server without SymPy: the symbolic tools say so, the numeric ones still answer."""

    def setUp(self) -> None:
        self._saved = SympyClient._unavailable
        SympyClient._unavailable = True

    def tearDown(self) -> None:
        SympyClient._unavailable = self._saved

    def assert_needs_sympy(self, result: str) -> None:
        self.assertTrue(result.startswith("Error: "), result)
        self.assertIn("needs SymPy, and the server's SymPy engine is not available", result)

    def assert_numeric(self, result: str, expected: float) -> None:
        self.assertFalse(result.startswith("Error"), result)
        self.assertIn("SymPy engine is not available", result)
        self.assertAlmostEqual(value_of(result), expected, places=9, msg=result)

    def test_symbolic_tools_need_sympy(self) -> None:
        self.assert_needs_sympy(MathUtils.derivative("x^2", "x"))
        self.assert_needs_sympy(MathUtils.simplify("ln(e^2)"))
        self.assert_needs_sympy(MathUtils.expand("(x+1)^2"))
        self.assert_needs_sympy(MathUtils.factor("x^2-1"))
        self.assert_needs_sympy(MathUtils.integral("x^2", "x"))

    def test_definite_integrals_are_computed_numerically(self) -> None:
        self.assert_numeric(MathUtils.integral("x^2", "x", 0, 3), 9.0)
        self.assert_numeric(MathUtils.integral("ln(x)", "x", 1, 2), 2 * math.log(2) - 1)
        self.assert_numeric(MathUtils.integral("sin(x^2)", "x", 0, 1), 0.3102683017233811)

    def test_limits_are_estimated_numerically(self) -> None:
        self.assert_numeric(MathUtils.limit("sin(x)/x", "x", "0"), 1.0)
        self.assert_numeric(MathUtils.limit("ln(x)/x", "x", "inf"), 0.0)

    def test_equations_are_solved_numerically(self) -> None:
        result = MathUtils.solve("x^2-4=0", "x")
        self.assertTrue(result.startswith("[-2,2] (the real roots found numerically"), result)
        self.assertEqual(real_roots(MathUtils.solve("ln(x)=2", "x")), [round(math.exp(2), 9)])
        self.assertEqual(real_roots(MathUtils.solve("(1-x)*e^(-x)=0", "x")), [1.0])
        self.assertTrue(MathUtils.solve("x^2+1=0", "x").startswith("[] (no real root was found numerically"))
        self.assertTrue(MathUtils.solve("x=x", "x").startswith("Every real x satisfies x = x"))
        self.assertEqual(MathUtils.solve("1e-8*x=2e-8", "x").split(" (")[0], "[2]")
        with_parameter = MathUtils.solve("x + y = 3", "x")
        self.assertTrue(with_parameter.startswith("Error: "), with_parameter)
        self.assertIn("needs an equation in x alone; besides x it contains y", with_parameter)

    def test_roots_in_an_interval_are_found_numerically(self) -> None:
        self.assertEqual(MathUtils.solve("sin(x) = 0", "x", "[0, 2pi)").split(" (")[0], "[0,pi]")
        self.assertEqual(MathUtils.solve("x^2 = 4", "x", "[-2, 2)").split(" (")[0], "[-2]")
        self.assertEqual(MathUtils.solve("(x-1)(x+2) = 0", "x", "[-5, 5]").split(" (")[0], "[-2,1]")

    def test_systems_are_solved_numerically(self) -> None:
        result = MathUtils.solve_system_of_equations(["y = x^2 - 5", "x^2 + y^2 = 25"])
        self.assertTrue(result.startswith("x1 = -3, y1 = 4, x2 = 0, y2 = -5, x3 = 3, y3 = 4 ("), result)
        self.assertEqual(MathUtils.solve_system_of_equations(["x+y=2", "x-y=0"]), "x = 1, y = 1")

    def test_linear_systems_are_solved_by_elimination(self) -> None:
        self.assertEqual(MathUtils.solve_system_of_equations(["x/3+y/7=1", "x-y=2"]), "x = 2.7, y = 0.7")
        dependent = MathUtils.solve_system_of_equations(["x+y=1", "2x+2y=2"])
        self.assertEqual(
            dependent,
            "Infinitely many solutions: the equations are dependent (1 independent equation for 2 unknowns): "
            "x = 1-y, for any y.",
        )
        inconsistent = MathUtils.solve_system_of_equations(["x+y=1", "x+y=2"])
        self.assertEqual(inconsistent, "No solution: the equations are inconsistent (they contradict each other).")

    def test_numeric_integrate_reads_ln(self) -> None:
        result = MathUtils.numeric_integrate("ln(x)", "x", 1, 2)
        self.assertAlmostEqual(result["value"], 2 * math.log(2) - 1, places=9)

    def test_numeric_integrate_names_other_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "may contain only x; it also contains a"):
            MathUtils.numeric_integrate("a*x", "x", 0, 1)


class TestCasSympyAnswers(unittest.TestCase):
    """SymPy's exact answers."""

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

    def test_dense_periodic_roots_shrink_the_window(self) -> None:
        result = MathUtils.solve("tan(60*x)=0", "x")
        self.assertIn("(the roots in [-29*pi/60, 29*pi/60]; all solutions: x = pi*n/60", result)
        self.assertEqual(len(roots_of(result)), 59)

    def test_solutions_not_affine_in_n_have_no_general_form(self) -> None:
        result = MathUtils.solve("sin(x^2)=0", "x")
        self.assertIn("there are infinitely many", result)
        self.assertNotIn("any integer n", result)
