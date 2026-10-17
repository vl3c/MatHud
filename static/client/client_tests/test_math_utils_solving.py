"""Regression tests for MathUtils equation solving, definite integrals, and asymptote detection."""

from __future__ import annotations

import json
import math
import unittest
from typing import List, Tuple

from browser import window

from utils.math_utils import MathUtils

from .cas_engines import WithoutSympy, real_roots, value_of


def _solution_points(result: str) -> List[Tuple[float, float]]:
    """'x1 = a, y1 = b, x2 = c, y2 = d' (or 'x = a, y = b') as sorted (x, y) tuples, the values
    evaluated by math.js (1/2-sqrt(73)/2)."""
    values = dict(item.split(" = ") for item in result.split(" (")[0].split(", "))
    xs = sorted(key for key in values if key.startswith("x"))
    return sorted(
        (float(window.math.evaluate(values[key])), float(window.math.evaluate(values["y" + key[1:]]))) for key in xs
    )


class _WithSympy:
    """Context manager that changes nothing: the block uses the server's SymPy engine."""

    def __enter__(self) -> "_WithSympy":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


class TestMathUtilsSolving(unittest.TestCase):
    def assert_points_close(self, actual: List[Tuple[float, float]], expected: List[Tuple[float, float]]) -> None:
        self.assertEqual(len(actual), len(expected), f"Expected {expected}, got {actual}")
        for (ax, ay), (ex, ey) in zip(sorted(actual), sorted(expected)):
            self.assertAlmostEqual(ax, ex, places=6)
            self.assertAlmostEqual(ay, ey, places=6)

    # ------------------------------------------------------------------
    # solve_system_of_equations
    # ------------------------------------------------------------------
    def test_solve_system_reciprocal_and_line_finds_both_intersections(self) -> None:
        result = MathUtils.solve_system_of_equations(["y = 1/x", "y = x"])
        self.assertEqual(result, "x1 = -1, y1 = -1, x2 = 1, y2 = 1")

    def test_line_and_parabola_with_scaled_y(self) -> None:
        result = MathUtils.solve_system_of_equations(["2*y = x + 1", "y = x^2"])
        self.assertEqual(result, "x1 = -1/2, y1 = 1/4, x2 = 1, y2 = 1")

    def test_solve_system_with_vertical_line_and_parabola(self) -> None:
        self.assertEqual(MathUtils.solve_system_of_equations(["x = 3", "y = x^2"]), "x = 3, y = 9")

    def test_line_and_parabola(self) -> None:
        result = MathUtils.solve_system_of_equations(["y = x + 2", "y = x^2"])
        self.assertEqual(result, "x1 = -1, y1 = 1, x2 = 2, y2 = 4")

    def test_two_parabolas_with_irrational_roots(self) -> None:
        result = MathUtils.solve_system_of_equations(["y = x^2", "y = 3 - x^2"])
        root = math.sqrt(1.5)
        self.assert_points_close(_solution_points(result), [(-root, 1.5), (root, 1.5)])

    def test_circle_and_parabola(self) -> None:
        result = MathUtils.solve_system_of_equations(["x^2 + y^2 = 25", "y = x^2 - 5"])
        self.assertEqual(result, "x1 = -3, y1 = 4, x2 = 0, y2 = -5, x3 = 3, y3 = 4")

    def test_system_without_explicit_y(self) -> None:
        result = MathUtils.solve_system_of_equations(["x^2 + y^2 = 25", "x^2 + y = 7"])
        points = _solution_points(result)
        self.assertEqual(len(points), 4, result)
        for x, y in points:
            self.assertAlmostEqual(x * x + y * y, 25.0, places=9)
            self.assertAlmostEqual(x * x + y, 7.0, places=9)

    def test_two_circles(self) -> None:
        result = MathUtils.solve_system_of_equations(["x^2 + y^2 = 25", "(x-6)^2 + y^2 = 25"])
        self.assertEqual(result, "x1 = 3, y1 = -4, x2 = 3, y2 = 4")

    def test_solve_system_cubic_and_line_returns_all_intersections(self) -> None:
        result = MathUtils.solve_system_of_equations(["y = x^3", "y = x"])
        self.assertEqual(result, "x1 = -1, y1 = -1, x2 = 0, y2 = 0, x3 = 1, y3 = 1")

    def test_solve_system_high_order_single_intersection_keeps_plain_format(self) -> None:
        result = MathUtils.solve_system_of_equations(["x^3 + x - 4 = y", "x^5 - x^4 + x^3 - x^2 + x - 1 = y"])
        self.assertEqual(result, "x = -1, y = -6")

    def test_no_real_intersection(self) -> None:
        result = MathUtils.solve_system_of_equations(["y = x^2 + 1", "y = -x^2"])
        self.assertEqual(result, "No real solution (the equations have 2 complex solutions).")

    def test_three_unknowns(self) -> None:
        result = MathUtils.solve_system_of_equations(["x + y + z = 6", "x - y = 0", "x + z = 5"])
        self.assertEqual(result, "x = 1, y = 1, z = 4")

    def test_parameters_are_not_unknowns(self) -> None:
        result = MathUtils.solve_system_of_equations(["x + y = a", "x - y = 1"])
        self.assertEqual(result, "x = a/2+1/2, y = a/2-1/2")

    def test_dependent_systems_have_infinitely_many_solutions(self) -> None:
        linear = MathUtils.solve_system_of_equations(["x + y = 2", "2*x + 2*y = 4"])
        self.assertTrue(linear.startswith("Infinitely many solutions: the equations are dependent"), linear)
        curves = MathUtils.solve_system_of_equations(["x^2 + y^2 = 1", "2*x^2 + 2*y^2 = 2"])
        self.assertTrue(curves.startswith("Infinitely many solutions (whole curves of them)"), curves)

    def test_transcendental_system_by_substitution(self) -> None:
        # SymPy's solve finds one of the three; substitution lists every real root
        result = MathUtils.solve_system_of_equations(["y = 2^x", "y = x^2"])
        self.assertTrue(result.startswith("x1 = -0.766664695962123, y1 = 0.587774756034695, x2 = 2, y2 = 4,"), result)
        self.assertIn("x3 = 4, y3 = 16 (the real roots found numerically", result)

    def test_single_equation_is_two_curves(self) -> None:
        self.assertEqual(MathUtils.solve_system_of_equations(["x^2 = x + 2"]), "x1 = -1, y1 = 1, x2 = 2, y2 = 4")

    def test_systems_read_input_as_solve_does(self) -> None:
        self.assertEqual(MathUtils.solve_system_of_equations(["y=|x|", "y=2"]), "x1 = -2, y1 = 2, x2 = 2, y2 = 2")
        self.assertEqual(MathUtils.solve_system_of_equations(["2x(x+1)=y", "y=4"]), "x1 = -2, y1 = 4, x2 = 1, y2 = 4")

    def test_glued_products_with_an_unknown_are_refused(self) -> None:
        glued = MathUtils.solve_system_of_equations(["y=mx+b", "y=-x+3"])
        self.assertEqual(glued, "Error: mx would be read as one name; write products with * (m*x), as in y = m*x + b.")
        self.assertTrue(MathUtils.solve_system_of_equations(["xy=6", "x+y=5"]).startswith("Error: xy would be read"))
        # without x, y or z alone they are names: coordinates and components
        self.assertEqual(MathUtils.solve_system_of_equations(["xA + xB = 10", "xA - xB = 2"]), "xA = 6, xB = 4")

    def test_a_kink_is_not_taken_for_a_line(self) -> None:
        self.assertEqual(MathUtils.solve_system_of_equations(["y = max(x,1)", "y=3"]).split(" (")[0], "x = 3, y = 3")

    def test_line_and_sine_list_every_intersection(self) -> None:
        result = MathUtils.solve_system_of_equations(["y=sin(x)", "y=x/10"])
        self.assertIn("x7 = 8.42320393236049", result)

    def test_other_systems_go_to_the_numeric_solver(self) -> None:
        data = json.loads(MathUtils.solve_system_of_equations(["sin(x) + y = 1", "x - y = 0"]))
        self.assertEqual(data["method"], "newton_raphson")
        for solution in data["solutions"]:
            self.assertAlmostEqual(math.sin(solution["x"]) + solution["y"], 1.0, places=6)

    # ------------------------------------------------------------------
    # integral (definite)
    # ------------------------------------------------------------------
    def test_integral_rejects_interior_pole(self) -> None:
        result = MathUtils.integral("1/x^2", "x", -1, 1)
        self.assertTrue(result.startswith("Error"), result)

    def test_integral_rejects_tangent_across_pole(self) -> None:
        self.assertTrue(MathUtils.integral("tan(x)", "x", 1, 2).startswith("Error"))
        # Crosses two poles but F(b) - F(a) happens to be real
        self.assertTrue(MathUtils.integral("tan(x)", "x", 1, 5).startswith("Error"))

    def test_integral_rejects_log_singularity(self) -> None:
        result = MathUtils.integral("1/x", "x", -1, 2)
        self.assertTrue(result.startswith("Error"), result)

    def test_integral_regular_cases_unchanged(self) -> None:
        for engines in (_WithSympy(), WithoutSympy()):
            with engines:
                self.assertAlmostEqual(value_of(MathUtils.integral("x^2", "x", 0, 3)), 9.0, places=9)
                self.assertAlmostEqual(value_of(MathUtils.integral("sin(x)", "x", 0, "pi")), 2.0, places=9)
                self.assertAlmostEqual(value_of(MathUtils.integral("e^x", "x", 0, 1)), math.e - 1, places=9)
                self.assertAlmostEqual(value_of(MathUtils.integral("1/x", "x", 1, 2)), math.log(2), places=9)
                self.assertAlmostEqual(
                    value_of(MathUtils.integral("tan(x)", "x", 0, 1)), -math.log(math.cos(1)), places=9
                )

    def test_integral_across_removable_discontinuity(self) -> None:
        # math.js yields NaN (0/0) at the removable point, which must not be reported as a pole
        self.assertAlmostEqual(value_of(MathUtils.integral("(x^2-1)/(x-1)", "x", 0, 2)), 4.0, places=9)
        for lower, upper in ((-1, 1), (-1, 2)):
            result = MathUtils.integral("sin(x)/x", "x", lower, upper)
            self.assertFalse(result.startswith("Error"), result)

    def test_integral_still_rejects_pole_evaluating_to_nan(self) -> None:
        # x/x^3 is 0/0 at x = 0 but blows up like 1/x^2 around it
        self.assertTrue(MathUtils.integral("x/x^3", "x", -1, 1).startswith("Error"))
        self.assertTrue(MathUtils.integral("sin(x)/x^2", "x", -1, 1).startswith("Error"))

    def test_integral_convergent_endpoint_singularity(self) -> None:
        self.assertAlmostEqual(float(MathUtils.integral("1/sqrt(x)", "x", 0, 1)), 2.0, places=9)

    # ------------------------------------------------------------------
    # calculate_vertical_asymptotes / calculate_point_discontinuities
    # ------------------------------------------------------------------
    def test_vertical_asymptotes_arctangent_has_none(self) -> None:
        self.assertEqual(MathUtils.calculate_vertical_asymptotes("atan(x)", -5, 5), [])

    def test_vertical_asymptotes_shifted_tangent(self) -> None:
        result = MathUtils.calculate_vertical_asymptotes("tan(x-1)", -5, 5)
        expected = [1 + math.pi / 2 + n * math.pi for n in range(-2, 1)]
        self.assertEqual([round(x, 6) for x in result], [round(x, 6) for x in expected])

    def test_vertical_asymptotes_scaled_tangent(self) -> None:
        result = MathUtils.calculate_vertical_asymptotes("tan(2*x)", -2, 2)
        expected = [math.pi / 4 + n * math.pi / 2 for n in range(-1, 1)]
        self.assertEqual([round(x, 6) for x in result], [round(x, 6) for x in expected])

    def test_vertical_asymptotes_shifted_log(self) -> None:
        self.assertEqual(MathUtils.calculate_vertical_asymptotes("log(x-3)"), [3])

    def test_vertical_asymptotes_denominator_followed_by_terms(self) -> None:
        self.assertEqual(MathUtils.calculate_vertical_asymptotes("1/(x-2)+3"), [2])

    def test_vertical_asymptotes_irrational_roots(self) -> None:
        result = MathUtils.calculate_vertical_asymptotes("1/(x^2-2)")
        self.assertEqual([round(x, 9) for x in result], [round(-math.sqrt(2), 9), round(math.sqrt(2), 9)])

    def test_vertical_asymptotes_multiple_denominators(self) -> None:
        self.assertEqual(MathUtils.calculate_vertical_asymptotes("1/(x-1)+1/(x+2)"), [-2, 1])

    def test_point_discontinuities_abs_with_irrational_corners(self) -> None:
        result = MathUtils.calculate_point_discontinuities("abs(x^2-2)")
        self.assertEqual([round(x, 9) for x in result], [round(-math.sqrt(2), 9), round(math.sqrt(2), 9)])

    def test_point_discontinuities_abs_with_nested_parentheses(self) -> None:
        self.assertEqual(MathUtils.calculate_point_discontinuities("abs((x-1)*2)"), [1])

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------
    def _roots(self, result: str) -> List[object]:
        self.assertTrue(result.startswith("[") and result.endswith("]"), result)
        inner = result[1:-1]
        return [window.math.evaluate(part) for part in inner.split(",")] if inner else []

    def test_solve_drops_roots_that_do_not_satisfy_equation(self) -> None:
        roots = self._roots(MathUtils.solve("x^3-2=0", "x"))
        self.assertGreater(len(roots), 0)
        for root in roots:
            residual = window.math.abs(window.math.subtract(window.math.pow(root, 3), 2))
            self.assertLess(float(residual), 1e-9)
        self.assertTrue(any(abs(float(window.math.re(r)) - 2 ** (1 / 3)) < 1e-9 for r in roots))

    def test_solve_keeps_valid_roots_of_large_coefficient_equations(self) -> None:
        for coefficient in (200000, 2000000):
            roots = sorted(
                float(window.math.re(r)) for r in self._roots(MathUtils.solve(f"x^2 - {coefficient}*x + 1 = 0", "x"))
            )
            self.assertEqual(len(roots), 2, roots)
            self.assertAlmostEqual(roots[0] * coefficient, 1.0, places=2)
            self.assertAlmostEqual(roots[1] / coefficient, 1.0, places=6)

    def test_solve_keeps_valid_complex_roots(self) -> None:
        self.assertEqual(MathUtils.solve("x^2+1=0", "x"), "[-i,i]")

    def test_solve_keeps_regular_output(self) -> None:
        self.assertEqual(MathUtils.solve("x^2 - 4", "x"), "[-2,2]")
        self.assertEqual(MathUtils.solve("x^2-2=0", "x"), "[-sqrt(2),sqrt(2)]")

    def test_solve_keeps_symbolic_roots_it_cannot_check(self) -> None:
        self.assertEqual(MathUtils.solve("x + y = 3", "x"), "[3-y]")

    def test_solve_a_formula_for_a_variable(self) -> None:
        self.assertEqual(MathUtils.solve("R = V/I", "I"), "[V/R]")
        self.assertEqual(MathUtils.solve("T = 2*pi*sqrt(L/g)", "g"), "[4*pi^2*L/T^2]")
        self.assertEqual(MathUtils.solve("a^x = b", "x"), "[log(b)/log(a)]")

    def test_solve_reads_arc_names(self) -> None:
        self.assertEqual(MathUtils.solve("arctan(x) = 1", "x"), "[tan(1)]")
        self.assertEqual(MathUtils.derivative("arcsin(x)", "x"), "1/sqrt(1-x^2)")

    def test_trig_equations_that_are_not_periodic_are_solved_on_the_whole_line(self) -> None:
        result = MathUtils.solve("x + sin(x) = 10", "x")
        self.assertTrue(result.startswith("[10.9999902496552] (the real roots found numerically for |x| up to"), result)
        self.assertEqual(len(MathUtils.solve("sin(x) = x/10", "x").split(" (")[0].split(",")), 7)
        # roots without end: those in [-2*pi, 2*pi], saying there are more
        crossing = MathUtils.solve("x*sin(x) = 1", "x")
        self.assertIn("in [-2*pi, 2*pi] (it has more beyond, as |x| grows)", crossing)
        self.assertEqual(len(crossing.split(" (")[0].split(",")), 4)

    def test_solve_lists_the_roots_in_an_interval(self) -> None:
        self.assertEqual(
            MathUtils.solve("tan(x) = 1", "x", "[0, 20]"),
            "[pi/4,5*pi/4,9*pi/4,13*pi/4,17*pi/4,21*pi/4,25*pi/4] (the roots in [0, 20])",
        )
        self.assertEqual(MathUtils.solve("sin(x) = 0", "x", "[0, 2pi)"), "[0,pi] (the roots in [0, 2*pi))")
        self.assertEqual(MathUtils.solve("sin(x) = 0", "x", "[0, 2pi]"), "[0,pi,2*pi] (the roots in [0, 2*pi])")
        many = MathUtils.solve("sin(x) = 0", "x", "[0, 1000]")
        self.assertTrue(many.endswith(",59*pi] (the first 60 of 319 roots in [0, 1000])"), many)
        self.assertEqual(MathUtils.solve("x^2 = 4", "x", "[-2, 2)"), "[-2] (the roots in [-2, 2))")
        self.assertEqual(MathUtils.solve("x^2 = 2", "x", "[0, 5]"), "[sqrt(2)] (the roots in [0, 5])")
        self.assertEqual(MathUtils.solve("x^2 + 1 = 0", "x", "[0, 5]"), "[] (the roots in [0, 5])")
        self.assertEqual(MathUtils.solve("x^2 = 4", "x", "(-Infinity, 0]"), "[-2] (the roots in (-Infinity, 0])")
        self.assertEqual(real_roots(MathUtils.solve("(x-1)(x+2) = 0", "x", "[-5, 5]")), [-2.0, 1.0])
        for engine in (_WithSympy, WithoutSympy):
            with engine():
                self.assertTrue(MathUtils.solve("x^2 = 4", "x", "[2, 1]").startswith("Error:"))
                self.assertEqual(MathUtils.solve("x^2 = 4", "x", None), MathUtils.solve("x^2 = 4", "x"))
                # One variable only, as for inequalities
                self.assertEqual(
                    MathUtils.solve("x + y = 3", "x", "[0, 5]"),
                    "Error: solve lists the roots in an interval for equations in one variable; "
                    "besides x it also contains y.",
                )
        with WithoutSympy():
            # The numeric root finder honours the interval's ends
            self.assertEqual(MathUtils.solve("x^2 = 4", "x", "[-2, 2)").split(" (")[0], "[-2]")
            self.assertEqual(MathUtils.solve("x^2 = 4", "x", "(-Infinity, 0]").split(" (")[0], "[-2]")
            self.assertEqual(MathUtils.solve("x^2 + 1 = 0", "x", "[0, 5]").split(" (")[0], "[]")
