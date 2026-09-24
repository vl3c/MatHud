"""Regression tests for MathUtils equation solving, definite integrals, and asymptote detection."""

from __future__ import annotations

import json
import math
import re
import unittest
from typing import List, Tuple

from browser import window

from utils.math_utils import MathUtils


def _parse_point_pairs(result: str) -> List[Tuple[float, float]]:
    """Parse '(x = a, y = b), ...' strings into sorted (x, y) tuples."""
    pairs = re.findall(r"\(x = ([^,]+), y = ([^)]+)\)", result)
    return sorted((float(x), float(y)) for x, y in pairs)


def _parse_indexed_solutions(result: str) -> List[Tuple[float, float]]:
    """Parse 'x1 = a, y1 = b, x2 = c, y2 = d' (or 'x = a, y = b') into sorted (x, y) tuples."""
    values = dict(item.split(" = ") for item in result.split(", "))
    xs = sorted(key for key in values if key.startswith("x"))
    return sorted((float(values[key]), float(values["y" + key[1:]])) for key in xs)


def _parse_numeric_solutions(result: str) -> List[Tuple[float, float]]:
    """Parse the numeric solver JSON output into sorted (x, y) tuples."""
    data = json.loads(result)
    return sorted((float(s["x"]), float(s["y"])) for s in data["solutions"])


class TestMathUtilsSolving(unittest.TestCase):
    def assert_points_close(self, actual: List[Tuple[float, float]], expected: List[Tuple[float, float]]) -> None:
        self.assertEqual(len(actual), len(expected), f"Expected {expected}, got {actual}")
        for (ax, ay), (ex, ey) in zip(sorted(actual), sorted(expected)):
            self.assertAlmostEqual(ax, ex, places=6)
            self.assertAlmostEqual(ay, ey, places=6)

    # ------------------------------------------------------------------
    # get_equation_type
    # ------------------------------------------------------------------
    def test_equation_type_reciprocal_is_not_linear(self) -> None:
        self.assertNotEqual(MathUtils.get_equation_type("y = 1/x"), "Linear")

    def test_equation_type_exponential_is_not_linear(self) -> None:
        self.assertNotEqual(MathUtils.get_equation_type("y = e^x"), "Linear")
        self.assertNotEqual(MathUtils.get_equation_type("y = 2^x"), "Linear")

    def test_equation_type_fractional_power_is_not_linear(self) -> None:
        self.assertNotEqual(MathUtils.get_equation_type("y = x^0.5"), "Linear")

    def test_equation_type_quadratic_plus_reciprocal_is_not_quadratic(self) -> None:
        self.assertNotEqual(MathUtils.get_equation_type("y = x^2 + 1/x"), "Quadratic")

    def test_equation_type_keeps_linear_with_constant_fraction(self) -> None:
        self.assertEqual(MathUtils.get_equation_type("y = x/2 + 1"), "Linear")
        self.assertEqual(MathUtils.get_equation_type("2*y = x + 1"), "Linear")

    def test_solve_system_reciprocal_and_line_finds_both_intersections(self) -> None:
        result = MathUtils.solve_system_of_equations(["y = 1/x", "y = x"])
        self.assert_points_close(_parse_numeric_solutions(result), [(-1.0, -1.0), (1.0, 1.0)])

    # ------------------------------------------------------------------
    # solve_linear_quadratic_system
    # ------------------------------------------------------------------
    def test_linear_quadratic_with_scaled_y(self) -> None:
        result = MathUtils.solve_linear_quadratic_system(["2*y = x + 1", "y = x^2"])
        self.assert_points_close(_parse_indexed_solutions(result), [(-0.5, 0.25), (1.0, 1.0)])

    def test_linear_quadratic_with_vertical_line(self) -> None:
        result = MathUtils.solve_linear_quadratic_system(["x = 3", "y = x^2"])
        self.assert_points_close(_parse_indexed_solutions(result), [(3.0, 9.0)])

    def test_solve_system_with_vertical_line_and_parabola(self) -> None:
        result = MathUtils.solve_system_of_equations(["x = 3", "y = x^2"])
        self.assert_points_close(_parse_indexed_solutions(result), [(3.0, 9.0)])

    def test_linear_quadratic_fast_path_unchanged(self) -> None:
        result = MathUtils.solve_linear_quadratic_system(["y = x + 2", "y = x^2"])
        self.assertEqual(_parse_indexed_solutions(result), [(-1.0, 1.0), (2.0, 4.0)])

    # ------------------------------------------------------------------
    # solve_quadratic_system
    # ------------------------------------------------------------------
    def test_quadratic_system_with_irrational_roots(self) -> None:
        result = MathUtils.solve_quadratic_system(["y = x^2", "y = 3 - x^2"])
        root = math.sqrt(1.5)
        self.assert_points_close(_parse_point_pairs(result), [(-root, 1.5), (root, 1.5)])

    def test_quadratic_system_circle_and_parabola(self) -> None:
        result = MathUtils.solve_system_of_equations(["x^2 + y^2 = 25", "y = x^2 - 5"])
        self.assert_points_close(_parse_point_pairs(result), [(-3.0, 4.0), (0.0, -5.0), (3.0, 4.0)])

    def test_quadratic_system_without_explicit_y_uses_numeric_solver(self) -> None:
        result = MathUtils.solve_quadratic_system(["x^2 + y^2 = 25", "x^2 + y = 7"])
        points = _parse_numeric_solutions(result)
        self.assertGreater(len(points), 0)
        for x, y in points:
            self.assertAlmostEqual(x * x + y * y, 25.0, places=6)
            self.assertAlmostEqual(x * x + y, 7.0, places=6)

    def test_quadratic_system_no_real_intersection_is_not_empty(self) -> None:
        result = MathUtils.solve_quadratic_system(["y = x^2 + 1", "y = -x^2"])
        self.assertNotEqual(result, "")
        self.assertEqual(_parse_numeric_solutions(result), [])

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
        self.assertAlmostEqual(float(MathUtils.integral("x^2", "x", 0, 3)), 9.0, places=9)
        self.assertAlmostEqual(float(MathUtils.integral("sin(x)", "x", 0, "pi")), 2.0, places=9)
        self.assertAlmostEqual(float(MathUtils.integral("e^x", "x", 0, 1)), math.e - 1, places=9)
        self.assertAlmostEqual(float(MathUtils.integral("1/x", "x", 1, 2)), math.log(2), places=9)
        self.assertAlmostEqual(float(MathUtils.integral("tan(x)", "x", 0, 1)), -math.log(math.cos(1)), places=9)

    def test_integral_convergent_endpoint_singularity(self) -> None:
        self.assertAlmostEqual(float(MathUtils.integral("1/sqrt(x)", "x", 0, 1)), 2.0, places=9)

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

    def test_solve_keeps_valid_complex_roots(self) -> None:
        self.assertEqual(MathUtils.solve("x^2+1=0", "x"), "[i,-i]")

    def test_solve_keeps_regular_output(self) -> None:
        self.assertEqual(MathUtils.solve("x^2 - 4", "x"), "[2,-2]")
        self.assertEqual(MathUtils.solve("x^2-2=0", "x"), "[sqrt(2),-sqrt(2)]")

    def test_solve_keeps_symbolic_roots_it_cannot_check(self) -> None:
        self.assertEqual(MathUtils.solve("x + y = 3", "x"), "[-(-3+y)]")
