"""Tests for inequalities in the solve tool (utils.inequality_solver, utils.real_sets in Brython).

TestInequalitySolverSympy pins SymPy's exact answers and skips without the server's SymPy
engine; TestInequalitySolverWithoutSympy pins the numeric sign chart, which answers when
SymPy is unavailable. TestInequalityHelpers pins what Brython prints for numbers (CPython's
pytest covers the same helpers) and the checks that refute a wrong set.
"""

from __future__ import annotations

import math
import unittest

from utils.inequality_solver import InequalitySolver
from utils.math_utils import MathUtils
from utils.real_sets import Piece, format_solution, nice_number, number_text, verify_solution_set

from .cas_engines import WithoutSympy, sympy_available

NUMERIC = "(found numerically because SymPy was not available"


def _solve(text: str, interval: object = None) -> str:
    return str(MathUtils.solve(text, "x", interval))


class TestInequalityHelpers(unittest.TestCase):
    def test_brython_prints_numbers_like_cpython(self) -> None:
        self.assertEqual(nice_number(math.pi / 6)[0], "pi/6")
        self.assertEqual(nice_number(-5 * math.pi / 6)[0], "-5*pi/6")
        self.assertEqual(nice_number(0.75)[0], "3/4")
        self.assertIsNone(nice_number(math.sqrt(2)))
        self.assertEqual(number_text(0.4384471871911697), "0.438447187191")
        self.assertEqual(number_text(-3.0), "-3")
        self.assertEqual(number_text(-math.inf), "-Infinity")
        piece = Piece(0.4384471871911697, 4.561552812808831, False, False, "5/2-sqrt(17)/2", "sqrt(17)/2+5/2")
        self.assertEqual(
            format_solution("x", [piece], "x^2 - 5x + 2 < 0"),
            "5/2-sqrt(17)/2 < x < sqrt(17)/2+5/2; interval notation: (5/2-sqrt(17)/2, sqrt(17)/2+5/2); "
            "endpoints ≈ 0.438447187191, 4.56155281281",
        )

    def test_math_js_checks_refute_wrong_sets(self) -> None:
        relations = InequalitySolver._relations(["x^2 - 5x + 6", "0"], ["<"], "x")
        self.assertEqual(verify_solution_set(relations, [Piece(2, 3, False, False)]), "confirmed")
        self.assertEqual(verify_solution_set(relations, [Piece(2, 3, True, False)]), "refuted")
        self.assertEqual(verify_solution_set(relations, [Piece(2, 3.5, False, False)]), "refuted")
        self.assertEqual(verify_solution_set(relations, []), "refuted")

    def test_refusals(self) -> None:
        cases = {
            "x != 2": "exclude its roots",
            "x > y": "also contains y",
            "a*x > 2": "also contains a",
            "x > 1 and x < 5": "on its own",
            "1 < x > 0": "point the same way",
            "y > 2": "also contains y",
        }
        for text, message in cases.items():
            result = _solve(text)
            self.assertTrue(result.startswith("Error:"), result)
            self.assertIn(message, result)
        for interval in ("[2, 1]", "[0, y]", "[0, 1"):
            result = _solve("x > 0", interval)
            self.assertTrue(result.startswith("Error:"), f"{interval}: {result}")

    def test_equations_are_unchanged(self) -> None:
        self.assertEqual(_solve("x^2 - 5x + 6 = 0"), "[2,3]")


class TestInequalitySolverSympy(unittest.TestCase):
    def setUp(self) -> None:
        if not sympy_available():
            self.skipTest("the server's SymPy engine is not available")

    def test_exact_sets(self) -> None:
        cases = {
            "x^2 - 5x + 6 < 0": "2 < x < 3; interval notation: (2, 3)",
            "|x - 1| >= 2": "x <= -1 or x >= 3; interval notation: (-Infinity, -1] ∪ [3, Infinity)",
            "(x-1)/(x+2) > 0": "x < -2 or x > 1; interval notation: (-Infinity, -2) ∪ (1, Infinity)",
            "1 < 2x+1 <= 5": "0 < x <= 2; interval notation: (0, 2]",
            "x(x+2) <= 0": "-2 <= x <= 0; interval notation: [-2, 0]",
            "x^2 <= 0": "x = 0; interval notation: {0}",
            "x^2 + 1 < 0": "No real x satisfies x^2 + 1 < 0 (the solution set is empty).",
            "x^2 + 1 > 0": "Every real x satisfies x^2 + 1 > 0; interval notation: (-Infinity, Infinity)",
            "ln(x) < 2": "0 < x < exp(2); interval notation: (0, exp(2)); endpoints ≈ 7.38905609893",
            "x ≥ 2": "x >= 2; interval notation: [2, Infinity)",
        }
        for text, expected in cases.items():
            self.assertEqual(_solve(text), expected, text)

    def test_periodic_sets(self) -> None:
        self.assertEqual(
            _solve("sin(x) > 1/2", "[0, 2pi]"),
            "pi/6 < x < 5*pi/6 (on [0, 2*pi]); interval notation: (pi/6, 5*pi/6); "
            "endpoints ≈ 0.523598775598, 2.61799387799",
        )
        self.assertTrue(
            _solve("sin(x) > 1/2").startswith(
                "-11*pi/6 < x < -7*pi/6 or pi/6 < x < 5*pi/6 (on [-2*pi, 2*pi]; the pattern repeats every 2*pi)"
            )
        )
        self.assertTrue(_solve("sin(x) >= 0", "[0, 2*pi)").startswith("0 <= x <= pi (on [0, 2*pi));"))
        self.assertTrue(_solve("sin(x) >= 0", "[0, 2*pi]").startswith("0 <= x <= pi or x = 2*pi (on [0, 2*pi]);"))
        self.assertTrue(
            _solve("tan(x) > 1", "[0, 2*pi]").startswith("pi/4 < x < pi/2 or 5*pi/4 < x < 3*pi/2 (on [0, 2*pi]);")
        )

    def test_a_wrong_sympy_answer_gives_way_to_the_sign_chart(self) -> None:
        # SymPy answers [0, pi/2) per period
        result = _solve("2sin(x)^2 - sin(x) - 1 < 0", "[0, 2*pi)")
        self.assertTrue(
            result.startswith("0 <= x < pi/2 or pi/2 < x < 7*pi/6 or 11*pi/6 < x < 2*pi (on [0, 2*pi));"), result
        )
        self.assertIn("(found numerically because SymPy's answer failed the numeric check)", result)


class TestInequalitySolverWithoutSympy(unittest.TestCase):
    def test_sign_chart_answers(self) -> None:
        cases = {
            "x^2 - 5x + 6 < 0": "2 < x < 3; interval notation: (2, 3)",
            "|x - 1| >= 2": "x <= -1 or x >= 3; interval notation: (-Infinity, -1] ∪ [3, Infinity)",
            "x^2 <= 0": "x = 0; interval notation: {0}",
            "(x-1)^2 > 0": "x < 1 or x > 1 (every real x except 1); interval notation: (-Infinity, 1) ∪ (1, Infinity)",
        }
        with WithoutSympy():
            for text, expected in cases.items():
                result = _solve(text)
                self.assertTrue(result.startswith(expected + " " + NUMERIC), f"{text}: {result}")
                self.assertTrue(result.endswith("checked out to |x| = 10^6)"), result)
            result = _solve("sin(x) > 1/2", "[0, 2pi]")
        self.assertTrue(result.startswith("pi/6 < x < 5*pi/6 (on [0, 2*pi]); interval notation:"), result)
        self.assertTrue(result.endswith(NUMERIC + ")"), result)
