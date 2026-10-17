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


EVERYWHERE = "; interval notation: (-Infinity, Infinity)"

# Inputs from the review of the first version, with the set each engine must give (the
# sign chart's answer is followed by its note). Exact sets first, then the numeric ones.
REVIEW_EXACT = {
    # exp underflows to 0 far out, which is no sign change
    ("x*exp(x) < 0", None): "x < 0; interval notation: (-Infinity, 0)",
    ("(x+1)*exp(x) < 0", None): "x < -1; interval notation: (-Infinity, -1)",
    ("x*exp(-x) > 0", None): "x > 0; interval notation: (0, Infinity)",
    ("exp(-x^2) > 0", None): "Every real x satisfies exp(-x^2) > 0" + EVERYWHERE,
    ("exp(x) > 0", None): "Every real x satisfies exp(x) > 0" + EVERYWHERE,
    ("2^x > 0", None): "Every real x satisfies 2^x > 0" + EVERYWHERE,
    ("1/(1+exp(-x)) > 0", None): "Every real x satisfies 1/(1+exp(-x)) > 0" + EVERYWHERE,
    # implicit products after ")"
    ("(x-1)(x+2) < 0", None): "-2 < x < 1; interval notation: (-2, 1)",
    ("(x-2)^2 (x+1) >= 0", None): "x >= -1; interval notation: [-1, Infinity)",
    ("|x-1|(x+1) > 0", None): "-1 < x < 1 or x > 1; interval notation: (-1, 1) ∪ (1, Infinity)",
    ("(x+1)e^x < 0", None): "x < -1; interval notation: (-Infinity, -1)",
    ("(x+1)x > 0", None): "x < -1 or x > 0; interval notation: (-Infinity, -1) ∪ (0, Infinity)",
    ("(x+1)2 > 4", None): "x > 1; interval notation: (1, Infinity)",
    # identities read as equalities
    ("(x+1)^2 >= x^2 + 2x + 1", None): "Every real x satisfies (x+1)^2 >= x^2 + 2x + 1" + EVERYWHERE,
    # open interval ends
    ("cos(x) > 0", "[0, 2*pi)"): "0 <= x < pi/2 or 3*pi/2 < x < 2*pi (on [0, 2*pi)); "
    "interval notation: [0, pi/2) ∪ (3*pi/2, 2*pi); endpoints ≈ 1.57079632679, 4.71238898038, 6.28318530718",
    ("x^2 < 2", "(0, 5)"): "0 < x < sqrt(2) (on (0, 5)); interval notation: (0, sqrt(2)); endpoints ≈ 1.41421356237",
    ("x < 3", "(2, 5)"): "2 < x < 3 (on (2, 5)); interval notation: (2, 3)",
    ("x >= 1", "(1, 3)"): "Every x in (1, 3) satisfies x >= 1; interval notation: (1, 3)",
    # the last period copy
    ("cos(6x) >= 1", "[0, pi]"): "x = 0 or x = pi/3 or x = 2*pi/3 or x = pi (on [0, pi]); "
    "interval notation: {0} ∪ {pi/3} ∪ {2*pi/3} ∪ {pi}; endpoints ≈ 1.0471975512, 2.09439510239, 3.14159265359",
    # flat touching roots at the window's ends
    ("cos(x) >= 1", None): "x = -2*pi or x = 0 or x = 2*pi (on [-2*pi, 2*pi]; the pattern repeats every 2*pi); "
    "interval notation: {-2*pi} ∪ {0} ∪ {2*pi}; endpoints ≈ -6.28318530718, 6.28318530718",
    ("cos(x) >= 1", "[0, 2*pi]"): "x = 0 or x = 2*pi (on [0, 2*pi]); interval notation: {0} ∪ {2*pi}; "
    "endpoints ≈ 6.28318530718",
    ("cos(x)^2 >= 1", "[0, 3*pi]"): "x = 0 or x = pi or x = 2*pi or x = 3*pi (on [0, 3*pi]); "
    "interval notation: {0} ∪ {pi} ∪ {2*pi} ∪ {3*pi}; endpoints ≈ 3.14159265359, 6.28318530718, 9.42477796077",
    # domain edges where float rounding leaves sqrt undefined
    ("sqrt(2 - x^2) >= 0", None): "-sqrt(2) <= x <= sqrt(2); interval notation: [-sqrt(2), sqrt(2)]; "
    "endpoints ≈ -1.41421356237, 1.41421356237",
    ("sqrt(x^2 - 3) >= 0", None): "x <= -sqrt(3) or x >= sqrt(3); interval notation: "
    "(-Infinity, -sqrt(3)] ∪ [sqrt(3), Infinity); endpoints ≈ -1.73205080757, 1.73205080757",
    # decimals stay decimals; interval ends are evaluated
    ("x >= 1.23456789", None): "x >= 1.23456789; interval notation: [1.23456789, Infinity)",
    ("x > 0", "[1e-300, 1]"): "Every x in [1e-300, 1] satisfies x > 0; interval notation: [1e-300, 1]",
    ("(x-1)*(x-1.0000001) < 0", None): "1 < x < 1.0000001; interval notation: (1, 1.0000001)",
    ("x < 3", "(1+1, 5)"): "2 < x < 3 (on (2, 5)); interval notation: (2, 3)",
    ("x < 3", "[-inf, 5]"): "x < 3 (on (-Infinity, 5]); interval notation: (-Infinity, 3)",
    ("x^2 + 1 < 0", "[-2, 2]"): "No x in [-2, 2] satisfies x^2 + 1 < 0 (the solution set is empty).",
    ("x > 2000000", None): "x > 2000000; interval notation: (2000000, Infinity)",
    ("x^2 - 5x + 6 < 0", "[-10000, 10000]"): "2 < x < 3 (on [-10000, 10000]); interval notation: (2, 3)",
}

# What the sign chart gives where its text differs from SymPy's (decimal ends, no period note)
REVIEW_NUMERIC = dict(REVIEW_EXACT)
REVIEW_NUMERIC.update(
    {
        ("x^2 < 2", "(0, 5)"): "0 < x < 1.41421356237 (on (0, 5)); interval notation: (0, 1.41421356237)",
        ("cos(x) >= 1", None): "x = -2*pi or x = 0 or x = 2*pi (on [-2*pi, 2*pi]); "
        "interval notation: {-2*pi} ∪ {0} ∪ {2*pi}; endpoints ≈ -6.28318530718, 6.28318530718",
        ("sqrt(2 - x^2) >= 0", None): "-1.41421356237 <= x <= 1.41421356237; "
        "interval notation: [-1.41421356237, 1.41421356237]",
        ("sqrt(x^2 - 3) >= 0", None): "x <= -1.73205080757 or x >= 1.73205080757; "
        "interval notation: (-Infinity, -1.73205080757] ∪ [1.73205080757, Infinity)",
        ("x^2 + 1 < 0", "[-2, 2]"): "No x in [-2, 2] satisfies x^2 + 1 < 0 (the solution set is empty",
        ("x > 1e20", None): "x > 100000000000000000000; interval notation: (100000000000000000000, Infinity)",
        ("x^2 > 1e-12", None): "x < -1e-06 or x > 1e-06; interval notation: (-Infinity, -1e-06) ∪ (1e-06, Infinity)",
    }
)


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
            # bars after ln: log(abs(x)) for math.js as well as SymPy
            "ln|x| < 1": "-e < x < 0 or 0 < x < e; interval notation: (-e, 0) ∪ (0, e); "
            "endpoints ≈ -2.71828182846, 2.71828182846",
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

    def test_review_inputs(self) -> None:
        for (text, interval), expected in REVIEW_EXACT.items():
            self.assertEqual(_solve(text, interval), expected, f"{text} on {interval}")

    def test_calculator_notation(self) -> None:
        # functions without parentheses and e or pi glued to x, read as calculators do
        cases = {
            ("sin x > 1/2", "[0, 2*pi]"): "pi/6 < x < 5*pi/6 (on [0, 2*pi]); interval notation: (pi/6, 5*pi/6); "
            "endpoints ≈ 0.523598775598, 2.61799387799",
            ("sinx > 1/2", "[0, 2*pi]"): "pi/6 < x < 5*pi/6 (on [0, 2*pi]); interval notation: (pi/6, 5*pi/6); "
            "endpoints ≈ 0.523598775598, 2.61799387799",
            ("2 sin x cos x > 0", "[0, pi]"): "0 < x < pi/2 (on [0, pi]); interval notation: (0, pi/2); "
            "endpoints ≈ 1.57079632679",
            ("xe^x > 0", None): "x > 0; interval notation: (0, Infinity)",
            ("x ln x < 0", None): "0 < x < 1; interval notation: (0, 1)",
            ("ln x < 1", None): "0 < x < e; interval notation: (0, e); endpoints ≈ 2.71828182846",
            ("xpi > 1", None): "x > 1/pi; interval notation: (1/pi, Infinity); endpoints ≈ 0.318309886184",
            ("sqrt x < 2", None): "0 <= x < 4; interval notation: [0, 4)",
        }
        for (text, interval), expected in cases.items():
            self.assertEqual(_solve(text, interval), expected, text)
        # equations, with an interval and without, read the same text
        equations = {
            ("sinx = 0.5", "[0, 2*pi]"): "[pi/6,5*pi/6] (the roots in [0, 2*pi])",
            ("xe^x = 1", "[0, 2]"): "[0.5671432904098] (the roots in [0, 2])",
            ("lnx = 1", "[0, 5]"): "[e] (the roots in [0, 5])",
            ("x ln x = 1", "[0.5, 3]"): "[1.7632228343519] (the roots in [0.5, 3])",
            ("sin 2x = 1", "[0, pi]"): "[pi/4] (the roots in [0, pi])",
            ("xe^x = 1", None): "[0.5671432904098]",
            ("ln x = 1", None): "[e]",
            ("xpi = 1", None): "[1/pi]",
            # sech is a function, not sec(h); log2x is log2(x) or log(2x), so it is not read
            ("sech x = 0.5", None): "[-asech(1/2),asech(1/2)]",
            ("coth x = 2", None): "[acoth(2)]",
            ("csch x = 1", None): "[log(1+sqrt(2))]",
            ("sinhx = 1", None): "[log(1+sqrt(2))]",
            ("log2x = 3", None): "[]",
        }
        for (text, interval), expected in equations.items():
            self.assertEqual(_solve(text, interval), expected, f"{text} on {interval}")
        self.assertTrue(_solve("cosx = x", "[0, 2]").startswith("[0.739085133215191] (the roots in [0, 2]"))
        # the variable being solved for is never split: cost is not cos(t), ex is not e*x
        self.assertEqual(str(MathUtils.solve("cost < 1", "cost")), "cost < 1; interval notation: (-Infinity, 1)")
        self.assertEqual(str(MathUtils.solve("ex > 1", "ex")), "ex > 1; interval notation: (1, Infinity)")
        self.assertEqual(str(MathUtils.solve("cost = 5", "cost")), "[5]")
        self.assertEqual(str(MathUtils.solve("costs > 5", "costs")), "costs > 5; interval notation: (5, Infinity)")

    def test_identities_without_an_exact_answer(self) -> None:
        result = _solve("sin(x)^2 + cos(x)^2 > 1")
        self.assertTrue(
            result.startswith("No x in [-2*pi, 2*pi] satisfies sin(x)^2 + cos(x)^2 > 1 (the solution set is empty"),
            result,
        )
        result = _solve("sin(x)^2 + cos(x)^2 <= 1", "[0, 10]")
        self.assertTrue(
            result.startswith("Every x in [0, 10] satisfies sin(x)^2 + cos(x)^2 <= 1; interval notation: [0, 10]"),
            result,
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

    def test_review_inputs(self) -> None:
        with WithoutSympy():
            for (text, interval), expected in REVIEW_NUMERIC.items():
                result = _solve(text, interval)
                self.assertTrue(result.startswith(expected), f"{text} on {interval}: {result}")
                self.assertIn("found numerically because SymPy was not available", result)
            identity = _solve("sin(x)^2 + cos(x)^2 > 1")
            empty = _solve("x^2 + 1 < 0", "[-2, 2]")
            reach = [_solve(text) for text in ("x > 2000000", "x > 1e20")]
        self.assertEqual(
            identity,
            "No x in [-2*pi, 2*pi] satisfies sin(x)^2 + cos(x)^2 > 1 "
            "(the solution set is empty; found numerically because SymPy was not available).",
        )
        self.assertEqual(
            empty,
            "No x in [-2, 2] satisfies x^2 + 1 < 0 (the solution set is empty; "
            "found numerically because SymPy was not available).",
        )
        # the grid reaches beyond the largest polynomial root, and the note says how far
        self.assertTrue(reach[0].endswith("checked out to |x| = 10^9)"), reach[0])
        self.assertTrue(reach[1].endswith("checked out to |x| = 10^22)"), reach[1])
