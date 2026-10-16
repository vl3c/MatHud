"""Tests for utils.real_sets: checking, finding and writing the solution sets of inequalities.

The browser evaluates the inequality with math.js; here compile_expression's Python
evaluators stand in for it (an overflow is +inf, as in math.js).
"""

from __future__ import annotations

import math
import unittest
from typing import List, Optional, Sequence, Tuple

from server_tests import client_renderer  # noqa: F401  (browser stub, so compiled_expression imports)
from compiled_expression import compile_expression
from utils.real_sets import (
    Piece,
    RealFunction,
    Relation,
    SignChartFailure,
    Window,
    format_solution,
    is_plain_number,
    nice_number,
    number_text,
    sign_chart,
    verify_solution_set,
)

TWO_PI = 2 * math.pi


def _function(text: str) -> RealFunction:
    bound = compile_expression(text, ("x",)).bind()

    def evaluate(point: float) -> Optional[float]:
        try:
            value = bound(point)
        except OverflowError:
            return math.inf
        except (ArithmeticError, ValueError, TypeError):
            return None
        return None if isinstance(value, complex) else float(value)

    return evaluate


def _relations(text: str) -> List[Relation]:
    """'lhs < rhs' or 'a < b <= c' as relations (lhs - rhs)."""
    signs = ("<=", ">=", "<", ">")
    parts: List[str] = []
    found: List[str] = []
    rest = text
    while True:
        positions = [(rest.find(sign), sign) for sign in signs if rest.find(sign) >= 0]
        if not positions:
            parts.append(rest)
            break
        index, sign = min(positions, key=lambda item: (item[0], -len(item[1])))
        parts.append(rest[:index])
        found.append(sign)
        rest = rest[index + len(sign) :]
    return [Relation(_function(f"({parts[index]}) - ({parts[index + 1]})"), sign) for index, sign in enumerate(found)]


def _window(
    lower: float, upper: float, lower_text: str, upper_text: str, closed: Tuple[bool, bool] = (True, True)
) -> Window:
    return Window(lower, upper, closed[0], closed[1], lower_text, upper_text)


def _notation(pieces: Sequence[Piece]) -> List[Tuple[str, str, bool, bool]]:
    return [(piece.lower_text, piece.upper_text, piece.lower_closed, piece.upper_closed) for piece in pieces]


def _solve(text: str, window: Optional[Window] = None, seeds: Sequence[float] = ()) -> str:
    relations = _relations(text)
    pieces = sign_chart(relations, window, seeds)
    return str(format_solution("x", pieces, text, window))


class TestNumbers(unittest.TestCase):
    def test_nice_numbers(self) -> None:
        self.assertEqual(nice_number(0.5), ("1/2", 0.5))
        self.assertEqual(nice_number(-3.0), ("-3", -3.0))
        self.assertEqual(nice_number(math.pi / 6)[0], "pi/6")
        self.assertEqual(nice_number(-5 * math.pi / 6)[0], "-5*pi/6")
        self.assertEqual(nice_number(2 * math.pi)[0], "2*pi")
        self.assertEqual(nice_number(-math.pi)[0], "-pi")
        self.assertIsNone(nice_number(math.sqrt(2)))
        # A value that merely resembles pi/3 is not labelled exact
        self.assertIsNone(nice_number(math.pi / 3 + 1e-8))

    def test_number_text(self) -> None:
        self.assertEqual(number_text(3.0), "3")
        self.assertEqual(number_text(0.4384471871911697), "0.438447187191")
        self.assertEqual(number_text(-math.inf), "-Infinity")

    def test_plain_numbers(self) -> None:
        for text in ("3", "-2", "0.5", "7/12", "-Infinity", "1e-5"):
            self.assertTrue(is_plain_number(text), text)
        for text in ("pi/6", "exp(2)", "sqrt(17)/2+5/2", "inf", "nan", "5*pi/6"):
            self.assertFalse(is_plain_number(text), text)


class TestFormatting(unittest.TestCase):
    def test_the_documented_forms(self) -> None:
        inf = math.inf
        self.assertEqual(
            format_solution("x", [Piece(2, 3, False, False)], "x^2 - 5x + 6 < 0"),
            "2 < x < 3; interval notation: (2, 3)",
        )
        self.assertEqual(
            format_solution("x", [Piece(-inf, -1, False, True), Piece(3, inf, True, False)], "|x - 1| >= 2"),
            "x <= -1 or x >= 3; interval notation: (-Infinity, -1] ∪ [3, Infinity)",
        )
        self.assertEqual(format_solution("x", [Piece(0, 0, True, True)], "x^2 <= 0"), "x = 0; interval notation: {0}")
        self.assertEqual(
            format_solution("x", [Piece(-inf, 1, False, False), Piece(1, inf, False, False)], "(x-1)^2 > 0"),
            "x < 1 or x > 1 (every real x except 1); interval notation: (-Infinity, 1) ∪ (1, Infinity)",
        )
        self.assertEqual(
            format_solution("x", [], "x^2 + 1 < 0"), "No real x satisfies x^2 + 1 < 0 (the solution set is empty)."
        )
        self.assertEqual(
            format_solution("x", [Piece(-inf, inf, False, False)], "x^2 + 1 > 0"),
            "Every real x satisfies x^2 + 1 > 0; interval notation: (-Infinity, Infinity)",
        )

    def test_windows_periods_and_approximations(self) -> None:
        window = _window(0, TWO_PI, "0", "2*pi")
        piece = Piece(math.pi / 6, 5 * math.pi / 6, False, False, "pi/6", "5*pi/6")
        self.assertEqual(
            format_solution("x", [piece], "sin(x) > 1/2", window),
            "pi/6 < x < 5*pi/6 (on [0, 2*pi]); interval notation: (pi/6, 5*pi/6); "
            "endpoints ≈ 0.523598775598, 2.61799387799",
        )
        default = _window(-TWO_PI, TWO_PI, "-2*pi", "2*pi")
        self.assertIn(
            "(on [-2*pi, 2*pi]; the pattern repeats every 2*pi)",
            format_solution("x", [piece], "sin(x) > 1/2", default, "2*pi"),
        )
        self.assertEqual(
            format_solution("x", [], "x > 2", _window(0, 1, "0", "1")),
            "No x in [0, 1] satisfies x > 2 (the solution set is empty) (on [0, 1]).",
        )
        self.assertEqual(
            format_solution("t", [Piece(0, 1, True, True)], "t^2 <= t", _window(0, 1, "0", "1")),
            "Every t in [0, 1] satisfies t^2 <= t (on [0, 1]); interval notation: [0, 1]",
        )


class TestSignChart(unittest.TestCase):
    def test_polynomials_rationals_and_absolute_values(self) -> None:
        self.assertEqual(_solve("x^2 - 5x + 6 < 0"), "2 < x < 3; interval notation: (2, 3)")
        self.assertEqual(
            _solve("abs(x - 1) >= 2"), "x <= -1 or x >= 3; interval notation: (-Infinity, -1] ∪ [3, Infinity)"
        )
        self.assertEqual(
            _solve("(x-1)/(x+2) > 0"), "x < -2 or x > 1; interval notation: (-Infinity, -2) ∪ (1, Infinity)"
        )
        self.assertEqual(_solve("1 < 2x+1 <= 5"), "0 < x <= 2; interval notation: (0, 2]")
        self.assertEqual(_solve("sqrt(x-1) < 3"), "1 <= x < 10; interval notation: [1, 10)")
        self.assertEqual(
            _solve("x^3 + x + 1 > 0"), "x > -0.682327803828; interval notation: (-0.682327803828, Infinity)"
        )

    def test_points_touching_roots_and_poles(self) -> None:
        self.assertEqual(_solve("x^2 <= 0"), "x = 0; interval notation: {0}")
        # Found between samples by the golden-section search
        self.assertEqual(_solve("(x - 0.3)^2 <= 0"), "x = 3/10; interval notation: {3/10}")
        self.assertIn("(every real x except 1)", _solve("(x-1)^2 > 0"))
        self.assertIn("(every real x except 1)", _solve("1/(x-1)^2 > 0", seeds=[1.0]))
        self.assertEqual(_solve("1/x < 1"), "x < 0 or x > 1; interval notation: (-Infinity, 0) ∪ (1, Infinity)")
        self.assertEqual(_solve("x^2 + 1 < 0"), "No real x satisfies x^2 + 1 < 0 (the solution set is empty).")

    def test_transcendental_and_jumps(self) -> None:
        self.assertEqual(_solve("cos(x) > x"), "x < 0.739085133215; interval notation: (-Infinity, 0.739085133215)")
        self.assertEqual(_solve("e^x > 2"), "x > 0.69314718056; interval notation: (0.69314718056, Infinity)")
        self.assertEqual(_solve("floor(x) > 2"), "x >= 3; interval notation: [3, Infinity)")

    def test_trig_on_windows(self) -> None:
        window = _window(0, TWO_PI, "0", "2*pi")
        self.assertTrue(_solve("sin(x) > 1/2", window).startswith("pi/6 < x < 5*pi/6 (on [0, 2*pi])"))
        self.assertTrue(_solve("sin(x) >= 0", window).startswith("0 <= x <= pi or x = 2*pi (on [0, 2*pi])"))
        half_open = _window(0, TWO_PI, "0", "2*pi", (True, False))
        self.assertTrue(_solve("sin(x) >= 0", half_open).startswith("0 <= x <= pi (on [0, 2*pi))"))
        self.assertTrue(
            _solve("tan(x) > 1", window).startswith("pi/4 < x < pi/2 or 5*pi/4 < x < 3*pi/2 (on [0, 2*pi])")
        )

    def test_rounding_near_a_touching_root(self) -> None:
        # 2*sin(x)^2 - sin(x) - 1 rounds to 0 within about 1e-8 of pi/2
        window = _window(0, TWO_PI, "0", "2*pi", (True, False))
        pieces = sign_chart(_relations("2*sin(x)^2 - sin(x) - 1 < 0"), window)
        self.assertEqual(
            _notation(pieces),
            [("0", "pi/2", True, False), ("pi/2", "7*pi/6", False, False), ("11*pi/6", "2*pi", False, False)],
        )

    def test_failures(self) -> None:
        with self.assertRaises(SignChartFailure):
            sign_chart(_relations("sin(x) > 0"))  # the sign keeps changing far out
        with self.assertRaises(SignChartFailure):
            sign_chart(_relations("sqrt(-1 - x^2) > 0"))
        with self.assertRaises(SignChartFailure):
            sign_chart(_relations("sin(20*x) > 0"), _window(0, 20, "0", "20"))  # too many intervals


class TestVerification(unittest.TestCase):
    def test_confirms_right_sets(self) -> None:
        inf = math.inf
        self.assertEqual(verify_solution_set(_relations("x^2 - 5x + 6 < 0"), [Piece(2, 3, False, False)]), "confirmed")
        self.assertEqual(verify_solution_set(_relations("x^2 <= 0"), [Piece(0, 0, True, True)]), "confirmed")
        root = (5 - math.sqrt(17)) / 2
        self.assertEqual(
            verify_solution_set(_relations("x^2 - 5x + 2 < 0"), [Piece(root, (5 + math.sqrt(17)) / 2, False, False)]),
            "confirmed",
        )
        self.assertEqual(
            verify_solution_set(_relations("e^x > 2"), [Piece(math.log(2), inf, False, False)]), "confirmed"
        )
        window = _window(0, TWO_PI, "0", "2*pi")
        tangent = [Piece(math.pi / 4, math.pi / 2, False, False), Piece(5 * math.pi / 4, 3 * math.pi / 2, False, False)]
        self.assertEqual(verify_solution_set(_relations("tan(x) > 1"), tangent, window), "confirmed")

    def test_refutes_wrong_sets(self) -> None:
        relations = _relations("x^2 - 5x + 6 < 0")
        self.assertEqual(verify_solution_set(relations, [Piece(2, 3, True, False)]), "refuted")  # closed end
        self.assertEqual(verify_solution_set(relations, [Piece(2, 3.001, False, False)]), "refuted")  # misplaced
        self.assertEqual(verify_solution_set(relations, [Piece(2, 2.5, False, False)]), "refuted")
        # A dropped isolated point: the seed (a polynomial root) shows it
        self.assertEqual(verify_solution_set(_relations("x^2 <= 0"), [], seeds=[0.0]), "refuted")
        # SymPy's tan(x) > 1 on [0, 2*pi] misses a period
        window = _window(0, TWO_PI, "0", "2*pi")
        self.assertEqual(
            verify_solution_set(_relations("tan(x) > 1"), [Piece(math.pi / 4, math.pi / 2, False, False)], window),
            "refuted",
        )
        # ... and loses the point 2*pi of sin(x) >= 0
        self.assertEqual(
            verify_solution_set(_relations("sin(x) >= 0"), [Piece(0, math.pi, True, True)], window), "refuted"
        )
        # An undefined point inside the set
        self.assertEqual(verify_solution_set(_relations("1/x < 1"), [Piece(-math.inf, 1, False, False)]), "refuted")

    def test_refutes_sympys_periodic_mistake(self) -> None:
        # SymPy answers [0, pi/2) per period for 2*sin(x)^2 - sin(x) - 1 < 0
        window = _window(-TWO_PI, TWO_PI, "-2*pi", "2*pi")
        claimed = [
            Piece(-TWO_PI, -1.5 * math.pi, True, False),
            Piece(0, math.pi / 2, True, False),
            Piece(TWO_PI, TWO_PI, True, True),
        ]
        self.assertEqual(verify_solution_set(_relations("2*sin(x)^2 - sin(x) - 1 < 0"), claimed, window), "refuted")

    def test_unchecked_when_nothing_evaluates(self) -> None:
        self.assertEqual(verify_solution_set(_relations("sqrt(-1 - x^2) > 0"), []), "unchecked")


if __name__ == "__main__":
    unittest.main()
