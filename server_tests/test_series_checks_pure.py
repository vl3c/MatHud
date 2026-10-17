"""Tests for utils.series_checks (is a polynomial a function's Taylor polynomial?) and the
taylor_polynomial tool's remainder notes (utils.taylor.remainder_note).

The browser evaluates the expressions with math.js; here compile_expression's Python
evaluators stand in for it (a complex or failed value is undefined).
"""

from __future__ import annotations

import math
import unittest
from typing import Optional, cast

from server_tests import client_renderer  # noqa: F401  (browser stub, so compiled_expression imports)
from compiled_expression import compile_expression
from utils.series_checks import RealFunction, taylor_matches
from utils.taylor import remainder_note

EXP_10 = "+".join(f"x^{k}/{math.factorial(k)}" for k in range(11))


def _function(text: str) -> RealFunction:
    bound = compile_expression(text, ("x",)).bind()

    def evaluate(x: float) -> Optional[float]:
        try:
            value = bound(x)
        except (ArithmeticError, ValueError, TypeError):
            return None
        if isinstance(value, complex):
            return None
        return float(value)

    return evaluate


def _matches(function: str, polynomial: str, center: float, order: int) -> Optional[bool]:
    return cast(Optional[bool], taylor_matches(_function(function), _function(polynomial), center, order))


class TestTaylorMatches(unittest.TestCase):
    def test_confirms_taylor_polynomials(self) -> None:
        for function, polynomial, center, order in (
            ("e^x", "1 + x + x^2/2 + x^3/6 + x^4/24", 0.0, 4),
            ("e^x", EXP_10, 0.0, 10),
            ("e^x", "1", 0.0, 0),
            ("sin(x)", "x - x^3/6 + x^5/120", 0.0, 5),
            ("ln(x)", "(x-1) - (x-1)^2/2 + (x-1)^3/3", 1.0, 3),
            ("cos(x)", "sqrt(2)/2 - sqrt(2)*(x-pi/4)/2 - sqrt(2)*(x-pi/4)^2/4", math.pi / 4, 2),
            ("e^x", "exp(-1) + (x+1)*exp(-1) + (x+1)^2*exp(-1)/2 + (x+1)^3*exp(-1)/6", -1.0, 3),
            ("sqrt(x)", "2 + (x-4)/4 - (x-4)^2/64 + (x-4)^3/512", 4.0, 3),
            # a small radius of convergence (0.2): the ratios are read close to the centre
            ("1/(1+25x^2)", "1 - 25*x^2 + 625*x^4", 0.0, 4),
            ("1/(1-x)", "+".join(f"x^{k}" for k in range(11)), 0.0, 10),
            # removable at the centre: math.js gives NaN there, the sides decide
            ("sin(x)/x", "1 - x^2/6 + x^4/120", 0.0, 4),
            # far from 0, with large values
            ("e^x", "e^100*(1 + (x-100) + (x-100)^2/2)", 100.0, 2),
            ("sin(x)", "sin(1000) + cos(1000)*(x-1000)", 1000.0, 1),
            # exact: the error is rounding at every step
            ("x^3+2x", "3 + 5*(x-1) + 3*(x-1)^2 + (x-1)^3", 1.0, 5),
            # smooth to the order only
            ("abs(x)^3", "0", 0.0, 2),
            # exact on the right; on the left the errors break off at x = 0, where abs turns
            ("abs(x)", "1 + (x-1)", 1.0, 3),
            ("max(x, 0)", "x", 1.0, 2),
        ):
            with self.subTest(function=function, center=center, order=order):
                self.assertIs(_matches(function, polynomial, center, order), True)

    def test_refutes_a_wrong_coefficient_at_each_degree(self) -> None:
        correct = [1.0 / math.factorial(k) for k in range(6)]
        for wrong in range(6):
            coefficients = list(correct)
            coefficients[wrong] *= 1.5
            polynomial = "+".join(f"{value!r}*x^{k}" for k, value in enumerate(coefficients))
            with self.subTest(wrong=wrong):
                self.assertIs(_matches("e^x", polynomial, 0.0, 5), False)
        self.assertIs(_matches("e^x", EXP_10.replace("x^10/3628800", "x^10/362880"), 0.0, 10), False)
        self.assertIs(_matches("ln(x)", "(x-1) - (x-1)^2/2 + (x-1)^3/4", 1.0, 3), False)
        self.assertIs(_matches("e^x", "2", 0.0, 0), False)
        # a polynomial at the wrong centre
        self.assertIs(_matches("sin(x)", "x - x^3/6", 1.0, 3), False)

    def test_undecided_when_the_errors_do_not_tell(self) -> None:
        # flat at 0: the error falls off faster than any power, then is lost in rounding
        self.assertIsNone(_matches("exp(-1/x^2)", "0", 0.0, 4))
        # undefined on both sides of the centre
        self.assertIsNone(_matches("sqrt(-x^2-1)", "0", 0.0, 2))

    def test_only_one_side_defined(self) -> None:
        # ln(x) is defined right of 1 to x = 9 and left of it down to 0: both sides run;
        # sqrt(x - 1) near 1 is undefined on the left, and the right side decides
        self.assertIs(_matches("sqrt(x-1)^2", "(x-1)", 1.0, 1), True)


class TestRemainderNote(unittest.TestCase):
    BASE = {"center": "0", "base": "x", "lookahead": 8}

    def test_next_term(self) -> None:
        note = remainder_note(dict(self.BASE, next_term="-x^7/5040"), "x")
        self.assertEqual(note, "Near x = 0 the error is about -x^7/5040 (it grows away from 0).")

    def test_exact(self) -> None:
        note = remainder_note(dict(self.BASE, exact=True, next_term=None), "x")
        self.assertEqual(note, "Exact: the function is its own Taylor polynomial of this order.")

    def test_no_term_within_the_lookahead(self) -> None:
        note = remainder_note({"center": "1", "base": "(x-1)", "lookahead": 8, "next_term": None}, "x")
        self.assertEqual(
            note,
            "There are no further nonzero terms up to (x-1)^8, so near x = 1 the error is smaller than that power.",
        )

    def test_flat_and_one_sided(self) -> None:
        self.assertIn(
            "near x = 0 it is smaller than any", remainder_note(dict(self.BASE, flat=True, next_term=None), "x")
        )
        self.assertIn("not smooth", remainder_note(dict(self.BASE, one_sided=True, next_term=None), "x"))


if __name__ == "__main__":
    unittest.main()
