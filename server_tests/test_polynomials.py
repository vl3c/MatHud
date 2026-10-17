"""Tests for utils.polynomials, the expander behind the asymptote analysis and the sign chart's seeds.

The trees are built from Python's ast by real_sets_harness.ast_tree, as MathUtils._math_tree
builds them from math.js's parse tree in the browser.
"""

from __future__ import annotations

import unittest
from fractions import Fraction

from server_tests import real_sets_harness as harness
from utils.polynomials import MAX_DEGREE, _exact_number, coefficients


def _coefficients(text: str):  # type: ignore[no-untyped-def]
    return coefficients(harness.ast_tree(text))


class TestPolynomials(unittest.TestCase):
    def test_sums_products_and_powers(self) -> None:
        self.assertEqual(_coefficients("(x+1)^2"), [1.0, 2.0, 1.0])
        self.assertEqual(_coefficients("2*x*(x+3)"), [0.0, 6.0, 2.0])
        self.assertEqual(_coefficients("-x^2+3"), [3.0, 0.0, -1.0])
        self.assertEqual(_coefficients("x/2+1/3"), [1 / 3, 0.5])
        self.assertEqual(
            _coefficients("(x-3)^8"), [6561.0, -17496.0, 20412.0, -13608.0, 5670.0, -1512.0, 252.0, -24.0, 1.0]
        )

    def test_numbers_are_read_exactly(self) -> None:
        # 0.1 + 0.2 is exactly 3/10 before the final conversion
        self.assertEqual(_coefficients("0.1*x+0.2*x"), [0.0, float(Fraction(3, 10))])
        self.assertEqual(_coefficients("x-x"), [0.0])
        self.assertEqual(_coefficients("5"), [5.0])
        self.assertEqual(_coefficients("(x-1)*(x-1.001)"), [1.001, -2.001, 1.0])

    def test_decimal_literals(self) -> None:
        # Brython's Fraction() cannot read "1.001", so the expander parses literals itself
        for text, expected in (
            ("1.001", Fraction(1001, 1000)),
            ("0.1", Fraction(1, 10)),
            (".5", Fraction(1, 2)),
            ("5.", Fraction(5)),
            ("2.5e3", Fraction(2500)),
            ("1e-07", Fraction(1, 10**7)),
            ("1E+2", Fraction(100)),
        ):
            self.assertEqual(_exact_number(text), expected, text)

    def test_constants(self) -> None:
        found = _coefficients("pi*x^2+sqrt(2)")
        assert found is not None
        self.assertAlmostEqual(found[0], 2**0.5)
        self.assertAlmostEqual(found[2], 3.141592653589793)

    def test_quotients_that_divide(self) -> None:
        self.assertEqual(_coefficients("(x^2-1)/(x-1)"), [1.0, 1.0])
        self.assertEqual(_coefficients("(x^3-8)/(x-2)"), [4.0, 2.0, 1.0])
        self.assertEqual(_coefficients("((x-2)^2/(x+1))*(x+1)"), [4.0, -4.0, 1.0])
        self.assertEqual(_coefficients("1/(1/x)"), [0.0, 1.0])
        self.assertEqual(_coefficients("(x+1)^(-1)*(x+1)"), [1.0])

    def test_not_polynomials(self) -> None:
        for text in ("1/x", "(x^2+1)/(x-1)", "sin(x)", "e^x", "2^x", "x^(1/2)", "x^y", "x^0.5", "1/(x-x)"):
            with self.subTest(text=text):
                self.assertIsNone(_coefficients(text))

    def test_degree_cap(self) -> None:
        self.assertIsNotNone(_coefficients(f"x^{MAX_DEGREE}"))
        self.assertIsNone(_coefficients(f"x^{MAX_DEGREE + 1}"))
        self.assertIsNone(_coefficients("(x+1)^100"))

    def test_huge_numbers_do_not_hang(self) -> None:
        self.assertIsNone(_coefficients("x*10^10^10"))

    def test_trees_by_hand(self) -> None:
        tree = ("op", "add", [("op", "pow", [("var",), ("num", "2")]), ("const", 1.5)])
        self.assertEqual(coefficients(tree), [1.5, 0.0, 1.0])
        self.assertIsNone(coefficients(("op", "mod", [("var",), ("num", "2")])))
        self.assertIsNone(coefficients(("other",)))
        self.assertIsNone(coefficients(("const", float("inf"))))


if __name__ == "__main__":
    unittest.main()
