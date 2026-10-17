"""Tests for partial fractions in the expand tool (MathUtils.expand with a variable, utils.partial_fractions).

TestPartialFractionsSympy pins SymPy's decompositions and skips without the server's SymPy
engine; TestPartialFractionsWithoutSympy pins what happens with no engine: an error for a
ratio of polynomials, the ordinary expansion for anything else.
"""

from __future__ import annotations

import unittest

from utils.math_utils import MathUtils
from utils.partial_fractions import UNCHANGED_NOTE, PartialFractions

from .cas_engines import WithoutSympy, sympy_available


class TestPartialFractionsSympy(unittest.TestCase):
    def setUp(self) -> None:
        if not sympy_available():
            self.skipTest("the server's SymPy engine is not available")

    def test_partial_fractions(self) -> None:
        self.assertEqual(MathUtils.expand("2/(x^2-1)", "x"), "1/(x-1) - 1/(x+1)")
        self.assertEqual(MathUtils.expand("(3x+5)/((x+1)(x+2))", "x"), "2/(x+1) + 1/(x+2)")
        self.assertEqual(MathUtils.expand("(x^2+1)/(x-1)", "x"), "x + 1 + 2/(x-1)")
        # x(x^2+1) is a product, as the solve tool reads it
        self.assertEqual(MathUtils.expand("1/(x(x^2+1))", "x"), "1/x - x/(x^2+1)")
        self.assertEqual(MathUtils.expand("1/(t^2-t)", "t"), "-1/t + 1/(t-1)")

    def test_single_fraction_is_noted(self) -> None:
        self.assertEqual(MathUtils.expand("1/(x^2-2)", "x"), "1/(x^2-2)" + UNCHANGED_NOTE)

    def test_a_spurious_variable_keeps_the_ordinary_expansion(self) -> None:
        # a polynomial, a second variable, a variable the expression lacks, a non-rational function
        self.assertEqual(MathUtils.expand("(x + 1)^2", "x"), MathUtils.expand("(x + 1)^2"))
        self.assertEqual(MathUtils.expand("(x+1)/(x*y)", "x"), MathUtils.expand("(x+1)/(x*y)"))
        self.assertEqual(MathUtils.expand("(a+b)^2", "x"), MathUtils.expand("(a+b)^2"))
        self.assertEqual(MathUtils.expand("sin(x)/(x+1)", "x"), MathUtils.expand("sin(x)/(x+1)"))
        self.assertEqual(MathUtils.expand("(x + 1)^2", ""), MathUtils.expand("(x + 1)^2"))

    def test_refusals(self) -> None:
        self.assertIn("degree up to 10", MathUtils.expand("1/(x^12-1)", "x"))
        self.assertIn("sin(x)", MathUtils.expand("1/(sinx+1)", "x"))

    def test_a_refuted_result_is_an_error(self) -> None:
        original = PartialFractions._ask_sympy
        try:
            PartialFractions._ask_sympy = staticmethod(  # type: ignore[method-assign]
                lambda expression, variable: {"result": {"fractions": "1/(x-1) + 1/(x+1)", "unchanged": False}}
            )
            result = MathUtils.expand("2/(x^2-1)", "x")
        finally:
            PartialFractions._ask_sympy = original  # type: ignore[method-assign]
        self.assertTrue(result.startswith("Error:"), result)
        self.assertIn("numeric check", result)


class TestPartialFractionsWithoutSympy(unittest.TestCase):
    def test_a_ratio_of_polynomials_needs_sympy(self) -> None:
        with WithoutSympy():
            result = MathUtils.expand("2/(x^2-1)", "x")
        self.assertTrue(result.startswith("Error:"), result)
        self.assertIn("SymPy", result)

    def test_anything_else_is_expanded_as_usual(self) -> None:
        with WithoutSympy():
            self.assertEqual(MathUtils.expand("(x + 1)^2", "x"), "1+2*x+x^2")
            self.assertEqual(MathUtils.expand("(x+1)/(x*y)", "x"), MathUtils.expand("(x+1)/(x*y)"))
            self.assertEqual(MathUtils.expand("(x + 1)^2", None), "1+2*x+x^2")

    def test_rationalize_reads_ratios_of_polynomials(self) -> None:
        self.assertTrue(PartialFractions._is_rational_in("2/(x^2-1)", "x"))
        self.assertTrue(PartialFractions._is_rational_in("e/(x-pi)", "x"))
        self.assertFalse(PartialFractions._is_rational_in("(x+1)^2", "x"))
        self.assertFalse(PartialFractions._is_rational_in("sin(x)/(x+1)", "x"))
        self.assertFalse(PartialFractions._is_rational_in("(x+1)/(x*y)", "x"))
