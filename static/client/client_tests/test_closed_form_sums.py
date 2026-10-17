"""Tests for closed forms of sums in the simplify tool (MathUtils.simplify of summation(term, k, a, b)).

TestClosedFormSumsSympy pins SymPy's closed forms and the numeric checks of them;
TestClosedFormSumsWithoutSympy pins the refusals, which come before any engine, and what
happens with no engine: numeric bounds are summed term by term, other bounds get an error.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict

from utils.closed_form_sums import ClosedFormSums
from utils.math_utils import MathUtils

from .cas_engines import WithoutSympy


def _simplify(text: str) -> str:
    return str(MathUtils.simplify(text))


class TestClosedFormSumsSympy(unittest.TestCase):
    def test_symbolic_bounds(self) -> None:
        for text, expected in (
            ("summation(k^2, k, 1, n)", "n*(n+1)*(2*n+1)/6"),
            ("summation('k^2', 'k', 1, n)", "n*(n+1)*(2*n+1)/6"),
            ("sum(2k-1, k, 1, n)", "n^2"),
            ("summation(1/(k(k+1)), k, 1, n)", "n/(n+1)"),
            ("summation(k, k, 1, 2n)", "n*(2*n+1)"),
            ("summation(k^2, k, n, 2n)", "n*(n+1)*(14*n+1)/6"),
            ("summation(1, k, 1, n)", "n"),
            # an index named like a constant is renamed, so neither engine reads i as sqrt(-1)
            ("summation(i^2, i, 1, n)", "n*(n+1)*(2*n+1)/6"),
            ("summation(r^k, k, 0, n)", "(r^(n+1)-1)/(r-1) (for r != 1)"),
        ):
            with self.subTest(text=text):
                self.assertEqual(_simplify(text), expected)

    def test_infinite_sums(self) -> None:
        for text, expected in (
            ("summation(1/k^2, k, 1, Infinity)", "pi^2/6 (about 1.64493406685)"),
            ("summation(1/2^k, k, 0, inf)", "2"),
            ("summation(1/k!, k, 0, Infinity)", "e (about 2.71828182846)"),
            ("summation(r^k, k, 0, Infinity)", "1/(1-r) (for -1 < r < 1)"),
            # other spellings of infinity are never a bound symbol named so
            ("summation(1/2^k, k, 1, infinity)", "1"),
            ("summation(r^k, k, 0, Inf)", "1/(1-r) (for -1 < r < 1)"),
            ("summation(1/k^2, k, 1, infty)", "pi^2/6 (about 1.64493406685)"),
            ("summation(k^2, k, 1, +INFINITY)", "The series diverges: its partial sums tend to Infinity."),
            # a condition that is a union of intervals
            ("summation(1/x^k, k, 0, Infinity)", "x/(x-1) (for x < -1 or x > 1)"),
            ("summation(C/(1+r)^k, k, 1, n)", "C/r-C/(r*(r+1)^n) (for r != -1 and r != 0)"),
        ):
            with self.subTest(text=text):
                self.assertEqual(_simplify(text), expected)

    def test_numeric_bounds(self) -> None:
        self.assertEqual(_simplify("summation(k^2, k, 1, 10)"), "385")
        self.assertEqual(_simplify("summation(k^2, k, 1, 10^6)"), "333333833333500000")
        self.assertEqual(
            _simplify("summation(1/k, k, 1, 100)"), "about 5.18737751764 (the exact value is too long to show)"
        )
        self.assertEqual(
            _simplify("summation(2^k, k, 1, 10^5)"),
            "about 1.99800418602877e+30103 (the closed form 2*2^N-2 at N = 10^5)",
        )
        self.assertEqual(
            _simplify("summation(k^2, k, 5, 1)"), "0 (an empty sum: the lower bound is above the upper bound)"
        )

    def test_divergent_series(self) -> None:
        self.assertEqual(
            _simplify("summation(1/k, k, 1, Infinity)"), "The series diverges: its partial sums tend to Infinity."
        )
        self.assertEqual(
            _simplify("summation((-1)^k, k, 0, Infinity)"),
            "The series diverges: its partial sums do not approach a finite value.",
        )

    def test_no_closed_form(self) -> None:
        self.assertEqual(
            _simplify("summation(1/k^3, k, 1, Infinity)"),
            "No closed form in calculator functions (SymPy writes zeta(3)); the sum is about 1.20205690316.",
        )
        result = _simplify("summation(1/k, k, 1, n)")
        self.assertTrue(result.startswith("Error:"), result)
        self.assertIn("harmonic(n)", result)
        self.assertIn("summation(1/k, k, 1, 100)", result)
        # SymPy calls this divergent: no verdict is taken from it
        result = _simplify("summation(sin(k)/k, k, 1, Infinity)")
        self.assertTrue(result.startswith("Error: SymPy found no closed form"), result)

    def test_undefined_terms(self) -> None:
        for text in ("summation(1/(k-3), k, 1, 5)", "summation(1/(k-3)^2, k, 1, Infinity)"):
            with self.subTest(text=text):
                self.assertIn("is undefined", _simplify(text))

    def test_a_term_without_the_index_summed_to_infinity(self) -> None:
        self.assertIn("does not contain the index k", _simplify("summation(x, k, 1, Infinity)"))

    def test_refuted_results_are_errors(self) -> None:
        for text, result in (
            ("summation(k^2, k, 1, 10)", {"value": "386"}),
            ("summation(k^2, k, 1, n)", {"value": "n^2*(n+1)/2"}),
            ("summation(1/k^2, k, 1, Infinity)", {"value": "pi^2/6+1/10000"}),
            ("summation(1/k^2, k, 1, Infinity)", {"diverges": True, "limit": "Infinity"}),
            ("summation(k^2, k, 1, 10^6)", {"value": "1", "general": "N^2", "general_symbol": "N"}),
        ):
            with self.subTest(text=text, result=result):
                answer = self._with_sympy_answer(text, result)
                self.assertTrue(answer.startswith("Error:"), answer)

    def _with_sympy_answer(self, text: str, result: Dict[str, Any]) -> str:
        original = ClosedFormSums._ask_sympy
        try:
            ClosedFormSums._ask_sympy = staticmethod(lambda given: {"result": result})  # type: ignore[method-assign]
            return _simplify(text)
        finally:
            ClosedFormSums._ask_sympy = original  # type: ignore[method-assign]


class TestClosedFormSumsWithoutSympy(unittest.TestCase):
    def test_refusals(self) -> None:
        for text, hint in (
            ("summation(kx^k, k, 0, Infinity)", "write k*x"),
            ("summation(x^k/2k!, k, 0, Infinity)", "write 2*k! or (2k)!"),
            ("summation(sink/k^2, k, 1, Infinity)", "sin(k)"),
            ("summation(1/k(k+1), k, 1, n)", "ambiguous"),
            ("2*summation(k, k, 1, n)", "summation(term, k, a, b)"),
            ("Sum(k^2, (k, 1, n))", "summation(term, k, a, b)"),
            ("∑(k^2, k, 1, n)", "summation(term, k, a, b)"),
            ("\\sum_{k=1}^{n} k^2", "summation(term, k, a, b)"),
            ("Σk^2", "summation(term, k, a, b)"),
            ("summation(k^2, k=1..n)", "four arguments"),
        ):
            with self.subTest(text=text):
                result = _simplify(text)
                self.assertTrue(result.startswith("Error:"), result)
                self.assertIn(hint, result)

    def test_numeric_bounds_are_summed_term_by_term(self) -> None:
        with WithoutSympy():
            result = _simplify("summation(k^2, k, 1, 10)")
        self.assertTrue(result.startswith("385 (summed numerically"), result)

    def test_other_bounds_need_sympy(self) -> None:
        with WithoutSympy():
            result = _simplify("summation(k^2, k, 1, n)")
        self.assertTrue(result.startswith("Error:"), result)
        self.assertIn("SymPy", result)

    def test_other_expressions_are_not_sums(self) -> None:
        self.assertIsNone(ClosedFormSums.simplify("x^2 + 2x + 1"))
        self.assertIsNone(ClosedFormSums.simplify("sum([1, 2, 3])"))
