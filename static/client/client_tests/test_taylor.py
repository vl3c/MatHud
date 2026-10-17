"""Tests for the taylor_polynomial tool (utils.taylor, utils.series_checks in Brython).

TestTaylorSympy pins the polynomials and notes, and skips without the server's SymPy engine;
TestTaylorWithoutSympy pins the input refusals, which need no engine, and the error that
says SymPy is needed.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict

from utils.series_checks import taylor_matches
from utils.symbolic_checks import SymbolicChecks
from utils.taylor import Taylor

from .cas_engines import WithoutSympy, sympy_available


def _taylor(expression: str, point: Any, order: Any, variable: str = "x") -> Dict[str, Any]:
    result = Taylor.taylor_polynomial(expression, variable, point, order)
    if not isinstance(result, dict):
        raise AssertionError(f"not a result: {result}")
    return result


def _error(expression: Any, point: Any, order: Any, variable: Any = "x") -> str:
    result = Taylor.taylor_polynomial(expression, variable, point, order)
    if not (isinstance(result, str) and result.startswith("Error:")):
        raise AssertionError(f"not an error: {result}")
    return result


class TestTaylorSympy(unittest.TestCase):
    def setUp(self) -> None:
        if not sympy_available():
            self.skipTest("the server's SymPy engine is not available")

    def test_maclaurin_polynomial(self) -> None:
        self.assertEqual(
            _taylor("sin(x)", "0", 5),
            {
                "polynomial": "x - x^3/6 + x^5/120",
                "center": "0",
                "order": 5,
                "next_term": "-x^7/5040",
                "remainder_note": "Near x = 0 the error is about -x^7/5040 (it grows away from 0).",
            },
        )

    def test_polynomial_at_another_point(self) -> None:
        result = _taylor("ln(x)", "1", 3)
        self.assertEqual(result["polynomial"], "(x-1) - (x-1)^2/2 + (x-1)^3/3")
        self.assertEqual(result["next_term"], "-(x-1)^4/4")
        result = _taylor("cos(x)", "pi/4", 2)
        self.assertEqual(result["polynomial"], "sqrt(2)/2 - sqrt(2)*(x-pi/4)/2 - sqrt(2)*(x-pi/4)^2/4")
        # a number given as a number
        self.assertEqual(_taylor("e^x", 0, 2)["polynomial"], "1 + x + x^2/2")

    def test_input_pipeline(self) -> None:
        # the solve tool's pipeline: Unicode, |x| bars, ln, x(x+2) as a product
        self.assertEqual(_taylor("x(x+2)", "0", 2)["polynomial"], "2*x + x^2")
        self.assertEqual(_taylor("|x|", "1", 1)["polynomial"], "1 + (x-1)")
        self.assertEqual(_taylor("e^(2t)", "0", 2, variable="t")["polynomial"], "1 + 2*t + 2*t^2")

    def test_exact_flat_and_refused(self) -> None:
        result = _taylor("x^3+2x", "1", 5)
        self.assertEqual(result["remainder_note"], "Exact: the function is its own Taylor polynomial of this order.")
        self.assertIsNone(result["next_term"])
        self.assertIn("flat at x = 0", _taylor("exp(-1/x^2)", "0", 4)["remainder_note"])
        self.assertIn("not smooth at x = 0", _error("abs(x)", "0", 2))
        self.assertIn("not smooth at x = 0", _error("sqrt(x)", "0", 2))

    def test_the_polynomial_plots(self) -> None:
        # The text is ready for draw_function: math.js reads it and it matches e^x near 0
        result = _taylor("e^x", "0", 3)
        polynomial = SymbolicChecks.real_function(result["polynomial"], "x")
        assert polynomial is not None
        self.assertAlmostEqual(polynomial(1.0), 1 + 1 + 1 / 2 + 1 / 6)
        self.assertAlmostEqual(polynomial(2.0), 1 + 2 + 2 + 8 / 6)


class TestTaylorWithoutSympy(unittest.TestCase):
    def test_needs_sympy(self) -> None:
        with WithoutSympy():
            self.assertIn("needs the server's SymPy engine", _error("e^x", "0", 3))

    def test_refuses_calculator_spellings(self) -> None:
        self.assertIn("sin(x)", _error("sinx", "0", 3))
        self.assertIn("sin(x)", _error("sin x", "0", 3))

    def test_refuses_bad_arguments(self) -> None:
        self.assertIn("order must be a whole number from 0 to 10", _error("e^x", "0", 11))
        self.assertIn("order must be", _error("e^x", "0", -1))
        self.assertIn("order must be", _error("e^x", "0", True))
        self.assertIn("order must be", _error("e^x", "0", "3"))
        self.assertIn("point must be a real number", _error("e^x", "a", 2))
        self.assertIn("point must be a real number", _error("e^x", "", 2))
        self.assertIn("also contains a", _error("a*x^2", "0", 2))
        self.assertIn("expression alone", _error("y = sin(x)", "0", 2))
        self.assertIn("x does not appear in the function", _error("5", "0", 2))
        self.assertIn("variable must be", _error("e^x", "0", 2, variable=""))

    def test_brython_check_reads_math_js_values(self) -> None:
        function = SymbolicChecks.real_function("e^x", "x")
        right = SymbolicChecks.real_function("1 + x + x^2/2", "x")
        wrong = SymbolicChecks.real_function("1 + x + x^2/3", "x")
        assert function is not None and right is not None and wrong is not None
        self.assertIs(taylor_matches(function, right, 0.0, 2), True)
        self.assertIs(taylor_matches(function, wrong, 0.0, 2), False)
        sinc = SymbolicChecks.real_function("sin(x)/x", "x")
        sinc_polynomial = SymbolicChecks.real_function("1 - x^2/6", "x")
        assert sinc is not None and sinc_polynomial is not None
        self.assertIs(taylor_matches(sinc, sinc_polynomial, 0.0, 2), True)
