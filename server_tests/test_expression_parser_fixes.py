"""Tests for ASCII forms the expression parser reads as calculators do.

sin^2(x) and sin^-1(x) are the ASCII forms of sin²(x) and sin⁻¹(x): they used to validate
and then fail at run time (sin**2*(x) raises TypeError). The client suite checks the same
forms end to end through math.js (client_tests/test_math_functions.py).
"""

from __future__ import annotations

import math
import unittest
from typing import Dict, Tuple

from server_tests import client_renderer  # noqa: F401  (installs the browser stub)
from expression_validator import ExpressionValidator

# Input -> (python_compatible=True, python_compatible=False)
CARET_CASES: Dict[str, Tuple[str, str]] = {
    "sin^2(x)": ("sin(x)**2", "sin(x)^2"),
    "sin**2(x)": ("sin(x)**2", "sin(x)^2"),
    "sin^-1(x)": ("asin(x)", "asin(x)"),
    "sin^(-1)(x)": ("asin(x)", "asin(x)"),
    "tan^-1(1)": ("atan(1)", "atan(1)"),
    "sine^2(x)": ("sin(x)**2", "sin(x)^2"),
    "2sin^2(x)": ("2*sin(x)**2", "2*sin(x)^2"),
    "ln^2(x)": ("log(x)**2", "log(x)^2"),
    "log^-1(x)": ("log(x)**(-1)", "log(x)^(-1)"),
    "sin^-2(x)": ("sin(x)**(-2)", "sin(x)^(-2)"),
    "sin ^ 2 (x)": ("sin(x)**2", "sin(x)^2"),
    "sin^2(cos^2(x))": ("sin(cos(x)**2)**2", "sin(cos(x)^2)^2"),
    "sin^2(x) + cos^2(x)": ("sin(x)**2 + cos(x)**2", "sin(x)^2 + cos(x)^2"),
    # Calculators read sin^2(x)^3 as (sin(x)^2)^3; this is sin(x)^(2^3), as sin²(x)³ is
    "sin^2(x)^3": ("sin(x)**2**3", "sin(x)^2^3"),
}

# Forms that keep their old reading
UNCHANGED_CASES: Dict[str, Tuple[str, str]] = {
    "x^2(x+1)": ("x**2*(x+1)", "x^2*(x+1)"),
    "e^2(x)": ("e**2*(x)", "e^2*(x)"),
    "x2sin^2(x)": ("x2sin**2*(x)", "x2sin^2*(x)"),
    "sin^2 x": ("sin**2 x", "sin^2 x"),
    "sin^2*(x)": ("sin**2*(x)", "sin^2*(x)"),
    "sin(x)^2": ("sin(x)**2", "sin(x)^2"),
    "sin^0.5(x)": ("sin**0.5*(x)", "sin^0.5*(x)"),
}


def _fix_both(expression: str) -> Tuple[str, str]:
    return (
        ExpressionValidator.fix_math_expression(expression, python_compatible=True),
        ExpressionValidator.fix_math_expression(expression, python_compatible=False),
    )


class TestCaretFunctionPowers(unittest.TestCase):
    def test_powers_on_function_names_move_after_the_call(self) -> None:
        for expression, expected in CARET_CASES.items():
            with self.subTest(expression=expression):
                self.assertEqual(_fix_both(expression), expected)

    def test_other_powers_are_unchanged(self) -> None:
        for expression, expected in UNCHANGED_CASES.items():
            with self.subTest(expression=expression):
                self.assertEqual(_fix_both(expression), expected)

    def test_converted_forms_evaluate(self) -> None:
        cases = {
            "sin^2(x)": math.sin(0.5) ** 2,
            "sin^-1(x)": math.asin(0.5),
            "sin^(-1)(x)": math.asin(0.5),
            "sin**2(x)": math.sin(0.5) ** 2,
            "sine^2(x)": math.sin(0.5) ** 2,
            "2sin^2(x)": 2 * math.sin(0.5) ** 2,
            "log^-1(x)": 1 / math.log(0.5),
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertAlmostEqual(ExpressionValidator.parse_function_string(expression)(0.5), expected)

    def test_cas_input_reads_the_same_forms(self) -> None:
        # The CAS tools get raw text: derive("sin^2(x)") used to reach nerdamer as sin^2*x
        from utils.math_utils import MathUtils

        self.assertEqual(MathUtils._cas_input("sin^2(x)"), "sin(x)^2")
        self.assertEqual(MathUtils._cas_input("sin^-1(x) + ln^2(x)"), "asin(x) + log(x)^2")
        self.assertEqual(ExpressionValidator.convert_caret_function_powers("x^2 + 1"), "x^2 + 1")


if __name__ == "__main__":
    unittest.main()
