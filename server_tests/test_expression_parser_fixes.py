"""Tests for ASCII forms the expression parser reads as calculators do.

sin^2(x) and sin^-1(x) are the ASCII forms of sin²(x) and sin⁻¹(x): they used to validate
and then fail at run time (sin**2*(x) raises TypeError). ln|x| became the unknown name
lnabs(x). The client suite checks the same forms end to end through math.js
(client_tests/test_math_functions.py).
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


# Absolute value bars right after a function are its argument (python_compatible=True)
# Call aliases before bars: the same reading in Python and, after rename_for_mathjs, in math.js
ALIAS_BAR_CASES: Dict[str, str] = {
    "cosine|x|": "cos(abs(x))",
    "sine|x|": "sin(abs(x))",
    "sgn|x|": "sign(abs(x))",
    "arcsine|x|": "asin(abs(x))",
    "hyperbolic sine|x|": "sinh(abs(x))",
    "real|x|": "re(abs(x))",
    "imag|x|": "im(abs(x))",
    "conjugate|x|": "conj(abs(x))",
    "2cosine|x|": "2*cos(abs(x))",
    "|sine| + cosine|x|": "abs(sine) + cos(abs(x))",
}

BAR_CASES: Dict[str, str] = {
    **ALIAS_BAR_CASES,
    "ln|x|": "log(abs(x))",
    "sin|x|": "sin(abs(x))",
    "2ln|x|": "2*log(abs(x))",
    "ln |x|": "log (abs(x))",
    "ln|x| + |x|ln|y|": "log(abs(x)) + abs(x)log(abs(y))",
    # Unchanged: not after a one-argument function spelling
    "kiln|x|": "kilnabs(x)",
    "x|y|": "xabs(y)",
    "|x|": "abs(x)",
    "mod|x|": "modabs(x)",
    "hybrid|text|": "hybridabs(text)",
    "xcosine|x|": "xcosineabs(x)",
}


class TestAbsoluteBars(unittest.TestCase):
    def test_bars_after_a_function_are_its_argument(self) -> None:
        for expression, expected in BAR_CASES.items():
            with self.subTest(expression=expression):
                self.assertEqual(ExpressionValidator.fix_math_expression(expression, python_compatible=True), expected)

    def test_math_js_text_keeps_bars_until_the_math_js_boundary(self) -> None:
        import expression_functions as ef

        self.assertEqual(ExpressionValidator.fix_math_expression("ln|x|"), "log|x|")
        cases = {
            "|x|": "abs(x)",
            "log|x|": "log(abs(x))",
            "|x - 3|": "abs(x - 3)",
            "|x|%|y|": "abs(x) mod abs(y)",
            "trunc|x|": "fix(abs(x))",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(ef.rename_for_mathjs(text), expected)
                self.assertEqual(ef.rename_for_mathjs(expected), expected)

    def test_math_js_text_reads_call_aliases_before_bars(self) -> None:
        import expression_functions as ef

        self.assertEqual(ExpressionValidator.fix_math_expression("cosine|x|"), "cos|x|")
        self.assertEqual(ExpressionValidator.fix_math_expression("|sine|"), "|sine|")
        for expression, expected in ALIAS_BAR_CASES.items():
            with self.subTest(expression=expression):
                self.assertEqual(ef.rename_for_mathjs(ExpressionValidator.fix_math_expression(expression)), expected)

    def test_ln_of_bars_plots(self) -> None:
        function = ExpressionValidator.parse_function_string("ln|x|")
        self.assertAlmostEqual(function(-math.e), 1.0)
        self.assertAlmostEqual(function(math.e), 1.0)


if __name__ == "__main__":
    unittest.main()
