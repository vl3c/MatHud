"""Every symbol the chat input can insert, checked against the expression parser.

The symbol palette, Alt shortcuts and backslash completion insert the symbols of
math_symbols.SYMBOLS. This file gives each of them an explicit expectation: the exact
rewrite of a representative expression by fix_math_expression (Python and math.js modes)
and normalize_unicode_math (nerdamer), and either its value in the Python plotting
namespace or the clean ValueError validation raises. Symbols with no arithmetic meaning
(∈, ∪, ∠, ∫, ...) pass through unchanged and are rejected by validation, never crash.

A symbol added to the table without an expectation here fails
test_every_symbol_has_an_expectation.
"""

from __future__ import annotations

import ast
import math
import unittest
from typing import Any, Dict, NamedTuple, Optional

from server_tests import client_renderer  # noqa: F401  (browser stub, so MathUtils imports)
from expression_validator import ExpressionValidator
from math_symbols import SYMBOLS


class Expectation(NamedTuple):
    expression: str  # a representative expression using the symbol
    python: str  # fix_math_expression(expression, python_compatible=True)
    mathjs: str  # fix_math_expression(expression, python_compatible=False)
    nerdamer: str  # normalize_unicode_math(expression)
    value: Optional[float] = None  # value in the Python namespace, when it evaluates
    variables: Optional[Dict[str, Any]] = None
    error: Optional[str] = None  # part of the ValueError validation raises, when it does not


def _same(expression: str, error: str) -> Expectation:
    """A symbol left untouched by every normaliser and rejected by validation."""
    return Expectation(expression, expression, expression, expression, error=error)


DEG_30 = str(30 * math.pi / 180)

OPERATORS: Dict[str, Expectation] = {
    "×": Expectation("3×4", "3*4", "3*4", "3*4", 12),
    "÷": Expectation("6÷4", "6/4", "6/4", "6/4", 1.5),
    "·": Expectation("a·b", "a*b", "a*b", "a*b", 6, {"a": 2, "b": 3}),
    "√": Expectation("x√y", "x*sqrt(y)", "x*sqrt(y)", "x*sqrt(y)", 6, {"x": 3, "y": 4}),
    # Comparisons are rewritten but are not expressions the evaluators accept
    "≠": Expectation("x ≠ 3", "x != 3", "x != 3", "x != 3", error="Compare"),
    "≤": Expectation("x ≤ 3", "x <= 3", "x <= 3", "x <= 3", error="Compare"),
    "≥": Expectation("x ≥ 3", "x >= 3", "x >= 3", "x >= 3", error="Compare"),
    "°": Expectation("sin(30°)", f"sin({DEG_30})", f"sin({DEG_30})", "sin((30*pi/180))", 0.5),
    # Not arithmetic: left as typed
    "±": _same("x ± 1", "invalid character '±'"),
    "≈": _same("x ≈ 1", "invalid character '≈'"),
}

SUPERSCRIPTS: Dict[str, Expectation] = {
    "⁰": Expectation("x⁰", "x**0", "x^0", "x^0", 1, {"x": 5}),
    "¹": Expectation("x¹", "x**1", "x^1", "x^1", 5, {"x": 5}),
    "²": Expectation("x²", "x**2", "x^2", "x^2", 25, {"x": 5}),
    "³": Expectation("x³", "x**3", "x^3", "x^3", 125, {"x": 5}),
    "⁴": Expectation("x⁴", "x**4", "x^4", "x^4", 16, {"x": 2}),
    "⁵": Expectation("x⁵", "x**5", "x^5", "x^5", 32, {"x": 2}),
    "⁶": Expectation("x⁶", "x**6", "x^6", "x^6", 64, {"x": 2}),
    "⁷": Expectation("x⁷", "x**7", "x^7", "x^7", 128, {"x": 2}),
    "⁸": Expectation("x⁸", "x**8", "x^8", "x^8", 256, {"x": 2}),
    "⁹": Expectation("x⁹", "x**9", "x^9", "x^9", 512, {"x": 2}),
    "⁻": Expectation("x⁻¹", "x**(-1)", "x^(-1)", "x^(-1)", 0.25, {"x": 4}),
    "ⁿ": Expectation("xⁿ", "x**(n)", "x^(n)", "x^(n)", 8, {"x": 2, "n": 3}),
}

CONSTANTS: Dict[str, Expectation] = {
    "π": Expectation("2π", "2*pi", "2*pi", "2*pi", 2 * math.pi),
    "∞": Expectation("1/∞", "1/inf", "1/Infinity", "1/Infinity", 0),
}

# Greek letters stay variable names; each is a single-letter factor (2θ is 2*θ)
GREEK: Dict[str, Expectation] = {
    letter: Expectation(f"2{letter}+1", f"2*{letter}+1", f"2*{letter}+1", f"2*{letter}+1", 4, {letter: 1.5})
    for letter in "αβγδεζηθκλμξρστφχψωΓΔΘΠΣΦΩ"
}

GEOMETRY: Dict[str, Expectation] = {
    "∠": _same("∠ABC", "invalid character '∠'"),
    "⊥": _same("a ⊥ b", "invalid character '⊥'"),
    "∥": _same("a ∥ b", "invalid character '∥'"),
    "△": _same("△ABC", "invalid character '△'"),
    "≅": _same("a ≅ b", "invalid character '≅'"),
    "∼": _same("a ∼ b", "invalid character '∼'"),
    "′": _same("f′(x)", "invalid character '′'"),
    "″": _same("f″(x)", "invalid character '″'"),
}

SETS_AND_LOGIC: Dict[str, Expectation] = {
    "∈": _same("x ∈ A", "invalid character '∈'"),
    "∉": _same("x ∉ A", "invalid character '∉'"),
    "⊂": _same("A ⊂ B", "invalid character '⊂'"),
    "⊆": _same("A ⊆ B", "invalid character '⊆'"),
    "∪": _same("A ∪ B", "invalid character '∪'"),
    "∩": _same("A ∩ B", "invalid character '∩'"),
    "∅": _same("∅", "invalid character '∅'"),
    # Python would fold these letterlike symbols into the names R, N, Z, Q and C
    "ℝ": _same("2ℝ", "Unsupported symbol 'ℝ'"),
    "ℕ": _same("2ℕ", "Unsupported symbol 'ℕ'"),
    "ℤ": _same("2ℤ", "Unsupported symbol 'ℤ'"),
    "ℚ": _same("2ℚ", "Unsupported symbol 'ℚ'"),
    "ℂ": _same("2ℂ", "Unsupported symbol 'ℂ'"),
    "∧": _same("p ∧ q", "invalid character '∧'"),
    "∨": _same("p ∨ q", "invalid character '∨'"),
    "¬": _same("¬p", "invalid character '¬'"),
    "→": _same("x → 0", "invalid character '→'"),
    "⇒": _same("p ⇒ q", "invalid character '⇒'"),
    "⇔": _same("p ⇔ q", "invalid character '⇔'"),
    "∀": _same("∀x", "invalid character '∀'"),
    "∃": _same("∃x", "invalid character '∃'"),
}

CALCULUS: Dict[str, Expectation] = {
    "∫": _same("∫x", "invalid character '∫'"),
    "∑": _same("∑x", "invalid character '∑'"),
    "∏": _same("∏x", "invalid character '∏'"),
    "∂": _same("∂f", "invalid character '∂'"),
    "∇": _same("∇f", "invalid character '∇'"),
    # "lim" is inserted as text; fix_math_expression spells it limit( for nerdamer
    "lim": Expectation("lim(sin(x)/x, x, 0)", "limit(sin(x)/x, x, 0)", "limit(sin(x)/x, x, 0)", "lim(sin(x)/x, x, 0)"),
}

CATEGORIES: Dict[str, Dict[str, Expectation]] = {
    "operators": OPERATORS,
    "superscripts": SUPERSCRIPTS,
    "constants": CONSTANTS,
    "greek": GREEK,
    "geometry": GEOMETRY,
    "sets and logic": SETS_AND_LOGIC,
    "calculus": CALCULUS,
}
EXPECTATIONS: Dict[str, Expectation] = {}
for _category in CATEGORIES.values():
    EXPECTATIONS.update(_category)


class TestMathSymbolParsing(unittest.TestCase):
    def test_every_symbol_has_an_expectation(self) -> None:
        self.assertEqual(sum(len(category) for category in CATEGORIES.values()), len(EXPECTATIONS))
        self.assertEqual(set(EXPECTATIONS), {entry.symbol for entry in SYMBOLS})

    def test_representative_expressions_use_the_inserted_text(self) -> None:
        for entry in SYMBOLS:
            with self.subTest(symbol=entry.symbol, latex=entry.latex):
                self.assertIn(entry.insert, EXPECTATIONS[entry.symbol].expression)

    def test_normalised_forms(self) -> None:
        for symbol, expected in EXPECTATIONS.items():
            with self.subTest(symbol=symbol, expression=expected.expression):
                self.assertEqual(ExpressionValidator.fix_math_expression(expected.expression, True), expected.python)
                self.assertEqual(ExpressionValidator.fix_math_expression(expected.expression, False), expected.mathjs)
                self.assertEqual(ExpressionValidator.normalize_unicode_math(expected.expression), expected.nerdamer)

    def test_evaluable_symbols_evaluate(self) -> None:
        for symbol, expected in EXPECTATIONS.items():
            if expected.value is None:
                continue
            with self.subTest(symbol=symbol, expression=expected.expression):
                ExpressionValidator.validate_expression_tree(expected.python)
                namespace = ExpressionValidator._get_variables_and_functions(0)
                namespace.update(expected.variables or {})
                code = compile(ast.parse(expected.python, mode="eval"), "<symbol>", "eval")
                self.assertAlmostEqual(eval(code, namespace), expected.value)

    def test_unsupported_symbols_fail_validation_cleanly(self) -> None:
        for symbol, expected in EXPECTATIONS.items():
            if expected.error is None:
                continue
            with self.subTest(symbol=symbol, expression=expected.expression):
                with self.assertRaises(ValueError) as context:
                    ExpressionValidator.validate_expression_tree(expected.python)
                self.assertIn(expected.error, str(context.exception))

    def test_every_symbol_is_evaluable_or_rejected(self) -> None:
        # Only "lim" needs nerdamer, which the client tests exercise
        undecided = [
            symbol for symbol, expected in EXPECTATIONS.items() if expected.value is None and not expected.error
        ]
        self.assertEqual(undecided, ["lim"])

    def test_counts_per_category(self) -> None:
        counts = {name: len(category) for name, category in CATEGORIES.items()}
        self.assertEqual(
            counts,
            {
                "operators": 10,
                "superscripts": 12,
                "constants": 2,
                "greek": 26,
                "geometry": 8,
                "sets and logic": 20,
                "calculus": 6,
            },
        )


if __name__ == "__main__":
    unittest.main()
