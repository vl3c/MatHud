"""Goldens for parsed and translated expressions (functions of x, parametric curves of t).

They record what ``ExpressionValidator.parse_function_string`` and
``parse_parametric_expression`` return at a few sample points, the messages of
expressions refused at parse time, and the text ``Function.translate`` and
``PiecewiseFunction.translate`` write. Refactors of the expression compiler must keep
every section equal; a deliberate behaviour change regenerates the golden and says which
entries changed.

The sample points are floats, as the renderers pass them (``x!`` at 3 gives 6.0 but
raises at 3.0). Rows backed by MathUtils (sqrt, pow, det, mean...) evaluate through the
server stub's fake ``window.math`` (``server_tests/client_renderer``), so their values
are the stub's, not the browser's; ``client_tests/test_compiled_expression.py`` checks
real values in the browser. Messages are normalised where they quote Python's own
wording (``ast.dump`` and syntax-error details differ between Python versions).

Regenerate with ``python -m server_tests.test_compiled_expression_golden``.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Any, Callable, Dict, List, Optional, Tuple

from server_tests import client_renderer  # noqa: F401  (installs the browser stub)
from drawables.function import Function
from drawables.piecewise_function import PiecewiseFunction
from expression_validator import ExpressionValidator

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "data", "compiled_expressions_golden.json")

SAMPLE_POINTS: Tuple[float, ...] = (-2.5, -1.0, 0.0, 0.5, 1.0, 3.0)
OFFSETS: Tuple[float, ...] = (2, -2, -0.5, 0)

FUNCTION_CORPUS: Tuple[str, ...] = (
    # ASCII
    "x",
    "x^2",
    "x**2",
    "-x",
    "2x",
    "2 x",
    "3x^2+2x-1",
    "x^x",
    "1/x",
    "x/0",
    "1/(x^2 - 1)",
    "x-(-3)",
    "sin(x)",
    "cos(x)*x",
    "tan(x)",
    "sinx",
    "cosx + x",
    "sin(x)^2 + cos(x)^2",
    "sine(x)",
    "arcsine(x)",
    "asin(x)",
    "sinh(x)",
    "sec(x)",
    "acot(x)",
    "exp(x)",
    "exp(-x^2/2)/sqrt(2*pi)",
    "e^x",
    "e^-x",
    "xe^x",
    "x*exp(x)",
    "log(x)",
    "ln(x)",
    "log10(x)",
    "log2(x)",
    "sqrt(x)",
    "pow(x, 2)",
    "abs(x)",
    "|x|",
    "x!",
    "(x+1)!",
    "factorial(x)",
    "max(x,2)",
    "min(x, 1)",
    "floor(x)",
    "ceil(x)",
    "round(x)",
    "trunc(x)",
    "mean([x, 1])",
    "det([[x, 1], [0, 1]])",
    "1e-5x",
    "2.5e3x",
    "pi*x",
    "2pi x",
    "sin(30°) + x",
    "x°",
    "x + i",
    "i*x",
    "a*x",
    "xy",
    "x2 + x",
    "x # comment",
    "inf",
    "min(x, Infinity)",
    "x % 2",
    # Unicode (characters the validator's one-for-one table covers)
    "x²",
    "x³ - 4x",
    "sin²(x)",
    "sin⁻¹(x)",
    "sin²x + cos²x",
    "√x",
    "2πx",
    "e⁻ˣ",
    "x²⁺¹",
    "3×x − x÷2",
    "x·sinπ",
    "min(x, ∞)",
    "θ",
    "ϑ*x",
    "Δx + x",
    "µ",
)

PARAMETRIC_CORPUS: Tuple[str, ...] = (
    "t",
    "-t",
    "2t",
    "cos(t)",
    "5*sin(t)",
    "8*sin(3*t)",
    "t*cos(t)",
    "6*cos(4*t)*cos(t)",
    "t^2",
    "t/pi",
    "exp(t)",
    "log(t)",
    "ln(t)",
    "sqrt(t)",
    "abs(t)",
    "|t|",
    "round(t)",
    "max(t, 1)",
    "min(t, -1)",
    "floor(t)",
    "trunc(t)",
    "pow(t, 2)",
    "t!",
    "factorial(t)",
    "tan(t)",
    "acot(t)",
    "derive(t, t)",
    "mean([t])",
    "x",
    "t⁻¹",
    "t⁽²⁺¹⁾",
    "sec²(t) - tan²(t)",
    "2πt",
    "max(t, -∞)",
    "θ",
)

ERROR_CORPUS: Tuple[str, ...] = (
    "__import__('os')",
    "open('f')",
    "a.x",
    "a[0]",
    "lambda: 1",
    "",
    "x=1",
    "x > 1",
    "x // 2",
    "ℝ",
    "x(x+1)",
    "2in + x",
    "sin(x",
)

TRANSLATE_CORPUS: Tuple[str, ...] = (
    "x",
    "x + 1",
    "sin(x)",
    "x^2",
    "1/x",
    "sqrt(x)",
    "2*x + 3",
    "2x + 3",
    "exp(x)",
    "log(x)",
    "max(x,2)",
    "x*exp(x)",
    "x2 + x",
    "xy + x",
    "Δx + x",
    "θ + x",
    "1e5*x",
    "1e-5x",
    "sin(x)+x^2",
    "|x|",
    "x!",
    "e^x",
    "sin²(x)",
    "2πx",
    "a*x",
)

PIECEWISE_CORPUS: Tuple[Tuple[Tuple[str, Optional[float], Optional[float]], ...], ...] = (
    (("x^2", None, 0), ("max(x,1)", 0, None)),
    (("sin(x)", None, None),),
    (("exp(x)", None, 1), ("x*exp(x)", 1, 5), ("|x|", 5, None)),
    (("2x", -3, 3),),
)


def _message(error: Exception) -> str:
    """The error's class and message, without the parts that quote Python's own wording."""
    text = str(error)
    marker = "Disallowed function: "
    if marker in text:
        text = text[: text.index(marker) + len(marker)] + "<ast>"
    if text.startswith("Syntax error in expression '") and "': " in text:
        text = text[: text.rindex("': ") + 3] + "<detail>"
    return f"{type(error).__name__}: {text}"


def _samples(parse: Callable[[str], Callable[[float], float]], expression: str) -> Any:
    try:
        function = parse(expression)
    except Exception as error:
        return "PARSE " + _message(error)
    values: List[str] = []
    for point in SAMPLE_POINTS:
        try:
            values.append(repr(function(point)))
        except Exception as error:
            values.append("EXCEPTION " + type(error).__name__)
    return values


def _translated(make: Callable[[], Any], read: Callable[[Any], Any]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for dx in OFFSETS:
        for dy in OFFSETS:
            try:
                drawable = make()
                drawable.translate(dx, dy)
                results[f"{dx},{dy}"] = read(drawable)
            except Exception as error:
                results[f"{dx},{dy}"] = "EXCEPTION " + _message(error)
    return results


def _make_function(expression: str) -> Function:
    # Precomputed analysis skips the asymptote and periodicity scans (not under test here)
    return Function(
        expression,
        name="f",
        vertical_asymptotes=[],
        horizontal_asymptotes=[],
        point_discontinuities=[],
        is_periodic=False,
    )


def _make_piecewise(pieces: Tuple[Tuple[str, Optional[float], Optional[float]], ...]) -> PiecewiseFunction:
    return PiecewiseFunction(
        [{"expression": expression, "left": left, "right": right} for expression, left, right in pieces],
        name="p",
        vertical_asymptotes=[],
        horizontal_asymptotes=[],
        point_discontinuities=[],
    )


def _functions() -> Dict[str, Any]:
    return {
        expression: _samples(ExpressionValidator.parse_function_string, expression) for expression in FUNCTION_CORPUS
    }


def _parametric() -> Dict[str, Any]:
    return {
        expression: _samples(ExpressionValidator.parse_parametric_expression, expression)
        for expression in PARAMETRIC_CORPUS
    }


def _errors() -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}
    for expression in ERROR_CORPUS:
        entry: Dict[str, str] = {}
        for kind, parse in (
            ("function", ExpressionValidator.parse_function_string),
            ("parametric", ExpressionValidator.parse_parametric_expression),
        ):
            try:
                parse(expression)
                entry[kind] = "parsed"
            except Exception as error:
                entry[kind] = _message(error)
        results[expression] = entry
    return results


def _function_translations() -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for expression in TRANSLATE_CORPUS:
        results[expression] = {
            "stored": _make_function(expression).function_string,
            "translated": _translated(lambda: _make_function(expression), lambda f: f.function_string),
        }
    return results


def _piecewise_translations() -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for pieces in PIECEWISE_CORPUS:
        key = " | ".join(f"{expression} on [{left}, {right})" for expression, left, right in pieces)
        results[key] = _translated(
            lambda: _make_piecewise(pieces),
            lambda p: [[interval.expression, interval.left, interval.right] for interval in p.intervals],
        )
    return results


def collect() -> Dict[str, Any]:
    """Every golden section, computed from the current code."""
    return {
        "functions": _functions(),
        "parametric": _parametric(),
        "errors": _errors(),
        "function_translate": _function_translations(),
        "piecewise_translate": _piecewise_translations(),
    }


class TestCompiledExpressionGolden(unittest.TestCase):
    """Parsed values, parse errors and translated text match the recorded golden."""

    golden: Dict[str, Any] = {}

    @classmethod
    def setUpClass(cls) -> None:
        with open(GOLDEN_PATH, encoding="utf-8") as golden_file:
            cls.golden = json.load(golden_file)

    def _assert_section(self, name: str, actual: Dict[str, Any]) -> None:
        expected = self.golden[name]
        self.assertEqual(sorted(actual), sorted(expected), f"{name}: keys differ")
        mismatches = {key: (actual[key], expected[key]) for key in expected if actual[key] != expected[key]}
        self.assertEqual(mismatches, {}, f"{name}: (actual, expected) per key")

    def test_function_values(self) -> None:
        self._assert_section("functions", _functions())

    def test_parametric_values(self) -> None:
        self._assert_section("parametric", _parametric())

    def test_parse_errors(self) -> None:
        self._assert_section("errors", _errors())

    def test_function_translate(self) -> None:
        self._assert_section("function_translate", _function_translations())

    def test_piecewise_translate(self) -> None:
        self._assert_section("piecewise_translate", _piecewise_translations())

    def test_known_counts(self) -> None:
        # Sanity checks on the golden itself, so an empty or truncated file cannot pass
        self.assertEqual(len(self.golden["functions"]), len(FUNCTION_CORPUS))
        self.assertEqual(len(self.golden["parametric"]), len(PARAMETRIC_CORPUS))
        self.assertEqual(self.golden["functions"]["x^2"], ["6.25", "1.0", "0.0", "0.25", "1.0", "9.0"])
        self.assertEqual(self.golden["function_translate"]["x^2"]["translated"]["2,0"], "(x - 2)^2")


if __name__ == "__main__":
    with open(GOLDEN_PATH, "w", encoding="utf-8") as output:
        json.dump(collect(), output, indent=1, ensure_ascii=False, sort_keys=True)
        output.write("\n")
    print(f"Wrote {GOLDEN_PATH}", file=sys.stderr)
