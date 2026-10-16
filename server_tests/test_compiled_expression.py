"""Tests for compile_expression (static/client/compiled_expression.py).

Named variables, free names, variable checks, binding, the parse cache and
substitute_variable (shifting a graph). The browser
counterpart (client_tests/test_compiled_expression.py) runs the Unicode and binding cases
through Brython.
"""

from __future__ import annotations

import math
import random
import re
import unittest
from typing import Sequence, Tuple

from server_tests import client_renderer  # noqa: F401  (browser stub, so MathUtils resolves)
import compiled_expression as ce
from compiled_expression import FUNCTIONS_ALL, FUNCTIONS_CURVE, compile_expression, substitute_variable
from expression_validator import ExpressionValidator


def _free(text: str, variables: Sequence[str] = ("x",), functions: str = FUNCTIONS_ALL) -> Tuple[str, ...]:
    return compile_expression(text, variables, functions).free_names


class TestFreeNames(unittest.TestCase):
    def test_free_names_in_x(self) -> None:
        self.assertEqual(_free("x^2 + 1"), ())
        self.assertEqual(_free("a*x+b"), ("a", "b"))
        self.assertEqual(_free("x + i"), ())  # i is the imaginary unit (j in Python), reserved
        self.assertEqual(_free("sin + x"), ())  # a bare function name is bound
        self.assertEqual(_free("pi*x + e + inf"), ())
        self.assertEqual(_free("θ1 + Δx + x"), ("Δx", "θ1"))

    def test_free_names_in_t_with_curve_functions(self) -> None:
        self.assertEqual(_free("t*cos(t)", ("t",), FUNCTIONS_CURVE), ())
        # A call of a function the curve namespace lacks is not a name read
        self.assertEqual(_free("derive(t)", ("t",), FUNCTIONS_CURVE), ())
        self.assertEqual(_free("x + t", ("t",), FUNCTIONS_CURVE), ("x",))

    def test_free_names_in_theta_and_k(self) -> None:
        self.assertEqual(_free("2θ", ("θ",), FUNCTIONS_CURVE), ())
        self.assertEqual(_free("rθ", ("θ",), FUNCTIONS_CURVE), ("r",))
        self.assertEqual(_free("sinθ", ("θ",), FUNCTIONS_CURVE), ())
        self.assertEqual(_free("kπ", ("k",)), ())
        self.assertEqual(_free("1/k!", ("k",)), ())

    def test_free_names_in_x_and_y(self) -> None:
        self.assertEqual(_free("x^2 + y^2 - 1", ("x", "y")), ())
        # ASCII runs stay one name, as in math.js and nerdamer
        self.assertEqual(_free("xy", ("x", "y")), ("xy",))
        self.assertEqual(_free("ax + y", ("x", "y")), ("ax",))

    def test_variables_are_normalised(self) -> None:
        compiled = compile_expression("ϑ + 1", ("ϑ",), FUNCTIONS_CURVE)
        self.assertEqual(compiled.variables, ("θ",))
        self.assertEqual(compiled.free_names, ())
        self.assertEqual(compiled.bind()(2.0), 3.0)

    def test_attributes(self) -> None:
        compiled = compile_expression("2x^2", ("x",))
        self.assertEqual(compiled.source, "2x^2")
        self.assertEqual(compiled.python_text, ExpressionValidator.fix_math_expression("2x^2", True))
        self.assertEqual(compiled.functions, FUNCTIONS_ALL)


class TestVariableChecks(unittest.TestCase):
    def test_reserved_and_malformed_names_are_rejected(self) -> None:
        for variables in (("e",), ("ln",), ("pi",), ("sin",), ("i",), ("j",), ("_a",), ("1x",), ("",), ("a b",)):
            with self.subTest(variables=variables):
                with self.assertRaises(ValueError):
                    compile_expression("1", variables)

    def test_keywords_are_rejected(self) -> None:
        for name in ("in", "is", "or", "if", "as"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "keyword"):
                    compile_expression("1", (name,))

    def test_duplicates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compile_expression("x", ("x", "x"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compile_expression("θ", ("θ", "ϑ"))

    def test_a_string_and_an_unknown_function_set_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compile_expression("x*y", "xy")
        with self.assertRaisesRegex(ValueError, "Unknown function set"):
            compile_expression("x", ("x",), "polar")
        with self.assertRaises(ValueError):
            compile_expression("x", (["x"],))  # type: ignore[list-item]

    def test_call_only_aliases_are_usable_names(self) -> None:
        self.assertEqual(compile_expression("2*var", ("var",)).bind()(3.0), 6.0)


class TestBind(unittest.TestCase):
    def test_arity_zero_to_three(self) -> None:
        self.assertEqual(compile_expression("2 + 3", ()).bind()(), 5.0)
        self.assertEqual(compile_expression("x^2", ("x",)).bind()(3.0), 9.0)
        self.assertEqual(compile_expression("x^2 + y^2 - 1", ("x", "y")).bind()(1.0, 2.0), 4.0)
        self.assertEqual(compile_expression("x + 2y + 3z", ("x", "y", "z")).bind()(1.0, 2.0, 3.0), 14.0)
        # The order of variables is the order of the arguments
        self.assertEqual(compile_expression("x - y", ("y", "x")).bind()(1.0, 5.0), 4.0)

    def test_greek_variables(self) -> None:
        self.assertAlmostEqual(compile_expression("cos(2θ)", ("θ",), FUNCTIONS_CURVE).bind()(math.pi / 2), -1.0)
        self.assertAlmostEqual(compile_expression("1/k!", ("k",)).bind()(4), 1 / 24)

    def test_results_are_floats(self) -> None:
        value = compile_expression("x + 1", ("x",)).bind()(2)
        self.assertIsInstance(value, float)

    def test_binds_share_no_mutable_state(self) -> None:
        first = compile_expression("x + 1").bind()
        second = compile_expression("x + 1").bind()
        other = compile_expression("2x").bind()
        self.assertEqual(first(1.0), 2.0)
        self.assertEqual(second(5.0), 6.0)
        self.assertEqual(other(7.0), 14.0)
        self.assertEqual(first(1.0), 2.0)
        for functions in (FUNCTIONS_ALL, FUNCTIONS_CURVE):
            self.assertNotIn("x", ce.base_namespace(functions))
            self.assertNotIn("__builtins__", ce.base_namespace(functions))
            self.assertNotIn("x", ce._lambda_globals(functions))

    def test_one_lambda_per_text_variables_and_functions(self) -> None:
        self.assertIs(compile_expression("x + 1").bind(), compile_expression("x + 1").bind())
        self.assertIsNot(
            compile_expression("x + 1").bind(), compile_expression("x + 1", ("x",), FUNCTIONS_CURVE).bind()
        )
        self.assertEqual(compile_expression("t + x", ("x", "t")).bind()(1.0, 10.0), 11.0)
        self.assertEqual(compile_expression("t + x", ("t", "x")).bind()(1.0, 10.0), 11.0)
        self.assertEqual(compile_expression("x - t", ("t", "x")).bind()(1.0, 10.0), 9.0)

    def test_text_with_a_comment_and_a_variable_named_float(self) -> None:
        # The lambda wraps the validated text; a trailing comment must not swallow it
        self.assertEqual(compile_expression("x + 1 # shift").bind()(1.0), 2.0)
        # The lambda converts with _float, so a variable may be called float
        self.assertEqual(compile_expression("2*float", ("float",)).bind()(3.0), 6.0)

    def test_evaluation_errors_are_raised_by_the_callable(self) -> None:
        with self.assertRaises(NameError):
            compile_expression("a*x").bind()(1.0)
        with self.assertRaises(ZeroDivisionError):
            compile_expression("1/x").bind()(0.0)
        with self.assertRaises(TypeError):  # factorial of a float, as before
            compile_expression("x!").bind()(3.0)

    def test_curve_namespace_lacks_non_curve_functions(self) -> None:
        curve = ce.base_namespace(FUNCTIONS_CURVE)
        self.assertNotIn("derive", curve)
        self.assertIn("cos", curve)
        self.assertIn("derive", ce.base_namespace(FUNCTIONS_ALL))
        with self.assertRaises(NameError):
            compile_expression("derive(t, t)", ("t",), FUNCTIONS_CURVE).bind()(1.0)

    def test_base_namespace_is_built_once(self) -> None:
        self.assertIs(ce.base_namespace(FUNCTIONS_ALL), ce.base_namespace(FUNCTIONS_ALL))
        with self.assertRaises(ValueError):
            ce.base_namespace("polar")


class TestParseErrorsAndCache(unittest.TestCase):
    def test_messages_are_the_validators(self) -> None:
        cases = {
            "a.x": "Error validating expression 'a.x': Disallowed node type: Attribute",
            "ℝ": "Unsupported symbol 'ℝ' in expression 'ℝ'",
            "lambda: 1": "Error validating expression 'lambda: 1': Disallowed node type: Lambda",
        }
        for text, message in cases.items():
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as context:
                    compile_expression(text)
                self.assertEqual(str(context.exception), message)
        with self.assertRaisesRegex(ValueError, "^Syntax error in expression 'x=1'"):
            compile_expression("x=1")

    def test_repeated_text_reuses_the_parse(self) -> None:
        first = compile_expression("x^3 - x")
        second = compile_expression("x^3 - x", ("x",), FUNCTIONS_CURVE)
        self.assertIs(first._parsed, second._parsed)

    def test_failures_are_not_cached(self) -> None:
        for _ in range(2):
            with self.assertRaises(ValueError):
                compile_expression("__import__('os')")
        self.assertNotIn("__import__('os')", ce._CACHE)

    def test_cache_is_cleared_when_full(self) -> None:
        for index in range(ce._CACHE_LIMIT + 5):
            compile_expression(f"x + {index}")
        self.assertLessEqual(len(ce._CACHE), ce._CACHE_LIMIT)
        self.assertEqual(compile_expression("x + 3").bind()(1.0), 4.0)


def _old_translate_x(text: str, x_offset: float) -> str:
    """The regex Function.translate and PiecewiseFunction.translate used before (the oracle)."""
    protected = sorted(ExpressionValidator.ALLOWED_FUNCTIONS, key=len, reverse=True)
    pattern = rf"\b(x)\b|({'|'.join(map(re.escape, protected))})"

    def replace(match: re.Match[str]) -> str:
        if match.group(1):
            return f"(x - {x_offset})"
        return match.group(2) or match.group(0)

    return re.sub(pattern, replace, text)


# Inputs as typed; the stored text is their fix_math_expression form
_TRANSLATE_CORPUS = [
    "x", "x^2", "sin(x)", "2x", "3x^2+2x-1", "x*exp(x)", "max(x,2)", "exp(x)", "1e-5x", "x2+xy",
    "|x|", "x!", "(x+1)!", "sqrt(x)", "√x", "x²", "sin²x", "2πx", "θx", "Δx+x", "x°",
    "sin(30°)+x", "ln(x)", "log2(x)", "log10(x)+x", "2.5e3x", "x/(x-1)", "e^x", "e^-x", "xe^x",
    "x e", "x(x+1)", "0x10", "x_1+x", "1/x", "x^x", "sinx", "cosx+x", "tan(x)", "x**2", "-x",
    "x-(-3)", "degrees(x)", "2 x",
]  # fmt: skip
_OFFSETS = (2, -3, -0.5, 1e-7, 1e20)


class TestSubstituteVariable(unittest.TestCase):
    def test_whole_tokens_only(self) -> None:
        shift = "(x - 2)"
        self.assertEqual(substitute_variable("x^2 + max(x, 1)", "x", shift), "(x - 2)^2 + max((x - 2), 1)")
        self.assertEqual(substitute_variable("exp(x) + x2 + xy + Δx", "x", shift), "exp((x - 2)) + x2 + xy + Δx")
        self.assertEqual(substitute_variable("1e5*x + 2.5E-3", "x", shift), "1e5*(x - 2) + 2.5E-3")
        self.assertEqual(substitute_variable("x_1 + 0x10", "x", shift), "x_1 + 0x10")
        self.assertEqual(substitute_variable("x(x+1)", "x", shift), "x((x - 2)+1)")  # a call is not the variable

    def test_greek_and_other_variables(self) -> None:
        self.assertEqual(substitute_variable("2*θ + θ1 + sin(θ)", "θ", "(θ - 1)"), "2*(θ - 1) + θ1 + sin((θ - 1))")
        self.assertEqual(substitute_variable("cos(2*θ)", "ϑ", "(θ + 1)"), "cos(2*(θ + 1))")  # ϑ is θ
        self.assertEqual(substitute_variable("1/k! + k2 + kπ", "k", "(k + 1)"), "1/(k + 1)! + k2 + kπ")
        self.assertEqual(substitute_variable("Δx*x", "Δx", "h"), "h*x")

    def test_matches_the_old_regex_on_stored_text(self) -> None:
        for text in _TRANSLATE_CORPUS:
            stored = ExpressionValidator.fix_math_expression(text)
            for offset in _OFFSETS:
                with self.subTest(text=text, offset=offset):
                    expected = _old_translate_x(stored, offset)
                    actual = substitute_variable(stored, "x", f"(x - {offset})")
                    if text == "x(x+1)":  # never parses, so never stored
                        self.assertNotEqual(actual, expected)
                    else:
                        self.assertEqual(actual, expected)

    def test_matches_the_old_regex_on_random_text_that_parses(self) -> None:
        rng = random.Random(1)
        alphabet = list("x2e.()+*-/asinθΔ_ 9^|!") + ["x", "x", "exp", "max", "1e5"]
        compared = 0
        for _ in range(4000):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 8)))
            try:
                ExpressionValidator.parse_function_string(text)
                stored = ExpressionValidator.fix_math_expression(text)
            except Exception:
                continue
            compared += 1
            self.assertEqual(substitute_variable(stored, "x", "(x - 2)"), _old_translate_x(stored, 2), text)
        self.assertGreater(compared, 100)


if __name__ == "__main__":
    unittest.main()
