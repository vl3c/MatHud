"""Tests for compile_expression (static/client/compiled_expression.py).

Named variables, free names, variable checks, binding and the parse cache. The browser
counterpart (client_tests/test_compiled_expression.py) runs the Unicode and binding cases
through Brython.
"""

from __future__ import annotations

import math
import unittest
from typing import Sequence, Tuple

from server_tests import client_renderer  # noqa: F401  (browser stub, so MathUtils resolves)
import compiled_expression as ce
from compiled_expression import FUNCTIONS_ALL, FUNCTIONS_CURVE, compile_expression
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


if __name__ == "__main__":
    unittest.main()
