"""compile_expression checked in the real runtime (Brython and math.js).

The server tests (server_tests/test_compiled_expression.py) cover the API under CPython.
These need the browser: Brython's ast, compile and eval with named and Greek variables
(Brython does not NFKC-normalise identifiers), its keyword module, its re with Unicode
word characters (substitute_variable), and the values of the functions MathUtils computes
with math.js (the server stub fakes them).
"""

from __future__ import annotations

import math
import unittest

import compiled_expression as ce
from compiled_expression import FUNCTIONS_ALL, FUNCTIONS_CURVE, compile_expression, substitute_variable
from drawables.function import Function
from drawables.piecewise_function import PiecewiseFunction
from expression_validator import ExpressionValidator


class TestCompiledExpression(unittest.TestCase):
    def test_named_variables_evaluate(self) -> None:
        self.assertAlmostEqual(compile_expression("cos(2θ)", ("θ",), FUNCTIONS_CURVE).bind()(math.pi / 2), -1.0)
        self.assertAlmostEqual(compile_expression("2θ + sinθ", ("θ",), FUNCTIONS_CURVE).bind()(1.0), 2 + math.sin(1))
        self.assertAlmostEqual(compile_expression("1/k!", ("k",)).bind()(4), 1 / 24)
        self.assertAlmostEqual(compile_expression("(-1)^k", ("k",)).bind()(3), -1.0)
        self.assertEqual(compile_expression("x^2 + y^2 - 1", ("x", "y")).bind()(1.0, 2.0), 4.0)
        self.assertEqual(compile_expression("x - y", ("y", "x")).bind()(1.0, 5.0), 4.0)
        self.assertEqual(compile_expression("x + 2y + 3z", ("x", "y", "z")).bind()(1.0, 2.0, 3.0), 14.0)
        self.assertEqual(compile_expression("2 + 3", ()).bind()(), 5.0)

    def test_greek_variant_variable_is_normalised(self) -> None:
        compiled = compile_expression("ϑ + 1", ("ϑ",), FUNCTIONS_CURVE)
        self.assertEqual(compiled.variables, ("θ",))
        self.assertEqual(compiled.bind()(2.0), 3.0)
        self.assertEqual(compile_expression("µ*x", ("µ", "x")).bind()(2.0, 3.0), 6.0)

    def test_free_names(self) -> None:
        self.assertEqual(compile_expression("rθ", ("θ",), FUNCTIONS_CURVE).free_names, ("r",))
        self.assertEqual(compile_expression("θ1 + Δx + x").free_names, ("Δx", "θ1"))
        self.assertEqual(compile_expression("kπ + αk", ("k",)).free_names, ("α",))  # αk is α*k
        self.assertEqual(compile_expression("xy", ("x", "y")).free_names, ("xy",))
        self.assertEqual(compile_expression("x + i + sin").free_names, ())
        self.assertEqual(compile_expression("derive(t)", ("t",), FUNCTIONS_CURVE).free_names, ())

    def test_reserved_and_keyword_names_are_rejected(self) -> None:
        for name in ("in", "is", "or", "if", "as", "e", "ln", "pi", "i", "_a", "1x"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    compile_expression("1", (name,))
        with self.assertRaises(ValueError):
            compile_expression("θ", ("θ", "ϑ"))

    def test_binds_share_no_mutable_state(self) -> None:
        first = compile_expression("x + 1").bind()
        second = compile_expression("x + 1").bind()
        self.assertEqual(first(1.0), 2.0)
        self.assertEqual(second(5.0), 6.0)
        self.assertEqual(first(1.0), 2.0)
        for functions in (FUNCTIONS_ALL, FUNCTIONS_CURVE):
            self.assertNotIn("x", ce.base_namespace(functions))
            self.assertNotIn("__builtins__", ce.base_namespace(functions))
            self.assertNotIn("x", ce._lambda_globals(functions))

    def test_bind_compiles_one_lambda(self) -> None:
        # Brython compiles the lambda once; a comment in the text must not swallow its end
        self.assertIs(compile_expression("x + 1").bind(), compile_expression("x + 1").bind())
        self.assertEqual(compile_expression("x + 1 # shift").bind()(1.0), 2.0)
        self.assertEqual(compile_expression("2*float", ("float",)).bind()(3.0), 6.0)
        self.assertEqual(compile_expression("x - t", ("t", "x")).bind()(1.0, 10.0), 9.0)
        self.assertIsInstance(compile_expression("x + 1").bind()(2), float)

    def test_the_lambda_comes_from_the_validated_tree(self) -> None:
        # Brython compiles the hand-built lambda; it adds no readable name and no brackets
        with self.assertRaises(NameError):
            compile_expression("_float + x").bind()(1.0)
        self.assertEqual(compile_expression("x + 1 # )) , open('f'").bind()(1.0), 2.0)
        self.assertEqual(compile_expression("(" * 50 + "x" + ")" * 50).bind()(2.0), 2.0)

    def test_repeated_text_reuses_the_parse(self) -> None:
        self.assertIs(compile_expression("x^3 + 7")._parsed, compile_expression("x^3 + 7")._parsed)
        for _ in range(2):
            with self.assertRaises(ValueError):
                compile_expression("__import__('os')")
        self.assertNotIn("__import__('os')", ce._CACHE)

    def test_messages_are_the_validators(self) -> None:
        with self.assertRaises(ValueError) as context:
            compile_expression("a.x")
        self.assertEqual(str(context.exception), "Error validating expression 'a.x': Disallowed node type: Attribute")
        with self.assertRaises(ValueError) as context:
            compile_expression("x=1")
        self.assertTrue(str(context.exception).startswith("Syntax error in expression 'x=1'"))

    def test_evaluation_errors_are_raised_per_call(self) -> None:
        with self.assertRaises(NameError):
            compile_expression("a*x").bind()(1.0)
        with self.assertRaises(ZeroDivisionError):
            compile_expression("1/x").bind()(0.0)
        with self.assertRaises(NameError):
            compile_expression("derive(t, t)", ("t",), FUNCTIONS_CURVE).bind()(1.0)

    def test_math_utils_functions_have_real_values(self) -> None:
        # The server stub fakes window.math; here math.js computes them
        cases = {
            "sqrt(x)": (2.0, math.sqrt(2)),
            "pow(x, 3)": (2.0, 8.0),
            "round(x)": (2.5, 3.0),
            "det([[x, 1], [0, 1]])": (3.0, 3.0),
            "mean([x, 1])": (3.0, 2.0),
        }
        for text, (value, expected) in cases.items():
            with self.subTest(text=text):
                self.assertAlmostEqual(compile_expression(text).bind()(value), expected)
                self.assertAlmostEqual(ExpressionValidator.parse_function_string(text)(value), expected)

    def test_parametric_wrapper_uses_the_curve_namespace(self) -> None:
        self.assertAlmostEqual(ExpressionValidator.parse_parametric_expression("t*cos(t)")(math.pi), -math.pi)
        with self.assertRaises(NameError):
            ExpressionValidator.parse_parametric_expression("mean([t])")(1.0)

    def test_substitute_variable_with_unicode_names(self) -> None:
        # Brython's re must treat θ and Δ as word characters
        self.assertEqual(substitute_variable("2*θ + θ1 + sin(θ)", "θ", "(θ - 1)"), "2*(θ - 1) + θ1 + sin((θ - 1))")
        self.assertEqual(
            substitute_variable("Δx + x + max(x, 1e5)", "x", "(x - 2)"), "Δx + (x - 2) + max((x - 2), 1e5)"
        )
        self.assertEqual(substitute_variable("cos(2*θ)", "ϑ", "(θ + 1)"), "cos(2*(θ + 1))")
        shifted = compile_expression(substitute_variable("θ^2", "θ", "(θ - 1)"), ("θ",), FUNCTIONS_CURVE).bind()
        self.assertEqual(shifted(3.0), 4.0)

    def test_translate_rewrites_whole_tokens(self) -> None:
        function = Function("max(x, 2) + exp(x)", name="f")
        function.translate(-3, 1)
        self.assertEqual(function.function_string, "(max((x - -3), 2) + exp((x - -3))) + 1")
        self.assertAlmostEqual(function.function(-3.0), 2 + 1 + 1)
        piecewise = PiecewiseFunction(
            [{"expression": "x^2", "left": None, "right": 0}, {"expression": "max(x, 1)", "left": 0, "right": None}],
            name="p",
        )
        piecewise.translate(2, 0)
        self.assertEqual([interval.expression for interval in piecewise.intervals], ["(x - 2)^2", "max((x - 2), 1)"])
