"""Expression function tables checked in the real runtime (Brython and math.js 14.5.2).

The server goldens (server_tests/test_expression_function_tables.py) record the tables
under CPython; these checks need the browser: the identity of the MathUtils functions in
the namespaces, Python against math.js values, and which names math.js knows.
"""

from __future__ import annotations

import unittest
from typing import Dict, List, Sequence

from browser import window

from expression_validator import ExpressionValidator
from utils.math_utils import MathUtils

X_NAMESPACE_KEYS: List[str] = [
    "Infinity", "abs", "acos", "acosh", "acot", "acsc", "arithmetic_sum", "arrangements", "asec", "asin",
    "asinh", "atan", "atanh", "bin", "ceil", "combinations", "cos", "cosh", "cot", "csc", "derive", "det",
    "divisors", "e", "exp", "expand", "factor", "factorial", "floor", "gcd", "geometric_sum",
    "geometric_sum_infinite", "inf", "integrate", "is_prime", "lcm", "limit", "log", "log10", "log2", "max",
    "mean", "median", "min", "mod_inverse", "mod_pow", "mode", "next_prime", "p_series_test", "permutations",
    "pi", "pow", "prev_prime", "prime_factors", "product", "randint", "random", "ratio_test", "root_test",
    "round", "sec", "simplify", "sin", "sinh", "solve", "sqrt", "stdev", "sum", "summation", "tan", "tanh",
    "totient", "trunc", "variance",
]  # fmt: skip
T_NAMESPACE_KEYS: List[str] = [
    "Infinity", "abs", "acos", "acosh", "acot", "acsc", "asec", "asin", "asinh", "atan", "atanh", "ceil",
    "cos", "cosh", "cot", "csc", "e", "exp", "factorial", "floor", "inf", "log", "log10", "log2", "max", "min",
    "pi", "pow", "round", "sec", "sin", "sinh", "sqrt", "t", "tan", "tanh", "trunc",
]  # fmt: skip
# The names math.js evaluates the whitelisted functions under (stdev is std, trunc is fix)
MATHJS_NAMES: List[str] = [
    "abs", "acos", "acosh", "acot", "acsc", "asec", "asin", "asinh", "atan", "atanh", "bin", "ceil",
    "combinations", "cos", "cosh", "cot", "csc", "det", "exp", "factorial", "fix", "floor", "gcd", "lcm",
    "log", "log10", "log2", "max", "mean", "median", "min", "mode", "permutations", "pow", "random", "round",
    "sec", "simplify", "sin", "sinh", "sqrt", "std", "sum", "tan", "tanh", "variance",
]  # fmt: skip
# Whitelisted names math.js lacks: renamed (arrangements, stdev, trunc) or handled elsewhere
MATHJS_ABSENT: List[str] = [
    "arrangements",
    "derive",
    "expand",
    "factor",
    "integrate",
    "limit",
    "solve",
    "stdev",
    "trunc",
]
# The functions of one argument allowed in parametric curves, with points in their domains
CURVE_UNARY_SAMPLES: Dict[str, Sequence[float]] = {
    "sin": (0.3, -1.2),
    "cos": (0.3, -1.2),
    "tan": (0.3, -1.2),
    "sqrt": (2.5, 0.04),
    "log10": (2.5, 300.0),
    "log2": (2.5, 0.3),
    "factorial": (0, 5),
    "asin": (0.4, -0.9),
    "acos": (0.4, -0.9),
    "atan": (0.4, -3.0),
    "sinh": (0.4, -2.0),
    "cosh": (0.4, -2.0),
    "tanh": (0.4, -2.0),
    "sec": (0.7, -2.0),
    "csc": (0.7, -2.0),
    "cot": (0.7, -2.0),
    "asec": (1.5, -2.0),
    "acsc": (1.5, -2.0),
    "acot": (0.7, -2.0),
    "asinh": (0.7, -2.0),
    "acosh": (1.5, 3.0),
    "atanh": (0.5, -0.3),
    "exp": (0.7, -2.0),
    "abs": (0.7, -2.0),
    "ceil": (2.7, -2.7),
    "floor": (2.7, -2.7),
    "trunc": (2.7, -2.7),
}


class TestExpressionFunctions(unittest.TestCase):
    def test_namespace_keys(self) -> None:
        self.assertEqual(sorted(ExpressionValidator._build_functions()), X_NAMESPACE_KEYS)
        self.assertEqual(sorted(ExpressionValidator._get_variables_and_functions_parametric(0)), T_NAMESPACE_KEYS)

    def test_namespaces_hold_the_math_utils_functions(self) -> None:
        namespace = ExpressionValidator._build_functions()
        self.assertIs(namespace["derive"], MathUtils.derivative)
        self.assertIs(namespace["integrate"], MathUtils.integral)
        self.assertIs(namespace["sqrt"], MathUtils.sqrt)
        self.assertIs(ExpressionValidator._get_variables_and_functions_parametric(0)["round"], MathUtils.round)

    def test_python_and_mathjs_agree_on_curve_functions(self) -> None:
        for name in sorted(CURVE_UNARY_SAMPLES):
            for point in CURVE_UNARY_SAMPLES[name]:
                with self.subTest(name=name, point=point):
                    expression = f"{name}(x)"
                    python_value = ExpressionValidator.parse_function_string(expression)(point)
                    mathjs_value = MathUtils.evaluate(expression, {"x": point})
                    self.assertIsInstance(mathjs_value, (int, float), f"{expression} at {point}: {mathjs_value}")
                    self.assertAlmostEqual(python_value, float(mathjs_value), delta=1e-9 * max(1.0, abs(python_value)))

    def test_mathjs_knows_the_mapped_names(self) -> None:
        for name in MATHJS_NAMES:
            with self.subTest(name=name):
                self.assertTrue(hasattr(window.math, name), f"math.{name} is missing")
                self.assertTrue(callable(getattr(window.math, name)), f"math.{name} is not a function")

    def test_mathjs_lacks_the_other_names(self) -> None:
        for name in MATHJS_ABSENT:
            with self.subTest(name=name):
                self.assertFalse(hasattr(window.math, name), f"math.{name} exists; it may not need special handling")
