"""Expression function rows checked in the real runtime (Brython and math.js 14.5.2).

The server tests (server_tests/test_expression_functions.py) check the registry under
CPython; these checks need the browser: the identity of the MathUtils functions in the
namespaces, Python against math.js values, and which names math.js knows.
"""

from __future__ import annotations

import math
import unittest
from typing import Dict, List, Sequence, Tuple

from browser import window

import expression_functions as ef
from expression_validator import ExpressionValidator
from utils.math_utils import MathUtils

# Rows math.js lacks that are not python-only: other tools compute them (pinned, so a new
# one is a decision)
MATHJS_ABSENT_ROWS: List[str] = ["derive", "expand", "factor", "integrate", "limit", "solve"]
# Points in the domain of every curve-safe function of exactly one argument (a new row needs one)
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
    "cbrt": (2.5, -3.3),
    "sign": (0.7, -2.0),
    "gamma": (0.7, 4.3),
    "erf": (0.3, -1.2),
}
# Argument pairs of the two-argument calculator rows, compared between Python and math.js
BINARY_SAMPLES: Dict[str, Sequence[Tuple[float, float]]] = {
    "mod": ((17, 5), (-7, 3), (7, -3), (5.5, 2), (7, 0), (-7.5, 2)),
    "atan2": ((1, -1), (-1, -1), (0, 0), (1, 1), (-1, 1)),
    "nthroot": ((-8, 3), (32, 5), (16, -2), (2.5, 2)),
}

# Aliases are matched as whole tokens; the same cases as server_tests/test_expression_functions.py, run through Brython's re
ALIAS_CASES = {
    "2ln(x) + sine(x)": "2log(x) + sin(x)",
    "1e3ln(x)": "1e3log(x)",
    "2.5e-3ln(x)": "2.5e-3log(x)",
    "θ2ln(x)": "θ2log(x)",
    "ln (x)": "log (x)",
    "hyperbolic sine(x) + 2square root(x)": "sinh(x) + 2sqrt(x)",
    "arcsine(x)*cosine(x)": "asin(x)*cos(x)",
    "std(x)+var(y)": "stdev(x)+variance(y)",
    "kiln": "kiln",
    "ln2": "ln2",
    "x2ln(x)": "x2ln(x)",
    "a1sine(x)": "a1sine(x)",
    "artifact(3)": "artifact(3)",
    "myvar(2)": "myvar(2)",
    "sine (x)": "sine (x)",
    "var": "var",
    "sgn(x)": "sign(x)",
    "nthRoot(8, 3)": "nthroot(8, 3)",
    "xsgn(x)": "xsgn(x)",
}


class TestExpressionFunctions(unittest.TestCase):
    def test_namespace_keys(self) -> None:
        constants = set(ef.CONSTANTS)
        self.assertEqual(sorted(ExpressionValidator._build_functions()), sorted(set(ef.BY_NAME) | constants))
        self.assertEqual(
            sorted(ExpressionValidator._get_variables_and_functions_parametric(0)),
            sorted(ef.CURVE_NAMES | constants | {"t"}),
        )

    def test_namespaces_hold_the_math_utils_functions(self) -> None:
        namespace = ExpressionValidator._build_functions()
        self.assertIs(namespace["derive"], MathUtils.derivative)
        self.assertIs(namespace["integrate"], MathUtils.integral)
        self.assertIs(namespace["sqrt"], MathUtils.sqrt)
        self.assertIs(ExpressionValidator._get_variables_and_functions_parametric(0)["round"], MathUtils.round)

    def test_every_unary_curve_function_has_sample_points(self) -> None:
        unary = sorted(row.name for row in ef.FUNCTIONS if row.curve_safe and row.arity == (1, 1))
        self.assertEqual(unary, sorted(CURVE_UNARY_SAMPLES))

    def test_python_and_mathjs_agree_on_curve_functions(self) -> None:
        for name in sorted(CURVE_UNARY_SAMPLES):
            for point in CURVE_UNARY_SAMPLES[name]:
                with self.subTest(name=name, point=point):
                    expression = f"{name}(x)"
                    python_value = ExpressionValidator.parse_function_string(expression)(point)
                    mathjs_value = MathUtils.evaluate(expression, {"x": point})
                    self.assertIsInstance(mathjs_value, (int, float), f"{expression} at {point}: {mathjs_value}")
                    self.assertAlmostEqual(python_value, float(mathjs_value), delta=1e-9 * max(1.0, abs(python_value)))

    def test_python_and_mathjs_agree_on_binary_functions(self) -> None:
        for name in sorted(BINARY_SAMPLES):
            for a, b in BINARY_SAMPLES[name]:
                with self.subTest(name=name, a=a, b=b):
                    expression = f"{name}(x, y)"
                    python_value = ExpressionValidator.evaluate_expression(
                        ExpressionValidator.fix_math_expression(expression, True), a, {"y": b}
                    )
                    mathjs_value = MathUtils.evaluate(expression, {"x": a, "y": b})
                    self.assertIsInstance(mathjs_value, (int, float), f"{expression} at {a}, {b}: {mathjs_value}")
                    self.assertAlmostEqual(python_value, float(mathjs_value), delta=1e-9 * max(1.0, abs(python_value)))

    def test_brython_float_remainder_is_floored(self) -> None:
        self.assertEqual(-7.5 % 2, 0.5)
        self.assertEqual(7.5 % -2, -0.5)
        self.assertEqual(ExpressionValidator.parse_function_string("mod(x, 2)")(-7.5), 0.5)

    def test_complex_parts_in_brython(self) -> None:
        namespace = ExpressionValidator._build_functions()
        self.assertEqual(namespace["conj"](3 + 4j), 3 - 4j)
        self.assertEqual((namespace["re"](3 + 4j), namespace["im"](3 + 4j)), (3.0, 4.0))
        self.assertEqual((namespace["re"](2.5), namespace["im"](2.5)), (2.5, 0.0))
        self.assertAlmostEqual(namespace["arg"](-1), math.pi)

    def test_mathjs_knows_the_mapped_names(self) -> None:
        names = sorted({row.mathjs for row in ef.FUNCTIONS if row.mathjs is not None})
        self.assertGreater(len(names), 40)
        for name in names:
            with self.subTest(name=name):
                self.assertTrue(hasattr(window.math, name), f"math.{name} is missing")
                self.assertTrue(callable(getattr(window.math, name)), f"math.{name} is not a function")

    def test_mathjs_lacks_the_other_names(self) -> None:
        absent = sorted(row.name for row in ef.FUNCTIONS if row.mathjs is None and not row.python_only)
        self.assertEqual(absent, MATHJS_ABSENT_ROWS)
        for name in absent + sorted(ef.MATHJS_RENAMES):
            with self.subTest(name=name):
                self.assertFalse(hasattr(window.math, name), f"math.{name} exists; it may not need special handling")

    def test_aliases_are_matched_as_whole_tokens(self) -> None:
        for text, expected in ALIAS_CASES.items():
            with self.subTest(text=text):
                self.assertEqual(ef.replace_aliases(text), expected)

    def test_mathjs_renames_reach_every_call_site(self) -> None:
        from utils.symbolic_checks import SymbolicChecks

        self.assertEqual(ef.rename_for_mathjs("trunc(x) + xtrunc(x)"), "fix(x) + xtrunc(x)")
        self.assertAlmostEqual(SymbolicChecks.numeric_value("trunc(2.5) + stdev([1, 3])"), 2 + 2**0.5)
        probe = MathUtils._mathjs_probe_evaluator("arrangements(5, 2) + trunc(x)")
        self.assertIsNotNone(probe)
        self.assertEqual(probe(1.5), 21.0)
