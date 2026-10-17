"""Server tests for the pure-Python CAS helpers: utils.cas_text and utils.cas_numeric."""

from __future__ import annotations

import cmath
import math
import os
import sys
import unittest
from typing import List, Optional, Sequence

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "client"))

from utils.cas_numeric import (  # noqa: E402
    adaptive_integral,
    distinct_values,
    format_complex,
    format_number,
    polynomial_roots,
    snap,
)
from utils.cas_text import (  # noqa: E402
    find_closing_parenthesis,
    has_inequality,
    glued_call,
    has_word_names,
    insert_calculator_calls,
    insert_implicit_products,
    insert_letter_call_products,
    is_unevaluated,
    parse_interval,
    rewrite_for_nerdamer,
    split_constant_products,
    split_relation,
    split_top_level_commas,
    split_top_level_terms,
    unknown_function_names,
)


class TestInequalityText(unittest.TestCase):
    FUNCTIONS = ("sin", "exp", "abs", "log", "sqrt")

    def test_letter_call_products(self) -> None:
        cases = {
            "x(x+2)": "x*(x+2)",
            "2a(b+1)": "2a*(b+1)",
            "(x+1)x(x-1)": "(x+1)x*(x-1)",
            "θ(θ+1)": "θ*(θ+1)",
            "e(x+1)": "e*(x+1)",
            "exp(x) + sin (x)": "exp(x) + sin (x)",
            "x2(3)": "x2(3)",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(insert_letter_call_products(text, self.FUNCTIONS), expected)

    def test_implicit_products(self) -> None:
        cases = {
            "(x-1)(x+2)": "(x-1)*(x+2)",
            "(x-2)^2 (x+1)": "(x-2)^2*(x+1)",
            "abs(x-1)(x+1)": "abs(x-1)*(x+1)",
            "(x+1)x": "(x+1)*x",
            "(x+1)2": "(x+1)*2",
            "(x+1) e^x": "(x+1)*e^x",
            "1e5 x": "1e5*x",
            "2.5 (x+1)": "2.5*(x+1)",
            # unchanged: a number directly before a name, exponents, names with digits, operators
            "2x": "2x",
            "2(x+1)": "2(x+1)",
            "1e-5": "1e-5",
            "x2 + 1": "x2 + 1",
            "(x+1) + 2": "(x+1) + 2",
            "sin(x)^2": "sin(x)^2",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(insert_implicit_products(text), expected)

    def test_split_relation(self) -> None:
        self.assertEqual(split_relation("x^2 - 5x + 6 < 0"), (["x^2 - 5x + 6", "0"], ["<"]))
        self.assertEqual(split_relation("1 < 2x+1 <= 5"), (["1", "2x+1", "5"], ["<", "<="]))
        self.assertEqual(split_relation("x => 2"), (["x", "2"], [">="]))
        self.assertEqual(split_relation("x =< 2"), (["x", "2"], ["<="]))
        self.assertEqual(split_relation("5 >= x > 1"), (["5", "x", "1"], [">=", ">"]))

    def test_split_relation_refusals(self) -> None:
        cases = {
            "x != 2": "exclude its roots",
            "x == 2": "single '='",
            "x > 1 and x < 5": "on its own",
            "x < 1 or x > 3": "on its own",
            "1 < x > 0": "point the same way",
            "1 < x < 2 < 3": "at most two signs",
            "x = 2 < 3": "'='",
            "< 2": "missing",
            "x + 2": "needs one of",
        }
        for text, message in cases.items():
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as caught:
                    split_relation(text)
                self.assertIn(message, str(caught.exception))

    def test_split_top_level_terms(self) -> None:
        cases = {
            "x^3 - 2x^2 + x": [(1, "x^3"), (-1, "2x^2"), (1, "x")],
            "exp(2x) - 3exp(x) + 2": [(1, "exp(2x)"), (-1, "3exp(x)"), (1, "2")],
            "-x": [(-1, "x")],
            "-(x) - -1": [(-1, "(x)"), (1, "1")],
            "1e-10*(x-2)": [(1, "1e-10*(x-2)")],
            "1.5e+3 - x": [(1, "1.5e+3"), (-1, "x")],
            "2e-x": [(1, "2e"), (-1, "x")],
            "x^-1 - 1": [(1, "x^-1"), (-1, "1")],
            "2*-x+3": [(1, "2*-x"), (1, "3")],
            "abs(x-1) + abs(x+1)": [(1, "abs(x-1)"), (1, "abs(x+1)")],
            "x²-1": [(1, "x²"), (-1, "1")],
            "5": [(1, "5")],
            "x +": [(1, "x +")],
            "(x - 1": [(1, "(x - 1")],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(split_top_level_terms(text), expected)

    def test_parse_interval(self) -> None:
        self.assertEqual(parse_interval("[0, 2*pi]"), ("0", "2*pi", True, True))
        self.assertEqual(parse_interval("(0, 2pi)"), ("0", "2pi", False, False))
        self.assertEqual(parse_interval(" [0, 2*pi) "), ("0", "2*pi", True, False))
        self.assertEqual(parse_interval("(0, 1]"), ("0", "1", False, True))
        self.assertEqual(parse_interval("0, 1"), ("0", "1", True, True))
        self.assertEqual(parse_interval("0, f(1)"), ("0", "f(1)", True, True))
        self.assertEqual(parse_interval("(1+2), 3"), ("(1+2)", "3", True, True))
        for text in ("[0, 1", "0", "[0, 1, 2]", "", "[, 1]"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_interval(text)


class TestCalculatorNotation(unittest.TestCase):
    """Functions without parentheses and e or pi glued to a letter, read as calculators do."""

    UNARY = ("sin", "cos", "tan", "sec", "ln", "log", "exp", "sqrt", "abs")
    GLUED = ("sin", "cos", "ln", "log", "sqrt")

    def test_function_calls(self) -> None:
        cases = {
            "sin x": "sin(x)",
            "sin 2x": "sin(2x)",
            "2 sin x cos x": "2*sin(x)*cos(x)",
            "ln x + ln y": "ln(x) + ln(y)",
            "sin^2 x + cos^2 x": "sin^2(x) + cos^2(x)",
            "sin^2x": "sin^2(x)",
            "sin x^2": "sin(x^2)",
            "log x^2": "log(x^2)",
            "sin x^(2y)": "sin(x^(2y))",
            "x ln x": "x*ln(x)",
            "(x+1) ln x": "(x+1)*ln(x)",
            "sin cos x": "sin(cos(x))",
            "sin 30": "sin(30)",
            "sin 2pi": "sin(2pi)",
            "sec x": "sec(x)",
            "tan x/cos x": "tan(x)/cos(x)",
            "sinx": "sin(x)",
            "sin2x": "sin(2x)",
            "lnx^2": "ln(x^2)",
            "cost": "cos(t)",
            # the operand is a whole implicit product
            "sin pi x": "sin(pi*x)",
            "sin 2pi x": "sin(2pi*x)",
            "sin 2 pi x": "sin(2*pi*x)",
            "sin 2.5 x": "sin(2.5*x)",
            "sin^-1 x": "sin^-1(x)",
            # a letter before a glued call
            "xlnx - x": "x*ln(x) - x",
            "2xsinx": "2x*sin(x)",
            "e^xsinx": "e^x*sin(x)",
            "xsinx + cosx": "x*sin(x) + cos(x)",
            "tcost": "t*cos(t)",
            "nlogn": "n*log(n)",
            # a glued operand ends with its letter, so / and * after it are clear
            "sinx/x": "sin(x)/x",
            "lnx / x": "ln(x) / x",
            "sinx*2": "sin(x)*2",
            "2cosx*x": "2cos(x)*x",
            "sin2x/2": "sin(2x)/2",
            "1/sqrtx": "1/sqrt(x)",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(insert_calculator_calls(text, self.UNARY, self.GLUED), expected)

    def test_text_that_is_left_alone(self) -> None:
        for text in (
            "sin(x)",
            "sin (x)",
            "sin^2(x)",
            "2sin(x)cos(x)",
            "sin",
            "sin + 1",
            "sin -x",
            "x sin(x)",
            "time + rate",
            "sine",
            "sinh x2",
            "secret",
            "1e5x",
            "a.sin x",
            # not clear: left for the engines to refuse
            "sin 2*x",
            "sin x/2",
            "sin x y",
            "sin x(x+1)",
            "sin x^-1",
            "sinx(x+1)",
            "sinx^-1",
            # descriptive names
            "revenue - costs",
            "seconds*1000",
            "cos_theta^2",
            "sinxcosx",
            "xsinxy",
        ):
            with self.subTest(text=text):
                expected = "sinh(x2)" if text == "sinh x2" else text
                unary = self.UNARY + ("sinh", "sine")
                self.assertEqual(insert_calculator_calls(text, unary, self.GLUED), expected)

    def test_glued_calls(self) -> None:
        glued = ("sin", "asin", "ln", "log", "cos")
        self.assertEqual(glued_call("sinx", glued), ("", "sin", "x"))
        self.assertEqual(glued_call("asinx", glued), ("", "asin", "x"))
        self.assertEqual(glued_call("sin2x", glued), ("", "sin", "2x"))
        self.assertEqual(glued_call("xlnx", glued), ("x", "ln", "x"))
        for name in ("costs", "cos_t", "xycosx", "sin", "seconds", "logs2", "log2x", "log10x", "ln2x"):
            with self.subTest(name=name):
                self.assertIsNone(glued_call(name, glued))
        # a function name is never split into a glued call: sech is not sec(h)
        functions = ("sech", "csch", "coth", "sinc")
        for name in functions:
            with self.subTest(name=name):
                self.assertIsNone(glued_call(name, ("sec", "csc", "cot", "sin"), functions))
        self.assertEqual(glued_call("sinhx", ("sin", "sinh")), ("", "sinh", "x"))
        # the letter before is never the start of a function name: asinhx is not a*sinh(x)
        self.assertIsNone(glued_call("asinhx", ("sinh",), ("asinh",)))
        self.assertEqual(glued_call("asinhx", ("sinh", "asinh"), ("asinh",)), ("", "asinh", "x"))
        self.assertEqual(glued_call("xasinhx", ("sinh", "asinh"), ("asinh",)), ("x", "asinh", "x"))

    def test_function_names_are_not_split(self) -> None:
        unary = self.UNARY + ("sech", "coth")
        glued = self.GLUED + ("sec", "cot")
        functions = ("sech", "coth", "csch", "acoth")
        cases = {"sech x": "sech(x)", "coth x = 2": "coth(x) = 2", "csch + 1": "csch + 1", "sechx": "sechx"}
        cases.update({"acothx": "acothx", "acoth x": "acoth x"})
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(insert_calculator_calls(text, unary, glued, function_names=functions), expected)

    def test_glued_calls_beside_names_written_as_words(self) -> None:
        # beside profit and revenue, cost is a variable; Greek letters and xpi are not words
        for text in ("profit = revenue - cost", "revenue - cost + sinx", "cos_theta + secs"):
            with self.subTest(text=text):
                self.assertEqual(insert_calculator_calls(text, self.UNARY, self.GLUED), text)
        self.assertEqual(insert_calculator_calls("theta + sinx", self.UNARY, self.GLUED), "theta + sin(x)")
        self.assertEqual(insert_calculator_calls("xpi + cosx", self.UNARY, self.GLUED), "xpi + cos(x)")
        self.assertEqual(insert_calculator_calls("revenue + sin x", self.UNARY, self.GLUED), "revenue + sin(x)")
        self.assertTrue(has_word_names(["profit", "cost"], self.GLUED, self.UNARY))
        self.assertFalse(has_word_names(["cost", "x", "theta", "xpi", "sin"], self.GLUED, self.UNARY))

    def test_kept_names(self) -> None:
        self.assertEqual(insert_calculator_calls("cost < 1", self.UNARY, self.GLUED, keep=("cost",)), "cost < 1")
        self.assertEqual(insert_calculator_calls("sin cost", self.UNARY, self.GLUED, keep=("cost",)), "sin(cost)")

    def test_constant_products(self) -> None:
        reserved = ("re", "im", "exp", "e", "pi")
        cases = {
            "xe^x": "x*e^x",
            "(x+1)e^x": "(x+1)e^x",
            "2xe^(x^2)": "2x*e^(x^2)",
            "xpi": "x*pi",
            "pix + ex": "pi*x + e*x",
            "re(z) + ie": "re(z) + ie",
            "time + rate + xye": "time + rate + xye",
            "ex(2)": "ex(2)",
            "1e5 + 2.5e-3": "1e5 + 2.5e-3",
            "a.xe": "a.xe",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(split_constant_products(text, reserved), expected)
        self.assertEqual(split_constant_products("ex > 1", reserved, keep=("ex",)), "ex > 1")


class TestRewriteForNerdamer(unittest.TestCase):
    def test_renames_ln_and_sgn(self) -> None:
        self.assertEqual(rewrite_for_nerdamer("ln(x)+sgn(x)"), "log(x)+sign(x)")

    def test_log2_and_two_argument_log(self) -> None:
        self.assertEqual(rewrite_for_nerdamer("log2(x)"), "(log(x)/log(2))")
        self.assertEqual(rewrite_for_nerdamer("log(x, 10)"), "(log(x)/log(10))")

    def test_one_argument_log_is_unchanged(self) -> None:
        self.assertEqual(rewrite_for_nerdamer("log(x^2+1)"), "log(x^2+1)")

    def test_nested_calls(self) -> None:
        self.assertEqual(rewrite_for_nerdamer("ln(ln(x))"), "log(log(x))")
        self.assertEqual(rewrite_for_nerdamer("log(ln(x), 2)"), "(log(log(x))/log(2))")

    def test_names_containing_ln_are_untouched(self) -> None:
        self.assertEqual(rewrite_for_nerdamer("kln(x)+x.ln(2)"), "kln(x)+x.ln(2)")

    def test_unbalanced_text_is_returned_unchanged(self) -> None:
        self.assertEqual(rewrite_for_nerdamer("ln(x"), "ln(x")


class TestCasTextHelpers(unittest.TestCase):
    def test_unknown_function_names(self) -> None:
        known = ["sin", "log", "sqrt"]
        self.assertEqual(unknown_function_names("gamma(x)+sin(x)+Si(x)+gamma(2)", known), ["gamma", "Si"])

    def test_one_letter_names_are_products(self) -> None:
        self.assertEqual(unknown_function_names("x(x+1)", []), [])

    def test_has_inequality(self) -> None:
        for text in ("x < 2", "x>=1", "x != 3", "x ≤ 2", "x ≠ 1"):
            self.assertTrue(has_inequality(text), text)
        self.assertFalse(has_inequality("x^2 = 4"))

    def test_is_unevaluated(self) -> None:
        self.assertTrue(is_unevaluated("integrate(sin(x^2),x)"))
        self.assertTrue(is_unevaluated("2*limit(f,x,0)"))
        self.assertFalse(is_unevaluated("-cos(x)"))

    def test_find_closing_parenthesis(self) -> None:
        self.assertEqual(find_closing_parenthesis("f(a(b)c)d", 1), 7)
        self.assertEqual(find_closing_parenthesis("f(a", 1), -1)

    def test_split_top_level_commas(self) -> None:
        self.assertEqual(split_top_level_commas("a, f(b, c), [d, e]"), ["a", "f(b, c)", "[d, e]"])


def _sorted_roots(roots: Sequence[complex]) -> List[complex]:
    return sorted(roots, key=lambda z: (round(z.real, 6), round(z.imag, 6)))


class TestPolynomialRoots(unittest.TestCase):
    def assert_roots(self, coefficients: Sequence[float], expected: Sequence[complex], tolerance: float = 1e-9) -> None:
        roots = _sorted_roots(polynomial_roots(coefficients))
        wanted = _sorted_roots(expected)
        self.assertEqual(len(roots), len(wanted), roots)
        for root, value in zip(roots, wanted):
            self.assertLess(abs(root - value), tolerance, (roots, wanted))

    def test_quadratic(self) -> None:
        self.assert_roots([6, -5, 1], [2, 3])

    def test_complex_pair(self) -> None:
        self.assert_roots([1, 0, 1], [1j, -1j])

    def test_cubic_with_three_real_roots(self) -> None:
        # x^3 - 3x + 1 (roots 2cos(2pi/9), 2cos(4pi/9), 2cos(8pi/9)): nerdamer's closed form loses them
        expected = [2 * math.cos(2 * math.pi * k / 9) for k in (1, 2, 4)]
        self.assert_roots([1, -3, 0, 1], expected)
        for root in polynomial_roots([1, -3, 0, 1]):
            self.assertEqual(root.imag, 0.0)

    def test_real_roots_are_exactly_real(self) -> None:
        roots = polynomial_roots([1, 1, 0, 1])  # x^3 + x + 1
        real = [root for root in roots if root.imag == 0.0]
        self.assertEqual(len(real), 1)
        self.assertLess(abs(real[0].real - (-0.6823278038280193)), 1e-12)

    def test_multiple_root(self) -> None:
        # (x - 2)^5
        coefficients = [-32, 80, -80, 40, -10, 1]
        self.assert_roots(coefficients, [2] * 5, tolerance=1e-12)

    def test_zero_roots(self) -> None:
        self.assert_roots([0, 0, -1, 1], [0, 0, 1])

    def test_trailing_zero_coefficients_and_constants(self) -> None:
        self.assert_roots([-1, 1, 0, 0], [1])
        self.assertEqual(polynomial_roots([5]), [])
        self.assertEqual(polynomial_roots([]), [])

    def test_cube_roots_of_two(self) -> None:
        expected = [2 ** (1 / 3) * cmath.exp(2j * math.pi * k / 3) for k in range(3)]
        self.assert_roots([-2, 0, 0, 1], expected)


class TestAdaptiveIntegral(unittest.TestCase):
    def test_polynomial(self) -> None:
        estimate = adaptive_integral(lambda x: x * x, 0.0, 3.0)
        assert estimate is not None
        self.assertTrue(estimate["converged"])
        self.assertAlmostEqual(estimate["value"], 9.0, places=12)

    def test_oscillating_integrand(self) -> None:
        estimate = adaptive_integral(lambda x: math.sin(x * x), 0.0, 3.0)
        assert estimate is not None
        self.assertAlmostEqual(estimate["value"], 0.773562526893769, places=12)

    def test_integrable_endpoint_singularity(self) -> None:
        estimate = adaptive_integral(lambda x: 1 / math.sqrt(x), 0.0, 1.0)
        assert estimate is not None
        self.assertAlmostEqual(estimate["value"], 2.0, places=6)

    def test_semicircle(self) -> None:
        estimate = adaptive_integral(lambda x: math.sqrt(max(0.0, 1 - x * x)), -1.0, 1.0)
        assert estimate is not None
        self.assertAlmostEqual(estimate["value"], math.pi / 2, places=8)

    def test_divergent_integral_does_not_converge(self) -> None:
        estimate = adaptive_integral(lambda x: 1 / x, 0.0, 1.0)
        assert estimate is not None
        self.assertFalse(estimate["converged"])

    def test_undefined_integrand(self) -> None:
        def undefined_above_one(x: float) -> Optional[float]:
            return None if x > 1 else x

        self.assertIsNone(adaptive_integral(undefined_above_one, 0.0, 2.0))

    def test_empty_interval(self) -> None:
        self.assertEqual(adaptive_integral(lambda x: x, 1.0, 1.0), {"value": 0.0, "error": 0.0, "converged": True})


class TestFormatting(unittest.TestCase):
    def test_snap(self) -> None:
        self.assertEqual(snap(2.0000000000000004), 2.0)
        self.assertEqual(str(snap(-0.0)), "0.0")
        self.assertEqual(snap(0.5), 0.5)

    def test_format_number(self) -> None:
        self.assertEqual(format_number(3.0), "3")
        self.assertEqual(format_number(-0.6823278038280193), "-0.682327803828019")

    def test_format_complex(self) -> None:
        self.assertEqual(format_complex(complex(0.5, 1.25)), "0.5+1.25*i")
        self.assertEqual(format_complex(complex(-0.5, -1)), "-0.5-i")
        self.assertEqual(format_complex(complex(0, 2)), "2*i")
        self.assertEqual(format_complex(complex(0, -1)), "-i")
        self.assertEqual(format_complex(complex(4, 0)), "4")

    def test_distinct_values(self) -> None:
        self.assertEqual(distinct_values([2.0, 1.0, 2.0 + 1e-12, 1.0]), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
