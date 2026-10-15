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
    is_unevaluated,
    rewrite_for_nerdamer,
    split_top_level_commas,
    unknown_function_names,
)


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
