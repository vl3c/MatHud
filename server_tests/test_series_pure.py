"""Pure Python tests for series partial-sum utilities.

These tests verify the core series functions in utils.series
without requiring browser/Brython dependencies.
"""

from __future__ import annotations

import math
import unittest

from utils.series import (
    arithmetic_partial_sums,
    geometric_partial_sums,
    harmonic_partial_sums,
    fibonacci_partial_sums,
    taylor_exp_partial_sums,
    taylor_sin_partial_sums,
    taylor_cos_partial_sums,
    leibniz_partial_sums,
    _require_positive_int,
    _require_finite,
)


class TestValidation(unittest.TestCase):
    """Tests for input validation helpers."""

    def test_require_positive_int_accepts_valid(self) -> None:
        self.assertEqual(_require_positive_int(1, "n"), 1)
        self.assertEqual(_require_positive_int(100, "n"), 100)

    def test_require_positive_int_rejects_zero(self) -> None:
        with self.assertRaises(ValueError):
            _require_positive_int(0, "n")

    def test_require_positive_int_rejects_negative(self) -> None:
        with self.assertRaises(ValueError):
            _require_positive_int(-5, "n")

    def test_require_positive_int_rejects_float(self) -> None:
        with self.assertRaises(TypeError):
            _require_positive_int(3.0, "n")  # type: ignore[arg-type]

    def test_require_positive_int_rejects_bool(self) -> None:
        with self.assertRaises(TypeError):
            _require_positive_int(True, "n")  # type: ignore[arg-type]

    def test_require_finite_accepts_numbers(self) -> None:
        self.assertEqual(_require_finite(3.14, "x"), 3.14)
        self.assertEqual(_require_finite(0, "x"), 0.0)
        self.assertEqual(_require_finite(-7, "x"), -7.0)

    def test_require_finite_accepts_int_and_returns_float(self) -> None:
        result = _require_finite(5, "x")
        self.assertIsInstance(result, float)
        self.assertEqual(result, 5.0)

    def test_require_finite_rejects_inf(self) -> None:
        with self.assertRaises(ValueError):
            _require_finite(float("inf"), "x")
        with self.assertRaises(ValueError):
            _require_finite(float("-inf"), "x")

    def test_require_finite_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            _require_finite(float("nan"), "x")

    def test_require_finite_rejects_non_numeric(self) -> None:
        with self.assertRaises(TypeError):
            _require_finite("hello", "x")  # type: ignore[arg-type]

    def test_require_finite_rejects_none(self) -> None:
        with self.assertRaises(TypeError):
            _require_finite(None, "x")  # type: ignore[arg-type]

    def test_require_finite_rejects_bool(self) -> None:
        with self.assertRaises(TypeError):
            _require_finite(False, "x")  # type: ignore[arg-type]

    def test_all_functions_reject_n_zero(self) -> None:
        """Every series function should reject n=0."""
        with self.assertRaises(ValueError):
            arithmetic_partial_sums(1, 1, 0)
        with self.assertRaises(ValueError):
            geometric_partial_sums(1, 2, 0)
        with self.assertRaises(ValueError):
            harmonic_partial_sums(0)
        with self.assertRaises(ValueError):
            fibonacci_partial_sums(0)
        with self.assertRaises(ValueError):
            taylor_exp_partial_sums(1, 0)
        with self.assertRaises(ValueError):
            taylor_sin_partial_sums(1, 0)
        with self.assertRaises(ValueError):
            taylor_cos_partial_sums(1, 0)
        with self.assertRaises(ValueError):
            leibniz_partial_sums(0)

    def test_all_functions_reject_negative_n(self) -> None:
        """Every series function should reject negative n."""
        with self.assertRaises(ValueError):
            arithmetic_partial_sums(1, 1, -3)
        with self.assertRaises(ValueError):
            geometric_partial_sums(1, 2, -1)
        with self.assertRaises(ValueError):
            harmonic_partial_sums(-5)
        with self.assertRaises(ValueError):
            fibonacci_partial_sums(-2)
        with self.assertRaises(ValueError):
            taylor_exp_partial_sums(1, -1)
        with self.assertRaises(ValueError):
            taylor_sin_partial_sums(1, -4)
        with self.assertRaises(ValueError):
            taylor_cos_partial_sums(1, -1)
        with self.assertRaises(ValueError):
            leibniz_partial_sums(-10)

    def test_taylor_functions_reject_infinite_x(self) -> None:
        """Taylor functions should reject non-finite x values."""
        for func in (taylor_exp_partial_sums, taylor_sin_partial_sums, taylor_cos_partial_sums):
            with self.subTest(func=func.__name__):
                with self.assertRaises(ValueError):
                    func(float("inf"), 5)
                with self.assertRaises(ValueError):
                    func(float("nan"), 5)

    def test_all_functions_reject_bool_inputs(self) -> None:
        """Bools should not be accepted as numeric/int inputs."""
        with self.assertRaises(TypeError):
            arithmetic_partial_sums(1, 1, True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            geometric_partial_sums(1, 2, False)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            harmonic_partial_sums(True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            fibonacci_partial_sums(False)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            taylor_exp_partial_sums(True, 5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            taylor_sin_partial_sums(False, 5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            taylor_cos_partial_sums(True, 5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            leibniz_partial_sums(False)  # type: ignore[arg-type]


class TestArithmeticSeries(unittest.TestCase):
    """Tests for arithmetic_partial_sums."""

    def test_basic(self) -> None:
        result = arithmetic_partial_sums(1, 1, 5)
        self.assertEqual(result["partial_sums"], [1, 3, 6, 10, 15])

    def test_constant_series(self) -> None:
        result = arithmetic_partial_sums(5, 0, 4)
        self.assertEqual(result["partial_sums"], [5, 10, 15, 20])

    def test_negative_difference(self) -> None:
        result = arithmetic_partial_sums(10, -2, 4)
        # terms: 10, 8, 6, 4 → sums: 10, 18, 24, 28
        self.assertEqual(result["partial_sums"], [10, 18, 24, 28])

    def test_single_term(self) -> None:
        result = arithmetic_partial_sums(7, 3, 1)
        self.assertEqual(result["partial_sums"], [7])

    def test_zero_first_term(self) -> None:
        result = arithmetic_partial_sums(0, 3, 4)
        # terms: 0, 3, 6, 9 → sums: 0, 3, 9, 18
        self.assertEqual(result["partial_sums"], [0, 3, 9, 18])

    def test_fractional_values(self) -> None:
        result = arithmetic_partial_sums(0.5, 0.5, 3)
        # terms: 0.5, 1.0, 1.5 → sums: 0.5, 1.5, 3.0
        expected = [0.5, 1.5, 3.0]
        for actual, exp in zip(result["partial_sums"], expected):
            self.assertAlmostEqual(actual, exp)

    def test_result_structure(self) -> None:
        result = arithmetic_partial_sums(1, 2, 3)
        self.assertEqual(result["series_type"], "arithmetic")
        self.assertEqual(result["num_terms"], 3)
        self.assertEqual(result["parameters"], {"a": 1.0, "d": 2.0})
        self.assertIsInstance(result["partial_sums"], list)
        self.assertEqual(len(result["partial_sums"]), 3)

    def test_length_matches_n(self) -> None:
        for n in (1, 5, 10):
            with self.subTest(n=n):
                result = arithmetic_partial_sums(1, 1, n)
                self.assertEqual(len(result["partial_sums"]), n)


class TestGeometricSeries(unittest.TestCase):
    """Tests for geometric_partial_sums."""

    def test_basic(self) -> None:
        result = geometric_partial_sums(1, 2, 5)
        self.assertEqual(result["partial_sums"], [1, 3, 7, 15, 31])

    def test_ratio_one(self) -> None:
        result = geometric_partial_sums(3, 1, 4)
        self.assertEqual(result["partial_sums"], [3, 6, 9, 12])

    def test_ratio_negative(self) -> None:
        result = geometric_partial_sums(1, -1, 4)
        # terms: 1, -1, 1, -1 → sums: 1, 0, 1, 0
        self.assertEqual(result["partial_sums"], [1, 0, 1, 0])

    def test_ratio_fractional(self) -> None:
        result = geometric_partial_sums(1, 0.5, 4)
        # S_k = (1 - 0.5^k) / (1 - 0.5) = 2*(1 - 0.5^k)
        expected = [1.0, 1.5, 1.75, 1.875]
        for actual, exp in zip(result["partial_sums"], expected):
            self.assertAlmostEqual(actual, exp)

    def test_a_zero_returns_zeros(self) -> None:
        result = geometric_partial_sums(0, 2, 5)
        self.assertEqual(result["partial_sums"], [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_single_term(self) -> None:
        result = geometric_partial_sums(5, 3, 1)
        self.assertEqual(result["partial_sums"], [5])

    def test_ratio_zero(self) -> None:
        result = geometric_partial_sums(4, 0, 5)
        # terms: 4, 0, 0, 0, 0 → sums: 4, 4, 4, 4, 4
        self.assertEqual(result["partial_sums"], [4.0, 4.0, 4.0, 4.0, 4.0])

    def test_negative_a(self) -> None:
        result = geometric_partial_sums(-2, 3, 3)
        # terms: -2, -6, -18 → sums: -2, -8, -26
        expected = [-2.0, -8.0, -26.0]
        for actual, exp in zip(result["partial_sums"], expected):
            self.assertAlmostEqual(actual, exp)

    def test_result_structure(self) -> None:
        result = geometric_partial_sums(2, 3, 4)
        self.assertEqual(result["series_type"], "geometric")
        self.assertEqual(result["num_terms"], 4)
        self.assertEqual(result["parameters"], {"a": 2.0, "r": 3.0})


class TestHarmonicSeries(unittest.TestCase):
    """Tests for harmonic_partial_sums."""

    def test_basic(self) -> None:
        result = harmonic_partial_sums(4)
        expected = [1.0, 1.5, 11.0 / 6.0, 25.0 / 12.0]
        for actual, exp in zip(result["partial_sums"], expected):
            self.assertAlmostEqual(actual, exp)

    def test_single_term(self) -> None:
        result = harmonic_partial_sums(1)
        self.assertEqual(result["partial_sums"], [1.0])

    def test_large_n_approximation(self) -> None:
        # For large n, H_n ≈ ln(n) + γ (Euler-Mascheroni constant)
        euler_gamma = 0.5772156649015329
        result = harmonic_partial_sums(1000)
        h_1000 = result["partial_sums"][-1]
        approx = math.log(1000) + euler_gamma
        self.assertAlmostEqual(h_1000, approx, places=2)

    def test_monotonically_increasing(self) -> None:
        result = harmonic_partial_sums(20)
        sums = result["partial_sums"]
        for i in range(1, len(sums)):
            self.assertGreater(sums[i], sums[i - 1])

    def test_result_structure(self) -> None:
        result = harmonic_partial_sums(3)
        self.assertEqual(result["series_type"], "harmonic")
        self.assertEqual(result["num_terms"], 3)
        self.assertEqual(result["parameters"], {})
        self.assertEqual(len(result["partial_sums"]), 3)


class TestFibonacciSeries(unittest.TestCase):
    """Tests for fibonacci_partial_sums."""

    def test_basic(self) -> None:
        result = fibonacci_partial_sums(7)
        # Fib: 1, 1, 2, 3, 5, 8, 13
        # Sums: 1, 2, 4, 7, 12, 20, 33
        self.assertEqual(result["partial_sums"], [1, 2, 4, 7, 12, 20, 33])

    def test_single_term(self) -> None:
        result = fibonacci_partial_sums(1)
        self.assertEqual(result["partial_sums"], [1])

    def test_two_terms(self) -> None:
        result = fibonacci_partial_sums(2)
        self.assertEqual(result["partial_sums"], [1, 2])

    def test_known_identity(self) -> None:
        # S_n = F_{n+2} - 1 is a well-known Fibonacci identity
        n = 10
        result = fibonacci_partial_sums(n)
        s_n = result["partial_sums"][-1]
        # Compute F_{n+2} directly: F_1=1, F_2=1, F_3=2, ...
        fib = [0, 1, 1]
        while len(fib) <= n + 2:
            fib.append(fib[-1] + fib[-2])
        f_n_plus_2 = fib[n + 2]
        self.assertEqual(s_n, f_n_plus_2 - 1)

    def test_monotonically_increasing(self) -> None:
        result = fibonacci_partial_sums(15)
        sums = result["partial_sums"]
        for i in range(1, len(sums)):
            self.assertGreater(sums[i], sums[i - 1])

    def test_result_structure(self) -> None:
        result = fibonacci_partial_sums(5)
        self.assertEqual(result["series_type"], "fibonacci")
        self.assertEqual(result["num_terms"], 5)
        self.assertEqual(result["parameters"], {})


class TestTaylorExp(unittest.TestCase):
    """Tests for taylor_exp_partial_sums."""

    def test_at_zero(self) -> None:
        result = taylor_exp_partial_sums(0, 5)
        # e^0 = 1, all partial sums should be 1.0
        self.assertEqual(result["partial_sums"], [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_at_one_convergence(self) -> None:
        result = taylor_exp_partial_sums(1, 20)
        last = result["partial_sums"][-1]
        self.assertAlmostEqual(last, math.e, places=10)

    def test_negative_x(self) -> None:
        result = taylor_exp_partial_sums(-1, 20)
        last = result["partial_sums"][-1]
        self.assertAlmostEqual(last, 1.0 / math.e, places=10)

    def test_single_term(self) -> None:
        result = taylor_exp_partial_sums(5, 1)
        # First term is x^0/0! = 1
        self.assertEqual(result["partial_sums"], [1.0])

    def test_monotonically_increasing_for_positive_x(self) -> None:
        # All terms x^i/i! are positive when x > 0, so sums increase
        result = taylor_exp_partial_sums(2, 10)
        sums = result["partial_sums"]
        for i in range(1, len(sums)):
            self.assertGreater(sums[i], sums[i - 1])

    def test_at_two_convergence(self) -> None:
        result = taylor_exp_partial_sums(2, 25)
        last = result["partial_sums"][-1]
        self.assertAlmostEqual(last, math.exp(2), places=10)

    def test_result_structure(self) -> None:
        result = taylor_exp_partial_sums(3, 4)
        self.assertEqual(result["series_type"], "taylor_exp")
        self.assertEqual(result["num_terms"], 4)
        self.assertEqual(result["parameters"], {"x": 3.0})


class TestTaylorSin(unittest.TestCase):
    """Tests for taylor_sin_partial_sums."""

    def test_at_zero(self) -> None:
        result = taylor_sin_partial_sums(0, 5)
        # sin(0) = 0, all terms are 0
        self.assertEqual(result["partial_sums"], [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_at_pi_half(self) -> None:
        result = taylor_sin_partial_sums(math.pi / 2, 10)
        last = result["partial_sums"][-1]
        self.assertAlmostEqual(last, 1.0, places=10)

    def test_convergence(self) -> None:
        x = 1.5
        result = taylor_sin_partial_sums(x, 15)
        last = result["partial_sums"][-1]
        self.assertAlmostEqual(last, math.sin(x), places=10)

    def test_single_term(self) -> None:
        result = taylor_sin_partial_sums(2.0, 1)
        # First term is x
        self.assertEqual(result["partial_sums"], [2.0])

    def test_negative_x(self) -> None:
        x = -1.0
        result = taylor_sin_partial_sums(x, 15)
        last = result["partial_sums"][-1]
        self.assertAlmostEqual(last, math.sin(x), places=10)

    def test_at_pi_converges_to_zero(self) -> None:
        result = taylor_sin_partial_sums(math.pi, 15)
        last = result["partial_sums"][-1]
        self.assertAlmostEqual(last, 0.0, places=10)

    def test_result_structure(self) -> None:
        result = taylor_sin_partial_sums(1.0, 3)
        self.assertEqual(result["series_type"], "taylor_sin")
        self.assertEqual(result["parameters"], {"x": 1.0})


class TestTaylorCos(unittest.TestCase):
    """Tests for taylor_cos_partial_sums."""

    def test_at_zero(self) -> None:
        result = taylor_cos_partial_sums(0, 5)
        # cos(0) = 1, all partial sums should be 1.0
        self.assertEqual(result["partial_sums"], [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_convergence(self) -> None:
        x = 1.5
        result = taylor_cos_partial_sums(x, 15)
        last = result["partial_sums"][-1]
        self.assertAlmostEqual(last, math.cos(x), places=10)

    def test_single_term(self) -> None:
        result = taylor_cos_partial_sums(3.0, 1)
        # First term is 1.0
        self.assertEqual(result["partial_sums"], [1.0])

    def test_at_pi_converges_to_negative_one(self) -> None:
        result = taylor_cos_partial_sums(math.pi, 15)
        last = result["partial_sums"][-1]
        self.assertAlmostEqual(last, -1.0, places=10)

    def test_negative_x(self) -> None:
        # cos is even: cos(-x) == cos(x)
        x = -2.0
        result = taylor_cos_partial_sums(x, 15)
        last = result["partial_sums"][-1]
        self.assertAlmostEqual(last, math.cos(x), places=10)

    def test_result_structure(self) -> None:
        result = taylor_cos_partial_sums(2.0, 3)
        self.assertEqual(result["series_type"], "taylor_cos")
        self.assertEqual(result["parameters"], {"x": 2.0})


class TestLeibnizSeries(unittest.TestCase):
    """Tests for leibniz_partial_sums."""

    def test_basic(self) -> None:
        result = leibniz_partial_sums(4)
        expected = [
            1.0,
            1.0 - 1.0 / 3.0,
            1.0 - 1.0 / 3.0 + 1.0 / 5.0,
            1.0 - 1.0 / 3.0 + 1.0 / 5.0 - 1.0 / 7.0,
        ]
        for actual, exp in zip(result["partial_sums"], expected):
            self.assertAlmostEqual(actual, exp)

    def test_convergence(self) -> None:
        result = leibniz_partial_sums(10000)
        last = result["partial_sums"][-1]
        self.assertAlmostEqual(last, math.pi / 4, places=3)

    def test_alternating(self) -> None:
        result = leibniz_partial_sums(20)
        target = math.pi / 4
        sums = result["partial_sums"]
        # Consecutive partial sums should alternate above/below pi/4
        for i in range(1, len(sums)):
            above_prev = sums[i - 1] > target
            above_curr = sums[i] > target
            self.assertNotEqual(above_prev, above_curr)

    def test_single_term(self) -> None:
        result = leibniz_partial_sums(1)
        self.assertEqual(result["partial_sums"], [1.0])

    def test_result_structure(self) -> None:
        result = leibniz_partial_sums(5)
        self.assertEqual(result["series_type"], "leibniz")
        self.assertEqual(result["num_terms"], 5)
        self.assertEqual(result["parameters"], {})


if __name__ == "__main__":
    unittest.main()
