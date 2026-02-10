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
)


class TestSeries(unittest.TestCase):
    """Brython-side tests for series partial-sum utilities."""

    def test_arithmetic_partial_sums_basic(self) -> None:
        result = arithmetic_partial_sums(1, 1, 5)
        self.assertEqual(result["partial_sums"], [1, 3, 6, 10, 15])

    def test_arithmetic_constant_series(self) -> None:
        result = arithmetic_partial_sums(5, 0, 4)
        self.assertEqual(result["partial_sums"], [5, 10, 15, 20])

    def test_geometric_partial_sums_basic(self) -> None:
        result = geometric_partial_sums(1, 2, 5)
        self.assertEqual(result["partial_sums"], [1, 3, 7, 15, 31])

    def test_geometric_ratio_one(self) -> None:
        result = geometric_partial_sums(3, 1, 4)
        self.assertEqual(result["partial_sums"], [3, 6, 9, 12])

    def test_harmonic_partial_sums_basic(self) -> None:
        result = harmonic_partial_sums(4)
        expected = [1.0, 1.5, 11.0 / 6.0, 25.0 / 12.0]
        for actual, exp in zip(result["partial_sums"], expected):
            self.assertAlmostEqual(actual, exp)

    def test_fibonacci_partial_sums_basic(self) -> None:
        result = fibonacci_partial_sums(7)
        self.assertEqual(result["partial_sums"], [1, 2, 4, 7, 12, 20, 33])

    def test_taylor_exp_convergence(self) -> None:
        result = taylor_exp_partial_sums(1, 20)
        last = result["partial_sums"][-1]
        self.assertAlmostEqual(last, math.e, places=10)

    def test_taylor_sin_convergence(self) -> None:
        result = taylor_sin_partial_sums(math.pi / 2, 10)
        last = result["partial_sums"][-1]
        self.assertAlmostEqual(last, 1.0, places=10)

    def test_taylor_cos_convergence(self) -> None:
        result = taylor_cos_partial_sums(0, 5)
        self.assertEqual(result["partial_sums"], [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_taylor_cos_at_pi(self) -> None:
        result = taylor_cos_partial_sums(math.pi, 15)
        last = result["partial_sums"][-1]
        self.assertAlmostEqual(last, -1.0, places=10)

    def test_leibniz_convergence(self) -> None:
        result = leibniz_partial_sums(10000)
        last = result["partial_sums"][-1]
        self.assertAlmostEqual(last, math.pi / 4, places=3)

    def test_validation_rejects_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            arithmetic_partial_sums(1, 1, 0)
        with self.assertRaises(TypeError):
            arithmetic_partial_sums(1, 1, 2.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            taylor_exp_partial_sums(float("inf"), 5)
        with self.assertRaises(ValueError):
            harmonic_partial_sums(-1)

    def test_result_structure(self) -> None:
        result = arithmetic_partial_sums(1, 2, 3)
        self.assertIn("series_type", result)
        self.assertIn("partial_sums", result)
        self.assertIn("num_terms", result)
        self.assertIn("parameters", result)
        self.assertEqual(result["series_type"], "arithmetic")
        self.assertEqual(result["num_terms"], 3)
        self.assertIsInstance(result["partial_sums"], list)
        self.assertIsInstance(result["parameters"], dict)

    def test_all_series_types_correct(self) -> None:
        self.assertEqual(arithmetic_partial_sums(1, 1, 1)["series_type"], "arithmetic")
        self.assertEqual(geometric_partial_sums(1, 2, 1)["series_type"], "geometric")
        self.assertEqual(harmonic_partial_sums(1)["series_type"], "harmonic")
        self.assertEqual(fibonacci_partial_sums(1)["series_type"], "fibonacci")
        self.assertEqual(taylor_exp_partial_sums(1, 1)["series_type"], "taylor_exp")
        self.assertEqual(taylor_sin_partial_sums(1, 1)["series_type"], "taylor_sin")
        self.assertEqual(taylor_cos_partial_sums(1, 1)["series_type"], "taylor_cos")
        self.assertEqual(leibniz_partial_sums(1)["series_type"], "leibniz")

    def test_single_term_returns_length_one(self) -> None:
        self.assertEqual(len(arithmetic_partial_sums(1, 1, 1)["partial_sums"]), 1)
        self.assertEqual(len(geometric_partial_sums(1, 2, 1)["partial_sums"]), 1)
        self.assertEqual(len(harmonic_partial_sums(1)["partial_sums"]), 1)
        self.assertEqual(len(fibonacci_partial_sums(1)["partial_sums"]), 1)
        self.assertEqual(len(taylor_exp_partial_sums(1, 1)["partial_sums"]), 1)
        self.assertEqual(len(taylor_sin_partial_sums(1, 1)["partial_sums"]), 1)
        self.assertEqual(len(taylor_cos_partial_sums(1, 1)["partial_sums"]), 1)
        self.assertEqual(len(leibniz_partial_sums(1)["partial_sums"]), 1)
