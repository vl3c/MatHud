from __future__ import annotations

import json
import math
import random
import statistics
import time
import unittest

from utils.statistics.descriptive import compute_descriptive_statistics
from utils.statistics.tool import SAMPLE_FIGURES_NOTE, compute_descriptive_statistics_tool
from utils.statistics.two_variable import compute_two_variable_statistics, paired_sums

X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]


class TestKnownValues(unittest.TestCase):
    def test_textbook_pairs(self) -> None:
        # Sxx = 10, Syy = 6, Sxy = 6: r = 6 / sqrt(60)
        result = compute_two_variable_statistics(X, Y)
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["r"], 0.7745966692414834)
        self.assertAlmostEqual(result["r_squared"], 0.6, places=15)
        self.assertEqual(result["sample_covariance"], 1.5)
        self.assertEqual(result["population_covariance"], 1.2)

    def test_nested_one_variable_statistics(self) -> None:
        result = compute_two_variable_statistics(X, Y)
        self.assertEqual(result["x"], compute_descriptive_statistics(X))
        self.assertEqual(result["y"], compute_descriptive_statistics(Y))
        self.assertEqual(result["x"]["sample_standard_deviation"], 1.5811388300841898)

    def test_key_order(self) -> None:
        result = compute_two_variable_statistics(X, Y)
        self.assertEqual(
            list(result.keys()),
            ["count", "x", "y", "r", "r_squared", "sample_covariance", "population_covariance"],
        )

    def test_reversed_y_gives_minus_one(self) -> None:
        result = compute_two_variable_statistics(X, [5, 4, 3, 2, 1])
        self.assertEqual(result["r"], -1.0)
        self.assertEqual(result["r_squared"], 1.0)
        self.assertEqual(result["sample_covariance"], -2.5)

    def test_r_is_clamped(self) -> None:
        rng = random.Random(7)
        for _ in range(200):
            x = [rng.uniform(-10, 10) for _ in range(rng.randint(2, 30))]
            scale = rng.uniform(-1e3, 1e3)
            r = compute_two_variable_statistics(x, [scale * v for v in x])["r"]
            assert r is not None
            self.assertLessEqual(abs(r), 1.0)
            self.assertAlmostEqual(abs(r), 1.0, places=12)


class TestEqualValues(unittest.TestCase):
    def test_constant_x(self) -> None:
        result = compute_two_variable_statistics([3, 3, 3], [1, 2, 3])
        self.assertIsNone(result["r"])
        self.assertIsNone(result["r_squared"])
        self.assertEqual(result["sample_covariance"], 0.0)
        self.assertEqual(result["population_covariance"], 0.0)

    def test_constant_y(self) -> None:
        result = compute_two_variable_statistics([1, 2, 3], [0.1, 0.1, 0.1])
        self.assertIsNone(result["r"])
        self.assertEqual(result["sample_covariance"], 0.0)

    def test_constant_lists_whose_mean_misses_the_value(self) -> None:
        # The fsum mean of [0.1] * 3 is 0.10000000000000002; the test is min == max
        for value in (0.1, 0.7, 1e-300, 123456.789):
            with self.subTest(value=value):
                result = compute_two_variable_statistics([1, 2, 3], [value] * 3)
                self.assertIsNone(result["r"])
                self.assertEqual(result["sample_covariance"], 0.0)

    def test_large_values_with_small_spread_are_not_constant(self) -> None:
        # An absolute rule like fit_linear's would call this x constant
        result = compute_two_variable_statistics([1e13 + 1, 1e13 + 2, 1e13 + 3], [1, 2, 3])
        self.assertEqual(result["r"], 1.0)

    def test_notes_name_the_equal_axis(self) -> None:
        cases = (
            ([3, 3, 3], [1, 2, 3], "r and r_squared are undefined because all x values are equal"),
            ([1, 2, 3], [4, 4, 4], "r and r_squared are undefined because all y values are equal"),
            ([1, 1], [2, 2], "r and r_squared are undefined because all x values and all y values are equal"),
        )
        for x, y, note in cases:
            with self.subTest(x=x, y=y):
                self.assertEqual(compute_descriptive_statistics_tool(x, y)["note"], note)

    def test_no_note_when_r_is_defined(self) -> None:
        self.assertNotIn("note", compute_descriptive_statistics_tool(X, Y))


class TestNumerics(unittest.TestCase):
    def test_shifted_x_keeps_r(self) -> None:
        shifted = compute_two_variable_statistics([v + 1e9 for v in X], Y)["r"]
        assert shifted is not None
        self.assertAlmostEqual(shifted, 0.7745966692414834, places=9)

    def test_huge_values_keep_r_finite(self) -> None:
        data = [1e200, 2e200, 3e200]
        result = compute_two_variable_statistics(data, data)
        self.assertEqual(result["r"], 1.0)
        self.assertEqual(result["sample_covariance"], math.inf)
        self.assertEqual(json.loads(json.dumps(result))["sample_covariance"], math.inf)

    def test_deviations_beyond_the_float_limit(self) -> None:
        # The deviation of -1.7e308 from the mean overflows; r is still exact
        result = compute_two_variable_statistics([-1.7e308, 1.7e308, 1.7e308], [1, 2, 2])
        self.assertEqual(result["r"], 1.0)
        self.assertEqual(result["sample_covariance"], math.inf)

    def test_tiny_values(self) -> None:
        result = compute_two_variable_statistics([1e-300, 2e-300, 3e-300], [3, 2, 1])
        self.assertEqual(result["r"], -1.0)

    def test_agrees_with_the_statistics_module(self) -> None:
        rng = random.Random(20261016)
        for trial in range(200):
            n = rng.randint(2, 40)
            x = [rng.uniform(-1e3, 1e3) for _ in range(n)]
            y = [rng.uniform(-50, 50) + 0.3 * v for v in x]
            result = compute_two_variable_statistics(x, y)
            r = result["r"]
            assert r is not None
            with self.subTest(trial=trial):
                self.assertTrue(math.isclose(r, statistics.correlation(x, y), rel_tol=1e-12, abs_tol=1e-15))
                self.assertTrue(math.isclose(result["sample_covariance"], statistics.covariance(x, y), rel_tol=1e-12))
                self.assertTrue(
                    math.isclose(
                        result["population_covariance"], statistics.covariance(x, y) * (n - 1) / n, rel_tol=1e-12
                    )
                )

    def test_large_input_runs_in_linear_time(self) -> None:
        rng = random.Random(3)
        x = [rng.random() for _ in range(100_000)]
        y = [v + rng.random() for v in x]
        started = time.perf_counter()
        result = compute_two_variable_statistics(x, y)
        self.assertLess(time.perf_counter() - started, 10.0)
        self.assertEqual(result["count"], 100_000)

    def test_paired_sums_exposes_the_scaled_sums(self) -> None:
        sums = paired_sums(X, Y)
        self.assertAlmostEqual(sums.x_scale**2 * sums.sxx, 10.0, places=12)
        self.assertAlmostEqual(sums.y_scale**2 * sums.syy, 6.0, places=12)
        self.assertAlmostEqual(sums.x_scale * sums.y_scale * sums.sxy, 6.0, places=12)
        self.assertEqual(sums.covariance(4), 1.5)


class TestValidation(unittest.TestCase):
    def test_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, r"data and y_data must have the same length \(5 and 4\)"):
            compute_two_variable_statistics(X, Y[:4])

    def test_one_pair(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            compute_two_variable_statistics([1], [2])

    def test_errors_name_y_data(self) -> None:
        cases = (
            ([1, True, 3], TypeError, r"y_data\[1\] is a bool"),
            ([1, math.nan, 3], ValueError, r"y_data\[1\] is not finite"),
            ([1, math.inf, 3], ValueError, r"y_data\[1\] is not finite"),
            ([1, "2", 3], TypeError, r"y_data\[1\] must be int or float"),
            ((1, 2, 3), TypeError, "y_data must be a list"),
        )
        for y, error, message in cases:
            with self.subTest(y=y):
                with self.assertRaisesRegex(error, message):
                    compute_two_variable_statistics([1, 2, 3], y)  # type: ignore[arg-type]

    def test_errors_name_data(self) -> None:
        with self.assertRaisesRegex(ValueError, r"^data\[0\] is not finite"):
            compute_two_variable_statistics([math.inf, 2], [1, 2])


class TestToolEntryPoint(unittest.TestCase):
    def test_without_y_data(self) -> None:
        for y_data in (None, []):
            with self.subTest(y_data=y_data):
                result = compute_descriptive_statistics_tool([10, 20, 30, 40, 50], y_data)
                self.assertEqual(result, dict(compute_descriptive_statistics([10, 20, 30, 40, 50])))
                self.assertNotIn("note", result)

    def test_default_is_one_list(self) -> None:
        self.assertEqual(compute_descriptive_statistics_tool([1, 2])["count"], 2)

    def test_single_value_note(self) -> None:
        result = compute_descriptive_statistics_tool([5], None)
        self.assertIsNone(result["sample_standard_deviation"])
        self.assertEqual(result["note"], SAMPLE_FIGURES_NOTE)

    def test_with_y_data(self) -> None:
        result = compute_descriptive_statistics_tool(X, Y)
        self.assertEqual(result, dict(compute_two_variable_statistics(X, Y)))

    def test_result_is_json(self) -> None:
        result = compute_descriptive_statistics_tool([3, 3], [1, 2])
        self.assertEqual(json.loads(json.dumps(result)), result)


if __name__ == "__main__":
    unittest.main()
