from __future__ import annotations

import math
import random
import statistics
import unittest

from utils.statistics.descriptive import (
    compute_descriptive_statistics,
)

SPREAD_KEYS = (
    "sample_standard_deviation",
    "population_standard_deviation",
    "sample_variance",
    "population_variance",
)


class TestValidation(unittest.TestCase):
    def test_empty_list_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            compute_descriptive_statistics([])

    def test_non_list_input_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            compute_descriptive_statistics((1, 2, 3))
        with self.assertRaises(TypeError):
            compute_descriptive_statistics("123")
        with self.assertRaises(TypeError):
            compute_descriptive_statistics(42)

    def test_non_numeric_elements_raise_type_error(self) -> None:
        with self.assertRaises(TypeError):
            compute_descriptive_statistics([1, "two", 3])

    def test_non_finite_values_raise_value_error(self) -> None:
        for bad in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    compute_descriptive_statistics([1.0, bad, 3.0])

    def test_boolean_values_raise_type_error(self) -> None:
        with self.assertRaises(TypeError):
            compute_descriptive_statistics([1, True, 3])


class TestMean(unittest.TestCase):
    def test_integer_dataset(self) -> None:
        result = compute_descriptive_statistics([1, 2, 3, 4, 5])
        self.assertAlmostEqual(result["mean"], 3.0)

    def test_float_dataset(self) -> None:
        result = compute_descriptive_statistics([1.5, 2.5, 3.5])
        self.assertAlmostEqual(result["mean"], 2.5)

    def test_negative_numbers(self) -> None:
        result = compute_descriptive_statistics([-3, -1, 0, 1, 3])
        self.assertAlmostEqual(result["mean"], 0.0)

    def test_single_element(self) -> None:
        result = compute_descriptive_statistics([42])
        self.assertAlmostEqual(result["mean"], 42.0)


class TestMedian(unittest.TestCase):
    def test_odd_count(self) -> None:
        result = compute_descriptive_statistics([1, 3, 5])
        self.assertAlmostEqual(result["median"], 3.0)

    def test_even_count(self) -> None:
        result = compute_descriptive_statistics([1, 2, 3, 4])
        self.assertAlmostEqual(result["median"], 2.5)

    def test_single_element(self) -> None:
        result = compute_descriptive_statistics([7])
        self.assertAlmostEqual(result["median"], 7.0)

    def test_unsorted_input(self) -> None:
        result = compute_descriptive_statistics([5, 1, 3])
        self.assertAlmostEqual(result["median"], 3.0)


class TestMode(unittest.TestCase):
    def test_single_mode(self) -> None:
        result = compute_descriptive_statistics([1, 2, 2, 3])
        self.assertEqual(result["mode"], [2.0])

    def test_multi_modal(self) -> None:
        result = compute_descriptive_statistics([1, 1, 2, 2, 3])
        self.assertEqual(result["mode"], [1.0, 2.0])

    def test_all_unique_no_mode(self) -> None:
        result = compute_descriptive_statistics([1, 2, 3, 4])
        self.assertEqual(result["mode"], [])

    def test_all_same(self) -> None:
        result = compute_descriptive_statistics([5, 5, 5])
        self.assertEqual(result["mode"], [5.0])

    def test_all_same_frequency_multiple_distinct(self) -> None:
        result = compute_descriptive_statistics([1, 1, 2, 2, 3, 3])
        self.assertEqual(result["mode"], [])

    def test_single_element_is_the_mode(self) -> None:
        # One distinct value is the mode however often it appears, like [7, 7]
        result = compute_descriptive_statistics([42])
        self.assertEqual(result["mode"], [42.0])

    def test_two_identical_elements(self) -> None:
        result = compute_descriptive_statistics([7, 7])
        self.assertEqual(result["mode"], [7.0])

    def test_mode_elements_are_float(self) -> None:
        result = compute_descriptive_statistics([1, 2, 2, 3])
        for m in result["mode"]:
            self.assertIsInstance(m, float)


class TestVarianceAndStdDev(unittest.TestCase):
    def test_known_dataset(self) -> None:
        # [2,4,4,4,5,5,7,9]: squared deviations sum to 32, so sigma^2 = 32/8 and s^2 = 32/7
        result = compute_descriptive_statistics([2, 4, 4, 4, 5, 5, 7, 9])
        self.assertEqual(result["population_variance"], 4.0)
        self.assertEqual(result["population_standard_deviation"], 2.0)
        self.assertEqual(result["sample_variance"], 32 / 7)
        self.assertEqual(result["sample_standard_deviation"], math.sqrt(32 / 7))
        self.assertAlmostEqual(result["sample_standard_deviation"], 2.138089935299395, places=15)

    def test_scenario_dataset(self) -> None:
        # ST-04's data: sigma = sqrt(200), s = sqrt(250)
        result = compute_descriptive_statistics([10, 20, 30, 40, 50])
        self.assertEqual(result["population_standard_deviation"], 14.142135623730951)
        self.assertEqual(result["sample_standard_deviation"], 15.811388300841896)
        self.assertEqual(result["population_variance"], 200.0)
        self.assertEqual(result["sample_variance"], 250.0)

    def test_all_same_values(self) -> None:
        result = compute_descriptive_statistics([3, 3, 3])
        for key in SPREAD_KEYS:
            with self.subTest(key=key):
                self.assertEqual(result[key], 0.0)

    def test_population_and_sample_formulas(self) -> None:
        # [2, 8]: mean=5, squared deviations sum to 18: population 18/2, sample 18/1
        result = compute_descriptive_statistics([2, 8])
        self.assertAlmostEqual(result["population_variance"], 9.0)
        self.assertAlmostEqual(result["sample_variance"], 18.0)

    def test_single_value_has_no_sample_figures(self) -> None:
        result = compute_descriptive_statistics([5])
        self.assertIsNone(result["sample_standard_deviation"])
        self.assertIsNone(result["sample_variance"])
        self.assertEqual(result["population_standard_deviation"], 0.0)
        self.assertEqual(result["population_variance"], 0.0)

    def test_numerical_robustness(self) -> None:
        result = compute_descriptive_statistics([1.0, 1.0, 1.0000000000000002])
        for key in SPREAD_KEYS:
            with self.subTest(key=key):
                self.assertGreaterEqual(result[key], 0.0)

    def test_agrees_with_the_statistics_module(self) -> None:
        rng = random.Random(20261016)
        for trial in range(200):
            data = [rng.uniform(-1e3, 1e3) for _ in range(rng.randint(2, 40))]
            result = compute_descriptive_statistics(data)
            with self.subTest(trial=trial):
                self.assertTrue(
                    math.isclose(result["sample_standard_deviation"], statistics.stdev(data), rel_tol=1e-12)
                )
                self.assertTrue(
                    math.isclose(result["population_standard_deviation"], statistics.pstdev(data), rel_tol=1e-12)
                )
                self.assertTrue(math.isclose(result["sample_variance"], statistics.variance(data), rel_tol=1e-12))
                self.assertTrue(math.isclose(result["population_variance"], statistics.pvariance(data), rel_tol=1e-12))


class TestMinMax(unittest.TestCase):
    def test_mixed_values(self) -> None:
        result = compute_descriptive_statistics([-5, 0, 3, 10])
        self.assertAlmostEqual(result["min"], -5.0)
        self.assertAlmostEqual(result["max"], 10.0)

    def test_single_element(self) -> None:
        result = compute_descriptive_statistics([42])
        self.assertAlmostEqual(result["min"], 42.0)
        self.assertAlmostEqual(result["max"], 42.0)

    def test_negative_only(self) -> None:
        result = compute_descriptive_statistics([-3, -1, -7])
        self.assertAlmostEqual(result["min"], -7.0)
        self.assertAlmostEqual(result["max"], -1.0)

    def test_min_max_are_float(self) -> None:
        result = compute_descriptive_statistics([1, 2, 3])
        self.assertIsInstance(result["min"], float)
        self.assertIsInstance(result["max"], float)


class TestQuartiles(unittest.TestCase):
    def test_odd_n_textbook(self) -> None:
        # [1,2,3,4,5,6,7]: lower=[1,2,3] Q1=2.0, upper=[5,6,7] Q3=6.0
        result = compute_descriptive_statistics([1, 2, 3, 4, 5, 6, 7])
        self.assertAlmostEqual(result["q1"], 2.0)
        self.assertAlmostEqual(result["q3"], 6.0)
        self.assertAlmostEqual(result["iqr"], 4.0)

    def test_even_n(self) -> None:
        # [1,2,3,4,5,6,7,8]: lower=[1,2,3,4] Q1=2.5, upper=[5,6,7,8] Q3=6.5
        result = compute_descriptive_statistics([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertAlmostEqual(result["q1"], 2.5)
        self.assertAlmostEqual(result["q3"], 6.5)
        self.assertAlmostEqual(result["iqr"], 4.0)

    def test_n_equals_1(self) -> None:
        result = compute_descriptive_statistics([99])
        self.assertAlmostEqual(result["q1"], 99.0)
        self.assertAlmostEqual(result["q3"], 99.0)
        self.assertAlmostEqual(result["iqr"], 0.0)

    def test_n_equals_2(self) -> None:
        result = compute_descriptive_statistics([10, 20])
        self.assertAlmostEqual(result["q1"], 10.0)
        self.assertAlmostEqual(result["q3"], 20.0)

    def test_n_equals_3(self) -> None:
        # [1,2,3]: lower=[1] Q1=1.0, upper=[3] Q3=3.0
        result = compute_descriptive_statistics([1, 2, 3])
        self.assertAlmostEqual(result["q1"], 1.0)
        self.assertAlmostEqual(result["q3"], 3.0)
        self.assertAlmostEqual(result["iqr"], 2.0)


class TestRange(unittest.TestCase):
    def test_basic_range(self) -> None:
        result = compute_descriptive_statistics([1, 5, 10])
        self.assertAlmostEqual(result["range"], 9.0)

    def test_single_element_range(self) -> None:
        result = compute_descriptive_statistics([42])
        self.assertAlmostEqual(result["range"], 0.0)

    def test_range_is_float(self) -> None:
        result = compute_descriptive_statistics([1, 10])
        self.assertIsInstance(result["range"], float)


class TestEdgeCases(unittest.TestCase):
    def test_large_numbers(self) -> None:
        data = [1e15, 2e15, 3e15]
        result = compute_descriptive_statistics(data)
        self.assertAlmostEqual(result["mean"], 2e15)
        self.assertEqual(result["count"], 3)

    def test_very_small_numbers(self) -> None:
        data = [1e-15, 2e-15, 3e-15]
        result = compute_descriptive_statistics(data)
        self.assertAlmostEqual(result["mean"], 2e-15)

    def test_mixed_large_and_small(self) -> None:
        data = [0.001, 1000000]
        result = compute_descriptive_statistics(data)
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["min"], 0.001)
        self.assertAlmostEqual(result["max"], 1000000.0)


class TestResultStructure(unittest.TestCase):
    def test_all_expected_keys_present(self) -> None:
        result = compute_descriptive_statistics([1, 2, 3, 4, 5])
        expected_keys = {
            "count",
            "mean",
            "median",
            "mode",
            "sample_standard_deviation",
            "population_standard_deviation",
            "sample_variance",
            "population_variance",
            "min",
            "max",
            "q1",
            "q3",
            "iqr",
            "range",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_keys_are_in_calculator_order(self) -> None:
        # Sx before sigma x, as TI's 1-Var Stats lists them
        result = compute_descriptive_statistics([1, 2, 3])
        keys = list(result.keys())
        self.assertEqual(keys[4:8], list(SPREAD_KEYS))

    def test_count_is_int(self) -> None:
        result = compute_descriptive_statistics([1, 2, 3])
        self.assertIsInstance(result["count"], int)
        self.assertEqual(result["count"], 3)

    def test_mode_is_list(self) -> None:
        result = compute_descriptive_statistics([1, 2, 3])
        self.assertIsInstance(result["mode"], list)

    def test_numeric_fields_are_float(self) -> None:
        result = compute_descriptive_statistics([1, 2, 3, 4, 5])
        float_keys = [
            "mean",
            "median",
            "sample_standard_deviation",
            "population_standard_deviation",
            "sample_variance",
            "population_variance",
            "min",
            "max",
            "q1",
            "q3",
            "iqr",
            "range",
        ]
        for key in float_keys:
            with self.subTest(key=key):
                self.assertIsInstance(result[key], float)


class TestPrecision(unittest.TestCase):
    def test_mean_keeps_small_values_next_to_large_ones(self) -> None:
        result = compute_descriptive_statistics([1e16, 1, -1e16])
        self.assertAlmostEqual(result["mean"], 1.0 / 3.0, places=15)

    def test_mean_with_cancelling_large_values(self) -> None:
        result = compute_descriptive_statistics([1e16, 1, -1e16, 1])
        self.assertAlmostEqual(result["mean"], 0.5, places=15)

    def test_values_near_float_limit_do_not_overflow(self) -> None:
        result = compute_descriptive_statistics([1e308, 1e308])
        self.assertEqual(result["mean"], 1e308)
        self.assertEqual(result["median"], 1e308)
        self.assertEqual(result["population_variance"], 0.0)
        self.assertEqual(result["sample_variance"], 0.0)

        result = compute_descriptive_statistics([1.5e308, 1.7e308, 1.6e308])
        self.assertAlmostEqual(result["mean"] / 1.6e308, 1.0, places=12)
        # Deviations -1e307, 0, 1e307: the standard deviations stay finite, the variances overflow
        expected_population = math.sqrt(2.0 / 3.0) * 1e307
        self.assertAlmostEqual(result["population_standard_deviation"] / expected_population, 1.0, places=9)
        self.assertAlmostEqual(result["sample_standard_deviation"] / 1e307, 1.0, places=9)
        self.assertEqual(result["population_variance"], math.inf)
        self.assertEqual(result["sample_variance"], math.inf)

    def test_large_offset_variance(self) -> None:
        result = compute_descriptive_statistics([1e9 + 4, 1e9 + 7, 1e9 + 13, 1e9 + 16])
        self.assertAlmostEqual(result["population_variance"], 22.5, places=6)
        self.assertAlmostEqual(result["sample_variance"], 30.0, places=6)
