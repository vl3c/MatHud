from __future__ import annotations

import unittest

from utils.statistics.descriptive import (
    compute_descriptive_statistics,
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

    def test_single_element_no_mode(self) -> None:
        result = compute_descriptive_statistics([42])
        self.assertEqual(result["mode"], [])

    def test_two_identical_elements(self) -> None:
        result = compute_descriptive_statistics([7, 7])
        self.assertEqual(result["mode"], [7.0])

    def test_mode_elements_are_float(self) -> None:
        result = compute_descriptive_statistics([1, 2, 2, 3])
        for m in result["mode"]:
            self.assertIsInstance(m, float)


class TestVarianceAndStdDev(unittest.TestCase):
    def test_known_dataset(self) -> None:
        # Population variance of [2,4,4,4,5,5,7,9] = 4.0, std dev = 2.0
        result = compute_descriptive_statistics([2, 4, 4, 4, 5, 5, 7, 9])
        self.assertAlmostEqual(result["variance"], 4.0)
        self.assertAlmostEqual(result["standard_deviation"], 2.0)

    def test_all_same_values(self) -> None:
        result = compute_descriptive_statistics([3, 3, 3])
        self.assertAlmostEqual(result["variance"], 0.0)
        self.assertAlmostEqual(result["standard_deviation"], 0.0)

    def test_population_formula(self) -> None:
        # [2, 8]: mean=5, pop variance = ((2-5)^2 + (8-5)^2) / 2 = 9.0
        result = compute_descriptive_statistics([2, 8])
        self.assertAlmostEqual(result["variance"], 9.0)

    def test_numerical_robustness(self) -> None:
        result = compute_descriptive_statistics([1.0, 1.0, 1.0000000000000002])
        self.assertGreaterEqual(result["variance"], 0.0)
        self.assertGreaterEqual(result["standard_deviation"], 0.0)


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
            "standard_deviation",
            "variance",
            "min",
            "max",
            "q1",
            "q3",
            "iqr",
            "range",
        }
        self.assertEqual(set(result.keys()), expected_keys)

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
            "standard_deviation",
            "variance",
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
