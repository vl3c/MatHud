"""The distribution table (``utils.statistics.distributions.DISTRIBUTIONS``) and its lookup."""

from __future__ import annotations

import math
import unittest

from utils.statistics.distributions import (
    DEFAULT_DISTRIBUTION_TYPE,
    DISTRIBUTIONS,
    default_normal_bounds,
    get_distribution,
    normal_pdf,
    normal_pdf_expression,
)


class TestGetDistribution(unittest.TestCase):
    def test_names_match_without_case_or_spaces(self) -> None:
        for name in ("normal", " Normal ", "NORMAL"):
            with self.subTest(name=name):
                self.assertIs(get_distribution(name), DISTRIBUTIONS["normal"])

    def test_the_default_applies_only_to_an_empty_type(self) -> None:
        self.assertEqual(DEFAULT_DISTRIBUTION_TYPE, "normal")
        for empty in (None, "", "  "):
            with self.subTest(value=empty):
                self.assertIs(get_distribution(empty, default="normal"), DISTRIBUTIONS["normal"])
        with self.assertRaises(ValueError) as caught:
            get_distribution("poisson", default="normal")
        self.assertEqual(str(caught.exception), "Unsupported distribution_type 'poisson'")

    def test_an_empty_type_without_a_default_is_unsupported(self) -> None:
        cases = [(None, "Unsupported distribution_type 'None'"), ("", "Unsupported distribution_type ''")]
        for value, message in cases:
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as caught:
                    get_distribution(value)
                self.assertEqual(str(caught.exception), message)

    def test_unknown_types_keep_the_raw_value_in_the_message(self) -> None:
        with self.assertRaises(ValueError) as caught:
            get_distribution(" Poisson ")
        self.assertEqual(str(caught.exception), "Unsupported distribution_type ' Poisson '")


class TestNormalSpec(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = DISTRIBUTIONS["normal"]

    def test_name(self) -> None:
        self.assertEqual(self.spec.name, "normal")

    def test_params_default_and_keep_their_order(self) -> None:
        cases = [
            (None, {"mean": 0.0, "sigma": 1.0}),
            ([1, 2], {"mean": 0.0, "sigma": 1.0}),
            ({}, {"mean": 0.0, "sigma": 1.0}),
            ({"mean": None, "sigma": None}, {"mean": 0.0, "sigma": 1.0}),
            ({"sigma": 2, "mean": 1}, {"mean": 1.0, "sigma": 2.0}),
            ({"mean": "1.5", "sigma": "0.5", "other": 3}, {"mean": 1.5, "sigma": 0.5}),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                params = self.spec.parse_params(raw)
                self.assertEqual(params, expected)
                self.assertEqual(list(params), ["mean", "sigma"])
                self.assertTrue(all(type(value) is float for value in params.values()))

    def test_invalid_params_keep_their_messages(self) -> None:
        cases = [
            ({"mean": float("nan")}, ValueError, "mean must be finite"),
            ({"mean": float("inf")}, ValueError, "mean must be finite"),
            ({"sigma": float("nan")}, ValueError, "sigma must be finite"),
            ({"sigma": 0}, ValueError, "sigma must be > 0"),
            ({"sigma": -1}, ValueError, "sigma must be > 0"),
            ({"mean": "abc"}, ValueError, None),
            ({"mean": [1]}, TypeError, None),
        ]
        for raw, error, message in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(error) as caught:
                    self.spec.parse_params(raw)
                if message is not None:
                    self.assertEqual(str(caught.exception), message)

    def test_strict_parsing_rejects_present_null_params(self) -> None:
        for raw in ({"mean": None}, {"sigma": None}, {"mean": 0, "sigma": None}):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    self.spec.parse_params(raw, strict=True)
        self.assertEqual(self.spec.parse_params({}, strict=True), {"mean": 0.0, "sigma": 1.0})
        self.assertEqual(self.spec.parse_params(None, strict=True), {"mean": 0.0, "sigma": 1.0})
        self.assertEqual(self.spec.parse_params({"mean": 1, "x": None}, strict=True), {"mean": 1.0, "sigma": 1.0})

    def test_functions_match_the_normal_functions(self) -> None:
        for mean, sigma in ((0.0, 1.0), (1.25, 2.5), (-3.0, 0.1), (1e-05, 7.0)):
            params = {"mean": mean, "sigma": sigma}
            with self.subTest(params=params):
                self.assertEqual(self.spec.pdf_expression(params), normal_pdf_expression(mean, sigma))
                self.assertEqual(self.spec.default_bounds(params), default_normal_bounds(mean, sigma, k=4.0))
                for x in (-10.0, -1.0, 0.0, 0.3, mean, 4.0):
                    self.assertEqual(self.spec.pdf(x, params), normal_pdf(x, mean, sigma))
        self.assertTrue(math.isclose(self.spec.pdf(0.0, {"mean": 0.0, "sigma": 1.0}), 1 / math.sqrt(2 * math.pi)))


if __name__ == "__main__":
    unittest.main()
