"""The distribution table (``utils.statistics.distributions.DISTRIBUTIONS``) and its lookup."""

from __future__ import annotations

import math
import unittest

from utils.statistics.distributions import (
    DEFAULT_DISTRIBUTION_TYPE,
    DISTRIBUTIONS,
    SHARED_PARAM_NAMES,
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
        self.assertIs(get_distribution("poisson", default="normal"), DISTRIBUTIONS["poisson"])

    def test_an_empty_type_without_a_default_is_unsupported(self) -> None:
        valid = "; use one of normal, uniform, exponential, t, chi_squared, f, binomial, poisson, geometric"
        cases = [
            (None, "Unsupported distribution_type 'None'" + valid),
            ("", "Unsupported distribution_type ''" + valid),
        ]
        for value, message in cases:
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as caught:
                    get_distribution(value)
                self.assertEqual(str(caught.exception), message)

    def test_unknown_types_keep_the_raw_value_in_the_message(self) -> None:
        with self.assertRaises(ValueError) as caught:
            get_distribution(" Gauss ")
        self.assertTrue(str(caught.exception).startswith("Unsupported distribution_type ' Gauss '; use one of normal,"))

    def test_no_aliases(self) -> None:
        for alias in ("gaussian", "chi2", "chi-squared", "student", "student_t", "exp", "binom"):
            with self.subTest(alias=alias):
                with self.assertRaises(ValueError):
                    get_distribution(alias)
        self.assertIs(get_distribution(" Poisson "), DISTRIBUTIONS["poisson"])


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


class TestDistributionRows(unittest.TestCase):
    """The parameters of every row: defaults, needed keys, checks, ignored keys, strict mode."""

    def test_rows_and_their_keys(self) -> None:
        expected = {
            "normal": (False, ("mean", "sigma")),
            "uniform": (False, ("lower", "upper")),
            "exponential": (False, ("rate",)),
            "t": (False, ("df",)),
            "chi_squared": (False, ("df",)),
            "f": (False, ("df", "df2")),
            "binomial": (True, ("n", "p")),
            "poisson": (True, ("rate",)),
            "geometric": (True, ("p",)),
        }
        self.assertEqual(list(DISTRIBUTIONS), list(expected))
        for name, (discrete, keys) in expected.items():
            spec = DISTRIBUTIONS[name]
            with self.subTest(distribution=name):
                self.assertEqual(spec.name, name)
                self.assertIs(spec.discrete, discrete)
                self.assertEqual(spec.param_names, keys)
                self.assertTrue(set(keys) <= set(SHARED_PARAM_NAMES))
                self.assertEqual(spec.pdf_expression is None, discrete)

    def test_every_shared_key_is_read_by_some_row(self) -> None:
        used = {key for spec in DISTRIBUTIONS.values() for key in spec.param_names}
        self.assertEqual(used, set(SHARED_PARAM_NAMES))

    def test_defaults(self) -> None:
        cases = {
            "uniform": {"lower": 0.0, "upper": 1.0},
            "exponential": {"rate": 1.0},
        }
        for name, want in cases.items():
            with self.subTest(distribution=name):
                self.assertEqual(DISTRIBUTIONS[name].parse_params(None), want)
                self.assertEqual(DISTRIBUTIONS[name].parse_params({key: None for key in SHARED_PARAM_NAMES}), want)

    def test_needed_params_are_named(self) -> None:
        cases = {
            "t": "t needs df (degrees of freedom)",
            "chi_squared": "chi_squared needs df (degrees of freedom)",
            "f": "f needs df (numerator degrees of freedom) and df2 (denominator degrees of freedom)",
            "binomial": "binomial needs n (number of trials) and p (success probability)",
            "poisson": "poisson needs rate (λ)",
            "geometric": "geometric needs p (success probability)",
        }
        for name, message in cases.items():
            for raw in (None, {}, {key: None for key in SHARED_PARAM_NAMES}, {"mean": 0, "sigma": 1}):
                with self.subTest(distribution=name, raw=raw):
                    with self.assertRaises(ValueError) as caught:
                        DISTRIBUTIONS[name].parse_params(raw)
                    self.assertEqual(str(caught.exception), message)
        with self.assertRaises(ValueError) as caught:
            DISTRIBUTIONS["f"].parse_params({"df": 3})
        self.assertEqual(str(caught.exception), "f needs df2 (denominator degrees of freedom)")

    def test_invalid_values(self) -> None:
        cases = [
            ("uniform", {"lower": 2, "upper": 2}, "lower must be less than upper"),
            ("uniform", {"lower": 3, "upper": 1}, "lower must be less than upper"),
            ("exponential", {"rate": 0}, "rate must be > 0"),
            ("t", {"df": -1}, "df must be > 0 and at most 1e+06"),
            ("t", {"df": 2e6}, "df must be > 0 and at most 1e+06"),
            ("chi_squared", {"df": float("inf")}, "df must be finite"),
            ("f", {"df": 1, "df2": 0}, "df2 must be > 0 and at most 1e+06"),
            ("binomial", {"n": 2.5, "p": 0.5}, "n must be a whole number"),
            ("binomial", {"n": 0, "p": 0.5}, "n must be between 1 and 1e+06"),
            ("binomial", {"n": 10, "p": 1}, "p must be between 0 and 1 (exclusive)"),
            ("binomial", {"n": 10, "p": 0}, "p must be between 0 and 1 (exclusive)"),
            ("poisson", {"rate": -2}, "rate must be > 0 and at most 1e+06"),
            ("geometric", {"p": 1.5}, "p must be between 0 and 1 (exclusive)"),
        ]
        for name, raw, message in cases:
            with self.subTest(distribution=name, raw=raw):
                with self.assertRaises(ValueError) as caught:
                    DISTRIBUTIONS[name].parse_params(raw)
                self.assertEqual(str(caught.exception), message)

    def test_parsed_params_keep_only_their_own_keys_in_order(self) -> None:
        raw = {"p": 0.25, "n": 12, "mean": 0, "sigma": 1, "rate": None, "lambda": 3}
        spec = DISTRIBUTIONS["binomial"]
        params = spec.parse_params(raw)
        self.assertEqual(params, {"n": 12.0, "p": 0.25})
        self.assertEqual(list(params), ["n", "p"])
        self.assertEqual(spec.ignored_params(raw), ["mean", "sigma", "lambda"])
        self.assertEqual(spec.ignored_params(None), [])
        self.assertEqual(DISTRIBUTIONS["normal"].ignored_params({"mean": 1, "sigma": 2, "df": None}), [])

    def test_strict_parsing(self) -> None:
        spec = DISTRIBUTIONS["poisson"]
        with self.assertRaises(ValueError):
            spec.parse_params({"rate": None}, strict=True)
        self.assertEqual(spec.parse_params({"rate": 2, "mean": None}, strict=True), {"rate": 2.0})

    def test_default_bounds(self) -> None:
        cases = [
            ("uniform", {"lower": -1, "upper": 3}, (-1.0, 3.0)),
            ("exponential", {"rate": 2}, (0.0, -math.log(0.001) / 2)),
            ("t", {"df": 1}, (-10.0, 10.0)),
            ("t", {"df": 200}, (-4.0, 4.0)),
            ("f", {"df": 1, "df2": 1}, (0.0, 10.0)),
            ("binomial", {"n": 10, "p": 0.3}, (-5.0, 11.0)),
            ("geometric", {"p": 0.5}, (1.0, 10.0)),
        ]
        for name, raw, want in cases:
            spec = DISTRIBUTIONS[name]
            with self.subTest(distribution=name, raw=raw):
                got = spec.default_bounds(spec.parse_params(raw))
                self.assertAlmostEqual(got[0], want[0], places=9)
                self.assertAlmostEqual(got[1], want[1], places=9)
        chi = DISTRIBUTIONS["chi_squared"]
        self.assertEqual(chi.default_bounds({"df": 4.0})[0], 0.0)
        self.assertGreater(chi.default_bounds({"df": 1e4})[0], 9000.0)

    def test_moments(self) -> None:
        cases = [
            ("uniform", {"lower": 0, "upper": 6}, (3.0, 3.0)),
            ("exponential", {"rate": 4}, (0.25, 0.0625)),
            ("t", {"df": 1}, (None, None)),
            ("t", {"df": 2}, (0.0, None)),
            ("t", {"df": 4}, (0.0, 2.0)),
            ("chi_squared", {"df": 3}, (3.0, 6.0)),
            ("f", {"df": 3, "df2": 2}, (None, None)),
            ("f", {"df": 3, "df2": 4}, (2.0, None)),
            ("f", {"df": 3, "df2": 6}, (1.5, 5.25)),
            ("binomial", {"n": 10, "p": 0.5}, (5.0, 2.5)),
            ("poisson", {"rate": 3}, (3.0, 3.0)),
            ("geometric", {"p": 0.25}, (4.0, 12.0)),
        ]
        for name, raw, want in cases:
            spec = DISTRIBUTIONS[name]
            with self.subTest(distribution=name, raw=raw):
                self.assertEqual(spec.moments(spec.parse_params(raw)), want)


if __name__ == "__main__":
    unittest.main()
