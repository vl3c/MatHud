from __future__ import annotations

import unittest

from utils.statistics.distributions import default_normal_bounds, normal_pdf_expression


class TestStatisticsDistributions(unittest.TestCase):
    def test_normal_pdf_expression_contains_expected_tokens(self) -> None:
        expr = normal_pdf_expression(0.0, 1.0)
        self.assertIn("exp", expr)
        self.assertIn("sqrt", expr)
        self.assertIn("pi", expr)
        self.assertIn("^", expr)

    def test_normal_pdf_expression_rejects_invalid_sigma(self) -> None:
        with self.assertRaises(ValueError):
            normal_pdf_expression(0.0, 0.0)

        with self.assertRaises(ValueError):
            normal_pdf_expression(0.0, -1.0)

    def test_normal_pdf_expression_avoids_scientific_notation(self) -> None:
        expr = normal_pdf_expression(-2.5e-7, 1e-5)
        self.assertIn("(0.00001)", expr)
        self.assertIn("(-0.00000025)", expr)
        self.assertNotIn("e-", expr.replace("exp", ""))

    def test_default_normal_bounds(self) -> None:
        left, right = default_normal_bounds(0.0, 1.0)
        self.assertAlmostEqual(left, -4.0)
        self.assertAlmostEqual(right, 4.0)

        left2, right2 = default_normal_bounds(2.0, 0.5, k=2.0)
        self.assertAlmostEqual(left2, 1.0)
        self.assertAlmostEqual(right2, 3.0)


class TestDistributionNumericsInBrython(unittest.TestCase):
    """The special functions under Brython's float and math, against values from mpmath."""

    def assert_relative(self, got: float, want: float, tolerance: float = 1e-10) -> None:
        self.assertLessEqual(abs(got - want) / abs(want), tolerance, f"got {got!r}, want {want!r}")

    def test_tails_and_quantiles(self) -> None:
        from utils.statistics.distributions import DISTRIBUTIONS

        cases = [
            ("normal", {}, 5.0, 2.866515718791939e-07),
            ("t", {"df": 15.0}, 2.1, 0.026527628076021367),
            ("chi_squared", {"df": 3.0}, 30.0, 1.3800570312932547e-06),
            ("f", {"df": 4.0, "df2": 9.0}, 7.5, 0.006078342979072856),
            ("poisson", {"rate": 4.0}, 12.0, 0.000273716822855503),
            ("binomial", {"n": 40.0, "p": 0.3}, 20.0, 0.0024193598971335934),
        ]
        for name, params, x, upper in cases:
            spec = DISTRIBUTIONS[name]
            parsed = spec.parse_params(params)
            with self.subTest(distribution=name):
                self.assert_relative(spec.sf(x, parsed), upper, 1e-9)
        normal = DISTRIBUTIONS["normal"]
        self.assert_relative(normal.quantile(0.975, normal.parse_params({})), 1.959963984540054)
        t = DISTRIBUTIONS["t"]
        self.assert_relative(t.quantile(0.995, t.parse_params({"df": 1.0})), 63.65674116287399)
        binomial = DISTRIBUTIONS["binomial"]
        self.assertEqual(binomial.quantile(0.9, binomial.parse_params({"n": 10, "p": 0.3})), 5.0)
