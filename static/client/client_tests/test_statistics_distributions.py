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

    def test_default_normal_bounds(self) -> None:
        left, right = default_normal_bounds(0.0, 1.0)
        self.assertAlmostEqual(left, -4.0)
        self.assertAlmostEqual(right, 4.0)

        left2, right2 = default_normal_bounds(2.0, 0.5, k=2.0)
        self.assertAlmostEqual(left2, 1.0)
        self.assertAlmostEqual(right2, 3.0)



