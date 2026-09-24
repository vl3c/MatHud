from __future__ import annotations

import math
import unittest

from utils.statistics.distributions import normal_pdf_expression


def _evaluate_pdf_expression(expression: str, x: float) -> float:
    namespace = {"x": x, "sqrt": math.sqrt, "exp": math.exp, "pi": math.pi}
    return float(eval(expression.replace("^", "**"), {"__builtins__": {}}, namespace))


class TestNormalPdfExpressionNumberFormat(unittest.TestCase):
    def test_small_and_large_parameters_avoid_scientific_notation(self) -> None:
        for mean, sigma in ((0.0, 1e-5), (-2.5e-7, 3e-6), (1e16, 1e20), (1.25, 2.5)):
            with self.subTest(mean=mean, sigma=sigma):
                expr = normal_pdf_expression(mean, sigma)
                self.assertNotIn("e-", expr.replace("exp", ""))
                self.assertNotIn("e+", expr)

    def test_small_sigma_expression_values(self) -> None:
        sigma = 1e-5
        expr = normal_pdf_expression(0.0, sigma)
        self.assertIn("(0.00001)", expr)
        peak = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        self.assertAlmostEqual(_evaluate_pdf_expression(expr, 0.0) / peak, 1.0, places=12)
        self.assertAlmostEqual(_evaluate_pdf_expression(expr, sigma) / (peak * math.exp(-0.5)), 1.0, places=12)

    def test_parameters_round_trip_exactly(self) -> None:
        for value in (1e-5, 3e-6, 1.2345678901234567e-8, 0.1, 2.5, 123456.789, 1e20, -4.5e-9):
            with self.subTest(value=value):
                expr = normal_pdf_expression(value, 1.0)
                mean_text = expr.split("(x-(")[1].split("))")[0]
                self.assertEqual(float(mean_text), value)

    def test_regular_values_keep_their_previous_text(self) -> None:
        expr = normal_pdf_expression(1.25, 2.5)
        self.assertEqual(expr, "(1/((2.5)*sqrt(2*pi)))*exp(-(((x-(1.25))^2)/(2*(2.5)^2)))")


if __name__ == "__main__":
    unittest.main()
