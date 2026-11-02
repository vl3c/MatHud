"""The inference tools under Brython's float and math layer (fsum, hypot, tiny tails).

The full suite runs under pytest (server_tests/test_inference_tools.py); this checks the
same modules import and give the same numbers in the browser.
"""

from __future__ import annotations

import unittest

from utils.statistics.inference_tool import chi_squared_test_tool, confidence_interval_tool, hypothesis_test_tool

SLEEP_1 = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0]
SLEEP_2 = [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]


def _sample(**fields: object) -> dict:
    sample: dict = {key: None for key in ("data", "mean", "sd", "sigma", "n", "successes")}
    sample.update(fields)
    return sample


class TestInferenceToolsInBrython(unittest.TestCase):
    def assert_relative(self, got: float, want: float, tolerance: float = 1e-9) -> None:
        self.assertLessEqual(abs(got - want) / abs(want), tolerance, f"got {got!r}, want {want!r}")

    def test_welch_t_test_matches_r(self) -> None:
        result = hypothesis_test_tool("two_sample_t", _sample(data=SLEEP_1), _sample(data=SLEEP_2))
        self.assert_relative(result["df"], 17.7764735161, 1e-9)
        self.assert_relative(result["p_value"], 0.0793941401, 1e-8)
        self.assertFalse(result["reject_null"])

    def test_a_tiny_p_value_keeps_its_digits(self) -> None:
        result = hypothesis_test_tool("z", _sample(mean=112, sigma=1, n=1), None, 100)
        self.assert_relative(result["p_value"], 3.55296422416e-33)
        self.assertEqual(hypothesis_test_tool("z", _sample(mean=40, sigma=1, n=1), None, 0)["p_value"], 0.0)

    def test_paired_interval_matches_r(self) -> None:
        result = confidence_interval_tool("paired_t", _sample(data=SLEEP_1), _sample(data=SLEEP_2))
        self.assertAlmostEqual(result["lower"], -2.4598858, places=6)
        self.assertAlmostEqual(result["upper"], -0.7001142, places=6)

    def test_chi_squared_independence_matches_r(self) -> None:
        result = chi_squared_test_tool("independence", None, None, [[762, 327, 468], [484, 239, 477]])
        self.assertEqual(result["df"], 2)
        self.assert_relative(result["p_value"], 2.95358918e-07, 1e-6)

    def test_refusals_are_value_errors(self) -> None:
        with self.assertRaises(ValueError):
            hypothesis_test_tool("t", _sample(data=[1, 2, 3]), _sample(data=[4, 5, 6]), 0)
