"""The compute_distribution tool (``utils.statistics.distribution_tool``)."""

from __future__ import annotations

import unittest
from typing import Any, Dict

from static.tool_argument_validator import ToolArgumentValidator
from utils.statistics.distribution_tool import compute_distribution_tool, round_significant


def _run(distribution_type: str, params: Any = None, **kwargs: Any) -> Dict[str, Any]:
    return compute_distribution_tool(distribution_type, params, **kwargs)


class TestComputeDistribution(unittest.TestCase):
    def test_binomial_gives_every_inequality(self) -> None:
        result = _run("binomial", {"n": 10, "p": 0.3}, x=3, left_bound=2, right_bound=5, probability=0.9)
        expected = {
            "distribution": "binomial",
            "params": {"n": 10.0, "p": 0.3},
            "mean": 3.0,
            "variance": 2.1,
            "standard_deviation": 1.44913767462,
            "support": "0, 1, ..., 10",
            "x": 3.0,
            "P(X = x)": 0.266827932,
            "P(X <= x)": 0.6496107184,
            "P(X < x)": 0.3827827864,
            "P(X >= x)": 0.6172172136,
            "P(X > x)": 0.3503892816,
            "a": 2.0,
            "b": 5.0,
            "P(a <= X <= b)": 0.8033426667,
            "P(a < X < b)": 0.466948881,
            "probability": 0.9,
            "quantile": 5.0,
            "quantile_note": "the smallest value k with P(X <= k) >= probability",
        }
        self.assertEqual(result, expected)
        self.assertEqual(list(result), list(expected))

    def test_continuous_results(self) -> None:
        result = _run("t", {"df": 15}, x=2.1, probability=0.975)
        self.assertEqual(result["support"], "(-inf, inf)")
        self.assertAlmostEqual(result["P(X > x)"], 0.026527628076, places=12)
        self.assertAlmostEqual(result["quantile"], 2.13144954556, places=10)
        self.assertNotIn("P(X = x)", result)
        normal = _run("normal", {"mean": 100, "sigma": 15}, left_bound=85, right_bound=115)
        self.assertAlmostEqual(normal["P(a <= X <= b)"], 0.682689492137, places=12)
        self.assertIn("interval_note", normal)

    def test_open_interval_sides(self) -> None:
        upper = _run("normal", None, left_bound=8)
        self.assertEqual((upper["a"], upper["b"]), (8.0, None))
        self.assertAlmostEqual(upper["P(a <= X <= b)"] / 6.22096057427e-16, 1.0, places=9)
        lower = _run("poisson", {"rate": 4}, right_bound=2)
        self.assertAlmostEqual(lower["P(a <= X <= b)"], 0.238103305554, places=12)

    def test_discrete_values_between_integers(self) -> None:
        result = _run("poisson", {"rate": 2}, x=2.5)
        self.assertEqual(result["P(X = x)"], 0.0)
        self.assertEqual(result["P(X <= x)"], result["P(X < x)"])
        self.assertEqual(result["support"], "0, 1, 2, ...")
        self.assertEqual(_run("geometric", {"p": 0.5})["support"], "1, 2, 3, ...")

    def test_moments_only_and_undefined_moments(self) -> None:
        result = _run("f", {"df": 3, "df2": 2})
        self.assertEqual((result["mean"], result["variance"], result["standard_deviation"]), (None, None, None))
        self.assertIn("df2 <= 2", result["moments_note"])
        cauchy = _run("t", {"df": 1})
        self.assertIn("df <= 1", cauchy["moments_note"])
        self.assertIn("infinite", _run("t", {"df": 2})["moments_note"])

    def test_ignored_params_are_named(self) -> None:
        result = _run("exponential", {"rate": 2, "mean": 0, "sigma": 1, "df": None})
        self.assertEqual(result["ignored_params"], ["mean", "sigma"])

    def test_invalid_input(self) -> None:
        cases = [
            (("normal", None), {"probability": 1.0}, "probability must be between 0 and 1 (exclusive)"),
            (("normal", None), {"probability": 0}, "probability must be between 0 and 1 (exclusive)"),
            (("normal", None), {"left_bound": 2, "right_bound": 1}, "left_bound must be <= right_bound"),
            (("normal", None), {"x": float("nan")}, "x must be finite"),
            (("binomial", {"n": 10}), {}, "binomial needs p (success probability)"),
        ]
        for args, kwargs, message in cases:
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(ValueError) as caught:
                    _run(*args, **kwargs)
                self.assertEqual(str(caught.exception), message)

    def test_round_significant(self) -> None:
        self.assertEqual(round_significant(0.30000000000000004), 0.3)
        self.assertEqual(round_significant(-6.95729106167942e-17), -6.95729106168e-17)
        self.assertIsNone(round_significant(None))
        self.assertEqual(round_significant(0.0), 0.0)


class TestNestedNullFill(unittest.TestCase):
    def test_omitted_nested_nullable_keys_are_filled(self) -> None:
        result = ToolArgumentValidator.validate(
            "compute_distribution",
            {"distribution_type": "normal", "distribution_params": {"mean": 0, "sigma": 1}, "x": 1},
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(
            result["arguments"]["distribution_params"],
            {
                "mean": 0,
                "sigma": 1,
                "lower": None,
                "upper": None,
                "rate": None,
                "df": None,
                "df2": None,
                "n": None,
                "p": None,
            },
        )
        self.assertIsNone(result["arguments"]["probability"])

    def test_unknown_nested_keys_still_fail(self) -> None:
        result = ToolArgumentValidator.validate(
            "compute_distribution",
            {"distribution_type": "poisson", "distribution_params": {"lambda": 3}},
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("distribution_params.lambda" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
