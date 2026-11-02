"""The hypothesis_test, confidence_interval and chi_squared_test tools (``utils.statistics.inference_tool``).

Published answers (R's sleep data, the R chisq.test examples, textbook z tests) are the
reference values; every p-value and critical value is also checked against mpmath, an
engine independent of ``special_functions``.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, List, Optional

import mpmath

from server_tests.test_plot_tool_schemas import _find_tool
from static.tool_argument_validator import ToolArgumentValidator
from utils.statistics.chi_squared_tests import CHI_SQUARED_TEST_TYPES
from utils.statistics.confidence_intervals import INTERVAL_NAMES
from utils.statistics.hypothesis_tests import TEST_NAMES
from utils.statistics.inference_samples import SAMPLE_KEYS
from utils.statistics.inference_setup import SAMPLE_KINDS
from utils.statistics.inference_tool import chi_squared_test_tool, confidence_interval_tool, hypothesis_test_tool

mpmath.mp.dps = 40

# R's sleep data: extra hours of sleep under two drugs, the same ten patients.
SLEEP_1 = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0]
SLEEP_2 = [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]


def _data(values: List[float]) -> Dict[str, Any]:
    return {"data": values, "mean": None, "sd": None, "sigma": None, "n": None, "successes": None}


def _summary(**fields: Any) -> Dict[str, Any]:
    sample: Dict[str, Any] = {key: None for key in ("data", "mean", "sd", "sigma", "n", "successes")}
    sample.update(fields)
    return sample


def _test(test_type: str, sample1: Any, sample2: Any = None, h0: Any = None, **kwargs: Any) -> Dict[str, Any]:
    return hypothesis_test_tool(
        test_type,
        sample1,
        sample2,
        h0,
        kwargs.get("alternative"),
        kwargs.get("alpha"),
        kwargs.get("pooled"),
    )


def _interval(
    interval_type: str, sample1: Any, sample2: Any = None, level: Any = None, pooled: Any = None
) -> Dict[str, Any]:
    return confidence_interval_tool(interval_type, sample1, sample2, level, pooled)


def _normal_upper(z: float) -> float:
    return float(mpmath.ncdf(-mpmath.mpf(z)))


def _t_upper_mp(t: Any, df: Any) -> Any:
    t, df = mpmath.mpf(t), mpmath.mpf(df)
    tail = mpmath.betainc(df / 2, mpmath.mpf(1) / 2, 0, df / (df + t * t), regularized=True) / 2
    return tail if t >= 0 else 1 - tail


def _t_upper(t: float, df: float) -> float:
    """P(T > t) for Student's t, from the regularized incomplete beta."""
    return float(_t_upper_mp(t, df))


def _chi_squared_upper(x: float, df: float) -> float:
    return float(mpmath.gammainc(mpmath.mpf(df) / 2, mpmath.mpf(x) / 2, mpmath.inf, regularized=True))


def _t_critical(tail: float, df: float) -> float:
    return float(
        mpmath.findroot(lambda t: _t_upper_mp(t, df) - tail, (mpmath.mpf(0.5), mpmath.mpf(20)), solver="illinois")
    )


class _Close(unittest.TestCase):
    def assertRel(self, actual: Optional[float], expected: float, tol: float = 1e-10) -> None:
        assert actual is not None
        self.assertLessEqual(abs(actual - expected), tol * max(abs(expected), 1e-300), (actual, expected))


class TestMeanTests(_Close):
    def test_z_test_from_summary_matches_the_textbook(self) -> None:
        result = _test("z", _summary(mean=103, sigma=15, n=36), h0=100, alternative="greater")
        self.assertEqual(result["statistic"], 1.2)
        self.assertRel(result["p_value"], _normal_upper(1.2))
        self.assertAlmostEqual(result["p_value"], 0.1151, places=4)
        self.assertFalse(result["reject_null"])
        self.assertEqual(result["null_hypothesis"], "mu = 100")
        self.assertEqual(result["alternative_hypothesis"], "mu > 100")
        self.assertEqual(len(result["critical_values"]), 1)
        self.assertRel(result["critical_values"][0], 1.6448536269514722)
        self.assertNotIn("df", result)
        self.assertTrue(result["conclusion"].startswith("p = 0.1151 > alpha = 0.05: fail to reject H0"))

    def test_z_tails_and_a_tiny_p_value(self) -> None:
        less = _test("z", _summary(mean=97, sigma=15, n=36), h0=100, alternative="less")
        self.assertRel(less["p_value"], _normal_upper(1.2))
        self.assertRel(less["critical_values"][0], -1.6448536269514722)
        two = _test("z", _summary(mean=112, sigma=1, n=1), h0=100)
        self.assertRel(two["p_value"], 2 * _normal_upper(12.0))  # two-sided, about 3.55e-33
        self.assertGreater(two["p_value"], 0.0)
        self.assertEqual(two["critical_values"], [-1.95996398454, 1.95996398454])
        self.assertTrue(two["reject_null"])

    def test_p_value_below_1e_300_is_reported_as_zero(self) -> None:
        result = _test("z", _summary(mean=40, sigma=1, n=1), h0=0)
        self.assertEqual(result["p_value"], 0.0)
        self.assertEqual(result["p_value_note"], "p < 1e-300")
        self.assertTrue(result["conclusion"].startswith("p < 1e-300 <= alpha = 0.05: reject H0"))

    def test_z_test_from_data_needs_sigma_and_ignores_sd(self) -> None:
        result = _test("z", _summary(data=[98, 102, 105, 99], sigma=4, sd=3), h0=100)
        self.assertEqual(result["samples"], [{"name": "sample1", "n": 4, "mean": 101.0, "sigma": 4.0}])
        self.assertEqual(result["statistic"], 0.5)
        self.assertEqual(result["ignored_inputs"], ["sample1.sd"])

    def test_one_sample_t_from_data(self) -> None:
        result = _test("t", _data(SLEEP_1), h0=0)
        mean = sum(SLEEP_1) / 10
        sd = math.sqrt(sum((v - mean) ** 2 for v in SLEEP_1) / 9)
        t = mean / (sd / math.sqrt(10))
        self.assertRel(result["statistic"], t)
        self.assertEqual(result["df"], 9)
        self.assertRel(result["p_value"], 2 * _t_upper(abs(t), 9))
        self.assertRel(result["critical_values"][1], _t_critical(0.025, 9))

    def test_paired_t_matches_r_sleep_data(self) -> None:
        result = _test("paired_t", _data(SLEEP_1), _data(SLEEP_2))
        self.assertAlmostEqual(result["statistic"], -4.0621, places=4)
        self.assertEqual(result["df"], 9)
        self.assertAlmostEqual(result["p_value"], 0.002833, places=6)
        self.assertRel(result["p_value"], 2 * _t_upper(abs(result["statistic"]), 9), 1e-9)
        self.assertEqual(result["null_hypothesis"], "mu_d = 0")
        self.assertEqual(result["samples"][0]["name"], "differences")

    def test_welch_and_pooled_t_match_r_sleep_data(self) -> None:
        welch = _test("two_sample_t", _data(SLEEP_1), _data(SLEEP_2))
        self.assertAlmostEqual(welch["statistic"], -1.8608, places=4)
        self.assertAlmostEqual(welch["df"], 17.776, places=3)
        self.assertIsInstance(welch["df"], float)
        self.assertAlmostEqual(welch["p_value"], 0.07939, places=5)
        self.assertRel(welch["p_value"], 2 * _t_upper(abs(welch["statistic"]), welch["df"]), 1e-9)
        self.assertIn("Welch", welch["method"])
        pooled = _test("two_sample_t", _data(SLEEP_1), _data(SLEEP_2), pooled=True)
        self.assertEqual(pooled["df"], 18)
        self.assertAlmostEqual(pooled["p_value"], 0.07919, places=5)
        self.assertIn("pooled_sd", pooled)

    def test_two_sample_t_from_mixed_data_and_summary(self) -> None:
        result = _test("two_sample_t", _data(SLEEP_1), _summary(mean=2.33, sd=2.002, n=10), h0=0, alternative="less")
        self.assertLess(result["statistic"], 0)
        self.assertRel(result["p_value"], 1 - _t_upper(result["statistic"], result["df"]), 1e-9)

    def test_two_sample_t_with_one_constant_sample_uses_the_other_df(self) -> None:
        result = _test("two_sample_t", _summary(mean=5, sd=0, n=8), _summary(mean=4, sd=2, n=12))
        self.assertAlmostEqual(result["df"], 11.0, places=9)

    def test_two_sample_z(self) -> None:
        result = _test("two_sample_z", _summary(mean=80, sigma=10, n=50), _summary(mean=75, sigma=12, n=60), h0=0)
        z = 5 / math.sqrt(100 / 50 + 144 / 60)
        self.assertRel(result["statistic"], z)
        self.assertRel(result["p_value"], 2 * _normal_upper(z))

    def test_huge_spreads_stay_finite(self) -> None:
        result = _test("two_sample_t", _summary(mean=1e200, sd=1e200, n=10), _summary(mean=0, sd=2e200, n=12))
        self.assertTrue(math.isfinite(result["df"]))
        self.assertTrue(math.isfinite(result["statistic"]))
        z = _test("two_sample_z", _summary(mean=1e200, sigma=1e200, n=4), _summary(mean=0, sigma=1e200, n=4))
        self.assertRel(z["statistic"], math.sqrt(2.0))
        with self.assertRaisesRegex(ValueError, "too large or too small"):
            _test("two_sample_z", _summary(mean=1e308, sigma=1, n=4), _summary(mean=-1e308, sigma=1, n=4))

    def test_large_samples_are_not_capped(self) -> None:
        result = _test("t", _summary(mean=10.01, sd=1, n=2_000_000), h0=10)
        self.assertEqual(result["df"], 1_999_999)
        self.assertRel(result["p_value"], 2 * _t_upper(result["statistic"], 1_999_999), 1e-8)


class TestProportionTests(_Close):
    def test_one_proportion(self) -> None:
        result = _test("one_proportion", _summary(successes=45, n=100), h0=0.5)
        self.assertRel(result["statistic"], -1.0)
        self.assertRel(result["p_value"], 2 * _normal_upper(1.0))
        self.assertEqual(result["samples"][0]["p_hat"], 0.45)
        self.assertNotIn("notes", result)

    def test_one_proportion_notes_small_expected_counts(self) -> None:
        result = _test("one_proportion", _summary(successes=2, n=20), h0=0.1, alternative="greater")
        self.assertIn("n*p0 and n*(1 - p0)", result["notes"][0])

    def test_two_proportion_uses_the_pooled_proportion(self) -> None:
        result = _test("two_proportion", _summary(successes=45, n=100), _summary(successes=30, n=100))
        pooled = 75 / 200
        z = 0.15 / math.sqrt(pooled * (1 - pooled) * (2 / 100))
        self.assertRel(result["statistic"], z)
        self.assertRel(result["p_value"], 2 * _normal_upper(z))
        self.assertEqual(result["pooled_p_hat"], 0.375)
        self.assertEqual(result["null_hypothesis"], "p1 - p2 = 0")

    def test_two_proportion_zero_hypothesized_value_is_accepted(self) -> None:
        result = _test("two_proportion", _summary(successes=45, n=100), _summary(successes=30, n=100), h0=0)
        self.assertEqual(result["null_hypothesis"], "p1 - p2 = 0")


class TestIntervals(_Close):
    def test_z_interval(self) -> None:
        result = _interval("z", _summary(mean=103, sigma=15, n=36))
        self.assertRel(result["lower"], 103 - 1.959963984540054 * 2.5)
        self.assertRel(result["upper"], 103 + 1.959963984540054 * 2.5)
        self.assertEqual(result["confidence_level"], 0.95)
        self.assertEqual(result["parameter"], "mu")

    def test_percentage_level(self) -> None:
        self.assertEqual(_interval("z", _summary(mean=0, sigma=1, n=1), level=90)["confidence_level"], 0.9)

    def test_paired_and_welch_intervals_match_r(self) -> None:
        paired = _interval("paired_t", _data(SLEEP_1), _data(SLEEP_2))
        self.assertAlmostEqual(paired["lower"], -2.4598858, places=6)
        self.assertAlmostEqual(paired["upper"], -0.7001142, places=6)
        welch = _interval("two_sample_t", _data(SLEEP_1), _data(SLEEP_2))
        self.assertAlmostEqual(welch["lower"], -3.3654832, places=6)
        self.assertAlmostEqual(welch["upper"], 0.2054832, places=6)
        pooled = _interval("two_sample_t", _data(SLEEP_1), _data(SLEEP_2), pooled=True)
        self.assertAlmostEqual(pooled["lower"], -3.363874, places=5)
        self.assertAlmostEqual(pooled["upper"], 0.203874, places=5)

    def test_t_interval_critical_value(self) -> None:
        result = _interval("t", _summary(mean=12.4, sd=2.53, n=10), level=0.99)
        self.assertRel(result["critical_value"], _t_critical(0.005, 9))
        self.assertRel(result["margin_of_error"], result["critical_value"] * 2.53 / math.sqrt(10))

    def test_proportion_intervals_are_wald(self) -> None:
        one = _interval("one_proportion", _summary(successes=45, n=100))
        self.assertRel(one["standard_error"], math.sqrt(0.45 * 0.55 / 100))
        two = _interval("two_proportion", _summary(successes=45, n=100), _summary(successes=30, n=100))
        self.assertRel(two["standard_error"], math.sqrt(0.45 * 0.55 / 100 + 0.3 * 0.7 / 100))
        self.assertNotIn("pooled_p_hat", two)

    def test_all_successes_gives_a_point_interval_with_notes(self) -> None:
        result = _interval("one_proportion", _summary(successes=20, n=20))
        self.assertEqual(result["lower"], 1.0)
        self.assertEqual(result["upper"], 1.0)
        self.assertEqual(len(result["notes"]), 2)


class TestChiSquared(_Close):
    def test_goodness_of_fit_matches_r(self) -> None:
        result = chi_squared_test_tool("goodness_of_fit", [89, 37, 30, 28, 2], [0.40, 0.20, 0.20, 0.19, 0.01])
        self.assertAlmostEqual(result["statistic"], 5.7947, places=4)
        self.assertEqual(result["df"], 4)
        self.assertAlmostEqual(result["p_value"], 0.215, places=3)
        self.assertRel(result["p_value"], _chi_squared_upper(result["statistic"], 4))
        self.assertIn("below 5", result["notes"][0])
        self.assertRel(result["critical_value"], 9.487729036781154)

    def test_independence_matches_r(self) -> None:
        result = chi_squared_test_tool("independence", None, None, [[762, 327, 468], [484, 239, 477]])
        self.assertAlmostEqual(result["statistic"], 30.07, places=2)
        self.assertEqual(result["df"], 2)
        self.assertAlmostEqual(result["p_value"], 2.954e-07, delta=1e-10)
        self.assertRel(result["p_value"], _chi_squared_upper(result["statistic"], 2))
        self.assertEqual(len(result["expected"]), 2)
        self.assertNotIn("continuity", result["method"])

    def test_two_by_two_says_no_continuity_correction(self) -> None:
        result = chi_squared_test_tool("independence", None, None, [[10, 20], [30, 40]])
        self.assertIn("no continuity correction", result["method"])

    def test_equal_expected_by_default_and_df_override(self) -> None:
        result = chi_squared_test_tool("goodness_of_fit", [20, 30, 50], None, None, 1)
        self.assertEqual(result["expected"], [33.3333333333] * 3)
        self.assertEqual(result["df"], 1)

    def test_rounded_probabilities_are_rescaled(self) -> None:
        result = chi_squared_test_tool("goodness_of_fit", [30, 30, 40], [0.333, 0.333, 0.333])
        self.assertEqual(result["expected"], [33.3333333333] * 3)
        self.assertIn("rescaled", result["notes"][0])

    def test_rounded_counts_are_used_as_given(self) -> None:
        result = chi_squared_test_tool("goodness_of_fit", [30, 30, 40], [33.3, 33.3, 33.3])
        self.assertEqual(result["expected"], [33.3, 33.3, 33.3])
        self.assertIn("used as given", result["notes"][0])

    def test_large_statistic_keeps_a_small_p_value(self) -> None:
        result = chi_squared_test_tool("goodness_of_fit", [500, 10], None)
        self.assertRel(result["p_value"], _chi_squared_upper(result["statistic"], 1), 1e-9)
        self.assertGreater(result["p_value"], 0.0)

    def test_small_alpha_critical_value(self) -> None:
        result = chi_squared_test_tool("goodness_of_fit", [20, 30, 50], None, None, None, 1e-10)
        self.assertRel(_chi_squared_upper(result["critical_value"], 2), 1e-10, 1e-9)


class TestRefusals(unittest.TestCase):
    def assertRefused(self, pattern: str, call: Any) -> None:
        with self.assertRaisesRegex(ValueError, pattern):
            call()

    def test_sample_refusals(self) -> None:
        self.assertRefused("not both", lambda: _test("t", _summary(data=[1, 2, 3], mean=2), h0=0))
        self.assertRefused("not both", lambda: _test("t", _summary(data=[1, 2, 3], sd=1), h0=0))
        self.assertRefused("n is 4 but sample1.data has 3", lambda: _test("t", _summary(data=[1, 2, 3], n=4), h0=0))
        self.assertEqual(_test("t", _summary(data=[1, 2, 4], n=3), h0=0)["samples"][0]["n"], 3)
        self.assertRefused("use t", lambda: _test("z", _summary(mean=1, sd=1, n=5), h0=0))
        self.assertRefused("needs sample1.sigma", lambda: _test("z", _summary(mean=1, n=5), h0=0))
        self.assertRefused("means a z test", lambda: _test("t", _summary(mean=1, sd=1, sigma=1, n=5), h0=0))
        self.assertRefused("whole number", lambda: _test("t", _summary(mean=1, sd=1, n=5.5), h0=0))
        self.assertRefused("sd is needed", lambda: _test("t", _summary(mean=1, n=5), h0=0))
        self.assertRefused("at least 2", lambda: _test("t", _data([4.0]), h0=0))
        self.assertRefused("no spread", lambda: _test("t", _data([4.0, 4.0, 4.0]), h0=0))
        self.assertRefused(
            "both samples have no spread",
            lambda: _test("two_sample_t", _summary(mean=1, sd=0, n=3), _summary(mean=2, sd=0, n=3)),
        )
        self.assertRefused("sample1 is needed", lambda: _test("t", None, h0=0))
        self.assertRefused("sample2 is needed", lambda: _test("two_sample_t", _data([1, 2, 3])))
        self.assertRefused("unknown keys", lambda: _test("t", {"data": [1, 2], "values": [3]}, h0=0))

    def test_foreign_groups_are_refused(self) -> None:
        self.assertRefused(
            "use two_sample_t, or paired_t", lambda: _test("t", _data([1, 2, 3]), _data([4, 5, 6]), h0=0)
        )
        self.assertRefused(
            "use two_proportion",
            lambda: _test("one_proportion", _summary(successes=4, n=10), _summary(successes=5, n=10), h0=0.5),
        )
        self.assertRefused("proportion tests", lambda: _test("t", _summary(successes=3, n=10), h0=0))
        self.assertRefused(
            "belong to tests of a mean", lambda: _test("one_proportion", _summary(mean=0.4, n=10), h0=0.5)
        )
        self.assertRefused("successes and n", lambda: _test("one_proportion", _summary(n=10), h0=0.5))

    def test_proportion_and_paired_refusals(self) -> None:
        self.assertRefused("cannot exceed", lambda: _test("one_proportion", _summary(successes=11, n=10), h0=0.5))
        self.assertRefused(
            "strictly between 0 and 1", lambda: _test("one_proportion", _summary(successes=4, n=10), h0=1.2)
        )
        self.assertRefused(
            "p1 = p2 only",
            lambda: _test("two_proportion", _summary(successes=4, n=10), _summary(successes=5, n=10), h0=0.1),
        )
        self.assertRefused(
            "all successes or all failures",
            lambda: _test("two_proportion", _summary(successes=10, n=10), _summary(successes=8, n=8)),
        )
        self.assertRefused("equal length", lambda: _test("paired_t", _data([1, 2, 3]), _data([1, 2])))
        self.assertRefused("both data lists", lambda: _test("paired_t", _summary(mean=1, sd=1, n=5), _data([1, 2])))
        self.assertRefused("same difference", lambda: _test("paired_t", _data([1, 2, 3]), _data([0, 1, 2])))
        self.assertRefused("mu0", lambda: _test("t", _data([1, 2, 3])))
        self.assertRefused("p0", lambda: _test("one_proportion", _summary(successes=4, n=10)))

    def test_level_and_alpha_refusals(self) -> None:
        sample = _summary(mean=1, sigma=1, n=4)
        self.assertRefused("confidence level", lambda: _test("z", sample, h0=0, alpha=0.95))
        self.assertRefused("write 0.05", lambda: _test("z", sample, h0=0, alpha=5))
        self.assertRefused("alpha must be > 0", lambda: _test("z", sample, h0=0, alpha=0))
        self.assertRefused("exclusive", lambda: _interval("z", sample, level=100))
        self.assertRefused("exclusive", lambda: _interval("z", sample, level=1))
        self.assertRefused("alternative must be", lambda: _test("z", sample, h0=0, alternative="two-tailed"))
        self.assertRefused("Unsupported test_type", lambda: _test("anova", sample))
        self.assertRefused("Unsupported interval_type", lambda: _interval("chi_squared_gof", sample))

    def test_pooled_outside_two_sample_t_is_listed(self) -> None:
        result = _test("z", _summary(mean=1, sigma=1, n=4), h0=0, pooled=True)
        self.assertEqual(result["ignored_inputs"], ["pooled"])

    def test_chi_squared_refusals(self) -> None:
        self.assertRefused(
            "looks like percentages", lambda: chi_squared_test_tool("goodness_of_fit", [20, 30, 40], [25, 25, 50])
        )
        self.assertRefused(
            "give counts summing to 90", lambda: chi_squared_test_tool("goodness_of_fit", [20, 30, 40], [20, 20, 20])
        )
        self.assertRefused(
            "probabilities or counts",
            lambda: chi_squared_test_tool("goodness_of_fit", [0.2, 0.3, 0.5], [0.3, 0.3, 0.4]),
        )
        self.assertRefused(
            "one per category", lambda: chi_squared_test_tool("goodness_of_fit", [20, 30, 40], [0.5, 0.5])
        )
        self.assertRefused(
            "must be > 0", lambda: chi_squared_test_tool("goodness_of_fit", [20, 30, 40], [0.5, 0.5, 0.0])
        )
        self.assertRefused("at least 2 categories", lambda: chi_squared_test_tool("goodness_of_fit", [20], None))
        self.assertRefused("must be >= 0", lambda: chi_squared_test_tool("goodness_of_fit", [20, -1], None))
        self.assertRefused("at most 2", lambda: chi_squared_test_tool("goodness_of_fit", [20, 30, 40], None, None, 3))
        self.assertRefused(
            "table is for test_type independence",
            lambda: chi_squared_test_tool("goodness_of_fit", [1, 2], None, [[1, 2], [3, 4]]),
        )
        self.assertRefused(
            "observed is for test_type goodness_of_fit",
            lambda: chi_squared_test_tool("independence", [1, 2], None, [[1, 2], [3, 4]]),
        )
        self.assertRefused(
            "same number", lambda: chi_squared_test_tool("independence", None, None, [[1, 2], [3, 4, 5]])
        )
        self.assertRefused("row 2", lambda: chi_squared_test_tool("independence", None, None, [[1, 2], [0, 0]]))
        self.assertRefused("column 1", lambda: chi_squared_test_tool("independence", None, None, [[0, 2], [0, 4]]))
        self.assertRefused("needs table", lambda: chi_squared_test_tool("independence", None, None, None))
        self.assertRefused("needs observed", lambda: chi_squared_test_tool("goodness_of_fit", None, None, None))


class TestReviewRoundOne(_Close):
    def test_rounded_thirds_are_probabilities(self) -> None:
        result = chi_squared_test_tool("goodness_of_fit", [30, 30, 40], [0.33, 0.33, 0.33])
        self.assertEqual(result["expected"], [33.3333333333] * 3)

    def test_empty_lists_count_as_null(self) -> None:
        sample = _summary(data=[], mean=5.5, sd=1, n=10)
        self.assertEqual(_test("t", sample, _summary(data=[]), h0=5)["df"], 9)
        result = chi_squared_test_tool("independence", [], [], [[10, 20], [30, 40]])
        self.assertEqual(result["df"], 1)
        self.assertEqual(chi_squared_test_tool("goodness_of_fit", [20, 30, 50], [], [])["df"], 2)

    def test_matching_df_is_accepted_on_independence(self) -> None:
        self.assertEqual(chi_squared_test_tool("independence", None, None, [[10, 20], [30, 40]], 1)["df"], 1)
        with self.assertRaisesRegex(ValueError, r"\(rows - 1\)\(columns - 1\) = 1"):
            chi_squared_test_tool("independence", None, None, [[10, 20], [30, 40]], 2)

    def test_z_with_only_sd_names_both_readings(self) -> None:
        with self.assertRaisesRegex(ValueError, "send it as sample1.sigma.*use t"):
            _test("z", _summary(mean=105, sd=15, n=36), h0=100)

    def test_conclusion_never_prints_p_equal_to_alpha(self) -> None:
        result = _test("z", _summary(mean=1.95996, sigma=1, n=1), h0=0)
        self.assertFalse(result["reject_null"])
        self.assertTrue(result["conclusion"].startswith("p = 0.050000"), result["conclusion"])
        self.assertNotIn("p = 0.05 >", result["conclusion"])

    def test_proportion_hints(self) -> None:
        with self.assertRaisesRegex(ValueError, "send 90"):
            _test("one_proportion", _summary(successes=0.45, n=200), h0=0.5)
        with self.assertRaisesRegex(ValueError, "write 0.5 for 50%"):
            _test("one_proportion", _summary(successes=90, n=200), h0=50)

    def test_a_fractional_count_asks_for_the_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "ask for the count"):
            _test("one_proportion", _summary(successes=0.453, n=200), h0=0.5)

    def test_the_schema_states_the_paired_direction(self) -> None:
        props = _find_tool("hypothesis_test")["function"]["parameters"]["properties"]
        self.assertIn("sample1 - sample2", props["test_type"]["description"])
        self.assertIn("after as sample1 with greater", props["alternative"]["description"])

    def test_pooled_false_is_not_listed(self) -> None:
        self.assertNotIn("ignored_inputs", _test("z", _summary(mean=1, sigma=1, n=4), h0=0, pooled=False))

    def test_levels_below_half_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "below 50%"):
            _interval("z", _summary(mean=0, sigma=1, n=1), level=1.5)
        with self.assertRaisesRegex(ValueError, "below 50%"):
            _interval("z", _summary(mean=0, sigma=1, n=1), level=0.05)

    def test_ratio_hint(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide each by their sum"):
            chi_squared_test_tool("goodness_of_fit", [315, 108, 101, 32], [9, 3, 3, 1])


class TestSchemas(unittest.TestCase):
    def _props(self, name: str) -> Dict[str, Any]:
        props: Dict[str, Any] = _find_tool(name)["function"]["parameters"]["properties"]
        return props

    def test_enums_follow_the_dispatch_tables(self) -> None:
        self.assertEqual(self._props("hypothesis_test")["test_type"]["enum"], list(TEST_NAMES))
        self.assertEqual(self._props("confidence_interval")["interval_type"]["enum"], list(INTERVAL_NAMES))
        self.assertEqual(list(TEST_NAMES), list(SAMPLE_KINDS))
        self.assertEqual(list(INTERVAL_NAMES), list(SAMPLE_KINDS))
        self.assertEqual(self._props("chi_squared_test")["test_type"]["enum"], list(CHI_SQUARED_TEST_TYPES))

    def test_the_sample_object_is_the_sample_keys(self) -> None:
        for tool in ("hypothesis_test", "confidence_interval"):
            for sample in ("sample1", "sample2"):
                self.assertEqual(self._props(tool)[sample]["required"], list(SAMPLE_KEYS))

    def test_omitted_nested_keys_are_filled_with_null(self) -> None:
        result = ToolArgumentValidator.validate(
            "hypothesis_test",
            {"test_type": "t", "sample1": {"data": [1, 2, 4]}, "hypothesized_value": 0},
        )
        self.assertTrue(result["valid"], result["errors"])
        arguments = result["arguments"]
        self.assertEqual(
            arguments["sample1"],
            {"data": [1, 2, 4], "mean": None, "sd": None, "sigma": None, "n": None, "successes": None},
        )
        self.assertIsNone(arguments["sample2"])
        self.assertEqual(hypothesis_test_tool(**arguments)["df"], 2)

    def test_a_table_validates(self) -> None:
        result = ToolArgumentValidator.validate(
            "chi_squared_test", {"test_type": "independence", "table": [[10, 20], [30, 40]]}
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(chi_squared_test_tool(**result["arguments"])["df"], 1)


if __name__ == "__main__":
    unittest.main()
