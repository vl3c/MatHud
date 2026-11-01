"""Hypothesis tests of means and proportions: z, t, two-sample, paired and proportion tests.

``run_hypothesis_test`` returns the statistic, its df, the p-value for the chosen
alternative, the critical values at alpha, the decision and a one-line conclusion, like a
calculator's Z-Test, T-Test, 2-SampTTest, 1-PropZTest and 2-PropZTest screens. H0 is
rejected when p <= alpha, decided on the unrounded p.

The shared helpers (p-values, critical values, conclusions, rounding) serve the interval
and χ² modules too. Tails come from the ``DISTRIBUTIONS`` specs, called with plain
parameter dicts (``parse_params`` caps df at 1e6). Pure Python with no ``browser`` import.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from utils.statistics.distribution_tool import round_significant
from utils.statistics.distributions import DistributionSpec, get_distribution
from utils.statistics.inference_samples import finite_number, parse_alpha
from utils.statistics.inference_setup import Setup, setup_for

ALTERNATIVES = ("two_sided", "less", "greater")
_RELATION = {"two_sided": "!=", "less": "<", "greater": ">"}
# Smaller p-values are subnormal floats with few correct digits; they are reported as 0.
SMALLEST_P_VALUE = 1e-300
STANDARD_NORMAL = {"mean": 0.0, "sigma": 1.0}


def reference_distribution(df: Optional[float]) -> Tuple[DistributionSpec, Dict[str, float]]:
    """The standard normal (df None) or Student's t with ``df`` degrees of freedom."""
    if df is None:
        return get_distribution("normal"), dict(STANDARD_NORMAL)
    return get_distribution("t"), {"df": df}


def p_value(spec: DistributionSpec, params: Dict[str, float], statistic: float, alternative: str) -> float:
    """The p-value; two-sided doubles the smaller tail (z and t are symmetric). Tails use sf, not 1 - cdf."""
    lower = float(spec.cdf(statistic, params))
    upper = float(spec.sf(statistic, params))
    if alternative == "less":
        return lower
    if alternative == "greater":
        return upper
    return min(1.0, 2.0 * min(lower, upper))


def symmetric_critical(spec: DistributionSpec, params: Dict[str, float], tail: float) -> float:
    """The positive value with P(X > value) = tail, from the lower tail for full precision."""
    return -float(spec.quantile(tail, params))


def critical_values(spec: DistributionSpec, params: Dict[str, float], alpha: float, alternative: str) -> List[float]:
    if alternative == "two_sided":
        upper = symmetric_critical(spec, params, alpha / 2.0)
        return [-upper, upper]
    if alternative == "less":
        return [-symmetric_critical(spec, params, alpha)]
    return [symmetric_critical(spec, params, alpha)]


def parse_alternative(value: Any) -> str:
    if value is None:
        return "two_sided"
    if value not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {', '.join(ALTERNATIVES)}")
    return str(value)


def number_text(value: float) -> str:
    """A short exact-looking number: 0, 0.5, -2.25, 1e-07."""
    rounded = round_significant(value)
    assert rounded is not None
    if rounded == math.floor(rounded) and abs(rounded) < 1e15:
        return str(int(rounded))
    return repr(rounded)


def decide(p: float, alpha: float, alternative_hypothesis: str) -> Dict[str, Any]:
    """p_value (0 below 1e-300, with a note), reject_null and the conclusion, on the unrounded p."""
    reject = p <= alpha
    tiny = p < SMALLEST_P_VALUE
    p_text = f"p < {SMALLEST_P_VALUE:g}" if tiny else f"p = {_distinct_text(p, alpha)}"
    sign = "<=" if reject else ">"
    if reject:
        verdict = f"reject H0; the data support H1 ({alternative_hypothesis})"
    else:
        verdict = f"fail to reject H0; not enough evidence for H1 ({alternative_hypothesis})"
    result: Dict[str, Any] = {
        "p_value": 0.0 if tiny else p,
        "alpha": alpha,
        "reject_null": reject,
        "conclusion": f"{p_text} {sign} alpha = {alpha:g}: {verdict}",
    }
    if tiny:
        result["p_value_note"] = f"p < {SMALLEST_P_VALUE:g}"
    return result


def _distinct_text(p: float, alpha: float) -> str:
    """p with 4 significant digits, more when 4 would print it equal to alpha (p = 0.0500004)."""
    for digits in range(4, 13):
        text = f"{p:.{digits}g}"
        if p == alpha or float(text) != alpha:
            return text
    return repr(p)


def require_finite(*values: Optional[float]) -> None:
    """Raise the tools' "too large or too small" error for an inf or NaN figure."""
    for value in values:
        if value is not None and not math.isfinite(value):
            raise ValueError("the inputs are too large or too small to compute")


def rounded(value: Any) -> Any:
    """Every float in a result rounded to 12 significant digits (ints, text and booleans kept)."""
    if isinstance(value, bool) or isinstance(value, int) or isinstance(value, str) or value is None:
        return value
    if isinstance(value, float):
        return round_significant(value)
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def df_value(setup: Setup) -> Any:
    """Welch's df stays a float; every other df is a whole number."""
    if setup.df is None:
        return None
    return setup.df if setup.fractional_df else int(setup.df)


def _hypothesized_value(test_type: str, value: Any) -> float:
    if test_type in ("z", "t"):
        if value is None:
            raise ValueError("hypothesized_value is needed: mu0, the mean in H0")
        return float(finite_number(value, "hypothesized_value"))
    if test_type == "one_proportion":
        if value is None:
            raise ValueError("hypothesized_value is needed: p0, the proportion in H0")
        p0 = float(finite_number(value, "hypothesized_value"))
        if not 0.0 < p0 < 1.0:
            hint = f"; write {p0 / 100:g} for {p0:g}%" if 1.0 < p0 < 100.0 else ""
            raise ValueError(f"hypothesized_value (p0) must be strictly between 0 and 1{hint}")
        return p0
    if test_type == "two_proportion":
        if value is not None and finite_number(value, "hypothesized_value") != 0.0:
            raise ValueError("two_proportion tests H0: p1 = p2 only; send hypothesized_value as null or 0")
        return 0.0
    return 0.0 if value is None else finite_number(value, "hypothesized_value")


def run_hypothesis_test(
    test_type: str,
    sample1: Any,
    sample2: Any,
    hypothesized_value: Any,
    alternative: Any,
    alpha: Any,
    pooled: Any,
) -> Dict[str, Any]:
    """Run one test; raises ValueError for invalid or contradictory input."""
    if test_type not in TEST_NAMES:
        raise ValueError(f"Unsupported test_type '{test_type}'; use one of {', '.join(TEST_NAMES)}")
    ignored: List[str] = []
    side = parse_alternative(alternative)
    level = parse_alpha(alpha)
    h0 = _hypothesized_value(test_type, hypothesized_value)
    setup = setup_for(test_type, sample1, sample2, pooled, h0, True, ignored)
    statistic = (setup.estimate - h0) / setup.standard_error
    spec, params = reference_distribution(setup.df)
    require_finite(setup.estimate, setup.standard_error, setup.df, statistic)
    p = p_value(spec, params, statistic, side)
    criticals = critical_values(spec, params, level, side)
    require_finite(p, *criticals)
    alternative_hypothesis = f"{setup.parameter} {_RELATION[side]} {number_text(h0)}"
    result: Dict[str, Any] = {
        "test": test_type,
        "method": f"{TEST_NAMES[test_type]}: {setup.method}",
        "null_hypothesis": f"{setup.parameter} = {number_text(h0)}",
        "alternative_hypothesis": alternative_hypothesis,
        "statistic_name": "z" if setup.df is None else "t",
        "statistic": statistic,
    }
    if setup.df is not None:
        result["df"] = df_value(setup)
    result.update(decide(p, level, alternative_hypothesis))
    result["critical_values"] = criticals
    result["estimate"] = setup.estimate
    result["standard_error"] = setup.standard_error
    result.update(setup.extras)
    result["samples"] = setup.samples
    if setup.notes:
        result["notes"] = setup.notes
    if ignored:
        result["ignored_inputs"] = ignored
    final: Dict[str, Any] = rounded(result)
    return final


TEST_NAMES = {
    "z": "Z-Test",
    "t": "T-Test",
    "two_sample_z": "2-SampZTest",
    "two_sample_t": "2-SampTTest",
    "paired_t": "paired T-Test",
    "one_proportion": "1-PropZTest",
    "two_proportion": "2-PropZTest",
}
