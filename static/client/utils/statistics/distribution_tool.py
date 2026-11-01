"""The compute_distribution tool: probabilities, quantiles and moments of one distribution.

The function registry binds the tool straight to this pure entry point; it needs no canvas.
One call answers everything it is given: the density and every tail at ``x``, the
probability of the interval from ``left_bound`` to ``right_bound``, and the quantile of
``probability``, next to the distribution's mean, variance and standard deviation.
Numbers are rounded to 12 significant digits.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from utils.statistics.distribution_functions import floor_integer
from utils.statistics.distributions import DistributionSpec, get_distribution

SIGNIFICANT_DIGITS = 12


def round_significant(value: Optional[float], digits: int = SIGNIFICANT_DIGITS) -> Optional[float]:
    """``value`` rounded to ``digits`` significant digits (None, 0 and non-finite values unchanged)."""
    if value is None or value == 0.0 or not math.isfinite(value):
        return value
    magnitude = int(math.floor(math.log10(abs(value))))
    return round(value, digits - 1 - magnitude) + 0.0


def compute_distribution_tool(
    distribution_type: str,
    distribution_params: Optional[Dict[str, Any]] = None,
    x: Optional[float] = None,
    left_bound: Optional[float] = None,
    right_bound: Optional[float] = None,
    probability: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the tool; raises ValueError for an invalid distribution, parameter or input."""
    spec = get_distribution(distribution_type)
    params = spec.parse_params(distribution_params)
    try:
        return _compute(spec, params, distribution_params, x, left_bound, right_bound, probability)
    except (OverflowError, ZeroDivisionError):
        raise ValueError("the parameters or inputs are too large or too small to compute") from None


def _compute(
    spec: DistributionSpec,
    params: Dict[str, float],
    distribution_params: Optional[Dict[str, Any]],
    x: Optional[float],
    left_bound: Optional[float],
    right_bound: Optional[float],
    probability: Optional[float],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"distribution": spec.name, "params": dict(params)}
    _add_moments(result, spec, params)
    result["support"] = describe_support(spec, params)
    if x is not None:
        _add_point(result, spec, params, _finite(x, "x"))
    if left_bound is not None or right_bound is not None:
        _add_interval(result, spec, params, left_bound, right_bound)
    if probability is not None:
        _add_quantile(result, spec, params, _finite(probability, "probability"))
    ignored = spec.ignored_params(distribution_params)
    if ignored:
        result["ignored_params"] = ignored
    return result


def describe_support(spec: DistributionSpec, params: Dict[str, float]) -> str:
    """The support in words: "0, 1, ..., 10", "1, 2, 3, ...", "[0, inf)" or "(-inf, inf)"."""
    lo, hi = spec.support(params)
    if spec.discrete:
        first = _number_text(lo)
        if math.isfinite(hi):
            return f"{first}, {_number_text(lo + 1)}, ..., {_number_text(hi)}"
        return f"{first}, {_number_text(lo + 1)}, {_number_text(lo + 2)}, ..."
    left = "(-inf" if math.isinf(lo) else f"[{_number_text(lo)}"
    right = "inf)" if math.isinf(hi) else f"{_number_text(hi)}]"
    return f"{left}, {right}"


def _number_text(value: float) -> str:
    return str(int(value)) if value == math.floor(value) and abs(value) < 1e15 else repr(value)


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _add_moments(result: Dict[str, Any], spec: DistributionSpec, params: Dict[str, float]) -> None:
    mean, variance = spec.moments(params)
    result["mean"] = round_significant(mean)
    result["variance"] = round_significant(variance)
    result["standard_deviation"] = round_significant(math.sqrt(variance)) if variance is not None else None
    if mean is None or variance is None:
        result["moments_note"] = _undefined_moments_note(spec, params, mean is None)


def _undefined_moments_note(spec: DistributionSpec, params: Dict[str, float], mean_undefined: bool) -> str:
    if spec.name == "t":
        if mean_undefined:
            return "the mean is undefined for df <= 1 and the variance for df <= 2"
        return "the variance is infinite for 1 < df <= 2"
    if mean_undefined:
        return "the mean is undefined for df2 <= 2 and the variance for df2 <= 4"
    return "the variance is undefined for df2 <= 4"


def _add_point(result: Dict[str, Any], spec: DistributionSpec, params: Dict[str, float], x: float) -> None:
    result["x"] = x
    below = spec.cdf(x, params)
    above = spec.sf(x, params)
    if not spec.discrete:
        result["pdf"] = round_significant(spec.pdf(x, params))
        result["P(X <= x)"] = round_significant(below)
        result["P(X > x)"] = round_significant(above)
        return
    at = spec.pdf(x, params)
    result["P(X = x)"] = round_significant(at)
    result["P(X <= x)"] = round_significant(below)
    result["P(X < x)"] = round_significant(max(0.0, below - at))
    result["P(X >= x)"] = round_significant(min(1.0, above + at))
    result["P(X > x)"] = round_significant(above)


def _add_interval(
    result: Dict[str, Any],
    spec: DistributionSpec,
    params: Dict[str, float],
    left_bound: Optional[float],
    right_bound: Optional[float],
) -> None:
    a = -math.inf if left_bound is None else _finite(left_bound, "left_bound")
    b = math.inf if right_bound is None else _finite(right_bound, "right_bound")
    if a > b:
        raise ValueError("left_bound must be <= right_bound")
    result["a"] = None if left_bound is None else a
    result["b"] = None if right_bound is None else b
    result["P(a <= X <= b)"] = round_significant(interval_probability(spec, params, a, b))
    if spec.discrete:
        result["P(a < X < b)"] = round_significant(
            max(
                0.0,
                interval_probability(spec, params, a, b) - _point_mass(spec, params, a) - _point_mass(spec, params, b),
            )
            if a < b
            else 0.0
        )
    else:
        result["interval_note"] = "for a continuous distribution P(a < X < b) is the same"


def _point_mass(spec: DistributionSpec, params: Dict[str, float], x: float) -> float:
    return spec.pdf(x, params) if math.isfinite(x) else 0.0


def interval_probability(spec: DistributionSpec, params: Dict[str, float], a: float, b: float) -> float:
    """P(a <= X <= b), from whichever tail keeps more digits; a may be -inf and b inf."""
    if a > b:
        return 0.0
    if spec.discrete:
        # P(a <= X <= b) = P(X <= b) - P(X <= a - 1 step): the integers from ceil(a) to floor(b).
        first = -floor_integer(-a) if math.isfinite(a) else None
        below_a = 0.0 if first is None else spec.cdf(first - 1, params)
        above_a = 1.0 if first is None else spec.sf(first - 1, params)
    else:
        below_a = 0.0 if math.isinf(a) else spec.cdf(a, params)
        above_a = 1.0 if math.isinf(a) else spec.sf(a, params)
    below_b = 1.0 if math.isinf(b) else spec.cdf(b, params)
    above_b = 0.0 if math.isinf(b) else spec.sf(b, params)
    # Subtract the smaller pair: both tails, or both cdfs, whichever is further from 1.
    if below_b <= 0.5:
        return max(0.0, below_b - below_a)
    return max(0.0, above_a - above_b)


def _add_quantile(result: Dict[str, Any], spec: DistributionSpec, params: Dict[str, float], prob: float) -> None:
    if not 0.0 < prob < 1.0:
        raise ValueError("probability must be between 0 and 1 (exclusive)")
    result["probability"] = prob
    result["quantile"] = round_significant(spec.quantile(prob, params))
    if spec.discrete:
        result["quantile_note"] = "the smallest value k with P(X <= k) >= probability"
