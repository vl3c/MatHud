"""Descriptive statistics computation module.

Provides a pure-Python function to compute standard descriptive statistics
(mean, median, mode, standard deviation, variance, min, max, quartiles)
for a list of numbers. No browser imports — fully testable with pytest.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, TypedDict


class DescriptiveStatisticsResult(TypedDict):
    """Result of descriptive statistics computation."""

    count: int
    mean: float
    median: float
    mode: List[float]
    # Sample figures divide by n - 1 (s, Sx) and are None for one value;
    # population figures divide by n (sigma)
    sample_standard_deviation: Optional[float]
    population_standard_deviation: float
    sample_variance: Optional[float]
    population_variance: float
    min: float
    max: float
    q1: float
    q3: float
    iqr: float
    range: float


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_data(data: List[float], name: str = "data") -> None:
    """Validate that *data* is a non-empty list of finite numbers.

    *name* is the argument's name in the error messages.

    Raises:
        TypeError: If *data* is not a list or contains non-numeric elements
                   (including booleans).
        ValueError: If *data* is empty or contains non-finite values.
    """
    if not isinstance(data, list):
        raise TypeError(f"{name} must be a list, got {type(data).__name__}")

    if len(data) == 0:
        raise ValueError(f"{name} must not be empty")

    for i, v in enumerate(data):
        if isinstance(v, bool):
            raise TypeError(f"{name}[{i}] is a bool; only int and float are accepted")
        if not isinstance(v, (int, float)):
            raise TypeError(f"{name}[{i}] must be int or float, got {type(v).__name__}")
        if not math.isfinite(v):
            raise ValueError(f"{name}[{i}] is not finite: {v}")


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------


def _average_of_two(a: float, b: float) -> float:
    """Return (a + b) / 2 without overflowing for values near the float limit."""
    total = a + b
    if math.isfinite(total):
        return total / 2.0
    return a / 2.0 + b / 2.0


def _compute_mean(values: List[float]) -> float:
    """Return the mean using an exactly rounded sum (math.fsum).

    A plain running sum loses small values next to large ones of opposite
    sign ([1e16, 1, -1e16] would give 0). When the sum overflows although the
    mean is finite, the values are divided by the count before summing.
    """
    n = len(values)
    try:
        total = math.fsum(values)
    except OverflowError:
        total = math.inf
    if math.isfinite(total):
        return total / n
    return math.fsum(v / n for v in values)


# Deviations below this square without overflow, so the variance is summed directly
_SAFE_DEVIATION = 1e150


def _sum_of_squares(values: List[float], mean: float) -> Tuple[float, float]:
    """Return (scale, scaled): the sum of squared deviations from *mean* is scale**2 * scaled.

    The squares are summed with math.fsum. Deviations too large to square are
    divided by the largest one first (scale is then that deviation, otherwise 1),
    so the standard deviation stays finite when the variance alone would
    overflow. A deviation that overflows itself gives an infinite scale.
    """
    deviations = [v - mean for v in values]
    largest = max(abs(d) for d in deviations)
    if largest <= _SAFE_DEVIATION:
        return 1.0, math.fsum(d * d for d in deviations)
    if not math.isfinite(largest):
        return math.inf, math.inf
    return largest, math.fsum((d / largest) ** 2 for d in deviations)


def _variance_and_deviation(scale: float, scaled: float, divisor: int) -> Tuple[float, float]:
    """Return (variance, standard deviation) from _sum_of_squares' result.

    *divisor* is n for the population figures and n - 1 for the sample ones.
    """
    if not math.isfinite(scale):
        return math.inf, math.inf
    if scale == 1.0:
        # Summed unscaled (a scaled sum always has a scale above _SAFE_DEVIATION)
        variance = scaled / divisor
        return variance, math.sqrt(variance)
    std_dev = scale * math.sqrt(scaled / divisor)
    return std_dev * std_dev, std_dev


def population_spread(values: List[float], name: str = "data") -> Tuple[float, float]:
    """Return the population (variance, standard deviation) of *values*.

    The same routine as compute_descriptive_statistics, so expressions' pstdev and
    pvariance give the tool's figures bit for bit. *name* is the argument's name in
    the error messages.
    """
    _validate_data(values, name)
    scale, scaled = _sum_of_squares(values, _compute_mean(values))
    return _variance_and_deviation(scale, scaled, len(values))


def _compute_median(sorted_data: List[float]) -> float:
    """Return the median of an already-sorted list."""
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_data[mid])
    return _average_of_two(float(sorted_data[mid - 1]), float(sorted_data[mid]))


def _compute_quartiles(sorted_data: List[float]) -> tuple[float, float]:
    """Return (Q1, Q3) using the median-of-halves (exclusive) method.

    - Q1 = median of the lower half (excluding the median element for odd N).
    - Q3 = median of the upper half (excluding the median element for odd N).
    - Edge cases: N=1 -> Q1=Q3=value; N=2 -> Q1=min, Q3=max.
    """
    n = len(sorted_data)

    if n == 1:
        return (float(sorted_data[0]), float(sorted_data[0]))

    if n == 2:
        return (float(sorted_data[0]), float(sorted_data[1]))

    mid = n // 2
    if n % 2 == 1:
        lower = sorted_data[:mid]
        upper = sorted_data[mid + 1 :]
    else:
        lower = sorted_data[:mid]
        upper = sorted_data[mid:]

    return (_compute_median(lower), _compute_median(upper))


def _compute_mode(sorted_data: List[float]) -> List[float]:
    """Return the mode(s) of an already-sorted list.

    - Returns all values tied for the highest frequency, sorted ascending.
    - If there are two or more distinct values and they all appear the same
      number of times, returns ``[]`` (no meaningful mode).
    - A single distinct value is the mode however often it appears
      ([5] and [7, 7, 7] give [5] and [7]).
    """
    # Build frequency map (iterate sorted data for deterministic order)
    freq: dict[float, int] = {}
    for v in sorted_data:
        fv = float(v)
        freq[fv] = freq.get(fv, 0) + 1

    max_freq = max(freq.values())
    modes = sorted(v for v, c in freq.items() if c == max_freq)

    # All equal frequency with multiple distinct values → no mode
    if len(modes) == len(freq) and len(freq) > 1:
        return []

    return modes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_descriptive_statistics(
    data: List[float],
) -> DescriptiveStatisticsResult:
    """Compute descriptive statistics for a list of numbers.

    Args:
        data: Non-empty list of finite numbers (int or float).

    Returns:
        A ``DescriptiveStatisticsResult`` dict containing:
        count, mean, median, mode, sample_standard_deviation,
        population_standard_deviation, sample_variance, population_variance,
        min, max, q1, q3, iqr, range. The sample figures (divided by n - 1)
        are None when there is one value.

    Raises:
        TypeError: If *data* is not a list or contains non-numeric values.
        ValueError: If *data* is empty or contains non-finite values.
    """
    _validate_data(data)

    sorted_data = sorted(data)
    n = len(sorted_data)

    # Central tendency
    mean = _compute_mean(sorted_data)
    median = _compute_median(sorted_data)
    mode = _compute_mode(sorted_data)

    # Spread: population figures divide by n, sample figures by n - 1
    scale, scaled = _sum_of_squares(sorted_data, mean)
    population_variance, population_std_dev = _variance_and_deviation(scale, scaled, n)
    sample_variance: Optional[float] = None
    sample_std_dev: Optional[float] = None
    if n > 1:
        sample_variance, sample_std_dev = _variance_and_deviation(scale, scaled, n - 1)

    # Extremes
    data_min = float(sorted_data[0])
    data_max = float(sorted_data[-1])
    data_range = data_max - data_min

    # Quartiles
    q1, q3 = _compute_quartiles(sorted_data)
    iqr = q3 - q1

    return DescriptiveStatisticsResult(
        count=n,
        mean=mean,
        median=median,
        mode=mode,
        sample_standard_deviation=sample_std_dev,
        population_standard_deviation=population_std_dev,
        sample_variance=sample_variance,
        population_variance=population_variance,
        min=data_min,
        max=data_max,
        q1=q1,
        q3=q3,
        iqr=iqr,
        range=data_range,
    )
