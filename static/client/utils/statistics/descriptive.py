"""Descriptive statistics computation module.

Provides a pure-Python function to compute standard descriptive statistics
(mean, median, mode, standard deviation, variance, min, max, quartiles)
for a list of numbers. No browser imports — fully testable with pytest.
"""

from __future__ import annotations

import math
from typing import List, TypedDict


class DescriptiveStatisticsResult(TypedDict):
    """Result of descriptive statistics computation."""

    count: int
    mean: float
    median: float
    mode: List[float]
    standard_deviation: float
    variance: float
    min: float
    max: float
    q1: float
    q3: float
    iqr: float
    range: float


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_data(data: List[float]) -> None:
    """Validate that *data* is a non-empty list of finite numbers.

    Raises:
        TypeError: If *data* is not a list or contains non-numeric elements
                   (including booleans).
        ValueError: If *data* is empty or contains non-finite values.
    """
    if not isinstance(data, list):
        raise TypeError(f"data must be a list, got {type(data).__name__}")

    if len(data) == 0:
        raise ValueError("data must not be empty")

    for i, v in enumerate(data):
        if isinstance(v, bool):
            raise TypeError(f"data[{i}] is a bool; only int and float are accepted")
        if not isinstance(v, (int, float)):
            raise TypeError(f"data[{i}] must be int or float, got {type(v).__name__}")
        if not math.isfinite(v):
            raise ValueError(f"data[{i}] is not finite: {v}")


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


def _compute_spread(values: List[float], mean: float) -> tuple[float, float]:
    """Return the population (variance, standard deviation).

    Squared deviations are summed with math.fsum. Deviations too large to square
    are scaled first, so the standard deviation stays finite when the variance
    alone would overflow (the variance is then inf).
    """
    n = len(values)
    deviations = [v - mean for v in values]
    largest = max(abs(d) for d in deviations)
    if largest <= _SAFE_DEVIATION:
        variance = math.fsum(d * d for d in deviations) / n
        return variance, math.sqrt(variance)
    if not math.isfinite(largest):
        return math.inf, math.inf
    std_dev = largest * math.sqrt(math.fsum((d / largest) ** 2 for d in deviations) / n)
    return std_dev * std_dev, std_dev


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
        count, mean, median, mode, standard_deviation, variance,
        min, max, q1, q3, iqr, range.

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

    # Spread — population formula (divide by N)
    variance, std_dev = _compute_spread(sorted_data, mean)

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
        standard_deviation=std_dev,
        variance=variance,
        min=data_min,
        max=data_max,
        q1=q1,
        q3=q3,
        iqr=iqr,
        range=data_range,
    )
