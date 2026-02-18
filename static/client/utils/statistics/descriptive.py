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


def _compute_median(sorted_data: List[float]) -> float:
    """Return the median of an already-sorted list."""
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_data[mid])
    return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0


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
    - If every value appears the same number of times (multiple distinct
      values), returns ``[]`` — no meaningful mode.
    - Exception: if all values are identical, that value IS the mode.
    """
    # Build frequency map (iterate sorted data for deterministic order)
    freq: dict[float, int] = {}
    for v in sorted_data:
        fv = float(v)
        freq[fv] = freq.get(fv, 0) + 1

    max_freq = max(freq.values())

    # All unique → no mode
    if max_freq == 1:
        return []

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
    mean = sum(sorted_data) / n
    median = _compute_median(sorted_data)
    mode = _compute_mode(sorted_data)

    # Spread — population formula (divide by N)
    variance = sum((x - mean) ** 2 for x in sorted_data) / n
    variance = max(0.0, variance)  # clamp tiny negatives from float rounding
    std_dev = math.sqrt(variance)

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
