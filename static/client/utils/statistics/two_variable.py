"""Two-variable statistics: Pearson's r and covariance of paired x and y lists.

Pure Python like ``descriptive``; ``fit_linear`` takes its r from the same
``paired_sums``, so both tools report one r for the same data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict

from utils.statistics.descriptive import (
    DescriptiveStatisticsResult,
    _compute_mean,
    _validate_data,
    compute_descriptive_statistics,
)


class TwoVariableStatisticsResult(TypedDict):
    """Result of two-variable statistics; x and y hold each list's one-variable statistics."""

    count: int
    x: DescriptiveStatisticsResult
    y: DescriptiveStatisticsResult
    r: Optional[float]
    r_squared: Optional[float]
    sample_covariance: float
    population_covariance: float


@dataclass(frozen=True)
class PairedSums:
    """Centred sums of paired data, each axis divided by its largest |deviation|.

    The unscaled sums are Sxx = x_scale**2 * sxx, Syy = y_scale**2 * syy and
    Sxy = x_scale * y_scale * sxy; the scaled ones never overflow. A scale is 0
    when every value on that axis is equal, and r is then None.
    """

    count: int
    x_scale: float
    y_scale: float
    sxx: float
    syy: float
    sxy: float
    r: Optional[float]

    def covariance(self, divisor: int) -> float:
        """Sxy / divisor: n - 1 for the sample covariance, n for the population one."""
        if self.x_scale == 0.0 or self.y_scale == 0.0:
            return 0.0
        # sxy / divisor is at most 2 in size, so this overflows only when the result does
        return self.x_scale * (self.y_scale * (self.sxy / divisor))


def _scaled_deviations(values: List[float]) -> Tuple[float, List[float]]:
    """Return (scale, deviations from the mean divided by scale); (0, zeros) for equal values.

    Equal values are tested directly: the fsum mean of a constant list can miss
    the value by an ulp, which would leave tiny nonzero deviations. A deviation
    too large for a float (values near the float limit with opposite signs) is
    taken from the halved values, and the scale doubled.
    """
    if min(values) == max(values):
        return 0.0, [0.0] * len(values)
    mean = _compute_mean(values)
    factor = 1.0
    deviations = [v - mean for v in values]
    largest = max(abs(d) for d in deviations)
    if not math.isfinite(largest):
        half_mean = mean / 2.0
        deviations = [v / 2.0 - half_mean for v in values]
        largest = max(abs(d) for d in deviations)
        factor = 2.0
    if largest == 0.0:
        return 0.0, [0.0] * len(values)
    return factor * largest, [d / largest for d in deviations]


def paired_sums(x: List[float], y: List[float]) -> PairedSums:
    """Return the scaled centred sums and Pearson's r of equally long, validated lists.

    r is clamped to [-1, 1] and is None when all x values or all y values are
    equal. It never raises for finite input, so ``fit_linear`` can call it on
    any data it accepts.
    """
    x_scale, dx = _scaled_deviations(x)
    y_scale, dy = _scaled_deviations(y)
    sxx = math.fsum(d * d for d in dx)
    syy = math.fsum(d * d for d in dy)
    sxy = math.fsum(a * b for a, b in zip(dx, dy))
    r: Optional[float] = None
    if x_scale > 0.0 and y_scale > 0.0:
        # Each scaled sum of squares lies in [1, n], so the product cannot overflow
        r = max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))
    return PairedSums(count=len(x), x_scale=x_scale, y_scale=y_scale, sxx=sxx, syy=syy, sxy=sxy, r=r)


def compute_two_variable_statistics(data: List[float], y_data: List[float]) -> TwoVariableStatisticsResult:
    """Compute Pearson's r, r squared and the covariances of paired data.

    Args:
        data: The x values, finite numbers.
        y_data: The y values, as many as x values.

    Returns:
        count, x and y (each list's one-variable statistics), r and r_squared
        (None when all x or all y values are equal), sample_covariance (divided
        by n - 1) and population_covariance (divided by n). A covariance too
        large for a float is inf.

    Raises:
        TypeError: If either list is not a list or holds a non-number.
        ValueError: If a list is empty or holds a non-finite value, the lengths
            differ, or there are fewer than two pairs.
    """
    _validate_data(data, "data")
    _validate_data(y_data, "y_data")
    if len(data) != len(y_data):
        raise ValueError(f"data and y_data must have the same length ({len(data)} and {len(y_data)})")
    n = len(data)
    if n < 2:
        raise ValueError("two-variable statistics need at least two (x, y) pairs")
    sums = paired_sums(data, y_data)
    return TwoVariableStatisticsResult(
        count=n,
        x=compute_descriptive_statistics(data),
        y=compute_descriptive_statistics(y_data),
        r=sums.r,
        r_squared=None if sums.r is None else sums.r * sums.r,
        sample_covariance=sums.covariance(n - 1),
        population_covariance=sums.covariance(n),
    )
