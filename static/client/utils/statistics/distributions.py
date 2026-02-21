"""Probability distribution expression generators.

This module provides functions for generating MatHud-compatible expression
strings for probability distributions.

Key Features:
    - Normal PDF expression generation with mean and sigma
    - Default bounds calculation for normal distributions
    - Finite value validation for parameters
"""

from __future__ import annotations

import math
from typing import Tuple


def _require_finite(value: float, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def normal_pdf_expression(mean: float, sigma: float) -> str:
    """
    Return a MatHud-compatible function expression string for the normal PDF.

    Uses '^' for exponentiation to match MatHud function expression conventions.
    """
    mean = _require_finite(float(mean), "mean")
    sigma = _require_finite(float(sigma), "sigma")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")

    # f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-((x-mean)^2) / (2*sigma^2))
    return f"(1/(({sigma})*sqrt(2*pi)))*exp(-(((x-({mean}))^2)/(2*({sigma})^2)))"


def default_normal_bounds(mean: float, sigma: float, k: float = 4.0) -> Tuple[float, float]:
    mean = _require_finite(float(mean), "mean")
    sigma = _require_finite(float(sigma), "sigma")
    k = _require_finite(float(k), "k")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    if k <= 0.0:
        raise ValueError("k must be > 0")
    return (mean - k * sigma, mean + k * sigma)
