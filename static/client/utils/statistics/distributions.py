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


def _format_plain_number(value: float) -> str:
    """Format a float without scientific notation, keeping its full (round-trip) precision.

    str(1e-05) gives "1e-05", which expression parsers can misread as "1*e - 05".
    """
    text = repr(float(value))
    if "e" not in text and "E" not in text:
        return text
    mantissa, exponent = text.lower().split("e")
    sign = ""
    if mantissa.startswith("-"):
        sign, mantissa = "-", mantissa[1:]
    integer_part, _, fraction_part = mantissa.partition(".")
    digits = integer_part + fraction_part
    point = len(integer_part) + int(exponent)
    if point <= 0:
        plain = "0." + "0" * (-point) + digits
    elif point >= len(digits):
        plain = digits + "0" * (point - len(digits)) + ".0"
    else:
        plain = digits[:point] + "." + digits[point:]
    return sign + plain


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
    mean_text = _format_plain_number(mean)
    sigma_text = _format_plain_number(sigma)
    return f"(1/(({sigma_text})*sqrt(2*pi)))*exp(-(((x-({mean_text}))^2)/(2*({sigma_text})^2)))"


def default_normal_bounds(mean: float, sigma: float, k: float = 4.0) -> Tuple[float, float]:
    mean = _require_finite(float(mean), "mean")
    sigma = _require_finite(float(sigma), "sigma")
    k = _require_finite(float(k), "k")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    if k <= 0.0:
        raise ValueError("k must be > 0")
    return (mean - k * sigma, mean + k * sigma)
