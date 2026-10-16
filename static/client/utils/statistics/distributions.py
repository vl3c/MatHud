"""Probability distributions: the distribution table and the normal distribution's functions.

``DISTRIBUTIONS`` maps each ``distribution_type`` the plot tools accept to a
``DistributionSpec``: how to parse its parameters, evaluate its pdf, write its pdf as a
MatHud expression and pick default plot bounds. ``get_distribution`` looks a type up.

Key Features:
    - Normal PDF expression generation with mean and sigma
    - Normal PDF evaluation at a point
    - Default bounds calculation for normal distributions
    - Finite value validation for parameters
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple


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


def normal_pdf(x: float, mean: float, sigma: float) -> float:
    """Evaluate the normal PDF at x; far tails underflow to 0.0."""
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    z = (float(x) - float(mean)) / float(sigma)
    return math.exp(-0.5 * z * z) / (float(sigma) * math.sqrt(2.0 * math.pi))


def default_normal_bounds(mean: float, sigma: float, k: float = 4.0) -> Tuple[float, float]:
    mean = _require_finite(float(mean), "mean")
    sigma = _require_finite(float(sigma), "sigma")
    k = _require_finite(float(k), "k")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    if k <= 0.0:
        raise ValueError("k must be > 0")
    return (mean - k * sigma, mean + k * sigma)


DEFAULT_DISTRIBUTION_TYPE = "normal"


@dataclass(frozen=True)
class DistributionSpec:
    """One distribution: its parameters, pdf, pdf expression and default plot bounds.

    ``parse_params(raw, strict=False)`` turns a raw ``distribution_params`` value (a dict, or
    anything else for none) into the canonical parameters, in a fixed key order, and raises
    ValueError for invalid ones. A missing or null parameter takes its default; with
    ``strict=True`` a parameter that is present but null is an error instead.
    """

    name: str
    parse_params: Callable[..., Dict[str, float]]
    pdf: Callable[[float, Mapping[str, float]], float]
    pdf_expression: Callable[[Mapping[str, float]], str]
    default_bounds: Callable[[Mapping[str, float]], Tuple[float, float]]


def _parse_normal_params(params: Any, strict: bool = False) -> Dict[str, float]:
    """``{"mean", "sigma"}`` from raw params: defaults 0 and 1, sigma > 0, both finite."""
    params = params if isinstance(params, dict) else {}

    mean_raw = params.get("mean")
    sigma_raw = params.get("sigma")
    if strict:
        for key in ("mean", "sigma"):
            if key in params and params[key] is None:
                raise ValueError(f"{key} must be a number")

    mean = 0.0 if mean_raw is None else float(mean_raw)
    sigma = 1.0 if sigma_raw is None else float(sigma_raw)

    if not math.isfinite(mean):
        raise ValueError("mean must be finite")
    if not math.isfinite(sigma):
        raise ValueError("sigma must be finite")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")

    return {"mean": mean, "sigma": sigma}


_NORMAL = DistributionSpec(
    name="normal",
    parse_params=_parse_normal_params,
    pdf=lambda x, params: normal_pdf(x, params["mean"], params["sigma"]),
    pdf_expression=lambda params: normal_pdf_expression(params["mean"], params["sigma"]),
    default_bounds=lambda params: default_normal_bounds(params["mean"], params["sigma"], k=4.0),
)

DISTRIBUTIONS: Dict[str, DistributionSpec] = {"normal": _NORMAL}


def get_distribution(distribution_type: Any, *, default: Optional[str] = None) -> DistributionSpec:
    """The spec for a distribution type, matched without case or surrounding spaces.

    An empty or null type takes ``default`` when one is given. An unknown type, or an empty
    one with no default, raises ValueError naming the type as given.
    """
    key = str(distribution_type or "").strip().lower()
    if not key and default is not None:
        key = default
    spec = DISTRIBUTIONS.get(key)
    if spec is None:
        raise ValueError(f"Unsupported distribution_type '{distribution_type}'")
    return spec
