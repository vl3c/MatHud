"""The functions of each probability distribution: density, cdf, survival, quantile, moments.

Every function takes the distribution's parsed parameters (``utils.statistics.distributions``
parses and validates them) and works in plain floats. Upper tails come from their own
formulas, never from 1 - cdf, so small tail probabilities keep their digits. Discrete
distributions take any real x: P(X = 2.5) = 0 and P(X <= 2.5) = P(X <= 2).

Conventions: the quantile of a continuous distribution is the x with P(X <= x) = p; for a
discrete one it is the smallest k with P(X <= k) >= p. The geometric distribution counts the
trials up to and including the first success (support 1, 2, 3, ...).
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Optional, Tuple

from utils.statistics.special_functions import (
    beta_inc,
    beta_inc_upper,
    beta_kernel,
    bisect_increasing,
    gamma_kernel,
    gamma_p,
    gamma_q,
    log_gamma,
)

Params = Mapping[str, float]
Moments = Tuple[Optional[float], Optional[float]]

# A discrete x within this distance of an integer counts as that integer (bar midpoints).
_INTEGER_TOLERANCE = 1e-9
# The discrete quantile accepts P(X <= k) this close below p, so p = P(X <= k) gives k.
_QUANTILE_TOLERANCE = 1e-12


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


def _num(value: float) -> str:
    """A number for an expression, in parentheses so a sign never glues to an operator."""
    return f"({_format_plain_number(value)})"


# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------


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


def _standard_normal_lower(z: float) -> float:
    """P(Z <= z) for z <= 0, as Q(1/2, z^2/2)/2 (erfc(-z/√2)/2, accurate far into the tail)."""
    return 0.5 * gamma_q(0.5, 0.5 * z * z)


def normal_cdf(x: float, params: Params) -> float:
    z = (x - params["mean"]) / params["sigma"]
    return _standard_normal_lower(z) if z <= 0.0 else 1.0 - _standard_normal_lower(-z)


def normal_sf(x: float, params: Params) -> float:
    z = (x - params["mean"]) / params["sigma"]
    return _standard_normal_lower(-z) if z >= 0.0 else 1.0 - _standard_normal_lower(z)


def normal_moments(params: Params) -> Moments:
    return params["mean"], params["sigma"] ** 2


# ---------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------


def uniform_pdf(x: float, params: Params) -> float:
    lower, upper = params["lower"], params["upper"]
    return 1.0 / (upper - lower) if lower <= x <= upper else 0.0


def uniform_cdf(x: float, params: Params) -> float:
    lower, upper = params["lower"], params["upper"]
    return min(1.0, max(0.0, (x - lower) / (upper - lower)))


def uniform_sf(x: float, params: Params) -> float:
    lower, upper = params["lower"], params["upper"]
    return min(1.0, max(0.0, (upper - x) / (upper - lower)))


def uniform_quantile(prob: float, params: Params) -> float:
    return params["lower"] + prob * (params["upper"] - params["lower"])


def uniform_moments(params: Params) -> Moments:
    lower, upper = params["lower"], params["upper"]
    return 0.5 * (lower + upper), (upper - lower) ** 2 / 12.0


def uniform_pdf_expression(params: Params) -> str:
    return _num(1.0 / (params["upper"] - params["lower"]))


# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------


def exponential_pdf(x: float, params: Params) -> float:
    rate = params["rate"]
    return rate * math.exp(-rate * x) if x >= 0.0 else 0.0


def exponential_cdf(x: float, params: Params) -> float:
    return -math.expm1(-params["rate"] * x) if x > 0.0 else 0.0


def exponential_sf(x: float, params: Params) -> float:
    return math.exp(-params["rate"] * x) if x > 0.0 else 1.0


def exponential_quantile(prob: float, params: Params) -> float:
    return -math.log1p(-prob) / params["rate"]


def exponential_moments(params: Params) -> Moments:
    rate = params["rate"]
    return 1.0 / rate, 1.0 / (rate * rate)


def exponential_pdf_expression(params: Params) -> str:
    rate = _num(params["rate"])
    return f"{rate}*exp(-{rate}*x)"


# ---------------------------------------------------------------------------
# Student's t
# ---------------------------------------------------------------------------


def _t_log_constant(df: float) -> float:
    return log_gamma(0.5 * (df + 1.0)) - log_gamma(0.5 * df) - 0.5 * math.log(df * math.pi)


def t_pdf(x: float, params: Params) -> float:
    df = params["df"]
    return math.exp(_t_log_constant(df) - 0.5 * (df + 1.0) * math.log1p(x * x / df))


def _t_lower_tail(t: float, df: float) -> float:
    """P(T <= -|t|) = I_(df/(df+t^2))(df/2, 1/2) / 2, through the side that keeps digits."""
    t2 = t * t
    near = t2 / (df + t2)
    far = df / (df + t2)
    if t2 < df:
        return 0.5 * beta_inc_upper(0.5, 0.5 * df, near, far)
    return 0.5 * beta_inc(0.5 * df, 0.5, far, near)


def t_cdf(x: float, params: Params) -> float:
    tail = _t_lower_tail(x, params["df"])
    return tail if x <= 0.0 else 1.0 - tail


def t_sf(x: float, params: Params) -> float:
    tail = _t_lower_tail(x, params["df"])
    return tail if x >= 0.0 else 1.0 - tail


def t_moments(params: Params) -> Moments:
    df = params["df"]
    return (0.0 if df > 1.0 else None), (df / (df - 2.0) if df > 2.0 else None)


def t_pdf_expression(params: Params) -> str:
    df = params["df"]
    constant = _num(math.exp(_t_log_constant(df)))
    return f"{constant}*(1+x^2/{_num(df)})^(-{_num(0.5 * (df + 1.0))})"


# ---------------------------------------------------------------------------
# Chi-squared
# ---------------------------------------------------------------------------


def _density_at_zero(shape: float, at_two: float) -> float:
    """A density x^(shape-1)·... at 0: infinite below shape 1, ``at_two`` at 1, 0 above."""
    if shape < 1.0:
        return math.inf
    return at_two if shape == 1.0 else 0.0


def chi_squared_pdf(x: float, params: Params) -> float:
    k = params["df"]
    if x < 0.0:
        return 0.0
    if x == 0.0:
        return _density_at_zero(0.5 * k, 0.5)
    return gamma_kernel(0.5 * k, 0.5 * x) / x


def chi_squared_cdf(x: float, params: Params) -> float:
    return gamma_p(0.5 * params["df"], 0.5 * x) if x > 0.0 else 0.0


def chi_squared_sf(x: float, params: Params) -> float:
    return gamma_q(0.5 * params["df"], 0.5 * x) if x > 0.0 else 1.0


def chi_squared_moments(params: Params) -> Moments:
    k = params["df"]
    return k, 2.0 * k


def chi_squared_pdf_expression(params: Params) -> str:
    """exp(L + (k/2 - 1)·ln(x) - x/2): one exponential, so large df neither overflows nor underflows."""
    k = params["df"]
    log_constant = -0.5 * k * math.log(2.0) - log_gamma(0.5 * k)
    return f"exp({_num(log_constant)}+{_num(0.5 * k - 1.0)}*ln(x)-x/2)"


# ---------------------------------------------------------------------------
# F
# ---------------------------------------------------------------------------


def f_pdf(x: float, params: Params) -> float:
    d1, d2 = params["df"], params["df2"]
    if x < 0.0:
        return 0.0
    if x == 0.0:
        return _density_at_zero(0.5 * d1, 1.0)
    return beta_kernel(0.5 * d1, 0.5 * d2, d1 * x / (d1 * x + d2), d2 / (d1 * x + d2)) / x


def _f_tails(x: float, d1: float, d2: float) -> Tuple[float, float]:
    """(P(F <= x), P(F > x)) for x > 0, each from the side of the beta function that keeps digits."""
    y = d1 * x / (d1 * x + d2)
    rest = d2 / (d1 * x + d2)
    if y < 0.5:
        return beta_inc(0.5 * d1, 0.5 * d2, y, rest), beta_inc_upper(0.5 * d1, 0.5 * d2, y, rest)
    return beta_inc_upper(0.5 * d2, 0.5 * d1, rest, y), beta_inc(0.5 * d2, 0.5 * d1, rest, y)


def f_cdf(x: float, params: Params) -> float:
    return _f_tails(x, params["df"], params["df2"])[0] if x > 0.0 else 0.0


def f_sf(x: float, params: Params) -> float:
    return _f_tails(x, params["df"], params["df2"])[1] if x > 0.0 else 1.0


def f_moments(params: Params) -> Moments:
    d1, d2 = params["df"], params["df2"]
    mean = d2 / (d2 - 2.0) if d2 > 2.0 else None
    variance = 2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0) ** 2 * (d2 - 4.0)) if d2 > 4.0 else None
    return mean, variance


def f_pdf_expression(params: Params) -> str:
    """exp(L + (d1/2 - 1)·ln(x) - ((d1+d2)/2)·ln(1 + (d1/d2)·x)), one exponential like χ²'s."""
    d1, d2 = params["df"], params["df2"]
    log_beta = log_gamma(0.5 * d1) + log_gamma(0.5 * d2) - log_gamma(0.5 * (d1 + d2))
    log_constant = 0.5 * d1 * math.log(d1 / d2) - log_beta
    return f"exp({_num(log_constant)}+{_num(0.5 * d1 - 1.0)}*ln(x)-{_num(0.5 * (d1 + d2))}*ln(1+{_num(d1 / d2)}*x))"


# ---------------------------------------------------------------------------
# Discrete helpers
# ---------------------------------------------------------------------------


def as_integer(x: float) -> Optional[int]:
    """The integer x stands for (within 1e-9), or None."""
    nearest = round(x)
    return int(nearest) if abs(x - nearest) <= _INTEGER_TOLERANCE else None


def floor_integer(x: float) -> int:
    """The largest integer k <= x, treating x within 1e-9 below an integer as that integer."""
    return int(math.floor(x + _INTEGER_TOLERANCE))


# ---------------------------------------------------------------------------
# Binomial
# ---------------------------------------------------------------------------


def binomial_pmf(x: float, params: Params) -> float:
    k = as_integer(x)
    n, p = int(params["n"]), params["p"]
    if k is None or k < 0 or k > n:
        return 0.0
    if k == 0:
        return math.exp(n * math.log1p(-p))
    if k == n:
        return math.exp(n * math.log(p))
    return n / (k * (n - k)) * beta_kernel(k, n - k, p, 1.0 - p)


def binomial_cdf(x: float, params: Params) -> float:
    k = floor_integer(x)
    n, p = int(params["n"]), params["p"]
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return beta_inc(n - k, k + 1, 1.0 - p, p)


def binomial_sf(x: float, params: Params) -> float:
    k = floor_integer(x)
    n, p = int(params["n"]), params["p"]
    if k < 0:
        return 1.0
    if k >= n:
        return 0.0
    return beta_inc_upper(n - k, k + 1, 1.0 - p, p)


def binomial_moments(params: Params) -> Moments:
    n, p = params["n"], params["p"]
    return n * p, n * p * (1.0 - p)


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------


def poisson_pmf(x: float, params: Params) -> float:
    k = as_integer(x)
    rate = params["rate"]
    if k is None or k < 0:
        return 0.0
    if k == 0:
        return math.exp(-rate)
    return gamma_kernel(k, rate) / k


def poisson_cdf(x: float, params: Params) -> float:
    k = floor_integer(x)
    return gamma_q(k + 1.0, params["rate"]) if k >= 0 else 0.0


def poisson_sf(x: float, params: Params) -> float:
    k = floor_integer(x)
    return gamma_p(k + 1.0, params["rate"]) if k >= 0 else 1.0


def poisson_moments(params: Params) -> Moments:
    return params["rate"], params["rate"]


# ---------------------------------------------------------------------------
# Geometric (trials up to and including the first success)
# ---------------------------------------------------------------------------


def geometric_pmf(x: float, params: Params) -> float:
    k = as_integer(x)
    p = params["p"]
    if k is None or k < 1:
        return 0.0
    return p * math.exp((k - 1) * math.log1p(-p))


def geometric_cdf(x: float, params: Params) -> float:
    k = floor_integer(x)
    return -math.expm1(k * math.log1p(-params["p"])) if k >= 1 else 0.0


def geometric_sf(x: float, params: Params) -> float:
    k = floor_integer(x)
    return math.exp(k * math.log1p(-params["p"])) if k >= 1 else 1.0


def geometric_moments(params: Params) -> Moments:
    p = params["p"]
    return 1.0 / p, (1.0 - p) / (p * p)


# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------


def continuous_quantile(
    prob: float,
    cdf: Callable[[float], float],
    sf: Callable[[float], float],
    support: Tuple[float, float],
    center: float,
    scale: float,
) -> float:
    """The x with P(X <= x) = prob, by bisection; the upper half solves P(X > x) = 1 - prob."""
    if prob <= 0.5:

        def increasing(x: float) -> float:
            return cdf(x)

        target = prob
    else:

        def increasing(x: float) -> float:
            return -sf(x)

        target = -(1.0 - prob)
    lo = _expand(increasing, target, support[0], center, -scale, below=True)
    hi = _expand(increasing, target, support[1], center, scale, below=False)
    return bisect_increasing(increasing, target, (lo, hi))


def _expand(
    increasing: Callable[[float], float],
    target: float,
    end: float,
    center: float,
    step: float,
    *,
    below: bool,
) -> float:
    """A bracket end on one side of the target: the support's end when finite, else a doubling search."""
    if math.isfinite(end):
        return end
    x = center + step
    for _ in range(2000):
        value = increasing(x)
        if (value < target) if below else (value > target):
            return x
        step *= 2.0
        x = center + step
    raise ArithmeticError("quantile bracket not found")


def discrete_quantile(
    prob: float,
    cdf: Callable[[float], float],
    sf: Callable[[float], float],
    lowest: int,
    highest: Optional[int],
    center: float,
    scale: float,
) -> int:
    """The smallest k in the support with P(X <= k) >= prob (within 1e-12 of prob)."""

    def reaches(k: int) -> bool:
        if prob <= 0.5:
            return cdf(k) >= prob * (1.0 - _QUANTILE_TOLERANCE)
        return sf(k) <= (1.0 - prob) * (1.0 + _QUANTILE_TOLERANCE)

    lo = lowest - 1  # P(X <= lowest - 1) = 0 < prob
    if highest is not None:
        hi = highest
    else:
        step = max(1.0, scale)
        hi = max(lowest, int(math.ceil(center + step)))
        while not reaches(hi):
            step *= 2.0
            hi = max(lowest, int(math.ceil(center + step)))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if reaches(mid):
            hi = mid
        else:
            lo = mid
    return hi
