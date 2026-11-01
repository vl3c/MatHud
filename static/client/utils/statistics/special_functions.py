"""Special functions for the probability distributions: log-gamma and the incomplete gamma and beta.

Pure Python with no ``browser`` import, so the same code runs in Brython and under pytest.
The normal distribution's tails are built on ``gamma_q`` too, so every distribution shares
one code path that keeps far tails to full relative precision.

Algorithms (reimplemented from the standard textbook forms):
    - ``log_gamma``: the Lanczos approximation (g = 7, nine coefficients).
    - ``gamma_p`` / ``gamma_q``: the power series for x < a + 1, otherwise the continued
      fraction evaluated with the modified Lentz method. Each computes the smaller tail
      directly, so far tails keep their digits.
    - ``beta_inc`` / ``beta_inc_upper``: the continued fraction of the regularized
      incomplete beta function, with I_x(a, b) = 1 - I_(1-x)(b, a) on the slow side.
    - ``bisect_increasing``: bisection for x where an increasing function crosses a target.

Every iteration is bounded; a series or fraction that does not converge raises
ArithmeticError instead of returning an inaccurate value.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

_LANCZOS_G = 7.0
_LANCZOS_COEFFICIENTS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)

_MAX_ITERATIONS = 1000


def _iteration_limit(size: float) -> int:
    """Series and fractions need about sqrt(size) terms near their transition point."""
    return _MAX_ITERATIONS + int(40.0 * math.sqrt(size))


_EPSILON = 1e-16
_TINY = 1e-300


def log_gamma(x: float) -> float:
    """ln Γ(x) for x > 0."""
    if not x > 0.0:
        raise ValueError("log_gamma needs x > 0")
    if x < 0.5:
        # Γ(x) = Γ(x + 1) / x keeps the Lanczos sum in its accurate range.
        return log_gamma(x + 1.0) - math.log(x)
    z = x - 1.0
    total = _LANCZOS_COEFFICIENTS[0]
    for index in range(1, len(_LANCZOS_COEFFICIENTS)):
        total += _LANCZOS_COEFFICIENTS[index] / (z + index)
    t = z + _LANCZOS_G + 0.5
    return _HALF_LOG_TWO_PI + (z + 0.5) * math.log(t) - t + math.log(total)


def log_beta(a: float, b: float) -> float:
    """ln B(a, b) for a, b > 0."""
    return log_gamma(a) + log_gamma(b) - log_gamma(a + b)


def _stirling_correction(z: float) -> float:
    """ln Γ(z) minus its Stirling main part (z - 1/2) ln z - z + ln(2π)/2, for z > 0.

    The asymptotic series is used from z = 15, where its first omitted term is below 1e-16;
    below that the difference is taken directly (both parts are small there).
    """
    if z >= 15.0:
        inv = 1.0 / z
        inv2 = inv * inv
        return inv * (1.0 / 12.0 - inv2 * (1.0 / 360.0 - inv2 * (1.0 / 1260.0 - inv2 * (1.0 / 1680.0 - inv2 / 1188.0))))
    return log_gamma(z) - ((z - 0.5) * math.log(z) - z + _HALF_LOG_TWO_PI)


def _t_minus_log1p(t: float) -> float:
    """t - ln(1 + t) for t > -1, without cancellation near 0."""
    if abs(t) < 0.1:
        # sum over k >= 2 of (-1)^k t^k / k
        total = 0.0
        power = t * t
        for k in range(2, 40):
            term = power / k
            total += term if k % 2 == 0 else -term
            if abs(term) < 1e-17 * abs(total):
                break
            power *= t
        return total
    return t - math.log1p(t)


def _relative_deviation(value: float, center: float) -> float:
    """t - ln(1 + t) with t = (value - center)/center, for value, center > 0.

    Far below the center 1 + t loses digits (t ≈ -1), so the log takes value/center directly.
    """
    t = (value - center) / center
    if t < -0.5:
        return t - math.log(value / center)
    return _t_minus_log1p(t)


def gamma_kernel(a: float, x: float) -> float:
    """x^a e^(-x) / Γ(a) for a > 0, x > 0.

    Written as exp(-a·(t - ln(1+t))) · sqrt(a / 2π) / exp(correction) with t = (x - a)/a,
    so large a and x keep their digits (the naive logs cancel to ~a·1e-16).
    """
    if x <= 0.0:
        return 0.0
    if x / a == 0.0:
        # x/a underflows: the plain logs lose nothing here.
        return math.exp(a * math.log(x) - x - log_gamma(a))
    log_value = -a * _relative_deviation(x, a) + 0.5 * math.log(a) - _HALF_LOG_TWO_PI - _stirling_correction(a)
    return math.exp(log_value)


def beta_kernel(a: float, b: float, x: float, y: Optional[float] = None) -> float:
    """x^a y^b / B(a, b) with y = 1 - x, for a, b > 0 and 0 < x < 1, accurate for large a and b.

    ``y`` is 1 - x when the caller knows it exactly (1 - x rounds away a small y's digits).
    With s = a + b, x0 = a / s and y0 = b / s, the logs of the powers and the Stirling parts
    of the three gammas combine into -a·l(t1) - b·l(t2) + ln(a·b/s)/2 - ln(2π)/2 plus the
    corrections, where l(t) = t - ln(1+t), t1 = (x - x0)/x0 and t2 = (y - y0)/y0
    (``_relative_deviation``).
    """
    if y is None:
        y = 1.0 - x
    s = a + b
    x0 = a / s
    y0 = b / s
    if x / x0 == 0.0 or y / y0 == 0.0:
        # x/x0 or y/y0 underflows: use the plain logs.
        return math.exp(a * math.log(x) + b * math.log(y) - log_beta(a, b))
    log_value = (
        -a * _relative_deviation(x, x0)
        - b * _relative_deviation(y, y0)
        + 0.5 * math.log(a * b / s)
        - _HALF_LOG_TWO_PI
        - _stirling_correction(a)
        - _stirling_correction(b)
        + _stirling_correction(s)
    )
    return math.exp(log_value)


def _gamma_series(a: float, x: float) -> float:
    """P(a, x) by its power series; converges fast for x < a + 1."""
    term = 1.0 / a
    total = term
    denominator = a
    for _ in range(_iteration_limit(a)):
        denominator += 1.0
        term *= x / denominator
        total += term
        if abs(term) < abs(total) * _EPSILON:
            return total * gamma_kernel(a, x)
    raise ArithmeticError("incomplete gamma series did not converge")


def _gamma_continued_fraction(a: float, x: float) -> float:
    """Q(a, x) by its continued fraction (modified Lentz); converges fast for x >= a + 1."""
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, _iteration_limit(a)):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPSILON:
            return h * gamma_kernel(a, x)
    raise ArithmeticError("incomplete gamma continued fraction did not converge")


def _check_gamma_args(a: float, x: float) -> None:
    if not a > 0.0:
        raise ValueError("incomplete gamma needs a > 0")
    if not x >= 0.0:
        raise ValueError("incomplete gamma needs x >= 0")


def gamma_p(a: float, x: float) -> float:
    """The regularized lower incomplete gamma P(a, x) = γ(a, x) / Γ(a)."""
    _check_gamma_args(a, x)
    if x == 0.0:
        return 0.0
    if math.isinf(x):
        return 1.0
    if x < a + 1.0:
        return min(1.0, _gamma_series(a, x))
    return max(0.0, 1.0 - _gamma_continued_fraction(a, x))


def gamma_q(a: float, x: float) -> float:
    """The regularized upper incomplete gamma Q(a, x) = 1 - P(a, x)."""
    _check_gamma_args(a, x)
    if x == 0.0:
        return 1.0
    if math.isinf(x):
        return 0.0
    if x < a + 1.0:
        return max(0.0, 1.0 - _gamma_series(a, x))
    return min(1.0, _gamma_continued_fraction(a, x))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """The continued fraction of I_x(a, b) without its prefactor (modified Lentz)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, _iteration_limit(a + b)):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPSILON:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _beta_tail(a: float, b: float, x: float, y: float) -> float:
    """I_x(a, b) evaluated directly (y = 1 - x); accurate when x < (a + 1) / (a + b + 2)."""
    return beta_kernel(a, b, x, y) * _beta_continued_fraction(a, b, x) / a


def _check_beta_args(a: float, b: float, x: float) -> None:
    if not (a > 0.0 and b > 0.0):
        raise ValueError("incomplete beta needs a > 0 and b > 0")
    if not 0.0 <= x <= 1.0:
        raise ValueError("incomplete beta needs 0 <= x <= 1")


def beta_inc(a: float, b: float, x: float, y: Optional[float] = None) -> float:
    """The regularized incomplete beta I_x(a, b); ``y`` is 1 - x when known exactly."""
    _check_beta_args(a, b, x)
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    y = 1.0 - x if y is None else y
    if x < (a + 1.0) / (a + b + 2.0):
        return min(1.0, _beta_tail(a, b, x, y))
    return max(0.0, 1.0 - _beta_tail(b, a, y, x))


def beta_inc_upper(a: float, b: float, x: float, y: Optional[float] = None) -> float:
    """1 - I_x(a, b) = I_y(b, a) with y = 1 - x, computed without cancellation."""
    _check_beta_args(a, b, x)
    if x == 0.0:
        return 1.0
    if x == 1.0:
        return 0.0
    y = 1.0 - x if y is None else y
    if x < (a + 1.0) / (a + b + 2.0):
        return max(0.0, 1.0 - _beta_tail(a, b, x, y))
    return min(1.0, _beta_tail(b, a, y, x))


def bisect_increasing(
    function: Callable[[float], float],
    target: float,
    bracket: Tuple[float, float],
    *,
    max_steps: int = 400,
) -> float:
    """The x in ``bracket`` where an increasing ``function`` reaches ``target``.

    ``function(lo) <= target <= function(hi)`` must hold. Returns a midpoint that hits the
    target exactly (the median of a symmetric distribution is then 0.0). Stops when the interval is
    narrower than 1e-15 relative (or 1e-300 absolute) or after ``max_steps`` halvings.
    """
    lo, hi = bracket
    for _ in range(max_steps):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        value = function(mid)
        if value == target:
            return mid
        if value < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(abs(lo), abs(hi)) or hi - lo <= 1e-300:
            break
    return 0.5 * (lo + hi)
