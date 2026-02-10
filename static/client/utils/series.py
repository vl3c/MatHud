"""Series partial-sum utilities.

Pure-Python functions for computing partial sums of well-known series:
- Arithmetic series
- Geometric series
- Harmonic series
- Fibonacci series (cumulative partial sums)
- Taylor approximations (e^x, sin(x), cos(x))
- Leibniz series for pi/4
"""

from __future__ import annotations

import math
from typing import Dict, List, TypedDict


class SeriesResult(TypedDict):
    """Result of a series partial-sum computation."""

    series_type: str
    partial_sums: List[float]
    num_terms: int
    parameters: Dict[str, float]


def _require_positive_int(n: int, name: str) -> int:
    """Validate that *n* is a positive integer.

    Raises:
        TypeError: If *n* is not an ``int``.
        ValueError: If *n* is not positive.
    """
    if not isinstance(n, int):
        raise TypeError(f"{name} must be an integer")
    if n <= 0:
        raise ValueError(f"{name} must be positive")
    return n


def _require_finite(value: float, name: str) -> float:
    """Validate that *value* is a finite number.

    Raises:
        TypeError: If *value* is not numeric.
        ValueError: If *value* is infinite or NaN.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


# ------------------------------------------------------------------
# Arithmetic series
# ------------------------------------------------------------------


def arithmetic_partial_sums(a: float, d: float, n: int) -> SeriesResult:
    """Compute partial sums of an arithmetic series.

    The k-th partial sum is S_k = sum_{i=0}^{k-1} (a + i*d) for k = 1..n.
    Uses the closed form S_k = k*a + d*k*(k-1)/2.

    Args:
        a: First term of the series.
        d: Common difference.
        n: Number of terms (positive integer).

    Returns:
        A ``SeriesResult`` with the list of partial sums.
    """
    a = _require_finite(a, "a")
    d = _require_finite(d, "d")
    n = _require_positive_int(n, "n")

    sums: List[float] = []
    for k in range(1, n + 1):
        sums.append(k * a + d * k * (k - 1) / 2)
    return SeriesResult(
        series_type="arithmetic",
        partial_sums=sums,
        num_terms=n,
        parameters={"a": a, "d": d},
    )


# ------------------------------------------------------------------
# Geometric series
# ------------------------------------------------------------------


def geometric_partial_sums(a: float, r: float, n: int) -> SeriesResult:
    """Compute partial sums of a geometric series.

    The k-th partial sum is S_k = sum_{i=0}^{k-1} a*r^i for k = 1..n.
    When r == 1, S_k = k*a.  Otherwise S_k = a*(1 - r^k)/(1 - r).

    Args:
        a: First term of the series.
        r: Common ratio.
        n: Number of terms (positive integer).

    Returns:
        A ``SeriesResult`` with the list of partial sums.
    """
    a = _require_finite(a, "a")
    r = _require_finite(r, "r")
    n = _require_positive_int(n, "n")

    sums: List[float] = []
    if r == 1.0:
        for k in range(1, n + 1):
            sums.append(k * a)
    else:
        for k in range(1, n + 1):
            sums.append(a * (1 - r**k) / (1 - r))
    return SeriesResult(
        series_type="geometric",
        partial_sums=sums,
        num_terms=n,
        parameters={"a": a, "r": r},
    )


# ------------------------------------------------------------------
# Harmonic series
# ------------------------------------------------------------------


def harmonic_partial_sums(n: int) -> SeriesResult:
    """Compute partial sums of the harmonic series.

    H_k = sum_{i=1}^{k} 1/i for k = 1..n.

    Args:
        n: Number of terms (positive integer).

    Returns:
        A ``SeriesResult`` with the list of partial sums.
    """
    n = _require_positive_int(n, "n")

    sums: List[float] = []
    running = 0.0
    for k in range(1, n + 1):
        running += 1.0 / k
        sums.append(running)
    return SeriesResult(
        series_type="harmonic",
        partial_sums=sums,
        num_terms=n,
        parameters={},
    )


# ------------------------------------------------------------------
# Fibonacci partial sums
# ------------------------------------------------------------------


def fibonacci_partial_sums(n: int) -> SeriesResult:
    """Compute cumulative partial sums of the Fibonacci sequence.

    Fibonacci numbers: F_1=1, F_2=1, F_3=2, F_4=3, ...
    Partial sums: S_k = sum_{i=1}^{k} F_i for k = 1..n.

    Args:
        n: Number of terms (positive integer).

    Returns:
        A ``SeriesResult`` with the list of partial sums.
    """
    n = _require_positive_int(n, "n")

    sums: List[float] = []
    fib_prev, fib_curr = 0, 1
    running = 0.0
    for _ in range(n):
        running += fib_curr
        sums.append(running)
        fib_prev, fib_curr = fib_curr, fib_prev + fib_curr
    return SeriesResult(
        series_type="fibonacci",
        partial_sums=sums,
        num_terms=n,
        parameters={},
    )


# ------------------------------------------------------------------
# Taylor series: e^x
# ------------------------------------------------------------------


def taylor_exp_partial_sums(x: float, n: int) -> SeriesResult:
    """Compute partial sums of the Taylor series for e^x.

    S_k = sum_{i=0}^{k-1} x^i / i! for k = 1..n.
    Uses incremental term computation to avoid factorial overflow.

    Args:
        x: Evaluation point (finite float).
        n: Number of terms (positive integer).

    Returns:
        A ``SeriesResult`` with the list of partial sums.
    """
    x = _require_finite(x, "x")
    n = _require_positive_int(n, "n")

    sums: List[float] = []
    term = 1.0  # x^0 / 0!
    running = 0.0
    for i in range(n):
        if i > 0:
            term *= x / i
        running += term
        sums.append(running)
    return SeriesResult(
        series_type="taylor_exp",
        partial_sums=sums,
        num_terms=n,
        parameters={"x": x},
    )


# ------------------------------------------------------------------
# Taylor series: sin(x)
# ------------------------------------------------------------------


def taylor_sin_partial_sums(x: float, n: int) -> SeriesResult:
    """Compute partial sums of the Taylor series for sin(x).

    S_k = sum_{i=0}^{k-1} (-1)^i * x^(2i+1) / (2i+1)! for k = 1..n.
    Uses incremental term computation to avoid factorial overflow.

    Args:
        x: Evaluation point (finite float).
        n: Number of terms (positive integer).

    Returns:
        A ``SeriesResult`` with the list of partial sums.
    """
    x = _require_finite(x, "x")
    n = _require_positive_int(n, "n")

    sums: List[float] = []
    term = x  # first term: x^1 / 1!
    running = 0.0
    for i in range(n):
        if i > 0:
            term *= -x * x / ((2 * i) * (2 * i + 1))
        running += term
        sums.append(running)
    return SeriesResult(
        series_type="taylor_sin",
        partial_sums=sums,
        num_terms=n,
        parameters={"x": x},
    )


# ------------------------------------------------------------------
# Taylor series: cos(x)
# ------------------------------------------------------------------


def taylor_cos_partial_sums(x: float, n: int) -> SeriesResult:
    """Compute partial sums of the Taylor series for cos(x).

    S_k = sum_{i=0}^{k-1} (-1)^i * x^(2i) / (2i)! for k = 1..n.
    Uses incremental term computation to avoid factorial overflow.

    Args:
        x: Evaluation point (finite float).
        n: Number of terms (positive integer).

    Returns:
        A ``SeriesResult`` with the list of partial sums.
    """
    x = _require_finite(x, "x")
    n = _require_positive_int(n, "n")

    sums: List[float] = []
    term = 1.0  # first term: x^0 / 0! = 1
    running = 0.0
    for i in range(n):
        if i > 0:
            term *= -x * x / ((2 * i - 1) * (2 * i))
        running += term
        sums.append(running)
    return SeriesResult(
        series_type="taylor_cos",
        partial_sums=sums,
        num_terms=n,
        parameters={"x": x},
    )


# ------------------------------------------------------------------
# Leibniz series for pi/4
# ------------------------------------------------------------------


def leibniz_partial_sums(n: int) -> SeriesResult:
    """Compute partial sums of the Leibniz series for pi/4.

    S_k = sum_{i=0}^{k-1} (-1)^i / (2i+1) for k = 1..n.
    This series converges to pi/4.

    Args:
        n: Number of terms (positive integer).

    Returns:
        A ``SeriesResult`` with the list of partial sums.
    """
    n = _require_positive_int(n, "n")

    sums: List[float] = []
    running = 0.0
    for i in range(n):
        running += (-1) ** i / (2 * i + 1)
        sums.append(running)
    return SeriesResult(
        series_type="leibniz",
        partial_sums=sums,
        num_terms=n,
        parameters={},
    )
