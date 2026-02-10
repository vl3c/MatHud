"""Pure-Python numeric integration helpers.

This module intentionally has no browser/Brython dependencies so it can be
validated via server-side pytest suites.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, TypedDict


class NumericIntegrationResult(TypedDict):
    """Result payload for numeric integration."""

    method: str
    steps: int
    value: float
    error_estimate: float


def _require_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _safe_eval(eval_fn: Callable[[float], float], x: float) -> float:
    y = eval_fn(float(x))
    if isinstance(y, bool) or not isinstance(y, (int, float)):
        raise TypeError("integrand must evaluate to a number")
    y = float(y)
    if not math.isfinite(y):
        raise ValueError("integrand produced a non-finite value")
    return y


def _trapezoid(eval_fn: Callable[[float], float], a: float, b: float, steps: int) -> float:
    h = (b - a) / steps
    total = 0.5 * (_safe_eval(eval_fn, a) + _safe_eval(eval_fn, b))
    for i in range(1, steps):
        total += _safe_eval(eval_fn, a + i * h)
    return total * h


def _midpoint(eval_fn: Callable[[float], float], a: float, b: float, steps: int) -> float:
    h = (b - a) / steps
    total = 0.0
    for i in range(steps):
        total += _safe_eval(eval_fn, a + (i + 0.5) * h)
    return total * h


def _simpson(eval_fn: Callable[[float], float], a: float, b: float, steps: int) -> tuple[float, int]:
    # Simpson requires an even partition count.
    if steps % 2 == 1:
        steps += 1
    h = (b - a) / steps
    total = _safe_eval(eval_fn, a) + _safe_eval(eval_fn, b)
    for i in range(1, steps):
        weight = 4 if i % 2 == 1 else 2
        total += weight * _safe_eval(eval_fn, a + i * h)
    return total * h / 3.0, steps


def integrate(
    eval_fn: Callable[[float], float],
    lower_bound: float,
    upper_bound: float,
    method: str = "simpson",
    steps: int = 200,
) -> NumericIntegrationResult:
    """Numerically integrate an integrand function over a finite interval."""
    if not callable(eval_fn):
        raise TypeError("eval_fn must be callable")

    a = _require_finite(lower_bound, "lower_bound")
    b = _require_finite(upper_bound, "upper_bound")
    if a >= b:
        raise ValueError("lower_bound must be less than upper_bound")

    steps = _require_positive_int(steps, "steps")
    method = str(method).strip().lower()
    if method not in ("trapezoid", "midpoint", "simpson"):
        raise ValueError("method must be one of: trapezoid, midpoint, simpson")

    if method == "trapezoid":
        coarse = _trapezoid(eval_fn, a, b, steps)
        fine = _trapezoid(eval_fn, a, b, steps * 2)
        err = abs(fine - coarse) / 3.0
        return NumericIntegrationResult(
            method=method,
            steps=steps,
            value=fine,
            error_estimate=err,
        )

    if method == "midpoint":
        coarse = _midpoint(eval_fn, a, b, steps)
        fine = _midpoint(eval_fn, a, b, steps * 2)
        err = abs(fine - coarse) / 3.0
        return NumericIntegrationResult(
            method=method,
            steps=steps,
            value=fine,
            error_estimate=err,
        )

    coarse, coarse_steps = _simpson(eval_fn, a, b, steps)
    fine, fine_steps = _simpson(eval_fn, a, b, coarse_steps * 2)
    err = abs(fine - coarse) / 15.0
    return NumericIntegrationResult(
        method="simpson",
        steps=fine_steps,
        value=fine,
        error_estimate=err,
    )
