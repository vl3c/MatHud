"""Comparing golden values that hold numbers as repr text.

Functions such as cbrt, acosh, atanh and erf come from the platform's libm, whose last digit can
differ between runner images, so numbers match within a few units in the last place; everything
else (exception names, translated text) must match exactly.
"""

from __future__ import annotations

import math
from typing import Any

REL_TOL = 1e-14


def _as_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def golden_match(actual: Any, expected: Any) -> bool:
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(golden_match(a, e) for a, e in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return sorted(actual) == sorted(expected) and all(golden_match(actual[k], expected[k]) for k in expected)
    if isinstance(actual, str) and isinstance(expected, str) and actual != expected:
        a, e = _as_float(actual), _as_float(expected)
        return a is not None and e is not None and math.isclose(a, e, rel_tol=REL_TOL, abs_tol=0.0)
    return bool(actual == expected)
