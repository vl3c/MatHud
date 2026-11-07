"""Comparison of golden values recorded as text, shared by the golden-file tests."""

from __future__ import annotations

import math
from typing import Any


def same_outcome(actual: Any, expected: Any) -> bool:
    """Equal, except that floats written as text may differ in their last bits: the platform's
    libm rounds acosh, atanh, erf and cbrt differently across runner images."""
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(same_outcome(a, e) for a, e in zip(actual, expected))
    if actual == expected:
        return True
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    try:
        actual_value, expected_value = float(actual), float(expected)
    except ValueError:
        return False
    return math.isclose(actual_value, expected_value, rel_tol=1e-13, abs_tol=1e-300)
