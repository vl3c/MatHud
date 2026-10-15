"""Helpers for CAS tests: run a block on the nerdamer path, and read results by value.

The CAS tools ask the server's SymPy engine first (utils/sympy_client.py) and use nerdamer
when it is unavailable. ``WithoutSympy`` makes it unavailable for a block, so a test can
pin nerdamer's behaviour whatever the server has.
"""

from __future__ import annotations

from typing import Any, List, Optional

from browser import window

from utils.math_utils import MathUtils
from utils.sympy_client import SympyClient


class WithoutSympy:
    """Context manager: the CAS tools use nerdamer inside the block."""

    def __enter__(self) -> "WithoutSympy":
        self._saved = SympyClient._unavailable
        SympyClient._unavailable = True
        return self

    def __exit__(self, *exc_info: Any) -> None:
        SympyClient._unavailable = self._saved


def value_of(result: Any) -> float:
    """A numeric CAS result ("1/3", "pi/2", "0.5 (numeric estimate; ...)") as a float."""
    text = str(result).split(" (")[0]
    return float(window.math.evaluate(text))


def roots_of(result: str) -> List[complex]:
    """The roots in a solve() result "[r1,r2,...] (note)", evaluated by math.js."""
    listed = result.split(" (")[0].strip()
    if not (listed.startswith("[") and listed.endswith("]")):
        raise AssertionError(f"not a root list: {result}")
    roots: List[complex] = []
    for text in MathUtils._split_top_level_commas(listed[1:-1]):
        if not text:
            continue
        value = MathUtils._complex_value(text)
        if value is None:
            raise AssertionError(f"cannot evaluate root {text} in {result}")
        roots.append(value)
    return roots


def real_roots(result: str, digits: Optional[int] = 9) -> List[float]:
    """The real roots of a solve() result, sorted (rounded to digits)."""
    values = [root.real for root in roots_of(result) if abs(root.imag) < 1e-12]
    return sorted(round(value, digits) if digits is not None else value for value in values)


class AnyEngine:
    """Context manager that changes nothing: the block uses whichever engine is available."""

    def __enter__(self) -> "AnyEngine":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None
