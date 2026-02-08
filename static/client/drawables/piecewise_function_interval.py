"""
MatHud Piecewise Function Interval

Represents a single interval of a piecewise function, storing the expression,
its parsed evaluator, interval bounds, and explicit undefined points (holes).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class PiecewiseFunctionInterval:
    """Represents a single interval of a piecewise function.

    Stores the expression, its parsed evaluator, interval bounds with
    inclusivity flags, and explicit undefined points (holes).
    """

    def __init__(
        self,
        expression: str,
        evaluator: Callable[[float], float],
        left: Optional[float],
        right: Optional[float],
        left_inclusive: bool,
        right_inclusive: bool,
        undefined_at: Optional[List[float]] = None,
    ) -> None:
        self.expression: str = expression
        self.evaluator: Callable[[float], float] = evaluator
        self.left: Optional[float] = left
        self.right: Optional[float] = right
        self.left_inclusive: bool = left_inclusive
        self.right_inclusive: bool = right_inclusive
        self.undefined_at: List[float] = undefined_at or []

    def contains(self, x: float) -> bool:
        """Check if x falls within this interval (excludes undefined points)."""
        # Check explicit undefined points first
        for hole in self.undefined_at:
            if abs(x - hole) < 1e-12:
                return False

        if self.left is not None:
            if self.left_inclusive:
                if x < self.left:
                    return False
            else:
                if x <= self.left:
                    return False

        if self.right is not None:
            if self.right_inclusive:
                if x > self.right:
                    return False
            else:
                if x >= self.right:
                    return False

        return True

    def is_undefined_at(self, x: float) -> bool:
        """Check if x is an explicit undefined point (hole)."""
        for hole in self.undefined_at:
            if abs(x - hole) < 1e-12:
                return True
        return False

    def evaluate(self, x: float) -> float:
        """Evaluate this interval's expression at x. Returns NaN for undefined points."""
        if self.is_undefined_at(x):
            return float('nan')
        return self.evaluator(x)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize interval to dictionary."""
        result: Dict[str, Any] = {
            "expression": self.expression,
            "left": self.left,
            "right": self.right,
            "left_inclusive": self.left_inclusive,
            "right_inclusive": self.right_inclusive,
        }
        if self.undefined_at:
            result["undefined_at"] = self.undefined_at
        return result

