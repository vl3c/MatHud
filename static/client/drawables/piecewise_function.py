"""
MatHud Piecewise Function Drawable

Represents a piecewise-defined mathematical function with multiple expression intervals,
each valid over a specific domain. Implements the same interface as Function for
seamless integration with FunctionRenderable.

Key Features:
    - Multiple expression intervals with domain bounds
    - Automatic discontinuity detection at interval boundaries
    - Function-compatible interface for rendering reuse
    - Support for unbounded intervals (extending to infinity)

Interface Compatibility:
    - function(x): Evaluates the correct interval for given x
    - left_bound, right_bound: Derived from outermost interval boundaries
    - vertical_asymptotes: Union of per-interval asymptotes
    - point_discontinuities: Interval boundaries where limits differ

Dependencies:
    - constants: Default styling values
    - expression_validator: Expression parsing
    - utils.math_utils: Asymptote and discontinuity calculations
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from constants import default_color
from drawables.drawable import Drawable
from drawables.piecewise_function_interval import PiecewiseFunctionInterval
from expression_validator import ExpressionValidator
from utils.math_utils import MathUtils


class PiecewiseFunction(Drawable):
    """A piecewise-defined mathematical function.

    Stores multiple expression intervals, each valid over a specific domain.
    Implements the same interface as Function for rendering compatibility.
    """

    def __init__(
        self,
        pieces: List[Dict[str, Any]],
        name: Optional[str] = None,
        color: str = default_color,
        vertical_asymptotes: Optional[List[float]] = None,
        horizontal_asymptotes: Optional[List[float]] = None,
        point_discontinuities: Optional[List[float]] = None,
    ) -> None:
        """Initialize a piecewise function.

        Args:
            pieces: List of interval definitions, each with:
                - expression: Mathematical expression string
                - left: Left interval bound (None for negative infinity)
                - right: Right interval bound (None for positive infinity)
                - left_inclusive: Whether left bound is included
                - right_inclusive: Whether right bound is included
            name: Optional name for the function
            color: Color for rendering
            vertical_asymptotes: Pre-computed vertical asymptotes (optional)
            horizontal_asymptotes: Pre-computed horizontal asymptotes (optional)
            point_discontinuities: Pre-computed discontinuities (optional)
        """
        self.intervals: List[PiecewiseFunctionInterval] = []
        self.is_periodic: bool = False
        self.estimated_period: Optional[float] = None

        self._parse_intervals(pieces)
        self._sort_intervals()
        self._compute_bounds()

        if vertical_asymptotes is not None:
            self.vertical_asymptotes = vertical_asymptotes
        else:
            self.vertical_asymptotes = []

        if horizontal_asymptotes is not None:
            self.horizontal_asymptotes = horizontal_asymptotes
        else:
            self.horizontal_asymptotes = []

        if point_discontinuities is not None:
            self.point_discontinuities = point_discontinuities
        else:
            self._compute_discontinuities()

        self._compute_asymptotes()

        super().__init__(name=name or "f", color=color)

    def _parse_intervals(self, pieces: List[Dict[str, Any]]) -> None:
        """Parse interval definitions into PiecewiseFunctionInterval objects."""
        if not pieces:
            raise ValueError("At least one interval is required for a piecewise function")

        for piece_def in pieces:
            expression = piece_def.get("expression", "")
            if not expression:
                raise ValueError("Each interval must have an expression")

            fixed_expr = ExpressionValidator.fix_math_expression(expression)
            evaluator = ExpressionValidator.parse_function_string(expression)

            self.intervals.append(
                PiecewiseFunctionInterval(
                    expression=fixed_expr,
                    evaluator=evaluator,
                    left=piece_def.get("left"),
                    right=piece_def.get("right"),
                    left_inclusive=piece_def.get("left_inclusive", True),
                    right_inclusive=piece_def.get("right_inclusive", False),
                    undefined_at=piece_def.get("undefined_at"),
                )
            )

    def _sort_intervals(self) -> None:
        """Sort intervals by their left bound for consistent evaluation."""

        def sort_key(interval: PiecewiseFunctionInterval) -> float:
            if interval.left is None:
                return float("-inf")
            return interval.left

        self.intervals.sort(key=sort_key)

    def _compute_bounds(self) -> None:
        """Compute overall left and right bounds from intervals."""
        self.left_bound: Optional[float] = None
        self.right_bound: Optional[float] = None

        for interval in self.intervals:
            if interval.left is not None:
                if self.left_bound is None or interval.left < self.left_bound:
                    self.left_bound = interval.left
            else:
                self.left_bound = None
                break

        for interval in self.intervals:
            if interval.right is not None:
                if self.right_bound is None or interval.right > self.right_bound:
                    self.right_bound = interval.right
            else:
                self.right_bound = None
                break

    def _compute_discontinuities(self) -> None:
        """Compute point discontinuities at interval boundaries and explicit undefined points."""
        self.point_discontinuities: List[float] = []

        # Collect explicit undefined points from all intervals
        for interval in self.intervals:
            for hole in interval.undefined_at:
                if hole not in self.point_discontinuities:
                    self.point_discontinuities.append(hole)

        # Collect boundary points (excluding the function's overall bounds)
        # The overall bounds are not discontinuities - they're just where the function ends
        boundary_points: List[float] = []
        for interval in self.intervals:
            if interval.left is not None:
                boundary_points.append(interval.left)
            if interval.right is not None:
                boundary_points.append(interval.right)

        boundary_points = sorted(set(boundary_points))

        for boundary in boundary_points:
            # Skip the function's overall bounds - these aren't discontinuities
            if boundary == self.left_bound or boundary == self.right_bound:
                continue
            if self._is_jump_discontinuity(boundary):
                if boundary not in self.point_discontinuities:
                    self.point_discontinuities.append(boundary)

        for interval in self.intervals:
            interval_discontinuities = MathUtils.calculate_point_discontinuities(
                interval.expression,
                interval.left,
                interval.right,
            )
            for disc in interval_discontinuities:
                if disc not in self.point_discontinuities:
                    self.point_discontinuities.append(disc)

        self.point_discontinuities.sort()

    def _is_jump_discontinuity(self, x: float) -> bool:
        """Check if there's a jump discontinuity at x."""
        epsilon = 1e-9

        left_value: Optional[float] = None
        right_value: Optional[float] = None

        for interval in self.intervals:
            if interval.contains(x - epsilon):
                try:
                    left_value = interval.evaluate(x - epsilon)
                except Exception:
                    pass
            if interval.contains(x + epsilon):
                try:
                    right_value = interval.evaluate(x + epsilon)
                except Exception:
                    pass

        if left_value is None or right_value is None:
            return True

        try:
            if abs(left_value - right_value) > 1e-6:
                return True
        except Exception:
            return True

        return False

    def _compute_asymptotes(self) -> None:
        """Compute vertical asymptotes from all intervals."""
        all_asymptotes: List[float] = list(self.vertical_asymptotes)

        for interval in self.intervals:
            interval_asymptotes, _, _ = MathUtils.calculate_asymptotes_and_discontinuities(
                interval.expression,
                interval.left,
                interval.right,
            )
            for asymp in interval_asymptotes:
                if asymp not in all_asymptotes:
                    all_asymptotes.append(asymp)

        self.vertical_asymptotes = sorted(all_asymptotes)

    def function(self, x: float) -> float:
        """Evaluate the piecewise function at x.

        Finds the appropriate interval and evaluates it.
        Returns NaN if x is not covered by any interval.
        """
        for interval in self.intervals:
            if interval.contains(x):
                return interval.evaluate(x)
        return float("nan")

    def get_class_name(self) -> str:
        return "PiecewiseFunction"

    def get_state(self) -> Dict[str, Any]:
        """Serialize function state for persistence."""
        pieces_data = [interval.to_dict() for interval in self.intervals]

        state: Dict[str, Any] = {
            "name": self.name,
            "args": {
                "pieces": pieces_data,
            },
        }

        if self.vertical_asymptotes:
            state["args"]["vertical_asymptotes"] = self.vertical_asymptotes
        if self.horizontal_asymptotes:
            state["args"]["horizontal_asymptotes"] = self.horizontal_asymptotes
        if self.point_discontinuities:
            state["args"]["point_discontinuities"] = self.point_discontinuities

        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        if id(self) in memo:
            return cast(PiecewiseFunction, memo[id(self)])

        pieces_data = [interval.to_dict() for interval in self.intervals]

        new_function = PiecewiseFunction(
            pieces=pieces_data,
            name=self.name,
            color=self.color,
            vertical_asymptotes=self.vertical_asymptotes.copy() if self.vertical_asymptotes else None,
            horizontal_asymptotes=self.horizontal_asymptotes.copy() if self.horizontal_asymptotes else None,
            point_discontinuities=self.point_discontinuities.copy() if self.point_discontinuities else None,
        )
        memo[id(self)] = new_function
        return new_function

    def translate(self, x_offset: float, y_offset: float) -> None:
        """Translate the piecewise function."""
        if x_offset == 0 and y_offset == 0:
            return

        import re

        new_pieces: List[Dict[str, Any]] = []

        for interval in self.intervals:
            new_expr = interval.expression

            if x_offset != 0:
                protected_funcs: list[str] = sorted(ExpressionValidator.ALLOWED_FUNCTIONS, key=len, reverse=True)
                func_pattern: str = "|".join(map(re.escape, protected_funcs))
                pattern: str = rf"\b(x)\b|({func_pattern})"

                def replace_match(match: Any) -> str:
                    if match.group(1):
                        return f"(x - {x_offset})"
                    elif match.group(2):
                        return cast(str, match.group(2))
                    return cast(str, match.group(0))

                new_expr = re.sub(pattern, replace_match, new_expr)

            if y_offset != 0:
                new_expr = f"({new_expr}) + {y_offset}"

            new_left = interval.left + x_offset if interval.left is not None else None
            new_right = interval.right + x_offset if interval.right is not None else None
            new_undefined_at = [h + x_offset for h in interval.undefined_at] if interval.undefined_at else None

            new_pieces.append(
                {
                    "expression": new_expr,
                    "left": new_left,
                    "right": new_right,
                    "left_inclusive": interval.left_inclusive,
                    "right_inclusive": interval.right_inclusive,
                    "undefined_at": new_undefined_at,
                }
            )

        self.intervals = []
        self._parse_intervals(new_pieces)
        self._sort_intervals()
        self._compute_bounds()
        self._compute_discontinuities()
        self._compute_asymptotes()

    def rotate(self, angle: float) -> None:
        """Rotation is not applicable to functions."""
        pass

    def update_color(self, color: str) -> None:
        """Update the function color metadata."""
        self.color = str(color)

    def has_point_discontinuity_between_x(self, x1: float, x2: float) -> bool:
        """Check if there is a point discontinuity between x1 and x2."""
        return any(x1 < x < x2 for x in self.point_discontinuities)

    def has_vertical_asymptote_between_x(self, x1: float, x2: float) -> bool:
        """Check if there is a vertical asymptote between x1 and x2."""
        return any(x1 <= x < x2 for x in self.vertical_asymptotes)

    def get_vertical_asymptote_between_x(self, x1: float, x2: float) -> Optional[float]:
        """Get the x value of a vertical asymptote between x1 and x2."""
        for x in self.vertical_asymptotes:
            if x1 <= x < x2:
                return x
        return None
