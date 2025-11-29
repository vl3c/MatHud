from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, cast

from constants import default_color, default_point_size
from drawables.drawable import Drawable
from expression_validator import ExpressionValidator
from utils.math_utils import MathUtils


class Function(Drawable):
    def __init__(self, function_string: str, name: Optional[str] = None, step: float = default_point_size, color: str = default_color, left_bound: Optional[float] = None, right_bound: Optional[float] = None, vertical_asymptotes: Optional[List[float]] = None, horizontal_asymptotes: Optional[List[float]] = None, point_discontinuities: Optional[List[float]] = None, is_periodic: Optional[bool] = None, estimated_period: Optional[float] = None) -> None:
        self.step: float = step
        self.left_bound: Optional[float] = left_bound
        self.right_bound: Optional[float] = right_bound
        self.is_periodic: bool = False
        self.estimated_period: Optional[float] = None
        try:
            self.function_string = ExpressionValidator.fix_math_expression(function_string)
            self.function = ExpressionValidator.parse_function_string(function_string)
            if vertical_asymptotes is not None:
                self.vertical_asymptotes = vertical_asymptotes
            if horizontal_asymptotes is not None:
                self.horizontal_asymptotes = horizontal_asymptotes
            if point_discontinuities is not None:
                self.point_discontinuities = point_discontinuities
            if vertical_asymptotes is None and horizontal_asymptotes is None and point_discontinuities is None:
                self._calculate_asymptotes_and_discontinuities()
            if is_periodic is not None:
                self.is_periodic = is_periodic
                self.estimated_period = estimated_period
            else:
                self._detect_periodicity()
        except Exception as e:
            raise ValueError(f"Failed to parse function string '{function_string}': {str(e)}")
        super().__init__(name=name or "f", color=color)

    def _detect_periodicity(self) -> None:
        """Detect if function is periodic and estimate its period."""
        range_hint = None
        if self.left_bound is not None and self.right_bound is not None:
            range_hint = abs(self.right_bound - self.left_bound)
        self.is_periodic, self.estimated_period = MathUtils.detect_function_periodicity(
            self.function, range_hint=range_hint
        )
    
    def get_class_name(self) -> str:
        return 'Function'

    def get_state(self) -> Dict[str, Any]:
        function_string: str = self.function_string
        state: Dict[str, Any] = {
            "name": self.name, 
            "args": {
                "function_string": function_string,
                "left_bound": self.left_bound, 
                "right_bound": self.right_bound
            }
        }
        
        # Only include asymptotes and discontinuities lists that have values
        if hasattr(self, 'vertical_asymptotes') and self.vertical_asymptotes:
            state["args"]["vertical_asymptotes"] = self.vertical_asymptotes
        if hasattr(self, 'horizontal_asymptotes') and self.horizontal_asymptotes:
            state["args"]["horizontal_asymptotes"] = self.horizontal_asymptotes
        if hasattr(self, 'point_discontinuities') and self.point_discontinuities:
            state["args"]["point_discontinuities"] = self.point_discontinuities
            
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        if id(self) in memo:
            return cast(Function, memo[id(self)])
        new_function: Function = Function(
            function_string=self.function_string,
            name=self.name,
            step=self.step,
            color=self.color,
            left_bound=self.left_bound,
            right_bound=self.right_bound,
            vertical_asymptotes=self.vertical_asymptotes.copy() if hasattr(self, 'vertical_asymptotes') and self.vertical_asymptotes is not None else None,
            horizontal_asymptotes=self.horizontal_asymptotes.copy() if hasattr(self, 'horizontal_asymptotes') and self.horizontal_asymptotes is not None else None,
            point_discontinuities=self.point_discontinuities.copy() if hasattr(self, 'point_discontinuities') and self.point_discontinuities is not None else None,
            is_periodic=self.is_periodic,
            estimated_period=self.estimated_period,
        )
        memo[id(self)] = new_function
        return new_function

    def translate(self, x_offset: float, y_offset: float) -> None:
        if x_offset == 0 and y_offset == 0:
            return

        # Translate bounds if they exist
        if self.left_bound is not None:
            self.left_bound += x_offset
        if self.right_bound is not None:
            self.right_bound += x_offset

        try:
            # First handle horizontal translation by replacing x with (x - x_offset)
            if x_offset != 0:
                import re
                # Use all allowed functions from ExpressionValidator
                protected_funcs: list[str] = sorted(ExpressionValidator.ALLOWED_FUNCTIONS, key=len, reverse=True)
                
                # Create a regex pattern that matches standalone x while protecting function names
                func_pattern: str = '|'.join(map(re.escape, protected_funcs))
                # Use word boundaries to match standalone 'x'
                pattern: str = rf'\b(x)\b|({func_pattern})'
                
                def replace_match(match: Any) -> str:
                    if match.group(1):  # If it's a standalone 'x'
                        return f'(x - {x_offset})'
                    elif match.group(2):  # If it's a function name
                        return cast(str, match.group(2))  # Return the function name unchanged
                    return cast(str, match.group(0))
                    
                new_function_string: str = re.sub(pattern, replace_match, self.function_string)
            else:
                new_function_string = self.function_string

            # Then handle vertical translation by adding y_offset
            if y_offset != 0:
                new_function_string = f"({new_function_string}) + {y_offset}"

            # Update function string and parse new function
            self.function_string = ExpressionValidator.fix_math_expression(new_function_string)
            self.function = ExpressionValidator.parse_function_string(new_function_string)
            
        except Exception as e:
            print(f"Warning: Could not translate function: {str(e)}")
            # If translation fails, revert bounds
            if self.left_bound is not None:
                self.left_bound -= x_offset
            if self.right_bound is not None:
                self.right_bound -= x_offset

    def rotate(self, angle: float) -> None:
        pass 

    def update_color(self, color: str) -> None:
        """Update the function color metadata."""
        self.color = str(color)

    def update_left_bound(self, left_bound: Optional[float]) -> None:
        """Update the left bound (None clears the bound)."""
        self.left_bound = None if left_bound is None else float(left_bound)

    def update_right_bound(self, right_bound: Optional[float]) -> None:
        """Update the right bound (None clears the bound)."""
        self.right_bound = None if right_bound is None else float(right_bound)

    def _calculate_asymptotes_and_discontinuities(self) -> None:
        """Calculate vertical and horizontal asymptotes and point discontinuities of the function"""
        from utils.math_utils import MathUtils
        
        # Calculate asymptotes and discontinuities using MathUtil
        self.vertical_asymptotes, self.horizontal_asymptotes, self.point_discontinuities = MathUtils.calculate_asymptotes_and_discontinuities(
            self.function_string,
            self.left_bound,
            self.right_bound
        )

    def has_point_discontinuity_between_x(self, x1: float, x2: float) -> bool:
        """Check if there is a point discontinuity between x1 and x2"""
        return (hasattr(self, 'point_discontinuities') and any(x1 < x < x2 for x in self.point_discontinuities))
    
    def has_vertical_asymptote_between_x(self, x1: float, x2: float) -> bool:
        """Check if there is a vertical asymptote between x1 and x2"""
        return (hasattr(self, 'vertical_asymptotes') and any(x1 <= x < x2 for x in self.vertical_asymptotes))

    def get_vertical_asymptote_between_x(self, x1: float, x2: float) -> Optional[float]:
        """Get the x value of a vertical asymptote between x1 and x2, if any exists"""
        if hasattr(self, 'vertical_asymptotes'):
            for x in self.vertical_asymptotes:
                if x1 <= x < x2:
                    return x
        return None
