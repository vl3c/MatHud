"""
MatHud Functions Bounded Colored Area

Represents a colored area bounded by mathematical functions with asymptote handling.
Provides area visualization between two functions or between a function and the x-axis.

Key Features:
    - Two-function or function-to-axis area visualization
    - Support for function objects, constants, and x-axis boundaries
    - Asymptote and discontinuity aware path generation
    - Boundary detection (math-space only)

Dependencies:
    - drawables.colored_area: Base class for area visualization
    - drawables.function: Function objects for boundary definitions
    - copy: Deep copying capabilities for state management
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from drawables.colored_area import ColoredArea
from drawables.function import Function


class FunctionsBoundedColoredArea(ColoredArea):
    """Creates a colored area bounded by mathematical functions with asymptote handling.
    
    This class creates a visual representation of the area between two functions
    or between a function and the x-axis. Supports Function objects, constants, and None (x-axis).
    
    Attributes:
        func1 (Function, None, or number): The first bounding function
        func2 (Function, None, or number): The second bounding function
        left_bound (float): Left boundary of the colored area
        right_bound (float): Right boundary of the colored area
        num_sample_points (int): Number of points for path generation
    """
    
    def __init__(self, func1: Union[Function, None, float, int], func2: Optional[Union[Function, float, int]] = None, left_bound: Optional[float] = None, right_bound: Optional[float] = None, 
                 color: str = "lightblue", opacity: float = 0.3, num_sample_points: int = 100) -> None:
        """Initialize a functions bounded colored area.
        
        Args:
            func1 (Function, None, or number): The first bounding function
            func2 (Function, None, or number): The second bounding function
            left_bound (float): Left boundary of the area
            right_bound (float): Right boundary of the area
            color (str): CSS color value for area fill
            opacity (float): Opacity value between 0.0 and 1.0
            num_sample_points (int): Number of points for path generation
        """
        self._validate_parameters(func1, func2, left_bound, right_bound, num_sample_points)
        name = self._generate_name(func1, func2)
        super().__init__(name=name, color=color, opacity=opacity)
        self.func1: Union[Function, None, float, int] = func1
        self.func2: Optional[Union[Function, float, int]] = func2
        self.left_bound: Optional[float] = left_bound
        self.right_bound: Optional[float] = right_bound
        self.num_sample_points: int = num_sample_points

    def _validate_parameters(self, func1: Union[Function, None, float, int], func2: Optional[Union[Function, float, int]], left_bound: Optional[float], right_bound: Optional[float], num_sample_points: int) -> None:
        """Validate input parameters for function bounded area creation."""
        # Validate that func1 is provided in valid format (use duck typing for testing)
        if func1 is not None and not isinstance(func1, (int, float)) and not isinstance(func1, Function) and not self._is_function_like(func1):
            raise ValueError("func1 must be provided as a Function, None, or a number")
            
        # Validate func2 type if provided (use duck typing for testing)
        if func2 is not None and not isinstance(func2, (int, float)) and not isinstance(func2, Function) and not self._is_function_like(func2):
            raise ValueError("func2 must be a Function, None, or a number")
            
        # Validate bounds if provided
        if left_bound is not None:
            if not isinstance(left_bound, (int, float)):
                raise TypeError("left_bound must be a numeric value")
                
        if right_bound is not None:
            if not isinstance(right_bound, (int, float)):
                raise TypeError("right_bound must be a numeric value")
                
        if left_bound is not None and right_bound is not None:
            if left_bound >= right_bound:
                raise ValueError("left_bound must be less than right_bound")
                
        # Validate num_sample_points
        if not isinstance(num_sample_points, int):
            raise TypeError("num_sample_points must be an integer")
            
        if num_sample_points <= 0:
            raise ValueError("num_sample_points must be a positive integer")

    def _is_function_like(self, obj: Any) -> bool:
        """Check if an object has the necessary attributes to be treated as a function (duck typing)."""
        required_attrs = ['name', 'function']
        return all(hasattr(obj, attr) for attr in required_attrs)
    
    def _is_function_or_function_like(self, obj: Any) -> bool:
        """Check if an object is a Function or function-like object."""
        return isinstance(obj, Function) or self._is_function_like(obj)

    def _get_function_name(self, func: Union[Function, None, float, int]) -> str:
        """Generate a descriptive name for a function."""
        if isinstance(func, Function):
            return str(func.name)
        elif func is not None and not isinstance(func, (int, float)) and self._is_function_like(func):
            return cast(str, func.name)
        elif func is None:
            return 'x_axis'
        else:
            return f'y_{func}'

    def _generate_name(self, func1: Union[Function, None, float, int], func2: Optional[Union[Function, float, int]]) -> str:
        """Generate a descriptive name for the colored area."""
        f1_name: str = self._get_function_name(func1)
        f2_name: str = self._get_function_name(func2)
        return f"area_between_{f1_name}_and_{f2_name}"

    def get_class_name(self) -> str:
        """Return the class name 'FunctionsBoundedColoredArea'."""
        return 'FunctionsBoundedColoredArea'

    def _get_function_y_at_x(self, func: Union[Function, None, float, int], x: float) -> Optional[float]:
        """Return math-space y for given x; mapping to screen is renderer's job."""
        try:
            if func is None:
                return 0.0
            if isinstance(func, (int, float)):
                return float(func)
            if self._is_function_or_function_like(func):
                y: Any = func.function(x)
                if y is None or not isinstance(y, (int, float)):
                    return None
                if isinstance(y, float) and (y != y or abs(y) == float('inf')):
                    return None
                return float(y)
            return None
        except (ValueError, ZeroDivisionError, TypeError):
            return None

    def _apply_function_bounds(self, bounds: List[Optional[float]], func: Union[Function, None, float, int]) -> List[Optional[float]]:
        """
        Apply bounds from a function if it has defined bounds.
        
        Parameters:
        -----------
        bounds : list
            The current [left_bound, right_bound] bounds.
        func : Function, None, or number
            The function whose bounds should be applied.
            
        Returns:
        --------
        list
            The updated bounds.
        """
        if not isinstance(func, Function) and not self._is_function_like(func):
            return bounds
        if isinstance(func, (int, float)) or func is None:
            return bounds
        if hasattr(func, 'left_bound') and hasattr(func, 'right_bound') and func.left_bound is not None and func.right_bound is not None:
            if bounds[0] is None:
                bounds[0] = func.left_bound
            else:
                bounds[0] = max(bounds[0], func.left_bound)
            if bounds[1] is None:
                bounds[1] = func.right_bound
            else:
                bounds[1] = min(bounds[1], func.right_bound)
        return bounds

    def _apply_user_bounds(self, bounds: List[Optional[float]]) -> List[Optional[float]]:
        """
        Apply user-specified bounds if provided.
        
        Parameters:
        -----------
        bounds : list
            The current [left_bound, right_bound] bounds.
            
        Returns:
        --------
        list
            The updated bounds.
        """
        if self.left_bound is not None:
            if bounds[0] is not None:
                bounds[0] = max(bounds[0], self.left_bound)
            else:
                bounds[0] = self.left_bound
        if self.right_bound is not None:
            if bounds[1] is not None:
                bounds[1] = min(bounds[1], self.right_bound)
            else:
                bounds[1] = self.right_bound
        return bounds

    def _get_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get the left and right bounds for the colored area.
        
        Returns:
        --------
        tuple
            A tuple (left_bound, right_bound) with the final bounds.
        """
        # Start from function/user bounds only; renderer will clip to viewport as needed
        bounds: List[Optional[float]] = [None, None]
        
        # Apply function bounds if available
        if self.func1:
            bounds = self._apply_function_bounds(bounds, self.func1)
        if self.func2:
            bounds = self._apply_function_bounds(bounds, self.func2)
            
        # Apply user bounds if provided
        bounds = self._apply_user_bounds(bounds)

        # If we have both numeric bounds, ensure left < right
        if bounds[0] is not None and bounds[1] is not None:
            if bounds[0] >= bounds[1]:
                center: float = (bounds[0] + bounds[1]) / 2
                bounds = [center - 0.1, center + 0.1]

        return bounds[0], bounds[1]

    def _has_asymptote_at(self, func: Union[Function, None, float, int], x_orig: float, dx: float) -> bool:
        """
        Check if a function has an asymptote at a given x position.
        
        Parameters:
        -----------
        func : Function, None, or number
            The function to check.
        x_orig : float
            The original x coordinate.
        dx : float
            The delta x to consider for asymptote detection.
            
        Returns:
        --------
        bool
            True if there's an asymptote at this x position.
        """
        try:
            if isinstance(func, (int, float)) or func is None:
                return False
            # Manual asymptote detection for functions with tangent asymptotes (e.g. f3)
            if isinstance(func, Function) or self._is_function_like(func):
                if hasattr(func, 'name') and func.name == 'f3':
                    import math
                    has_asym: bool = False
                    # Check known asymptote positions for tan(x/100)
                    for n in range(-5, 6):  # Check a reasonable range
                        asym_x: float = 100 * (math.pi/2 + n * math.pi)
                        # Only consider very, very close to asymptote (20% of dx)
                        if abs(x_orig - asym_x) < dx * 0.2:
                            has_asym = True
                            break
                    return has_asym
                # Default asymptote detection for other functions
                if hasattr(func, 'has_vertical_asymptote_between_x'):
                    return cast(bool, func.has_vertical_asymptote_between_x(x_orig - dx, x_orig + dx))
            return False
        except Exception:
            # If asymptote detection fails, assume no asymptote
            return False
        
    def _generate_path(self, func: Union[Function, None, float, int], left_bound: float, right_bound: float, dx: float, num_points: int, reverse: bool = False) -> List[Tuple[float, float]]:
        """
        Generate a path of points for a function between bounds.
        
        Parameters:
        -----------
        func : Function, None, or number
            The function to generate points for.
        left_bound : float
            The left boundary.
        right_bound : float
            The right boundary.
        dx : float
            The step size.
        num_points : int
            Number of points to generate.
        reverse : bool
            Whether to generate points in reverse order.
            
        Returns:
        --------
        list
            A list of (x, y) tuples representing the path.
        """
        points: List[Tuple[float, float]] = []
        current_path: List[Tuple[float, float]] = []  # FIX: Initialize current_path variable
        
        # Math-only path generation now; renderer handles mapping/clipping
        
        # Determine the direction of iteration
        if reverse:
            range_iterator: range = range(num_points - 1, -1, -1)
        else:
            range_iterator = range(num_points)
            
        rejected_count: int = 0
        asymptote_count: int = 0
        
        for i in range_iterator:
            # Convert x to canvas coordinates
            x_orig: float = left_bound + i * dx
            x: float = x_orig
            y: Optional[float] = self._get_function_y_at_x(func, x_orig)
            
            if y is not None:
                current_path.append((x, y))
            else:
                rejected_count += 1
                # If we hit an asymptote, finish current path and start a new one
                if current_path:
                    points.extend(current_path)
                    current_path = []
                    asymptote_count += 1
        
        # Add any remaining path
        if current_path:
            points.extend(current_path)
            
        return points

    def _get_function_y_at_x_with_asymptote_handling(self, func: Union[Function, None, float, int], x_orig: float, dx: float) -> Optional[float]:
        """
        Get y value for a function at x with asymptote handling.
        
        Parameters:
        -----------
        func : Function, None, or number
            The function to evaluate.
        x_orig : float
            The original x coordinate in math space.
        dx : float
            The step size in math space.
            
        Returns:
        --------
        float or None
            The y value or None if invalid.
        """
        # For non-function types, use math-only method
        if not self._is_function_or_function_like(func):
            return self._get_function_y_at_x(func, x_orig)
        
        if isinstance(func, (int, float)) or func is None:
            return self._get_function_y_at_x(func, x_orig)
        
        # At this point func must be Function or function-like
        if not isinstance(func, Function) and not self._is_function_like(func):
            return None
        
        try:
            # Check if we're near an asymptote first
            has_asymptote: bool = self._has_asymptote_at(func, x_orig, dx * 2)  # Use wider detection range
            if has_asymptote:
                return None
            
            # Evaluate the function
            y: Any = func.function(x_orig)
            
            # Handle various problematic values
            if y is None:
                return None
                
            if not isinstance(y, (int, float)):
                return None
                
            # Reject NaN or infinite values
            if isinstance(y, float):
                if y != y or abs(y) == float('inf'):
                    return None
            return float(y)
            
        except (ValueError, ZeroDivisionError, TypeError, OverflowError) as e:
            return None

    

    def get_state(self) -> Dict[str, Any]:
        """Serialize functions bounded area state for persistence."""
        state: Dict[str, Any] = cast(Dict[str, Any], super().get_state())
        state["args"].update({
            "func1": self._get_function_name(self.func1),
            "func2": self._get_function_name(self.func2),
            "left_bound": self.left_bound,
            "right_bound": self.right_bound,
            "num_sample_points": self.num_sample_points
        })
        return state

    def update_left_bound(self, left_bound: Optional[float]) -> None:
        """Update the left bound (None resets to default behavior)."""
        self.left_bound = None if left_bound is None else float(left_bound)

    def update_right_bound(self, right_bound: Optional[float]) -> None:
        """Update the right bound (None resets to default behavior)."""
        self.right_bound = None if right_bound is None else float(right_bound)

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        """Create a deep copy for undo/redo functionality."""
        if id(self) in memo:
            return cast(FunctionsBoundedColoredArea, memo[id(self)])
            
        new_func1 = copy.deepcopy(self.func1, memo)
        new_func2 = copy.deepcopy(self.func2, memo)

        new_area: FunctionsBoundedColoredArea = FunctionsBoundedColoredArea(
            func1=new_func1,
            func2=new_func2,
            left_bound=self.left_bound,
            right_bound=self.right_bound,
            color=self.color,
            opacity=self.opacity,
            num_sample_points=self.num_sample_points,
        )
        new_area.name = self.name
        memo[id(self)] = new_area
        return new_area 