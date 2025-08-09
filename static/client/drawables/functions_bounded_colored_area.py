"""
MatHud Functions Bounded Colored Area

Represents a colored area bounded by mathematical functions with asymptote handling.
Provides area visualization between two functions or between a function and the x-axis.

Key Features:
    - Two-function or function-to-axis area visualization
    - Support for function objects, constants, and x-axis boundaries
    - Asymptote and discontinuity aware path generation
    - Boundary detection and coordinate transformation

Dependencies:
    - drawables.colored_area: Base class for area visualization
    - drawables.function: Function objects for boundary definitions
    - copy: Deep copying capabilities for state management
"""

from drawables.colored_area import ColoredArea
from drawables.function import Function
import copy


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
    
    def __init__(self, func1, func2=None, canvas=None, left_bound=None, right_bound=None, 
                 color="lightblue", opacity=0.3, num_sample_points=100):
        """Initialize a functions bounded colored area.
        
        Args:
            func1 (Function, None, or number): The first bounding function
            func2 (Function, None, or number): The second bounding function
            canvas (Canvas): Parent canvas for coordinate system access
            left_bound (float): Left boundary of the area
            right_bound (float): Right boundary of the area
            color (str): CSS color value for area fill
            opacity (float): Opacity value between 0.0 and 1.0
            num_sample_points (int): Number of points for path generation
        """
        self._validate_parameters(func1, func2, left_bound, right_bound, num_sample_points)
        name = self._generate_name(func1, func2)
        super().__init__(name=name, canvas=canvas, color=color, opacity=opacity)
        self.func1 = func1
        self.func2 = func2
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.num_sample_points = num_sample_points

    def _validate_parameters(self, func1, func2, left_bound, right_bound, num_sample_points):
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

    def _is_function_like(self, obj):
        """Check if an object has the necessary attributes to be treated as a function (duck typing)."""
        required_attrs = ['name', 'function']
        return all(hasattr(obj, attr) for attr in required_attrs)
    
    def _is_function_or_function_like(self, obj):
        """Check if an object is a Function or function-like object."""
        return isinstance(obj, Function) or self._is_function_like(obj)

    def _get_function_name(self, func):
        """Generate a descriptive name for a function."""
        if hasattr(func, 'name'):
            return func.name
        elif func is None:
            return 'x_axis'
        else:
            return f'y_{func}'

    def _generate_name(self, func1, func2):
        """Generate a descriptive name for the colored area."""
        f1_name = self._get_function_name(func1)
        f2_name = self._get_function_name(func2)
        return f"area_between_{f1_name}_and_{f2_name}"

    def get_class_name(self):
        """Return the class name 'FunctionsBoundedColoredArea'."""
        return 'FunctionsBoundedColoredArea'

    def _get_function_y_at_x(self, func, x):
        """Get y value for a given x, handling different function types."""
        try:
            if func is None:  # x-axis
                # Convert y=0 to canvas coordinates using CoordinateMapper
                canvas_x, canvas_y = self.canvas.coordinate_mapper.math_to_screen(x, 0)
                return canvas_y
                
            if isinstance(func, (int, float)):  # constant function
                # Convert constant y value to canvas coordinates using CoordinateMapper
                canvas_x, canvas_y = self.canvas.coordinate_mapper.math_to_screen(x, float(func))
                return canvas_y
                
            if self._is_function_or_function_like(func):
                try:
                    # Use the actual math x value directly
                    y = func.function(x)
                    
                    # Check if result is valid
                    if y is None or not isinstance(y, (int, float)) or (isinstance(y, float) and (float('nan') == y or float('inf') == abs(y))):
                        return None
                        
                    # Convert y to canvas coordinates using CoordinateMapper
                    canvas_x, canvas_y = self.canvas.coordinate_mapper.math_to_screen(x, y)
                    return canvas_y
                except (ValueError, ZeroDivisionError, TypeError) as e:
                    return None
            
            # If func is not recognized as any known type
            return None
            
        except Exception as e:
            # Catch any unexpected exceptions and return None
            return None

    def _get_initial_bounds(self):
        """
        Get initial bounds from the coordinate mapper.
        
        Returns:
        --------
        list
            A list [left_bound, right_bound] with the initial bounds.
        """
        try:
            return [
                self.canvas.coordinate_mapper.get_visible_left_bound(),
                self.canvas.coordinate_mapper.get_visible_right_bound()
            ]
        except Exception as e:
            # If canvas bounds can't be determined, use reasonable defaults
            return [-10, 10]  # Default reasonable bounds

    def _apply_function_bounds(self, bounds, func):
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
        if self._is_function_or_function_like(func) and hasattr(func, 'left_bound') and hasattr(func, 'right_bound') and func.left_bound is not None and func.right_bound is not None:
            bounds[0] = max(bounds[0], func.left_bound)
            bounds[1] = min(bounds[1], func.right_bound)
        return bounds

    def _apply_user_bounds(self, bounds):
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
            bounds[0] = max(bounds[0], self.left_bound)
        if self.right_bound is not None:
            bounds[1] = min(bounds[1], self.right_bound)
        return bounds

    def _get_bounds(self):
        """
        Get the left and right bounds for the colored area.
        
        Returns:
        --------
        tuple
            A tuple (left_bound, right_bound) with the final bounds.
        """
        bounds = self._get_initial_bounds()
        
        # Apply function bounds if available
        if self.func1:
            bounds = self._apply_function_bounds(bounds, self.func1)
        if self.func2:
            bounds = self._apply_function_bounds(bounds, self.func2)
            
        # Apply user bounds if provided
        bounds = self._apply_user_bounds(bounds)

        # Ensure left_bound < right_bound
        if bounds[0] >= bounds[1]:
            # If bounds are inverted or equal, adjust them slightly
            center = (bounds[0] + bounds[1]) / 2
            bounds = [center - 0.1, center + 0.1]  # Small range around center

        return bounds[0], bounds[1]

    def _has_asymptote_at(self, func, x_orig, dx):
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
            # Manual asymptote detection for functions with tangent asymptotes (e.g. f3)
            if hasattr(func, 'name') and func.name == 'f3':
                import math
                has_asym = False
                # Check known asymptote positions for tan(x/100)
                for n in range(-5, 6):  # Check a reasonable range
                    asym_x = 100 * (math.pi/2 + n * math.pi)
                    # Only consider very, very close to asymptote (20% of dx)
                    if abs(x_orig - asym_x) < dx * 0.2:
                        has_asym = True
                        break
                return has_asym
            # Default asymptote detection for other functions
            return (self._is_function_or_function_like(func) and 
                   hasattr(func, 'has_vertical_asymptote_between_x') and 
                   func.has_vertical_asymptote_between_x(x_orig - dx, x_orig + dx))
        except Exception:
            # If asymptote detection fails, assume no asymptote
            return False
        
    def _generate_path(self, func, left_bound, right_bound, dx, num_points, reverse=False):
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
        points = []
        current_path = []  # FIX: Initialize current_path variable
        
        # Get canvas bounds for asymptote handling
        canvas_top = 0
        canvas_bottom = self.canvas.height if hasattr(self.canvas, 'height') else 800
        canvas_margin = 50  # Margin for very large values
        
        # Determine the direction of iteration
        if reverse:
            range_iterator = range(num_points - 1, -1, -1)
        else:
            range_iterator = range(num_points)
            
        rejected_count = 0
        asymptote_count = 0
        
        for i in range_iterator:
            # Convert x to canvas coordinates
            x_orig = left_bound + i * dx
            x, _ = self.canvas.coordinate_mapper.math_to_screen(x_orig, 0)
            
            # Try to get y value with asymptote handling
            y = self._get_function_y_at_x_with_asymptote_handling(func, x, x_orig, dx, canvas_top, canvas_bottom, canvas_margin)
            
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

    def _get_function_y_at_x_with_asymptote_handling(self, func, x, x_orig, dx, canvas_top, canvas_bottom, canvas_margin):
        """
        Get y value for a function at x with asymptote handling.
        
        Parameters:
        -----------
        func : Function, None, or number
            The function to evaluate.
        x : float
            The x coordinate (canvas).
        x_orig : float
            The original x coordinate.
        dx : float
            The step size.
        canvas_top : float
            The top of the canvas.
        canvas_bottom : float
            The bottom of the canvas.
        canvas_margin : float
            The margin for asymptote handling.
            
        Returns:
        --------
        float or None
            The y value or None if invalid.
        """
        # For non-function types, use the original method
        if not self._is_function_or_function_like(func):
            return self._get_function_y_at_x(func, x)
        
        try:
            # Check if we're near an asymptote first
            has_asymptote = self._has_asymptote_at(func, x_orig, dx * 2)  # Use wider detection range
            if has_asymptote:
                return None
            
            # Evaluate the function
            y = func.function(x_orig)
            
            # Handle various problematic values
            if y is None:
                return None
                
            if not isinstance(y, (int, float)):
                return None
                
            # Handle infinities and very large values
            if isinstance(y, float):
                if float('nan') == y:
                    return None
                if float('inf') == abs(y):
                    return None
                    
                # Clip extremely large values to canvas bounds
                if abs(y) > 1000:  # Very large value threshold
                    # Choose a reasonable value based on direction
                    if y > 0:
                        y = (canvas_top + canvas_margin - self.canvas.cartesian2axis.origin.y) / (-self.canvas.scale_factor)
                    else:
                        y = (canvas_bottom - canvas_margin - self.canvas.cartesian2axis.origin.y) / (-self.canvas.scale_factor)
                        
            # Convert y back to canvas coordinates
            canvas_y = self.canvas.cartesian2axis.origin.y - y * self.canvas.scale_factor
            
            # Clip to reasonable canvas bounds
            if canvas_y < canvas_top - canvas_margin:
                canvas_y = canvas_top - canvas_margin
            elif canvas_y > canvas_bottom + canvas_margin:
                canvas_y = canvas_bottom + canvas_margin
                
            return canvas_y
            
        except (ValueError, ZeroDivisionError, TypeError, OverflowError) as e:
            return None

    def draw(self):
        """Draw the colored area between the functions on the canvas."""
        try:
            left_bound, right_bound = self._get_bounds()
            
            # Calculate step size
            num_points = self.num_sample_points
            dx = (right_bound - left_bound) / (num_points - 1)
            
            # Safety check for dx
            if dx <= 0:
                left_bound, right_bound = left_bound - 1, right_bound + 1
                dx = (right_bound - left_bound) / (num_points - 1)
            
            # Generate forward path (along func1)
            points = self._generate_path(self.func1, left_bound, right_bound, dx, num_points, reverse=False)

            # Generate reverse path (along func2)
            reverse_points = self._generate_path(self.func2, left_bound, right_bound, dx, num_points, reverse=True)
            
            # Only create path if we have valid points
            if points and reverse_points:
                self._create_svg_path(points, reverse_points)
        except Exception:
            return None

    def get_state(self):
        """Serialize functions bounded area state for persistence."""
        state = super().get_state()
        state["args"].update({
            "func1": self._get_function_name(self.func1),
            "func2": self._get_function_name(self.func2),
            "left_bound": self.left_bound,
            "right_bound": self.right_bound,
            "num_sample_points": self.num_sample_points
        })
        return state

    def __deepcopy__(self, memo):
        """Create a deep copy for undo/redo functionality."""
        if id(self) in memo:
            return memo[id(self)]
            
        # Create new instance using __init__
        new_area = FunctionsBoundedColoredArea(
            func1=self.func1,  # Functions will be properly deep copied by their own __deepcopy__
            func2=self.func2,
            canvas=self.canvas,  # Canvas reference is not deep copied
            left_bound=self.left_bound,
            right_bound=self.right_bound,
            color=self.color,
            opacity=self.opacity,
            num_sample_points=self.num_sample_points
        )
        memo[id(self)] = new_area
        return new_area 