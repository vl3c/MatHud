"""
MatHud Function Segment Bounded Colored Area

Represents a colored area bounded by a mathematical function and a line segment.
Provides area visualization between a function and a segment with coordinate transformation.

Key Features:
    - Function-to-segment area visualization
    - Support for function objects, constants, and x-axis boundaries
    - Coordinate transformation between canvas and mathematical space
    - Boundary intersection calculation

Dependencies:
    - drawables.colored_area: Base class for area visualization
    - drawables.function: Function objects for boundary definitions
"""

from drawables.colored_area import ColoredArea
from drawables.function import Function

class FunctionSegmentBoundedColoredArea(ColoredArea):
    """Creates a colored area bounded by a mathematical function and a line segment.
    
    This class creates a visual representation of the area between a function
    and a segment with coordinate transformation support.
    
    Attributes:
        func (Function, None, or number): The bounding function
        segment (Segment): The bounding line segment
    """
    def __init__(self, func, segment, canvas=None, color="lightblue", opacity=0.3):
        """Initialize a function segment bounded colored area.
        
        Args:
            func (Function, None, or number): The bounding function
            segment (Segment): The bounding line segment
            canvas (Canvas): Parent canvas for coordinate system access
            color (str): CSS color value for area fill
            opacity (float): Opacity value between 0.0 and 1.0
        """
        name = self._generate_name(func, segment)
        super().__init__(name=name, canvas=canvas, color=color, opacity=opacity)
        self.func = func
        self.segment = segment

    def _generate_name(self, func, segment):
        """Generate a descriptive name for the colored area."""
        f_name = self._get_function_display_name(func)
        s_name = segment.name
        return f"area_between_{f_name}_and_{s_name}"

    def _get_function_display_name(self, func):
        """Extract function name for display purposes."""
        if hasattr(func, 'name'):
            return func.name
        elif func is None:
            return 'x_axis'
        else:
            return f'y_{func}'

    def get_class_name(self):
        """Return the class name 'FunctionSegmentBoundedColoredArea'."""
        return 'FunctionSegmentBoundedColoredArea'

    def _get_function_y_at_x(self, x):
        """Get y value for a given x from the function."""
        if self.func is None:  # x-axis
            return 0
        if isinstance(self.func, (int, float)):  # constant function
            return float(self.func)
        if isinstance(self.func, Function):
            return self._calculate_function_y_value(x)
        return None

    def _calculate_function_y_value(self, x):
        """Calculate y value for Function objects with coordinate conversion."""
        try:
            # Convert from canvas coordinates to original coordinates
            orig_x = self.canvas.coordinate_mapper.screen_to_math(x, 0)[0]
            y = self.func.function(orig_x)
            # Convert y back to canvas coordinates
            return self.canvas.coordinate_mapper.math_to_screen(orig_x, y)[1]
        except (ValueError, ZeroDivisionError):
            return None

    def _get_bounds(self):
        """Calculate the left and right bounds for the colored area."""
        # Get segment bounds
        seg_left, seg_right = self._get_segment_bounds()
        
        # For function bounds
        if isinstance(self.func, Function):
            return self._get_intersection_bounds(seg_left, seg_right)
        else:
            # For x-axis or constant function, use segment bounds
            return seg_left, seg_right

    def _get_segment_bounds(self):
        """Get the left and right bounds of the segment."""
        x1, x2 = self.segment.point1.x, self.segment.point2.x
        return min(x1, x2), max(x1, x2)

    def _get_intersection_bounds(self, seg_left, seg_right):
        """Get intersection of segment and function bounds."""
        func_left = self.func.left_bound
        func_right = self.func.right_bound
        # Use intersection of bounds
        left_bound = max(seg_left, func_left)
        right_bound = min(seg_right, func_right)
        return left_bound, right_bound

    def draw(self):
        """Draw the colored area between the function and segment on the canvas."""
        left_bound, right_bound = self._get_bounds()
        
        # Sample points for the function
        num_points = 100
        dx = (right_bound - left_bound) / (num_points - 1)
        
        # Get path points
        forward_points = self._generate_function_points(left_bound, right_bound, num_points, dx)
        reverse_points = self._generate_segment_points()
        
        # Create SVG path using base class method
        self._create_svg_path(forward_points, reverse_points)

    def _generate_segment_points(self):
        """Generate points for the segment path (in reverse order)."""
        return [(self.segment.point2.x, self.segment.point2.y),
               (self.segment.point1.x, self.segment.point1.y)]

    def _generate_function_points(self, left_bound, right_bound, num_points, dx):
        """Generate points along the function curve from left to right bound."""
        points = []
        for i in range(num_points):
            x = left_bound + i * dx
            y = self._get_function_y_at_x(x)
            if y is not None:
                points.append((x, y))
        return points

    def uses_segment(self, segment):
        """Check if this colored area uses a specific segment."""
        return (self.segment.point1.x == segment.point1.x and 
                self.segment.point1.y == segment.point1.y and
                self.segment.point2.x == segment.point2.x and
                self.segment.point2.y == segment.point2.y)

    def get_state(self):
        """Serialize function segment bounded area state for persistence."""
        state = super().get_state()
        state["args"].update({
            "func": self.func.name if hasattr(self.func, 'name') else str(self.func),
            "segment": self.segment.name
        })
        return state

    def __deepcopy__(self, memo):
        """Create a deep copy for undo/redo functionality."""
        if id(self) in memo:
            return memo[id(self)]
            
        # Create new instance using __init__
        new_area = FunctionSegmentBoundedColoredArea(
            func=self.func,  # Function will be properly deep copied by its own __deepcopy__
            segment=self.segment,  # Segment will be properly deep copied by its own __deepcopy__
            canvas=self.canvas,  # Canvas reference is not deep copied
            color=self.color,
            opacity=self.opacity
        )
        memo[id(self)] = new_area
        return new_area 