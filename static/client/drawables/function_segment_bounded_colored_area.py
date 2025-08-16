"""
MatHud Function Segment Bounded Colored Area

Represents a colored area bounded by a mathematical function and a line segment.
Provides area geometry between a function and a segment in math space; the renderer maps to screen.

Key Features:
	- Function-to-segment area visualization
	- Support for function objects, constants, and x-axis boundaries
	- Math-space boundary intersection calculation

Dependencies:
	- drawables.colored_area: Base class for area visualization
	- drawables.function: Function objects for boundary definitions
"""

from drawables.colored_area import ColoredArea
from drawables.function import Function
from utils.math_utils import MathUtils

class FunctionSegmentBoundedColoredArea(ColoredArea):
	"""Creates a colored area bounded by a mathematical function and a line segment.
	
	This class creates a visual representation of the area between a function
	and a segment using math-space geometry. The renderer handles mapping to screen.
	
	Attributes:
		func (Function, None, or number): The bounding function
		segment (Segment): The bounding line segment
	"""
	def __init__(self, func, segment, color="lightblue", opacity=0.3):
		"""Initialize a function segment bounded colored area.
		
		Args:
			func (Function, None, or number): The bounding function
			segment (Segment): The bounding line segment
			color (str): CSS color value for area fill
			opacity (float): Opacity value between 0.0 and 1.0
		"""
		name = self._generate_name(func, segment)
		super().__init__(name=name, color=color, opacity=opacity)
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

	def _is_function_like(self, obj):
		"""Check if an object has the necessary attributes to be treated as a function (duck typing)."""
		required_attrs = ['name', 'function']
		return all(hasattr(obj, attr) for attr in required_attrs)

	def _get_function_y_at_x(self, x):
		"""Get y value for a given x from the function. Returns math coordinates."""
		if self.func is None:  # x-axis
			return 0  # Math coordinate: y = 0
		if isinstance(self.func, (int, float)):  # constant function
			return float(self.func)  # Math coordinate: y = constant
		if isinstance(self.func, Function) or self._is_function_like(self.func):
			return self._calculate_function_y_value(x)  # Math coordinate
		return None

	def _calculate_function_y_value(self, x):
		"""Calculate y value for Function objects with coordinate conversion."""
		try:
			# x is already in math coordinates
			y = self.func.function(x)
			# Return math coordinate
			return y
		except (ValueError, ZeroDivisionError):
			return None

	def _get_bounds(self):
		"""Calculate the left and right bounds for the colored area."""
		# Get segment bounds
		seg_left, seg_right = self._get_segment_bounds()
		
		# For function bounds
		if isinstance(self.func, Function) or self._is_function_like(self.func):
			return self._get_intersection_bounds(seg_left, seg_right)
		else:
			# For x-axis or constant function, use segment bounds
			return seg_left, seg_right

	def _get_segment_bounds(self):
		"""Get the left and right bounds of the segment."""
		# Use math-space x directly from points
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

	def _generate_segment_points(self):
		"""Generate points for the segment path (in reverse order)."""
		return [(self.segment.point2.x, self.segment.point2.y),
		       (self.segment.point1.x, self.segment.point1.y)]

	def _generate_function_points(self, left_bound, right_bound, num_points, dx):
		"""Generate math-space points; renderer does mapping."""
		points = []
		for i in range(num_points):
			x_math = left_bound + i * dx
			y_math = self._get_function_y_at_x(x_math)
			if y_math is not None:
				points.append((x_math, y_math))
		return points

	def uses_segment(self, segment):
		"""Check if this colored area uses a specific segment.

		Supports comparison in math space.
		"""
		try:
			def same_point(ax, ay, bx, by):
				return abs(ax - bx) < MathUtils.EPSILON and abs(ay - by) < MathUtils.EPSILON

			# Math space comparison (both orders)
			a1x, a1y = self.segment.point1.x, self.segment.point1.y
			a2x, a2y = self.segment.point2.x, self.segment.point2.y
			b1x, b1y = segment.point1.x, segment.point1.y
			b2x, b2y = segment.point2.x, segment.point2.y

			if (same_point(a1x, a1y, b1x, b1y) and same_point(a2x, a2y, b2x, b2y)):
				return True
			if (same_point(a1x, a1y, b2x, b2y) and same_point(a2x, a2y, b1x, b1y)):
				return True
		except Exception:
			return False
		return False

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
			color=self.color,
			opacity=self.opacity
		)
		memo[id(self)] = new_area
		return new_area 