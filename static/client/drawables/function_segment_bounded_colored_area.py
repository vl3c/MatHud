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

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from drawables.colored_area import ColoredArea
from drawables.function import Function
from drawables.segment import Segment
from utils.math_utils import MathUtils

class FunctionSegmentBoundedColoredArea(ColoredArea):
	"""Creates a colored area bounded by a mathematical function and a line segment.

	This class creates a visual representation of the area between a function
	and a segment using math-space geometry. The renderer handles mapping to screen.

	Attributes:
		func (Function, None, or number): The bounding function
		segment (Segment): The bounding line segment
	"""
	def __init__(self, func: Union[Function, None, float, int], segment: Segment, color: str = "lightblue", opacity: float = 0.3) -> None:
		"""Initialize a function segment bounded colored area.

		Args:
			func (Function, None, or number): The bounding function
			segment (Segment): The bounding line segment
			color (str): CSS color value for area fill
			opacity (float): Opacity value between 0.0 and 1.0
		"""
		name = self._generate_name(func, segment)
		super().__init__(name=name, color=color, opacity=opacity)
		self.func: Union[Function, None, float, int] = func
		self.segment: Segment = segment

	def _generate_name(self, func: Union[Function, None, float, int], segment: Segment) -> str:
		"""Generate a descriptive name for the colored area."""
		f_name: str = self._get_function_display_name(func)
		s_name: str = segment.name
		return f"area_between_{f_name}_and_{s_name}"

	def _get_function_display_name(self, func: Union[Function, None, float, int]) -> str:
		"""Extract function name for display purposes."""
		if isinstance(func, Function):
			return str(func.name)
		elif func is not None and not isinstance(func, (int, float)) and self._is_function_like(func):
			return cast(str, func.name)
		elif func is None:
			return 'x_axis'
		else:
			return f'y_{func}'

	def get_class_name(self) -> str:
		"""Return the class name 'FunctionSegmentBoundedColoredArea'."""
		return 'FunctionSegmentBoundedColoredArea'

	def _is_function_like(self, obj: Any) -> bool:
		"""Check if an object has the necessary attributes to be treated as a function (duck typing)."""
		required_attrs = ['name', 'function']
		return all(hasattr(obj, attr) for attr in required_attrs)

	def _get_function_y_at_x(self, x: float) -> Optional[float]:
		"""Get y value for a given x from the function. Returns math coordinates."""
		if self.func is None:  # x-axis
			return 0  # Math coordinate: y = 0
		if isinstance(self.func, (int, float)):  # constant function
			return float(self.func)  # Math coordinate: y = constant
		if isinstance(self.func, Function) or self._is_function_like(self.func):
			return self._calculate_function_y_value(x)  # Math coordinate
		return None

	def _calculate_function_y_value(self, x: float) -> Optional[float]:
		"""Calculate y value for Function objects with coordinate conversion."""
		try:
			if not isinstance(self.func, Function) and not self._is_function_like(self.func):
				return None
			if isinstance(self.func, (int, float)) or self.func is None:
				return None
			# x is already in math coordinates
			y: Any = self.func.function(x)
			# Return math coordinate
			if isinstance(y, (int, float)):
				result: float = float(y)
				if isinstance(result, float) and (result != result or abs(result) == float('inf')):
					return None
				return result
			return None
		except (ValueError, ZeroDivisionError):
			return None

	def _get_bounds(self) -> Tuple[float, float]:
		"""Calculate the left and right bounds for the colored area."""
		# Get segment bounds
		seg_left: float
		seg_right: float
		seg_left, seg_right = self._get_segment_bounds()

		# For function bounds
		if isinstance(self.func, Function) or self._is_function_like(self.func):
			return self._get_intersection_bounds(seg_left, seg_right)
		else:
			# For x-axis or constant function, use segment bounds
			return seg_left, seg_right

	def _get_segment_bounds(self) -> Tuple[float, float]:
		"""Get the left and right bounds of the segment."""
		# Use math-space x directly from points
		x1: float
		x2: float
		x1, x2 = self.segment.point1.x, self.segment.point2.x
		return min(x1, x2), max(x1, x2)

	def _get_intersection_bounds(self, seg_left: float, seg_right: float) -> Tuple[float, float]:
		"""Get intersection of segment and function bounds."""
		if not isinstance(self.func, Function) and not self._is_function_like(self.func):
			return seg_left, seg_right
		if isinstance(self.func, (int, float)) or self.func is None:
			return seg_left, seg_right
		func_left: Optional[float] = self.func.left_bound if hasattr(self.func, 'left_bound') else None
		func_right: Optional[float] = self.func.right_bound if hasattr(self.func, 'right_bound') else None
		# Use intersection of bounds
		left_bound: float = max(seg_left, func_left) if func_left is not None else seg_left
		right_bound: float = min(seg_right, func_right) if func_right is not None else seg_right
		return left_bound, right_bound

	def _generate_segment_points(self) -> List[Tuple[float, float]]:
		"""Generate points for the segment path (in reverse order)."""
		return [(self.segment.point2.x, self.segment.point2.y),
		       (self.segment.point1.x, self.segment.point1.y)]

	def _generate_function_points(self, left_bound: float, right_bound: float, num_points: int, dx: float) -> List[Tuple[float, float]]:
		"""Generate math-space points; renderer does mapping."""
		points: List[Tuple[float, float]] = []
		for i in range(num_points):
			x_math: float = left_bound + i * dx
			y_math: Optional[float] = self._get_function_y_at_x(x_math)
			if y_math is not None:
				points.append((x_math, y_math))
		return points

	def uses_segment(self, segment: Segment) -> bool:
		"""Check if this colored area uses a specific segment.

		Supports comparison in math space.
		"""
		try:
			def same_point(ax: float, ay: float, bx: float, by: float) -> bool:
				return cast(bool, abs(ax - bx) < MathUtils.EPSILON and abs(ay - by) < MathUtils.EPSILON)

			# Math space comparison (both orders)
			a1x: float
			a1y: float
			a2x: float
			a2y: float
			b1x: float
			b1y: float
			b2x: float
			b2y: float
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

	def get_state(self) -> Dict[str, Any]:
		"""Serialize function segment bounded area state for persistence."""
		state: Dict[str, Any] = super().get_state()
		func_name: str
		if isinstance(self.func, Function) or self._is_function_like(self.func):
			if isinstance(self.func, (int, float)) or self.func is None:
				func_name = str(self.func)
			else:
				func_name = cast(str, self.func.name)
		else:
			func_name = str(self.func)
		state["args"].update({
			"func": func_name,
			"segment": self.segment.name
		})
		return state

	def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
		"""Create a deep copy for undo/redo functionality."""
		if id(self) in memo:
			return cast(FunctionSegmentBoundedColoredArea, memo[id(self)])

		new_area: FunctionSegmentBoundedColoredArea = FunctionSegmentBoundedColoredArea(
			func=copy.deepcopy(self.func, memo),
			segment=copy.deepcopy(self.segment, memo),
			color=self.color,
			opacity=self.opacity
		)
		new_area.name = self.name
		memo[id(self)] = new_area
		return new_area
