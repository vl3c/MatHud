"""
MatHud Line Segment Geometric Object

Represents a line segment between two points in 2D mathematical space.
Provides line equation calculation, visibility detection, and rotation capabilities.

Key Features:
    - Two-endpoint line segment representation
    - Automatic line equation calculation (ax + by + c = 0)
    - Designed to be renderer-agnostic (no viewport logic here)
    - Translation and rotation transformations
    - Midpoint-based rotation around segment center

Mathematical Properties:
    - line_formula: Algebraic line equation coefficients
    - Endpoint coordinate tracking through Point objects
    - Visibility optimization for performance

Dependencies:
    - constants: Default styling values
    - drawables.drawable: Base class interface
    - drawables.position: Coordinate calculations
    - utils.math_utils: Line equation and intersection calculations
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, cast

from constants import default_color, point_label_font_size
from drawables.attached_label import AttachedLabel
from drawables.drawable import Drawable
from drawables.point import Point
from drawables.position import Position
from utils.math_utils import MathUtils

class Segment(Drawable):
    """Represents a line segment between two points with mathematical line properties.

    Maintains references to two Point objects and calculates line equation properties
    for mathematical operations and geometric intersections.

    Attributes:
        point1 (Point): First endpoint of the segment
        point2 (Point): Second endpoint of the segment
        line_formula (dict): Algebraic line equation coefficients (a, b, c for ax + by + c = 0)
    """
    def __init__(
        self,
        p1: Point,
        p2: Point,
        color: str = default_color,
        *,
        label_text: str = "",
        label_visible: bool = False,
    ) -> None:
        """Initialize a line segment between two points.

        Args:
            p1 (Point): First endpoint of the segment
            p2 (Point): Second endpoint of the segment
            color (str): CSS color value for segment visualization
        """
        self.point1: Point = p1
        self.point2: Point = p2
        self.line_formula: Dict[str, float] = self._calculate_line_algebraic_formula()
        name: str = self.point1.name + self.point2.name
        super().__init__(name=name, color=color)
        midpoint_x, midpoint_y = self._get_midpoint()
        self.label = AttachedLabel(
            midpoint_x,
            midpoint_y,
            str(label_text or ""),
            color=self.color,
            font_size=point_label_font_size,
            visible=bool(label_visible),
            text_format="text_only",
        )

    def get_class_name(self) -> str:
        return 'Segment'

    def _calculate_line_algebraic_formula(self) -> Dict[str, float]:
        p1: Point = self.point1
        p2: Point = self.point2
        line_formula: Dict[str, float] = MathUtils.get_line_formula(p1.x, p1.y, p2.x, p2.y)
        return line_formula

    def get_state(self) -> Dict[str, Any]:
        # Keep endpoint ordering consistent with in-memory references so downstream
        # consumers (workspace saves, dependency checks) preserve segment identity.
        self._sync_label_position()
        state: Dict[str, Any] = {
            "name": self.name,
            "args": {
                "p1": self.point1.name,
                "p2": self.point2.name,
            },
            # Include coordinates for render cache invalidation
            "_p1_coords": [self.point1.x, self.point1.y],
            "_p2_coords": [self.point2.x, self.point2.y],
        }
        # Keep attached label state minimal in the exported canvas state.
        # Only include label metadata when there is non-empty label text.
        try:
            label_text = str(getattr(self.label, "text", "") or "")
        except Exception:
            label_text = ""
        if label_text:
            try:
                label_visible = bool(getattr(self.label, "visible", False))
            except Exception:
                label_visible = False
            state["args"]["label"] = {
                "text": label_text,
                "visible": label_visible,
            }
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        # Check if the segment has already been deep copied
        if id(self) in memo:
            from typing import cast
            return cast(Segment, memo[id(self)])

        # Deepcopy points that define the segment
        new_p1: Point = deepcopy(self.point1, memo)
        new_p2: Point = deepcopy(self.point2, memo)
        # Create a new Segment instance with the copied points
        new_segment: Segment = Segment(
            new_p1,
            new_p2,
            color=self.color,
            label_text=self.label.text,
            label_visible=bool(getattr(self.label, "visible", True)),
        )
        memo[id(self)] = new_segment
        # Preserve label styling details
        try:
            new_segment.label.update_font_size(getattr(self.label, "font_size", point_label_font_size))
            new_segment.label.update_rotation(getattr(self.label, "rotation_degrees", 0.0))
            new_segment.label.update_color(getattr(self.label, "color", self.color))
            new_segment.label.render_mode = getattr(self.label, "render_mode", new_segment.label.render_mode)
        except Exception:
            pass
        new_segment.name = self.name
        new_segment._sync_label_position()

        return new_segment

    def translate(self, x_offset: float, y_offset: float) -> None:
        self.point1.x += x_offset
        self.point1.y += y_offset
        self.point2.x += x_offset
        self.point2.y += y_offset
        # Keep analytic state in sync after translation.
        self.line_formula = self._calculate_line_algebraic_formula()
        self._sync_label_position()

    def update_color(self, color: str) -> None:
        """Update the segment color metadata."""
        self.color = str(color)
        try:
            self.label.update_color(self.color)
        except Exception:
            pass

    def _get_midpoint(self) -> Tuple[float, float]:
        """Calculate the midpoint of the segment"""
        x: float = (self.point1.x + self.point2.x) / 2
        y: float = (self.point1.y + self.point2.y) / 2
        return (x, y)

    def _sync_label_position(self) -> None:
        """Keep the embedded label anchored at the segment midpoint."""
        if not hasattr(self, "label"):
            return
        try:
            mid_x, mid_y = self._get_midpoint()
            self.label.update_position(mid_x, mid_y)
        except Exception:
            pass

    def update_label_text(self, text: str) -> None:
        """Update the embedded label text (validated by Label)."""
        self.label.update_text(text)

    def set_label_visibility(self, visible: bool) -> None:
        """Toggle embedded label visibility."""
        self.label.visible = bool(visible)
        self._sync_label_position()

    def _rotate_point_around_center(self, point: Point, center_x: float, center_y: float, angle_rad: float) -> None:
        """Rotate a single point around a center by given angle in radians"""
        dx: float = point.x - center_x
        dy: float = point.y - center_y

        point.x = center_x + (dx * math.cos(angle_rad) - dy * math.sin(angle_rad))
        point.y = center_y + (dx * math.sin(angle_rad) + dy * math.cos(angle_rad))

    def reflect(self, axis: str, a: float = 0, b: float = 0, c: float = 0) -> None:
        """Reflect the segment across the specified axis."""
        self.point1.reflect(axis, a, b, c)
        self.point2.reflect(axis, a, b, c)
        self.line_formula = self._calculate_line_algebraic_formula()
        self._sync_label_position()

    def scale(self, sx: float, sy: float, cx: float, cy: float) -> None:
        """Scale the segment relative to center (cx, cy)."""
        self.point1.scale(sx, sy, cx, cy)
        self.point2.scale(sx, sy, cx, cy)
        self.line_formula = self._calculate_line_algebraic_formula()
        self._sync_label_position()

    def shear(self, axis: str, factor: float, cx: float, cy: float) -> None:
        """Shear the segment relative to center (cx, cy)."""
        self.point1.shear(axis, factor, cx, cy)
        self.point2.shear(axis, factor, cx, cy)
        self.line_formula = self._calculate_line_algebraic_formula()
        self._sync_label_position()

    def rotate_around(self, angle_deg: float, cx: float, cy: float) -> None:
        """Rotate the segment around an arbitrary center (cx, cy)."""
        self.point1.rotate_around(angle_deg, cx, cy)
        self.point2.rotate_around(angle_deg, cx, cy)
        self.line_formula = self._calculate_line_algebraic_formula()
        self._sync_label_position()

    def rotate(self, angle: float) -> Tuple[bool, Optional[str]]:
        """Rotate the segment around its midpoint by the given angle in degrees"""
        # Get midpoint
        center_x: float
        center_y: float
        center_x, center_y = self._get_midpoint()

        # Convert angle to radians
        angle_rad: float = math.radians(angle)

        # Rotate both endpoints
        self._rotate_point_around_center(self.point1, center_x, center_y, angle_rad)
        self._rotate_point_around_center(self.point2, center_x, center_y, angle_rad)

        # Update line formula
        self.line_formula = self._calculate_line_algebraic_formula()
        self._sync_label_position()

        # Return tuple (should_proceed, message) to match interface
        return True, None

    def __eq__(self, other: object) -> Any:
        """Checks if two segments are equal based on their endpoints."""
        if not isinstance(other, Segment):
            return NotImplemented
        # Use sets to ignore endpoint order
        # Assumes self.point1 and self.point2 are not None
        # Requires Point class to have proper __eq__ and __hash__
        if self.point1 is None or self.point2 is None or other.point1 is None or other.point2 is None:
            return False # Or handle appropriately if None points are possible during comparison
        points_self: set[Point] = {self.point1, self.point2}
        points_other: set[Point] = {other.point1, other.point2}
        return points_self == points_other

    def __hash__(self) -> int:
        """Computes hash based on a frozenset of the hashes of its endpoint Points."""
        # Hash is based on the IDs of the point objects, order-independent
        if self.point1 is None or self.point2 is None:
             # Consistent with __eq__ if points can be None
             return hash((None, None))
        # Use frozenset of point hashes to ensure hash is consistent regardless of point1/point2 order
        # and relies on Point.__hash__ which is value-based.
        return hash(frozenset([hash(self.point1), hash(self.point2)]))
