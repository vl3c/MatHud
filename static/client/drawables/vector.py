"""
MatHud Vector Geometric Object

Represents a directed line segment (vector) with origin and tip points, displayed with an arrow tip.
Built on top of the Segment class with additional directional visualization.

Key Features:
    - Directed line segment with origin and tip designation
    - Automatic arrow head calculation and rendering
    - Translation and rotation operations maintaining direction
    - Integration with segment properties for mathematical operations

Visual Elements:
    - Line segment: Rendered using underlying Segment object
    - Arrow tip: Triangular polygon calculated from direction and size
    - Directional properties: Origin and tip point distinction

Dependencies:
    - constants: Default styling and arrow sizing
    - drawables.drawable: Base class interface
    - drawables.point: Endpoint objects
    - drawables.segment: Underlying line representation
    - utils.math_utils: Angle and geometric calculations
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, cast

from constants import default_color
from drawables.drawable import Drawable
from drawables.point import Point
from drawables.segment import Segment


class Vector(Drawable):
    """Represents a directed line segment (vector) with origin, tip, and arrow head visualization.

    Extends the concept of a line segment to include directionality, displayed with
    an arrow head at the tip to indicate vector direction and magnitude.

    Attributes:
        segment (Segment): Underlying line segment providing mathematical properties
        origin (Point): Starting point of the vector (property access to segment.point1)
        tip (Point): Ending point of the vector (property access to segment.point2)
    """

    def __init__(self, origin: Point, tip: Point, color: str = default_color) -> None:
        """Initialize a vector with origin and tip points.

        Args:
            origin (Point): Starting point of the vector
            tip (Point): Ending point of the vector (where arrow head is drawn)
            color (str): CSS color value for vector visualization
        """
        self.segment: Segment = Segment(origin, tip, color=color)
        name: str = self.segment.name
        super().__init__(name=name, color=color)

    @property
    def origin(self) -> Point:
        """Get the origin point of the vector."""
        return self.segment.point1

    @property
    def tip(self) -> Point:
        """Get the tip point of the vector."""
        return self.segment.point2

    def get_class_name(self) -> str:
        return "Vector"

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "args": {
                "origin": self.segment.point1.name,
                "tip": self.segment.point2.name,
            },
            # Include coordinates for render cache invalidation
            "_origin_coords": [self.segment.point1.x, self.segment.point1.y],
            "_tip_coords": [self.segment.point2.x, self.segment.point2.y],
        }

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        # Check if the vector has already been deep copied
        if id(self) in memo:
            return cast(Vector, memo[id(self)])
        # Deepcopy the origin and tip points that define the vector
        new_origin: Point = deepcopy(self.segment.point1, memo)
        new_tip: Point = deepcopy(self.segment.point2, memo)
        # Create a new Vector instance with the deep-copied points
        new_vector: Vector = Vector(new_origin, new_tip, color=self.color)
        # Store the newly created vector in the memo dictionary
        memo[id(self)] = new_vector
        return new_vector

    def translate(self, x_offset: float, y_offset: float) -> None:
        self.segment.translate(x_offset, y_offset)

    def reflect(self, axis: str, a: float = 0, b: float = 0, c: float = 0) -> None:
        """Reflect the vector across the specified axis."""
        self.segment.reflect(axis, a, b, c)

    def scale(self, sx: float, sy: float, cx: float, cy: float) -> None:
        """Scale the vector relative to center (cx, cy)."""
        self.segment.scale(sx, sy, cx, cy)

    def shear(self, axis: str, factor: float, cx: float, cy: float) -> None:
        """Shear the vector relative to center (cx, cy)."""
        self.segment.shear(axis, factor, cx, cy)

    def rotate_around(self, angle_deg: float, cx: float, cy: float) -> None:
        """Rotate the vector around an arbitrary center (cx, cy)."""
        self.segment.rotate_around(angle_deg, cx, cy)

    def rotate(self, angle: float) -> Tuple[bool, Optional[str]]:
        """Rotate the vector around its origin by the given angle in degrees"""
        # Use segment's rotation method to rotate the line portion
        should_proceed: bool
        message: Optional[str]
        should_proceed, message = self.segment.rotate(angle)
        if not should_proceed:
            return False, message
        return True, None

    def update_color(self, color: str) -> None:
        """Update the vector color and underlying segment color."""
        sanitized = str(color)
        self.color = sanitized
        if hasattr(self.segment, "update_color"):
            self.segment.update_color(sanitized)
        else:
            self.segment.color = sanitized
