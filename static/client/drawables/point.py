"""
MatHud Point Geometric Object

Fundamental geometric building block representing a point in 2D mathematical space.
Provides coordinate tracking, labeling, and serves as endpoints for other geometric objects.

Key Features:
    - Math coordinate tracking
    - Automatic label display with coordinates
    - Translation operations for object manipulation

Coordinate Systems:
    - x, y: Mathematical coordinates

Dependencies:
    - constants: Point sizing and labeling configuration
    - drawables.drawable: Base class interface
    - drawables.position: Coordinate container
    - utils.math_utils: Mathematical operations
    - CoordinateMapper: View-layer service (used by renderer)
"""

from __future__ import annotations

import math
from typing import Any, Dict, cast

from constants import default_color
from drawables.attached_label import AttachedLabel
from drawables.drawable import Drawable
from drawables.position import Position
from utils.math_utils import MathUtils


class Point(Drawable):
    """Represents a point in 2D mathematical space with coordinate tracking and labeling.

    Fundamental building block for all geometric constructions, maintaining mathematical
    coordinates. Rendering and coordinate transformations are handled by the renderer
    via a CoordinateMapper.

    Attributes:
        x, y (float): Mathematical coordinates (unaffected by zoom/pan)
    """

    def __init__(self, x: float, y: float, name: str = "", color: str = default_color) -> None:
        """Initialize a point with mathematical coordinates.

        Args:
            x (float): Mathematical x-coordinate in the coordinate system
            y (float): Mathematical y-coordinate in the coordinate system
            name (str): Unique identifier for the point
            color (str): CSS color value for point visualization
        """
        self._x: float = float(x)
        self._y: float = float(y)
        super().__init__(name=name, color=color)
        self.label = AttachedLabel(
            self._x,
            self._y,
            str(self.name or ""),
            color=self.color,
            text_format="text_with_anchor_coords",
            coord_precision=3,
            offset_from_point_radius=True,
            non_selectable=True,
            font_size_source="style",
            font_size_key="point_label_font_size",
            font_family_key="point_label_font_family",
        )

    def get_class_name(self) -> str:
        return "Point"

    def __str__(self) -> str:
        def fmt(v: float) -> str:
            return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)

        return f"{fmt(self.x)},{fmt(self.y)}"

    def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"name": self.name, "args": {"position": {"x": self.x, "y": self.y}}}
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> Point:
        if id(self) in memo:
            return cast(Point, memo[id(self)])
        # For undo / redo / archive functionality
        # Create a new Point instance with the same coordinates and properties
        new_point: Point = Point(self.x, self.y, name=self.name, color=self.color)
        memo[id(self)] = new_point
        return new_point

    def translate(self, x_offset: float, y_offset: float) -> None:
        self.x += x_offset
        self.y += y_offset
        self._sync_label_position()

    def update_position(self, x: float, y: float) -> None:
        """Update both coordinates at once to keep math-space state consistent."""
        self.x = float(x)
        self.y = float(y)
        self._sync_label_position()

    def update_color(self, color: str) -> None:
        """Update the visual color metadata for the point."""
        self.color = color
        self._sync_label_color()

    def update_name(self, name: str) -> None:
        """Rename the point using the Drawable base property."""
        self.name = name
        self._sync_label_text()

    def reflect(self, axis: str, a: float = 0, b: float = 0, c: float = 0) -> None:
        """Reflect the point across the specified axis.

        Args:
            axis: 'x_axis', 'y_axis', or 'line' (ax + by + c = 0)
            a, b, c: Coefficients for line reflection
        """
        if axis == "x_axis":
            self._y = -self._y
        elif axis == "y_axis":
            self._x = -self._x
        elif axis == "line":
            denom = a * a + b * b
            if denom < 1e-18:
                return
            dot = a * self._x + b * self._y + c
            self._x = self._x - 2 * a * dot / denom
            self._y = self._y - 2 * b * dot / denom
        self._sync_label_position()

    def scale(self, sx: float, sy: float, cx: float, cy: float) -> None:
        """Scale the point relative to center (cx, cy)."""
        self._x = cx + sx * (self._x - cx)
        self._y = cy + sy * (self._y - cy)
        self._sync_label_position()

    def shear(self, axis: str, factor: float, cx: float, cy: float) -> None:
        """Shear the point relative to center (cx, cy).

        Args:
            axis: 'horizontal' or 'vertical'
            factor: Shear factor
            cx, cy: Center of shear
        """
        dx = self._x - cx
        dy = self._y - cy
        if axis == "horizontal":
            self._x = cx + dx + factor * dy
            self._y = cy + dy
        elif axis == "vertical":
            self._x = cx + dx
            self._y = cy + dy + factor * dx
        self._sync_label_position()

    def rotate_around(self, angle_deg: float, cx: float, cy: float) -> None:
        """Rotate the point around an arbitrary center (cx, cy).

        Args:
            angle_deg: Rotation angle in degrees (positive = counterclockwise)
            cx, cy: Center of rotation
        """
        angle_rad = math.radians(angle_deg)
        dx = self._x - cx
        dy = self._y - cy
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        self._x = cx + dx * cos_a - dy * sin_a
        self._y = cy + dx * sin_a + dy * cos_a
        self._sync_label_position()

    def rotate(self, angle: float) -> None:
        pass

    def __eq__(self, other: object) -> Any:
        """Checks if two points are equal based on coordinates within tolerance."""
        if not isinstance(other, Point):
            return NotImplemented
        # Use MathUtils for tolerance comparison
        x_match = abs(self.x - other.x) < MathUtils.EPSILON
        y_match = abs(self.y - other.y) < MathUtils.EPSILON
        return x_match and y_match

    def __hash__(self) -> int:
        """Computes hash based on rounded coordinates."""
        # Hash based on coordinates rounded to a few decimal places
        # Adjust precision as needed, should be coarser than EPSILON allows differences
        precision: int = 6  # e.g., 6 decimal places
        rounded_x: float = round(self.x, precision)
        rounded_y: float = round(self.y, precision)
        return hash((rounded_x, rounded_y))

    @property
    def x(self) -> float:
        """Math x-coordinate (will be flipped to math in this migration step)."""
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        self._x = float(value)
        self._sync_label_position()

    @property
    def y(self) -> float:
        """Math y-coordinate (will be flipped to math in this migration step)."""
        return self._y

    @y.setter
    def y(self, value: float) -> None:
        self._y = float(value)
        self._sync_label_position()

    def _sync_label_position(self) -> None:
        try:
            if hasattr(self, "label"):
                self.label.update_position(self._x, self._y)
        except Exception:
            return

    def _sync_label_text(self) -> None:
        try:
            if hasattr(self, "label"):
                self.label.update_text(str(self.name or ""))
        except Exception:
            return

    def _sync_label_color(self) -> None:
        try:
            if hasattr(self, "label"):
                self.label.update_color(str(self.color))
        except Exception:
            return

    def zoom(self) -> None:
        """Empty zoom method for backward compatibility.

        Zoom transformations are now handled centrally by CoordinateMapper
        when drawing, so individual drawable zoom() methods do nothing.
        """
        pass

    def pan(self) -> None:
        """Empty pan method for backward compatibility.

        Pan transformations are now handled centrally by CoordinateMapper
        when drawing, so individual drawable pan() methods do nothing.
        """
        pass
