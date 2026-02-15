"""
MatHud Ellipse Geometric Object

Represents an ellipse defined by center point, radii, and rotation angle in 2D mathematical space.
Provides algebraic equation calculation and rotation transformation capabilities.

Key Features:
    - Center point and dual radius definition (rx, ry)
    - Rotation angle support for arbitrary ellipse orientation
    - Automatic ellipse equation calculation
    - Pure math model; renderer applies screen scaling and rotation transforms

Mathematical Properties:
    - ellipse_formula: Algebraic equation coefficients
    - Center point tracking through Point object
    - Renderer-agnostic; no viewport scaling stored in model
    - Rotation angle preservation and application

Dependencies:
    - constants: Default styling values
    - drawables.drawable: Base class interface
    - utils.math_utils: Ellipse equation calculations
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, cast

from constants import default_color
from drawables.drawable import Drawable
from drawables.point import Point
from utils.math_utils import MathUtils

class Ellipse(Drawable):
    """Represents an ellipse with center point, dual radii, and rotation angle.

    Maintains center Point object, x/y radius values, and rotation angle,
    calculating ellipse equation properties for mathematical operations.
    Screen radii are calculated dynamically using CoordinateMapper.

    Attributes:
        center (Point): Center point of the ellipse
        radius_x (float): Horizontal radius in mathematical coordinate units
        radius_y (float): Vertical radius in mathematical coordinate units
        rotation_angle (float): Rotation angle in degrees for ellipse orientation
        ellipse_formula (dict): Algebraic ellipse equation coefficients
    """
    def __init__(self, center_point: Point, radius_x: float, radius_y: float, rotation_angle: float = 0, color: str = default_color) -> None:
        """Initialize an ellipse with center point, radii, and rotation.

        Args:
            center_point (Point): Center point of the ellipse
            radius_x (float): Horizontal radius in mathematical coordinate units
            radius_y (float): Vertical radius in mathematical coordinate units
            rotation_angle (float): Rotation angle in degrees (default: 0)
            color (str): CSS color value for ellipse visualization
        """
        self.center: Point = center_point
        self.radius_x: float = radius_x
        self.radius_y: float = radius_y
        self.rotation_angle: float = rotation_angle  # Initialize with provided angle
        self.ellipse_formula: Dict[str, float] = self._calculate_ellipse_algebraic_formula()
        super().__init__(name=self._generate_default_name(), color=color)

    def get_class_name(self) -> str:
        return 'Ellipse'

    def _calculate_ellipse_algebraic_formula(self) -> Dict[str, float]:
        x: float = self.center.x
        y: float = self.center.y
        result: Any = MathUtils.get_ellipse_formula(x, y, self.radius_x, self.radius_y, self.rotation_angle)
        return cast(Dict[str, float], result)

    def get_state(self) -> Dict[str, Any]:
        """Return the state of the ellipse including rotation"""
        state: Dict[str, Any] = {
            "name": self.name,
            "args": {
                "center": self.center.name,
                "radius_x": self.radius_x,
                "radius_y": self.radius_y,
                "rotation_angle": self.rotation_angle,
                "ellipse_formula": self.ellipse_formula
            }
        }
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        if id(self) in memo:
            return cast(Ellipse, memo[id(self)])
        # Deep copy the center point
        new_center: Point = deepcopy(self.center, memo)
        # Create a new Ellipse instance with the copied center point and other properties
        new_ellipse: Ellipse = Ellipse(new_center, self.radius_x, self.radius_y,
                             color=self.color,
                             rotation_angle=self.rotation_angle)
        memo[id(self)] = new_ellipse
        return new_ellipse

    def translate(self, x_offset: float, y_offset: float) -> None:
        self.center.x += x_offset
        self.center.y += y_offset
        self.ellipse_formula = self._calculate_ellipse_algebraic_formula()
        self.regenerate_name()

    def reflect(self, axis: str, a: float = 0, b: float = 0, c: float = 0) -> None:
        """Reflect the ellipse across the specified axis.

        Center is reflected; rotation_angle is adjusted to preserve shape orientation.
        """
        self.center.reflect(axis, a, b, c)
        if axis == "x_axis":
            self.rotation_angle = (-self.rotation_angle) % 360
        elif axis == "y_axis":
            self.rotation_angle = (180 - self.rotation_angle) % 360
        elif axis == "line":
            denom = a * a + b * b
            if denom >= 1e-18:
                line_angle_deg = math.degrees(math.atan2(-a, b))
                self.rotation_angle = (2 * line_angle_deg - self.rotation_angle) % 360
        self.ellipse_formula = self._calculate_ellipse_algebraic_formula()
        self.regenerate_name()

    def scale(self, sx: float, sy: float, cx: float, cy: float) -> None:
        """Scale the ellipse from center (cx, cy).

        Supports uniform scaling and axis-aligned non-uniform scaling.

        Raises:
            ValueError: If non-uniform scaling on a rotated ellipse, or zero factor.
        """
        if abs(sx) < 1e-18 or abs(sy) < 1e-18:
            raise ValueError("Scale factor must not be zero")
        uniform = abs(sx - sy) < 1e-9
        rotated = (self.rotation_angle % 180) > 1e-9
        if not uniform and rotated:
            raise ValueError(
                "Non-uniform scaling of a rotated ellipse is not supported"
            )
        self.center.scale(sx, sy, cx, cy)
        if uniform:
            self.radius_x = abs(self.radius_x * sx)
            self.radius_y = abs(self.radius_y * sx)
        else:
            self.radius_x = abs(self.radius_x * sx)
            self.radius_y = abs(self.radius_y * sy)
        self.ellipse_formula = self._calculate_ellipse_algebraic_formula()
        self.regenerate_name()

    def shear(self, axis: str, factor: float, cx: float, cy: float) -> None:
        """Shearing an ellipse is not supported.

        Raises:
            ValueError: Always raised.
        """
        raise ValueError("Shearing an ellipse is not supported")

    def rotate_around(self, angle_deg: float, cx: float, cy: float) -> None:
        """Rotate the ellipse around an arbitrary point (cx, cy)."""
        self.center.rotate_around(angle_deg, cx, cy)
        self.rotation_angle = (self.rotation_angle + angle_deg) % 360
        self.ellipse_formula = self._calculate_ellipse_algebraic_formula()
        self.regenerate_name()

    def rotate(self, angle: float) -> Tuple[bool, Optional[str]]:
        """Rotate the ellipse around its center by the given angle in degrees"""
        # Update rotation angle (keep it between 0 and 360 degrees)
        self.rotation_angle = (self.rotation_angle + angle) % 360

        # Update ellipse formula if needed
        self.ellipse_formula = self._calculate_ellipse_algebraic_formula()

        # Return tuple (should_proceed, message) to match interface
        return True, None

    def update_color(self, color: str) -> None:
        """Update the ellipse color metadata."""
        self.color = str(color)

    def update_center_position(self, x: float, y: float) -> None:
        """Move the ellipse center and refresh cached state."""
        self.center.update_position(x, y)
        self.ellipse_formula = self._calculate_ellipse_algebraic_formula()
        self.regenerate_name()

    def update_radius_x(self, radius_x: float) -> None:
        """Update the horizontal radius."""
        self.radius_x = float(radius_x)
        self.ellipse_formula = self._calculate_ellipse_algebraic_formula()
        self.regenerate_name()

    def update_radius_y(self, radius_y: float) -> None:
        """Update the vertical radius."""
        self.radius_y = float(radius_y)
        self.ellipse_formula = self._calculate_ellipse_algebraic_formula()
        self.regenerate_name()

    def update_rotation_angle(self, rotation_angle: float) -> None:
        """Set the rotation angle directly."""
        self.rotation_angle = float(rotation_angle) % 360
        self.ellipse_formula = self._calculate_ellipse_algebraic_formula()

    def _generate_default_name(self) -> str:
        return f"{self.center.name}({self._format_radius(self.radius_x)}, {self._format_radius(self.radius_y)})"

    def _format_radius(self, value: float) -> str:
        value = float(value)
        if value.is_integer():
            return str(int(value))
        return str(value)

    def regenerate_name(self) -> None:
        """Refresh the ellipse name from its current center/radii."""
        self.name = self._generate_default_name()
