"""
Tangent and Normal Line Manager for MatHud

Manages the creation of tangent and normal lines to curves (functions, parametric
functions, circles, ellipses). Lines are created as standard Segment drawables.

The manager handles different curve types:
- Function y=f(x): parameter is the x-coordinate
- ParametricFunction x(t), y(t): parameter is the t value
- Circle: parameter is the angle in radians from positive x-axis
- Ellipse: parameter is the angle in radians (parameter angle, not geometric)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from constants import default_color
from utils.math_utils import MathUtils

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.segment_manager import SegmentManager
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from name_generator.drawable import DrawableNameGenerator
    from drawables.segment import Segment
    from drawables.function import Function
    from drawables.parametric_function import ParametricFunction
    from drawables.circle import Circle
    from drawables.ellipse import Ellipse

# Type alias for supported curve types
CurveType = Union["Function", "ParametricFunction", "Circle", "Ellipse"]

# Default line segment length in math units
DEFAULT_TANGENT_LENGTH = 4.0


class TangentManager:
    """
    Manages tangent and normal line creation for various curve types.

    Creates segment drawables representing tangent or normal lines at specified
    points on functions, parametric curves, circles, and ellipses.

    Attributes:
        canvas: Reference to the parent Canvas instance
        drawables: Container for all drawable objects
        segment_manager: Manager for creating segment drawables
        name_generator: Generates unique names for segments
        dependency_manager: Tracks object dependencies
        proxy: Manager proxy for inter-manager communication
    """

    def __init__(
        self,
        canvas: "Canvas",
        drawables: "DrawablesContainer",
        segment_manager: "SegmentManager",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        proxy: "DrawableManagerProxy",
    ) -> None:
        """
        Initialize the TangentManager.

        Args:
            canvas: Parent Canvas instance
            drawables: Container for storing drawables
            segment_manager: Manager for creating segments
            name_generator: Generator for unique drawable names
            dependency_manager: Manager for tracking dependencies
            proxy: Proxy for inter-manager communication
        """
        self.canvas: "Canvas" = canvas
        self.drawables: "DrawablesContainer" = drawables
        self.segment_manager: "SegmentManager" = segment_manager
        self.name_generator: "DrawableNameGenerator" = name_generator
        self.dependency_manager: "DrawableDependencyManager" = dependency_manager
        self.proxy: "DrawableManagerProxy" = proxy

    def _archive_for_undo(self) -> None:
        """Archive current state before making changes for undo support."""
        undo_redo = getattr(self.canvas, "undo_redo_manager", None)
        if undo_redo:
            undo_redo.archive()

    def _get_curve(self, curve_name: str) -> Optional[CurveType]:
        """
        Retrieve a curve by name from the drawables container.

        Searches for functions, parametric functions, circles, and ellipses.

        Args:
            curve_name: Name of the curve to find

        Returns:
            The curve drawable if found, None otherwise
        """
        if not curve_name:
            return None

        # Check each drawable type
        for func in self.drawables.Functions:
            if getattr(func, "name", None) == curve_name:
                return func  # type: ignore[return-value]

        for pfunc in self.drawables.ParametricFunctions:
            if getattr(pfunc, "name", None) == curve_name:
                return pfunc  # type: ignore[return-value]

        for circle in self.drawables.Circles:
            if getattr(circle, "name", None) == curve_name:
                return circle  # type: ignore[return-value]

        for ellipse in self.drawables.Ellipses:
            if getattr(ellipse, "name", None) == curve_name:
                return ellipse  # type: ignore[return-value]

        return None

    def _compute_tangent_data(
        self,
        curve: CurveType,
        parameter: float,
        length: float,
    ) -> Dict[str, Any]:
        """
        Compute tangent line data for a curve at a given parameter.

        Args:
            curve: The curve drawable
            parameter: x-coord (Function), t-value (Parametric), or angle (Circle/Ellipse)
            length: Desired length of the tangent line segment

        Returns:
            Dict with keys: 'point' (x, y), 'slope', 'endpoints' ((x1, y1), (x2, y2))

        Raises:
            ValueError: If tangent cannot be computed
        """
        class_name = curve.get_class_name()

        if class_name == "Function":
            return self._compute_function_tangent(curve, parameter, length)  # type: ignore[arg-type]
        elif class_name == "ParametricFunction":
            return self._compute_parametric_tangent(curve, parameter, length)  # type: ignore[arg-type]
        elif class_name == "Circle":
            return self._compute_circle_tangent(curve, parameter, length)  # type: ignore[arg-type]
        elif class_name == "Ellipse":
            return self._compute_ellipse_tangent(curve, parameter, length)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unsupported curve type: {class_name}")

    def _compute_function_tangent(
        self,
        func: "Function",
        x: float,
        length: float,
    ) -> Dict[str, Any]:
        """Compute tangent data for y=f(x) at x-coordinate."""
        # Evaluate function at x
        try:
            y = func.function(x)
        except Exception as e:
            raise ValueError(f"Cannot evaluate function at x={x}: {e}")

        if not isinstance(y, (int, float)):
            raise ValueError(f"Function returns non-numeric value at x={x}")
        y = float(y)
        if math.isnan(y):
            raise ValueError(f"Function is undefined at x={x}")

        # Compute derivative numerically
        slope = MathUtils.numerical_derivative_at(func.function, x)
        if slope is None:
            raise ValueError(f"Cannot compute derivative at x={x}")

        point = (x, y)
        endpoints = MathUtils.tangent_line_endpoints(slope, point, length)

        return {
            "point": point,
            "slope": slope,
            "endpoints": endpoints,
        }

    def _compute_parametric_tangent(
        self,
        pfunc: "ParametricFunction",
        t: float,
        length: float,
    ) -> Dict[str, Any]:
        """Compute tangent data for parametric curve at parameter t."""
        # Evaluate parametric function at t
        try:
            point = pfunc.evaluate(t)
        except Exception as e:
            raise ValueError(f"Cannot evaluate parametric function at t={t}: {e}")

        x, y = point

        # Compute derivatives dx/dt and dy/dt numerically
        dx_dt = MathUtils.numerical_derivative_at(pfunc.evaluate_x, t)
        dy_dt = MathUtils.numerical_derivative_at(pfunc.evaluate_y, t)

        if dx_dt is None or dy_dt is None:
            raise ValueError(f"Cannot compute derivatives at t={t}")

        # Calculate slope dy/dx = (dy/dt) / (dx/dt)
        if abs(dx_dt) < MathUtils.EPSILON:
            # Vertical tangent
            slope: Optional[float] = None
        else:
            slope = dy_dt / dx_dt

        endpoints = MathUtils.tangent_line_endpoints(slope, point, length)

        return {
            "point": point,
            "slope": slope,
            "endpoints": endpoints,
        }

    def _compute_circle_tangent(
        self,
        circle: "Circle",
        angle: float,
        length: float,
    ) -> Dict[str, Any]:
        """Compute tangent data for circle at given angle."""
        point, slope = MathUtils.circle_tangent_slope_at_angle(
            circle.center.x,
            circle.center.y,
            circle.radius,
            angle,
        )

        endpoints = MathUtils.tangent_line_endpoints(slope, point, length)

        return {
            "point": point,
            "slope": slope,
            "endpoints": endpoints,
        }

    def _compute_ellipse_tangent(
        self,
        ellipse: "Ellipse",
        angle: float,
        length: float,
    ) -> Dict[str, Any]:
        """Compute tangent data for ellipse at given parameter angle."""
        point, slope = MathUtils.ellipse_tangent_slope_at_angle(
            ellipse.center.x,
            ellipse.center.y,
            ellipse.radius_x,
            ellipse.radius_y,
            angle,
            ellipse.rotation_angle,
        )

        endpoints = MathUtils.tangent_line_endpoints(slope, point, length)

        return {
            "point": point,
            "slope": slope,
            "endpoints": endpoints,
        }

    def _compute_normal_data(
        self,
        curve: CurveType,
        parameter: float,
        length: float,
    ) -> Dict[str, Any]:
        """
        Compute normal line data for a curve at a given parameter.

        The normal line is perpendicular to the tangent line.

        Args:
            curve: The curve drawable
            parameter: x-coord (Function), t-value (Parametric), or angle (Circle/Ellipse)
            length: Desired length of the normal line segment

        Returns:
            Dict with keys: 'point' (x, y), 'slope', 'endpoints' ((x1, y1), (x2, y2))
        """
        # First compute tangent data
        tangent_data = self._compute_tangent_data(curve, parameter, length)

        # Calculate normal slope
        tangent_slope = tangent_data["slope"]
        normal_slope = MathUtils.normal_slope(tangent_slope)

        # Calculate normal endpoints
        endpoints = MathUtils.tangent_line_endpoints(
            normal_slope, tangent_data["point"], length
        )

        return {
            "point": tangent_data["point"],
            "slope": normal_slope,
            "endpoints": endpoints,
        }

    def create_tangent_line(
        self,
        curve_name: str,
        parameter: float,
        name: Optional[str] = None,
        length: Optional[float] = None,
        color: Optional[str] = None,
    ) -> "Segment":
        """
        Create a tangent line segment to a curve at a specified point.

        Args:
            curve_name: Name of the target curve (function, parametric, circle, or ellipse)
            parameter: Location on curve (x for functions, t for parametric, angle for circle/ellipse)
            name: Optional name for the created segment
            length: Total length of tangent segment (default: 4.0 math units)
            color: Display color (default: same as curve or default_color)

        Returns:
            The created Segment drawable

        Raises:
            ValueError: If curve not found or tangent cannot be computed
        """
        # Get the curve
        curve = self._get_curve(curve_name)
        if curve is None:
            raise ValueError(f"Curve '{curve_name}' not found")

        # Set defaults
        if length is None:
            length = DEFAULT_TANGENT_LENGTH
        if color is None:
            color = getattr(curve, "color", default_color)

        # Compute tangent data
        tangent_data = self._compute_tangent_data(curve, parameter, length)

        # Extract endpoints
        (x1, y1), (x2, y2) = tangent_data["endpoints"]

        # Archive for undo before creating segment
        self._archive_for_undo()

        # Create the segment using segment manager
        segment = self.segment_manager.create_segment(
            x1, y1, x2, y2,
            name=name or "",
            color=color,
            extra_graphics=True,
        )

        return segment

    def create_normal_line(
        self,
        curve_name: str,
        parameter: float,
        name: Optional[str] = None,
        length: Optional[float] = None,
        color: Optional[str] = None,
    ) -> "Segment":
        """
        Create a normal line segment to a curve at a specified point.

        The normal line is perpendicular to the tangent line at the same point.

        Args:
            curve_name: Name of the target curve (function, parametric, circle, or ellipse)
            parameter: Location on curve (x for functions, t for parametric, angle for circle/ellipse)
            name: Optional name for the created segment
            length: Total length of normal segment (default: 4.0 math units)
            color: Display color (default: same as curve or default_color)

        Returns:
            The created Segment drawable

        Raises:
            ValueError: If curve not found or normal cannot be computed
        """
        # Get the curve
        curve = self._get_curve(curve_name)
        if curve is None:
            raise ValueError(f"Curve '{curve_name}' not found")

        # Set defaults
        if length is None:
            length = DEFAULT_TANGENT_LENGTH
        if color is None:
            color = getattr(curve, "color", default_color)

        # Compute normal data
        normal_data = self._compute_normal_data(curve, parameter, length)

        # Extract endpoints
        (x1, y1), (x2, y2) = normal_data["endpoints"]

        # Archive for undo before creating segment
        self._archive_for_undo()

        # Create the segment using segment manager
        segment = self.segment_manager.create_segment(
            x1, y1, x2, y2,
            name=name or "",
            color=color,
            extra_graphics=True,
        )

        return segment
