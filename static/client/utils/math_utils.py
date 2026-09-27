"""
MatHud Mathematical Utilities Module

Comprehensive mathematical computation library for geometric analysis, symbolic algebra, and numerical calculations.
Provides the mathematical foundation for all geometric objects and canvas operations.

Key Features:
    - Geometric analysis: point matching, distance, area, angle calculations
    - Coordinate validation and tolerance-based comparisons
    - Line and curve equation generation (lines, circles, ellipses)
    - Symbolic mathematics: derivatives, integrals, limits, simplification
    - System of equations solving (linear, quadratic, mixed systems)
    - Statistical functions: mean, median, mode, variance
    - Asymptote and discontinuity analysis for function plotting
    - Rectangle and triangle validation algorithms

Mathematical Categories:
    - Point/Segment Operations: coordinate matching, distance, collinearity
    - Shape Analysis: area calculations, centroid finding, geometric validation
    - Equation Generation: algebraic formulas for geometric objects
    - Symbolic Computation: calculus operations via MathJS integration
    - Numerical Methods: equation solving, statistical analysis
    - Function Analysis: asymptotes, discontinuities, behavior analysis

Tolerance System:
    - EPSILON = 1e-9: Global tolerance for floating-point comparisons
    - Adaptive thresholds for segment-based calculations
    - Coordinate-aware precision handling

Dependencies:
    - browser.window: MathJS library integration for symbolic math
    - math: Standard mathematical functions and constants
    - statistics: Statistical computation functions
    - drawables.position: Coordinate container for geometric calculations
"""

import json
import math
import random
import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union, cast

from browser import window

Number = Union[int, float]
PointLike = Any
SegmentLike = Any


class MathUtils:
    """Comprehensive mathematical utilities class for geometric analysis and symbolic computation.

    Provides static methods for all mathematical operations required by the MatHud canvas system,
    including coordinate validation, geometric calculations, equation generation, and symbolic mathematics.

    Class Attributes:
        EPSILON (float): Global tolerance constant (1e-9) for floating-point comparisons
    """

    # Epsilon (tolerance)
    EPSILON = 1e-9
    MAX_SERIES_TERMS = 1000
    MAX_NUMERIC_INTEGRATION_STEPS = 10000

    @staticmethod
    def _ensure_non_negative_integer(value: Number, name: str) -> int:
        """Validate that a value is a non-negative integer and return it as int."""
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a non-negative integer")
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            raise TypeError(f"{name} must be a non-negative integer")
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    @staticmethod
    def format_number_for_cartesian(n: Number, max_digits: int = 6) -> str:
        """
        Formats the number to a string with a maximum number of significant digits or in scientific notation.
        Trailing zeros after the decimal point are stripped.
        """
        if n == 0:
            return "0"
        # Use scientific notation for very large or very small numbers but not zero
        elif abs(n) >= 10**max_digits or (abs(n) < 10 ** (-max_digits + 1)):
            formatted_number = f"{n:.1e}"
        else:
            formatted_number = f"{n:.{max_digits}g}"
        # Process scientific notation to adjust exponent formatting
        if "e" in formatted_number:
            base, exponent = formatted_number.split("e")
            base = base.rstrip("0").rstrip(".")
            # Fix handling for exponent sign
            sign = exponent[0] if exponent.startswith("-") else "+"
            exponent_number = exponent.lstrip("+").lstrip("-").lstrip("0") or "0"
            formatted_number = f"{base}e{sign}{exponent_number}"
        else:
            # Truncate to max_digits significant digits for non-scientific notation
            if "." in formatted_number:
                formatted_number = formatted_number[: formatted_number.find(".") + max_digits]
        return formatted_number

    @staticmethod
    def point_matches_coordinates(point: PointLike, x: Number, y: Number) -> bool:
        """Check if a point matches given coordinates within tolerance.

        Uses global EPSILON tolerance for floating-point comparison to handle
        precision issues in coordinate matching.

        Args:
            point: Point object with x and y attributes
            x (float): Target x-coordinate to match
            y (float): Target y-coordinate to match

        Returns:
            bool: True if point coordinates match within tolerance, False otherwise
        """
        # Check if differences are within epsilon
        px = float(point.x)
        py = float(point.y)
        x_match = abs(px - float(x)) < MathUtils.EPSILON
        y_match = abs(py - float(y)) < MathUtils.EPSILON
        return bool(x_match and y_match)

    @staticmethod
    def segment_matches_coordinates(
        segment: SegmentLike,
        x1: Number,
        y1: Number,
        x2: Number,
        y2: Number,
    ) -> bool:
        """Check if a segment matches given endpoint coordinates in any order.

        Tests both possible orderings of endpoints since segments are undirected.
        Uses tolerance-based coordinate matching.

        Args:
            segment: Segment object with point1 and point2 attributes
            x1, y1 (float): First endpoint coordinates
            x2, y2 (float): Second endpoint coordinates

        Returns:
            bool: True if segment endpoints match coordinates (in either order), False otherwise
        """
        first_direction_match = MathUtils.point_matches_coordinates(
            segment.point1, x1, y1
        ) and MathUtils.point_matches_coordinates(segment.point2, x2, y2)
        second_direction_match = MathUtils.point_matches_coordinates(
            segment.point1, x2, y2
        ) and MathUtils.point_matches_coordinates(segment.point2, x1, y1)
        return bool(first_direction_match or second_direction_match)

    @staticmethod
    def segment_matches_point_names(segment: SegmentLike, p1_name: str, p2_name: str) -> bool:
        """Check if a segment connects two points by their names.

        Tests both possible orderings of point names since segments are undirected.

        Args:
            segment: Segment object with point1 and point2 attributes
            p1_name (str): Name of first point
            p2_name (str): Name of second point

        Returns:
            bool: True if segment connects the named points (in either order), False otherwise
        """
        return bool(
            (segment.point1.name == p1_name and segment.point2.name == p2_name)
            or (segment.point1.name == p2_name and segment.point2.name == p1_name)
        )

    @staticmethod
    def segment_has_end_point(segment: SegmentLike, x: Number, y: Number) -> bool:
        """Check if a segment has an endpoint at the given coordinates.

        Uses tolerance-based coordinate matching to check if either endpoint
        of the segment matches the provided coordinates.

        Args:
            segment: Segment object with point1 and point2 attributes
            x, y (float): Coordinates to check as potential endpoint

        Returns:
            bool: True if coordinates match either segment endpoint, False otherwise
        """
        return bool(
            MathUtils.point_matches_coordinates(segment.point1, x, y)
            or MathUtils.point_matches_coordinates(segment.point2, x, y)
        )

    @staticmethod
    def get_2D_distance(p1: PointLike, p2: PointLike) -> float:
        """Calculate Euclidean distance between two points in 2D space.

        Uses the standard distance formula: sqrt((x2-x1)² + (y2-y1)²)

        Args:
            p1: Point object with x and y attributes
            p2: Point object with x and y attributes

        Returns:
            float: Euclidean distance between the two points
        """
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        distance = math.sqrt(dx**2 + dy**2)
        return float(distance)

    @staticmethod
    def project_point_onto_circle(
        point: PointLike,
        center_x: Number,
        center_y: Number,
        radius: Number,
        *,
        tolerance: Optional[float] = None,
    ) -> None:
        """Project a point onto the circumference of a circle defined by a center and radius.

        If the point already lies on the circle (within tolerance) nothing happens. If the point
        coincides with the center (within tolerance) a ValueError is raised since projection
        would be undefined. Otherwise, the point is moved along the ray from the center through
        the point until it lies on the circle.

        Args:
            point: Point object with mutable x and y attributes.
            center_x: Circle center x-coordinate.
            center_y: Circle center y-coordinate.
            radius: Circle radius (must be positive).
            tolerance: Optional absolute tolerance override for near-zero comparisons.
        """
        if radius is None or float(radius) <= 0:
            raise ValueError("Circle radius must be a positive number.")

        dx = float(point.x) - float(center_x)
        dy = float(point.y) - float(center_y)
        distance = math.hypot(dx, dy)

        if tolerance is not None:
            radius_tol = float(tolerance)
        else:
            radius_value = abs(float(radius))
            radius_tol = MathUtils.EPSILON * max(1.0, radius_value)
            radius_tol = max(radius_tol, MathUtils.EPSILON)

        if math.isclose(distance, float(radius), abs_tol=radius_tol):
            return

        center_tol = max(MathUtils.EPSILON, 1e-12)
        if distance <= center_tol:
            raise ValueError("Cannot project a point that coincides with the circle center.")

        scale = float(radius) / distance
        point.x = float(center_x) + dx * scale
        point.y = float(center_y) + dy * scale

    @staticmethod
    def point_on_circle(
        point: PointLike,
        *,
        center_x: Number,
        center_y: Number,
        radius: Number,
        tolerance: Optional[float] = None,
        strict: bool = True,
    ) -> bool:
        """Validate that a single point lies on a circle."""
        if tolerance is not None:
            tol = float(tolerance)
        else:
            radius_value = abs(float(radius))
            tol = MathUtils.EPSILON * max(1.0, radius_value)
            tol = max(tol, MathUtils.EPSILON)
        distance = math.hypot(point.x - float(center_x), point.y - float(center_y))
        if abs(distance - float(radius)) > tol:
            if strict:
                raise ValueError(f"Point '{getattr(point, 'name', '')}' is not on the expected circle.")
            return False
        return True

    @staticmethod
    def get_2D_midpoint(p1: PointLike, p2: PointLike) -> Tuple[float, float]:
        """Calculate the midpoint between two points in 2D space.

        Returns the point exactly halfway between the two input points.

        Args:
            p1: Point object with x and y attributes
            p2: Point object with x and y attributes

        Returns:
            tuple: (x, y) coordinates of the midpoint
        """
        x = (p1.x + p2.x) / 2
        y = (p1.y + p2.y) / 2
        return float(x), float(y)

    @staticmethod
    def is_point_on_segment(
        px: Number,
        py: Number,
        sp1x: Number,
        sp1y: Number,
        sp2x: Number,
        sp2y: Number,
    ) -> bool:
        """Check if a point lies on a line segment between two endpoints.

        Uses bounding box check followed by adaptive collinearity test.
        Handles vertical and horizontal lines specially for better precision.

        Args:
            px, py (float): Coordinates of point to test
            sp1x, sp1y (float): Coordinates of segment first endpoint
            sp2x, sp2y (float): Coordinates of segment second endpoint

        Returns:
            bool: True if point lies on the segment, False otherwise
        """
        # Check if point is within bounding box of the segment
        if not ((min(sp1x, sp2x) <= px <= max(sp1x, sp2x)) and (min(sp1y, sp2y) <= py <= max(sp1y, sp2y))):
            return False

        # For vertical lines, check if x values match
        if abs(sp1x - sp2x) < 1e-10:
            return abs(px - sp1x) < 1e-5

        # For horizontal lines, check if y values match
        if abs(sp1y - sp2y) < 1e-10:
            return abs(py - sp1y) < 1e-5

        # Check if point is on the line defined by the segment
        # Using the cross product approach to check if three points are collinear
        from drawables.point import Position

        origin = Position(sp1x, sp1y)
        p1 = Position(sp2x, sp2y)
        p2 = Position(px, py)
        cross_product = MathUtils.cross_product(origin, p1, p2)

        # Calculate segment length for a better threshold
        segment_length = math.sqrt((sp2x - sp1x) ** 2 + (sp2y - sp1y) ** 2)

        # |cross| / length is the point's distance from the segment's line. Allow up to 0.01
        # (tolerates rounded coordinates), but scale with the segment length so tiny segments
        # are not over-matched and very long ones are not under-matched.
        distance = abs(cross_product) / segment_length
        threshold = max(1e-5 * segment_length, min(0.01, 1e-3 * segment_length))

        return distance < threshold

    @staticmethod
    def _segment_endpoints(segment: SegmentLike) -> Tuple[float, float, float, float]:
        """
        Normalize different segment representations into endpoint tuples.
        """
        if hasattr(segment, "point1") and hasattr(segment, "point2"):
            return (
                float(segment.point1.x),
                float(segment.point1.y),
                float(segment.point2.x),
                float(segment.point2.y),
            )
        if isinstance(segment, (list, tuple)) and len(segment) == 2:
            (x1, y1), (x2, y2) = segment
            return float(x1), float(y1), float(x2), float(y2)
        raise ValueError("Unsupported segment representation")

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        tau = 2 * math.pi
        return float(angle % tau)

    @staticmethod
    def _arc_angle_sequence(
        start_angle: float,
        end_angle: float,
        num_samples: int,
        *,
        clockwise: bool = False,
    ) -> List[float]:
        """
        Generate a sequence of angles along an arc including both endpoints.
        """
        if num_samples < 2:
            num_samples = 2

        start = MathUtils._normalize_angle(start_angle)
        end = MathUtils._normalize_angle(end_angle)
        tau = 2 * math.pi

        if clockwise:
            span = (start - end) % tau
            direction = -1.0
        else:
            span = (end - start) % tau
            direction = 1.0

        if span == 0.0:
            span = tau

        step = span / (num_samples - 1)
        angles: List[float] = []
        for idx in range(num_samples):
            angle = start + direction * step * idx
            angles.append(MathUtils._normalize_angle(angle))
        angles[-1] = end
        return angles

    @staticmethod
    def circle_segment_intersections(
        cx: Number,
        cy: Number,
        radius: Number,
        segment: SegmentLike,
        *,
        epsilon: float = 1e-9,
    ) -> List[Dict[str, float]]:
        """
        Compute intersection points (if any) between a circle and a segment.
        Returns a list of dicts with x, y, and angle (radians from positive x-axis).
        """
        r = float(radius)
        if r <= 0:
            return []
        x1, y1, x2, y2 = MathUtils._segment_endpoints(segment)
        dx = x2 - x1
        dy = y2 - y1
        fx = x1 - float(cx)
        fy = y1 - float(cy)
        a = dx * dx + dy * dy
        if abs(a) < epsilon:
            return []
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r
        discriminant = b * b - 4 * a * c
        if discriminant < -epsilon:
            return []
        discriminant = max(discriminant, 0.0)
        sqrt_disc = math.sqrt(discriminant)
        intersections: List[Dict[str, float]] = []
        signs = [-1.0, 1.0] if sqrt_disc > epsilon else [0.0]
        for sign in signs:
            t = (-b + sign * sqrt_disc) / (2 * a)
            if t < -epsilon or t > 1 + epsilon:
                continue
            t = min(max(t, 0.0), 1.0)
            ix = x1 + t * dx
            iy = y1 + t * dy
            angle = math.atan2(iy - float(cy), ix - float(cx))
            intersections.append({"x": ix, "y": iy, "angle": MathUtils._normalize_angle(angle)})
        return intersections

    @staticmethod
    def ellipse_segment_intersections(
        cx: Number,
        cy: Number,
        radius_x: Number,
        radius_y: Number,
        rotation_degrees: Number,
        segment: SegmentLike,
        *,
        epsilon: float = 1e-9,
    ) -> List[Dict[str, float]]:
        """
        Compute intersection points between an ellipse and a segment.
        Returns dicts with x, y, and parameter angle (radians on the ellipse).
        """
        rx = float(radius_x)
        ry = float(radius_y)
        if rx <= 0 or ry <= 0:
            return []
        rot_rad = math.radians(float(rotation_degrees))
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)

        x1, y1, x2, y2 = MathUtils._segment_endpoints(segment)

        def to_local(px: float, py: float) -> Tuple[float, float]:
            tx = px - float(cx)
            ty = py - float(cy)
            local_x = tx * cos_r + ty * sin_r
            local_y = -tx * sin_r + ty * cos_r
            return local_x, local_y

        def to_world(px: float, py: float) -> Tuple[float, float]:
            world_x = px * cos_r - py * sin_r + float(cx)
            world_y = px * sin_r + py * cos_r + float(cy)
            return world_x, world_y

        lx1, ly1 = to_local(x1, y1)
        lx2, ly2 = to_local(x2, y2)

        dx = lx2 - lx1
        dy = ly2 - ly1
        a = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry)
        b = 2 * ((lx1 * dx) / (rx * rx) + (ly1 * dy) / (ry * ry))
        c = (lx1 * lx1) / (rx * rx) + (ly1 * ly1) / (ry * ry) - 1

        if abs(a) < epsilon:
            return []

        discriminant = b * b - 4 * a * c
        if discriminant < -epsilon:
            return []
        discriminant = max(discriminant, 0.0)
        sqrt_disc = math.sqrt(discriminant)

        intersections: List[Dict[str, float]] = []
        signs = [-1.0, 1.0] if sqrt_disc > epsilon else [0.0]
        for sign in signs:
            t = (-b + sign * sqrt_disc) / (2 * a)
            if t < -epsilon or t > 1 + epsilon:
                continue
            t = min(max(t, 0.0), 1.0)
            local_x = lx1 + t * dx
            local_y = ly1 + t * dy
            world_x, world_y = to_world(local_x, local_y)
            theta = math.atan2(local_y / ry, local_x / rx)
            intersections.append({"x": world_x, "y": world_y, "angle": MathUtils._normalize_angle(theta)})
        return intersections

    @staticmethod
    def sample_circle_arc(
        cx: Number,
        cy: Number,
        radius: Number,
        start_angle: Number,
        end_angle: Number,
        *,
        num_samples: int = 64,
        clockwise: bool = False,
    ) -> List[Tuple[float, float]]:
        """
        Sample points along a circular arc.
        """
        r = float(radius)
        if r <= 0:
            return []
        angles = MathUtils._arc_angle_sequence(float(start_angle), float(end_angle), num_samples, clockwise=clockwise)
        points: List[Tuple[float, float]] = []
        for angle in angles:
            points.append(
                (
                    float(cx) + r * math.cos(angle),
                    float(cy) + r * math.sin(angle),
                )
            )
        return points

    @staticmethod
    def sample_ellipse_arc(
        cx: Number,
        cy: Number,
        radius_x: Number,
        radius_y: Number,
        start_angle: Number,
        end_angle: Number,
        *,
        rotation_degrees: Number = 0.0,
        num_samples: int = 64,
        clockwise: bool = False,
    ) -> List[Tuple[float, float]]:
        """
        Sample points along an elliptical arc (respecting rotation).
        The start and end angles are parameter angles before ellipse rotation.
        """
        rx = float(radius_x)
        ry = float(radius_y)
        if rx <= 0 or ry <= 0:
            return []

        rot_rad = math.radians(float(rotation_degrees))
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        angles = MathUtils._arc_angle_sequence(float(start_angle), float(end_angle), num_samples, clockwise=clockwise)
        points: List[Tuple[float, float]] = []
        for angle in angles:
            local_x = rx * math.cos(angle)
            local_y = ry * math.sin(angle)
            world_x = local_x * cos_r - local_y * sin_r + float(cx)
            world_y = local_x * sin_r + local_y * cos_r + float(cy)
            points.append((world_x, world_y))
        return points

    @staticmethod
    def get_triangle_area(p1: PointLike, p2: PointLike, p3: PointLike) -> float:
        """Calculate the area of a triangle using Heron's formula.

        Computes triangle area from three vertices using side lengths
        and the semi-perimeter formula.

        Args:
            p1, p2, p3: Point objects with x and y attributes representing triangle vertices

        Returns:
            float: Area of the triangle
        """
        # Calculate the area of the triangle using Heron's formula
        a = MathUtils.get_2D_distance(p1, p2)
        b = MathUtils.get_2D_distance(p2, p3)
        c = MathUtils.get_2D_distance(p3, p1)
        s = (a + b + c) / 2
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        return area

    @staticmethod
    def get_triangle_centroid(p1: PointLike, p2: PointLike, p3: PointLike) -> Tuple[float, float]:
        """Calculate the centroid (geometric center) of a triangle.

        Returns the point where the three medians of the triangle intersect.

        Args:
            p1, p2, p3: Point objects with x and y attributes representing triangle vertices

        Returns:
            tuple: (x, y) coordinates of the triangle centroid
        """
        x = (p1.x + p2.x + p3.x) / 3
        y = (p1.y + p2.y + p3.y) / 3
        return x, y

    @staticmethod
    def get_rectangle_area(diagonal_p1: PointLike, diagonal_p2: PointLike) -> float:
        """Calculate the area of a rectangle from diagonal points.

        Assumes the rectangle is axis-aligned and computes area
        from the width and height derived from diagonal endpoints.

        Args:
            diagonal_p1: Point object representing one corner of rectangle
            diagonal_p2: Point object representing opposite corner of rectangle

        Returns:
            float: Area of the rectangle
        """
        width = abs(diagonal_p1.x - diagonal_p2.x)
        height = abs(diagonal_p1.y - diagonal_p2.y)
        area = width * height
        return float(area)

    @staticmethod
    def cross_product(origin: PointLike, p1: PointLike, p2: PointLike) -> float:
        """Calculate the 2D cross product of two vectors from an origin point.

        Computes the z-component of the cross product of vectors (origin->p1) and (origin->p2).
        Used for orientation testing and area calculations.

        Args:
            origin: Point object representing vector origin
            p1: Point object representing end of first vector
            p2: Point object representing end of second vector

        Returns:
            float: Cross product value (positive for counter-clockwise, negative for clockwise)
        """
        result = (p1.x - origin.x) * (p2.y - origin.y) - (p2.x - origin.x) * (p1.y - origin.y)
        return float(result)

    @staticmethod
    def dot_product(origin: PointLike, p1: PointLike, p2: PointLike) -> float:
        """Calculate the dot product of two vectors from an origin point.

        Computes the dot product of vectors (origin->p1) and (origin->p2).
        Used for angle calculations and orthogonality testing.

        Args:
            origin: Point object representing vector origin
            p1: Point object representing end of first vector
            p2: Point object representing end of second vector

        Returns:
            float: Dot product value
        """
        vec1 = (p1.x - origin.x, p1.y - origin.y)
        vec2 = (p2.x - origin.x, p2.y - origin.y)
        result = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        return float(result)

    @staticmethod
    def calculate_angle_degrees(
        vertex_coords: Sequence[Number],
        arm1_coords: Sequence[Number],
        arm2_coords: Sequence[Number],
    ) -> Optional[float]:
        """
        Calculates the angle in degrees formed by three points: vertex, point on arm1, point on arm2.
        The angle is measured counter-clockwise from the vector (vertex -> arm1) to (vertex -> arm2).
        Returns the angle in the range [0, 360) degrees, or None if calculation is not possible.

        Args:
            vertex_coords (tuple): (x, y) coordinates of the vertex.
            arm1_coords (tuple): (x, y) coordinates of a point on the first arm.
            arm2_coords (tuple): (x, y) coordinates of a point on the second arm.
        """
        if not MathUtils.are_points_valid_for_angle_geometry(vertex_coords, arm1_coords, arm2_coords):
            return None

        vx, vy = vertex_coords
        p1x, p1y = arm1_coords
        p2x, p2y = arm2_coords

        # Vector from vertex to arm1_point
        v1x = p1x - vx
        v1y = p1y - vy
        # Vector from vertex to arm2_point
        v2x = p2x - vx
        v2y = p2y - vy

        # Angle of v1 and v2 with respect to positive x-axis
        angle1_rad = math.atan2(v1y, v1x)
        angle2_rad = math.atan2(v2y, v2x)

        # Angle difference in radians
        angle_rad = angle2_rad - angle1_rad

        # Normalize to be between -pi and pi (though atan2 typically gives this range for each angle)
        # The difference, however, might be outside. Normalizing the *difference* is key.
        if angle_rad > math.pi:
            angle_rad -= 2 * math.pi
        elif angle_rad < -math.pi:
            angle_rad += 2 * math.pi

        # Convert to degrees and normalize to [0, 360)
        angle_degrees = math.degrees(angle_rad)
        if angle_degrees < 0:
            angle_degrees += 360

        return angle_degrees

    @staticmethod
    def are_points_valid_for_angle_geometry(
        vertex_coords: Sequence[Number],
        arm1_coords: Sequence[Number],
        arm2_coords: Sequence[Number],
    ) -> bool:
        """
        Checks if three points can form a geometrically valid, non-degenerate angle.
        Specifically, arm points must be distinct from the vertex and from each other.

        Args:
            vertex_coords (tuple): (x, y) coordinates of the vertex.
            arm1_coords (tuple): (x, y) coordinates of a point on the first arm.
            arm2_coords (tuple): (x, y) coordinates of a point on the second arm.

        Returns:
            bool: True if the points form a valid angle geometry, False otherwise.
        """
        vx, vy = vertex_coords
        p1x, p1y = arm1_coords
        p2x, p2y = arm2_coords

        # Check if arm1_point is coincident with vertex_point (zero length arm1)
        if abs(p1x - vx) < MathUtils.EPSILON and abs(p1y - vy) < MathUtils.EPSILON:
            return False

        # Check if arm2_point is coincident with vertex_point (zero length arm2)
        if abs(p2x - vx) < MathUtils.EPSILON and abs(p2y - vy) < MathUtils.EPSILON:
            return False

        # Check if arm1_point is coincident with arm2_point (overlapping arms)
        if abs(p1x - p2x) < MathUtils.EPSILON and abs(p1y - p2y) < MathUtils.EPSILON:
            return False

        return True

    @staticmethod
    def is_right_angle(origin: PointLike, p1: PointLike, p2: PointLike) -> bool:
        """Check if two vectors from an origin form a right angle (90 degrees).

        Uses dot product test with tolerance for floating-point precision.
        Two vectors are perpendicular if their dot product is zero.

        Args:
            origin: Point object representing vertex of the angle
            p1: Point object representing end of first vector
            p2: Point object representing end of second vector

        Returns:
            bool: True if vectors form a right angle, False otherwise
        """
        dot_product = MathUtils.dot_product(origin, p1, p2)
        # Normalize by the vector lengths so the tolerance is independent of scale
        norms = math.hypot(p1.x - origin.x, p1.y - origin.y) * math.hypot(p2.x - origin.x, p2.y - origin.y)
        if norms == 0:
            return False
        return abs(dot_product) / norms < 1e-9

    @staticmethod
    def is_rectangle(
        x1: Number,
        y1: Number,
        x2: Number,
        y2: Number,
        x3: Number,
        y3: Number,
        x4: Number,
        y4: Number,
    ) -> bool:  # points must be in clockwise or counterclockwise order
        from drawables.point import Position

        points = [Position(x, y) for x, y in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]]

        # Check for duplicate points with tolerance
        TOLERANCE = 1e-10
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i != j and abs(p1.x - p2.x) < TOLERANCE and abs(p1.y - p2.y) < TOLERANCE:
                    return False

        # Calculate all pairwise distances
        distances = [
            MathUtils.get_2D_distance(p1, p2) for i, p1 in enumerate(points) for j, p2 in enumerate(points) if i < j
        ]

        # Group similar distances using a tolerance relative to their magnitude
        grouped_distances: List[List[float]] = []
        for d in distances:
            found_group = False
            for group in grouped_distances:
                if abs(group[0] - d) < 1e-9 * max(group[0], d):
                    group.append(d)
                    found_group = True
                    break
            if not found_group:
                grouped_distances.append([d])

        # Count occurrences in each group
        distance_counts = [len(group) for group in grouped_distances]
        distance_counts.sort()

        # Check for valid rectangle patterns (2 groups with [2,4] counts for squares, or 3 groups with [2,2,2] counts for rectangles)
        if len(distance_counts) not in [2, 3]:
            return False
        if len(distance_counts) == 2 and distance_counts != [2, 4]:
            return False
        if len(distance_counts) == 3 and distance_counts != [2, 2, 2]:
            return False

        # Check for right angles with tolerance
        for i in range(4):
            vertex = points[i]
            next_point = points[(i + 1) % 4]
            prev_point = points[(i - 1) % 4]
            if not MathUtils.is_right_angle(vertex, next_point, prev_point):
                return False

        return True

    # DEPRECATED BUT FASTER
    @staticmethod
    def evaluate_expression_using_python(expression: str) -> float:
        """[DEPRECATED] Evaluate a mathematical expression at x=0.

        Legacy method for quick expression evaluation. Use symbolic methods instead.

        Args:
            expression (str): Mathematical expression string

        Returns:
            float: Result of evaluating expression at x=0
        """
        from expression_validator import ExpressionValidator

        result = ExpressionValidator.parse_function_string(expression)(0)
        return float(result)

    @staticmethod
    def points_orientation(
        p1x: Number,
        p1y: Number,
        p2x: Number,
        p2y: Number,
        p3x: Number,
        p3y: Number,
    ) -> int:
        """Determine the orientation of three points in 2D space.

        Uses cross product to determine if three points form a clockwise,
        counter-clockwise, or collinear arrangement.

        Args:
            p1x, p1y (float): Coordinates of first point
            p2x, p2y (float): Coordinates of second point
            p3x, p3y (float): Coordinates of third point

        Returns:
            int: 0 for collinear, 1 for clockwise, 2 for counter-clockwise
        """
        # Calculate orientation of triplet (p1, p2, p3)
        val = float((p2y - p1y) * (p3x - p2x) - (p2x - p1x) * (p3y - p2y))
        if val == 0:
            return 0  # Collinear
        elif val > 0:
            return 1  # Clockwise
        else:
            return 2  # Counterclockwise

    @staticmethod
    def segments_intersect(
        s1x1: Number,
        s1y1: Number,
        s1x2: Number,
        s1y2: Number,
        s2x1: Number,
        s2y1: Number,
        s2x2: Number,
        s2y2: Number,
    ) -> bool:
        # Find orientations
        o1 = MathUtils.points_orientation(s1x1, s1y1, s1x2, s1y2, s2x1, s2y1)
        o2 = MathUtils.points_orientation(s1x1, s1y1, s1x2, s1y2, s2x2, s2y2)
        o3 = MathUtils.points_orientation(s2x1, s2y1, s2x2, s2y2, s1x1, s1y1)
        o4 = MathUtils.points_orientation(s2x1, s2y1, s2x2, s2y2, s1x2, s1y2)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special Cases using the revised is_point_on_segment
        if o1 == 0 and MathUtils.is_point_on_segment(s2x1, s2y1, s1x1, s1y1, s1x2, s1y2):
            return True
        if o2 == 0 and MathUtils.is_point_on_segment(s2x2, s2y2, s1x1, s1y1, s1x2, s1y2):
            return True
        if o3 == 0 and MathUtils.is_point_on_segment(s1x1, s1y1, s2x1, s2y1, s2x2, s2y2):
            return True
        if o4 == 0 and MathUtils.is_point_on_segment(s1x2, s1y2, s2x1, s2y1, s2x2, s2y2):
            return True

        return False

    @staticmethod
    def get_line_formula(x1: Number, y1: Number, x2: Number, y2: Number) -> str:
        # Calculate the slope
        if x2 - x1 != 0:  # Avoid division by zero
            m = (y2 - y1) / (x2 - x1)
        else:
            return "x = " + str(x1)  # The line is vertical
        # Calculate the y-intercept
        b = y1 - m * x1
        # Return the algebraic expression
        if b >= 0:
            return f"y = {m} * x + {b}"
        else:
            return f"y = {m} * x - {-b}"  # Use -b to make sure the minus sign is printed correctly

    @staticmethod
    def get_segments_intersection(
        s1_x1: Number,
        s1_y1: Number,
        s1_x2: Number,
        s1_y2: Number,
        s2_x1: Number,
        s2_y1: Number,
        s2_x2: Number,
        s2_y2: Number,
    ) -> Optional[Tuple[float, float]]:
        # Generate line formulas for both segments
        line1_formula = MathUtils.get_line_formula(s1_x1, s1_y1, s1_x2, s1_y2)
        line2_formula = MathUtils.get_line_formula(s2_x1, s2_y1, s2_x2, s2_y2)
        # Assuming solve_system_of_equations exists and handles these formulas
        solution = MathUtils.solve_system_of_equations([line1_formula, line2_formula])
        # Check if the solution is an error message
        if isinstance(solution, str) and solution.startswith("Error:"):
            return None
        # Parse the solution if it is in the form "x = 0.5, y = 0.5"
        if isinstance(solution, str) and ", " in solution:
            x_str, y_str = solution.split(", ")
            x = float(x_str.split(" = ")[1])
            y = float(y_str.split(" = ")[1])
            return x, y
        return None

    @staticmethod
    def get_circle_formula(x: Number, y: Number, r: Number) -> str:
        # Return the algebraic expression
        return f"(x - {x})**2 + (y - {y})**2 = {r}**2"

    @staticmethod
    def get_ellipse_formula(
        x: Number,
        y: Number,
        rx: Number,
        ry: Number,
        rotation_angle: Number = 0,
    ) -> str:
        """
        Get the algebraic formula for an ellipse, optionally rotated.
        Args:
            x, y: center coordinates
            rx, ry: radii in x and y directions
            rotation_angle: rotation in degrees (default 0)
        Returns:
            String representation of the ellipse formula
        """

        def fmt_num(n: Any) -> str:
            try:
                n_float = float(n)
                if n_float.is_integer():
                    return str(int(n_float))
                return str(n_float).rstrip("0").rstrip(".")
            except Exception:
                return str(n)

        fx, fy, frx, fry = fmt_num(x), fmt_num(y), fmt_num(rx), fmt_num(ry)

        if rotation_angle == 0:
            # Standard ellipse formula without rotation
            return f"((x - {fx})**2)/{frx}**2 + ((y - {fy})**2)/{fry}**2 = 1"
        else:
            # Convert angle to radians
            angle_rad = math.radians(rotation_angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            # Calculate coefficients for the rotated ellipse equation
            A = (cos_a**2 / rx**2) + (sin_a**2 / ry**2)
            B = 2 * cos_a * sin_a * (1 / rx**2 - 1 / ry**2)
            C = (sin_a**2 / rx**2) + (cos_a**2 / ry**2)

            # Format coefficients with significant digits in plain decimal notation (no exponent),
            # since fixed decimal places collapse small coefficients (large radii) to 0
            coef_scale = max(abs(A), abs(C))

            def fmt_coef(value: float) -> str:
                if not math.isfinite(value):
                    return str(value)
                if abs(value) <= 1e-12 * coef_scale:  # floating-point noise, e.g. cross term at 90 degrees
                    return "0"
                decimals = max(0, 9 - int(math.floor(math.log10(abs(value)))))
                text = format(value, "." + str(decimals) + "f")
                if "." in text:
                    text = text.rstrip("0").rstrip(".")
                return text

            a_str, b_str, c_str = fmt_coef(A), fmt_coef(B), fmt_coef(C)

            # Handle special cases for coefficient signs in the formula
            b_term = f"- {b_str[1:]}" if b_str.startswith("-") else f"+ {b_str}"

            return f"{a_str}*(x - {fx})**2 {b_term}*(x - {fx})*(y - {fy}) + {c_str}*(y - {fy})**2 = 1"

    @staticmethod
    def try_convert_to_number(value: Any) -> Any:
        try:
            return float(value)
        except Exception:
            return value

    @staticmethod
    def sqrt(x: Number) -> Any:
        try:
            result = window.math.format(window.math.sqrt(x))
            return MathUtils.try_convert_to_number(result)
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

    @staticmethod
    def pow(x: Number, exp: Number) -> Any:
        try:
            result = window.math.format(window.math.pow(x, exp))
            return MathUtils.try_convert_to_number(result)
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

    @staticmethod
    def det(matrix: Sequence[Sequence[Number]]) -> Any:
        try:
            result = window.math.format(window.math.det(matrix))
            return MathUtils.try_convert_to_number(result)
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

    @staticmethod
    def convert(value: Number, from_unit: str, to_unit: str) -> Any:
        try:
            return window.math.format(window.math.evaluate(f"{value} {from_unit} to {to_unit}"))
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

    @staticmethod
    def _normalize_symbols(value: Any) -> Any:
        """Rewrite Unicode math notation (π, x², ×, −, ≤, ∞, ...) in a raw expression for nerdamer or math.js.

        For expressions that do not pass through ExpressionValidator.fix_math_expression;
        ASCII strings and non-string values are returned unchanged.
        """
        if not isinstance(value, str):
            return value
        from expression_validator import ExpressionValidator

        return ExpressionValidator.normalize_unicode_math(value)

    @staticmethod
    def _normalize_variable(value: Any) -> Any:
        """Rewrite a variable name argument (ϕ -> φ) with character replacements only.

        A variable is one name, so it is never split into factors: "Δx" stays "Δx".
        Non-string values are returned unchanged.
        """
        if not isinstance(value, str):
            return value
        from expression_validator import ExpressionValidator

        return ExpressionValidator.normalize_unicode_name(value)

    # Number theory functions that require Python evaluation (not available in Math.js)
    _PYTHON_ONLY_FUNCTIONS = {
        "is_prime",
        "prime_factors",
        "mod_pow",
        "mod_inverse",
        "next_prime",
        "prev_prime",
        "totient",
        "divisors",
        "summation",
        "product",
        "arithmetic_sum",
        "geometric_sum",
        "geometric_sum_infinite",
        "ratio_test",
        "root_test",
        "p_series_test",
    }

    @staticmethod
    def evaluate(expression: str, variables: Optional[Dict[str, Number]] = None) -> Any:
        """Evaluate a mathematical expression numerically with optional variables.

        Uses Math.js for evaluation with expression validation and error handling.
        Supports complex mathematical expressions, functions, and variable substitution.
        For number theory functions (is_prime, prime_factors, etc.), uses Python evaluation.

        Args:
            expression (str): Mathematical expression string (e.g., "sin(x) + 2*y")
            variables (dict): Optional variable substitutions (e.g., {"x": 3.14, "y": 2})

        Returns:
            float or str: Numerical result or error message if evaluation fails
        """
        try:
            from expression_validator import ExpressionValidator

            js_expression = ExpressionValidator.fix_math_expression(expression, python_compatible=False)
            python_expression = ExpressionValidator.fix_math_expression(expression, python_compatible=True)
            ExpressionValidator.validate_expression_tree(python_expression)
            if variables:
                # The expression's ϕ became φ, so the scope's names must too
                variables = {
                    ExpressionValidator.normalize_unicode_name(name): value for name, value in variables.items()
                }

            # Check if expression contains Python-only functions (number theory; randint has no
            # inclusive math.js equivalent)
            if "randint(" in expression or any(func in expression for func in MathUtils._PYTHON_ONLY_FUNCTIONS):
                # Use Python evaluation for number theory functions
                result = ExpressionValidator.evaluate_expression(
                    python_expression, variables.get("x", 0) if variables else 0
                )
                # Preserve boolean and list types for better display
                if isinstance(result, bool):
                    return "True" if result else "False"
                if isinstance(result, list):
                    return str(result)
                return result

            # Map advertised names onto their math.js equivalents
            js_expression = js_expression.replace("arrangements(", "permutations(")
            js_expression = js_expression.replace("stdev(", "std(")
            js_expression = js_expression.replace("trunc(", "fix(")

            try:
                if not variables:
                    result = window.math.format(window.math.evaluate(js_expression))
                else:
                    result = window.math.format(window.math.evaluate(js_expression, variables))
            except Exception as e:
                # Brython cannot convert integer-valued JS numbers >= 2^53 ("not a big int"),
                # so let math.js format such results before they cross into Python
                if "not a big int" not in str(e):
                    raise
                formatted_expression = f"format({js_expression})"
                if not variables:
                    result = window.math.evaluate(formatted_expression)
                else:
                    result = window.math.evaluate(formatted_expression, variables)

            # math.format wraps string results (e.g. bin) in JSON quotes
            unquoted_result = MathUtils._unquote_formatted_string(result)
            if unquoted_result is not None:
                return unquoted_result
            result = MathUtils._unwrap_single_mode_result(js_expression, result)

            converted_result = MathUtils.try_convert_to_number(result)

            # Check for division by zero
            if (
                "lim" not in expression
                and "limit" not in expression
                and (
                    converted_result == float("-inf")
                    or converted_result == float("inf")
                    or str(converted_result).lower() in ["-inf", "inf", "infinity", "-infinity"]
                )
            ):
                raise ZeroDivisionError()

            return converted_result
        except ZeroDivisionError:
            return (
                "Error: Result is infinite (division by zero, overflow, "
                "or a value outside the function's domain such as log(0))"
            )
        except OverflowError:
            return "Error: Overflow - the result is too large to represent"
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

    @staticmethod
    def _unquote_formatted_string(result: Any) -> Optional[str]:
        """Return the plain text of a JSON-quoted math.format string result, else None."""
        if not isinstance(result, str) or len(result) < 2 or result[0] != '"' or result[-1] != '"':
            return None
        try:
            unquoted = json.loads(result)
        except Exception:
            return result[1:-1]
        return unquoted if isinstance(unquoted, str) else None

    @staticmethod
    def _unwrap_single_mode_result(js_expression: str, result: Any) -> Any:
        """Return the lone value of a top-level mode(...) result; math.js always returns an array of modes."""
        expression = js_expression.strip()
        if not expression.startswith("mode(") or not expression.endswith(")"):
            return result
        depth = 0
        for index, char in enumerate(expression[len("mode") :], start=len("mode")):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(expression) - 1:
                    return result  # mode(...) is only part of a larger expression
        if not isinstance(result, str) or not (result.startswith("[") and result.endswith("]")):
            return result
        values = MathUtils._split_top_level_commas(result[1:-1])
        return values[0] if len(values) == 1 else result

    @staticmethod
    def derivative(expression: str, variable: str) -> str:
        """Calculate the derivative of a mathematical expression.

        Uses Nerdamer symbolic computation for analytical differentiation.
        Supports all standard functions and multi-variable expressions.

        Args:
            expression (str): Mathematical expression to differentiate
            variable (str): Variable to differentiate with respect to (e.g., "x")

        Returns:
            str: Derivative expression as string or error message
        """
        expression, variable = MathUtils._normalize_symbols(expression), MathUtils._normalize_variable(variable)
        try:
            result = MathUtils._guarded_nerdamer_text(f"diff({expression}, {variable})", MathUtils.NERDAMER_TOOL_MAX_MS)
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"
        return MathUtils._took_too_long_message(f"the derivative of {expression}") if result is None else result

    @staticmethod
    def limit(expression: str, variable: str, value_to_approach: Union[Number, str]) -> str:
        """Calculate the limit of a mathematical expression.

        Uses Nerdamer symbolic computation for limit evaluation.
        Supports finite limits and limits at infinity.

        Args:
            expression (str): Mathematical expression
            variable (str): Variable approaching the limit (e.g., "x")
            value_to_approach (str/float): Value or "inf"/"-inf" for infinity

        Returns:
            str: Limit result as string or error message. nerdamer's limit runs under a step and
            time budget (static/nerdamer_guard.js): its L'Hopital loop never ends for quotients
            such as abs(x)/x, so a limit that runs out of budget is an error, with a numeric
            estimate when one is clear.
        """
        expression, variable = MathUtils._normalize_symbols(expression), MathUtils._normalize_variable(variable)
        try:
            value_to_approach = str(MathUtils._normalize_symbols(value_to_approach)).lower().replace(" ", "")
            if value_to_approach in ["inf", "infinity", "+inf", "+infinity"]:
                value_to_approach = "Infinity"
            elif value_to_approach in ["-inf", "-infinity"]:
                value_to_approach = "-Infinity"
            result = MathUtils._guarded_nerdamer_text(f"limit({expression}, {variable}, {value_to_approach})")
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"
        if result is None:
            return MathUtils._abandoned_limit_message(expression, variable, str(value_to_approach))
        return result

    # A guarded nerdamer computation (static/nerdamer_guard.js) is stopped after this many
    # derivative and limit steps (limits only) or once this many milliseconds have passed; it
    # stops within a few hundred milliseconds of the deadline. Limits, and the solve() behind
    # asymptote detection, get NERDAMER_MAX_MS; the derive, integrate, simplify, expand,
    # factor and solve tools get NERDAMER_TOOL_MAX_MS.
    NERDAMER_MAX_STEPS = 2000
    NERDAMER_MAX_MS = 1500
    NERDAMER_TOOL_MAX_MS = 5000
    _NO_STEP_LIMIT = 10**9

    @staticmethod
    def _guarded_nerdamer_text(
        nerdamer_input: str, max_ms: Optional[int] = None, decimals: bool = False
    ) -> Optional[str]:
        """Evaluate nerdamer input under the time budget; None if the budget ran out.

        Only limits count derivative steps (max_ms None means a limit: NERDAMER_MAX_MS and
        NERDAMER_MAX_STEPS). With decimals the result is evaluated and printed as decimals.
        Falls back to plain nerdamer when the guard script is not loaded. Errors raised by
        nerdamer are raised again as ValueError.
        """
        if not hasattr(window, "MatHudGuardedNerdamer"):
            parsed = window.nerdamer(nerdamer_input)
            return str(parsed.evaluate().text("decimals") if decimals else parsed.text())
        max_steps = MathUtils.NERDAMER_MAX_STEPS if max_ms is None else MathUtils._NO_STEP_LIMIT
        outcome = window.MatHudGuardedNerdamer(
            nerdamer_input, max_steps, max_ms or MathUtils.NERDAMER_MAX_MS, bool(decimals)
        )
        if outcome.exceeded:
            return None
        if outcome.error:
            raise ValueError(str(outcome.error))
        return str(outcome.text)

    @staticmethod
    def _took_too_long_message(what: str) -> str:
        """Error text for a nerdamer computation the guard stopped."""
        seconds = MathUtils.NERDAMER_TOOL_MAX_MS / 1000
        return (
            f"Error: Computing {what} took too long and was stopped after about {seconds:g} s; "
            "the expression is probably too large for the symbolic engine."
        )

    @staticmethod
    def _abandoned_limit_message(expression: str, variable: str, value_to_approach: str) -> str:
        """Error text for a limit nerdamer could not finish, with a numeric estimate where one is clear."""
        message = (
            f"Error: The limit of {expression} as {variable} -> {value_to_approach} could not be computed "
            "symbolically: the computation did not finish (nerdamer's L'Hopital loop does not end for "
            "quotients with abs(), sqrt(x^2) or similar)."
        )
        estimate = MathUtils._numeric_limit_estimate(expression, variable, value_to_approach)
        return f"{message} {estimate}" if estimate else message

    @staticmethod
    def _numeric_limit_estimate(expression: str, variable: str, value_to_approach: str) -> Optional[str]:
        """Describe what f does near the target numerically, or None when the samples are unclear."""
        try:
            evaluate_ieee = MathUtils._mathjs_probe_evaluator(expression, variable)
            if evaluate_ieee is None:
                return None

            def evaluate(point: float) -> Optional[float]:
                value = evaluate_ieee(point)
                return None if value is None or math.isnan(value) else value

            if value_to_approach in ("Infinity", "-Infinity"):
                sign = -1.0 if value_to_approach.startswith("-") else 1.0
                limit_value = MathUtils._limit_at_infinity(evaluate_ieee, sign)
                if limit_value is None:
                    return None
                return (
                    f"Numerically, the expression approaches {limit_value:.12g} as {variable} -> {value_to_approach}."
                )
            x0 = float(window.math.evaluate(value_to_approach))
            if not math.isfinite(x0):
                return None
            start = MathUtils._PROBE_START * max(1.0, abs(x0))
            sides = [
                MathUtils._settled_value(MathUtils._probe_side(evaluate, x0, side, start) or []) for side in (-1.0, 1.0)
            ]
        except Exception:
            return None
        left, right = sides
        if left is None or right is None:
            return None
        if abs(left - right) <= 1e-9 * max(1.0, abs(left), abs(right)):
            return f"Numerically, the expression approaches {left:.12g} from both sides."
        return (
            f"Numerically, it approaches {left:.12g} from the left and {right:.12g} from the right, "
            "so the two-sided limit does not exist."
        )

    @staticmethod
    def integral(
        expression: str,
        variable: str,
        lower_bound: Optional[Number] = None,
        upper_bound: Optional[Number] = None,
    ) -> str:
        """Calculate the integral of a mathematical expression.

        Uses Nerdamer symbolic computation for integration.
        Supports both indefinite and definite integrals.

        Args:
            expression (str): Mathematical expression to integrate
            variable (str): Variable of integration (e.g., "x")
            lower_bound (float): Optional lower bound for definite integral
            upper_bound (float): Optional upper bound for definite integral

        Returns:
            str: Integral result as string or error message
        """
        import re

        expression, variable = MathUtils._normalize_symbols(expression), MathUtils._normalize_variable(variable)
        lower_bound, upper_bound = MathUtils._normalize_symbols(lower_bound), MathUtils._normalize_symbols(upper_bound)
        try:
            antiderivative = MathUtils._guarded_nerdamer_text(
                f"integrate({expression}, {variable})", MathUtils.NERDAMER_TOOL_MAX_MS
            )
            if antiderivative is None:
                return MathUtils._took_too_long_message(f"the integral of {expression}")
            if lower_bound is None and upper_bound is None:
                return antiderivative
            indefinite_integral = window.nerdamer(antiderivative)
            evaluated_at_upper = indefinite_integral.sub(variable, upper_bound).text()
            evaluated_at_lower = indefinite_integral.sub(variable, lower_bound).text()
            result = str(window.nerdamer(f"{evaluated_at_upper} - {evaluated_at_lower}").evaluate().text())
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

        # F(b) - F(a) is only valid when the integrand has no singularity inside the interval
        singular_point = (
            MathUtils._find_interior_singularity(expression, variable, lower_bound, upper_bound)
            if lower_bound is not None and upper_bound is not None
            else None
        )
        if singular_point is not None:
            return (
                f"Error: The integrand {expression} is singular or undefined near {variable} = {singular_point:.6g} "
                f"inside [{lower_bound}, {upper_bound}], so this is an improper integral that may diverge; "
                "the antiderivative cannot be evaluated across it."
            )
        if re.search(r"\bi\b", result):
            return (
                f"Error: The definite integral of {expression} over [{lower_bound}, {upper_bound}] is not a real "
                "number; the integrand is likely singular or undefined in the interval (improper or divergent integral)."
            )
        return result

    @staticmethod
    def _find_interior_singularity(
        expression: str,
        variable: str,
        lower_bound: Union[Number, str],
        upper_bound: Union[Number, str],
    ) -> Optional[float]:
        """Scan the integrand on a fine grid for interior points where it is undefined or blows up.

        Returns the approximate location of such a point, or None when the integrand looks
        finite inside the interval or cannot be evaluated numerically. Singularities exactly
        at an endpoint are ignored; they are handled by the antiderivative evaluation.
        """
        try:
            from expression_validator import ExpressionValidator

            compiled = window.math.compile(ExpressionValidator.fix_math_expression(str(expression)))
            a = float(window.math.evaluate(str(lower_bound)))
            b = float(window.math.evaluate(str(upper_bound)))
            if not (math.isfinite(a) and math.isfinite(b)) or a == b:
                return None

            def value_at(x: float) -> Optional[float]:
                value = compiled.evaluate({variable: x})
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return None  # complex or non-numeric: undefined over the reals
                return float(value)

            steps = 2000
            step = (b - a) / steps
            xs = [a + k * step for k in range(steps + 1)]
            raw_values = [value_at(x) for x in xs]
            finite_magnitudes = sorted(abs(v) for v in raw_values if v is not None and math.isfinite(v))
            typical_scale = finite_magnitudes[len(finite_magnitudes) // 2] if finite_magnitudes else 0.0
            probe_step = 1e-7 * abs(b - a)

            def magnitude(x: float, value: Optional[float] = None) -> Optional[float]:
                if value is None:
                    value = value_at(x)
                if value is not None and math.isnan(value):
                    # 0/0 at a removable point (e.g. sin(x)/x at 0) is NaN but not a singularity
                    value = MathUtils._removable_limit(value_at, x, probe_step, typical_scale)
                if value is None or not math.isfinite(value):
                    return None
                return abs(value)

            values = [magnitude(x, value) for x, value in zip(xs, raw_values)]
            for k in range(1, steps):
                if values[k] is None:
                    return xs[k]

            # Zoom into the largest local maxima of |f|: a pole keeps growing, a smooth peak does not
            def at(k: int) -> float:
                value = values[k]
                return math.inf if value is None else value

            peaks = [k for k in range(1, steps) if at(k) >= at(k - 1) and at(k) >= at(k + 1)]
            peaks.sort(key=at, reverse=True)
            for k in peaks[:20]:
                lo, hi = sorted((xs[k - 1], xs[k + 1]))
                location = MathUtils._refine_blow_up(magnitude, lo, hi, at(k))
                if location is not None and min(abs(location - a), abs(location - b)) > abs(step) * 1e-6:
                    return location
            return None
        except Exception:
            return None

    @staticmethod
    def _removable_limit(
        value_at: Callable[[float], Optional[float]], x: float, probe_step: float, typical_scale: float
    ) -> Optional[float]:
        """Return the limit of f at x when f is undefined there but continuous around it, else None.

        f is probed at x +/- h and x +/- 2h: a removable point gives four nearly equal finite
        values, while a pole or a jump makes them differ markedly.
        """
        h = max(probe_step, 1e-12 * max(1.0, abs(x)))
        probes = [value_at(x + offset) for offset in (-2 * h, -h, h, 2 * h)]
        finite = [p for p in probes if p is not None and math.isfinite(p)]
        if len(finite) != len(probes):
            return None
        tolerance = 1e-3 * max(typical_scale, max(abs(p) for p in finite))
        if max(finite) - min(finite) > tolerance:
            return None
        return (finite[1] + finite[2]) / 2

    @staticmethod
    def _refine_blow_up(magnitude: Any, lo: float, hi: float, grid_value: float) -> Optional[float]:
        """Ternary-search the maximum of |f| on [lo, hi]; return its location if it grows without bound.

        magnitude(x) returns |f(x)|, or None where f is undefined or infinite.
        """
        for _ in range(100):
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            v1 = magnitude(m1)
            if v1 is None:
                return m1
            v2 = magnitude(m2)
            if v2 is None:
                return m2
            if v1 < v2:
                lo = m1
            else:
                hi = m2
        peak_location = (lo + hi) / 2
        peak = magnitude(peak_location)
        if peak is None or peak > 1e6 * max(1.0, grid_value):
            return peak_location
        return None

    @staticmethod
    def numeric_integrate(
        expression: str,
        variable: str,
        lower_bound: Number,
        upper_bound: Number,
        method: str = "simpson",
        steps: int = 200,
    ) -> Dict[str, Number]:
        """Numerically integrate an expression over a finite interval.

        This is intended for fast approximation workflows where symbolic
        integration is unnecessary or unavailable.
        """
        from utils.numeric_integration import integrate as integrate_numeric

        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("expression must be a non-empty string")
        if not isinstance(variable, str) or not variable.strip():
            raise ValueError("variable must be a non-empty string")
        expression, variable = MathUtils._normalize_symbols(expression), MathUtils._normalize_variable(variable)

        lower = float(lower_bound)
        upper = float(upper_bound)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("lower_bound and upper_bound must be finite")
        if lower >= upper:
            raise ValueError("lower_bound must be less than upper_bound")

        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if steps > MathUtils.MAX_NUMERIC_INTEGRATION_STEPS:
            raise ValueError(f"steps cannot exceed {MathUtils.MAX_NUMERIC_INTEGRATION_STEPS}")

        expr = window.nerdamer(expression)

        def eval_fn(x: float) -> float:
            return float(expr.sub(variable, x).evaluate().text())

        result = integrate_numeric(
            eval_fn=eval_fn,
            lower_bound=lower,
            upper_bound=upper,
            method=method,
            steps=steps,
        )
        payload: Dict[str, Any] = {
            "value": result["value"],
            "error_estimate": result["error_estimate"],
            "steps": result["steps"],
        }
        if "warning" in result:
            payload["warning"] = result["warning"]
        return payload

    @staticmethod
    def simplify(expression: str) -> str:
        """Simplify a mathematical expression to its simplest form.

        Uses Nerdamer symbolic computation for algebraic simplification.
        Combines like terms, factors, and reduces expressions.

        Args:
            expression (str): Mathematical expression to simplify

        Returns:
            str: Simplified expression as string or error message
        """
        expression = MathUtils._normalize_symbols(expression)
        try:
            result = MathUtils._guarded_nerdamer_text(f"simplify({expression})", MathUtils.NERDAMER_TOOL_MAX_MS)
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"
        return MathUtils._took_too_long_message(f"simplify({expression})") if result is None else result

    @staticmethod
    def expand(expression: str) -> str:
        """Expand a mathematical expression by distributing operations.

        Uses Nerdamer symbolic computation for algebraic expansion.
        Expands products, powers, and nested expressions.

        Args:
            expression (str): Mathematical expression to expand

        Returns:
            str: Expanded expression as string or error message
        """
        expression = MathUtils._normalize_symbols(expression)
        try:
            result = MathUtils._guarded_nerdamer_text(f"expand({expression})", MathUtils.NERDAMER_TOOL_MAX_MS)
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"
        return MathUtils._took_too_long_message(f"expand({expression})") if result is None else result

    @staticmethod
    def factor(expression: str) -> str:
        """Factor a mathematical expression into its factored form.

        Uses Nerdamer symbolic computation for algebraic factorization.
        Factors polynomials and extracts common factors.

        Args:
            expression (str): Mathematical expression to factor

        Returns:
            str: Factored expression as string or error message
        """
        expression = MathUtils._normalize_symbols(expression)
        try:
            result = MathUtils._guarded_nerdamer_text(f"factor({expression})", MathUtils.NERDAMER_TOOL_MAX_MS)
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"
        return MathUtils._took_too_long_message(f"factor({expression})") if result is None else result

    @staticmethod
    def get_equation_type(equation: str) -> str:
        import re

        try:
            # Preprocess the equation by expanding it to eliminate parentheses
            expanded_equation = MathUtils.expand(equation)
            # Remove whitespaces for easier processing
            expanded_equation = expanded_equation.replace(" ", "")

            # Split into left and right sides if equation contains =
            if "=" in expanded_equation:
                left, right = expanded_equation.split("=")
                # If one side is just 'y', use the other side for analysis
                if left.strip() == "y":
                    expanded_equation = right
                elif right.strip() == "y":
                    expanded_equation = left

            # Check for higher order equations (power >= 5)
            # Pattern: x^5, y^6, z^10, etc.
            # Matches: 'x^5', 'y^9', 'x^10', 'y^123'
            # Does not match: 'x^2', 'x^3', 'x^4'
            higher_order_match = re.search(r"\b[a-zA-Z]\^([5-9]|\d{2,})\b", expanded_equation)
            if higher_order_match:
                power = higher_order_match.group(1)
                return f"Order {power}"

            # Check for multiple variables
            # Pattern: any letters a-z or A-Z
            # Matches: 'x', 'y', 'X', 'Y'
            set(re.findall(r"[a-zA-Z]", expanded_equation))

            # Check for trigonometric equations
            # Pattern: trig function followed by parentheses and content
            # Matches: 'sin(x)', 'cos(2x)', 'tan(x+y)'
            # Does not match: 'sin', 'cos x', 'tan[x]'
            trigonometric_match = re.search(r"\b(sin|cos|tan|csc|sec|cot)\s*\(([^)]+)\)", expanded_equation)
            if trigonometric_match:
                return "Trigonometric"

            # Check for non-polynomial terms
            # Pattern: a power that is not a plain integer, or a division by a variable/group
            # Matches: 'x^(-1)' (expanded 1/x), 'e^x', '2^x', 'x^0.5', '1/(x+1)'
            # Does not match: 'x^2', '(1/2)*x', 'x/2'
            if re.search(r"\^(?!\d+(?![\d.]))|/\s*[(a-zA-Z]", expanded_equation):
                return "Other Non-linear"

            # Check for non-linear terms with multiple variables
            # Pattern: letter followed optionally by * followed by letter
            # Matches: 'xy', 'x*y', 'x y', 'yx'
            # Does not match: 'x+y', 'x-y'
            # Note: Only check for variable products, not just multiple variables
            # (linear equations like x + y = 4 should not be flagged as non-linear)
            if re.search(r"[a-zA-Z]\s*[*]?\s*[a-zA-Z]", expanded_equation):
                return "Other Non-linear"

            # Check for quartic equations
            # Pattern: letter followed by ^4
            # Matches: 'x^4', 'y^4'
            # Does not match: 'x^2', 'x^5', 'x4'
            quartic_match = re.search(r"\b[a-zA-Z]\^4\b", expanded_equation)
            if quartic_match:
                return "Quartic"

            # Check for cubic equations
            # Pattern: letter followed by ^3
            # Matches: 'x^3', 'y^3'
            # Does not match: 'x^2', 'x^4', 'x3'
            cubic_match = re.search(r"\b[a-zA-Z]\^3\b", expanded_equation)
            if cubic_match:
                return "Cubic"

            # Check for quadratic equations
            # Pattern: letter followed by ^2
            # Matches: 'x^2', 'y^2'
            # Does not match: 'x^3', 'x2', 'x^'
            quadratic_match = re.search(r"\b[a-zA-Z]\^2\b", expanded_equation)
            if quadratic_match:
                return "Quadratic"

            # Check for linear equations
            # Pattern: single letter
            # Matches: 'x', 'y' (when not part of another term)
            # Does not match: 'x^2', 'xy', '2'
            linear_match = re.search(r"\b[a-zA-Z]\b", expanded_equation)
            if linear_match:
                return "Linear"

            return "Unknown"

        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def determine_max_number_of_solutions(equations: Sequence[str]) -> int:
        try:
            if not equations:  # Checking for an empty list of equations
                return 0  # Indicates no solutions can be determined without equations

            if len(equations) < 2:
                return 0  # Need at least two equations to find intersections

            # Analyze the types of equations
            equation_types = [MathUtils.get_equation_type(eq) for eq in equations]

            # Create a dictionary mapping equation types to their degrees
            type_to_degree = {"Linear": 1, "Quadratic": 2, "Cubic": 3, "Quartic": 4}

            # If any equation type is unknown, trigonometric, or contains 'Error'
            if any(t in ["Unknown", "Trigonometric"] or "Error" in t for t in equation_types):
                return 0  # Cannot determine solution count for these types

            # If any equation type is "Other Non-linear"
            if any(t == "Other Non-linear" for t in equation_types):
                return 0  # Cannot determine solution count for general non-linear equations

            # Get the degrees of the equations if they're polynomial
            degrees = []
            for eq_type in equation_types:
                if eq_type in type_to_degree:
                    degrees.append(type_to_degree[eq_type])
                elif eq_type.startswith("Order"):
                    try:
                        # Extract the order number from strings like "Order 5"
                        degree = int(eq_type.split()[1])
                        degrees.append(degree)
                    except (IndexError, ValueError):
                        return 0  # If we can't parse the order, return 0

            if len(degrees) != 2:
                return 0  # We need exactly two polynomial equations

            # The maximum number of intersections is the product of the degrees
            return degrees[0] * degrees[1]

        except Exception as e:
            print(f"Error in determine_max_number_of_solutions: {e}")
            return 0

    @staticmethod
    def solve(equation: str, variable: str) -> str:
        """Solve an equation for a specific variable.

        Uses Nerdamer symbolic computation for equation solving.
        Supports linear, quadratic, polynomial, and transcendental equations.

        Args:
            equation (str): Mathematical equation (e.g., "x^2 + 2*x - 3 = 0")
            variable (str): Variable to solve for (e.g., "x")

        Returns:
            str: JSON string of solutions or error message
        """
        equation, variable = MathUtils._normalize_symbols(equation), MathUtils._normalize_variable(variable)
        try:
            raw_solutions = MathUtils._guarded_nerdamer_text(
                f"solve({equation}, {variable})", MathUtils.NERDAMER_TOOL_MAX_MS
            )
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"
        if raw_solutions is None:
            return MathUtils._took_too_long_message(f"the solutions of {equation} for {variable}")
        return MathUtils._drop_invalid_roots(raw_solutions, equation, variable)

    @staticmethod
    def _drop_invalid_roots(raw_solutions: str, equation: str, variable: str) -> str:
        """Remove roots that clearly fail the equation when substituted back numerically.

        Roots that cannot be checked (e.g. symbolic parameters) are kept, and the
        original text is returned unchanged when every root checks out.
        """
        if not (raw_solutions.startswith("[") and raw_solutions.endswith("]")):
            return raw_solutions
        roots = MathUtils._split_top_level_commas(raw_solutions[1:-1])
        sides = equation.split("=")
        if not roots or len(sides) > 2:
            return raw_solutions
        kept = []
        for root in roots:
            try:
                scope = {variable: window.math.evaluate(root)}
            except Exception:
                kept.append(root)
                continue
            if MathUtils._equation_holds(sides, scope) is not False:
                kept.append(root)
        if len(kept) == len(roots):
            return raw_solutions
        return "[" + ",".join(kept) + "]"

    @staticmethod
    def _equation_holds(sides: Sequence[str], scope: Dict[str, Any]) -> Optional[bool]:
        """Check lhs = rhs (or expression = 0) numerically with math.js.

        The tolerance also scales with how steeply the residual changes with each variable,
        so a root that is accurate to rounding error still passes when the equation has large
        coefficients (e.g. the small root of x^2 - 2000000*x + 1 = 0).
        Returns None when the equation cannot be evaluated for the given scope.
        """

        def residual_at(values: Dict[str, Any]) -> Any:
            lhs = window.math.evaluate(sides[0], values)
            rhs = window.math.evaluate(sides[1], values) if len(sides) == 2 else 0
            return window.math.subtract(lhs, rhs)

        try:
            lhs = window.math.evaluate(sides[0], scope)
            rhs = window.math.evaluate(sides[1], scope) if len(sides) == 2 else 0
            residual = float(window.math.abs(window.math.subtract(lhs, rhs)))
            scale = max(1.0, float(window.math.abs(lhs)), float(window.math.abs(rhs)))
        except Exception:
            return None
        if not math.isfinite(residual) or not math.isfinite(scale):
            return None
        for name, value in scope.items():
            try:
                size = max(1.0, float(window.math.abs(value)))
                step = 1e-7 * size
                forward = residual_at({**scope, name: window.math.subtract(value, -step)})
                backward = residual_at({**scope, name: window.math.subtract(value, step)})
                slope = float(window.math.abs(window.math.subtract(forward, backward))) / (2 * step)
            except Exception:
                continue
            if math.isfinite(slope):
                scale = max(scale, size * slope)
        return residual <= 1e-6 * scale

    @staticmethod
    def _split_top_level_commas(text: str) -> List[str]:
        """Split text on commas that are not nested inside brackets."""
        parts: List[str] = []
        current: List[str] = []
        depth = 0
        for char in text:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            if char == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        parts.append("".join(current).strip())
        return [part for part in parts if part]

    @staticmethod
    def _to_real_float(value_text: str) -> Optional[float]:
        """Convert a numeric root string to a float, or None if it is not a finite real number."""
        try:
            value = float(value_text)
        except Exception:
            try:
                result = window.math.evaluate(value_text)
                if isinstance(result, (int, float)):
                    value = float(result)
                else:
                    real, imag = float(result.re), float(result.im)
                    if abs(imag) > 1e-9 * max(1.0, abs(real)):
                        return None
                    value = real
            except Exception:
                return None
        return value if math.isfinite(value) else None

    @staticmethod
    def _numeric_real_roots(expression: str, variable: str) -> List[float]:
        """Solve expression = 0 with nerdamer and return the distinct real roots as floats.

        Raises ValueError when nerdamer fails or runs past NERDAMER_MAX_MS.
        """
        raw_roots = MathUtils._guarded_nerdamer_text(
            f"solve({expression}, {variable})", MathUtils.NERDAMER_MAX_MS, decimals=True
        )
        if raw_roots is None:
            raise ValueError(f"solve({expression}, {variable}) took too long")
        if not (raw_roots.startswith("[") and raw_roots.endswith("]")):
            return []
        roots: List[float] = []
        for root_text in MathUtils._split_top_level_commas(raw_roots[1:-1]):
            root = MathUtils._to_real_float(root_text)
            if root is not None and not any(abs(root - r) <= 1e-9 * max(1.0, abs(r)) for r in roots):
                roots.append(root)
        return roots

    @staticmethod
    def solve_linear_system(equations: Sequence[str]) -> str:
        try:
            if len(equations) == 0:
                raise ValueError("The system of equations must contain at least 1 equation.")

            print(f"Attempting to solve a system of linear equations: {equations}")
            # Use nerdamer to solve the system of equations
            solutions = window.nerdamer.solveEquations(equations)  # returns [['x', 3], ['y', 1]]
            print(f"Solutions: {solutions}")
            # Prepare the solution dictionary
            solution_dict = {sol[0]: sol[1] for sol in solutions}
            # Convert solution_dict to string format
            solution_strings = [f"{k} = {v}" for k, v in solution_dict.items()]
            return ", ".join(solution_strings)
        except ValueError as ve:
            raise ve
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def solve_linear_quadratic_system(equations: Sequence[str]) -> str:
        try:
            from ast import literal_eval

            if len(equations) in [0, 1] or len(equations) > 2:
                raise ValueError("The system of equations must contain at most 2 equations.")

            print(f"Attempting to solve a system of linear and quadratic equations: {equations}")

            # The coefficient fast path below only works when both equations are 'y = f(x)'
            if any(MathUtils._explicit_y_expression(eq) is None for eq in equations):
                substitution_solutions = MathUtils._solve_by_substitution(equations)
                if not substitution_solutions:
                    return MathUtils.solve_numeric(equations)
                if len(substitution_solutions) == 1:
                    return f"x = {substitution_solutions[0][0]}, y = {substitution_solutions[0][1]}"
                indexed = [f"x{i} = {x}, y{i} = {y}" for i, (x, y) in enumerate(substitution_solutions, start=1)]
                return ", ".join(indexed)

            from expression_validator import ExpressionValidator

            eq1 = MathUtils.expand(equations[0])
            eq1 = ExpressionValidator.fix_math_expression(eq1, python_compatible=False)
            # Split by '=' to separate the left and right sides of the equation and take the side containing the variable
            eq1 = eq1.split("=")[0] if "x" in eq1.split("=")[0] else eq1.split("=")[1]

            eq2 = MathUtils.expand(equations[1])
            eq2 = ExpressionValidator.fix_math_expression(eq2, python_compatible=False)
            # Split by '=' to separate the left and right sides of the equation and take the side containing the variable
            eq2 = eq2.split("=")[0] if "x" in eq2.split("=")[0] else eq2.split("=")[1]

            linear, quadratic = (eq1, eq2) if "^2" in eq2 else (eq2, eq1)

            system_eq = f"{quadratic} - ({linear})"
            system_eq = MathUtils.expand(system_eq)

            # Extract m, n = coefficients of the linear equation (assuming y = mx + n form)
            lin_coeffs_str = window.nerdamer.coeffs(
                linear, "x"
            ).text()  # The coefficients are placed in the index of their power. So constants are in the 0th place, x^2 would be in the 2nd place, etc.
            lin_coeffs = literal_eval(lin_coeffs_str)
            m, n = lin_coeffs[1], lin_coeffs[0]

            # Extract a, b, c = coefficients of system equation
            quadratic_coeffs_str = window.nerdamer.coeffs(system_eq, "x").text()
            quadratic_coeffs = literal_eval(quadratic_coeffs_str)
            a, b, c = quadratic_coeffs[2], quadratic_coeffs[1], quadratic_coeffs[0]

            # Solve the quadratic equation of the system
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                raise ValueError(f"No real solution for the quadratic equation {quadratic}.")

            x1 = (-b + math.sqrt(discriminant)) / (2 * a)
            x2 = (-b - math.sqrt(discriminant)) / (2 * a)

            # If the linear equation is not directly in terms of y, adjust accordingly
            y1 = m * x1 + n
            y2 = m * x2 + n

            # Format solutions
            solutions = []
            if discriminant > 0:  # Two solutions
                solutions.append(("x1", x1))
                solutions.append(("y1", y1))
                solutions.append(("x2", x2))
                solutions.append(("y2", y2))
            elif discriminant == 0:  # One solution
                solutions.append(("x", x1))
                solutions.append(("y", y1))

            # Prepare the solution dictionary (assuming a single solution format for simplification)
            solution_dict = {sol[0]: sol[1] for sol in solutions}

            # Convert solution_dict to string format
            solution_strings = [f"{k} = {v}" for k, v in solution_dict.items()]
            return ", ".join(solution_strings)

        except ValueError as ve:
            raise ve
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

    @staticmethod
    def solve_quadratic_system(equations: Sequence[str]) -> str:
        try:
            if len(equations) != 2:
                raise ValueError("The system must contain exactly 2 quadratic equations.")

            print(f"Attempting to solve a system of quadratic equations: {equations}")
            # Substitute an explicit 'y = f(x)' equation into the other one and solve for x
            solutions = MathUtils._solve_by_substitution(equations)
            if not solutions:
                # Neither equation is explicit in y, or no real root was found symbolically
                print("Falling back to numeric solver for quadratic system")
                return MathUtils.solve_numeric(equations)

            solution_strings = [f"(x = {x}, y = {y})" for x, y in solutions]
            print(f"Solutions found: {solution_strings}")
            return ", ".join(solution_strings)

        except ValueError as ve:
            raise ve
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

    _Y_TOKEN_PATTERN = r"(?<![A-Za-z_])y(?![A-Za-z_])"

    @staticmethod
    def _explicit_y_expression(equation: str) -> Optional[str]:
        """Return f(x) when the equation is literally 'y = f(x)' or 'f(x) = y', else None."""
        import re

        sides = [side.strip() for side in equation.split("=")]
        if len(sides) != 2:
            return None
        for side, other in ((sides[0], sides[1]), (sides[1], sides[0])):
            if side == "y" and other and not re.search(MathUtils._Y_TOKEN_PATTERN, other):
                return other
        return None

    @staticmethod
    def _solve_by_substitution(equations: Sequence[str]) -> Optional[List[Tuple[float, float]]]:
        """Solve two equations in x and y by substituting an explicit 'y = f(x)' into the other.

        Returns the real (x, y) solutions that satisfy both equations, or None when
        neither equation is explicit in y or the reduced equation cannot be solved.
        """
        import re

        explicit_forms = [MathUtils._explicit_y_expression(eq) for eq in equations]
        index = next((i for i, form in enumerate(explicit_forms) if form is not None), None)
        if index is None:
            return None
        y_expression = explicit_forms[index]
        if y_expression is None:
            return None
        other_sides = equations[1 - index].split("=")
        if len(other_sides) > 2:
            return None
        if len(other_sides) == 1:
            other_sides.append("0")
        substituted = [re.sub(MathUtils._Y_TOKEN_PATTERN, f"({y_expression})", side) for side in other_sides]
        try:
            x_roots = MathUtils._numeric_real_roots(f"({substituted[0]}) - ({substituted[1]})", "x")
        except Exception as e:
            print(f"Substitution solve failed: {e}")
            return None

        equation_sides = [eq.split("=") for eq in equations]
        solutions: List[Tuple[float, float]] = []
        for x_value in x_roots:
            try:
                y_value = window.math.evaluate(y_expression, {"x": x_value})
            except Exception:
                continue
            if not isinstance(y_value, (int, float)) or not math.isfinite(y_value):
                continue
            y_value = float(y_value)
            scope = {"x": x_value, "y": y_value}
            if all(MathUtils._equation_holds(sides, scope) is not False for sides in equation_sides):
                solutions.append((x_value, y_value))
        return solutions

    @staticmethod
    def solve_system_of_equations(equations: Sequence[str]) -> str:
        if not isinstance(equations, list) or not equations or not all(isinstance(eq, str) for eq in equations):
            raise ValueError("Invalid input for equations. Expected a list of equations.")
        try:
            # Split single equation strings into two equations
            if len(equations) == 1 and "x" in equations[0] and "=" in equations[0]:
                eq1, eq2 = equations[0].split("=")
                eq1 += "= y"
                eq2 += "= y"
                equations = [eq1, eq2]

            from expression_validator import ExpressionValidator

            equations = [ExpressionValidator.fix_math_expression(eq, python_compatible=False) for eq in equations]

            max_solutions_of_system = MathUtils.determine_max_number_of_solutions(equations)
            print(f"Max solutions for system of equations {equations}: {max_solutions_of_system}")

            if max_solutions_of_system == 4:
                # Solve two quadratic equations
                print("Solving two quadratic equations")
                solutions = MathUtils.solve_quadratic_system(equations)
                return solutions
            elif max_solutions_of_system == 2:
                # Solve linear and quadratic equations
                print("Solving linear and quadratic equations")
                solutions = MathUtils.solve_linear_quadratic_system(equations)
                return solutions
            elif max_solutions_of_system == 1:
                # Solve two linear equations
                print("Solving two linear equations")
                solutions = MathUtils.solve_linear_system(equations)
                return solutions
            else:
                # Check if any equations are transcendental or non-polynomial
                equation_types = [MathUtils.get_equation_type(eq) for eq in equations]
                if any(t in ["Trigonometric", "Unknown", "Other Non-linear"] or "Error" in t for t in equation_types):
                    print("Falling back to numeric solver for transcendental/non-polynomial system")
                    return MathUtils.solve_numeric(equations)

                # Two x/y equations with an explicit 'y = f(x)': substitute to find every real intersection
                if len(equations) == 2:
                    substitution_solutions = MathUtils._solve_by_substitution(equations)
                    if substitution_solutions:
                        if len(substitution_solutions) == 1:
                            x_value, y_value = substitution_solutions[0]
                            return f"x = {x_value}, y = {y_value}"
                        indexed = [
                            f"x{i} = {x_value}, y{i} = {y_value}"
                            for i, (x_value, y_value) in enumerate(substitution_solutions, start=1)
                        ]
                        return ", ".join(indexed)

                # Try the nerdamer library solver, fall back to numeric on failure
                print("Solving using nerdamer, returning first solution found")
                try:
                    solutions = window.nerdamer.solveEquations(equations)
                    solution_strings = [f"{solution[0]} = {solution[1]}" for solution in solutions]
                    return ", ".join(solution_strings)
                except Exception as e:
                    print(f"Nerdamer failed ({e}), falling back to numeric solver")
                    return MathUtils.solve_numeric(equations)
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def solve_numeric(
        equations: Sequence[str],
        variables: Optional[Sequence[str]] = None,
        initial_guesses: Optional[Sequence[Sequence[float]]] = None,
        tolerance: float = 1e-10,
        max_iterations: int = 50,
    ) -> str:
        """Numerically solve a system of equations using multi-start Newton-Raphson.

        Use for transcendental, mixed nonlinear, or systems that can't be solved
        symbolically. Supports any number of variables.

        Args:
            equations: List of equation strings. Use '=' for equations.
                If no '=' is present, the expression is assumed equal to 0.
            variables: Optional list of variable names (auto-detected if not provided).
            initial_guesses: Optional list of starting point vectors.
            tolerance: Convergence tolerance for residuals.
            max_iterations: Maximum Newton-Raphson iterations per starting point.

        Returns:
            JSON string with solutions, variables, and method information.
        """
        from numeric_solver import solve_numeric as _solve_numeric

        if isinstance(equations, list):
            equations = [MathUtils._normalize_symbols(equation) for equation in equations]
        if isinstance(variables, list):
            variables = [MathUtils._normalize_variable(variable) for variable in variables]
        return str(_solve_numeric(equations, variables, initial_guesses, tolerance, max_iterations))

    @staticmethod
    def random(min_value: Number = 0, max_value: Number = 1) -> float:
        return random.uniform(min_value, max_value)

    @staticmethod
    def round(value: Number, ndigits: int = 0) -> float:
        return round(value, ndigits)

    @staticmethod
    def gcd(*values: Number) -> int:
        ints = [int(v) for v in values]
        return math.gcd(*ints)

    @staticmethod
    def lcm(*values: Number) -> int:
        ints = [int(v) for v in values]
        return math.lcm(*ints)

    # ========== Number Theory Functions ==========

    @staticmethod
    def is_prime(n: Number) -> bool:
        """Check if a number is prime.

        Args:
            n: Integer to check for primality

        Returns:
            bool: True if n is prime, False otherwise

        Raises:
            ValueError: If n is negative
            TypeError: If n is not an integer
        """
        if not isinstance(n, (int, float)) or (isinstance(n, float) and not n.is_integer()):
            raise TypeError("is_prime requires an integer argument")
        n = int(n)
        if n < 0:
            raise ValueError("is_prime requires a non-negative integer")
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def prime_factors(n: Number) -> List[int]:
        """Return the prime factorization of n with multiplicity.

        Args:
            n: Positive integer to factorize

        Returns:
            list[int]: List of prime factors (with repetition for multiplicity)

        Raises:
            ValueError: If n is less than 1
            TypeError: If n is not an integer

        Examples:
            prime_factors(12) -> [2, 2, 3]
            prime_factors(1) -> []
        """
        if not isinstance(n, (int, float)) or (isinstance(n, float) and not n.is_integer()):
            raise TypeError("prime_factors requires an integer argument")
        n = int(n)
        if n < 1:
            raise ValueError("prime_factors requires a positive integer")
        if n == 1:
            return []
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    @staticmethod
    def mod_pow(base: Number, exp: Number, mod: Number) -> int:
        """Compute modular exponentiation: (base^exp) mod mod.

        Args:
            base: Base of the exponentiation
            exp: Exponent (must be non-negative)
            mod: Modulus (must be positive)

        Returns:
            int: Result of (base^exp) mod mod

        Raises:
            ValueError: If exp is negative or mod is not positive
            TypeError: If arguments are not integers
        """
        for name, val in [("base", base), ("exp", exp), ("mod", mod)]:
            if not isinstance(val, (int, float)) or (isinstance(val, float) and not val.is_integer()):
                raise TypeError(f"mod_pow requires integer arguments, got non-integer for {name}")
        base, exp, mod = int(base), int(exp), int(mod)
        if exp < 0:
            raise ValueError("mod_pow requires a non-negative exponent")
        if mod <= 0:
            raise ValueError("mod_pow requires a positive modulus")
        return pow(base, exp, mod)

    @staticmethod
    def mod_inverse(a: Number, mod: Number) -> int:
        """Compute the modular multiplicative inverse of a modulo mod.

        Finds x such that (a * x) % mod == 1.

        Args:
            a: Integer to find the inverse of
            mod: Modulus (must be positive)

        Returns:
            int: Modular inverse of a modulo mod

        Raises:
            ValueError: If inverse does not exist (gcd(a, mod) != 1) or mod is not positive
            TypeError: If arguments are not integers
        """
        for name, val in [("a", a), ("mod", mod)]:
            if not isinstance(val, (int, float)) or (isinstance(val, float) and not val.is_integer()):
                raise TypeError(f"mod_inverse requires integer arguments, got non-integer for {name}")
        a, mod = int(a), int(mod)
        if mod <= 0:
            raise ValueError("mod_inverse requires a positive modulus")

        # Extended Euclidean Algorithm
        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            if a == 0:
                return b, 0, 1
            gcd_val, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd_val, x, y

        gcd_val, x, _ = extended_gcd(a % mod, mod)
        if gcd_val != 1:
            raise ValueError(f"Modular inverse does not exist: gcd({a}, {mod}) = {gcd_val} != 1")
        return x % mod

    @staticmethod
    def next_prime(n: Number) -> int:
        """Find the smallest prime number >= n.

        Args:
            n: Starting value

        Returns:
            int: Smallest prime >= n

        Raises:
            TypeError: If n is not an integer
        """
        if not isinstance(n, (int, float)) or (isinstance(n, float) and not n.is_integer()):
            raise TypeError("next_prime requires an integer argument")
        n = int(n)
        if n <= 2:
            return 2
        candidate = n if n % 2 != 0 else n + 1
        while not MathUtils.is_prime(candidate):
            candidate += 2
        return candidate

    @staticmethod
    def prev_prime(n: Number) -> int:
        """Find the largest prime number <= n.

        Args:
            n: Starting value

        Returns:
            int: Largest prime <= n

        Raises:
            ValueError: If n < 2 (no prime exists <= n)
            TypeError: If n is not an integer
        """
        if not isinstance(n, (int, float)) or (isinstance(n, float) and not n.is_integer()):
            raise TypeError("prev_prime requires an integer argument")
        n = int(n)
        if n < 2:
            raise ValueError("No prime exists less than or equal to n (n must be >= 2)")
        if n == 2:
            return 2
        candidate = n if n % 2 != 0 else n - 1
        while candidate >= 2 and not MathUtils.is_prime(candidate):
            candidate -= 2
        if candidate < 2:
            return 2
        return candidate

    @staticmethod
    def totient(n: Number) -> int:
        """Compute Euler's totient function φ(n).

        Returns the count of integers from 1 to n that are coprime with n.

        Args:
            n: Positive integer

        Returns:
            int: φ(n), the count of coprimes

        Raises:
            ValueError: If n < 1
            TypeError: If n is not an integer
        """
        if not isinstance(n, (int, float)) or (isinstance(n, float) and not n.is_integer()):
            raise TypeError("totient requires an integer argument")
        n = int(n)
        if n < 1:
            raise ValueError("totient requires a positive integer")
        if n == 1:
            return 1
        # Use the formula: φ(n) = n * Π(1 - 1/p) for each prime p dividing n
        result = n
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
            p += 1
        if temp > 1:
            result -= result // temp
        return result

    @staticmethod
    def divisors(n: Number) -> List[int]:
        """Return all positive divisors of n in sorted order.

        Args:
            n: Positive integer

        Returns:
            list[int]: Sorted list of all positive divisors

        Raises:
            ValueError: If n < 1
            TypeError: If n is not an integer
        """
        if not isinstance(n, (int, float)) or (isinstance(n, float) and not n.is_integer()):
            raise TypeError("divisors requires an integer argument")
        n = int(n)
        if n < 1:
            raise ValueError("divisors requires a positive integer")
        result = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                result.append(i)
                if i != n // i:
                    result.append(n // i)
            i += 1
        return sorted(result)

    # ========== Sequence and Series Functions ==========

    @staticmethod
    def summation(expression: str, variable: str, start: int, end: int) -> str:
        """Compute sum of expression for integer values from start to end.

        Calculates Σ expression for variable = start to end.

        Args:
            expression: Mathematical expression containing the variable
            variable: Variable name to substitute (e.g., "n")
            start: Starting integer value (inclusive)
            end: Ending integer value (inclusive)

        Returns:
            str: Sum result as string

        Raises:
            ValueError: If start or end are not valid integers
        """
        start = int(start)
        end = int(end)
        if start > end:
            return "0"
        term_count = end - start + 1
        if term_count > MathUtils.MAX_SERIES_TERMS:
            raise ValueError(f"summation supports at most {MathUtils.MAX_SERIES_TERMS} terms")
        total = 0.0
        for i in range(start, end + 1):
            value = float(window.nerdamer(expression).sub(variable, i).evaluate().text())
            total += value
        # Return integer if it's a whole number
        if total == int(total):
            return str(int(total))
        return str(total)

    @staticmethod
    def product(expression: str, variable: str, start: int, end: int) -> str:
        """Compute product of expression for integer values from start to end.

        Calculates Π expression for variable = start to end.

        Args:
            expression: Mathematical expression containing the variable
            variable: Variable name to substitute (e.g., "n")
            start: Starting integer value (inclusive)
            end: Ending integer value (inclusive)

        Returns:
            str: Product result as string

        Raises:
            ValueError: If start or end are not valid integers
        """
        start = int(start)
        end = int(end)
        if start > end:
            return "1"
        term_count = end - start + 1
        if term_count > MathUtils.MAX_SERIES_TERMS:
            raise ValueError(f"product supports at most {MathUtils.MAX_SERIES_TERMS} terms")
        total = 1.0
        for i in range(start, end + 1):
            value = float(window.nerdamer(expression).sub(variable, i).evaluate().text())
            total *= value
        # Return integer if it's a whole number
        if total == int(total):
            return str(int(total))
        return str(total)

    @staticmethod
    def arithmetic_sum(first: Number, diff: Number, n: int) -> Number:
        """Sum of arithmetic series using closed-form formula.

        Calculates S_n = n/2 * (2a + (n-1)d) where:
        - a is the first term
        - d is the common difference
        - n is the number of terms

        Args:
            first: First term of the arithmetic sequence (a)
            diff: Common difference between consecutive terms (d)
            n: Number of terms to sum

        Returns:
            Number: Sum of the arithmetic series

        Raises:
            ValueError: If n < 1
        """
        n = int(n)
        if n < 1:
            raise ValueError("Number of terms must be at least 1")
        result = n / 2 * (2 * first + (n - 1) * diff)
        # Return integer if it's a whole number
        if result == int(result):
            return int(result)
        return result

    @staticmethod
    def geometric_sum(first: Number, ratio: Number, n: int) -> Number:
        """Sum of finite geometric series using closed-form formula.

        Calculates S_n = a(1-r^n)/(1-r) for r != 1, or S_n = a*n for r = 1
        where:
        - a is the first term
        - r is the common ratio
        - n is the number of terms

        Args:
            first: First term of the geometric sequence (a)
            ratio: Common ratio between consecutive terms (r)
            n: Number of terms to sum

        Returns:
            Number: Sum of the geometric series

        Raises:
            ValueError: If n < 1
        """
        n = int(n)
        if n < 1:
            raise ValueError("Number of terms must be at least 1")
        if ratio == 1:
            return first * n
        result = first * (1 - ratio**n) / (1 - ratio)
        # Return integer if it's a whole number
        if result == int(result):
            return int(result)
        return result

    @staticmethod
    def geometric_sum_infinite(first: Number, ratio: Number) -> Number:
        """Sum of infinite geometric series.

        Calculates S = a/(1-r) where |r| < 1
        where:
        - a is the first term
        - r is the common ratio

        Args:
            first: First term of the geometric sequence (a)
            ratio: Common ratio between consecutive terms (r), must satisfy |r| < 1

        Returns:
            Number: Sum of the infinite geometric series

        Raises:
            ValueError: If |ratio| >= 1 (series diverges)
        """
        if abs(ratio) >= 1:
            raise ValueError("Infinite geometric series diverges when |ratio| >= 1")
        result = first / (1 - ratio)
        # Return integer if it's a whole number
        if result == int(result):
            return int(result)
        return result

    @staticmethod
    def ratio_test(expression: str, n_var: str) -> str:
        """Apply the ratio test for series convergence.

        Calculates L = lim |a_{n+1}/a_n| as n -> infinity
        - If L < 1: series converges absolutely
        - If L > 1 or L = infinity: series diverges
        - If L = 1: test is inconclusive

        Args:
            expression: Expression for the nth term a_n (e.g., "1/n!")
            n_var: Variable name representing n (e.g., "n")

        Returns:
            str: "Converges", "Diverges", or "Inconclusive" with the limit value
        """
        try:
            # Create a_{n+1} by substituting n -> n+1
            next_term = str(window.nerdamer(expression).sub(n_var, f"({n_var}+1)").text())
            # L = lim |a_{n+1}/a_n|
            ratio_expr = f"({next_term})/({expression})"
            return MathUtils._classify_series_limit(ratio_expr, f"abs({ratio_expr})", n_var)
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def root_test(expression: str, n_var: str) -> str:
        """Apply the root test for series convergence.

        Calculates L = lim |a_n|^{1/n} as n -> infinity
        - If L < 1: series converges absolutely
        - If L > 1 or L = infinity: series diverges
        - If L = 1: test is inconclusive

        Args:
            expression: Expression for the nth term a_n (e.g., "(1/2)^n")
            n_var: Variable name representing n (e.g., "n")

        Returns:
            str: "Converges", "Diverges", or "Inconclusive" with the limit value
        """
        try:
            # L = lim |a_n|^{1/n}
            root_expr = f"({expression})^(1/{n_var})"
            return MathUtils._classify_series_limit(root_expr, f"(abs({expression}))^(1/{n_var})", n_var)
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def _classify_series_limit(expression: str, abs_expression: str, n_var: str) -> str:
        """Classify L = |lim expression| as n -> infinity for the ratio and root tests.

        nerdamer's limit() mishandles abs() at infinity (e.g. lim |2^n|^(1/n) gives 1), so the
        limit of ``expression`` is taken first and its absolute value used; the limit of
        ``abs_expression`` (already wrapped in abs) is the fallback when that is not numeric.
        """
        try:
            # Let nerdamer simplify first (e.g. 2^(n+1)/2^n -> 2, (2^n)^(1/n) -> 2)
            expression = str(window.nerdamer(expression).text())
        except Exception:
            pass
        limit_result = MathUtils.limit(expression, n_var, "inf")
        value = MathUtils._numeric_limit_value(limit_result)
        if value is None:
            limit_result = MathUtils.limit(abs_expression, n_var, "inf")
            if "Error" in limit_result:
                return f"Error computing limit: {limit_result}"
            value = MathUtils._numeric_limit_value(limit_result)
            if value is None:
                # Limit is symbolic or could not be computed
                return f"Inconclusive (L = {limit_result})"

        L = abs(value)
        if math.isinf(L):
            return "Diverges (L = infinity)"
        L_text = f"{L:.12g}"
        if abs(L - 1) <= 1e-9:
            return f"Inconclusive (L = {L_text})"
        if L < 1:
            return f"Converges (L = {L_text})"
        return f"Diverges (L = {L_text})"

    @staticmethod
    def _numeric_limit_value(limit_result: str) -> Optional[float]:
        """Numerically evaluate a nerdamer limit result (e.g. "e^(-0.69...)", "Infinity").

        Returns None for errors, unevaluated limits, and non-numeric results.
        """
        if "Error" in limit_result or "limit" in limit_result:
            return None
        try:
            evaluated: Any = window.nerdamer(limit_result).evaluate()
            numeric_text = str(evaluated.text("decimals")).strip()
        except Exception:
            return None
        if numeric_text in ("Infinity", "+Infinity"):
            return float("inf")
        if numeric_text == "-Infinity":
            return float("-inf")
        try:
            value = float(numeric_text)
        except (ValueError, TypeError):
            return None
        return None if math.isnan(value) else value

    @staticmethod
    def p_series_test(p: Number) -> str:
        """Check if a p-series converges.

        A p-series is sum(1/n^p) for n = 1 to infinity.
        - Converges if p > 1
        - Diverges if p <= 1

        Args:
            p: The exponent in the p-series (1/n^p)

        Returns:
            str: "Converges" if p > 1, "Diverges" if p <= 1
        """
        if p > 1:
            return "Converges"
        else:
            return "Diverges"

    @staticmethod
    def permutations(n: int, k: Optional[int] = None) -> int:
        """Calculate permutations of n items optionally taken k at a time."""
        n = MathUtils._ensure_non_negative_integer(n, "n")
        if k is None:
            return math.factorial(n)
        k = MathUtils._ensure_non_negative_integer(k, "k")
        if k > n:
            raise ValueError("k must be less than or equal to n for permutations")
        if hasattr(math, "perm"):
            return math.perm(n, k)
        return math.factorial(n) // math.factorial(n - k)

    @staticmethod
    def arrangements(n: int, k: int) -> int:
        """Calculate arrangements (nPk) of n items taken k at a time."""
        return MathUtils.permutations(n, k)

    @staticmethod
    def combinations(n: int, k: int) -> int:
        """Calculate combinations (nCk) of n items taken k at a time."""
        n = MathUtils._ensure_non_negative_integer(n, "n")
        k = MathUtils._ensure_non_negative_integer(k, "k")
        if k > n:
            raise ValueError("k must be less than or equal to n for combinations")
        if hasattr(math, "comb"):
            return math.comb(n, k)
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

    @staticmethod
    def mean(values: Sequence[Number]) -> float:
        """Calculate the arithmetic mean (average) of a list of values.

        Args:
            values (list): List of numeric values

        Returns:
            float: Arithmetic mean of the values
        """
        return statistics.mean(values)

    @staticmethod
    def median(values: Sequence[Number]) -> float:
        """Calculate the median (middle value) of a list of values.

        Args:
            values (list): List of numeric values

        Returns:
            float: Median value
        """
        return statistics.median(values)

    @staticmethod
    def mode(values: Sequence[Number]) -> float:
        """Calculate the mode (most frequent value) of a list of values.

        Args:
            values (list): List of numeric values

        Returns:
            float: Most frequent value
        """
        return statistics.mode(values)

    @staticmethod
    def stdev(values: Sequence[Number]) -> float:
        """Calculate the sample standard deviation of a list of values.

        Args:
            values (list): List of numeric values

        Returns:
            float: Sample standard deviation
        """
        return statistics.stdev(values)

    @staticmethod
    def variance(values: Sequence[Number]) -> float:
        """Calculate the sample variance of a list of values.

        Args:
            values (list): List of numeric values

        Returns:
            float: Sample variance
        """
        return statistics.variance(values)

    @staticmethod
    def calculate_vertical_asymptotes(
        function_string: str,
        left_bound: Optional[Number] = None,
        right_bound: Optional[Number] = None,
    ) -> List[float]:
        """Calculate vertical asymptotes of a function within given bounds"""
        return MathUtils._vertical_asymptotes_and_discontinuities(function_string, left_bound, right_bound)[0]

    @staticmethod
    def _vertical_asymptotes_and_discontinuities(
        function_string: str,
        left_bound: Optional[Number] = None,
        right_bound: Optional[Number] = None,
    ) -> Tuple[List[float], List[float]]:
        """Return (vertical asymptotes, point discontinuities) found from the expression text.

        Log asymptotes come from the text. A zero of a denominator, or a pole of a tan() in the
        expression, is classified by sampling f beside it (_is_asymptote_at_denominator_zero):
        an asymptote if f grows on either side, otherwise a point discontinuity (a hole such as
        (x^2-1)/(x-1) at x = 1 or x/tan(x) at pi/2, a jump, or a bounded oscillation such as
        sin(1/x) at 0). Candidates closer than _PROBE_SAME_POINT (relative) are one point, so
        nerdamer's near-duplicate roots (0, 1.4e-9, 6.2e-9 for 1 - cos(x)) count once, and
        with bounds only points within them are listed.
        """
        from expression_validator import ExpressionValidator

        # Standardize the function string
        function_string = ExpressionValidator.fix_math_expression(function_string)
        log_zeros: List[float] = []
        denominator_zeros: List[float] = []

        # For logarithmic functions: where the (first) argument is zero
        for log_argument in MathUtils._function_call_arguments(function_string, "log|ln|log10|log2"):
            arguments = MathUtils._split_top_level_commas(log_argument)
            if arguments:
                log_zeros.extend(MathUtils._real_zeros_in_x(arguments[0]))

        # For rational functions: where any denominator that depends on x is zero
        for denominator in MathUtils._denominators(function_string):
            denominator_zeros.extend(MathUtils._real_zeros_in_x(denominator))

        # For tangent functions (word boundary so atan/arctan are excluded)
        left = left_bound if left_bound is not None else -1000
        right = right_bound if right_bound is not None else 1000
        tangent_poles: List[float] = []
        for tan_argument in MathUtils._function_call_arguments(function_string, "tan"):
            tangent_poles.extend(MathUtils._tangent_asymptotes(tan_argument, left, right))

        log_zeros = MathUtils._merge_close_points(log_zeros)
        denominator_zeros = MathUtils._merge_close_points(denominator_zeros)
        tangent_poles = MathUtils._merge_close_points(tangent_poles)
        singular_points = MathUtils._merge_close_points(log_zeros + denominator_zeros + tangent_poles)

        vertical_asymptotes: List[float] = list(log_zeros)
        discontinuities: List[float] = []
        evaluate = MathUtils._singularity_probe_evaluator(function_string)
        for zero in denominator_zeros:
            if evaluate is None or MathUtils._is_asymptote_at_denominator_zero(evaluate, zero, singular_points):
                vertical_asymptotes.append(zero)
            else:
                discontinuities.append(zero)
        tangent_asymptotes, tangent_holes = MathUtils._classify_tangent_poles(evaluate, tangent_poles, singular_points)
        vertical_asymptotes.extend(tangent_asymptotes)
        discontinuities.extend(tangent_holes)

        def within_bounds(x: float) -> bool:
            return (left_bound is None or x >= left_bound) and (right_bound is None or x <= right_bound)

        asymptotes = [x for x in MathUtils._merge_close_points(vertical_asymptotes) if within_bounds(x)]
        holes = [
            x
            for x in MathUtils._merge_close_points(discontinuities)
            if within_bounds(x) and MathUtils._nearest_other_distance(asymptotes, x, include_same=True) != 0.0
        ]
        return asymptotes, holes

    @staticmethod
    def _same_point(a: float, b: float) -> bool:
        """True when a and b are closer than _PROBE_SAME_POINT relative to max(1, |a|, |b|)."""
        return abs(a - b) <= MathUtils._PROBE_SAME_POINT * max(1.0, abs(a), abs(b))

    @staticmethod
    def _merge_close_points(points: List[float]) -> List[float]:
        """Sorted points with near-duplicates merged, keeping the one nearest zero (0 over 1.4e-9).

        One sort and one pass: a point joins the run of its predecessor when the two are the
        same point (_same_point), so a run of close points collapses to one.
        """
        merged: List[float] = []
        previous: Optional[float] = None
        for point in sorted(points):
            if previous is not None and MathUtils._same_point(point, previous):
                if abs(point) < abs(merged[-1]):
                    merged[-1] = point
            else:
                merged.append(point)
            previous = point
        return merged

    @staticmethod
    def _nearest_other_distance(sorted_points: List[float], x0: float, include_same: bool = False) -> float:
        """Distance from x0 to the nearest point of sorted_points that is not x0 itself.

        Points within _PROBE_SAME_POINT of x0 count as x0 and are skipped, unless include_same,
        in which case finding one returns 0.0. Returns inf when there is no other point.
        """
        import bisect

        index = bisect.bisect_left(sorted_points, x0)
        nearest = math.inf
        # Merged points are never the same point as each other, so at most one neighbour on
        # each side can be x0 itself; two on each side are enough.
        for neighbour in sorted_points[max(0, index - 2) : index + 2]:
            if MathUtils._same_point(neighbour, x0):
                if include_same:
                    return 0.0
                continue
            nearest = min(nearest, abs(neighbour - x0))
        return nearest

    @staticmethod
    def _classify_tangent_poles(
        evaluate: Optional[Callable[[float], Optional[float]]],
        poles: List[float],
        singular_points: List[float],
    ) -> Tuple[List[float], List[float]]:
        """Split tan() poles into (asymptotes, point discontinuities) by sampling f beside each one.

        A pole where f stays bounded is not an asymptote: x/tan(x) and 1/tan(x) tend to 0 at
        pi/2, and tan(x)*(x - 101*pi/2) tends to -1 at 101*pi/2 only.
        """
        if evaluate is None or not poles:
            return list(poles), []
        asymptotes: List[float] = []
        holes: List[float] = []
        for pole in poles:
            verdict = MathUtils._quick_pole_verdict(evaluate, pole, MathUtils._probe_start(pole, singular_points))
            if verdict is None:
                verdict = MathUtils._is_asymptote_at_denominator_zero(evaluate, pole, singular_points)
            (asymptotes if verdict else holes).append(pole)
        return asymptotes, holes

    # Quick verdict before the full classifier, which costs up to 56 evaluations per point
    # (thousands of tan() poles made that seconds in Brython). f is sampled at x0 +/- start/8^k
    # for k = 0, 1, 2 on a side and judged on the two differences, so adding a constant to f
    # changes nothing: the second at least _STEEP_GROWTH times the first, same sign, is growth
    # (a simple pole gives 8); at most _QUICK_SHRINK times the first on both sides is a
    # bounded, converging f (x/tan(x) gives 1/8, tan(x)*(x - x0) 1/64). Anything else, or an
    # undefined or infinite sample, goes to the full classifier.
    _STEEP_GROWTH = 4.0
    _QUICK_SHRINK = 0.25

    @staticmethod
    def _quick_pole_verdict(evaluate: Callable[[float], Optional[float]], x0: float, start: float) -> Optional[bool]:
        """True (asymptote), False (point discontinuity) or None (not clear: use the full classifier)."""
        converging_sides = 0
        for side in (1.0, -1.0):
            samples = [evaluate(x0 + side * start / 8.0**k) for k in range(3)]
            if any(sample is None or not math.isfinite(sample) for sample in samples):
                return None
            far, middle, near = cast(List[float], samples)
            first, second = middle - far, near - middle
            if first != 0 and (first > 0) == (second > 0) and abs(second) >= MathUtils._STEEP_GROWTH * abs(first):
                return True
            if abs(second) <= MathUtils._QUICK_SHRINK * abs(first):
                converging_sides += 1
        return False if converging_sides == 2 else None

    @staticmethod
    def _probe_start(x0: float, singular_points: List[float]) -> float:
        """The first sampling offset beside x0: 1e-2 * max(1, |x0|), or a tenth of the distance to the nearest other singular point."""
        scale = max(1.0, abs(x0))
        return min(MathUtils._PROBE_START * scale, MathUtils._nearest_other_distance(singular_points, x0) / 10)

    # Sampling beside a denominator zero x0: the first offset is 1e-2 * max(1, |x0|), or a tenth
    # of the distance to the nearest other singular point if smaller (points closer than
    # 1e-6 * max(1, |x0|) count as x0 itself); each next offset is 4 times smaller.
    _PROBE_START = 1e-2
    _PROBE_FACTOR = 4.0
    _PROBE_COUNT = 7
    _PROBE_SAME_POINT = 1e-6
    # A sample is float noise (cancellation) when moving x by up to 3e-6 of its offset changes f
    # by more than a tenth of the step from the previous sample; sampling stops before it.
    _PROBE_JITTER = 1e-6
    _PROBE_NOISE = 0.1
    # Successive differences shrinking to at most 0.9 of the previous one converge.
    _PROBE_SHRINK = 0.9
    # Log-like growth: ratios of successive differences rising by >= 0.005 each, to >= 0.7.
    _PROBE_SLOW_GROWTH_RATIO = 0.7
    _PROBE_SLOW_GROWTH_STEP = 0.005
    # Unbounded oscillation: the last two |f| exceed the first two by this factor.
    _PROBE_ENVELOPE = 100.0

    @staticmethod
    def _singularity_probe_evaluator(function_string: str) -> Optional[Callable[[float], Optional[float]]]:
        """Return f as a probe: a float, inf on overflow, or None where f is undefined or not real."""
        from expression_validator import ExpressionValidator

        try:
            base_function = ExpressionValidator.parse_function_string(function_string)
        except Exception:
            return None

        def evaluate(x: float) -> Optional[float]:
            try:
                value = base_function(x)
            except OverflowError:
                return float("inf")
            except Exception:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            value = float(value)
            return None if math.isnan(value) else value

        return evaluate

    @staticmethod
    def _mathjs_probe_evaluator(expression: str, variable: str = "x") -> Optional[Callable[[float], Optional[float]]]:
        """Return f compiled by math.js as a probe: a float, or None where f fails or is not real.

        math.js computes with IEEE floats, so an overflow is inf and carries through the rest of
        the expression (1/(1 + exp(-x)) is 0 where exp(-x) overflows), and inf/inf is NaN.
        Python floats raise OverflowError instead, which loses the value of the whole expression.
        """
        from expression_validator import ExpressionValidator

        try:
            # math.js calls trunc() fix()
            fixed = ExpressionValidator.fix_math_expression(str(expression)).replace("trunc(", "fix(")
            compiled = window.math.compile(fixed)
        except Exception:
            return None

        def evaluate(point: float) -> Optional[float]:
            try:
                value = compiled.evaluate({variable: point})
            except Exception:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value)

        return evaluate

    @staticmethod
    def _is_asymptote_at_denominator_zero(
        evaluate: Callable[[float], Optional[float]], x0: float, singular_points: List[float]
    ) -> bool:
        """True if the denominator zero x0 is a vertical asymptote, False for a point discontinuity.

        Each side is sampled at shrinking offsets and judged on the differences between
        successive samples, never on |f| relative to its own value, so adding a constant to
        f changes nothing. x0 is an asymptote if f grows on either side, or if f is undefined
        on both sides; otherwise (f converges, or stays bounded without converging, as
        sin(1/x) does at 0) it is a point discontinuity.
        """
        start = MathUtils._probe_start(x0, singular_points)
        verdicts = [
            MathUtils._probe_side_verdict(MathUtils._probe_side(evaluate, x0, side, start)) for side in (-1.0, 1.0)
        ]
        return "grow" in verdicts or verdicts == ["undefined", "undefined"]

    @staticmethod
    def _probe_side(
        evaluate: Callable[[float], Optional[float]], x0: float, side: float, start: float
    ) -> Optional[List[float]]:
        """Samples of f on one side of x0, nearest last; None if f is undefined there.

        Sampling stops at an infinite value (kept) or before a sample dominated by float noise.
        """
        values: List[float] = []
        for index in range(MathUtils._PROBE_COUNT):
            offset = start / MathUtils._PROBE_FACTOR**index
            value = evaluate(x0 + side * offset)
            if value is None:
                return None
            if math.isinf(value):
                values.append(value)
                break
            if values:
                step = abs(value - values[-1])
                jittered = [evaluate(x0 + side * offset * (1 + j * MathUtils._PROBE_JITTER)) for j in (1, 2, 3)]
                if any(
                    other is None or math.isinf(other) or abs(other - value) > MathUtils._PROBE_NOISE * step
                    for other in jittered
                ):
                    break
            values.append(value)
        return values

    @staticmethod
    def _probe_side_verdict(values: Optional[List[float]]) -> str:
        """Classify one side's samples as "grow", "converge", "neither" or "undefined".

        - grow: an infinite sample; or differences that do not shrink (each at least 0.9 of
          the one before) while f moves one way or |f| keeps rising (powers of any positive
          exponent, even 1/x^0.004, logs, sin(1/x)/x); or ratios of differences rising towards
          1 (log(log(1/x))); or, failing convergence, |f| rising 100-fold over the samples.
        - converge: every difference at most 0.9 of the one before (holes, jumps, and slow
          limits such as |x|^0.25). Edge: a limit approached like L + c*|h|^p with p below
          about 0.076 shrinks by less than 0.9 per 4-fold step and is read as growth.
        - neither: bounded without converging (sin(1/x)), or fewer than three clean samples.
        """
        if values is None:
            return "undefined"
        if any(math.isinf(value) for value in values):
            return "grow"
        if len(values) < 3:
            return "neither"
        differences = [later - earlier for earlier, later in zip(values, values[1:])]
        ratios = [
            (0.0 if later == 0 else float("inf")) if earlier == 0 else abs(later) / abs(earlier)
            for earlier, later in zip(differences, differences[1:])
        ]
        one_way = all(d > 0 for d in differences) or all(d < 0 for d in differences)
        rising = all(abs(later) > abs(earlier) for earlier, later in zip(values, values[1:]))
        if (one_way or rising) and all(ratio >= MathUtils._PROBE_SHRINK for ratio in ratios):
            return "grow"
        if (
            one_way
            and len(ratios) >= 2
            and ratios[-1] >= MathUtils._PROBE_SLOW_GROWTH_RATIO
            and all(later - earlier >= MathUtils._PROBE_SLOW_GROWTH_STEP for earlier, later in zip(ratios, ratios[1:]))
        ):
            return "grow"
        if all(ratio <= MathUtils._PROBE_SHRINK for ratio in ratios):
            return "converge"
        first = max(abs(value) for value in values[:2])
        if max(abs(value) for value in values[-2:]) > MathUtils._PROBE_ENVELOPE * first:
            return "grow"
        return "neither"

    _X_TOKEN_PATTERN = r"(?<![A-Za-z_])x(?![A-Za-z_])"

    @staticmethod
    def _balanced_group(text: str, open_index: int) -> Optional[str]:
        """Return the contents of the parenthesized group starting at text[open_index] == '('."""
        depth = 0
        for index in range(open_index, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    return text[open_index + 1 : index]
        return None

    @staticmethod
    def _function_call_arguments(text: str, names: str) -> List[str]:
        """Return the full (balanced) argument text of every call to one of the '|'-separated names."""
        import re

        arguments = []
        for match in re.finditer(rf"\b(?:{names})\s*\(", text):
            group = MathUtils._balanced_group(text, match.end() - 1)
            if group is not None:
                arguments.append(group)
        return arguments

    @staticmethod
    def _denominators(text: str) -> List[str]:
        """Return each denominator that depends on x: the group or token right after every '/'."""
        import re

        denominators = []
        for index, char in enumerate(text):
            if char != "/":
                continue
            rest = text[index + 1 :]
            start = index + 1 + (len(rest) - len(rest.lstrip()))
            if start < len(text) and text[start] == "(":
                denominator = MathUtils._balanced_group(text, start)
            else:
                token_match = re.match(r"[A-Za-z_][A-Za-z_0-9.]*|\d*\.?\d+", text[start:])
                denominator = token_match.group(0) if token_match else None
                if denominator is not None and text[start + len(denominator) : start + len(denominator) + 1] == "(":
                    call_arguments = MathUtils._balanced_group(text, start + len(denominator))
                    denominator = None if call_arguments is None else f"{denominator}({call_arguments})"
            if denominator and re.search(MathUtils._X_TOKEN_PATTERN, denominator):
                denominators.append(denominator)
        return denominators

    @staticmethod
    def _real_zeros_in_x(expression: str) -> List[float]:
        """Return the real zeros of an expression in x, or [] when it has none or cannot be solved."""
        import re

        if not re.search(MathUtils._X_TOKEN_PATTERN, expression):
            return []
        try:
            return MathUtils._numeric_real_roots(expression, "x")
        except Exception:
            return []

    @staticmethod
    def _tangent_asymptotes(argument: str, left: Number, right: Number) -> List[float]:
        """Return the asymptotes of tan(argument) within [left, right] for a linear argument a*x + b."""
        try:
            compiled = window.math.compile(argument)
            offset = float(compiled.evaluate({"x": 0}))
            slope = float(compiled.evaluate({"x": 1})) - offset
            for probe in (-2.5, 3.7):
                expected = offset + slope * probe
                if abs(float(compiled.evaluate({"x": probe})) - expected) > 1e-9 * max(1.0, abs(expected)):
                    return []  # non-linear argument: leave it to numeric discontinuity detection
        except Exception:
            return []
        if slope == 0 or not math.isfinite(slope) or not math.isfinite(offset):
            return []

        # Asymptotes occur where slope*x + offset = pi/2 + n*pi
        n_at_left = (slope * left + offset - math.pi / 2) / math.pi
        n_at_right = (slope * right + offset - math.pi / 2) / math.pi
        first_n = math.floor(min(n_at_left, n_at_right)) - 1
        last_n = math.ceil(max(n_at_left, n_at_right)) + 1
        asymptotes = []
        for n in range(first_n, last_n + 1):
            x = (math.pi / 2 + n * math.pi - offset) / slope
            if left <= x <= right:
                asymptotes.append(x)
        return asymptotes

    @staticmethod
    def calculate_horizontal_asymptotes(function_string: str) -> List[float]:
        """Calculate horizontal asymptotes of a function: the limit at +inf and at -inf, where finite.

        The limits are estimated numerically (_limit_at_infinity), never with nerdamer: its
        limit() never returns for abs(x)/x and similar quotients, which froze the tab on every
        draw, and it is wrong for others (floor(x)/x -> 0, sqrt(x^2+1)/x -> 0).
        """
        evaluate = MathUtils._mathjs_probe_evaluator(function_string)
        if evaluate is None:
            return []
        horizontal_asymptotes: List[float] = []
        for sign in (1.0, -1.0):
            value = MathUtils._limit_at_infinity(evaluate, sign)
            if value is not None:
                horizontal_asymptotes.append(value)
        return sorted(horizontal_asymptotes)

    # f is sampled at x = sign * 9.73 * 4^k for k < 12 (9.73 to 4.1e7). The start is not an
    # integer, so integer-periodic parts such as x - floor(x) are not sampled in phase; the
    # first four samples come before exp() overflows (x = 709.8), so quotients that become
    # inf/inf there, such as cosh(x)/sinh(x), keep four; and the largest x keeps x^2 below
    # 2^53, so cancellations such as sqrt(x^2 + x) - x stay exact enough.
    _INFINITY_PROBE_START = 9.73
    _INFINITY_PROBE_FACTOR = 4.0
    _INFINITY_PROBE_COUNT = 12
    # Convergence and Aitken extrapolation are judged on the trailing samples, so an offset or
    # a scale that keeps the first samples far from the limit ((x+500)/(x-500), atan(x/1000))
    # does not hide it.
    _SETTLED_TAIL = 5
    # Settled samples: the last four span at most 1e-4 * max(1, |value|) and at most a quarter
    # of the four before them (so a bounded oscillation such as 1 + 1e-5*sin(x) does not settle).
    _SETTLED_WINDOW = 4
    _SETTLED_SPREAD = 1e-4
    _SETTLED_SHRINK = 4.0
    # Two successive Aitken extrapolations must agree to this relative tolerance.
    _AITKEN_AGREEMENT = 1e-6
    # A settled value is never trusted beyond 12 significant digits.
    _SETTLED_PRECISION = 1e-12
    # Bisections from the last finite sample towards an infinite one (a gap 2^-40 of the first).
    _OVERFLOW_BISECTIONS = 40

    @staticmethod
    def _limit_at_infinity(evaluate: Callable[[float], Optional[float]], sign: float) -> Optional[float]:
        """Estimate lim f(x) as x -> sign * infinity from samples; None if f does not settle on a finite value.

        evaluate must use IEEE arithmetic (_mathjs_probe_evaluator), so an overflow inside f
        carries through it: 1/(1 + exp(-x)) is 0 where exp(-x) overflows. f undefined at a
        sample (not real) means no limit on that side. An infinite sample ends the sampling
        after the last finite value before the overflow (_value_before_overflow) is added, so
        exp(x-700) + 1, which reads 1 up to x = 622.7 and overflows by x = 2490.9, is judged on
        its value near 1e308 there. A NaN sample (inf/inf, e.g. cosh(x)/sinh(x) once both
        overflow) has lost the value of f: it ends the sampling, and the samples before it are
        judged.
        """
        values: List[float] = []
        previous_x = 0.0
        for index in range(MathUtils._INFINITY_PROBE_COUNT):
            x = sign * MathUtils._INFINITY_PROBE_START * MathUtils._INFINITY_PROBE_FACTOR**index
            value = evaluate(x)
            if value is None:
                return None
            if math.isnan(value):
                break
            if math.isinf(value):
                if not values:
                    return None
                before_overflow = MathUtils._value_before_overflow(evaluate, previous_x, x)
                if before_overflow is None:
                    return None
                values.append(before_overflow)
                break
            values.append(value)
            previous_x = x
        return MathUtils._settled_value(values)

    @staticmethod
    def _value_before_overflow(
        evaluate: Callable[[float], Optional[float]], finite_x: float, infinite_x: float
    ) -> Optional[float]:
        """f just before it stops being finite between finite_x and infinite_x; None where f is not real.

        An infinite f is either f itself overflowing, where f is huge just before (exp(x-700) + 1
        nears 1e308), or an overflow inside f that a later operation keeps infinite although f
        stays finite (log(1 + exp(x)) - x is log(inf) - x = inf, yet about 0 just before).
        """
        best = evaluate(finite_x)
        low, high = finite_x, infinite_x
        for _ in range(MathUtils._OVERFLOW_BISECTIONS):
            middle = (low + high) / 2
            value = evaluate(middle)
            if value is None:
                return None
            if math.isfinite(value):
                best, low = value, middle
            else:
                high = middle
        return best

    @staticmethod
    def _aitken(first: float, second: float, third: float) -> Optional[float]:
        """Aitken's delta-squared extrapolation of three successive samples; None if undefined."""
        later = third - second
        if later == 0:
            return third
        denominator = later - (second - first)
        if denominator == 0:
            return None
        return third - later * later / denominator

    @staticmethod
    def _settled_value(values: List[float]) -> Optional[float]:
        """The finite value samples (nearest last) settle on, rounded to the precision they support.

        When the trailing samples converge (MathUtils._probe_side_verdict on the last
        _SETTLED_TAIL), they are extrapolated with Aitken's delta-squared process, accepted
        when the last two extrapolations agree. Otherwise the samples settle if the last four
        barely spread and spread much less than the four before them (float noise near the
        limit, decaying oscillations such as sin(x)/x). The value is rounded to the digits its
        tolerance supports, so 1/x gives 0 and floor(x)/x gives 1.
        """
        if len(values) < 4 or not all(math.isfinite(value) for value in values):
            return None
        estimate: Optional[float] = None
        tolerance = 0.0
        # Slices start at max(0, ...): Brython returns only the last item for values[-5:] when
        # values has fewer than five items.
        tail = values[max(0, len(values) - MathUtils._SETTLED_TAIL) :]
        if MathUtils._probe_side_verdict(tail) == "converge":
            previous = MathUtils._aitken(values[-4], values[-3], values[-2])
            latest = MathUtils._aitken(values[-3], values[-2], values[-1])
            if previous is not None and latest is not None and math.isfinite(previous) and math.isfinite(latest):
                if abs(latest - previous) <= MathUtils._AITKEN_AGREEMENT * max(1.0, abs(latest)):
                    estimate, tolerance = latest, abs(latest - previous)
        if estimate is None:
            window = MathUtils._SETTLED_WINDOW
            last = values[len(values) - window :]
            earlier = values[max(0, len(values) - 2 * window) : len(values) - window]
            spread_last = max(last) - min(last)
            middle = sorted(last)[len(last) // 2]
            if spread_last > MathUtils._SETTLED_SPREAD * max(1.0, abs(middle)):
                return None
            if spread_last > 0:
                if not earlier or spread_last * MathUtils._SETTLED_SHRINK > max(earlier) - min(earlier):
                    return None
            estimate, tolerance = values[-1], spread_last
        tolerance = max(tolerance, MathUtils._SETTLED_PRECISION * max(1.0, abs(estimate)))
        digits = min(12, max(0, int(math.floor(-math.log10(tolerance)))))
        rounded = round(estimate, digits)
        return 0.0 if rounded == 0 else float(rounded)

    @staticmethod
    def calculate_asymptotes_and_discontinuities(
        function_string: str,
        left_bound: Optional[Number] = None,
        right_bound: Optional[Number] = None,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Calculate vertical and horizontal asymptotes and point discontinuities of a function"""
        from expression_validator import ExpressionValidator

        # Standardize the function string
        function_string = ExpressionValidator.fix_math_expression(function_string)
        vertical_asymptotes, denominator_discontinuities = MathUtils._vertical_asymptotes_and_discontinuities(
            function_string, left_bound, right_bound
        )
        horizontal_asymptotes = MathUtils.calculate_horizontal_asymptotes(function_string)
        point_discontinuities = MathUtils.calculate_point_discontinuities(function_string, left_bound, right_bound)
        # Denominator zeros and tan() poles that are not asymptotes (holes, jumps, bounded
        # oscillations); a set, since tan(10*x) alone has thousands
        listed = set(point_discontinuities)
        for point in denominator_discontinuities:
            within_bounds = (left_bound is None or point >= left_bound) and (
                right_bound is None or point <= right_bound
            )
            if within_bounds and point not in listed:
                listed.add(point)
                point_discontinuities.append(point)
        return vertical_asymptotes, horizontal_asymptotes, sorted(point_discontinuities)

    @staticmethod
    def calculate_point_discontinuities(
        function_string: str,
        left_bound: Optional[Number] = None,
        right_bound: Optional[Number] = None,
    ) -> List[float]:
        """Calculate point discontinuities of a function within given bounds"""
        import re
        from expression_validator import ExpressionValidator

        # Standardize the function string
        function_string = ExpressionValidator.fix_math_expression(function_string)
        point_discontinuities_set: Set[float] = set()

        # For piecewise functions (indicated by presence of conditional operators)
        # Match both Python-style (if/else) and mathematical notation (<, >, etc.)
        if any(op in function_string for op in ["if", "else", "<", ">", "<=", ">=", "=="]):
            # Extract transition points from conditions
            # Handle both styles of conditions
            condition_patterns = [
                r"(?:<=|>=|<|>|==)\s*(-?\d*\.?\d+)",  # Mathematical notation
                r"if\s+x\s*(?:<=|>=|<|>|==)\s*(-?\d*\.?\d+)",  # Python if notation
                r"(?:<=|>=|<|>|==)\s*x\s*(?:<=|>=|<|>|==)\s*(-?\d*\.?\d+)",  # Double conditions
            ]
            for pattern in condition_patterns:
                matches = re.findall(pattern, function_string)
                point_discontinuities_set.update(float(x) for x in matches)

        # For floor and ceil functions
        if "floor" in function_string or "ceil" in function_string:
            # If bounds are provided, check each integer within bounds
            if left_bound is not None and right_bound is not None:
                left = math.ceil(left_bound)
                right = math.floor(right_bound)
                point_discontinuities_set.update(range(left, right + 1))

        # For absolute value function at its corners
        if "abs" in function_string:
            # Solve each (balanced) abs argument = 0 to find the corner points
            for abs_argument in MathUtils._function_call_arguments(function_string, "abs"):
                point_discontinuities_set.update(MathUtils._real_zeros_in_x(abs_argument))

        # Convert to list and sort
        point_discontinuities_list = sorted(point_discontinuities_set)

        # Filter points within bounds if provided
        if left_bound is not None and right_bound is not None:
            point_discontinuities_list = [x for x in point_discontinuities_list if left_bound <= x <= right_bound]

        return point_discontinuities_list

    @staticmethod
    def triangle_matches_coordinates(
        triangle: Any,
        x1: Number,
        y1: Number,
        x2: Number,
        y2: Number,
        x3: Number,
        y3: Number,
    ) -> bool:
        """
        Check if a triangle matches the given coordinates (in any order).

        Args:
            triangle: The triangle object to check
            x1, y1, x2, y2, x3, y3: The coordinates to match against

        Returns:
            bool: True if the triangle matches the coordinates, False otherwise
        """
        # Get the unique vertices of the triangle
        vertices = triangle.get_vertices()

        # Extract the coordinates from the vertices
        triangle_points = set()
        for vertex in vertices:
            triangle_points.add((vertex.x, vertex.y))

        # Create a set of the target coordinates
        target_points = {(x1, y1), (x2, y2), (x3, y3)}

        # Check if the sets of coordinates are the same
        return triangle_points == target_points

    @staticmethod
    def find_diagonal_points(
        points: Sequence[PointLike],
        rect_name_for_warning: str,
    ) -> Tuple[Optional[PointLike], Optional[PointLike]]:
        """Helper to find two diagonal points from a list of four points.
        Finds the pair of points that would best serve as diagonal corners of a rectangle.

        Args:
            points: A list of four Point objects.
            rect_name_for_warning: The name of the rectangle for warning messages.

        Returns:
            A tuple (p_diag1, p_diag2) of diagonal points, or (None, None) if not found.
        """
        if len(points) != 4:
            return None, None

        # Points with no pair differing in both x and y (e.g. on a horizontal line) cannot form a rectangle
        has_2d_extent = any(
            abs(points[i].x - points[j].x) > MathUtils.EPSILON and abs(points[i].y - points[j].y) > MathUtils.EPSILON
            for i in range(len(points))
            for j in range(i + 1, len(points))
        )
        if not has_2d_extent:
            return None, None

        # The diagonals of a rectangle are its longest vertex pairs. Consider every pair
        # (a rotated rectangle's diagonal may be axis-aligned); the first longest one wins ties.
        best_pair: Tuple[Optional[PointLike], Optional[PointLike]] = (None, None)
        best_distance = -1.0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = math.hypot(points[i].x - points[j].x, points[i].y - points[j].y)
                if distance > best_distance:
                    best_distance = distance
                    best_pair = (points[i], points[j])
        return best_pair

    @staticmethod
    def rectangular_to_polar(x: float, y: float) -> Tuple[float, float]:
        """Convert rectangular (Cartesian) coordinates to polar coordinates.

        Args:
            x: The x-coordinate in rectangular system
            y: The y-coordinate in rectangular system

        Returns:
            Tuple of (r, theta) where r is the radius and theta is the angle in radians.
            Theta is in the range (-pi, pi].
        """
        r = math.sqrt(x * x + y * y)
        theta = math.atan2(y, x)
        return (r, theta)

    @staticmethod
    def polar_to_rectangular(r: float, theta: float) -> Tuple[float, float]:
        """Convert polar coordinates to rectangular (Cartesian) coordinates.

        Args:
            r: The radius (distance from origin)
            theta: The angle in radians from the positive x-axis

        Returns:
            Tuple of (x, y) coordinates in rectangular system
        """
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        return (x, y)

    @staticmethod
    def detect_function_periodicity(
        eval_func: Any,
        test_range: float = 20.0,
        probe_count: int = 20,
        range_hint: Optional[float] = None,
    ) -> Tuple[bool, Optional[float]]:
        """
        Detect if a function is periodic by probing for oscillations.

        Tests the function over a range centered at 0 to detect if
        midpoints deviate from chords, indicating oscillation.

        Args:
            eval_func: Function to evaluate y = f(x)
            test_range: Base range to test over (centered at 0)
            probe_count: Number of probe segments
            range_hint: Optional hint from function bounds to scale test_range

        Returns:
            Tuple of (is_periodic, estimated_period or None)
        """
        # Scale test_range based on range_hint for long-period functions
        if range_hint is not None and range_hint > test_range:
            test_range = min(range_hint, 1000.0)

        left = -test_range / 2
        segment_width = test_range / probe_count
        deviation_count = 0

        for i in range(probe_count):
            seg_left = left + i * segment_width
            seg_right = seg_left + segment_width
            seg_mid = (seg_left + seg_right) / 2
            try:
                y_left = eval_func(seg_left)
                y_mid = eval_func(seg_mid)
                y_right = eval_func(seg_right)
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in [y_left, y_mid, y_right]):
                    continue
                expected_mid = (y_left + y_right) / 2
                deviation = abs(y_mid - expected_mid)
                amplitude = abs(y_right - y_left) / 2
                if amplitude > 0.01 and deviation > amplitude * 0.1:
                    deviation_count += 1
            except Exception:
                continue

        if deviation_count >= probe_count // 4:
            estimated_periods = deviation_count
            estimated_period = test_range / estimated_periods
            return True, estimated_period

        return False, None

    @staticmethod
    def numerical_derivative_at(
        func: Any,
        x: float,
        h: float = 1e-7,
    ) -> Optional[float]:
        """
        Calculate the numerical derivative of a function at a specific point.

        Uses the central difference method: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

        Args:
            func: Callable that takes a float and returns a float
            x: Point at which to calculate the derivative
            h: Relative step size for the finite difference (default: 1e-7);
                the actual step is h * max(1, |x|) so it stays resolvable at large |x|

        Returns:
            The derivative value, or None if calculation fails
        """
        try:
            step = h * max(1.0, abs(x))
            x_plus = x + step
            x_minus = x - step
            y_plus = func(x_plus)
            y_minus = func(x_minus)

            if not (math.isfinite(y_plus) and math.isfinite(y_minus)):
                return None

            # Divide by the actually representable step to avoid rounding bias
            derivative = (y_plus - y_minus) / (x_plus - x_minus)

            if not math.isfinite(derivative):
                return None

            return float(derivative)
        except Exception:
            return None

    @staticmethod
    def tangent_line_endpoints(
        slope: Optional[float],
        point: Tuple[float, float],
        length: float,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Calculate endpoints of a tangent line segment centered at a point.

        Args:
            slope: Slope of the tangent line, or None for vertical line
            point: (x, y) coordinates of the tangent point
            length: Total length of the line segment

        Returns:
            Tuple of two (x, y) endpoints of the line segment
        """
        half_length = length / 2
        px, py = point

        if slope is None:
            # Vertical line
            return (px, py - half_length), (px, py + half_length)

        # Calculate dx from: length/2 = sqrt(dx^2 + (slope*dx)^2) = dx * sqrt(1 + slope^2)
        dx = half_length / math.sqrt(1 + slope**2)
        dy = slope * dx

        return (px - dx, py - dy), (px + dx, py + dy)

    @staticmethod
    def normal_slope(tangent_slope: Optional[float]) -> Optional[float]:
        """
        Calculate the slope of the normal line given the tangent slope.

        The normal line is perpendicular to the tangent line.

        Args:
            tangent_slope: Slope of the tangent line, or None for vertical tangent

        Returns:
            Slope of the normal line, or None for vertical normal
        """
        if tangent_slope is None:
            # Vertical tangent -> horizontal normal
            return 0.0
        if abs(tangent_slope) < MathUtils.EPSILON:
            # Horizontal tangent -> vertical normal
            return None
        return -1.0 / tangent_slope

    @staticmethod
    def circle_tangent_slope_at_angle(
        center_x: float,
        center_y: float,
        radius: float,
        angle: float,
    ) -> Tuple[Tuple[float, float], Optional[float]]:
        """
        Calculate the tangent slope at a point on a circle specified by angle.

        Args:
            center_x: X-coordinate of circle center
            center_y: Y-coordinate of circle center
            radius: Circle radius
            angle: Angle in radians from positive x-axis

        Returns:
            Tuple of (tangent_point, slope) where tangent_point is (x, y)
            and slope is None for vertical tangent
        """
        # Point on circle
        px = center_x + radius * math.cos(angle)
        py = center_y + radius * math.sin(angle)

        # Tangent is perpendicular to radius
        # Radius direction: (cos(angle), sin(angle))
        # Tangent direction: (-sin(angle), cos(angle))
        # Slope = dy/dx = cos(angle) / (-sin(angle)) = -cos(angle)/sin(angle)

        sin_a = math.sin(angle)
        cos_a = math.cos(angle)

        if abs(sin_a) < MathUtils.EPSILON:
            # sin(angle) ≈ 0 means angle ≈ 0 or π, tangent is vertical
            slope: Optional[float] = None
        else:
            slope = -cos_a / sin_a

        return (px, py), slope

    @staticmethod
    def ellipse_tangent_slope_at_angle(
        center_x: float,
        center_y: float,
        radius_x: float,
        radius_y: float,
        angle: float,
        rotation_degrees: float = 0.0,
    ) -> Tuple[Tuple[float, float], Optional[float]]:
        """
        Calculate the tangent slope at a point on an ellipse specified by parameter angle.

        Args:
            center_x: X-coordinate of ellipse center
            center_y: Y-coordinate of ellipse center
            radius_x: Horizontal radius (semi-major axis)
            radius_y: Vertical radius (semi-minor axis)
            angle: Parameter angle in radians (not geometric angle)
            rotation_degrees: Rotation of ellipse in degrees

        Returns:
            Tuple of (tangent_point, slope) where tangent_point is (x, y)
            and slope is None for vertical tangent
        """
        rot_rad = math.radians(rotation_degrees)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)

        # Point in local (unrotated) coordinates
        local_x = radius_x * math.cos(angle)
        local_y = radius_y * math.sin(angle)

        # Derivative in local coordinates: dx/dt = -a*sin(t), dy/dt = b*cos(t)
        local_dx = -radius_x * math.sin(angle)
        local_dy = radius_y * math.cos(angle)

        # Transform point to world coordinates
        world_x = local_x * cos_r - local_y * sin_r + center_x
        world_y = local_x * sin_r + local_y * cos_r + center_y

        # Transform tangent vector to world coordinates
        world_dx = local_dx * cos_r - local_dy * sin_r
        world_dy = local_dx * sin_r + local_dy * cos_r

        # Calculate slope
        if abs(world_dx) < MathUtils.EPSILON:
            slope: Optional[float] = None
        else:
            slope = world_dy / world_dx

        return (world_x, world_y), slope

    # ------------------- Construction Geometry Utilities -------------------

    @staticmethod
    def perpendicular_foot(
        px: float,
        py: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> Tuple[float, float]:
        """Project a point onto a line defined by two points.

        Uses the vector dot-product projection formula to find the closest
        point on the line through (x1, y1)-(x2, y2) to the point (px, py).

        Args:
            px: X-coordinate of the point to project
            py: Y-coordinate of the point to project
            x1: X-coordinate of the first line point
            y1: Y-coordinate of the first line point
            x2: X-coordinate of the second line point
            y2: Y-coordinate of the second line point

        Returns:
            (foot_x, foot_y) coordinates of the perpendicular foot

        Raises:
            ValueError: If the two line points coincide (degenerate segment)
        """
        dx = x2 - x1
        dy = y2 - y1
        len_sq = dx * dx + dy * dy
        if len_sq < MathUtils.EPSILON * MathUtils.EPSILON:
            raise ValueError("Degenerate segment: endpoints coincide")

        t = ((px - x1) * dx + (py - y1) * dy) / len_sq
        foot_x = x1 + t * dx
        foot_y = y1 + t * dy

        if not (math.isfinite(foot_x) and math.isfinite(foot_y)):
            raise ValueError("Perpendicular foot computation produced non-finite result")

        return foot_x, foot_y

    @staticmethod
    def angle_bisector_direction(
        vx: float,
        vy: float,
        p1x: float,
        p1y: float,
        p2x: float,
        p2y: float,
    ) -> Tuple[float, float]:
        """Compute the unit vector along the angle bisector.

        Given a vertex (vx, vy) and two arm endpoints (p1x, p1y) and
        (p2x, p2y), returns the unit direction vector from the vertex
        along the bisector of the angle formed by the two arms.

        The bisector direction is found by normalizing each arm vector
        and summing them.

        Args:
            vx: X-coordinate of the angle vertex
            vy: Y-coordinate of the angle vertex
            p1x: X-coordinate of the first arm endpoint
            p1y: Y-coordinate of the first arm endpoint
            p2x: X-coordinate of the second arm endpoint
            p2y: Y-coordinate of the second arm endpoint

        Returns:
            (dx, dy) unit vector along the bisector

        Raises:
            ValueError: If an arm has zero length or the arms are collinear
                        (180-degree angle, bisector undefined)
        """
        # Arm vectors from vertex
        a1x = p1x - vx
        a1y = p1y - vy
        a2x = p2x - vx
        a2y = p2y - vy

        len1 = math.sqrt(a1x * a1x + a1y * a1y)
        len2 = math.sqrt(a2x * a2x + a2y * a2y)

        if len1 < MathUtils.EPSILON:
            raise ValueError("First arm has zero length")
        if len2 < MathUtils.EPSILON:
            raise ValueError("Second arm has zero length")

        # Normalize
        u1x, u1y = a1x / len1, a1y / len1
        u2x, u2y = a2x / len2, a2y / len2

        # Sum of unit vectors gives bisector direction
        bx = u1x + u2x
        by = u1y + u2y

        blen = math.sqrt(bx * bx + by * by)
        if blen < MathUtils.EPSILON:
            raise ValueError("Arms are collinear (180° angle): bisector is undefined")

        if not (math.isfinite(bx / blen) and math.isfinite(by / blen)):
            raise ValueError("Bisector computation produced non-finite result")

        return bx / blen, by / blen

    @staticmethod
    def circumcenter(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        x3: float,
        y3: float,
    ) -> Tuple[float, float, float]:
        """Compute the circumcenter and circumradius of a triangle.

        The circumcircle passes through all three vertices. The center is
        found using the determinant-based formula (intersection of
        perpendicular bisectors).

        Args:
            x1, y1: First vertex
            x2, y2: Second vertex
            x3, y3: Third vertex

        Returns:
            (cx, cy, radius) of the circumscribed circle

        Raises:
            ValueError: If the three points are collinear or coincident
        """
        # Translate so the first vertex is the origin to avoid precision loss far from (0, 0)
        bx, by = x2 - x1, y2 - y1
        qx, qy = x3 - x1, y3 - y1

        d = 2.0 * (bx * qy - by * qx)
        # Relative collinearity test: d (four times the area) against the longest squared side
        scale = max(bx * bx + by * by, qx * qx + qy * qy, (qx - bx) ** 2 + (qy - by) ** 2)
        if scale == 0 or abs(d) <= MathUtils.EPSILON * scale:
            raise ValueError("Points are collinear: circumcircle is undefined")

        sq_b = bx * bx + by * by
        sq_q = qx * qx + qy * qy

        ux = (qy * sq_b - by * sq_q) / d
        uy = (bx * sq_q - qx * sq_b) / d

        cx = x1 + ux
        cy = y1 + uy
        radius = math.hypot(ux, uy)

        if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(radius)):
            raise ValueError("Circumcenter computation produced non-finite result")

        return cx, cy, radius

    @staticmethod
    def incenter_and_inradius(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        x3: float,
        y3: float,
    ) -> Tuple[float, float, float]:
        """Compute the incenter and inradius of a triangle.

        The incircle is tangent to all three sides. The incenter is the
        weighted average of the vertices, weighted by the length of the
        opposite side. The inradius is ``2 * area / perimeter``.

        Args:
            x1, y1: First vertex
            x2, y2: Second vertex
            x3, y3: Third vertex

        Returns:
            (cx, cy, radius) of the inscribed circle

        Raises:
            ValueError: If the triangle is degenerate (zero area)
        """
        # Side lengths (opposite to each vertex)
        a = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)  # opposite vertex 1
        b = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)  # opposite vertex 2
        c = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # opposite vertex 3

        perimeter = a + b + c
        if perimeter < MathUtils.EPSILON:
            raise ValueError("Degenerate triangle: zero perimeter")

        # Area via cross product: 2A = |(x2-x1)(y3-y1) - (x3-x1)(y2-y1)|
        area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0
        if area < MathUtils.EPSILON:
            raise ValueError("Degenerate triangle: zero area")

        # Incenter: weighted average by opposite side lengths
        cx = (a * x1 + b * x2 + c * x3) / perimeter
        cy = (a * y1 + b * y2 + c * y3) / perimeter

        # Inradius = 2 * area / perimeter
        radius = 2.0 * area / perimeter

        if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(radius)):
            raise ValueError("Incenter computation produced non-finite result")

        return cx, cy, radius
