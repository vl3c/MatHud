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

from expression_functions import calls_python_only, rename_for_mathjs, replace_absolute_bars
from utils.statistics.descriptive import population_spread

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
    MAX_SERIES_TERMS = 100000
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
        Returns a list of dicts with x, y, and angle (radians from positive x-axis),
        ordered along the segment: two points, one where the segment touches the circle,
        or none.

        Delegates to ``object_intersections.segment_circle_points`` (the closed form of
        find_intersections), which works relative to the circle's centre and size, so
        tiny circles and tangent segments at any scale are handled. ``epsilon`` is
        kept for compatibility and unused.
        """
        from utils.object_intersections import segment_circle_points

        r = float(radius)
        if r <= 0:
            return []
        x1, y1, x2, y2 = MathUtils._segment_endpoints(segment)
        center_x, center_y = float(cx), float(cy)
        intersections: List[Dict[str, float]] = []
        for ix, iy in segment_circle_points((x1, y1), (x2, y2), (center_x, center_y), r):
            angle = math.atan2(iy - center_y, ix - center_x)
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
        """The intersection of the lines through the two segments, or None when they are parallel."""
        direction1 = (s1_x2 - s1_x1, s1_y2 - s1_y1)
        direction2 = (s2_x2 - s2_x1, s2_y2 - s2_y1)
        cross = direction1[0] * direction2[1] - direction1[1] * direction2[0]
        if abs(cross) <= MathUtils.EPSILON * max(1.0, abs(direction1[0]) + abs(direction1[1])) * max(
            1.0, abs(direction2[0]) + abs(direction2[1])
        ):
            return None
        offset = (s2_x1 - s1_x1, s2_y1 - s1_y1)
        t = (offset[0] * direction2[1] - offset[1] * direction2[0]) / cross
        return float(s1_x1 + t * direction1[0]), float(s1_y1 + t * direction1[1])

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
        """Rewrite Unicode math notation (π, x², ×, −, ≤, ∞, ...) in a raw expression for SymPy or math.js.

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
            if calls_python_only(expression):
                # Use Python evaluation for number theory functions
                result = ExpressionValidator.evaluate_expression(
                    python_expression, variables.get("x", 0) if variables else 0, variables
                )
                # Preserve boolean and list types for better display
                if isinstance(result, bool):
                    return "True" if result else "False"
                if isinstance(result, list):
                    return str(result)
                return result

            # Map advertised names onto their math.js equivalents (trunc -> fix)
            js_expression = rename_for_mathjs(js_expression)

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
    def _error_text(error: Exception) -> str:
        """'Error: <message>' for an exception, adding a JavaScript error's message when it says more."""
        message = str(error)
        js_message = getattr(error, "message", None)
        if isinstance(js_message, str) and js_message and js_message not in message:
            message = f"{message} {js_message}"
        return f"Error: {message}"

    @staticmethod
    def _cas_input(expression: Any) -> Any:
        """An expression for SymPy and math.js: Unicode notation normalized, sin^2(x) read as
        sin(x)^2 (as the evaluator and plots do). Both engines read ln, sgn, log2 and log(a, b)."""
        from expression_validator import ExpressionValidator

        if not isinstance(expression, str):
            return expression
        normalized = MathUtils._normalize_symbols(expression)
        return ExpressionValidator.convert_caret_function_powers(normalized)

    @staticmethod
    def _symbolic_input(expression: Any) -> str:
        """An expression for the symbolic tools (derive, integrate, limit, simplify, expand,
        factor), read as solve reads it (InequalitySolver.prepared: |x| bars as abs, x(x+2) and
        (x-1)(x+2) as products), with n! as factorial(n).

        Raises:
            ValueError: when it is not text.
        """
        from expression_validator import ExpressionValidator
        from utils.inequality_solver import InequalitySolver

        return str(InequalitySolver.prepared(ExpressionValidator._replace_factorials(str(expression))))

    @staticmethod
    def _sympy_answer(operation: str, **args: Optional[str]) -> Optional[Dict[str, Any]]:
        """The server's SymPy answer (SympyClient.answer): None when SymPy is unavailable,
        {"error": ...} for an input it refuses, {"result": None, "timeout": True}, or {"result": ...}."""
        from utils.sympy_client import SympyClient

        return cast(Optional[Dict[str, Any]], SympyClient.answer(operation, **args))

    # The server's SymPy worker is stopped after this many seconds (static/sympy_engine.py).
    SYMPY_TIME_LIMIT_S = 6

    @staticmethod
    def _no_answer_message(answer: Optional[Dict[str, Any]], what: str) -> Optional[str]:
        """The error for an answer without a result: SymPy unavailable, a refused input or a
        timeout; None when SymPy answered (its result may still be None)."""
        from utils.sympy_client import SympyClient

        if answer is None:
            return f"Error: {what} needs SymPy, and {SympyClient.unavailable_reason()}."
        if "error" in answer:
            return f"Error: {answer['error']}"
        if answer.get("timeout"):
            return (
                f"Error: SymPy ran out of time ({MathUtils.SYMPY_TIME_LIMIT_S} s) computing {what}; "
                "the expression is probably too large."
            )
        return None

    @staticmethod
    def _missing_answer_reason(answer: Optional[Dict[str, Any]], no_result: str) -> str:
        """Why SymPy gave no result, as a clause for a numeric answer's note."""
        from utils.sympy_client import SympyClient

        if answer is None:
            return str(SympyClient.unavailable_reason())
        if "error" in answer:
            return f"SymPy cannot read it ({str(answer['error']).rstrip('.')})"
        if answer.get("timeout"):
            return "SymPy ran out of time"
        return no_result

    @staticmethod
    def derivative(expression: str, variable: str) -> str:
        """Calculate the derivative of a mathematical expression.

        The server's SymPy engine differentiates; the result is checked against central
        differences (SymbolicChecks.derivative_matches) and refused when they disagree.

        Args:
            expression (str): Mathematical expression to differentiate
            variable (str): Variable to differentiate with respect to (e.g., "x")

        Returns:
            str: Derivative expression as string or error message
        """
        from utils.symbolic_checks import SymbolicChecks

        try:
            expression, variable = MathUtils._symbolic_input(expression), MathUtils._normalize_variable(variable)
        except Exception as e:
            return MathUtils._error_text(e)
        what = f"the derivative of {expression}"
        answer = MathUtils._sympy_answer("derive", expression=expression, variable=variable)
        refusal = MathUtils._no_answer_message(answer, what)
        if refusal is not None or answer is None:
            return refusal or ""
        derivative = answer["result"]
        if derivative is None:
            return f"Error: SymPy cannot differentiate {expression}."
        if SymbolicChecks.derivative_matches(expression, derivative, variable) is False:
            return MathUtils._rejected_result_message(what, str(derivative))
        return str(derivative)

    @staticmethod
    def _rejected_result_message(what: str, result: str) -> str:
        """Error text for a symbolic result the numeric check showed to be wrong."""
        return (
            f"Error: The symbolic engine returned {result} for {what}, but a numeric check shows it is wrong, "
            "so no symbolic result is given."
        )

    @staticmethod
    def limit(expression: str, variable: str, value_to_approach: Union[Number, str]) -> str:
        """Calculate the limit of a mathematical expression.

        The symbolic limit comes from the server's SymPy engine, which checks both sides
        itself. Without it (unavailable, out of time, an input it refuses, or no single value)
        the expression's values on each side of the target answer (SymbolicChecks.limit_verdict):
        the value they settle on (marked as a numeric estimate), or that the limit does not
        exist when the sides differ or the values oscillate.

        Args:
            expression (str): Mathematical expression
            variable (str): Variable approaching the limit (e.g., "x")
            value_to_approach (str/float): Value or "inf"/"-inf" for infinity

        Returns:
            str: The limit ("Infinity"/"-Infinity" when it diverges), "The limit does not
            exist: ..." with the reason, or an error message.
        """
        from utils.symbolic_checks import SymbolicChecks

        try:
            expression, variable = MathUtils._symbolic_input(expression), MathUtils._normalize_variable(variable)
            target = MathUtils._limit_target(value_to_approach)
        except Exception as e:
            return MathUtils._error_text(e)
        answer = MathUtils._sympy_answer("limit", expression=expression, variable=variable, target=target)
        symbolic = answer.get("result") if answer is not None else None
        reason = MathUtils._missing_answer_reason(answer, "SymPy found no single value")
        verdict = SymbolicChecks.limit_verdict(expression, variable, target)
        description = f"{expression} as {variable} -> {target}"
        return MathUtils._reconciled_limit(symbolic, verdict, description, symbolic is not None, reason)

    @staticmethod
    def _limit_target(value_to_approach: Union[Number, str]) -> str:
        """The limit target as SymPy and math.js read it: "Infinity", "-Infinity" or the value's text."""
        target = str(MathUtils._normalize_symbols(value_to_approach)).lower().replace(" ", "")
        if target in ("inf", "infinity", "+inf", "+infinity", "oo", "+oo"):
            return "Infinity"
        if target in ("-inf", "-infinity", "-oo"):
            return "-Infinity"
        return str(MathUtils._normalize_symbols(value_to_approach))

    @staticmethod
    def _symbolic_limit_claim(symbolic: Optional[str]) -> Dict[str, Any]:
        """What a limit text claims: kind "value" (with value), "+inf", "-inf", "other" or "none".

        "none" covers no result and forms that are not one limit (e^Infinity, complex values).
        "other" is a result the numeric check cannot evaluate, such as one with a parameter.
        """
        import re

        from utils.symbolic_checks import SymbolicChecks

        if symbolic is None or "NaN" in symbolic:
            return {"kind": "none"}
        text = symbolic.strip()
        if text == "Infinity":
            return {"kind": "+inf"}
        if text == "-Infinity":
            return {"kind": "-inf"}
        if "Infinity" in text or re.search(r"(?<![A-Za-z_])i(?![A-Za-z_])", text):
            return {"kind": "none"}
        value = SymbolicChecks.numeric_value(text)
        return {"kind": "value", "value": value} if value is not None else {"kind": "other"}

    @staticmethod
    def _reconciled_limit(
        symbolic: Optional[str],
        verdict: Dict[str, Any],
        description: str,
        trusted: bool = False,
        reason: str = "the symbolic engine gave no usable result",
    ) -> str:
        """The limit to report from the symbolic text and the numeric verdict (see limit()).

        A trusted symbolic result (SymPy's, which checks both sides itself) is kept whatever the
        samples say, since rounding noise can fool them; any other must agree with them. reason
        says why there is no symbolic result, for the numeric answer's note.
        """
        from utils.symbolic_checks import LIMIT_AGREEMENT

        claim = MathUtils._symbolic_limit_claim(symbolic)
        kind = verdict["kind"]
        side_note = MathUtils._one_sided_note(verdict)
        if trusted and symbolic is not None and claim["kind"] != "none":
            return symbolic + (side_note if kind != "dne" else "")
        if kind == "dne":
            return f"The limit does not exist: {verdict['reason']}."
        if kind == "unclear":
            if claim["kind"] != "none" and symbolic is not None:
                return symbolic
            return (
                f"Error: The limit of {description} could not be computed: {reason}, "
                "and the expression's values do not settle numerically."
            )
        if kind in ("+inf", "-inf"):
            return ("Infinity" if kind == "+inf" else "-Infinity") + side_note
        value = verdict["value"]
        if claim["kind"] == "value" and abs(claim["value"] - value) <= LIMIT_AGREEMENT * max(1.0, abs(value)):
            return str(symbolic) + side_note
        if claim["kind"] != "none":
            reason = f"the symbolic result {symbolic} does not match the expression's values"
        return f"{value:.12g} (numeric estimate; {reason}){side_note}"

    @staticmethod
    def _one_sided_note(verdict: Dict[str, Any]) -> str:
        """' (one-sided: ...)' when the expression is real on one side of the target only, else ''."""
        if not verdict.get("one_sided"):
            return ""
        side = verdict.get("side", "one side")
        return f" (one-sided: the expression is real only on the {side} of the point)"

    @staticmethod
    def integral(
        expression: str,
        variable: str,
        lower_bound: Optional[Number] = None,
        upper_bound: Optional[Number] = None,
    ) -> str:
        """Calculate the integral of a mathematical expression.

        The server's SymPy engine integrates. An antiderivative is checked by differentiating
        it back numerically, and a definite integral against adaptive Gauss-Kronrod
        quadrature. A definite integral without an exact value (sin(x^2), or SymPy unavailable)
        is computed numerically and marked as such.

        Args:
            expression (str): Mathematical expression to integrate
            variable (str): Variable of integration (e.g., "x")
            lower_bound (float): Optional lower bound for definite integral
            upper_bound (float): Optional upper bound for definite integral

        Returns:
            str: Integral result as string or error message
        """
        from utils.symbolic_checks import SymbolicChecks

        if (lower_bound is None) != (upper_bound is None):
            return (
                "Error: A definite integral needs both lower_bound and upper_bound; give neither for an antiderivative."
            )
        try:
            expression, variable = MathUtils._symbolic_input(expression), MathUtils._normalize_variable(variable)
            lower_bound, upper_bound = (
                MathUtils._normalize_symbols(lower_bound),
                MathUtils._normalize_symbols(upper_bound),
            )
        except Exception as e:
            return MathUtils._error_text(e)
        bounds: Dict[str, Optional[str]] = (
            {} if lower_bound is None else {"lower": str(lower_bound), "upper": str(upper_bound)}
        )
        answer = MathUtils._sympy_answer("integrate", expression=expression, variable=variable, **bounds)
        result = answer.get("result") if answer is not None else None
        if lower_bound is not None:
            reason = MathUtils._missing_answer_reason(answer, "SymPy found no closed form")
            return MathUtils._definite_integral(expression, variable, lower_bound, upper_bound, result, reason)
        refusal = MathUtils._no_answer_message(answer, f"the integral of {expression}")
        if refusal is not None:
            return refusal
        if result is None:
            return MathUtils._no_antiderivative_message(expression)
        if SymbolicChecks.antiderivative_matches(result, expression, variable) is False:
            return MathUtils._rejected_result_message(f"the integral of {expression}", str(result))
        return str(result)

    @staticmethod
    def _no_antiderivative_message(expression: str) -> str:
        return (
            f"Error: SymPy found no closed-form antiderivative of {expression}. "
            "Give lower_bound and upper_bound to get a definite integral numerically."
        )

    @staticmethod
    def _definite_integral(
        expression: str,
        variable: str,
        lower_bound: Union[Number, str],
        upper_bound: Union[Number, str],
        exact_text: Optional[str],
        reason: str = "SymPy found no closed form",
    ) -> str:
        """The exact value (exact_text) when it matches adaptive quadrature, else the quadrature
        value (its note giving reason when there is no exact value) or an error."""
        from utils.symbolic_checks import AGREEMENT, SymbolicChecks

        bounds = f"[{lower_bound}, {upper_bound}]"
        a, b = SymbolicChecks.numeric_value(lower_bound), SymbolicChecks.numeric_value(upper_bound)
        if a is not None and b is not None:
            singular_point = MathUtils._find_interior_singularity(expression, variable, a, b)
            if singular_point is not None:
                return (
                    f"Error: The integrand {expression} is singular or undefined near {variable} = {singular_point:.6g} "
                    f"inside {bounds}, so this is an improper integral that may diverge; "
                    "the antiderivative cannot be evaluated across it."
                )
        symbolic_text = exact_text
        symbolic = SymbolicChecks.numeric_value(symbolic_text) if symbolic_text is not None else None
        estimate = (
            SymbolicChecks.numeric_integral(expression, variable, a, b) if a is not None and b is not None else None
        )
        if estimate is None or not estimate["converged"]:
            if symbolic is not None and (
                estimate is None or abs(symbolic - estimate["value"]) <= 1e-3 * max(1.0, abs(symbolic))
            ):
                return str(symbolic_text)
            if symbolic_text is not None and (
                a is None or b is None or SymbolicChecks.free_variables(expression) not in ([], [variable])
            ):
                return str(symbolic_text)  # a parameter in the integrand or a bound: no numeric check is possible
            if symbolic_text in ("Infinity", "-Infinity"):
                return f"{symbolic_text} (the integral diverges)"
            unknown = "" if symbolic_text is not None else f" ({reason})"
            return (
                f"Error: The definite integral of {expression} over {bounds} could not be computed{unknown}: it is "
                "likely improper (the integrand blows up or is not real in the interval) and may diverge."
            )
        value = estimate["value"]
        if symbolic is not None and abs(symbolic - value) <= AGREEMENT * max(1.0, abs(value)):
            return str(symbolic_text)
        if symbolic_text is not None and MathUtils._mathjs_fails(symbolic_text):
            # An exact value math.js cannot evaluate (Si(1), fresnels(...)): give both
            return f"{symbolic_text} (about {value:.12g} numerically)"
        if symbolic_text is not None:
            reason = f"the symbolic result {symbolic_text} does not match it"
        return f"{value:.12g} (computed numerically; {reason})"

    @staticmethod
    def _mathjs_fails(text: str) -> bool:
        """True when math.js raises on text (an unknown function such as Si), not when it gives a value."""
        try:
            window.math.evaluate(rename_for_mathjs(text))
        except Exception:
            return True
        return False

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

            compiled = window.math.compile(rename_for_mathjs(ExpressionValidator.fix_math_expression(str(expression))))
            a = float(window.math.evaluate(rename_for_mathjs(str(lower_bound))))
            b = float(window.math.evaluate(rename_for_mathjs(str(upper_bound))))
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
        method: Optional[str] = "simpson",
        steps: Optional[int] = 200,
    ) -> Dict[str, Number]:
        """Numerically integrate an expression over a finite interval.

        This is intended for fast approximation workflows where symbolic
        integration is unnecessary or unavailable. None method or steps take
        the defaults; equal bounds give 0 and reversed bounds the negated value;
        an integrand singular inside the interval raises ValueError.
        """
        from utils.numeric_integration import integrate as integrate_numeric

        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("expression must be a non-empty string")
        if not isinstance(variable, str) or not variable.strip():
            raise ValueError("variable must be a non-empty string")
        from expression_validator import ExpressionValidator

        expression, variable = MathUtils._normalize_symbols(expression), MathUtils._normalize_variable(variable)

        lower = float(lower_bound)
        upper = float(upper_bound)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("lower_bound and upper_bound must be finite")
        if steps is None:
            steps = 200
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if steps > MathUtils.MAX_NUMERIC_INTEGRATION_STEPS:
            raise ValueError(f"steps cannot exceed {MathUtils.MAX_NUMERIC_INTEGRATION_STEPS}")

        # read as the symbolic tools read it: ln, n!, x(x+1) and |x| as math.js has them
        text = ExpressionValidator.fix_math_expression(MathUtils._symbolic_input(expression))
        from utils.symbolic_checks import SymbolicChecks

        others = [name for name in SymbolicChecks.free_variables(text) if name != variable]
        if others:
            raise ValueError(f"the integrand may contain only {variable}; it also contains {', '.join(others)}")
        integrand = MathUtils._compiled_evaluator(text)

        def eval_fn(x: float) -> float:
            return cast(float, integrand({variable: x}))

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

        Uses the server's SymPy engine for algebraic simplification.
        Combines like terms, factors, and reduces expressions. A sum written as the whole
        expression, summation(term, k, a, b), gets its closed form (ClosedFormSums.simplify).

        Args:
            expression (str): Mathematical expression to simplify

        Returns:
            str: Simplified expression as string or error message
        """
        from utils.closed_form_sums import ClosedFormSums

        summed = ClosedFormSums.simplify(expression)
        if summed is not None:
            return str(summed)
        return MathUtils._checked_rewrite("simplify", expression)

    @staticmethod
    def expand(expression: str, variable: Optional[str] = None) -> str:
        """Expand a mathematical expression by distributing operations.

        Uses the server's SymPy engine for algebraic expansion.
        Expands products, powers, and nested expressions. With a variable, a ratio of
        polynomials in it is split into partial fractions instead (PartialFractions.expand).

        Args:
            expression (str): Mathematical expression to expand
            variable (str, optional): The variable of a rational function to split

        Returns:
            str: Expanded expression as string or error message
        """
        if isinstance(variable, str) and variable.strip():
            from utils.partial_fractions import PartialFractions

            return str(PartialFractions.expand(expression, variable))
        return MathUtils._checked_rewrite("expand", expression)

    @staticmethod
    def factor(expression: str) -> str:
        """Factor a mathematical expression into its factored form.

        Uses the server's SymPy engine to factor over the rationals: polynomials, common
        factors, and whole numbers into primes (360 is 2^3*3^2*5).

        Args:
            expression (str): Mathematical expression to factor

        Returns:
            str: Factored expression as string or error message
        """
        return MathUtils._checked_rewrite("factor", expression)

    @staticmethod
    def _checked_rewrite(operation: str, expression: str) -> str:
        """simplify/expand/factor of an expression by SymPy, refused when its values differ from the input's."""
        from utils.symbolic_checks import SymbolicChecks

        try:
            expression = MathUtils._symbolic_input(expression)
        except Exception as e:
            return MathUtils._error_text(e)
        answer = MathUtils._sympy_answer(operation, expression=expression)
        refusal = MathUtils._no_answer_message(answer, f"{operation}({expression})")
        if refusal is not None or answer is None:
            return refusal or ""
        result = answer["result"]
        if result is None:
            return f"Error: SymPy has no usable {operation} form of {expression}."
        if SymbolicChecks.equivalent(expression, result) is False:
            return MathUtils._rejected_result_message(f"{operation}({expression})", result)
        return str(result)

    @staticmethod
    def solve(equation: str, variable: str, interval: Optional[str] = None) -> str:
        """Solve an equation, or an inequality (InequalitySolver), for a specific variable.

        The server's SymPy engine solves it (solveset; with an interval, the real roots in
        it), then each root is checked by substituting it back: roots that fail the equation
        or make it undefined (x = 1 in x/(x-1) = 2) are dropped and numerically equal roots
        are listed once. When SymPy has no list of roots (cos(x) = x), runs out of time or is
        unavailable, the real roots are found numerically (NumericEquations), and the answer
        says so.

        Args:
            equation (str): Mathematical equation (e.g., "x^2 + 2*x - 3 = 0"), or an
                inequality ("x^2 - 5x + 6 < 0", "1 < 2x+1 <= 5")
            variable (str): Variable to solve for (e.g., "x")
            interval (str): The interval the roots, or an inequality's answer, are restricted
                to ("[0, 2*pi]", brackets honoured), or None

        Returns:
            str: The roots as "[r1,r2,...]" ("[]" when there are none), a sentence for an
            identity, or an error message. When a periodic equation has many numeric roots,
            only those in [-2*pi, 2*pi] are listed, with a note saying so; with an interval,
            the real roots in it, with a note naming it. An inequality's solution set as
            inequalities and in interval notation (InequalitySolver.solve).
        """
        from utils.cas_text import has_inequality

        if not isinstance(equation, str):
            return "Error: The equation must be given as text."
        from utils.inequality_solver import InequalitySolver

        if has_inequality(MathUtils._normalize_symbols(equation)):
            return str(InequalitySolver.solve(equation, variable, interval))
        try:
            roots_window = InequalitySolver.window(interval)
        except ValueError as e:
            return f"Error: {e}"
        try:
            variable = MathUtils._normalize_variable(variable)
            InequalitySolver.refuse_misspellings(equation, variable)
            # the text the inequalities get: |x| bars, ln|x| and x(x+2) as products
            equation = InequalitySolver.prepared(equation)
        except Exception as e:
            return MathUtils._error_text(e)
        sides = [side.strip() for side in equation.split("=")]
        if len(sides) > 2:
            return "Error: An equation has at most one '='."
        if roots_window is not None:
            try:
                others = InequalitySolver.other_names(sides, variable)
            except ValueError:
                others = []  # left to the engines, which say what they cannot read
            if others:
                return (
                    f"Error: solve lists the roots in an interval for equations in one variable; "
                    f"besides {variable} it also contains {', '.join(others)}."
                )
        interval_args = InequalitySolver.sympy_interval_args(roots_window)
        answer = MathUtils._sympy_answer("solve", equation=equation, variable=variable, **interval_args)
        if answer is not None and answer.get("result") is not None:
            return MathUtils._sympy_roots(answer["result"], sides, variable, roots_window)
        from utils.numeric_equations import NumericEquations

        reason = MathUtils._missing_answer_reason(answer, "SymPy found no closed form")
        return str(NumericEquations.solve(sides, variable, roots_window, reason))

    @staticmethod
    def _sympy_roots(solution: Dict[str, Any], sides: Sequence[str], variable: str, roots_window: Any = None) -> str:
        """SymPy's solution set as solve() reports it, its roots checked (_checked_roots).

        With roots_window (a real_sets.Window), SymPy listed the real roots in it: the checked
        roots are filtered to it again, and a note names it ("total" counts the roots when
        SymPy listed only the first ones).
        """
        if solution.get("identity"):
            return f"Every value of {variable} satisfies the equation (it is an identity)."
        roots = MathUtils._checked_roots(
            "[" + ",".join(solution.get("roots") or []) + "]", sides, variable, roots_window
        )
        if roots_window is not None:
            total = solution.get("total")
            listed = f"the first {len(solution.get('roots') or [])} of {total}" if total else "the"
            return f"{roots} ({listed} roots in {roots_window.text()})"
        window = solution.get("window")
        if not window:
            return roots
        general = solution.get("general")
        if not general:
            return f"{roots} (the roots in [-{window}, {window}]; there are infinitely many)"
        return f"{roots} (the roots in [-{window}, {window}]; all solutions: {', '.join(general)}, for any integer n)"

    @staticmethod
    def _equation_difference(sides: Sequence[str]) -> str:
        """lhs - rhs as one expression (lhs alone when there is no '=')."""
        return f"({sides[0]})-({sides[1]})" if len(sides) == 2 else str(sides[0])

    @staticmethod
    def _checked_roots(raw_solutions: str, sides: Sequence[str], variable: str, roots_window: Any = None) -> str:
        """A root list checked and tidied (see solve()); other text is returned unchanged.

        With roots_window (a real_sets.Window), only the real roots in it are kept, in order.
        """
        from utils.cas_numeric import format_complex

        if not (raw_solutions.startswith("[") and raw_solutions.endswith("]")):
            return raw_solutions
        roots: List[Tuple[str, Optional[complex]]] = []
        for text in MathUtils._split_top_level_commas(raw_solutions[1:-1]):
            value = MathUtils._complex_value(text)
            if value is not None:
                if MathUtils._root_status(sides, variable, value) in ("fails", "undefined"):
                    continue
                if MathUtils._has_close_value(roots, value):
                    continue
                if MathUtils._is_real_root_written_with_i(text, value):
                    # a real root written with i, 2*(-1)^(1/3)-style: give the number
                    value = complex(value.real, 0.0)
                    text = format_complex(value)
            roots.append((text, value))
        if roots_window is not None:
            roots = MathUtils._roots_in_window(roots, roots_window)
        return "[" + ",".join(text for text, _ in roots) + "]"

    @staticmethod
    def _roots_in_window(
        roots: List[Tuple[str, Optional[complex]]], roots_window: Any
    ) -> List[Tuple[str, Optional[complex]]]:
        """The real roots in roots_window, sorted; a root within rounding of an end counts as that end."""
        kept: List[Tuple[str, Optional[complex]]] = []
        for text, value in roots:
            if value is None or abs(value.imag) > 1e-12 * max(1.0, abs(value.real)):
                continue
            point = value.real
            ends = ((roots_window.lower, roots_window.lower_closed), (roots_window.upper, roots_window.upper_closed))
            for end, closed in ends:
                if math.isfinite(end) and abs(point - end) <= 1e-9 * max(1.0, abs(end)):
                    point = end if closed else math.nan
            if not math.isnan(point) and roots_window.contains(point):
                kept.append((text, value))
        return sorted(kept, key=lambda item: item[1].real if item[1] is not None else 0.0)

    @staticmethod
    def _is_real_root_written_with_i(text: str, value: complex) -> bool:
        import re

        return bool(re.search(r"(?<![A-Za-z0-9_])i(?![A-Za-z0-9_])", text)) and abs(value.imag) <= 1e-12 * max(
            1.0, abs(value.real)
        )

    @staticmethod
    def _complex_value(text: str) -> Optional[complex]:
        """A root's value as a complex number, or None when math.js cannot evaluate it (a parameter)."""
        try:
            value = window.math.evaluate(rename_for_mathjs(text))
        except Exception:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return complex(float(value), 0.0) if math.isfinite(value) else None
        try:
            result = complex(float(value.re), float(value.im))
        except Exception:
            return None
        return result if math.isfinite(result.real) and math.isfinite(result.imag) else None

    @staticmethod
    def _has_close_value(roots: Sequence[Tuple[str, Optional[complex]]], value: complex) -> bool:
        return any(other is not None and abs(other - value) <= 1e-9 * max(1.0, abs(value)) for _, other in roots)

    @staticmethod
    def _root_status(sides: Sequence[str], variable: str, value: complex) -> str:
        """ "holds", "fails", "undefined" (the equation is not finite there) or "unknown" (not checkable)."""
        sides = [rename_for_mathjs(side) for side in sides]
        point: Any = value.real if value.imag == 0 else window.math.complex(value.real, value.imag)
        scope = {variable: point}
        try:
            residual = window.math.abs(
                window.math.subtract(
                    window.math.evaluate(sides[0], scope),
                    window.math.evaluate(sides[1], scope) if len(sides) == 2 else 0,
                )
            )
            residual = float(residual)
        except Exception:
            return "unknown"
        if not math.isfinite(residual):
            return "undefined"
        holds = MathUtils._equation_holds(sides, scope)
        if holds is None:
            return "unknown"
        return "holds" if holds else "fails"

    # A polynomial's roots are found numerically up to this degree.
    _MAX_NUMERIC_POLYNOMIAL_DEGREE = 40

    @staticmethod
    def _polynomial_root_values(sides: Sequence[str], variable: str) -> Optional[List[complex]]:
        """The distinct roots of a polynomial equation with numeric coefficients, found numerically; else None."""
        from utils.cas_numeric import polynomial_roots

        coefficients = MathUtils._polynomial_coefficients(MathUtils._equation_difference(sides), variable)
        if coefficients is None or not 1 <= len(coefficients) - 1 <= MathUtils._MAX_NUMERIC_POLYNOMIAL_DEGREE:
            return None
        distinct: List[complex] = []
        for root in polynomial_roots(coefficients):
            if not any(abs(root - other) <= 1e-9 * max(1.0, abs(root)) for other in distinct):
                distinct.append(root)
        return distinct

    @staticmethod
    def _polynomial_coefficients(expression: str, variable: str) -> Optional[List[float]]:
        """[c0, c1, ...] when the expression is a polynomial in variable with numeric coefficients, else None.

        Read from math.js's parse tree by utils.polynomials (exact arithmetic, no server), and
        confirmed by comparing the polynomial's values with the expression's at sample points.
        """
        from utils.polynomials import coefficients as tree_coefficients
        from utils.symbolic_checks import SAMPLE_POINTS, SymbolicChecks

        try:
            tree = MathUtils._math_tree(expression, variable)
        except Exception:
            return None
        values: Optional[List[float]] = tree_coefficients(tree)
        if values is None:
            return None
        function = SymbolicChecks.real_function(expression, variable)
        if function is None:
            return None
        for point in SAMPLE_POINTS[:3]:
            expected = function(point)
            actual = sum(c * point**k for k, c in enumerate(values))
            if (
                expected is None
                or not math.isfinite(expected)
                or abs(expected - actual) > 1e-6 * max(1.0, abs(expected))
            ):
                return None
        return values

    _OPERATOR_NODES = ("add", "subtract", "multiply", "divide", "pow", "unaryMinus", "unaryPlus")

    @staticmethod
    def _math_tree(expression: str, variable: str) -> Any:
        """math.js's parse tree of expression as the tuples utils.polynomials reads: a part
        without the variable is ("const", value) when math.js evaluates it to a finite real.

        Raises:
            Exception: what math.js raises when it cannot parse the expression.
        """
        import re

        from expression_validator import ExpressionValidator

        pattern = re.compile(rf"(?<![A-Za-z_0-9.]){re.escape(variable)}(?![A-Za-z_0-9])")
        root = window.math.parse(rename_for_mathjs(ExpressionValidator.fix_math_expression(str(expression))))

        def constant(node: Any) -> Any:
            try:
                value = node.evaluate()
            except Exception:
                return ("other",)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return ("other",)
            return ("const", float(value))

        def convert(node: Any) -> Any:
            kind = str(node.type)
            if kind == "ParenthesisNode":
                return convert(node.content)
            if kind == "ConstantNode":
                # the literal's text: Brython cannot convert a JavaScript number such as 1e20 to an int
                text = str(node.toString())
                numeric = bool(text) and text[0] in "0123456789." and all(char in "0123456789.eE+-" for char in text)
                return ("num", text) if numeric else ("other",)
            if kind == "SymbolNode" and str(node.name) == variable:
                return ("var",)
            if kind == "OperatorNode" and str(node.fn) in MathUtils._OPERATOR_NODES:
                return ("op", str(node.fn), [convert(argument) for argument in node.args])
            if pattern.search(str(node.toString())):
                return ("other",)
            return constant(node)

        return convert(root)

    @staticmethod
    def _equation_holds(sides: Sequence[str], scope: Dict[str, Any]) -> Optional[bool]:
        """Check lhs = rhs (or expression = 0) numerically with math.js.

        The tolerance also scales with how steeply the residual changes with each variable,
        so a root that is accurate to rounding error still passes when the equation has large
        coefficients (e.g. the small root of x^2 - 2000000*x + 1 = 0).
        Returns None when the equation cannot be evaluated for the given scope.
        """
        sides = [rename_for_mathjs(side) for side in sides]

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
    def solve_system_of_equations(equations: Sequence[str]) -> str:
        """Solve a system of equations (a single equation in x is the intersection of y = lhs and y = rhs).

        The server's SymPy engine solves linear systems and systems of polynomials and their
        ratios (every solution, exact when short); each solution is checked by substituting
        it into every equation. Two equations where one gives y explicitly are solved by
        substituting it into the other (solve's one-variable path, numeric when SymPy has no
        closed form). Without SymPy, a linear system is solved by elimination
        (_numeric_linear_system). Anything else, or a system SymPy cannot answer, goes to the
        numeric solver (solve_numeric).

        Returns:
            "x = 3, y = 1"; "x1 = .., y1 = .., x2 = .., y2 = .." for several solutions; a
            sentence for no solution or infinitely many; solve_numeric's JSON; or "Error: ...".
        """
        if not isinstance(equations, list) or not equations or not all(isinstance(eq, str) for eq in equations):
            raise ValueError("Invalid input for equations. Expected a list of equations.")
        try:
            # Split single equation strings into two equations
            if len(equations) == 1 and "x" in equations[0] and "=" in equations[0]:
                eq1, eq2 = equations[0].split("=")
                equations = [eq1 + "= y", eq2 + "= y"]

            from expression_validator import ExpressionValidator
            from utils.inequality_solver import InequalitySolver

            # read as solve reads an equation: |x| bars, x(x+1) as a product
            equations = [
                ExpressionValidator.fix_math_expression(InequalitySolver.prepared(eq), python_compatible=False)
                for eq in equations
            ]
            if any(len(eq.split("=")) > 2 for eq in equations):
                return "Error: An equation has at most one '='."
            answer = MathUtils._sympy_answer("solve_system", equations=";".join(equations))
            if answer is not None and "error" in answer:
                return f"Error: {answer['error']}"
            result = answer.get("result") if answer is not None else None
            if result is None:
                result = MathUtils._numeric_linear_system(equations)
            if isinstance(result, dict):
                text = MathUtils._system_answer(result, equations)
                if text is not None:
                    return text
            substituted = MathUtils._solve_by_substitution(equations)
            if substituted is not None:
                return substituted
            return MathUtils.solve_numeric(equations)
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def _numeric_linear_system(equations: Sequence[str]) -> Optional[Dict[str, Any]]:
        """The answer of a linear system without SymPy (utils.linear_systems.solve_linear), every
        name an unknown; None unless each equation is linear, checked at sample points."""
        from utils.linear_systems import solve_linear
        from utils.symbolic_checks import SymbolicChecks

        differences = [MathUtils._equation_difference(equation.split("=")) for equation in equations]
        names = sorted({name for difference in differences for name in SymbolicChecks.free_variables(difference)})
        if not names or len(names) > MathUtils._MAX_LINEAR_UNKNOWNS:
            return None
        if any(name in ("x", "y", "z", "t") for name in names) and not set(names) <= {"x", "y", "z", "t"}:
            return None  # parameters (x + y = a) or a glued product (xy = 6): not for elimination
        try:
            compiled = [window.math.compile(rename_for_mathjs(difference)) for difference in differences]
        except Exception:
            return None

        def value(index: int, point: Sequence[float]) -> Optional[float]:
            try:
                result = compiled[index].evaluate({name: coordinate for name, coordinate in zip(names, point)})
            except Exception:
                return None
            if isinstance(result, bool) or not isinstance(result, (int, float)) or not math.isfinite(result):
                return None
            return float(result)

        origin = [0.0] * len(names)
        units = [[1.0 if other == column else 0.0 for other in range(len(names))] for column in range(len(names))]
        checks = [
            [0.37 + 0.71 * column for column in range(len(names))],
            [-1.3 + 0.53 * column for column in range(len(names))],
        ]
        coefficients: List[List[float]] = []
        constants: List[float] = []
        for index in range(len(differences)):
            at_origin = value(index, origin)
            at_units = [value(index, unit) for unit in units]
            if at_origin is None or any(item is None for item in at_units):
                return None
            row = [cast(float, item) - at_origin for item in at_units]
            for point in checks:
                expected = at_origin + sum(a * p for a, p in zip(row, point))
                actual = value(index, point)
                size = max(1.0, abs(at_origin), sum(abs(a * p) for a, p in zip(row, point)))
                if actual is None or abs(actual - expected) > 1e-9 * size:
                    return None  # not linear
            coefficients.append(row)
            constants.append(-at_origin)
        return dict(solve_linear(coefficients, constants, names))

    # A system with more names than this is not tried as a linear system without SymPy.
    _MAX_LINEAR_UNKNOWNS = 6

    @staticmethod
    def _system_answer(result: Dict[str, Any], equations: Sequence[str]) -> Optional[str]:
        """SymPy's solve_system result as the answer text, its solutions checked by substitution;
        None when no solution it gave holds (the numeric solver then answers)."""
        if result.get("none"):
            return "No solution: the equations are inconsistent (they contradict each other)."
        if result.get("infinite"):
            return MathUtils._infinite_system_answer(result)
        solutions = [
            solution
            for solution in result.get("solutions") or []
            if MathUtils._system_solution_holds(solution, equations) is not False
        ]
        if not solutions:
            if result.get("solutions"):
                return None  # every solution failed the check
            count = int(result.get("complex") or 0)
            if count:
                return f"No real solution (the equations have {count} complex solution{'s' if count != 1 else ''})."
            return "No real solution."
        if len(solutions) == 1:
            return ", ".join(f"{name} = {value}" for name, value in solutions[0].items())
        return ", ".join(
            f"{name}{index} = {value}"
            for index, solution in enumerate(solutions, start=1)
            for name, value in solution.items()
        )

    @staticmethod
    def _infinite_system_answer(result: Dict[str, Any]) -> str:
        free = [str(name) for name in result.get("free") or []]
        any_free = ", ".join(free)
        if result.get("curves"):
            branches = " or ".join(
                ", ".join(f"{name} = {value}" for name, value in curve.items()) for curve in result["curves"]
            )
            return f"Infinitely many solutions (whole curves of them): {branches}, for any {any_free}."
        general = result.get("general") or {}
        rank = int(result.get("rank") or len(general))
        unknowns = len(general) + len(free)
        values = ", ".join(f"{name} = {value}" for name, value in general.items())
        return (
            f"Infinitely many solutions: the equations are dependent ({rank} independent "
            f"equation{'s' if rank != 1 else ''} for {unknowns} unknowns): {values}, for any {any_free}."
        )

    # Values given to the parameters of a solution (x = a/2 + 1/2) to check it numerically.
    _PARAMETER_SAMPLES = (0.7310, 1.4142, 2.2360)

    @staticmethod
    def _system_solution_holds(solution: Dict[str, str], equations: Sequence[str]) -> Optional[bool]:
        """Whether every equation holds at the solution; values naming parameters are checked at a
        few sample values of the parameters. None when it cannot be evaluated."""
        from utils.symbolic_checks import SymbolicChecks

        names = set(solution)
        parameters = sorted(
            {
                name
                for text in list(solution.values()) + list(equations)
                for name in SymbolicChecks.free_variables(text)
                if name not in names
            }
        )
        for sample in MathUtils._PARAMETER_SAMPLES:
            scope: Dict[str, Any] = {name: sample + 0.1 * index for index, name in enumerate(parameters)}
            try:
                for name, value in solution.items():
                    computed = window.math.evaluate(rename_for_mathjs(str(value)), scope)
                    if isinstance(computed, bool) or not isinstance(computed, (int, float)):
                        return None
                    scope[name] = float(computed)
            except Exception:
                return None
            for equation in equations:
                holds = MathUtils._equation_holds(equation.split("="), scope)
                if holds is not True:
                    return holds
            if not parameters:
                break
        return True

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
    def _solve_by_substitution(equations: Sequence[str]) -> Optional[str]:
        """Two equations in x and y where one is 'y = f(x)': f substituted into the other and the
        result solved for x like solve() (SymPy, else numerically), y = f(x) at each real root.

        Returns the answer text (with solve's note, when it has one), or None when no equation
        is explicit in y or solve gives no list of roots.
        """
        import re

        if len(equations) != 2:
            return None
        explicit_forms = [MathUtils._explicit_y_expression(eq) for eq in equations]
        index = next((i for i, form in enumerate(explicit_forms) if form is not None), None)
        if index is None:
            return None
        y_expression = str(explicit_forms[index])
        other_sides = equations[1 - index].split("=") + (["0"] if "=" not in equations[1 - index] else [])
        substituted = [re.sub(MathUtils._Y_TOKEN_PATTERN, f"({y_expression})", side) for side in other_sides]
        answer = MathUtils.solve(f"{substituted[0]} = {substituted[1]}", "x")
        if not answer.startswith("["):
            return None
        listed, _, note = answer.partition("]")
        solutions: List[Tuple[str, str]] = []
        for x_text in MathUtils._split_top_level_commas(listed[1:]):
            x_value = MathUtils._complex_value(x_text)
            if x_value is None or abs(x_value.imag) > 1e-12 * max(1.0, abs(x_value.real)):
                continue
            y_value = MathUtils._complex_value(MathUtils._substituted_y(y_expression, x_value.real))
            if y_value is None or abs(y_value.imag) > 1e-12 * max(1.0, abs(y_value.real)):
                continue
            from utils.cas_numeric import format_number

            scope = {"x": x_value.real, "y": y_value.real}
            if all(MathUtils._equation_holds(eq.split("="), scope) is not False for eq in equations):
                solutions.append((x_text, format_number(y_value.real)))
        if not solutions:
            return f"No real solution{note.strip() and ' ' + note.strip()}"
        if len(solutions) == 1:
            text = f"x = {solutions[0][0]}, y = {solutions[0][1]}"
        else:
            text = ", ".join(f"x{i} = {x}, y{i} = {y}" for i, (x, y) in enumerate(solutions, start=1))
        return text + note

    @staticmethod
    def _substituted_y(y_expression: str, x_value: float) -> str:
        """f(x) with x replaced by the number, for math.js."""
        from compiled_expression import substitute_variable

        return str(substitute_variable(y_expression, "x", f"({x_value!r})"))

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
        """Round half away from zero, as math.js does (Python's round() rounds half to even:
        round(2.5) is 2), so an expression rounds alike whichever evaluator runs it."""
        if isinstance(value, int) and ndigits >= 0:
            return value  # already exact (a float would lose digits of a large integer)
        if 0 <= ndigits <= 15:
            return float(window.math.round(value, ndigits))  # math.js corrects 1.005 -> 1.01
        if ndigits < 0:
            step = 10.0**-ndigits  # round to tens, hundreds, ...
            return math.copysign(math.floor(abs(value) / step + 0.5) * step, value)
        scale = 10.0**ndigits
        return math.copysign(math.floor(abs(value) * scale + 0.5) / scale, value)

    @staticmethod
    def gcd(*values: Number) -> int:
        """Greatest common divisor; raises TypeError for a non-integer argument such as 4.5."""
        from utils.number_theory import gcd

        return cast(int, gcd(values))

    @staticmethod
    def lcm(*values: Number) -> int:
        """Least common multiple; raises TypeError for a non-integer argument such as 4.5."""
        from utils.number_theory import lcm

        return cast(int, lcm(values))

    # ========== Number Theory Functions ==========

    @staticmethod
    def is_prime(n: Number) -> bool:
        """Check if a number is prime.

        Uses trial division by small primes, then a deterministic Miller-Rabin test
        (exact for n < 3.3e24; above that a composite passes with probability below 4^-25).

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
        from utils.number_theory import is_prime

        return cast(bool, is_prime(n))

    @staticmethod
    def prime_factors(n: Number) -> List[int]:
        """Return the prime factorization of n with multiplicity.

        Trial division finds small factors and Pollard's rho the rest.

        Args:
            n: Positive integer to factorize

        Returns:
            list[int]: List of prime factors (with repetition for multiplicity)

        Raises:
            ValueError: If n is less than 1, above 10^30, or has only prime factors
                too large to find quickly
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
        from utils.number_theory import prime_factors

        return cast(List[int], prime_factors(n))

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
    def summation(expression: str, variable: str, start: int, end: int) -> Number:
        """Compute sum of expression for integer values from start to end.

        Calculates Σ expression for variable = start to end with math.js, adding the terms
        with math.fsum. The variable is passed in math.js's scope, which overrides its
        constants, so an index named i or e works.

        Args:
            expression: Mathematical expression containing the variable
            variable: Variable name to substitute (e.g., "n")
            start: Starting integer value (inclusive)
            end: Ending integer value (inclusive)

        Returns:
            The sum, an int when it is whole (0 for an empty range, start > end); a number, not
            text, so 3*summation(...) multiplies it

        Raises:
            ValueError: If there are more than MAX_SERIES_TERMS terms, or a term is not a finite real number
        """
        terms = MathUtils._series_terms("summation", expression, variable, start, end)
        return MathUtils._series_value(math.fsum(terms))

    @staticmethod
    def _series_terms(name: str, expression: str, variable: str, start: int, end: int) -> List[float]:
        """The values of expression (math.js, or Python's ** spelling) for variable = start .. end."""
        start = int(start)
        end = int(end)
        if start > end:
            return []
        term_count = end - start + 1
        if term_count > MathUtils.MAX_SERIES_TERMS:
            raise ValueError(f"{name} supports at most {MathUtils.MAX_SERIES_TERMS} terms")
        from utils.inequality_solver import InequalitySolver

        # evaluate_expression hands the quoted term over in Python syntax (k**2); k(k+1) is a
        # product, not a call, as simplify's sums read it
        text = str(expression).replace("**", "^")
        try:
            text = str(InequalitySolver.prepared(text))
        except Exception:
            pass  # math.js says what it cannot read
        evaluate = MathUtils._compiled_evaluator(text)
        terms: List[float] = []
        for index in range(start, end + 1):
            value = evaluate({variable: index})
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{expression} is not a finite real number at {variable} = {index}")
            terms.append(float(value))
        return terms

    @staticmethod
    def _series_value(total: float) -> Number:
        """A sum or product: an int when it is whole and exact in a float."""
        if not math.isfinite(total):
            raise ValueError("the result is too large to represent")
        if total.is_integer() and abs(total) < 2.0**53:
            return int(total)
        return total

    @staticmethod
    def product(expression: str, variable: str, start: int, end: int) -> Number:
        """Compute product of expression for integer values from start to end.

        Calculates Π expression for variable = start to end with math.js (see summation).

        Args:
            expression: Mathematical expression containing the variable
            variable: Variable name to substitute (e.g., "n")
            start: Starting integer value (inclusive)
            end: Ending integer value (inclusive)

        Returns:
            The product, an int when it is whole (1 for an empty range, start > end)

        Raises:
            ValueError: If there are more than MAX_SERIES_TERMS terms, or a term is not a finite real number
        """
        total = 1.0
        for value in MathUtils._series_terms("product", expression, variable, start, end):
            total *= value
        return MathUtils._series_value(total)

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
            n: Number of terms to sum (a whole number)

        Returns:
            Number: Sum of the arithmetic series; exact for integer inputs

        Raises:
            ValueError: If n < 1
            TypeError: If n is not a whole number
        """
        from utils.number_theory import arithmetic_sum

        return cast(Number, arithmetic_sum(first, diff, n))

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
            n: Number of terms to sum (a whole number)

        Returns:
            Number: Sum of the geometric series; exact for integer inputs
            (geometric_sum(1, 2, 64) is 2^64 - 1)

        Raises:
            ValueError: If n < 1 or the sum is too large to represent
            TypeError: If n is not a whole number
        """
        from utils.number_theory import geometric_sum

        return cast(Number, geometric_sum(first, ratio, n))

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
        from compiled_expression import substitute_variable

        try:
            # Create a_{n+1} by substituting n -> n+1
            next_term = str(substitute_variable(str(expression), str(n_var), f"({n_var}+1)"))
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

        The limit of ``expression`` is taken first and its absolute value used; the limit of
        ``abs_expression`` (already wrapped in abs) is the fallback when that is not numeric.
        """
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
        """Numerically evaluate a limit() result (e.g. "exp(-1)", "Infinity", "0 (numeric estimate; ...)").

        Returns None for errors, unevaluated limits, limits that do not exist, and non-numeric results.
        """
        import re

        if "Error" in limit_result or "limit" in limit_result:
            return None
        # limit() notes a numeric estimate or a one-sided limit after the value
        annotated = re.match(r"^(\S+) \((?:numeric estimate|one-sided)", limit_result)
        if annotated:
            limit_result = annotated.group(1)
        from utils.symbolic_checks import SymbolicChecks

        text = limit_result.strip()
        if text in ("Infinity", "+Infinity"):
            return float("inf")
        if text == "-Infinity":
            return float("-inf")
        value = SymbolicChecks.numeric_value(text)
        return None if value is None else float(value)

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
    def pstdev(values: List[float]) -> float:
        """Calculate the population standard deviation (dividing by n) of a list of values.

        Uses the compute_descriptive_statistics routine, so both report the same figure.

        Args:
            values (list): Non-empty list of finite numbers

        Returns:
            float: Population standard deviation
        """
        std_dev: float = population_spread(values, "pstdev's list")[1]
        return std_dev

    @staticmethod
    def pvariance(values: List[float]) -> float:
        """Calculate the population variance (dividing by n) of a list of values.

        Uses the compute_descriptive_statistics routine, so both report the same figure.

        Args:
            values (list): Non-empty list of finite numbers

        Returns:
            float: Population variance
        """
        variance: float = population_spread(values, "pvariance's list")[0]
        return variance

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
        sin(1/x) at 0). Candidates closer than _PROBE_SAME_POINT (relative) are one point, and
        with bounds only points within them are listed.
        """
        from expression_validator import ExpressionValidator

        # Standardize the function string; bars become abs() here only (the stored text keeps
        # them), so ln|x| and 1/|x| show their log argument and denominator
        function_string = replace_absolute_bars(ExpressionValidator.fix_math_expression(function_string))
        log_zeros: List[float] = []
        denominator_zeros: List[float] = []

        # For logarithmic functions: where the (first) argument is zero
        left = left_bound if left_bound is not None else -1000
        right = right_bound if right_bound is not None else 1000
        for log_argument in MathUtils._function_call_arguments(function_string, "log|ln|log10|log2"):
            arguments = MathUtils._split_top_level_commas(log_argument)
            if arguments:
                log_zeros.extend(MathUtils._real_zeros_in_x(arguments[0], left, right))

        # For rational functions: where any denominator that depends on x is zero
        for denominator in MathUtils._denominators(function_string):
            denominator_zeros.extend(MathUtils._real_zeros_in_x(denominator, left, right))

        # For tangent functions (word boundary so atan/arctan are excluded)
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
        for points in (denominator_zeros, tangent_poles):
            asymptotes_found, holes_found = MathUtils._classify_poles(evaluate, points, singular_points)
            vertical_asymptotes.extend(asymptotes_found)
            discontinuities.extend(holes_found)

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
    def _classify_poles(
        evaluate: Optional[Callable[[float], Optional[float]]],
        poles: List[float],
        singular_points: List[float],
    ) -> Tuple[List[float], List[float]]:
        """Split denominator zeros or tan() poles into (asymptotes, point discontinuities) by
        sampling f beside each one (the quick verdict, else the full classifier).

        A pole where f stays bounded is not an asymptote: (x^2-1)/(x-1) at 1, x/tan(x) and
        1/tan(x) at pi/2, and tan(x)*(x - 101*pi/2) at 101*pi/2 only.
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

    # Brython cannot bring a JavaScript number of magnitude 2^53 or more into Python (every such
    # double is integer-valued and it raises "not a big int"), so such a value is computed again
    # divided by 2^1000, which is exact, and scaled back.
    _BIG_NUMBER_SCALE = 2.0**1000

    @staticmethod
    def _compiled_evaluator(expression: str) -> Callable[[Any], Any]:
        """math.js compile(expression).evaluate as a function of the scope, safe for values >= 2^53.

        Raises what math.js raises when compiling; the returned function raises what
        evaluate raises, except Brython's "not a big int". Functions math.js knows under another
        name are renamed first (trunc -> fix).
        """
        expression = rename_for_mathjs(expression)
        compiled = window.math.compile(expression)
        scaled: List[Any] = []

        def evaluate(scope: Any) -> Any:
            try:
                return compiled.evaluate(scope)
            except Exception as error:
                if "not a big int" not in str(error):
                    raise
            if not scaled:
                scaled.append(window.math.compile(f"({expression}) / 2^1000"))
            value = scaled[0].evaluate(scope)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return value
            return float(value) * MathUtils._BIG_NUMBER_SCALE

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
            compiled = window.math.compile(rename_for_mathjs(ExpressionValidator.fix_math_expression(str(expression))))
        except Exception:
            return None
        probe = MathUtils._javascript_probe()

        def evaluate(point: float) -> Optional[float]:
            value = probe(compiled, variable, point)
            if isinstance(value, str):
                return float(value) if value else None  # "" for no real value
            return float(value)

        return evaluate

    # The probe runs in JavaScript: a value of 2^53 or more comes back as text, since Brython
    # turns a whole JavaScript number into an int and raises "not a big int" past 2^53 (the
    # exception made sampling e^x up to x = 700 take 0.3 s); failures and non-real values are
    # "" (Brython does not read a returned null as None).
    _PROBE_SOURCE = (
        "var scope = {}; scope[name] = x; var v;"
        " try { v = compiled.evaluate(scope); } catch (e) { return ''; }"
        " if (typeof v !== 'number') return '';"
        " return (isFinite(v) && Math.abs(v) >= 9007199254740992) ? String(v) : v;"
    )
    _probe_function: List[Any] = []

    @staticmethod
    def _javascript_probe() -> Any:
        if not MathUtils._probe_function:
            MathUtils._probe_function.append(window.Function("compiled", "name", "x", MathUtils._PROBE_SOURCE))
        return MathUtils._probe_function[0]

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
    def _real_zeros_in_x(expression: str, left: Number = -1000, right: Number = 1000) -> List[float]:
        """The real zeros of an expression in x: every one of a polynomial (from its
        coefficients), else those the scan finds in [left, right] (_scanned_zeros); [] when it
        has none or cannot be evaluated."""
        import re

        if not re.search(MathUtils._X_TOKEN_PATTERN, expression):
            return []
        roots = MathUtils._polynomial_root_values([expression], "x")
        if roots is not None:
            return [root.real for root in roots if abs(root.imag) <= 1e-9 * max(1.0, abs(root.real))]
        evaluate = MathUtils._mathjs_probe_evaluator(expression)
        if evaluate is None:
            return []
        return MathUtils._scanned_zeros(evaluate, float(left), float(right))

    # The zero scan samples this many points and lists at most this many zeros, those nearest 0 first.
    _ZERO_SCAN_POINTS = 4001
    _MAX_SCANNED_ZEROS = 500
    # Zeros closer than this many sample steps mean the scan may skip some (sin(10x) sampled every
    # 0.5): the middle 1/_FINE_SCAN_RATIO of the range is scanned again, that much finer.
    _CLOSE_ZEROS_STEPS = 4
    _FINE_SCAN_RATIO = 40

    @staticmethod
    def _scanned_zeros(evaluate: Callable[[float], Optional[float]], left: float, right: float) -> List[float]:
        """The zeros of f in [left, right]: exact zeros at samples, sign changes between
        neighbouring samples narrowed to 1e-12 (relative), and touching zeros (1 - cos(x) at 0)
        at small local minima of |f| (_touching_zeros). A sign change where |f| does not shrink
        towards it is a pole or a jump, not a zero; a run of zero samples (an underflow such as
        e^(-x) far out, or a stretch where f is 0) gives no zeros.

        At most _MAX_SCANNED_ZEROS, those nearest 0 (or the end of the range nearest it). When
        zeros lie closer together than the samples resolve, the middle of the range is scanned
        again, finer, so the zeros there are all listed (1/sin(10x) on [-1000, 1000]).
        """
        if not (math.isfinite(left) and math.isfinite(right)) or left >= right:
            return []
        center = min(max(0.0, left), right)
        zeros = MathUtils._grid_zeros(evaluate, left, right, center)
        step = (right - left) / (MathUtils._ZERO_SCAN_POINTS - 1)
        if any(b - a < MathUtils._CLOSE_ZEROS_STEPS * step for a, b in zip(zeros, zeros[1:])):
            half = (right - left) / MathUtils._FINE_SCAN_RATIO / 2
            inner_left, inner_right = max(left, center - half), min(right, center + half)
            inner = MathUtils._grid_zeros(evaluate, inner_left, inner_right, center)
            zeros = inner + [zero for zero in zeros if not inner_left <= zero <= inner_right]
        nearest = sorted(zeros, key=lambda zero: abs(zero - center))[: MathUtils._MAX_SCANNED_ZEROS]
        return sorted(nearest)

    @staticmethod
    def _grid_zeros(
        evaluate: Callable[[float], Optional[float]], left: float, right: float, center: float
    ) -> List[float]:
        """The zeros _scanned_zeros finds on one grid of [left, right], sorted; the sign changes
        nearest center are narrowed first, up to _MAX_SCANNED_ZEROS."""
        count = MathUtils._ZERO_SCAN_POINTS
        step = (right - left) / (count - 1)
        xs = [left + index * step for index in range(count - 1)] + [right]
        values = [evaluate(x) for x in xs]
        zeros: List[float] = []
        changes: List[int] = []
        for index, value in enumerate(values):
            if value == 0:
                neighbours = values[max(0, index - 1) : index] + values[index + 1 : index + 2]
                if all(other is not None and other != 0 for other in neighbours):
                    zeros.append(xs[index])
                continue
            following = values[index + 1] if index + 1 < count else None
            if value is None or following is None or following == 0 or (value < 0) == (following < 0):
                continue
            changes.append(index)
        changes.sort(key=lambda index: abs(xs[index] - center))
        for index in changes:
            if len(zeros) >= MathUtils._MAX_SCANNED_ZEROS:
                break
            zero = MathUtils._sign_change_zero(evaluate, xs[index], xs[index + 1], cast(float, values[index]))
            if zero is not None:
                zeros.append(zero)
        return sorted(zeros + MathUtils._touching_zeros(evaluate, xs, values, center))

    # A local minimum of |f| below this fraction of the median |f| is searched for a touching zero
    _TOUCHING_FRACTION = 1e-2
    _MAX_TOUCHING_SEARCHES = 200
    # Samples smaller than this are underflow, not the sides of a touching zero
    _UNDERFLOW_SIZE = 1e-100

    @staticmethod
    def _touching_zeros(
        evaluate: Callable[[float], Optional[float]],
        xs: Sequence[float],
        values: Sequence[Optional[float]],
        center: float = 0.0,
    ) -> List[float]:
        """Zeros where f touches 0 without changing sign between samples: the minimum of |f|
        found by golden-section search beside a small local minimum of the samples, kept when
        |f| there is rounding noise (below 1e-10 of the neighbouring samples). The minima
        nearest center are searched first, up to _MAX_TOUCHING_SEARCHES."""
        sizes = sorted(abs(value) for value in values if value is not None and value != 0)
        if not sizes:
            return []
        small = MathUtils._TOUCHING_FRACTION * sizes[len(sizes) // 2]
        found: List[float] = []
        searched = 0
        for index in sorted(range(1, len(values) - 1), key=lambda index: abs(xs[index] - center)):
            before, value, after = values[index - 1], values[index], values[index + 1]
            if before is None or value is None or after is None or value == 0 or abs(value) > small:
                continue
            if not ((before < 0) == (value < 0) == (after < 0)) or not abs(before) > abs(value) <= abs(after):
                continue
            searched += 1
            if searched > MathUtils._MAX_TOUCHING_SEARCHES:
                break
            if max(abs(before), abs(after)) < MathUtils._UNDERFLOW_SIZE:
                continue  # e^(-x) far out: values near underflow, not a zero
            location = MathUtils._golden_zero(evaluate, xs[index - 1], xs[index + 1])
            if location is None:
                continue
            reached = evaluate(location)
            if reached is not None and abs(reached) <= 1e-10 * max(1.0, abs(before), abs(after)):
                found.append(location)
        return found

    @staticmethod
    def _golden_zero(evaluate: Callable[[float], Optional[float]], low: float, high: float) -> Optional[float]:
        """The minimum of |f| on [low, high] by golden-section search (to 1e-12 relative)."""
        ratio = (math.sqrt(5) - 1) / 2

        def size(x: float) -> float:
            value = evaluate(x)
            return math.inf if value is None else abs(value)

        first, second = high - ratio * (high - low), low + ratio * (high - low)
        first_size, second_size = size(first), size(second)
        for _ in range(120):
            if high - low <= 1e-12 * max(1.0, abs(low), abs(high)):
                break
            if first_size <= second_size:
                high, second, second_size = second, first, first_size
                first = high - ratio * (high - low)
                first_size = size(first)
            else:
                low, first, first_size = first, second, second_size
                second = low + ratio * (high - low)
                second_size = size(second)
        middle = (low + high) / 2
        return middle if math.isfinite(size(middle)) else None

    @staticmethod
    def _sign_change_zero(
        evaluate: Callable[[float], Optional[float]], low: float, high: float, low_value: float
    ) -> Optional[float]:
        """The sign change between low and high narrowed to 1e-12 (relative); None when |f| there
        is not far below its values at low and high (a pole or a jump).

        Illinois false position, with a bisection step whenever three steps have not halved
        the bracket, so a smooth zero takes about 10 evaluations instead of 40 and a pole
        still narrows like bisection.
        """
        high_value = evaluate(high)
        if high_value is None:
            return None
        edge = max(abs(low_value), abs(high_value))
        weighted_low, weighted_high = low_value, high_value
        last_side = 0
        checkpoint, steps_since = high - low, 0
        for _ in range(200):
            width = high - low
            if width <= 1e-12 * max(1.0, abs(low), abs(high)):
                break
            if steps_since >= 3 and width > checkpoint / 2:
                middle = (low + high) / 2
            else:
                middle = high - weighted_high * width / (weighted_high - weighted_low)
                if not low < middle < high:
                    middle = (low + high) / 2
            steps_since += 1
            value = evaluate(middle)
            if value is None:
                return None
            if value == 0:
                return middle
            if (value < 0) == (low_value < 0):
                low, low_value, weighted_low = middle, value, value
                if last_side == -1:
                    weighted_high /= 2
                last_side = -1
            else:
                high, high_value, weighted_high = middle, value, value
                if last_side == 1:
                    weighted_low /= 2
                last_side = 1
            if high - low <= checkpoint / 2:
                checkpoint, steps_since = high - low, 0
        middle = (low + high) / 2
        value = evaluate(middle)
        if value is None or abs(value) > 1e-6 * edge:
            return None
        return middle

    @staticmethod
    def _tangent_asymptotes(argument: str, left: Number, right: Number) -> List[float]:
        """Return the asymptotes of tan(argument) within [left, right] for a linear argument a*x + b."""
        try:
            compiled = window.math.compile(rename_for_mathjs(argument))
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

        The limits are estimated numerically (_limit_at_infinity), never by asking the server:
        every drawn function needs them.
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
        sample (not real) means no limit on that side. An infinite sample or a NaN one (inf/inf,
        e.g. cosh(x)/sinh(x) once both overflow) ends the sampling, and the samples before it
        are judged. After an infinite sample, f just before the overflow (_value_before_overflow)
        must also be at least as close to the limit as the last sample: exp(x-700) + 1 reads 1
        up to x = 622.7 but nears 1e308 before it overflows, so it has no limit. That value is
        a check, not a sample: it lies off the 4-fold grid, which Aitken extrapolation needs.
        """
        values: List[float] = []
        previous_x = 0.0
        before_overflow: Optional[float] = None
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
                break
            values.append(value)
            previous_x = x
        limit = MathUtils._settled_value(values)
        if limit is None or before_overflow is None:
            return limit
        slack = MathUtils._SETTLED_SPREAD * max(1.0, abs(limit))
        if abs(before_overflow - limit) > abs(values[-1] - limit) + slack:
            return None
        return limit

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
            scan_left = float(left_bound) if left_bound is not None else -1000.0
            scan_right = float(right_bound) if right_bound is not None else 1000.0
            for abs_argument in MathUtils._function_call_arguments(function_string, "abs"):
                point_discontinuities_set.update(MathUtils._real_zeros_in_x(abs_argument, scan_left, scan_right))

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
        if perimeter == 0:
            raise ValueError("Degenerate triangle: zero perimeter")

        # Area via cross product: 2A = |(x2-x1)(y3-y1) - (x3-x1)(y2-y1)|
        area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0
        # Relative collinearity test, the one circumcenter uses: 4A against the longest squared side
        longest = max(a, b, c)
        if 4.0 * area <= MathUtils.EPSILON * longest * longest:
            raise ValueError("Degenerate triangle: zero area")

        # Incenter: weighted average by opposite side lengths
        cx = (a * x1 + b * x2 + c * x3) / perimeter
        cy = (a * y1 + b * y2 + c * y3) / perimeter

        # Inradius = 2 * area / perimeter
        radius = 2.0 * area / perimeter

        if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(radius)):
            raise ValueError("Incenter computation produced non-finite result")

        return cx, cy, radius
