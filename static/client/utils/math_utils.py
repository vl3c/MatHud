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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

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
        elif abs(n) >= 10**max_digits or (abs(n) < 10**(-max_digits + 1)):
            formatted_number = f"{n:.1e}"
        else:
            formatted_number = f"{n:.{max_digits}g}"
        # Process scientific notation to adjust exponent formatting
        if 'e' in formatted_number:
            base, exponent = formatted_number.split('e')
            base = base.rstrip('0').rstrip('.')
            # Fix handling for exponent sign
            sign = exponent[0] if exponent.startswith('-') else '+'
            exponent_number = exponent.lstrip('+').lstrip('-').lstrip('0') or '0'
            formatted_number = f"{base}e{sign}{exponent_number}"
        else:
            # Truncate to max_digits significant digits for non-scientific notation
            if '.' in formatted_number:
                formatted_number = formatted_number[:formatted_number.find('.') + max_digits]
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
        first_direction_match = MathUtils.point_matches_coordinates(segment.point1, x1, y1) and MathUtils.point_matches_coordinates(segment.point2, x2, y2)
        second_direction_match = MathUtils.point_matches_coordinates(segment.point1, x2, y2) and MathUtils.point_matches_coordinates(segment.point2, x1, y1)
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
        segment_length = math.sqrt((sp2x - sp1x)**2 + (sp2y - sp1y)**2)
        
        # Calculate a threshold as a proportion of the segment length
        # This makes it work well for both small and large coordinate values
        threshold = max(1e-5, segment_length * 0.01)  # 1% of segment length as threshold
        
        return abs(cross_product) < threshold

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
        for sign in (-1.0, 1.0):
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
        for sign in (-1.0, 1.0):
            t = (-b + sign * sqrt_disc) / (2 * a)
            if t < -epsilon or t > 1 + epsilon:
                continue
            t = min(max(t, 0.0), 1.0)
            local_x = lx1 + t * dx
            local_y = ly1 + t * dy
            world_x, world_y = to_world(local_x, local_y)
            # Parameter angle on the ellipse before scaling
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
        # Use a small tolerance for floating-point comparisons
        return abs(dot_product) < 1e-10

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
        distances = [MathUtils.get_2D_distance(p1, p2) for i, p1 in enumerate(points) for j, p2 in enumerate(points) if i < j]
        
        # Group similar distances using tolerance
        grouped_distances: List[List[float]] = []
        for d in distances:
            found_group = False
            for group in grouped_distances:
                if abs(group[0] - d) < TOLERANCE:
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
                return str(n_float).rstrip('0').rstrip('.')
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
            A = (cos_a**2/rx**2) + (sin_a**2/ry**2)
            B = 2*cos_a*sin_a*(1/rx**2 - 1/ry**2)
            C = (sin_a**2/rx**2) + (cos_a**2/ry**2)
            
            # Format coefficients to 4 decimal places for readability
            A = round(A, 4)
            B = round(B, 4)
            C = round(C, 4)
            
            # Handle special cases for coefficient signs in the formula
            b_term = f"+ {B}" if B >= 0 else f"- {abs(B)}"
            
            return f"{A}*(x - {fx})**2 {b_term}*(x - {fy})*(y - {fy}) + {C}*(y - {fy})**2 = 1"

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
    def evaluate(expression: str, variables: Optional[Dict[str, Number]] = None) -> Any:
        """Evaluate a mathematical expression numerically with optional variables.
        
        Uses Math.js for evaluation with expression validation and error handling.
        Supports complex mathematical expressions, functions, and variable substitution.
        
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
            
            js_expression = js_expression.replace("arrangements(", "permutations(")

            if not variables:
                result = window.math.format(window.math.evaluate(js_expression))
            else:
                result = window.math.format(window.math.evaluate(js_expression, variables))
            
            converted_result = MathUtils.try_convert_to_number(result)
            
            # Check for division by zero
            if "lim" not in expression and "limit" not in expression and \
                (converted_result == float('-inf') or \
                 converted_result == float('inf') or \
                    str(converted_result).lower() in ['-inf', 'inf', 'infinity', '-infinity']):
                raise ZeroDivisionError()
            
            return converted_result
        except ZeroDivisionError:
            return f"Error: ZeroDivisionError"
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

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
        try:
            return str(window.nerdamer(f"diff({expression}, {variable})").text())
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

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
            str: Limit result as string or error message
        """
        try:
            value_to_approach = str(value_to_approach).lower().replace(' ', '')
            if value_to_approach in ['inf', 'infinity']:
                value_to_approach = 'Infinity'
            elif value_to_approach in ['-inf', '-infinity']:
                value_to_approach = '-Infinity'
            return str(window.nerdamer(f"limit({expression}, {variable}, {value_to_approach})").text())
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

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
        try:
            indefinite_integral = window.nerdamer(f"integrate({expression}, {variable})")
            if lower_bound is None and upper_bound is None:
                return str(indefinite_integral.text())
            evaluated_at_upper = indefinite_integral.sub(variable, upper_bound).text()
            evaluated_at_lower = indefinite_integral.sub(variable, lower_bound).text()
            return str(window.nerdamer(f"{evaluated_at_upper} - {evaluated_at_lower}").evaluate().text())
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

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
        try:
            return str(window.nerdamer(f"simplify({expression})").text())
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

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
        try:
            return str(window.nerdamer(f"expand({expression})").text())
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

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
        try:
            return str(window.nerdamer(f"factor({expression})").text())
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

    @staticmethod
    def get_equation_type(equation: str) -> str:
        import re
        try:
            # Preprocess the equation by expanding it to eliminate parentheses
            expanded_equation = MathUtils.expand(equation)
            # Remove whitespaces for easier processing
            expanded_equation = expanded_equation.replace(' ', '')
            
            # Split into left and right sides if equation contains =
            if '=' in expanded_equation:
                left, right = expanded_equation.split('=')
                # If one side is just 'y', use the other side for analysis
                if left.strip() == 'y':
                    expanded_equation = right
                elif right.strip() == 'y':
                    expanded_equation = left

            # Check for higher order equations (power >= 5)
            # Pattern: x^5, y^6, z^10, etc.
            # Matches: 'x^5', 'y^9', 'x^10', 'y^123'
            # Does not match: 'x^2', 'x^3', 'x^4'
            higher_order_match = re.search(r'\b[a-zA-Z]\^([5-9]|\d{2,})\b', expanded_equation)
            if higher_order_match:
                power = higher_order_match.group(1)
                return f"Order {power}"
            
            # Check for multiple variables
            # Pattern: any letters a-z or A-Z
            # Matches: 'x', 'y', 'X', 'Y'
            variables = set(re.findall(r'[a-zA-Z]', expanded_equation))
            
            # Check for trigonometric equations
            # Pattern: trig function followed by parentheses and content
            # Matches: 'sin(x)', 'cos(2x)', 'tan(x+y)'
            # Does not match: 'sin', 'cos x', 'tan[x]'
            trigonometric_match = re.search(r'\b(sin|cos|tan|csc|sec|cot)\s*\(([^)]+)\)', expanded_equation)
            if trigonometric_match:
                return "Trigonometric"
            
            # Check for non-linear terms with multiple variables
            # Pattern: letter followed optionally by * followed by letter
            # Matches: 'xy', 'x*y', 'x y', 'yx'
            # Does not match: 'x+y', 'x-y'
            if len(variables) > 1 or re.search(r'[a-zA-Z]\s*[*]?\s*[a-zA-Z]', expanded_equation):
                return "Other Non-linear"
            
            # Check for quartic equations
            # Pattern: letter followed by ^4
            # Matches: 'x^4', 'y^4'
            # Does not match: 'x^2', 'x^5', 'x4'
            quartic_match = re.search(r'\b[a-zA-Z]\^4\b', expanded_equation)
            if quartic_match:
                return "Quartic"
            
            # Check for cubic equations
            # Pattern: letter followed by ^3
            # Matches: 'x^3', 'y^3'
            # Does not match: 'x^2', 'x^4', 'x3'
            cubic_match = re.search(r'\b[a-zA-Z]\^3\b', expanded_equation)
            if cubic_match:
                return "Cubic"
            
            # Check for quadratic equations
            # Pattern: letter followed by ^2
            # Matches: 'x^2', 'y^2'
            # Does not match: 'x^3', 'x2', 'x^'
            quadratic_match = re.search(r'\b[a-zA-Z]\^2\b', expanded_equation)
            if quadratic_match:
                return "Quadratic"
            
            # Check for linear equations
            # Pattern: single letter
            # Matches: 'x', 'y' (when not part of another term)
            # Does not match: 'x^2', 'xy', '2'
            linear_match = re.search(r'\b[a-zA-Z]\b', expanded_equation)
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
            type_to_degree = {
                "Linear": 1,
                "Quadratic": 2,
                "Cubic": 3,
                "Quartic": 4
            }

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
        try:
            return str(window.nerdamer(f"solve({equation}, {variable})").text())
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

    @staticmethod
    def solve_linear_system(equations: Sequence[str]) -> str:
        try:
            if len(equations) == 0:
                raise ValueError("The system of equations must contain at least 1 equation.")

            print(f"Attempting to solve a system of linear equations: {equations}")
            # Use nerdamer to solve the system of equations
            solutions = window.nerdamer.solveEquations(equations) # returns [['x', 3], ['y', 1]]
            print(f"Solutions: {solutions}")
            # Prepare the solution dictionary
            solution_dict = {sol[0]: sol[1] for sol in solutions}
            # Convert solution_dict to string format
            solution_strings = [f"{k} = {v}" for k, v in solution_dict.items()]
            return ', '.join(solution_strings)
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

            from expression_validator import ExpressionValidator
            eq1 = MathUtils.expand(equations[0])
            eq1 = ExpressionValidator.fix_math_expression(eq1, python_compatible=False)
            # Split by '=' to separate the left and right sides of the equation and take the side containing the variable
            eq1 = eq1.split('=')[0] if 'x' in eq1.split('=')[0] else eq1.split('=')[1]

            eq2 = MathUtils.expand(equations[1])
            eq2 = ExpressionValidator.fix_math_expression(eq2, python_compatible=False)
            # Split by '=' to separate the left and right sides of the equation and take the side containing the variable
            eq2 = eq2.split('=')[0] if 'x' in eq2.split('=')[0] else eq2.split('=')[1]
            
            linear, quadratic = (eq1, eq2) if "^2" in eq2 else (eq2, eq1)

            system_eq = f"{quadratic} - ({linear})"
            system_eq = MathUtils.expand(system_eq)

            # Extract m, n = coefficients of the linear equation (assuming y = mx + n form)
            lin_coeffs_str = window.nerdamer.coeffs(linear, 'x').text() # The coefficients are placed in the index of their power. So constants are in the 0th place, x^2 would be in the 2nd place, etc. 
            lin_coeffs = literal_eval(lin_coeffs_str)
            m, n = lin_coeffs[1], lin_coeffs[0]
            
            # Extract a, b, c = coefficients of system equation
            quadratic_coeffs_str = window.nerdamer.coeffs(system_eq, 'x').text()
            quadratic_coeffs = literal_eval(quadratic_coeffs_str)
            a, b, c = quadratic_coeffs[2], quadratic_coeffs[1], quadratic_coeffs[0]

            # Solve the quadratic equation of the system
            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                raise ValueError(f"No real solution for the quadratic equation {quadratic}.")

            x1 = (-b + math.sqrt(discriminant)) / (2*a)
            x2 = (-b - math.sqrt(discriminant)) / (2*a)
            
            # If the linear equation is not directly in terms of y, adjust accordingly
            y1 = m*x1 + n
            y2 = m*x2 + n
            
            # Format solutions
            solutions = []
            if discriminant > 0:  # Two solutions
                solutions.append(('x1', x1))
                solutions.append(('y1', y1))
                solutions.append(('x2', x2))
                solutions.append(('y2', y2))
            elif discriminant == 0:  # One solution
                solutions.append(('x', x1))
                solutions.append(('y', y1))
            
            # Prepare the solution dictionary (assuming a single solution format for simplification)
            solution_dict = {sol[0]: sol[1] for sol in solutions}
            
            # Convert solution_dict to string format
            solution_strings = [f"{k} = {v}" for k, v in solution_dict.items()]
            return ', '.join(solution_strings)

        except ValueError as ve:
            raise ve
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

    @staticmethod
    def solve_quadratic_system(equations: Sequence[str]) -> str:
        try:
            from expression_validator import ExpressionValidator

            if len(equations) != 2:
                raise ValueError("The system must contain exactly 2 quadratic equations.")

            print(f"Attempting to solve a system of quadratic equations: {equations}")
            eqs = []
            for equation in equations:
                eq = MathUtils.expand(equation)
                eq = ExpressionValidator.fix_math_expression(eq, python_compatible=False)
                eq = eq.split('=')[0] if 'x' in eq.split('=')[0] else eq.split('=')[1]
                eqs.append(eq)

            # Construct the system equation by setting the equations equal to each other
            system_eq = f"({eqs[0]}) - ({eqs[1]}) = 0"
            system_eq = MathUtils.expand(system_eq)

            # Use nerdamer to solve the system equation for x
            x_solutions_raw = MathUtils.solve(system_eq, 'x')
            x_solutions_data = json.loads(x_solutions_raw)
            x_solutions = [float(r) for r in x_solutions_data]

            solution_dict = {}
            for x_solution in x_solutions:
                # Substitute x_solution into both original equations to find y
                y_equation1 = eqs[0].replace('x', f"({x_solution})")
                y_equation2 = eqs[1].replace('x', f"({x_solution})")

                if 'y' not in y_equation1:
                    y_equation1 += ' = y'
                if 'y' not in y_equation2:
                    y_equation2 += ' = y'

                y1_raw = MathUtils.solve(y_equation1, 'y')
                y2_raw = MathUtils.solve(y_equation2, 'y')

                y1_value: Optional[float] = float(json.loads(y1_raw)[0]) if y1_raw else None
                y2_value: Optional[float] = float(json.loads(y2_raw)[0]) if y2_raw else None

                print(
                    f"Solving for x = {x_solution}: {y_equation1} = {y1_value}, {y_equation2} = {y2_value}"
                )

                if y1_value is not None and y2_value is not None and y1_value == y2_value:
                    solution_dict[x_solution] = y1_value

            solution_strings = [f"(x = {k}, y = {v})" for k, v in solution_dict.items()]
            print(f"Solutions found: {solution_strings}")
            return ', '.join(solution_strings)

        except ValueError as ve:
            raise ve
        except Exception as e:
            return f"Error: {e} {getattr(e, 'message', str(e))}"

    @staticmethod
    def solve_system_of_equations(equations: Sequence[str]) -> str:
        if not isinstance(equations, list) or not equations or not all(isinstance(eq, str) for eq in equations):
            raise ValueError("Invalid input for equations. Expected a list of equations.")
        try:
            # Split single equation strings into two equations
            if len(equations) == 1 and 'x' in equations[0] and '=' in equations[0]:
                eq1, eq2 = equations[0].split('=')
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
                # Return the first solution found by the nerdamer library solver
                print("Solving using nerdamer, returning first solution found")
                solutions = window.nerdamer.solveEquations(equations)
                solution_strings = [f"{solution[0]} = {solution[1]}" for solution in solutions]
                return ', '.join(solution_strings)
        except Exception as e:
            return f"Error: {e}"

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
        import re
        from expression_validator import ExpressionValidator
        
        # Standardize the function string
        function_string = ExpressionValidator.fix_math_expression(function_string)
        vertical_asymptotes: List[float] = []
        
        # For logarithmic functions
        if 'log' in function_string:
            vertical_asymptotes.append(0.0)
            
        # For rational functions
        if '/' in function_string:
            denominator = function_string.split('/')[-1].strip()
            try:
                # Try to solve denominator = 0
                zeros = json.loads(MathUtils.solve(denominator, 'x'))
                vertical_asymptotes.extend(float(x) for x in zeros)
            except:
                pass
                
        # For tangent functions
        if 'tan' in function_string:
            # Find all tangent terms in the function
            tan_matches = re.findall(r'tan\((.*?)(?:\)|$)', function_string)
            for tan_arg in tan_matches:
                coeff = 1.0
                # Check for x/divisor pattern first (e.g., x/100)
                div_match = re.search(r'x\s*/\s*(\d+\.?\d*)', tan_arg)
                if div_match:
                    divisor = float(div_match.group(1))
                    coeff = 1.0 / divisor if divisor != 0 else 1.0
                else:
                    # Check for coefficient*x pattern (e.g., 2*x or 2x)
                    coeff_match = re.search(r'([+-]?\d+\.?\d*)\s*\*?\s*x', tan_arg)
                    if coeff_match:
                        coeff = float(coeff_match.group(1))
                    
                # Get bounds
                left = left_bound if left_bound is not None else -1000
                right = right_bound if right_bound is not None else 1000
                
                # Calculate asymptotes within bounds
                # Asymptotes occur at x = (pi/2 + n*pi)/coeff
                n = math.floor(left * coeff / math.pi - 0.5)
                while True:
                    x = (math.pi/2 + n*math.pi) / coeff
                    if x > right:
                        break
                    if x >= left:
                        vertical_asymptotes.append(x)
                    n += 1
                    
        return sorted(vertical_asymptotes)

    @staticmethod
    def calculate_horizontal_asymptotes(function_string: str) -> List[float]:
        """Calculate horizontal asymptotes of a function"""
        from expression_validator import ExpressionValidator
        
        # Standardize the function string
        function_string = ExpressionValidator.fix_math_expression(function_string)
        horizontal_asymptotes: List[float] = []
        
        try:
            # Check limit as x approaches infinity
            limit_inf = float(MathUtils.limit(function_string, 'x', 'inf'))
            if not math.isinf(limit_inf) and not math.isnan(limit_inf):
                horizontal_asymptotes.append(limit_inf)
        except:
            pass
            
        try:
            # Check limit as x approaches negative infinity
            limit_neg_inf = float(MathUtils.limit(function_string, 'x', '-inf'))
            if not math.isinf(limit_neg_inf) and not math.isnan(limit_neg_inf):
                horizontal_asymptotes.append(limit_neg_inf)
        except:
            pass
            
        return sorted(horizontal_asymptotes)

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
        vertical_asymptotes = MathUtils.calculate_vertical_asymptotes(function_string, left_bound, right_bound)
        horizontal_asymptotes = MathUtils.calculate_horizontal_asymptotes(function_string)
        point_discontinuities = MathUtils.calculate_point_discontinuities(function_string, left_bound, right_bound)
        return vertical_asymptotes, horizontal_asymptotes, point_discontinuities

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
        if any(op in function_string for op in ['if', 'else', '<', '>', '<=', '>=', '==']):
            # Extract transition points from conditions
            # Handle both styles of conditions
            condition_patterns = [
                r'(?:<=|>=|<|>|==)\s*(-?\d*\.?\d+)',  # Mathematical notation
                r'if\s+x\s*(?:<=|>=|<|>|==)\s*(-?\d*\.?\d+)',  # Python if notation
                r'(?:<=|>=|<|>|==)\s*x\s*(?:<=|>=|<|>|==)\s*(-?\d*\.?\d+)'  # Double conditions
            ]
            for pattern in condition_patterns:
                matches = re.findall(pattern, function_string)
                point_discontinuities_set.update(float(x) for x in matches)
        
        # For floor and ceil functions
        if 'floor' in function_string or 'ceil' in function_string:
            # If bounds are provided, check each integer within bounds
            if left_bound is not None and right_bound is not None:
                left = math.ceil(left_bound)
                right = math.floor(right_bound)
                point_discontinuities_set.update(range(left, right + 1))
        
        # For absolute value function at its corners
        if 'abs' in function_string:
            # Extract all arguments of abs functions
            abs_pattern = r'abs\((.*?)\)'
            matches = re.findall(abs_pattern, function_string)
            for match in matches:
                try:
                    # Try to solve the argument = 0 to find the corner point
                    zeros = json.loads(MathUtils.solve(match, 'x'))
                    point_discontinuities_set.update(float(x) for x in zeros)
                except:
                    pass
        
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

        # Find all possible diagonal pairs (points that differ in both x and y)
        potential_diagonals = []
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p1 = points[i]
                p2 = points[j]
                
                # Check if points differ in both x and y coordinates (potential diagonal)
                dx = abs(p1.x - p2.x)
                dy = abs(p1.y - p2.y)
                
                if dx > MathUtils.EPSILON and dy > MathUtils.EPSILON:
                    distance = math.sqrt(dx**2 + dy**2)
                    potential_diagonals.append((p1, p2, distance, dx, dy))
        
        if not potential_diagonals:
            return None, None
            
        # Sort by a combination of factors that make good rectangle diagonals:
        # 1. Prefer more balanced rectangles (closer dx/dy ratio to 1.0)
        # 2. Then by distance as secondary criterion
        def diagonal_score(
            diag_info: Tuple[PointLike, PointLike, float, float, float]
        ) -> Tuple[float, float]:
            p1, p2, distance, dx, dy = diag_info
            # Calculate how balanced the rectangle would be (closer to 1.0 is better)
            aspect_ratio = max(dx, dy) / min(dx, dy) if min(dx, dy) > 0 else float('inf')
            balance_score = 1.0 / aspect_ratio  # Higher score for more balanced rectangles
            # Return tuple for sorting: (balance_score descending, distance descending)
            return (-balance_score, -distance)
        
        # Sort potential diagonals by our scoring criteria
        potential_diagonals.sort(key=diagonal_score)
        
        # Return the best diagonal pair
        best_diagonal = potential_diagonals[0]
        return best_diagonal[0], best_diagonal[1]

    @staticmethod
    def detect_function_periodicity(
        eval_func: Any,
        test_range: float = 20.0,
        probe_count: int = 20,
    ) -> Tuple[bool, Optional[float]]:
        """
        Detect if a function is periodic by probing for oscillations.
        
        Tests the function over a fixed range centered at 0 to detect if
        midpoints deviate from chords, indicating oscillation.
        
        Args:
            eval_func: Function to evaluate y = f(x)
            test_range: Range to test over (centered at 0)
            probe_count: Number of probe segments
            
        Returns:
            Tuple of (is_periodic, estimated_period or None)
        """
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
                if not all(
                    isinstance(v, (int, float)) and math.isfinite(v)
                    for v in [y_left, y_mid, y_right]
                ):
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