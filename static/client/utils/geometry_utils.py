"""
MatHud Geometry Utilities Module

Geometric analysis utilities for shape validation and classification.
Provides functions for analyzing geometric objects and validating shape structures.

Key Features:
    - Point name extraction from segment collections
    - Shape connectivity validation via graph theory
    - Polygon classification (triangle, quadrilateral, etc.)
    - Regularity detection for polygons

Use Cases:
    - Triangle validation and classification (equilateral, isosceles, scalene, right)
    - Rectangle validation (four connected segments in proper topology)
    - Polygon regularity checking
    - Shape completion checking

Dependencies:
    - itertools.combinations: Graph pair analysis
    - utils.math_utils: Segment matching operations
    - utils.graph_utils: Graph theory primitives
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple, Union

from itertools import combinations
from .math_utils import MathUtils
from .graph_utils import Edge, GraphUtils

if TYPE_CHECKING:
    from drawables.point import Point
    from drawables.segment import Segment

PointLike = Union["Point", Tuple[float, float]]


class GeometryUtils:
    """Geometric analysis utilities for shape validation and classification.
    
    Provides static methods for analyzing relationships between geometric objects,
    particularly for validating connectivity and classifying polygon shapes.
    Delegates graph theory operations to GraphUtils.
    """
    
    @staticmethod
    def _segments_to_edges(segments: List["Segment"]) -> List[Edge[str]]:
        """Convert segment objects to Edge objects using point names as vertex identifiers."""
        return [Edge(segment.point1.name, segment.point2.name) for segment in segments]
    
    @staticmethod
    def _build_point_graph(segments: List["Segment"]) -> Dict[str, Set[str]]:
        """
        Build an undirected adjacency map of point names connected by the provided segments.

        Args:
            segments: List of Segment objects

        Returns:
            dict: point name -> connected point names
        """
        edges = GeometryUtils._segments_to_edges(segments)
        return GraphUtils.build_adjacency_map(edges)

    @staticmethod
    def get_unique_point_names_from_segments(segments: List["Segment"]) -> List[str]:
        """
        Extract unique point names from a list of segments.
        
        Args:
            segments: List of Segment objects
            
        Returns:
            list: Sorted list of unique point names
        """
        # Flatten the list of points from each segment and extract the names
        points: List[str] = [point for segment in segments for point in [segment.point1.name, segment.point2.name]]
        # Remove duplicates by converting the list to a set, then convert it back to a sorted list
        unique_points: List[str] = sorted(set(points))
        return unique_points

    @staticmethod
    def is_fully_connected_graph(list_of_point_names: List[str], segments: List["Segment"]) -> bool:
        """
        Check if all points in the list are connected by segments.
        
        Args:
            list_of_point_names: List of point names to check
            segments: List of Segment objects
            
        Returns:
            bool: True if the graph is fully connected, False otherwise
        """
        # Iterate over all pairs of points
        for point_pair in combinations(list_of_point_names, 2):
            # Check if there's a segment connecting the pair
            if not any(MathUtils.segment_matches_point_names(segment, *point_pair) for segment in segments):
                return False
        return True

    # -------------------------------------------------------------------------
    # Closed-loop helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def segments_form_closed_loop(segments: List["Segment"]) -> bool:
        """
        Check if the provided segments form a simple closed loop (polygon).

        A valid loop requires:
            - At least three segments
            - Every vertex has degree two (exactly two incident segments)
            - The graph formed by the segments is connected
        """
        edges = GeometryUtils._segments_to_edges(segments)
        return GraphUtils.is_simple_cycle(edges)

    @staticmethod
    def order_segments_into_loop(segments: List["Segment"]) -> Optional[List["Point"]]:
        """
        Order a set of segments that form a closed loop into a vertex list.

        Returns:
            List of Point objects in traversal order (without repeating the first point)
            or None if the segments cannot be ordered into a valid loop.
        """
        if not GeometryUtils.segments_form_closed_loop(segments):
            return None

        adjacency: Dict[str, List["Segment"]] = {}
        for segment in segments:
            adjacency.setdefault(segment.point1.name, []).append(segment)
            adjacency.setdefault(segment.point2.name, []).append(segment)

        # Deterministic start: choose the segment whose smallest point name is minimal
        def segment_key(segment: "Segment") -> Tuple[str, str]:
            names = sorted([segment.point1.name, segment.point2.name])
            return names[0], names[1]

        ordered_segments = sorted(segments, key=segment_key)
        current_segment = ordered_segments[0]
        current_point = current_segment.point1

        loop_points: List["Point"] = [current_point]
        visited_segments: Set["Segment"] = set()

        while len(visited_segments) < len(segments):
            visited_segments.add(current_segment)
            next_point = current_segment.point2 if current_point is current_segment.point1 else current_segment.point1
            loop_points.append(next_point)
            available = adjacency.get(next_point.name, [])
            next_segment: Optional["Segment"] = None
            for candidate in available:
                if candidate not in visited_segments:
                    next_segment = candidate
                    break
            if next_segment is None:
                break
            current_segment = next_segment
            current_point = next_point

        if len(visited_segments) != len(segments):
            return None

        # The last point should complete the loop; drop the duplicate
        if loop_points[0].name != loop_points[-1].name:
            return None
        return loop_points[:-1]

    @staticmethod
    def polygon_math_coordinates_from_segments(segments: List["Segment"]) -> Optional[List[Tuple[float, float]]]:
        """
        Convenience helper returning ordered math-space coordinates for a closed polygon.
        """
        ordered_points = GeometryUtils.order_segments_into_loop(segments)
        if not ordered_points:
            return None
        return [(float(point.x), float(point.y)) for point in ordered_points]

    # -------------------------------------------------------------------------
    # Convex hull utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _cross_product(
        o: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> float:
        """Cross product of vectors OA and OB.
        
        Positive value indicates counter-clockwise turn from OA to OB.
        Negative value indicates clockwise turn.
        Zero indicates collinear points.
        """
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    @staticmethod
    def convex_hull(
        points: Sequence[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Compute convex hull using Andrew's monotone chain algorithm.
        
        Args:
            points: Sequence of (x, y) coordinate tuples.
            
        Returns:
            List of hull vertices in counter-clockwise order.
            Returns empty list if fewer than 3 unique points after deduplication.
            Returns 2 endpoints for collinear point sets.
        """
        # Remove duplicates and sort lexicographically
        sorted_points = sorted(set(points))
        n = len(sorted_points)
        
        if n < 3:
            return list(sorted_points)
        
        # Build lower hull (left to right)
        lower: List[Tuple[float, float]] = []
        for p in sorted_points:
            while len(lower) >= 2 and GeometryUtils._cross_product(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        # Build upper hull (right to left)
        upper: List[Tuple[float, float]] = []
        for p in reversed(sorted_points):
            while len(upper) >= 2 and GeometryUtils._cross_product(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        # Concatenate, removing duplicate endpoints
        return lower[:-1] + upper[:-1]

    @staticmethod
    def point_in_convex_hull(
        point: Tuple[float, float],
        hull: Sequence[Tuple[float, float]],
    ) -> bool:
        """Check if a point is inside or on the boundary of a convex hull.
        
        Args:
            point: The (x, y) point to test.
            hull: Convex hull vertices in counter-clockwise order.
            
        Returns:
            True if point is inside or on the boundary of the hull.
            False if point is outside or hull has fewer than 3 vertices.
        """
        n = len(hull)
        if n < 3:
            return False
        
        # For a CCW hull, point is inside if it's on the left side (or on)
        # of every edge when traversing CCW
        for i in range(n):
            o = hull[i]
            a = hull[(i + 1) % n]
            cross = GeometryUtils._cross_product(o, a, point)
            if cross < 0:
                return False
        return True

    # -------------------------------------------------------------------------
    # Open path helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def segments_form_open_path(segments: List["Segment"]) -> bool:
        """
        Check if the provided segments form a simple open path (chain).

        A valid open path requires:
            - At least one segment
            - Exactly 2 vertices with degree 1 (the endpoints)
            - All other vertices have degree 2
            - The graph formed by the segments is connected
        """
        edges = GeometryUtils._segments_to_edges(segments)
        return GraphUtils.is_simple_path(edges)

    @staticmethod
    def order_segments_into_path(segments: List["Segment"]) -> Optional[List["Point"]]:
        """
        Order a set of segments that form an open path into a vertex list.

        Returns:
            List of Point objects in traversal order from one endpoint to the other,
            or None if the segments cannot be ordered into a valid path.
        """
        if not GeometryUtils.segments_form_open_path(segments):
            return None

        adjacency: Dict[str, List["Segment"]] = {}
        point_by_name: Dict[str, "Point"] = {}
        for segment in segments:
            adjacency.setdefault(segment.point1.name, []).append(segment)
            adjacency.setdefault(segment.point2.name, []).append(segment)
            point_by_name[segment.point1.name] = segment.point1
            point_by_name[segment.point2.name] = segment.point2

        edges = GeometryUtils._segments_to_edges(segments)
        ordered_names = GraphUtils.order_path_vertices(edges)
        if ordered_names is None:
            return None

        return [point_by_name[name] for name in ordered_names]

    @staticmethod
    def path_math_coordinates_from_segments(segments: List["Segment"]) -> Optional[List[Tuple[float, float]]]:
        """
        Convenience helper returning ordered math-space coordinates for an open path.
        """
        ordered_points = GeometryUtils.order_segments_into_path(segments)
        if not ordered_points:
            return None
        return [(float(point.x), float(point.y)) for point in ordered_points]

    # -------------------------------------------------------------------------
    # General polygon helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def segments_form_polygon(segments: List["Segment"]) -> bool:
        """
        Check whether the provided segments form a simple polygon.

        This reuses the closed-loop validation logic but offers clearer semantics
        to callers that care specifically about polygon construction.
        """
        return GeometryUtils.segments_form_closed_loop(segments)

    @staticmethod
    def _coerce_point_coordinates(point: PointLike) -> Tuple[float, float]:
        if hasattr(point, "x") and hasattr(point, "y"):
            return float(getattr(point, "x")), float(getattr(point, "y"))
        if isinstance(point, (tuple, list)) and len(point) == 2:
            return float(point[0]), float(point[1])
        raise TypeError("Unsupported point representation for polygon analysis.")

    @staticmethod
    def _points_to_coordinates(points: Sequence[PointLike]) -> List[Tuple[float, float]]:
        coords: List[Tuple[float, float]] = [GeometryUtils._coerce_point_coordinates(point) for point in points]
        if len(coords) < 3:
            raise ValueError("At least three points are required to analyze a polygon.")
        return coords

    @staticmethod
    def _polygon_side_lengths(points: Sequence[PointLike]) -> List[float]:
        coords = GeometryUtils._points_to_coordinates(points)
        side_lengths: List[float] = []
        count = len(coords)
        for idx, (x1, y1) in enumerate(coords):
            x2, y2 = coords[(idx + 1) % count]
            side_lengths.append(math.hypot(x2 - x1, y2 - y1))
        return side_lengths

    @staticmethod
    def _polygon_internal_angles(points: Sequence[PointLike]) -> List[float]:
        coords = GeometryUtils._points_to_coordinates(points)
        count = len(coords)
        angles: List[float] = []
        for idx in range(count):
            x_prev, y_prev = coords[idx - 1]
            x_curr, y_curr = coords[idx]
            x_next, y_next = coords[(idx + 1) % count]

            v1x = x_prev - x_curr
            v1y = y_prev - y_curr
            v2x = x_next - x_curr
            v2y = y_next - y_curr

            mag1 = math.hypot(v1x, v1y)
            mag2 = math.hypot(v2x, v2y)
            if mag1 == 0.0 or mag2 == 0.0:
                raise ValueError("Degenerate polygon with overlapping points.")

            dot = v1x * v2x + v1y * v2y
            cross = v1x * v2y - v1y * v2x
            angle = math.degrees(math.atan2(abs(cross), dot))
            angles.append(angle)
        return angles

    @staticmethod
    def _comparison_tolerance(reference: float = 1.0, *, override: Optional[float] = None) -> float:
        if override is not None:
            return float(override)
        scale = max(abs(reference), 1.0)
        return max(MathUtils.EPSILON * scale * 10.0, 1e-6)

    @staticmethod
    def _is_close(value_a: float, value_b: float, *, tolerance: Optional[float] = None) -> bool:
        tol = GeometryUtils._comparison_tolerance(reference=max(abs(value_a), abs(value_b), 1.0), override=tolerance)
        return abs(value_a - value_b) <= tol

    @staticmethod
    def _all_close(values: Sequence[float], *, tolerance: Optional[float] = None) -> bool:
        if not values:
            return True
        first = values[0]
        return all(GeometryUtils._is_close(first, value, tolerance=tolerance) for value in values[1:])

    @staticmethod
    def _has_equal_side_pair(lengths: Sequence[float], *, tolerance: Optional[float] = None) -> bool:
        for idx, length in enumerate(lengths):
            for compare_index in range(idx + 1, len(lengths)):
                if GeometryUtils._is_close(length, lengths[compare_index], tolerance=tolerance):
                    return True
        return False

    @staticmethod
    def _has_right_angle(angles: Sequence[float], *, tolerance: float = 1e-3) -> bool:
        return any(GeometryUtils._is_close(angle, 90.0, tolerance=tolerance) for angle in angles)

    # -------------------------------------------------------------------------
    # Triangle classification helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def triangle_type_flags(points: Sequence[PointLike]) -> Dict[str, bool]:
        if len(points) != 3:
            raise ValueError("Triangle classification requires exactly three points.")
        side_lengths = GeometryUtils._polygon_side_lengths(points)
        angles = GeometryUtils._polygon_internal_angles(points)

        equilateral = GeometryUtils._all_close(side_lengths)
        isosceles = equilateral or GeometryUtils._has_equal_side_pair(side_lengths)
        scalene = not GeometryUtils._has_equal_side_pair(side_lengths)
        right = GeometryUtils._has_right_angle(angles)

        return {
            "equilateral": equilateral,
            "isosceles": isosceles,
            "scalene": scalene,
            "right": right,
        }

    @staticmethod
    def triangle_type_flags_from_segments(segments: List["Segment"]) -> Optional[Dict[str, bool]]:
        points = GeometryUtils.order_segments_into_loop(segments)
        if points is None or len(points) != 3:
            return None
        return GeometryUtils.triangle_type_flags(points)

    @staticmethod
    def is_equilateral_triangle(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.triangle_type_flags(points)["equilateral"]

    @staticmethod
    def is_isosceles_triangle(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.triangle_type_flags(points)["isosceles"]

    @staticmethod
    def is_scalene_triangle(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.triangle_type_flags(points)["scalene"]

    @staticmethod
    def is_right_triangle(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.triangle_type_flags(points)["right"]

    @staticmethod
    def is_equilateral_triangle_from_segments(segments: List["Segment"]) -> bool:
        flags = GeometryUtils.triangle_type_flags_from_segments(segments)
        return bool(flags and flags["equilateral"])

    @staticmethod
    def is_isosceles_triangle_from_segments(segments: List["Segment"]) -> bool:
        flags = GeometryUtils.triangle_type_flags_from_segments(segments)
        return bool(flags and flags["isosceles"])

    @staticmethod
    def is_scalene_triangle_from_segments(segments: List["Segment"]) -> bool:
        flags = GeometryUtils.triangle_type_flags_from_segments(segments)
        return bool(flags and flags["scalene"])

    @staticmethod
    def is_right_triangle_from_segments(segments: List["Segment"]) -> bool:
        flags = GeometryUtils.triangle_type_flags_from_segments(segments)
        return bool(flags and flags["right"])

    # -------------------------------------------------------------------------
    # Quadrilateral classification helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def quadrilateral_type_flags(points: Sequence[PointLike]) -> Dict[str, bool]:
        if len(points) != 4:
            raise ValueError("Quadrilateral classification requires exactly four points.")
        side_lengths = GeometryUtils._polygon_side_lengths(points)
        angles = GeometryUtils._polygon_internal_angles(points)

        all_sides_equal = GeometryUtils._all_close(side_lengths)
        opposite_sides_equal = (
            GeometryUtils._is_close(side_lengths[0], side_lengths[2])
            and GeometryUtils._is_close(side_lengths[1], side_lengths[3])
        )
        right_angles = all(GeometryUtils._is_close(angle, 90.0) for angle in angles)

        square = all_sides_equal and right_angles
        rectangle = right_angles and opposite_sides_equal
        rhombus = all_sides_equal
        irregular = not (square or rectangle or rhombus)

        return {
            "square": square,
            "rectangle": rectangle,
            "rhombus": rhombus,
            "irregular": irregular,
        }

    @staticmethod
    def quadrilateral_type_flags_from_segments(segments: List["Segment"]) -> Optional[Dict[str, bool]]:
        points = GeometryUtils.order_segments_into_loop(segments)
        if points is None or len(points) != 4:
            return None
        return GeometryUtils.quadrilateral_type_flags(points)

    @staticmethod
    def is_square(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.quadrilateral_type_flags(points)["square"]

    @staticmethod
    def is_rectangle(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.quadrilateral_type_flags(points)["rectangle"]

    @staticmethod
    def is_rhombus(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.quadrilateral_type_flags(points)["rhombus"]

    @staticmethod
    def is_irregular_quadrilateral(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.quadrilateral_type_flags(points)["irregular"]

    @staticmethod
    def is_square_from_segments(segments: List["Segment"]) -> bool:
        flags = GeometryUtils.quadrilateral_type_flags_from_segments(segments)
        return bool(flags and flags["square"])

    @staticmethod
    def is_rectangle_from_segments(segments: List["Segment"]) -> bool:
        flags = GeometryUtils.quadrilateral_type_flags_from_segments(segments)
        return bool(flags and flags["rectangle"])

    @staticmethod
    def is_rhombus_from_segments(segments: List["Segment"]) -> bool:
        flags = GeometryUtils.quadrilateral_type_flags_from_segments(segments)
        return bool(flags and flags["rhombus"])

    @staticmethod
    def is_irregular_quadrilateral_from_segments(segments: List["Segment"]) -> bool:
        flags = GeometryUtils.quadrilateral_type_flags_from_segments(segments)
        return bool(flags and flags["irregular"])

    # -------------------------------------------------------------------------
    # General polygon regularity helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def polygon_flags(points: Sequence[PointLike]) -> Dict[str, bool]:
        side_lengths = GeometryUtils._polygon_side_lengths(points)
        angles = GeometryUtils._polygon_internal_angles(points)
        regular = GeometryUtils._all_close(side_lengths) and GeometryUtils._all_close(angles, tolerance=1e-3)
        return {
            "regular": regular,
            "irregular": not regular,
        }

    @staticmethod
    def polygon_side_count(points: Sequence[PointLike]) -> int:
        coords = GeometryUtils._points_to_coordinates(points)
        return len(coords)

    @staticmethod
    def polygon_side_count_from_segments(segments: List["Segment"]) -> Optional[int]:
        points = GeometryUtils.order_segments_into_loop(segments)
        if points is None:
            return None
        return len(points)

    @staticmethod
    def is_polygon_with_sides(points: Sequence[PointLike], expected_sides: int) -> bool:
        return GeometryUtils.polygon_side_count(points) == expected_sides

    @staticmethod
    def is_polygon_with_sides_from_segments(segments: List["Segment"], expected_sides: int) -> bool:
        side_count = GeometryUtils.polygon_side_count_from_segments(segments)
        return side_count == expected_sides if side_count is not None else False

    @staticmethod
    def is_pentagon(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.is_polygon_with_sides(points, 5)

    @staticmethod
    def is_pentagon_from_segments(segments: List["Segment"]) -> bool:
        return GeometryUtils.is_polygon_with_sides_from_segments(segments, 5)

    @staticmethod
    def is_hexagon(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.is_polygon_with_sides(points, 6)

    @staticmethod
    def is_hexagon_from_segments(segments: List["Segment"]) -> bool:
        return GeometryUtils.is_polygon_with_sides_from_segments(segments, 6)

    @staticmethod
    def is_triangle(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.is_polygon_with_sides(points, 3)

    @staticmethod
    def is_triangle_from_segments(segments: List["Segment"]) -> bool:
        return GeometryUtils.is_polygon_with_sides_from_segments(segments, 3)

    @staticmethod
    def is_quadrilateral(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.is_polygon_with_sides(points, 4)

    @staticmethod
    def is_quadrilateral_from_segments(segments: List["Segment"]) -> bool:
        return GeometryUtils.is_polygon_with_sides_from_segments(segments, 4)

    @staticmethod
    def is_heptagon(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.is_polygon_with_sides(points, 7)

    @staticmethod
    def is_heptagon_from_segments(segments: List["Segment"]) -> bool:
        return GeometryUtils.is_polygon_with_sides_from_segments(segments, 7)

    @staticmethod
    def is_octagon(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.is_polygon_with_sides(points, 8)

    @staticmethod
    def is_octagon_from_segments(segments: List["Segment"]) -> bool:
        return GeometryUtils.is_polygon_with_sides_from_segments(segments, 8)

    @staticmethod
    def is_nonagon(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.is_polygon_with_sides(points, 9)

    @staticmethod
    def is_nonagon_from_segments(segments: List["Segment"]) -> bool:
        return GeometryUtils.is_polygon_with_sides_from_segments(segments, 9)

    @staticmethod
    def is_decagon(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.is_polygon_with_sides(points, 10)

    @staticmethod
    def is_decagon_from_segments(segments: List["Segment"]) -> bool:
        return GeometryUtils.is_polygon_with_sides_from_segments(segments, 10)

    @staticmethod
    def polygon_flags_from_segments(segments: List["Segment"]) -> Optional[Dict[str, bool]]:
        points = GeometryUtils.order_segments_into_loop(segments)
        if not points:
            return None
        return GeometryUtils.polygon_flags(points)

    @staticmethod
    def is_regular_polygon(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.polygon_flags(points)["regular"]

    @staticmethod
    def is_irregular_polygon(points: Sequence[PointLike]) -> bool:
        return GeometryUtils.polygon_flags(points)["irregular"]

    @staticmethod
    def is_regular_polygon_from_segments(segments: List["Segment"]) -> bool:
        flags = GeometryUtils.polygon_flags_from_segments(segments)
        return bool(flags and flags["regular"])

    @staticmethod
    def is_irregular_polygon_from_segments(segments: List["Segment"]) -> bool:
        flags = GeometryUtils.polygon_flags_from_segments(segments)
        return bool(flags and flags["irregular"])

    # -------------------------------------------------------------------------
    # Path element intersection utilities
    # -------------------------------------------------------------------------

    INTERSECTION_EPSILON = 1e-9

    @staticmethod
    def _points_equal(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        tol: float = 1e-9
    ) -> bool:
        """Check if two points are equal within tolerance."""
        return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol

    @staticmethod
    def _angle_in_arc_range(
        angle: float,
        start_angle: float,
        end_angle: float,
        clockwise: bool
    ) -> bool:
        """Check if angle is within the arc's angular range."""
        two_pi = 2 * math.pi
        
        arc_span = abs(end_angle - start_angle)
        if arc_span >= two_pi - 1e-9:
            return True
        
        def normalize(a: float) -> float:
            while a < 0:
                a += two_pi
            while a >= two_pi:
                a -= two_pi
            return a
        
        angle = normalize(angle)
        start = normalize(start_angle)
        end = normalize(end_angle)
        
        if not clockwise:
            if start <= end:
                return start <= angle <= end
            else:
                return angle >= start or angle <= end
        else:
            if start >= end:
                return end <= angle <= start
            else:
                return angle <= start or angle >= end

    @staticmethod
    def line_line_intersection(
        seg1_start: Tuple[float, float],
        seg1_end: Tuple[float, float],
        seg2_start: Tuple[float, float],
        seg2_end: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        """
        Find intersection point between two line segments.
        Uses parametric form: P = P1 + t*(P2-P1)
        """
        x1, y1 = seg1_start
        x2, y2 = seg1_end
        x3, y3 = seg2_start
        x4, y4 = seg2_end
        
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3
        
        denom = dx1 * dy2 - dy1 * dx2
        eps = GeometryUtils.INTERSECTION_EPSILON
        
        if abs(denom) < eps:
            return []
        
        dx3 = x3 - x1
        dy3 = y3 - y1
        
        t = (dx3 * dy2 - dy3 * dx2) / denom
        u = (dx3 * dy1 - dy3 * dx1) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            return [(x1 + t * dx1, y1 + t * dy1)]
        
        return []

    @staticmethod
    def line_circle_intersection(
        seg_start: Tuple[float, float],
        seg_end: Tuple[float, float],
        center: Tuple[float, float],
        radius: float,
        start_angle: float,
        end_angle: float,
        clockwise: bool = False
    ) -> List[Tuple[float, float]]:
        """
        Find intersection points between a line segment and a circular arc.
        Delegates to MathUtils for core calculation, then filters by arc range.
        """
        class SegmentAdapter:
            def __init__(self, p1: Tuple[float, float], p2: Tuple[float, float]):
                self.point1 = type('P', (), {'x': p1[0], 'y': p1[1]})()
                self.point2 = type('P', (), {'x': p2[0], 'y': p2[1]})()
        
        segment = SegmentAdapter(seg_start, seg_end)
        raw_intersections = MathUtils.circle_segment_intersections(
            center[0], center[1], radius, segment
        )
        
        results: List[Tuple[float, float]] = []
        for hit in raw_intersections:
            angle = hit['angle']
            if GeometryUtils._angle_in_arc_range(angle, start_angle, end_angle, clockwise):
                results.append((hit['x'], hit['y']))
        
        return results

    @staticmethod
    def line_ellipse_intersection(
        seg_start: Tuple[float, float],
        seg_end: Tuple[float, float],
        center: Tuple[float, float],
        radius_x: float,
        radius_y: float,
        rotation: float,
        start_angle: float,
        end_angle: float,
        clockwise: bool = False
    ) -> List[Tuple[float, float]]:
        """
        Find intersection points between a line segment and an elliptical arc.
        Delegates to MathUtils for core calculation, then filters by arc range.
        """
        class SegmentAdapter:
            def __init__(self, p1: Tuple[float, float], p2: Tuple[float, float]):
                self.point1 = type('P', (), {'x': p1[0], 'y': p1[1]})()
                self.point2 = type('P', (), {'x': p2[0], 'y': p2[1]})()
        
        segment = SegmentAdapter(seg_start, seg_end)
        rotation_degrees = math.degrees(rotation)
        raw_intersections = MathUtils.ellipse_segment_intersections(
            center[0], center[1], radius_x, radius_y, rotation_degrees, segment
        )
        
        results: List[Tuple[float, float]] = []
        for hit in raw_intersections:
            angle = hit['angle']
            if GeometryUtils._angle_in_arc_range(angle, start_angle, end_angle, clockwise):
                results.append((hit['x'], hit['y']))
        
        return results

    @staticmethod
    def circle_circle_intersection(
        center1: Tuple[float, float],
        radius1: float,
        start_angle1: float,
        end_angle1: float,
        clockwise1: bool,
        center2: Tuple[float, float],
        radius2: float,
        start_angle2: float,
        end_angle2: float,
        clockwise2: bool
    ) -> List[Tuple[float, float]]:
        """
        Find intersection points between two circular arcs.
        Uses the radical line method.
        """
        cx1, cy1 = center1
        cx2, cy2 = center2
        eps = GeometryUtils.INTERSECTION_EPSILON
        
        dx = cx2 - cx1
        dy = cy2 - cy1
        d = math.sqrt(dx * dx + dy * dy)
        
        if d < eps:
            return []
        if d > radius1 + radius2 + eps:
            return []
        if d < abs(radius1 - radius2) - eps:
            return []
        
        a = (radius1 * radius1 - radius2 * radius2 + d * d) / (2 * d)
        h_squared = radius1 * radius1 - a * a
        if h_squared < -eps:
            return []
        
        h = math.sqrt(max(0, h_squared))
        px = cx1 + a * dx / d
        py = cy1 + a * dy / d
        
        results: List[Tuple[float, float]] = []
        
        if h < eps:
            point = (px, py)
            angle1 = math.atan2(point[1] - cy1, point[0] - cx1)
            angle2 = math.atan2(point[1] - cy2, point[0] - cx2)
            if (GeometryUtils._angle_in_arc_range(angle1, start_angle1, end_angle1, clockwise1) and
                GeometryUtils._angle_in_arc_range(angle2, start_angle2, end_angle2, clockwise2)):
                results.append(point)
        else:
            offset_x = h * dy / d
            offset_y = h * dx / d
            
            for sign in [1, -1]:
                point = (px + sign * offset_x, py - sign * offset_y)
                angle1 = math.atan2(point[1] - cy1, point[0] - cx1)
                angle2 = math.atan2(point[1] - cy2, point[0] - cx2)
                if (GeometryUtils._angle_in_arc_range(angle1, start_angle1, end_angle1, clockwise1) and
                    GeometryUtils._angle_in_arc_range(angle2, start_angle2, end_angle2, clockwise2)):
                    results.append(point)
        
        return results

    @staticmethod
    def circle_ellipse_intersection(
        circle_center: Tuple[float, float],
        circle_radius: float,
        circle_start: float,
        circle_end: float,
        circle_cw: bool,
        ellipse_center: Tuple[float, float],
        ellipse_rx: float,
        ellipse_ry: float,
        ellipse_rotation: float,
        ellipse_start: float,
        ellipse_end: float,
        ellipse_cw: bool
    ) -> List[Tuple[float, float]]:
        """
        Find intersection points between a circular arc and an elliptical arc.
        Uses numerical sampling approach.
        """
        cx, cy = ellipse_center
        rot = ellipse_rotation
        
        cos_rot = math.cos(-rot)
        sin_rot = math.sin(-rot)
        
        circle_cx_local = (circle_center[0] - cx) * cos_rot - (circle_center[1] - cy) * sin_rot
        circle_cy_local = (circle_center[0] - cx) * sin_rot + (circle_center[1] - cy) * cos_rot
        
        circle_cx_scaled = circle_cx_local / ellipse_rx
        circle_cy_scaled = circle_cy_local / ellipse_ry
        
        results: List[Tuple[float, float]] = []
        num_samples = 360
        
        for i in range(num_samples):
            angle = 2 * math.pi * i / num_samples
            
            ex = math.cos(angle)
            ey = math.sin(angle)
            
            dist_x = (ex - circle_cx_scaled) * ellipse_rx
            dist_y = (ey - circle_cy_scaled) * ellipse_ry
            dist = math.sqrt(dist_x * dist_x + dist_y * dist_y)
            
            if abs(dist - circle_radius) < 0.01:
                cos_rot_inv = math.cos(rot)
                sin_rot_inv = math.sin(rot)
                world_x = (ex * ellipse_rx) * cos_rot_inv - (ey * ellipse_ry) * sin_rot_inv + cx
                world_y = (ex * ellipse_rx) * sin_rot_inv + (ey * ellipse_ry) * cos_rot_inv + cy
                
                circle_angle = math.atan2(world_y - circle_center[1], world_x - circle_center[0])
                
                if (GeometryUtils._angle_in_arc_range(angle, ellipse_start, ellipse_end, ellipse_cw) and
                    GeometryUtils._angle_in_arc_range(circle_angle, circle_start, circle_end, circle_cw)):
                    
                    is_duplicate = any(
                        GeometryUtils._points_equal((world_x, world_y), existing, tol=0.001)
                        for existing in results
                    )
                    if not is_duplicate:
                        results.append((world_x, world_y))
        
        return results

    @staticmethod
    def ellipse_ellipse_intersection(
        center1: Tuple[float, float],
        rx1: float,
        ry1: float,
        rotation1: float,
        start1: float,
        end1: float,
        cw1: bool,
        center2: Tuple[float, float],
        rx2: float,
        ry2: float,
        rotation2: float,
        start2: float,
        end2: float,
        cw2: bool
    ) -> List[Tuple[float, float]]:
        """
        Find intersection points between two elliptical arcs.
        Uses numerical sampling approach.
        """
        results: List[Tuple[float, float]] = []
        num_samples = 360
        
        for i in range(num_samples):
            angle1 = 2 * math.pi * i / num_samples
            
            if not GeometryUtils._angle_in_arc_range(angle1, start1, end1, cw1):
                continue
            
            cos_a = math.cos(angle1)
            sin_a = math.sin(angle1)
            cos_rot1 = math.cos(rotation1)
            sin_rot1 = math.sin(rotation1)
            
            local_x = rx1 * cos_a
            local_y = ry1 * sin_a
            world_x = cos_rot1 * local_x - sin_rot1 * local_y + center1[0]
            world_y = sin_rot1 * local_x + cos_rot1 * local_y + center1[1]
            
            cos_rot2 = math.cos(-rotation2)
            sin_rot2 = math.sin(-rotation2)
            dx = world_x - center2[0]
            dy = world_y - center2[1]
            local2_x = cos_rot2 * dx - sin_rot2 * dy
            local2_y = sin_rot2 * dx + cos_rot2 * dy
            
            normalized = (local2_x / rx2) ** 2 + (local2_y / ry2) ** 2
            
            if abs(normalized - 1.0) < 0.01:
                angle2 = math.atan2(local2_y / ry2, local2_x / rx2)
                
                if GeometryUtils._angle_in_arc_range(angle2, start2, end2, cw2):
                    is_duplicate = any(
                        GeometryUtils._points_equal((world_x, world_y), existing, tol=0.001)
                        for existing in results
                    )
                    if not is_duplicate:
                        results.append((world_x, world_y))
        
        return results

    # -------------------------------------------------------------------------
    # Area calculation utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def polygon_area(points: List[Tuple[float, float]]) -> float:
        """
        Calculate the signed area of a polygon using the shoelace formula.
        
        Positive area indicates counter-clockwise winding.
        Negative area indicates clockwise winding.
        
        Args:
            points: List of (x, y) vertices in order (not closing the loop)
            
        Returns:
            Signed area of the polygon
        """
        if len(points) < 3:
            return 0.0
        
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return area / 2.0

    @staticmethod
    def circular_sector_area(radius: float, angle_span: float) -> float:
        """
        Calculate the area of a circular sector.
        
        Args:
            radius: Circle radius
            angle_span: Angular span in radians (absolute value used)
            
        Returns:
            Area of the sector
        """
        return 0.5 * radius * radius * abs(angle_span)

    @staticmethod
    def circular_segment_area(
        center: Tuple[float, float],
        radius: float,
        start_angle: float,
        end_angle: float,
        clockwise: bool = False
    ) -> float:
        """
        Calculate the signed area contribution of a circular arc segment.
        
        Uses the formula: (1/2) * integral of (x*dy - y*dx) along the arc.
        For a circular arc, this integrates to:
        Area = (r^2/2) * (theta2 - theta1) + (1/2) * (x1*y2 - x2*y1)
        
        The second term accounts for the chord contribution.
        
        Args:
            center: (cx, cy) center of the circle
            radius: Circle radius
            start_angle: Start angle in radians
            end_angle: End angle in radians
            clockwise: Direction of traversal
            
        Returns:
            Signed area contribution (positive for CCW, negative for CW traversal)
        """
        cx, cy = center
        two_pi = 2 * math.pi
        
        start = start_angle % two_pi
        end = end_angle % two_pi
        
        if clockwise:
            if start <= end:
                span = -(two_pi - (end - start))
            else:
                span = -(start - end)
        else:
            if start <= end:
                span = end - start
            else:
                span = two_pi - (start - end)
        
        if abs(span) < GeometryUtils.INTERSECTION_EPSILON:
            span = two_pi if not clockwise else -two_pi
        
        x1 = cx + radius * math.cos(start_angle)
        y1 = cy + radius * math.sin(start_angle)
        x2 = cx + radius * math.cos(end_angle)
        y2 = cy + radius * math.sin(end_angle)
        
        sector_area = 0.5 * radius * radius * span
        
        chord_area = 0.5 * (x1 * y2 - x2 * y1)
        
        return sector_area + chord_area

    @staticmethod
    def elliptical_segment_area(
        center: Tuple[float, float],
        radius_x: float,
        radius_y: float,
        rotation: float,
        start_angle: float,
        end_angle: float,
        clockwise: bool = False
    ) -> float:
        """
        Calculate the signed area contribution of an elliptical arc segment.
        
        Uses numerical integration of (1/2) * (x*dy - y*dx) along the arc.
        
        Args:
            center: (cx, cy) center of the ellipse
            radius_x: Semi-major axis
            radius_y: Semi-minor axis
            rotation: Rotation angle in radians
            start_angle: Start parameter angle in radians
            end_angle: End parameter angle in radians
            clockwise: Direction of traversal
            
        Returns:
            Signed area contribution
        """
        cx, cy = center
        cos_rot = math.cos(rotation)
        sin_rot = math.sin(rotation)
        
        two_pi = 2 * math.pi
        start = start_angle % two_pi
        end = end_angle % two_pi
        
        if clockwise:
            if start <= end:
                span = -(two_pi - (end - start))
            else:
                span = -(start - end)
        else:
            if start <= end:
                span = end - start
            else:
                span = two_pi - (start - end)
        
        if abs(span) < GeometryUtils.INTERSECTION_EPSILON:
            span = two_pi if not clockwise else -two_pi
        
        num_steps = max(100, int(abs(span) * 50))
        dt = span / num_steps
        
        area = 0.0
        t = start_angle
        
        for _ in range(num_steps):
            local_x = radius_x * math.cos(t)
            local_y = radius_y * math.sin(t)
            x = cos_rot * local_x - sin_rot * local_y + cx
            y = sin_rot * local_x + cos_rot * local_y + cy
            
            dx_dt = -radius_x * math.sin(t)
            dy_dt = radius_y * math.cos(t)
            dx = cos_rot * dx_dt - sin_rot * dy_dt
            dy = sin_rot * dx_dt + cos_rot * dy_dt
            
            area += 0.5 * (x * dy - y * dx) * dt
            t += dt
        
        return area

    @staticmethod
    def line_segment_area_contribution(
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> float:
        """
        Calculate the signed area contribution of a line segment.
        
        Uses the formula: (1/2) * (x1*y2 - x2*y1)
        This is the shoelace contribution for one edge.
        
        Args:
            start: (x1, y1) start point
            end: (x2, y2) end point
            
        Returns:
            Signed area contribution
        """
        return 0.5 * (start[0] * end[1] - end[0] * start[1])