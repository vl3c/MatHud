"""
MatHud Geometry Utilities Module

Graph theory and geometric analysis utilities for connectivity and relationship validation.
Provides functions for analyzing connections between geometric objects and validating graph structures.

Key Features:
    - Point name extraction from segment collections
    - Graph connectivity analysis for geometric networks
    - Segment relationship validation
    - Unique identifier management for geometric objects

Graph Theory Operations:
    - Fully connected graph validation
    - Point-to-segment mapping
    - Connectivity analysis for shape construction
    - Network topology validation

Use Cases:
    - Triangle validation (three connected segments)
    - Rectangle validation (four connected segments in proper topology)
    - Shape completion checking
    - Geometric network analysis

Dependencies:
    - itertools.combinations: Graph pair analysis
    - utils.math_utils: Segment matching operations
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple, Union

from itertools import combinations
from .math_utils import MathUtils

if TYPE_CHECKING:
    from drawables.point import Point
    from drawables.segment import Segment

PointLike = Union["Point", Tuple[float, float]]


class GeometryUtils:
    """Graph theory and geometric analysis utilities for connectivity validation.
    
    Provides static methods for analyzing relationships between geometric objects,
    particularly for validating connectivity in geometric networks and shapes.
    """
    @staticmethod
    def _build_point_graph(segments: List["Segment"]) -> Dict[str, Set[str]]:
        """
        Build an undirected adjacency map of point names connected by the provided segments.

        Args:
            segments: List of Segment objects

        Returns:
            dict: point name -> connected point names
        """
        graph: Dict[str, Set[str]] = {}
        for segment in segments:
            p1 = segment.point1.name
            p2 = segment.point2.name
            graph.setdefault(p1, set()).add(p2)
            graph.setdefault(p2, set()).add(p1)
        return graph

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
        if len(segments) < 3:
            return False

        graph = GeometryUtils._build_point_graph(segments)
        if not graph:
            return False

        # All vertices in a simple polygon should have degree two
        for neighbors in graph.values():
            if len(neighbors) != 2:
                return False

        # Ensure the graph is connected
        start = next(iter(graph))
        visited: Set[str] = set()
        stack: List[str] = [start]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(graph[current] - visited)

        return len(visited) == len(graph)

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