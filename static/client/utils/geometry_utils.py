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

from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from itertools import combinations
from .math_utils import MathUtils

if TYPE_CHECKING:
    from drawables.point import Point
    from drawables.segment import Segment


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