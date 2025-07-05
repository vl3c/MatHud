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

from itertools import combinations
from .math_utils import MathUtils

class GeometryUtils:
    """Graph theory and geometric analysis utilities for connectivity validation.
    
    Provides static methods for analyzing relationships between geometric objects,
    particularly for validating connectivity in geometric networks and shapes.
    """
    @staticmethod
    def get_unique_point_names_from_segments(segments):
        """
        Extract unique point names from a list of segments.
        
        Args:
            segments: List of Segment objects
            
        Returns:
            list: Sorted list of unique point names
        """
        # Flatten the list of points from each segment and extract the names
        points = [point for segment in segments for point in [segment.point1.name, segment.point2.name]]
        # Remove duplicates by converting the list to a set, then convert it back to a sorted list
        unique_points = sorted(set(points))
        return unique_points

    @staticmethod
    def is_fully_connected_graph(list_of_point_names, segments):
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