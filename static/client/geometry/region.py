"""
MatHud Region Class

Represents a 2D region bounded by a closed path, optionally with holes.
Supports area calculation and point containment testing.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

from .path import (
    LineSegment,
    CircularArc,
    EllipticalArc,
    CompositePath,
)

_GEOMETRY_UTILS = None


def _get_geometry_utils():
    global _GEOMETRY_UTILS
    if _GEOMETRY_UTILS is None:
        from utils.geometry_utils import GeometryUtils as _GeometryUtils
        _GEOMETRY_UTILS = _GeometryUtils
    return _GEOMETRY_UTILS


class Region:
    """A 2D region defined by an outer boundary and optional holes.
    
    The outer boundary must be a closed CompositePath traversed counter-clockwise
    for positive area. Holes are closed CompositePaths traversed clockwise.
    
    Attributes:
        _outer_boundary: The outer closed path
        _holes: List of inner closed paths (holes)
    """
    
    def __init__(
        self,
        outer_boundary: CompositePath,
        holes: Optional[List[CompositePath]] = None
    ) -> None:
        """
        Create a Region from an outer boundary and optional holes.
        
        Args:
            outer_boundary: A closed CompositePath defining the outer edge
            holes: Optional list of closed CompositePaths defining holes
            
        Raises:
            ValueError: If outer_boundary is not closed
        """
        if not outer_boundary.is_closed():
            raise ValueError("Outer boundary must be a closed path")
        
        self._outer_boundary = outer_boundary
        self._holes: List[CompositePath] = []
        
        if holes:
            for hole in holes:
                if not hole.is_closed():
                    raise ValueError("All holes must be closed paths")
                self._holes.append(hole)
    
    @property
    def outer_boundary(self) -> CompositePath:
        """Get the outer boundary path."""
        return self._outer_boundary
    
    @property
    def holes(self) -> List[CompositePath]:
        """Get the list of hole paths."""
        return list(self._holes)
    
    def add_hole(self, hole: CompositePath) -> None:
        """Add a hole to the region.
        
        Args:
            hole: A closed CompositePath defining the hole
            
        Raises:
            ValueError: If hole is not closed
        """
        if not hole.is_closed():
            raise ValueError("Hole must be a closed path")
        self._holes.append(hole)
    
    def _path_area(self, path: CompositePath) -> float:
        """Calculate the signed area enclosed by a path using Green's theorem."""
        total_area = 0.0
        GeometryUtils = _get_geometry_utils()
        
        for element in path:
            if isinstance(element, LineSegment):
                total_area += GeometryUtils.line_segment_area_contribution(
                    element.start_point(),
                    element.end_point()
                )
            elif isinstance(element, CircularArc):
                total_area += GeometryUtils.circular_segment_area(
                    element.center,
                    element.radius,
                    element.start_angle,
                    element.end_angle,
                    element.clockwise
                )
            elif isinstance(element, EllipticalArc):
                total_area += GeometryUtils.elliptical_segment_area(
                    element.center,
                    element.radius_x,
                    element.radius_y,
                    element.rotation,
                    element.start_angle,
                    element.end_angle,
                    element.clockwise
                )
        
        return total_area
    
    def area(self) -> float:
        """Calculate the total area of the region.
        
        Returns the absolute area of the outer boundary minus the absolute
        areas of all holes.
        
        Returns:
            The area of the region
        """
        outer_area = abs(self._path_area(self._outer_boundary))
        
        hole_area = sum(abs(self._path_area(hole)) for hole in self._holes)
        
        return outer_area - hole_area
    
    def signed_area(self) -> float:
        """Calculate the signed area of the region.
        
        Positive for counter-clockwise outer boundary, negative for clockwise.
        Holes subtract from the total.
        
        Returns:
            The signed area of the region
        """
        outer_area = self._path_area(self._outer_boundary)
        
        hole_area = sum(self._path_area(hole) for hole in self._holes)
        
        return outer_area - hole_area
    
    def _point_in_polygon(
        self,
        x: float,
        y: float,
        points: List[Tuple[float, float]]
    ) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(points)
        if n < 3:
            return False
        
        inside = False
        j = n - 1
        
        for i in range(n):
            xi, yi = points[i]
            xj, yj = points[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def contains_point(self, x: float, y: float) -> bool:
        """Test if a point is inside the region.
        
        Uses ray casting algorithm on sampled boundary points.
        Point must be inside outer boundary and outside all holes.
        
        Args:
            x: X coordinate of point
            y: Y coordinate of point
            
        Returns:
            True if point is inside the region
        """
        outer_points = self._outer_boundary.sample(100)
        
        if not self._point_in_polygon(x, y, outer_points):
            return False
        
        for hole in self._holes:
            hole_points = hole.sample(100)
            if self._point_in_polygon(x, y, hole_points):
                return False
        
        return True
    
    @classmethod
    def from_polygon(cls, polygon: Any) -> Region:
        """Create a Region from a polygon drawable.
        
        Args:
            polygon: A polygon drawable with get_segments() method
            
        Returns:
            A Region representing the polygon's area
        """
        path = CompositePath.from_polygon(polygon)
        return cls(path)
    
    @classmethod
    def from_points(cls, points: List[Tuple[float, float]]) -> Region:
        """Create a Region from a list of points forming a closed polygon.
        
        Args:
            points: List of (x, y) coordinates. The path will be closed
                   automatically if the first and last points differ.
            
        Returns:
            A Region representing the polygon's area
        """
        if len(points) < 3:
            raise ValueError("At least 3 points required to form a region")
        
        pts = list(points)
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        
        path = CompositePath.from_points(pts)
        return cls(path)
    
    @classmethod
    def from_circle(cls, center: Tuple[float, float], radius: float) -> Region:
        """Create a Region from a circle.
        
        Args:
            center: (x, y) center of the circle
            radius: Radius of the circle
            
        Returns:
            A Region representing the circle's area
        """
        arc = CircularArc(center, radius, 0.0, 2 * math.pi)
        path = CompositePath([arc])
        return cls(path)
    
    @classmethod
    def from_ellipse(
        cls,
        center: Tuple[float, float],
        radius_x: float,
        radius_y: float,
        rotation: float = 0.0
    ) -> Region:
        """Create a Region from an ellipse.
        
        Args:
            center: (x, y) center of the ellipse
            radius_x: Semi-major axis
            radius_y: Semi-minor axis
            rotation: Rotation angle in radians
            
        Returns:
            A Region representing the ellipse's area
        """
        arc = EllipticalArc(center, radius_x, radius_y, 0.0, 2 * math.pi, rotation=rotation)
        path = CompositePath([arc])
        return cls(path)
    
    @classmethod
    def from_half_plane(
        cls,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        size: float = 10000.0
    ) -> Region:
        """Create a Region representing a half-plane bounded by a line.
        
        The half-plane is the area to the LEFT of the directed line from point1 to point2.
        To get the other half, swap the points.
        
        Args:
            point1: First point on the boundary line (x, y)
            point2: Second point on the boundary line (x, y)
            size: Size of the bounding box (should be larger than any shape it intersects)
            
        Returns:
            A Region representing the half-plane
        """
        x1, y1 = point1
        x2, y2 = point2
        
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        
        if length < 1e-10:
            raise ValueError("Points must be distinct to define a half-plane")
        
        dx /= length
        dy /= length
        
        nx, ny = -dy, dx
        
        ext1 = (x1 - dx * size, y1 - dy * size)
        ext2 = (x2 + dx * size, y2 + dy * size)
        
        far1 = (ext1[0] + nx * size, ext1[1] + ny * size)
        far2 = (ext2[0] + nx * size, ext2[1] + ny * size)
        
        points = [ext1, ext2, far2, far1]
        
        return cls.from_points(points)
    
    def __repr__(self) -> str:
        return f"Region(boundary_elements={len(self._outer_boundary)}, holes={len(self._holes)})"

    # -------------------------------------------------------------------------
    # Boolean operations
    # -------------------------------------------------------------------------

    def _sample_to_points(self, num_samples: int = 100) -> List[Tuple[float, float]]:
        """Sample the outer boundary to a list of points."""
        return self._outer_boundary.sample(num_samples)

    @staticmethod
    def _line_intersection(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Find intersection of lines (p1-p2) and (p3-p4)."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)

    @staticmethod
    def _is_inside_edge(
        point: Tuple[float, float],
        edge_start: Tuple[float, float],
        edge_end: Tuple[float, float]
    ) -> bool:
        """Check if point is on the inside (left) of a directed edge."""
        return ((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) -
                (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])) >= 0

    def _sutherland_hodgman_clip(
        self,
        subject: List[Tuple[float, float]],
        clip: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Clip subject polygon against clip polygon using Sutherland-Hodgman."""
        if len(subject) < 3 or len(clip) < 3:
            return []
        
        output = list(subject)
        
        for i in range(len(clip)):
            if len(output) == 0:
                break
            
            input_list = output
            output = []
            
            edge_start = clip[i]
            edge_end = clip[(i + 1) % len(clip)]
            
            for j in range(len(input_list)):
                current = input_list[j]
                previous = input_list[j - 1]
                
                current_inside = self._is_inside_edge(current, edge_start, edge_end)
                previous_inside = self._is_inside_edge(previous, edge_start, edge_end)
                
                if current_inside:
                    if not previous_inside:
                        intersection = self._line_intersection(
                            previous, current, edge_start, edge_end
                        )
                        if intersection:
                            output.append(intersection)
                    output.append(current)
                elif previous_inside:
                    intersection = self._line_intersection(
                        previous, current, edge_start, edge_end
                    )
                    if intersection:
                        output.append(intersection)
        
        return output

    @staticmethod
    def _ensure_ccw(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Ensure polygon points are in counterclockwise order.
        
        Sutherland-Hodgman requires consistent CCW winding for both polygons.
        """
        if len(points) < 3:
            return points
        
        # Calculate signed area using shoelace formula
        signed_area = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            signed_area += points[i][0] * points[j][1]
            signed_area -= points[j][0] * points[i][1]
        
        # If clockwise (negative area), reverse to make CCW
        if signed_area < 0:
            return list(reversed(points))
        return points

    def intersection(self, other: Region, num_samples: int = 100) -> Optional[Region]:
        """Compute the intersection of this region with another.
        
        Uses Sutherland-Hodgman polygon clipping on sampled boundaries.
        
        Args:
            other: Another Region to intersect with
            num_samples: Number of sample points for boundary approximation
            
        Returns:
            A new Region representing the intersection, or None if empty
        """
        self_points = self._sample_to_points(num_samples)
        other_points = other._sample_to_points(num_samples)
        
        if len(self_points) < 3 or len(other_points) < 3:
            return None
        
        # Ensure both polygons are CCW for correct clipping
        self_points = self._ensure_ccw(self_points)
        other_points = self._ensure_ccw(other_points)
        
        clipped = self._sutherland_hodgman_clip(self_points, other_points)
        
        if len(clipped) < 3:
            return None
        
        return Region.from_points(clipped)

    def union(self, other: Region, num_samples: int = 100) -> "Region":
        """Compute the union of this region with another.
        
        Uses area formula: union = A + B - intersection.
        Returns a CompositeRegion that computes area correctly.
        
        Args:
            other: Another Region to union with
            num_samples: Number of sample points for boundary approximation
            
        Returns:
            A Region representing the union
        """
        intersection = self.intersection(other, num_samples)
        return _CompositeRegion([self, other], intersection)

    @staticmethod
    def _convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Compute convex hull using Graham scan."""
        if len(points) < 3:
            return list(points)
        
        def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        sorted_points = sorted(set(points))
        
        if len(sorted_points) < 3:
            return sorted_points
        
        lower: List[Tuple[float, float]] = []
        for p in sorted_points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        upper: List[Tuple[float, float]] = []
        for p in reversed(sorted_points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        return lower[:-1] + upper[:-1]

    def difference(self, other: Region, num_samples: int = 100) -> Optional[Region]:
        """Compute the difference of this region minus another.
        
        Returns the area of this region that is not in the other region.
        Uses intersection as hole to correctly handle partial overlaps.
        
        Args:
            other: Region to subtract from this region
            num_samples: Number of sample points for boundary approximation
            
        Returns:
            A new Region with the intersection as a hole, or self if no overlap
        """
        inter = self.intersection(other, num_samples)
        
        if inter is None:
            return Region(self._outer_boundary, list(self._holes))
        
        new_holes = list(self._holes)
        new_holes.append(inter._outer_boundary)
        
        return Region(self._outer_boundary, new_holes)
    
    def symmetric_difference(self, other: Region, num_samples: int = 100) -> "Region":
        """Compute the symmetric difference of this region with another.
        
        Returns the area in either region but not in both.
        Uses formula: A ^ B = A + B - 2 * (A & B)
        
        Args:
            other: Another Region
            num_samples: Number of sample points for boundary approximation
            
        Returns:
            A Region representing the symmetric difference
        """
        inter = self.intersection(other, num_samples)
        return _SymmetricDifferenceRegion(self, other, inter)


class _CompositeRegion(Region):
    """A composite region representing the union of two regions.
    
    This is used internally to correctly compute union area using the
    formula: area(A | B) = area(A) + area(B) - area(A & B)
    """
    
    def __init__(self, regions: List[Region], intersection: Optional[Region]) -> None:
        self._regions = regions
        self._intersection = intersection
        self._outer_boundary = regions[0]._outer_boundary if regions else CompositePath([])
        self._holes: List[CompositePath] = []
    
    def area(self) -> float:
        """Calculate union area as sum of regions minus intersection."""
        total = sum(r.area() for r in self._regions)
        if self._intersection is not None:
            total -= self._intersection.area()
        return total
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is in any of the regions."""
        return any(r.contains_point(x, y) for r in self._regions)


class _SymmetricDifferenceRegion(Region):
    """A region representing the symmetric difference of two regions.
    
    Uses formula: area(A ^ B) = area(A) + area(B) - 2 * area(A & B)
    """
    
    def __init__(self, region1: Region, region2: Region, intersection: Optional[Region]) -> None:
        self._region1 = region1
        self._region2 = region2
        self._intersection = intersection
        self._outer_boundary = region1._outer_boundary
        self._holes: List[CompositePath] = []
    
    def area(self) -> float:
        """Calculate symmetric difference area."""
        total = self._region1.area() + self._region2.area()
        if self._intersection is not None:
            total -= 2 * self._intersection.area()
        return total
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is in exactly one of the regions."""
        in1 = self._region1.contains_point(x, y)
        in2 = self._region2.contains_point(x, y)
        return in1 != in2
