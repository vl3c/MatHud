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

    def __init__(self, outer_boundary: CompositePath, holes: Optional[List[CompositePath]] = None) -> None:
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
                total_area += GeometryUtils.line_segment_area_contribution(element.start_point(), element.end_point())
            elif isinstance(element, CircularArc):
                total_area += GeometryUtils.circular_segment_area(
                    element.center, element.radius, element.start_angle, element.end_angle, element.clockwise
                )
            elif isinstance(element, EllipticalArc):
                total_area += GeometryUtils.elliptical_segment_area(
                    element.center,
                    element.radius_x,
                    element.radius_y,
                    element.rotation,
                    element.start_angle,
                    element.end_angle,
                    element.clockwise,
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

    def _point_in_polygon(self, x: float, y: float, points: List[Tuple[float, float]]) -> bool:
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
        cls, center: Tuple[float, float], radius_x: float, radius_y: float, rotation: float = 0.0
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
    def from_half_plane(cls, point1: Tuple[float, float], point2: Tuple[float, float], size: float = 10000.0) -> Region:
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
        p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], p4: Tuple[float, float]
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
        point: Tuple[float, float], edge_start: Tuple[float, float], edge_end: Tuple[float, float]
    ) -> bool:
        """Check if point is on the inside (left) of a directed edge."""
        return (
            (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
            - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
        ) >= 0

    def _sutherland_hodgman_clip(
        self, subject: List[Tuple[float, float]], clip: List[Tuple[float, float]]
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
                        intersection = self._line_intersection(previous, current, edge_start, edge_end)
                        if intersection:
                            output.append(intersection)
                    output.append(current)
                elif previous_inside:
                    intersection = self._line_intersection(previous, current, edge_start, edge_end)
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

    @staticmethod
    def _is_convex(points: List[Tuple[float, float]]) -> bool:
        """Check whether a CCW polygon is convex (collinear vertices allowed)."""
        # Drop consecutive duplicates (e.g. a repeated closing point) so that
        # zero-length edges cannot hide a reflex turn.
        pts: List[Tuple[float, float]] = []
        for p in points:
            if not pts or abs(p[0] - pts[-1][0]) > 1e-12 or abs(p[1] - pts[-1][1]) > 1e-12:
                pts.append(p)
        if len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) <= 1e-12 and abs(pts[0][1] - pts[-1][1]) <= 1e-12:
            pts.pop()

        n = len(pts)
        if n < 4:
            return True
        for i in range(n):
            ax = pts[(i + 1) % n][0] - pts[i][0]
            ay = pts[(i + 1) % n][1] - pts[i][1]
            bx = pts[(i + 2) % n][0] - pts[(i + 1) % n][0]
            by = pts[(i + 2) % n][1] - pts[(i + 1) % n][1]
            cross = ax * by - ay * bx
            # Relative tolerance so nearly collinear sampled points still count as convex
            if cross < -1e-9 * math.hypot(ax, ay) * math.hypot(bx, by):
                return False
        return True

    # A region is also a signed sum of simple regions (no holes): its indicator is
    # sum(coefficient * indicator(term)). A plain region is its outer boundary (+1)
    # minus each hole (-1), and A & B is the sum over term pairs of the products.
    # Areas add up the same way, so nested operations keep every hole.

    def _signed_terms(self) -> List[Tuple[int, Region]]:
        """This region as (coefficient, simple region) terms."""
        terms: List[Tuple[int, Region]] = [(1, Region(self._outer_boundary))]
        terms.extend((-1, Region(hole)) for hole in self._holes)
        return terms

    def _clip_pair(self, first: Region, second: Region, num_samples: int) -> Optional[Region]:
        """Intersect two simple regions by Sutherland-Hodgman clipping of their sampled boundaries.

        The clip polygon must be convex, so a convex operand is used as the clip polygon
        (intersection is commutative).
        """
        first_points = first._sample_to_points(num_samples)
        second_points = second._sample_to_points(num_samples)

        if len(first_points) < 3 or len(second_points) < 3:
            return None

        # Ensure both polygons are CCW for correct clipping
        first_points = self._ensure_ccw(first_points)
        second_points = self._ensure_ccw(second_points)

        if not self._is_convex(second_points):
            if not self._is_convex(first_points):
                raise ValueError("Boolean operations on two non-convex regions are not supported")
            first_points, second_points = second_points, first_points

        clipped = self._sutherland_hodgman_clip(first_points, second_points)

        if len(clipped) < 3:
            return None

        return Region.from_points(clipped)

    def _intersection_terms(self, other: Region, num_samples: int) -> List[Tuple[int, Region]]:
        terms: List[Tuple[int, Region]] = []
        for coefficient, term in self._signed_terms():
            for other_coefficient, other_term in other._signed_terms():
                clipped = self._clip_pair(term, other_term, num_samples)
                if clipped is not None:
                    terms.append((coefficient * other_coefficient, clipped))
        return terms

    @staticmethod
    def _from_terms(terms: List[Tuple[int, Region]]) -> Optional[Region]:
        """The region for a signed sum of terms: a plain region when that is all it is."""
        if not terms:
            return None
        if len(terms) == 1 and terms[0][0] == 1:
            return terms[0][1]
        return _SignedRegion(terms)

    @staticmethod
    def _scaled(terms: List[Tuple[int, Region]], factor: int) -> List[Tuple[int, Region]]:
        return [(coefficient * factor, term) for coefficient, term in terms]

    def intersection(self, other: Region, num_samples: int = 100) -> Optional[Region]:
        """Compute the intersection of this region with another.

        Each pair of terms (outer boundaries and holes) is clipped with Sutherland-Hodgman
        on sampled boundaries; of each pair, at least one must be convex.

        Args:
            other: Another Region to intersect with
            num_samples: Number of sample points for boundary approximation

        Returns:
            A new Region representing the intersection, or None if empty

        Raises:
            ValueError: If two terms are both non-convex (not supported)
        """
        return self._from_terms(self._intersection_terms(other, num_samples))

    def union(self, other: Region, num_samples: int = 100) -> "Region":
        """Compute the union of this region with another: A + B - (A & B).

        Args:
            other: Another Region to union with
            num_samples: Number of sample points for boundary approximation

        Returns:
            A Region representing the union
        """
        overlap = self._intersection_terms(other, num_samples)
        terms = self._signed_terms() + other._signed_terms() + self._scaled(overlap, -1)
        return _SignedRegion(terms)

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
        """Compute the difference of this region minus another: A - (A & B).

        A plain region minus one overlapping plain region keeps the overlap as a hole.

        Args:
            other: Region to subtract from this region
            num_samples: Number of sample points for boundary approximation

        Returns:
            A new Region for the part of this region outside the other, or a copy of
            this region if they do not overlap
        """
        overlap = self._intersection_terms(other, num_samples)

        if not overlap:
            return self._copy()

        if self._is_plain() and not self._holes and len(overlap) == 1 and overlap[0][0] == 1:
            return Region(self._outer_boundary, [overlap[0][1]._outer_boundary])

        return _SignedRegion(self._signed_terms() + self._scaled(overlap, -1))

    def symmetric_difference(self, other: Region, num_samples: int = 100) -> "Region":
        """Compute the symmetric difference of this region with another: A + B - 2 (A & B).

        Args:
            other: Another Region
            num_samples: Number of sample points for boundary approximation

        Returns:
            A Region representing the symmetric difference
        """
        overlap = self._intersection_terms(other, num_samples)
        terms = self._signed_terms() + other._signed_terms() + self._scaled(overlap, -2)
        return _SignedRegion(terms)

    def _is_plain(self) -> bool:
        """True for a region given by one outer boundary and its holes."""
        return True

    def _copy(self) -> Region:
        return Region(self._outer_boundary, list(self._holes))

    def boundary_points(self, num_samples: int = 100) -> List[Tuple[float, float]]:
        """One closed point path whose fill (nonzero or even-odd rule) is exactly this region.

        Each term's ring runs counterclockwise for a positive coefficient and clockwise for
        a negative one, repeated |coefficient| times, so the winding number at any point is
        the region's indicator. The rings are joined through the first ring's first point;
        each bridge is walked once in each direction and cancels out.
        """
        rings: List[List[Tuple[float, float]]] = []
        for coefficient, term in self._signed_terms():
            ring = self._ensure_ccw(term._sample_to_points(num_samples))
            if len(ring) < 3:
                continue
            if coefficient < 0:
                ring = list(reversed(ring))
            rings.extend([ring] * abs(coefficient))
        if not rings:
            return []
        anchor = rings[0][0]
        path: List[Tuple[float, float]] = list(rings[0]) + [anchor]
        for ring in rings[1:]:
            path.extend(ring)
            path.append(ring[0])
            path.append(anchor)
        return path


class _SignedRegion(Region):
    """A region given as a signed sum of simple regions (see ``Region._signed_terms``).

    Union, symmetric difference and differences of regions with holes produce one; it
    combines further like any region, holes included.
    """

    def __init__(self, terms: List[Tuple[int, Region]]) -> None:
        self._terms: List[Tuple[int, Region]] = [(c, t) for c, t in terms if c != 0]
        positive = [t for c, t in self._terms if c > 0]
        source = positive[0] if positive else (self._terms[0][1] if self._terms else None)
        self._outer_boundary = source._outer_boundary if source is not None else CompositePath([])
        self._holes: List[CompositePath] = []

    def _signed_terms(self) -> List[Tuple[int, Region]]:
        return list(self._terms)

    def _is_plain(self) -> bool:
        return False

    def _copy(self) -> Region:
        return _SignedRegion(self._terms)

    def area(self) -> float:
        """Sum of each term's area times its coefficient."""
        return sum(coefficient * term.area() for coefficient, term in self._terms)

    def signed_area(self) -> float:
        return self.area()

    def contains_point(self, x: float, y: float) -> bool:
        """Check if the terms containing the point add up to at least one."""
        return sum(coefficient for coefficient, term in self._terms if term.contains_point(x, y)) > 0

    def _sample_to_points(self, num_samples: int = 100) -> List[Tuple[float, float]]:
        return self.boundary_points(num_samples)

    def __repr__(self) -> str:
        return f"Region(terms={len(self._terms)})"
