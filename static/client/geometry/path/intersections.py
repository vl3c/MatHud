"""
MatHud Intersection Utilities

Path element wrappers for geometric intersection calculations.
Delegates to GeometryUtils for core algorithms.
"""

from __future__ import annotations

from typing import List, Tuple

from .path_element import PathElement
from .line_segment import LineSegment
from .circular_arc import CircularArc
from .elliptical_arc import EllipticalArc
from .composite_path import CompositePath


Point = Tuple[float, float]

_GEOMETRY_UTILS = None


def _get_geometry_utils():
    global _GEOMETRY_UTILS
    if _GEOMETRY_UTILS is None:
        from utils.geometry_utils import GeometryUtils as _GeometryUtils
        _GEOMETRY_UTILS = _GeometryUtils
    return _GEOMETRY_UTILS


def line_line_intersection(seg1: LineSegment, seg2: LineSegment) -> List[Point]:
    """Find intersection point between two line segments."""
    GeometryUtils = _get_geometry_utils()
    return GeometryUtils.line_line_intersection(
        seg1.start_point(), seg1.end_point(),
        seg2.start_point(), seg2.end_point()
    )


def line_circle_intersection(seg: LineSegment, arc: CircularArc) -> List[Point]:
    """Find intersection points between a line segment and a circular arc."""
    GeometryUtils = _get_geometry_utils()
    return GeometryUtils.line_circle_intersection(
        seg.start_point(), seg.end_point(),
        arc.center, arc.radius,
        arc.start_angle, arc.end_angle, arc.clockwise
    )


def line_ellipse_intersection(seg: LineSegment, arc: EllipticalArc) -> List[Point]:
    """Find intersection points between a line segment and an elliptical arc."""
    GeometryUtils = _get_geometry_utils()
    return GeometryUtils.line_ellipse_intersection(
        seg.start_point(), seg.end_point(),
        arc.center, arc.radius_x, arc.radius_y, arc.rotation,
        arc.start_angle, arc.end_angle, arc.clockwise
    )


def circle_circle_intersection(arc1: CircularArc, arc2: CircularArc) -> List[Point]:
    """Find intersection points between two circular arcs."""
    GeometryUtils = _get_geometry_utils()
    return GeometryUtils.circle_circle_intersection(
        arc1.center, arc1.radius, arc1.start_angle, arc1.end_angle, arc1.clockwise,
        arc2.center, arc2.radius, arc2.start_angle, arc2.end_angle, arc2.clockwise
    )


def circle_ellipse_intersection(circle: CircularArc, ellipse: EllipticalArc) -> List[Point]:
    """Find intersection points between a circular arc and an elliptical arc."""
    GeometryUtils = _get_geometry_utils()
    return GeometryUtils.circle_ellipse_intersection(
        circle.center, circle.radius, circle.start_angle, circle.end_angle, circle.clockwise,
        ellipse.center, ellipse.radius_x, ellipse.radius_y, ellipse.rotation,
        ellipse.start_angle, ellipse.end_angle, ellipse.clockwise
    )


def ellipse_ellipse_intersection(arc1: EllipticalArc, arc2: EllipticalArc) -> List[Point]:
    """Find intersection points between two elliptical arcs."""
    GeometryUtils = _get_geometry_utils()
    return GeometryUtils.ellipse_ellipse_intersection(
        arc1.center, arc1.radius_x, arc1.radius_y, arc1.rotation,
        arc1.start_angle, arc1.end_angle, arc1.clockwise,
        arc2.center, arc2.radius_x, arc2.radius_y, arc2.rotation,
        arc2.start_angle, arc2.end_angle, arc2.clockwise
    )


def element_element_intersection(elem1: PathElement, elem2: PathElement) -> List[Point]:
    """Find intersection points between any two path elements."""
    if isinstance(elem1, LineSegment) and isinstance(elem2, LineSegment):
        return line_line_intersection(elem1, elem2)
    elif isinstance(elem1, LineSegment) and isinstance(elem2, CircularArc):
        return line_circle_intersection(elem1, elem2)
    elif isinstance(elem1, CircularArc) and isinstance(elem2, LineSegment):
        return line_circle_intersection(elem2, elem1)
    elif isinstance(elem1, LineSegment) and isinstance(elem2, EllipticalArc):
        return line_ellipse_intersection(elem1, elem2)
    elif isinstance(elem1, EllipticalArc) and isinstance(elem2, LineSegment):
        return line_ellipse_intersection(elem2, elem1)
    elif isinstance(elem1, CircularArc) and isinstance(elem2, CircularArc):
        return circle_circle_intersection(elem1, elem2)
    elif isinstance(elem1, CircularArc) and isinstance(elem2, EllipticalArc):
        return circle_ellipse_intersection(elem1, elem2)
    elif isinstance(elem1, EllipticalArc) and isinstance(elem2, CircularArc):
        return circle_ellipse_intersection(elem2, elem1)
    elif isinstance(elem1, EllipticalArc) and isinstance(elem2, EllipticalArc):
        return ellipse_ellipse_intersection(elem1, elem2)
    else:
        return []


def path_path_intersections(path1: CompositePath, path2: CompositePath) -> List[Point]:
    """Find all intersection points between two composite paths."""
    results: List[Point] = []
    GeometryUtils = _get_geometry_utils()
    
    for elem1 in path1:
        for elem2 in path2:
            points = element_element_intersection(elem1, elem2)
            for point in points:
                is_duplicate = any(
                    GeometryUtils._points_equal(point, existing)
                    for existing in results
                )
                if not is_duplicate:
                    results.append(point)
    
    return results
