"""
Geometry Package

Mathematical geometry abstractions separate from renderable drawables.
These classes represent exact geometric definitions for calculations.
"""

from .path import (
    PathElement,
    LineSegment,
    CircularArc,
    EllipticalArc,
    CompositePath,
    line_line_intersection,
    line_circle_intersection,
    line_ellipse_intersection,
    circle_circle_intersection,
    circle_ellipse_intersection,
    ellipse_ellipse_intersection,
    element_element_intersection,
    path_path_intersections,
)
from .region import Region

__all__ = [
    "PathElement",
    "LineSegment",
    "CircularArc",
    "EllipticalArc",
    "CompositePath",
    "Region",
    "line_line_intersection",
    "line_circle_intersection",
    "line_ellipse_intersection",
    "circle_circle_intersection",
    "circle_ellipse_intersection",
    "ellipse_ellipse_intersection",
    "element_element_intersection",
    "path_path_intersections",
]
