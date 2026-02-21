"""
Path subpackage for geometric path elements.

Contains abstract and concrete path element classes for representing
boundaries and composite paths, plus intersection utilities.
"""

from .path_element import PathElement
from .line_segment import LineSegment
from .circular_arc import CircularArc
from .elliptical_arc import EllipticalArc
from .composite_path import CompositePath
from .intersections import (
    line_line_intersection,
    line_circle_intersection,
    line_ellipse_intersection,
    circle_circle_intersection,
    circle_ellipse_intersection,
    ellipse_ellipse_intersection,
    element_element_intersection,
    path_path_intersections,
)

__all__ = [
    "PathElement",
    "LineSegment",
    "CircularArc",
    "EllipticalArc",
    "CompositePath",
    "line_line_intersection",
    "line_circle_intersection",
    "line_ellipse_intersection",
    "circle_circle_intersection",
    "circle_ellipse_intersection",
    "ellipse_ellipse_intersection",
    "element_element_intersection",
    "path_path_intersections",
]
