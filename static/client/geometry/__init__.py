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
)

__all__ = [
    'PathElement',
    'LineSegment',
    'CircularArc',
    'EllipticalArc',
    'CompositePath',
]
