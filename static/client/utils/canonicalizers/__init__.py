"""Polygon canonicalization utilities for vertex normalization.

This package provides canonicalization routines that transform arbitrary
vertex lists into well-formed polygon representations.

Key Features:
    - Triangle canonicalization with subtype support (equilateral, isosceles, right)
    - Quadrilateral canonicalization with subtype support (rectangle, square, parallelogram)
    - Point deduplication and CCW ordering
    - Best-fit algorithms preserving user-specified anchors
"""

from __future__ import annotations

from .common import (
    PointLike,
    PointTuple,
    PolygonCanonicalizationError,
    contains_point,
    nearest_point,
    point_like_to_tuple,
)
from .quadrilateral import (
    QuadrilateralCanonicalizer,
    canonicalize_quadrilateral,
    canonicalize_rectangle,
)
from .triangle import TriangleCanonicalizer, canonicalize_triangle

__all__ = [
    "PointLike",
    "PointTuple",
    "PolygonCanonicalizationError",
    "point_like_to_tuple",
    "contains_point",
    "nearest_point",
    "QuadrilateralCanonicalizer",
    "TriangleCanonicalizer",
    "canonicalize_quadrilateral",
    "canonicalize_rectangle",
    "canonicalize_triangle",
]



