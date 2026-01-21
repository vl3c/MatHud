"""Re-export module for polygon canonicalization utilities.

This module provides a convenient import point for all polygon
canonicalization functions and types from the canonicalizers subpackage.

Key Features:
    - Triangle and quadrilateral canonicalizers
    - Polygon subtype enumerations
    - Point conversion utilities
    - Canonicalization error types
"""

from __future__ import annotations

from utils.canonicalizers import (
    PointLike,
    PointTuple,
    PolygonCanonicalizationError,
    QuadrilateralCanonicalizer,
    TriangleCanonicalizer,
    canonicalize_quadrilateral,
    canonicalize_rectangle,
    canonicalize_triangle,
)
from utils.polygon_subtypes import QuadrilateralSubtype, TriangleSubtype

__all__ = [
    "PointLike",
    "PointTuple",
    "PolygonCanonicalizationError",
    "QuadrilateralCanonicalizer",
    "TriangleCanonicalizer",
    "QuadrilateralSubtype",
    "TriangleSubtype",
    "canonicalize_quadrilateral",
    "canonicalize_rectangle",
    "canonicalize_triangle",
]



