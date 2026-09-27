"""Re-export module for polygon canonicalization utilities.

This module provides a convenient import point for the rectangle
canonicalizer and its types from the canonicalizers subpackage.

Key Features:
    - Rectangle canonicalizer
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
    canonicalize_rectangle,
)
from utils.polygon_subtypes import QuadrilateralSubtype, TriangleSubtype

__all__ = [
    "PointLike",
    "PointTuple",
    "PolygonCanonicalizationError",
    "QuadrilateralCanonicalizer",
    "QuadrilateralSubtype",
    "TriangleSubtype",
    "canonicalize_rectangle",
]
