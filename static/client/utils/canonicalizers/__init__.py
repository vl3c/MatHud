"""Polygon canonicalization utilities for vertex normalization.

This package builds rectangle corners from a diagonal pair or a noisy vertex list.
create_polygon does not use it: it keeps the given vertices (utils/polygon_subtype_checks.py).

Key Features:
    - Rectangle and square canonicalization
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
from .quadrilateral import QuadrilateralCanonicalizer, canonicalize_rectangle

__all__ = [
    "PointLike",
    "PointTuple",
    "PolygonCanonicalizationError",
    "point_like_to_tuple",
    "contains_point",
    "nearest_point",
    "QuadrilateralCanonicalizer",
    "canonicalize_rectangle",
]
