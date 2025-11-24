from __future__ import annotations

from .common import (
    PointLike,
    PointTuple,
    PolygonCanonicalizationError,
    contains_point,
    nearest_point,
    point_like_to_tuple,
)
from .rectangle import RectangleCanonicalizer, canonicalize_rectangle
from .triangle import TriangleCanonicalizer, canonicalize_triangle

__all__ = [
    "PointLike",
    "PointTuple",
    "PolygonCanonicalizationError",
    "point_like_to_tuple",
    "contains_point",
    "nearest_point",
    "RectangleCanonicalizer",
    "TriangleCanonicalizer",
    "canonicalize_rectangle",
    "canonicalize_triangle",
]



