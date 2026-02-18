"""Common utilities for polygon canonicalization.

This module provides shared types and helper functions used by the
triangle and quadrilateral canonicalization routines.

Key Features:
    - PointLike type union for flexible vertex input
    - Point conversion from tuples, lists, dicts, and objects
    - Proximity-based point deduplication helpers
    - Nearest point search for vertex matching
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple, Union

PointTuple = Tuple[float, float]
PointLike = Union[PointTuple, Sequence[float], object]


class PolygonCanonicalizationError(ValueError):
    """Raised when provided vertices cannot be coerced into the requested polygon."""


def point_like_to_tuple(point: PointLike) -> PointTuple:
    """Convert various point representations into a float tuple."""
    if isinstance(point, tuple):
        if len(point) != 2:
            raise PolygonCanonicalizationError("Each vertex tuple must have 2 elements.")
        return float(point[0]), float(point[1])
    if isinstance(point, list):
        if len(point) != 2:
            raise PolygonCanonicalizationError("Each vertex list must have 2 elements.")
        return float(point[0]), float(point[1])
    if isinstance(point, dict):
        try:
            return float(point["x"]), float(point["y"])
        except KeyError as exc:
            raise PolygonCanonicalizationError("Dictionary vertices must provide x and y keys.") from exc
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(getattr(point, "x")), float(getattr(point, "y"))
    raise PolygonCanonicalizationError("Unsupported point representation.")


def contains_point(points: Iterable[PointTuple], candidate: PointTuple, tolerance: float) -> bool:
    """Return True if candidate lies within tolerance of any point in the collection."""
    cx, cy = candidate
    for px, py in points:
        if math.hypot(px - cx, py - cy) <= tolerance:
            return True
    return False


def nearest_point(points: Iterable[PointTuple], candidate: PointTuple) -> Optional[PointTuple]:
    """Return the point in the iterable closest to the candidate."""
    best_point: Optional[PointTuple] = None
    best_distance = float("inf")
    cx, cy = candidate
    for px, py in points:
        distance = math.hypot(px - cx, py - cy)
        if distance < best_distance:
            best_distance = distance
            best_point = (px, py)
    return best_point


__all__ = [
    "PointTuple",
    "PointLike",
    "PolygonCanonicalizationError",
    "point_like_to_tuple",
    "contains_point",
    "nearest_point",
]
