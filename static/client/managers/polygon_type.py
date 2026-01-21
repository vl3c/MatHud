"""Polygon type enumeration for polygon manager operations.

This module defines the PolygonType enum used to specify polygon types
when creating or querying polygons through the PolygonManager.

Key Features:
    - Named constants for all supported polygon types (3-10 sides)
    - Special GENERIC type for n-gons with n > 10
    - Case-insensitive string coercion via coerce() method
    - String subclass for direct string comparison
"""

from __future__ import annotations

from enum import Enum


class PolygonType(str, Enum):
    TRIANGLE = "triangle"
    QUADRILATERAL = "quadrilateral"
    RECTANGLE = "rectangle"
    SQUARE = "square"
    PENTAGON = "pentagon"
    HEXAGON = "hexagon"
    HEPTAGON = "heptagon"
    OCTAGON = "octagon"
    NONAGON = "nonagon"
    DECAGON = "decagon"
    GENERIC = "generic"

    @classmethod
    def coerce(cls, value: str) -> "PolygonType":
        try:
            return cls(value.strip().lower())
        except Exception as exc:
            raise ValueError(f"Unsupported polygon type '{value}'.") from exc

