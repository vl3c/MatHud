from __future__ import annotations

from enum import Enum


class PolygonType(str, Enum):
    TRIANGLE = "triangle"
    QUADRILATERAL = "quadrilateral"
    RECTANGLE = "rectangle"
    SQUARE = "square"
    PENTAGON = "pentagon"
    HEXAGON = "hexagon"

    @classmethod
    def coerce(cls, value: str) -> "PolygonType":
        try:
            return cls(value.strip().lower())
        except Exception as exc:
            raise ValueError(f"Unsupported polygon type '{value}'.") from exc

