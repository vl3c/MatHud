"""Polygon subtype enumerations for triangles and quadrilaterals.

This module defines enumeration types for polygon subtypes used by
canonicalization routines and polygon managers.

Key Features:
    - TriangleSubtype: equilateral, isosceles, scalene, right, right_isosceles
    - QuadrilateralSubtype: rectangle, square, parallelogram, rhombus, kite, trapezoids
    - Case-insensitive string parsing with normalization
    - Value listing and iteration utilities
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, List, Union


class _BaseSubtype(Enum):
    @classmethod
    def from_value(cls, value: Union[str, "_BaseSubtype"]) -> "_BaseSubtype":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        allowed = ", ".join(sorted(cls.values()))
        raise ValueError(f"Unsupported {cls.__name__} '{value}'. Expected one of: {allowed}.")

    @classmethod
    def values(cls) -> List[str]:
        return [member.value for member in cls]

    @classmethod
    def iter_values(cls) -> Iterable[str]:
        return (member.value for member in cls)

    def __str__(self) -> str:
        return self.value


class TriangleSubtype(_BaseSubtype):
    EQUILATERAL = "equilateral"
    ISOSCELES = "isosceles"
    SCALENE = "scalene"
    RIGHT = "right"
    RIGHT_ISOSCELES = "right_isosceles"


class QuadrilateralSubtype(_BaseSubtype):
    RECTANGLE = "rectangle"
    SQUARE = "square"
    PARALLELOGRAM = "parallelogram"
    RHOMBUS = "rhombus"
    KITE = "kite"
    TRAPEZOID = "trapezoid"
    ISOSCELES_TRAPEZOID = "isosceles_trapezoid"
    RIGHT_TRAPEZOID = "right_trapezoid"


__all__ = ["TriangleSubtype", "QuadrilateralSubtype"]
