"""Checks that given polygon vertices form a requested triangle or quadrilateral subtype.

``create_polygon`` keeps the vertices exactly as given (order and coordinates), so a
subtype is a claim about those vertices, not a request to move them. These checks
decide whether the claim holds, with the same tolerances as the polygon type flags
(``GeometryUtils``), so an accepted polygon also lists the subtype in its types.

Key Features:
    - Triangle subtypes: equilateral, isosceles, scalene, right, right_isosceles
    - Quadrilateral subtypes: rectangle, square, parallelogram, rhombus, kite,
      trapezoid, isosceles_trapezoid, right_trapezoid
    - Vertices are taken in the given cyclic order
    - A mismatch is described with the measured sides and angles
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from utils.geometry_utils import GeometryUtils
from utils.polygon_subtypes import QuadrilateralSubtype, TriangleSubtype

Coordinate = Tuple[float, float]


def triangle_subtype_mismatch(vertices: Sequence[Coordinate], subtype: TriangleSubtype) -> Optional[str]:
    """Return why the vertices do not form the subtype, or None when they do."""
    try:
        flags = GeometryUtils.triangle_type_flags(vertices)
    except ValueError:
        return _mismatch_message(subtype.value, "the vertices overlap")
    checks: Dict[TriangleSubtype, bool] = {
        TriangleSubtype.EQUILATERAL: flags["equilateral"],
        TriangleSubtype.ISOSCELES: flags["isosceles"],
        TriangleSubtype.SCALENE: flags["scalene"],
        TriangleSubtype.RIGHT: flags["right"],
        TriangleSubtype.RIGHT_ISOSCELES: flags["right"] and flags["isosceles"],
    }
    if checks[subtype]:
        return None
    return _mismatch_message(subtype.value, _describe(vertices))


def quadrilateral_subtype_mismatch(vertices: Sequence[Coordinate], subtype: QuadrilateralSubtype) -> Optional[str]:
    """Return why the vertices, in the given order, do not form the subtype, or None when they do."""
    try:
        matches = _QUADRILATERAL_CHECKS[subtype](list(vertices))
    except ValueError:
        return _mismatch_message(subtype.value, "the vertices overlap")
    if matches:
        return None
    return _mismatch_message(subtype.value, _describe(vertices))


# ---------------------------------------------------------------------------
# Quadrilateral predicates (vertices in cyclic order p0, p1, p2, p3)
# ---------------------------------------------------------------------------


def _flag(name: str) -> Callable[[List[Coordinate]], bool]:
    return lambda points: GeometryUtils.quadrilateral_type_flags(points)[name]


def _is_parallelogram(points: List[Coordinate]) -> bool:
    return _sides_parallel(points, 0, 2) and _sides_parallel(points, 1, 3)


def _is_kite(points: List[Coordinate]) -> bool:
    s = GeometryUtils._polygon_side_lengths(points)
    close = GeometryUtils._is_close
    return (close(s[0], s[1]) and close(s[2], s[3])) or (close(s[1], s[2]) and close(s[3], s[0]))


def _is_trapezoid(points: List[Coordinate]) -> bool:
    return _sides_parallel(points, 0, 2) or _sides_parallel(points, 1, 3)


def _is_isosceles_trapezoid(points: List[Coordinate]) -> bool:
    s = GeometryUtils._polygon_side_lengths(points)
    close = GeometryUtils._is_close
    diagonals_equal = close(_distance(points[0], points[2]), _distance(points[1], points[3]))
    if not diagonals_equal:
        return False
    return (_sides_parallel(points, 0, 2) and close(s[1], s[3])) or (
        _sides_parallel(points, 1, 3) and close(s[0], s[2])
    )


def _is_right_trapezoid(points: List[Coordinate]) -> bool:
    return _is_trapezoid(points) and GeometryUtils._has_right_angle(GeometryUtils._polygon_internal_angles(points))


_QUADRILATERAL_CHECKS: Dict[QuadrilateralSubtype, Callable[[List[Coordinate]], bool]] = {
    QuadrilateralSubtype.SQUARE: _flag("square"),
    QuadrilateralSubtype.RECTANGLE: _flag("rectangle"),
    QuadrilateralSubtype.RHOMBUS: _flag("rhombus"),
    QuadrilateralSubtype.PARALLELOGRAM: _is_parallelogram,
    QuadrilateralSubtype.KITE: _is_kite,
    QuadrilateralSubtype.TRAPEZOID: _is_trapezoid,
    QuadrilateralSubtype.ISOSCELES_TRAPEZOID: _is_isosceles_trapezoid,
    QuadrilateralSubtype.RIGHT_TRAPEZOID: _is_right_trapezoid,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _distance(a: Coordinate, b: Coordinate) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _side_vector(points: Sequence[Coordinate], index: int) -> Coordinate:
    start = points[index]
    end = points[(index + 1) % len(points)]
    return (end[0] - start[0], end[1] - start[1])


def _sides_parallel(points: Sequence[Coordinate], first: int, second: int) -> bool:
    """True when two sides are parallel within the shape-classification angle tolerance."""
    ux, uy = _side_vector(points, first)
    vx, vy = _side_vector(points, second)
    lengths = math.hypot(ux, uy) * math.hypot(vx, vy)
    if lengths == 0.0:
        raise ValueError("Degenerate polygon with overlapping points.")
    sine = abs(ux * vy - uy * vx) / lengths
    return sine <= math.sin(math.radians(GeometryUtils.ANGLE_TOLERANCE_DEGREES))


def _format(value: float) -> str:
    return f"{value:.6g}"


def _describe(vertices: Sequence[Coordinate]) -> str:
    sides = ", ".join(_format(length) for length in GeometryUtils._polygon_side_lengths(vertices))
    angles = ", ".join(_format(angle) for angle in GeometryUtils._polygon_internal_angles(vertices))
    return f"in the given order the sides are {sides} and the angles are {angles} degrees"


def _mismatch_message(subtype_value: str, details: str) -> str:
    label = subtype_value.replace("_", " ")
    article = "an" if label[0] in "aeiou" else "a"
    return (
        f"The vertices do not form {article} {label} ({details}). "
        f"Vertices are used exactly as given, in order: give coordinates that form {article} {label}, "
        "or omit the subtype."
    )


__all__ = ["quadrilateral_subtype_mismatch", "triangle_subtype_mismatch"]
