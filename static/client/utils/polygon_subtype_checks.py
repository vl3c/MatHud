"""Fit given polygon vertices to a requested triangle or quadrilateral subtype.

``create_polygon`` keeps the vertices in the given order. A subtype (or the
rectangle/square polygon types) is resolved against them in three tiers:

1. Vertices that already form the subtype, with the tolerances of the polygon
   type flags (``GeometryUtils``: 1e-4 relative on sides, 0.01 degrees on angles),
   are used verbatim.
2. Vertices close to the subtype (every vertex that has to move moves by at most
   ``ADJUST_RELATIVE_BOUND`` of the average side length) are adjusted, moving as
   few vertices as possible, preferring the last ones in the given order (the apex,
   the last corner), and never a vertex marked as fixed (an existing point). The
   result is exact, so the type flags list the subtype.
3. Anything else is refused with a message that names the shape, gives the
   sides and angles, states the tolerances and, where it helps, suggests a vertex.

Key Features:
    - Triangle subtypes: equilateral, isosceles, scalene, right, right_isosceles
    - Quadrilateral subtypes: rectangle, square, parallelogram, rhombus, kite,
      trapezoid, isosceles_trapezoid, right_trapezoid
    - Vertices are never re-ordered; quadrilateral corners out of cyclic order are refused
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from utils.geometry_utils import GeometryUtils
from utils.polygon_subtypes import QuadrilateralSubtype, TriangleSubtype

Coordinate = Tuple[float, float]
Subtype = Union[TriangleSubtype, QuadrilateralSubtype]
Move = Tuple[int, Coordinate, Coordinate]

# A vertex may move by at most this fraction of the average side length to fit the subtype.
ADJUST_RELATIVE_BOUND = 0.02
# Displacements below this fraction of the average side length count as "not moved".
_UNMOVED_RELATIVE = 1e-12


class SubtypeResolution:
    """Vertices to create, and the vertices that were moved to get them."""

    def __init__(self, vertices: List[Coordinate], moves: List[Move]) -> None:
        self.vertices: List[Coordinate] = vertices
        self.moves: List[Move] = moves  # (index, given, placed)


def resolve_polygon_subtype(
    vertices: Sequence[Coordinate],
    subtype: Subtype,
    *,
    fixed: Optional[Sequence[bool]] = None,
    labels: Optional[Sequence[str]] = None,
) -> SubtypeResolution:
    """Return the vertices to create for ``subtype``, or raise ValueError saying why it cannot be done.

    ``fixed[i]`` marks vertices that must not move (existing points); ``labels`` names the
    vertices in messages (default A, B, C, ...).
    """
    points = [(float(x), float(y)) for x, y in vertices]
    count = len(points)
    fixed_flags = list(fixed) if fixed is not None else [False] * count
    names = list(labels) if labels else [chr(ord("A") + index) for index in range(count)]
    shape = shape_name(subtype)

    _refuse_overlap(points, names, shape)
    if count == 4:
        _refuse_crossing(points, names, shape)
    _refuse_collinear(points, shape)

    if _matches(points, subtype):
        return SubtypeResolution(points, [])

    scale = _mean_side(points)
    bound = ADJUST_RELATIVE_BOUND * scale
    within_bound: List[Tuple[List[int], float, List[Coordinate]]] = []
    for candidate in _CONSTRUCTIONS[subtype](points):
        if not _valid_candidate(points, candidate, subtype):
            continue
        displacements = [_distance(a, b) for a, b in zip(points, candidate)]
        largest = max(displacements)
        if largest > bound:
            continue
        moved = [index for index, shift in enumerate(displacements) if shift > _UNMOVED_RELATIVE * scale]
        within_bound.append((moved, largest, candidate))

    allowed = [entry for entry in within_bound if not any(fixed_flags[index] for index in entry[0])]
    if allowed:
        moved, _, candidate = min(allowed, key=_preference)
        exact = [candidate[index] if index in moved else points[index] for index in range(count)]
        return SubtypeResolution(exact, [(index, points[index], exact[index]) for index in moved])
    if within_bound:
        moved, _, candidate = min(within_bound, key=_preference)
        raise ValueError(_existing_point_message(shape, names, points, candidate, moved, fixed_flags))
    raise ValueError(_mismatch_message(shape, subtype, names, points))


def _preference(entry: Tuple[List[int], float, List[Coordinate]]) -> Tuple[int, Tuple[int, ...], float]:
    """Fewest moved vertices, then the latest ones in the given order, then the smallest move."""
    moved, largest, _ = entry
    return (len(moved), tuple(sorted(-index for index in moved)), largest)


def cyclic_vertex_order(vertices: Sequence[Coordinate]) -> List[Coordinate]:
    """Return a quadrilateral's corners in cyclic order.

    Corners already in order (no two sides cross) are returned as given; otherwise they
    are sorted by angle around their centroid, starting from the first corner. Used for
    old workspace saves, which stored rectangle corners sorted by name.
    """
    points = [(float(x), float(y)) for x, y in vertices]
    if len(points) != 4 or _crossing_sides(points) is None:
        return points
    cx = sum(x for x, _ in points) / 4.0
    cy = sum(y for _, y in points) / 4.0
    start = math.atan2(points[0][1] - cy, points[0][0] - cx)
    return sorted(points, key=lambda p: (math.atan2(p[1] - cy, p[0] - cx) - start) % (2.0 * math.pi))


def format_coordinate(point: Coordinate) -> str:
    return f"({_format(point[0])}, {_format(point[1])})"


# ---------------------------------------------------------------------------
# Matching (the same tolerances as the polygon type flags)
# ---------------------------------------------------------------------------


def _matches(points: Sequence[Coordinate], subtype: Subtype) -> bool:
    try:
        return _MATCHERS[subtype](list(points))
    except ValueError:
        return False


def _triangle_flag(*names: str) -> Callable[[List[Coordinate]], bool]:
    return lambda points: all(GeometryUtils.triangle_type_flags(points)[name] for name in names)


def _quad_flag(name: str) -> Callable[[List[Coordinate]], bool]:
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
    if not close(_distance(points[0], points[2]), _distance(points[1], points[3])):
        return False
    return (_sides_parallel(points, 0, 2) and close(s[1], s[3])) or (
        _sides_parallel(points, 1, 3) and close(s[0], s[2])
    )


def _is_right_trapezoid(points: List[Coordinate]) -> bool:
    return _is_trapezoid(points) and GeometryUtils._has_right_angle(GeometryUtils._polygon_internal_angles(points))


_MATCHERS: Dict[Subtype, Callable[[List[Coordinate]], bool]] = {
    TriangleSubtype.EQUILATERAL: _triangle_flag("equilateral"),
    TriangleSubtype.ISOSCELES: _triangle_flag("isosceles"),
    TriangleSubtype.SCALENE: _triangle_flag("scalene"),
    TriangleSubtype.RIGHT: _triangle_flag("right"),
    TriangleSubtype.RIGHT_ISOSCELES: _triangle_flag("right", "isosceles"),
    QuadrilateralSubtype.SQUARE: _quad_flag("square"),
    QuadrilateralSubtype.RECTANGLE: _quad_flag("rectangle"),
    QuadrilateralSubtype.RHOMBUS: _quad_flag("rhombus"),
    QuadrilateralSubtype.PARALLELOGRAM: _is_parallelogram,
    QuadrilateralSubtype.KITE: _is_kite,
    QuadrilateralSubtype.TRAPEZOID: _is_trapezoid,
    QuadrilateralSubtype.ISOSCELES_TRAPEZOID: _is_isosceles_trapezoid,
    QuadrilateralSubtype.RIGHT_TRAPEZOID: _is_right_trapezoid,
}


# ---------------------------------------------------------------------------
# Constructions: exact shapes that keep some of the given vertices
# ---------------------------------------------------------------------------

Candidates = List[List[Coordinate]]


def _replace(points: Sequence[Coordinate], replacements: Dict[int, Coordinate]) -> List[Coordinate]:
    return [replacements.get(index, point) for index, point in enumerate(points)]


def _triangle_moves(points: Sequence[Coordinate]) -> List[Tuple[int, int, int]]:
    """(moved k, kept i, kept j) for each vertex k, with i, j the other two in cyclic order."""
    return [(k, (k + 1) % 3, (k + 2) % 3) for k in range(3)]


def _equilateral_candidates(points: Sequence[Coordinate]) -> Candidates:
    result: Candidates = []
    for k, i, j in _triangle_moves(points):
        apex = _nearest(points[k], _equilateral_apexes(points[i], points[j]))
        result.append(_replace(points, {k: apex}))
    return result


def _isosceles_candidates(points: Sequence[Coordinate]) -> Candidates:
    result: Candidates = []
    for k, i, j in _triangle_moves(points):
        pi, pj, pk = points[i], points[j], points[k]
        # k as the apex: onto the perpendicular bisector of the base.
        mid = _midpoint(pi, pj)
        result.append(_replace(points, {k: _project_on_line(pk, mid, _perpendicular(_sub(pj, pi)))}))
        # k as a base vertex: onto the circle around the apex through the other base vertex.
        for apex, other in ((pi, pj), (pj, pi)):
            result.append(_replace(points, {k: _on_circle(pk, apex, _distance(apex, other))}))
    return result


def _right_candidates(points: Sequence[Coordinate]) -> Candidates:
    result: Candidates = []
    for k, i, j in _triangle_moves(points):
        pi, pj, pk = points[i], points[j], points[k]
        # Right angle at k: onto the circle with diameter ij (Thales).
        result.append(_replace(points, {k: _on_circle(pk, _midpoint(pi, pj), _distance(pi, pj) / 2.0)}))
        # Right angle at i or j: onto the line through that vertex perpendicular to ij.
        for corner in (pi, pj):
            result.append(_replace(points, {k: _project_on_line(pk, corner, _perpendicular(_sub(pj, pi)))}))
    return result


def _right_isosceles_candidates(points: Sequence[Coordinate]) -> Candidates:
    result: Candidates = []
    for k, i, j in _triangle_moves(points):
        pi, pj, pk = points[i], points[j], points[k]
        result.append(_replace(points, {k: _nearest(pk, _right_isosceles_apexes(pi, pj))}))
        for corner, other in ((pi, pj), (pj, pi)):
            leg = _sub(other, corner)
            options = [_add(corner, _perpendicular(leg)), _add(corner, _scale(_perpendicular(leg), -1.0))]
            result.append(_replace(points, {k: _nearest(pk, options)}))
    return result


def _edge_frames(points: Sequence[Coordinate]) -> List[Tuple[int, int, int, int]]:
    """(a, b, c, d) index tuples: keep the side a-b, rebuild c (after b) and d (after c)."""
    return [(i, (i + 1) % 4, (i + 2) % 4, (i + 3) % 4) for i in range(4)]


def _square_candidates(points: Sequence[Coordinate]) -> Candidates:
    turn = 1.0 if _signed_area(points) > 0 else -1.0
    result: Candidates = []
    for a, b, c, d in _edge_frames(points):
        side = _sub(points[b], points[a])
        new_c = _add(points[b], _scale(_perpendicular(side), turn))
        new_d = _add(points[a], _sub(new_c, points[b]))
        result.append(_replace(points, {c: new_c, d: new_d}))
    return result


def _rectangle_candidates(points: Sequence[Coordinate]) -> Candidates:
    result: Candidates = []
    for a, b, c, d in _edge_frames(points):
        normal = _perpendicular(_sub(points[b], points[a]))
        new_c = _project_on_line(points[c], points[b], normal)
        new_d = _add(points[a], _sub(new_c, points[b]))
        result.append(_replace(points, {c: new_c, d: new_d}))
    return result


def _rhombus_candidates(points: Sequence[Coordinate]) -> Candidates:
    result: Candidates = []
    for a, b, c, d in _edge_frames(points):
        new_c = _on_circle(points[c], points[b], _distance(points[a], points[b]))
        new_d = _add(points[a], _sub(new_c, points[b]))
        result.append(_replace(points, {c: new_c, d: new_d}))
    return result


def _parallelogram_candidates(points: Sequence[Coordinate]) -> Candidates:
    result: Candidates = []
    for k in range(4):
        before, after, opposite = points[(k - 1) % 4], points[(k + 1) % 4], points[(k + 2) % 4]
        result.append(_replace(points, {k: _sub(_add(before, after), opposite)}))
    return result


def _kite_candidates(points: Sequence[Coordinate]) -> Candidates:
    result: Candidates = []
    for k in range(4):
        # Mirror the opposite vertex across the diagonal through the neighbours of k.
        mirrored = _reflect(points[(k + 2) % 4], points[(k + 1) % 4], points[(k + 3) % 4])
        result.append(_replace(points, {k: mirrored}))
    return result


def _parallel_side_moves() -> List[Tuple[int, int, int, int]]:
    """(moved k, its side partner m, opposite side start, opposite side end) for both side pairs."""
    moves: List[Tuple[int, int, int, int]] = []
    for first in (0, 1):
        side = (first, (first + 1) % 4)
        opposite = ((first + 2) % 4, (first + 3) % 4)
        for k, m in ((side[0], side[1]), (side[1], side[0])):
            moves.append((k, m, opposite[0], opposite[1]))
        for k, m in ((opposite[0], opposite[1]), (opposite[1], opposite[0])):
            moves.append((k, m, side[0], side[1]))
    return moves


def _trapezoid_candidates(points: Sequence[Coordinate]) -> Candidates:
    result: Candidates = []
    for k, m, start, end in _parallel_side_moves():
        direction = _sub(points[end], points[start])
        result.append(_replace(points, {k: _project_on_line(points[k], points[m], direction)}))
    return result


def _isosceles_trapezoid_candidates(points: Sequence[Coordinate]) -> Candidates:
    result: Candidates = []
    for k, m, start, end in _parallel_side_moves():
        # k mirrors its side partner m across the perpendicular bisector of the opposite side.
        mid = _midpoint(points[start], points[end])
        axis_point = _add(mid, _perpendicular(_sub(points[end], points[start])))
        result.append(_replace(points, {k: _reflect(points[m], mid, axis_point)}))
    return result


def _right_trapezoid_candidates(points: Sequence[Coordinate]) -> Candidates:
    result: Candidates = []
    for k in range(4):
        # Move only k: the right angle is at a neighbour of k, and k's other side is parallel
        # to the side opposite it. o is the vertex opposite k.
        o = points[(k + 2) % 4]
        for corner, other in ((points[(k + 1) % 4], points[(k - 1) % 4]), (points[(k - 1) % 4], points[(k + 1) % 4])):
            base = _sub(o, corner)
            result.append(_replace(points, {k: _intersect_lines(corner, _perpendicular(base), other, base)}))
    for a, b, c, d in _edge_frames(points):
        # Side a-b is the leg perpendicular to both bases b-c and a-d.
        normal = _perpendicular(_sub(points[b], points[a]))
        new_c = _project_on_line(points[c], points[b], normal)
        new_d = _project_on_line(points[d], points[a], normal)
        result.append(_replace(points, {c: new_c, d: new_d}))
    return result


def _no_candidates(points: Sequence[Coordinate]) -> Candidates:
    return []


_CONSTRUCTIONS: Dict[Subtype, Callable[[Sequence[Coordinate]], Candidates]] = {
    TriangleSubtype.EQUILATERAL: _equilateral_candidates,
    TriangleSubtype.ISOSCELES: _isosceles_candidates,
    TriangleSubtype.SCALENE: _no_candidates,
    TriangleSubtype.RIGHT: _right_candidates,
    TriangleSubtype.RIGHT_ISOSCELES: _right_isosceles_candidates,
    QuadrilateralSubtype.SQUARE: _square_candidates,
    QuadrilateralSubtype.RECTANGLE: _rectangle_candidates,
    QuadrilateralSubtype.RHOMBUS: _rhombus_candidates,
    QuadrilateralSubtype.PARALLELOGRAM: _parallelogram_candidates,
    QuadrilateralSubtype.KITE: _kite_candidates,
    QuadrilateralSubtype.TRAPEZOID: _trapezoid_candidates,
    QuadrilateralSubtype.ISOSCELES_TRAPEZOID: _isosceles_trapezoid_candidates,
    QuadrilateralSubtype.RIGHT_TRAPEZOID: _right_trapezoid_candidates,
}


def _valid_candidate(original: Sequence[Coordinate], candidate: Sequence[Coordinate], subtype: Subtype) -> bool:
    """An exact, non-degenerate shape of the subtype with the input's orientation and no crossing sides."""
    if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in candidate):
        return False
    if _signed_area(candidate) * _signed_area(original) <= 0:
        return False
    if len(candidate) == 4 and _crossing_sides(candidate) is not None:
        return False
    return _matches(candidate, subtype)


# ---------------------------------------------------------------------------
# Refusals and messages
# ---------------------------------------------------------------------------


def _refuse_overlap(points: Sequence[Coordinate], names: Sequence[str], shape: str) -> None:
    tolerance = 1e-9 * max(_mean_side(points), 1.0)
    for index, point in enumerate(points):
        for other in range(index + 1, len(points)):
            if _distance(point, points[other]) <= tolerance:
                raise ValueError(
                    f"The vertices cannot form {shape}: {names[index]} and {names[other]} are at the same position."
                )


def _refuse_collinear(points: Sequence[Coordinate], shape: str) -> None:
    scale = _mean_side(points)
    if abs(_signed_area(points)) <= 1e-9 * scale * scale:
        raise ValueError(f"The vertices cannot form {shape}: they lie on one line.")


def _refuse_crossing(points: Sequence[Coordinate], names: Sequence[str], shape: str) -> None:
    crossing = _crossing_sides(points)
    if crossing is None:
        return
    first, second = crossing
    raise ValueError(
        f"The vertices are not in order around {shape}: sides {_side_name(names, first)} and "
        f"{_side_name(names, second)} cross. Vertices are used in the given order and never re-ordered; "
        "list the corners in order around the shape."
    )


def _mismatch_message(shape: str, subtype: Subtype, names: Sequence[str], points: Sequence[Coordinate]) -> str:
    message = (
        f"The vertices do not form {shape}: {_describe(names, points)}. Vertices are used in the given order; "
        f"exact means sides equal within {GeometryUtils.SIDE_LENGTH_RELATIVE_TOLERANCE * 100:g}% and angles "
        f"within {GeometryUtils.ANGLE_TOLERANCE_DEGREES:g} degrees, and vertices off by at most "
        f"{ADJUST_RELATIVE_BOUND * 100:g}% of the average side length are adjusted to fit."
    )
    suggestion = _suggestion(subtype, names, points)
    if suggestion:
        message += " " + suggestion
    return message + f" Give coordinates that form {shape}, or omit the subtype."


def _existing_point_message(
    shape: str,
    names: Sequence[str],
    points: Sequence[Coordinate],
    candidate: Sequence[Coordinate],
    moved: Sequence[int],
    fixed: Sequence[bool],
) -> str:
    changes = ", ".join(
        f"{names[index]} from {format_coordinate(points[index])} to {format_coordinate(candidate[index])}"
        for index in moved
    )
    existing = [names[index] for index in moved if fixed[index]]
    what = f"{existing[0]} is an existing point" if len(existing) == 1 else f"{', '.join(existing)} are existing points"
    return (
        f"The vertices are close to {shape}, but fitting them would move {changes}, and {what}; existing "
        f"points are never moved. Give exact coordinates that form {shape}, or omit the subtype."
    )


def _suggestion(subtype: Subtype, names: Sequence[str], points: Sequence[Coordinate]) -> Optional[str]:
    if subtype is TriangleSubtype.EQUILATERAL:
        apex = _nearest(points[2], _equilateral_apexes(points[0], points[1]))
        return f"Keeping {names[0]} and {names[1]}, {names[2]} would be at {format_coordinate(apex)}."
    if subtype is TriangleSubtype.RIGHT_ISOSCELES:
        angles = GeometryUtils._polygon_internal_angles(points)
        k = min(range(3), key=lambda index: abs(angles[index] - 90.0))
        kept, moved = sorted(((k + 1) % 3, (k + 2) % 3))
        leg = _sub(points[kept], points[k])
        options = [_add(points[k], _perpendicular(leg)), _sub(points[k], _perpendicular(leg))]
        corner = _nearest(points[moved], options)
        return (
            f"With the right angle at {names[k]}, keeping {names[k]} and {names[kept]}, {names[moved]} would be at "
            f"{format_coordinate(corner)}."
        )
    return None


def _describe(names: Sequence[str], points: Sequence[Coordinate]) -> str:
    lengths = GeometryUtils._polygon_side_lengths(points)
    angles = GeometryUtils._polygon_internal_angles(points)
    sides = ", ".join(f"{_side_name(names, index)} = {_format(length)}" for index, length in enumerate(lengths))
    corners = ", ".join(f"{names[index]} = {_format(angle)}" for index, angle in enumerate(angles))
    return f"sides {sides}; angles {corners} degrees"


def _side_name(names: Sequence[str], index: int) -> str:
    return f"{names[index]}{names[(index + 1) % len(names)]}"


def shape_name(subtype: Subtype) -> str:
    """'an equilateral triangle', 'a square', ..."""
    noun = "triangle" if isinstance(subtype, TriangleSubtype) else ""
    label = subtype.value.replace("_", " ")
    text = f"{label} {noun}".strip() if noun else label
    article = "an" if text[0] in "aeiou" else "a"
    return f"{article} {text}"


def _format(value: float) -> str:
    return f"{value:.8g}"


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def _add(a: Coordinate, b: Coordinate) -> Coordinate:
    return (a[0] + b[0], a[1] + b[1])


def _sub(a: Coordinate, b: Coordinate) -> Coordinate:
    return (a[0] - b[0], a[1] - b[1])


def _scale(a: Coordinate, factor: float) -> Coordinate:
    return (a[0] * factor, a[1] * factor)


def _perpendicular(a: Coordinate) -> Coordinate:
    """a rotated by +90 degrees."""
    return (-a[1], a[0])


def _midpoint(a: Coordinate, b: Coordinate) -> Coordinate:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _distance(a: Coordinate, b: Coordinate) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _nearest(target: Coordinate, options: Sequence[Coordinate]) -> Coordinate:
    return min(options, key=lambda option: _distance(target, option))


def _project_on_line(point: Coordinate, origin: Coordinate, direction: Coordinate) -> Coordinate:
    length_sq = direction[0] ** 2 + direction[1] ** 2
    if length_sq == 0.0:
        return (math.nan, math.nan)
    t = ((point[0] - origin[0]) * direction[0] + (point[1] - origin[1]) * direction[1]) / length_sq
    return (origin[0] + t * direction[0], origin[1] + t * direction[1])


def _on_circle(point: Coordinate, center: Coordinate, radius: float) -> Coordinate:
    offset = _sub(point, center)
    length = math.hypot(offset[0], offset[1])
    if length == 0.0:
        return (math.nan, math.nan)
    return _add(center, _scale(offset, radius / length))


def _intersect_lines(p: Coordinate, u: Coordinate, q: Coordinate, v: Coordinate) -> Coordinate:
    """Intersection of the lines p + s*u and q + t*v (NaN when parallel)."""
    denominator = u[0] * v[1] - u[1] * v[0]
    if denominator == 0.0:
        return (math.nan, math.nan)
    s = ((q[0] - p[0]) * v[1] - (q[1] - p[1]) * v[0]) / denominator
    return (p[0] + s * u[0], p[1] + s * u[1])


def _reflect(point: Coordinate, line_a: Coordinate, line_b: Coordinate) -> Coordinate:
    foot = _project_on_line(point, line_a, _sub(line_b, line_a))
    return (2.0 * foot[0] - point[0], 2.0 * foot[1] - point[1])


def _equilateral_apexes(a: Coordinate, b: Coordinate) -> List[Coordinate]:
    mid = _midpoint(a, b)
    height = _scale(_perpendicular(_sub(b, a)), math.sqrt(3.0) / 2.0)
    return [_add(mid, height), _sub(mid, height)]


def _right_isosceles_apexes(a: Coordinate, b: Coordinate) -> List[Coordinate]:
    mid = _midpoint(a, b)
    half = _perpendicular(_scale(_sub(b, a), 0.5))
    return [_add(mid, half), _sub(mid, half)]


def _signed_area(points: Sequence[Coordinate]) -> float:
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _mean_side(points: Sequence[Coordinate]) -> float:
    count = len(points)
    return sum(_distance(points[index], points[(index + 1) % count]) for index in range(count)) / count


def _side_vector(points: Sequence[Coordinate], index: int) -> Coordinate:
    return _sub(points[(index + 1) % len(points)], points[index])


def _sides_parallel(points: Sequence[Coordinate], first: int, second: int) -> bool:
    """True when two sides are parallel within the shape-classification angle tolerance."""
    ux, uy = _side_vector(points, first)
    vx, vy = _side_vector(points, second)
    lengths = math.hypot(ux, uy) * math.hypot(vx, vy)
    if lengths == 0.0:
        raise ValueError("Degenerate polygon with overlapping points.")
    sine = abs(ux * vy - uy * vx) / lengths
    return sine <= math.sin(math.radians(GeometryUtils.ANGLE_TOLERANCE_DEGREES))


def _cross(o: Coordinate, a: Coordinate, b: Coordinate) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _crossing_sides(points: Sequence[Coordinate]) -> Optional[Tuple[int, int]]:
    """For a quadrilateral, the indices of two opposite sides that cross, or None."""
    for first, second in ((0, 2), (1, 3)):
        p1, p2 = points[first], points[(first + 1) % 4]
        q1, q2 = points[second], points[(second + 1) % 4]
        d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
        d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
        if d1 * d2 < 0 and d3 * d4 < 0:
            return first, second
    return None


__all__ = [
    "ADJUST_RELATIVE_BOUND",
    "SubtypeResolution",
    "cyclic_vertex_order",
    "format_coordinate",
    "resolve_polygon_subtype",
    "shape_name",
]
