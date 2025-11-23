from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple, Union

PointTuple = Tuple[float, float]
PointLike = Union[PointTuple, Sequence[float], object]


class PolygonCanonicalizationError(ValueError):
    """Raised when provided vertices cannot be coerced into the requested polygon."""


def _point_like_to_tuple(point: PointLike) -> PointTuple:
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
    # Fallback: try to read attributes (works for Point objects)
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(getattr(point, "x")), float(getattr(point, "y"))
    raise PolygonCanonicalizationError("Unsupported point representation.")


def _normalize_mode(construction_mode: str | None) -> str:
    if not construction_mode:
        return "vertices"
    mode = construction_mode.strip().lower()
    if mode not in {"vertices", "diagonal"}:
        raise PolygonCanonicalizationError(f"Unsupported construction_mode '{construction_mode}'.")
    return mode


def canonicalize_rectangle(
    vertices: Sequence[PointLike],
    *,
    construction_mode: str | None = None,
    tolerance: float | None = 1e-6,
) -> List[PointTuple]:
    """
    Produce a best-fit rectangle from provided vertices.

    Args:
        vertices: Sequence of point-like objects representing either four corners (vertices mode)
                  or the two diagonal corners (diagonal mode).
        construction_mode: Either "vertices" (default) or "diagonal".
        tolerance: Relative tolerance used when validating how close the supplied points are to
                   an ideal rectangle. Interpreted relative to the maximum rectangle dimension.
    """
    mode = _normalize_mode(construction_mode)
    tol = 1e-6 if tolerance is None else float(tolerance)
    if mode == "diagonal":
        return _canonicalize_rectangle_diagonal(vertices)
    return _canonicalize_rectangle_vertices(vertices, tol)


def _canonicalize_rectangle_diagonal(vertices: Sequence[PointLike]) -> List[PointTuple]:
    if len(vertices) != 2:
        raise PolygonCanonicalizationError(
            "Diagonal construction mode requires exactly two opposite corner points."
        )
    p1 = _point_like_to_tuple(vertices[0])
    p3 = _point_like_to_tuple(vertices[1])
    x1, y1 = p1
    x3, y3 = p3
    if math.isclose(x1, x3) or math.isclose(y1, y3):
        raise PolygonCanonicalizationError("Diagonal points must not share x or y coordinates.")
    return [
        (x1, y1),
        (x3, y1),
        (x3, y3),
        (x1, y3),
    ]


def _canonicalize_rectangle_vertices(vertices: Sequence[PointLike], tolerance: float) -> List[PointTuple]:
    points = [_point_like_to_tuple(vertex) for vertex in vertices]
    if len(points) < 4:
        raise PolygonCanonicalizationError("Rectangle canonicalization requires at least four vertices.")

    distinct = _dedupe_vertices(points, tolerance)
    primary_anchor, opposite_anchor = _extract_diagonal_anchors(points)
    ordered_vertices = _prioritize_diagonal_vertices(distinct, primary_anchor, opposite_anchor, tolerance)

    centroid_x, centroid_y = _compute_centroid(ordered_vertices)
    ux, uy, vx, vy = _compute_principal_axes(ordered_vertices, centroid_x, centroid_y)
    ux, uy, vx, vy = _align_axes_to_diagonal(ux, uy, vx, vy, ordered_vertices)

    projections = [
        _project_point(x, y, centroid_x, centroid_y, ux, uy, vx, vy)
        for x, y in ordered_vertices
    ]
    min_alpha, max_alpha, min_beta, max_beta = _bounds_from_projections(projections)
    width = max_alpha - min_alpha
    height = max_beta - min_beta
    if width <= 0 or height <= 0:
        raise PolygonCanonicalizationError("Provided vertices collapse to a line; cannot form a rectangle.")

    canonical = _build_corners_from_bounds(
        centroid_x,
        centroid_y,
        ux,
        uy,
        vx,
        vy,
        min_alpha,
        max_alpha,
        min_beta,
        max_beta,
    )

    return _synthesize_rectangle(
        canonical,
        primary_anchor,
        opposite_anchor,
        tolerance,
        ordered_vertices,
    )


def _dedupe_vertices(points: Sequence[PointTuple], tolerance: float) -> List[PointTuple]:
    distinct: List[PointTuple] = []
    for point in points:
        if not _contains_point(distinct, point, tolerance):
            distinct.append(point)
    if len(distinct) != 4:
        raise PolygonCanonicalizationError("Exactly four distinct vertices are required for rectangle canonicalization.")
    return distinct


def _extract_diagonal_anchors(points: Sequence[PointTuple]) -> tuple[PointTuple, PointTuple]:
    primary = points[0]
    opposite = points[2] if len(points) > 2 else points[-1]
    return primary, opposite


def _prioritize_diagonal_vertices(
    distinct: Sequence[PointTuple],
    primary_anchor: PointTuple,
    opposite_anchor: PointTuple,
    tolerance: float,
) -> List[PointTuple]:
    if not _contains_point(distinct, primary_anchor, tolerance):
        return list(distinct)

    ordered: List[PointTuple] = []
    for candidate in (primary_anchor, opposite_anchor):
        nearest = _nearest_point(distinct, candidate)
        if nearest is not None and nearest not in ordered:
            ordered.append(nearest)
    for vertex in distinct:
        if vertex not in ordered:
            ordered.append(vertex)
    return ordered


def _compute_centroid(vertices: Sequence[PointTuple]) -> tuple[float, float]:
    centroid_x = sum(x for x, _ in vertices) / len(vertices)
    centroid_y = sum(y for _, y in vertices) / len(vertices)
    return centroid_x, centroid_y


def _compute_principal_axes(
    vertices: Sequence[PointTuple],
    centroid_x: float,
    centroid_y: float,
) -> tuple[float, float, float, float]:
    a = b = c = 0.0
    for x, y in vertices:
        dx = x - centroid_x
        dy = y - centroid_y
        a += dx * dx
        b += dx * dy
        c += dy * dy
    count = float(len(vertices))
    a /= count
    b /= count
    c /= count

    disc = (a - c) * (a - c) + 4.0 * b * b
    if disc <= 1e-18:
        # Symmetric rectangles (e.g., squares) can collapse the covariance discriminant.
        # Fall back to an axis-aligned basis; later alignment will orient to the diagonal.
        return 1.0, 0.0, 0.0, 1.0
    sqrt_disc = math.sqrt(disc)
    lambda1 = (a + c + sqrt_disc) / 2.0
    if abs(b) > 1e-12:
        vx = lambda1 - c
        vy = b
    elif a >= c:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = 0.0, 1.0
    norm = math.hypot(vx, vy)
    if norm == 0:
        raise PolygonCanonicalizationError("Degenerate eigenvector while canonicalizing rectangle.")
    ux, uy = vx / norm, vy / norm
    vx, vy = -uy, ux
    return ux, uy, vx, vy


def _align_axes_to_diagonal(
    ux: float,
    uy: float,
    vx: float,
    vy: float,
    ordered_vertices: Sequence[PointTuple],
) -> tuple[float, float, float, float]:
    if len(ordered_vertices) < 3:
        return ux, uy, vx, vy
    diag_vec_x = ordered_vertices[2][0] - ordered_vertices[0][0]
    diag_vec_y = ordered_vertices[2][1] - ordered_vertices[0][1]
    if abs(diag_vec_x) + abs(diag_vec_y) <= 1e-12:
        return ux, uy, vx, vy
    base_angle = math.atan2(diag_vec_y, diag_vec_x)
    aligned_ux = math.cos(base_angle)
    aligned_uy = math.sin(base_angle)
    aligned_vx = -math.sin(base_angle)
    aligned_vy = math.cos(base_angle)
    return aligned_ux, aligned_uy, aligned_vx, aligned_vy


def _project_point(
    x: float,
    y: float,
    centroid_x: float,
    centroid_y: float,
    ux: float,
    uy: float,
    vx: float,
    vy: float,
) -> PointTuple:
    dx = x - centroid_x
    dy = y - centroid_y
    alpha = dx * ux + dy * uy
    beta = dx * vx + dy * vy
    return alpha, beta


def _bounds_from_projections(
    projections: Sequence[PointTuple],
) -> tuple[float, float, float, float]:
    min_alpha = min(alpha for alpha, _ in projections)
    max_alpha = max(alpha for alpha, _ in projections)
    min_beta = min(beta for _, beta in projections)
    max_beta = max(beta for _, beta in projections)
    return min_alpha, max_alpha, min_beta, max_beta


def _build_corners_from_bounds(
    centroid_x: float,
    centroid_y: float,
    ux: float,
    uy: float,
    vx: float,
    vy: float,
    min_alpha: float,
    max_alpha: float,
    min_beta: float,
    max_beta: float,
) -> List[PointTuple]:
    corners: List[PointTuple] = []
    for alpha, beta in (
        (min_alpha, min_beta),
        (max_alpha, min_beta),
        (max_alpha, max_beta),
        (min_alpha, max_beta),
    ):
        x = centroid_x + alpha * ux + beta * vx
        y = centroid_y + alpha * uy + beta * vy
        corners.append((x, y))
    return corners


def _synthesize_rectangle(
    canonical: List[PointTuple],
    primary_anchor: PointTuple,
    opposite_anchor: PointTuple,
    tolerance: float,
    source_vertices: Sequence[PointTuple],
) -> List[PointTuple]:
    start_index = _index_of_closest_corner(canonical, [primary_anchor])
    canonical = canonical[start_index:] + canonical[:start_index]
    return _scale_rectangle_to_diagonal(
        canonical,
        primary_anchor,
        opposite_anchor,
        tolerance,
        source_vertices,
    )


def _scale_rectangle_to_diagonal(
    canonical: Sequence[PointTuple],
    primary_anchor: PointTuple,
    opposite_anchor: PointTuple,
    tolerance: float,
    source_vertices: Sequence[PointTuple],
) -> List[PointTuple]:
    a_best = canonical[0]
    b_best = canonical[1]
    c_best = canonical[2]
    d_best = canonical[3]

    diag_vec_best = (c_best[0] - a_best[0], c_best[1] - a_best[1])
    diag_len_best = math.hypot(diag_vec_best[0], diag_vec_best[1])
    if diag_len_best <= tolerance:
        raise PolygonCanonicalizationError("Diagonal collapsed while canonicalizing rectangle.")
    u_best_unit = (diag_vec_best[0] / diag_len_best, diag_vec_best[1] / diag_len_best)
    v_best_unit = (-u_best_unit[1], u_best_unit[0])

    b_offset = (b_best[0] - a_best[0], b_best[1] - a_best[1])
    if b_offset[0] * v_best_unit[0] + b_offset[1] * v_best_unit[1] < 0:
        v_best_unit = (-v_best_unit[0], -v_best_unit[1])
    d_offset = (d_best[0] - a_best[0], d_best[1] - a_best[1])

    proj_b_u = b_offset[0] * u_best_unit[0] + b_offset[1] * u_best_unit[1]
    proj_b_v = b_offset[0] * v_best_unit[0] + b_offset[1] * v_best_unit[1]
    proj_d_u = d_offset[0] * u_best_unit[0] + d_offset[1] * u_best_unit[1]
    proj_d_v = d_offset[0] * v_best_unit[0] + d_offset[1] * v_best_unit[1]

    diag_vec_actual = (
        opposite_anchor[0] - primary_anchor[0],
        opposite_anchor[1] - primary_anchor[1],
    )
    diag_len_actual = math.hypot(diag_vec_actual[0], diag_vec_actual[1])
    if diag_len_actual <= tolerance:
        raise PolygonCanonicalizationError("Provided diagonal points collapse to a single location.")
    u_actual_unit = (diag_vec_actual[0] / diag_len_actual, diag_vec_actual[1] / diag_len_actual)
    v_actual_unit_base = (-u_actual_unit[1], u_actual_unit[0])

    scale = diag_len_actual / diag_len_best

    best_rectangle: List[PointTuple] | None = None
    best_error = float("inf")
    for orientation in (1.0, -1.0):
        v_actual_unit = (v_actual_unit_base[0] * orientation, v_actual_unit_base[1] * orientation)
        b_corner = (
            primary_anchor[0]
            + u_actual_unit[0] * (proj_b_u * scale)
            + v_actual_unit[0] * (proj_b_v * scale),
            primary_anchor[1]
            + u_actual_unit[1] * (proj_b_u * scale)
            + v_actual_unit[1] * (proj_b_v * scale),
        )
        d_corner = (
            primary_anchor[0]
            + u_actual_unit[0] * (proj_d_u * scale)
            + v_actual_unit[0] * (proj_d_v * scale),
            primary_anchor[1]
            + u_actual_unit[1] * (proj_d_u * scale)
            + v_actual_unit[1] * (proj_d_v * scale),
        )
        candidate = [primary_anchor, b_corner, opposite_anchor, d_corner]
        error = _rectangle_fit_error(candidate, source_vertices)
        if error < best_error:
            best_error = error
            best_rectangle = candidate

    if best_rectangle is None:
        raise PolygonCanonicalizationError("Failed to approximate rectangle from provided vertices.")
    return best_rectangle


def _contains_point(points: Iterable[PointTuple], candidate: PointTuple, tolerance: float) -> bool:
    for px, py in points:
        if math.hypot(px - candidate[0], py - candidate[1]) <= tolerance:
            return True
    return False


def _nearest_point(points: Iterable[PointTuple], candidate: PointTuple) -> PointTuple | None:
    best_point: PointTuple | None = None
    best_distance = float("inf")
    for px, py in points:
        distance = math.hypot(px - candidate[0], py - candidate[1])
        if distance < best_distance:
            best_distance = distance
            best_point = (px, py)
    return best_point


def _index_of_closest_corner(canonical: List[PointTuple], originals: List[PointTuple]) -> int:
    """Pick the corner closest to any of the original points to improve naming consistency."""
    best_index = 0
    best_distance = float("inf")
    for i, corner in enumerate(canonical):
        cx, cy = corner
        for ox, oy in originals:
            distance = math.hypot(cx - ox, cy - oy)
            if distance < best_distance:
                best_index = i
                best_distance = distance
    return best_index


def _rectangle_fit_error(rectangle: Sequence[PointTuple], originals: Sequence[PointTuple]) -> float:
    total = 0.0
    for ox, oy in originals:
        total += min(math.hypot(ox - rx, oy - ry) for rx, ry in rectangle)
    return total

