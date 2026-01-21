"""Quadrilateral canonicalization with subtype support.

This module provides canonicalization for quadrilaterals, transforming arbitrary
vertex sets into well-formed quadrilaterals of specified subtypes.

Key Features:
    - Rectangle/square canonicalization via PCA and diagonal anchors
    - Diagonal construction mode for axis-aligned rectangles
    - Parallelogram canonicalization via fourth vertex computation
    - Rhombus canonicalization with equal side enforcement
    - Kite canonicalization with axis of symmetry
    - Trapezoid variants (standard, isosceles, right)
    - CCW vertex ordering with original order alignment
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

from .common import (
    PointLike,
    PointTuple,
    PolygonCanonicalizationError,
    contains_point,
    nearest_point,
    point_like_to_tuple,
)
from utils.polygon_subtypes import QuadrilateralSubtype


class QuadrilateralCanonicalizer:
    """Best-fit quadrilateral canonicalization helpers with subtype support."""

    @staticmethod
    def canonicalize(
        vertices: Sequence[PointLike],
        *,
        subtype: QuadrilateralSubtype | str | None = None,
        tolerance: float = 1e-6,
        construction_mode: str | None = None,
        enforce_square: bool = False,
    ) -> List[PointTuple]:
        """Canonicalize vertices into a valid quadrilateral of the specified subtype.

        Args:
            vertices: Input vertices (at least 4 for most subtypes, 2 for diagonal mode).
            subtype: Quadrilateral subtype to enforce.
            tolerance: Distance tolerance for deduplication and validation.
            construction_mode: For rectangle/square only - "vertices" or "diagonal".
            enforce_square: For rectangle subtype - if True, produces a square.

        Returns:
            List of 4 canonicalized vertex tuples.
        """
        normalized_subtype = QuadrilateralCanonicalizer._normalize_subtype(subtype)

        # Rectangle/square subtypes have special handling with construction_mode
        if normalized_subtype in (QuadrilateralSubtype.RECTANGLE, QuadrilateralSubtype.SQUARE):
            return QuadrilateralCanonicalizer._canonicalize_rectangle(
                vertices,
                construction_mode=construction_mode,
                tolerance=tolerance,
                enforce_square=enforce_square or normalized_subtype is QuadrilateralSubtype.SQUARE,
            )

        # All other subtypes require 4 vertices
        points = [point_like_to_tuple(vertex) for vertex in vertices]
        distinct = QuadrilateralCanonicalizer._dedupe_vertices(points, tolerance)
        original_order = list(distinct)
        ordered = QuadrilateralCanonicalizer._order_ccw(distinct)

        QuadrilateralCanonicalizer._ensure_non_degenerate(ordered, tolerance)

        if normalized_subtype is None:
            return QuadrilateralCanonicalizer._align_to_original_order(ordered, original_order)

        if normalized_subtype is QuadrilateralSubtype.PARALLELOGRAM:
            result = QuadrilateralCanonicalizer._canonicalize_parallelogram(ordered, original_order)
        elif normalized_subtype is QuadrilateralSubtype.RHOMBUS:
            result = QuadrilateralCanonicalizer._canonicalize_rhombus(ordered, original_order, tolerance)
        elif normalized_subtype is QuadrilateralSubtype.KITE:
            result = QuadrilateralCanonicalizer._canonicalize_kite(ordered, original_order, tolerance)
        elif normalized_subtype is QuadrilateralSubtype.TRAPEZOID:
            result = QuadrilateralCanonicalizer._canonicalize_trapezoid(ordered, original_order, tolerance)
        elif normalized_subtype is QuadrilateralSubtype.ISOSCELES_TRAPEZOID:
            result = QuadrilateralCanonicalizer._canonicalize_isosceles_trapezoid(ordered, original_order, tolerance)
        elif normalized_subtype is QuadrilateralSubtype.RIGHT_TRAPEZOID:
            result = QuadrilateralCanonicalizer._canonicalize_right_trapezoid(ordered, original_order, tolerance)
        else:
            raise PolygonCanonicalizationError(f"Unsupported quadrilateral subtype '{subtype}'.")

        return QuadrilateralCanonicalizer._align_to_original_order(result, original_order)

    # ------------------------------------------------------------------ #
    # Rectangle/Square canonicalization
    # ------------------------------------------------------------------ #

    @staticmethod
    def _canonicalize_rectangle(
        vertices: Sequence[PointLike],
        *,
        construction_mode: str | None = None,
        tolerance: float = 1e-6,
        enforce_square: bool = False,
    ) -> List[PointTuple]:
        """Best-fit rectangle canonicalization with optional square enforcement."""
        mode = QuadrilateralCanonicalizer._normalize_construction_mode(construction_mode)
        tol = tolerance if tolerance is not None else 1e-6
        if mode == "diagonal":
            return QuadrilateralCanonicalizer._rectangle_from_diagonal(vertices)
        return QuadrilateralCanonicalizer._rectangle_from_vertices(vertices, tol, enforce_square)

    @staticmethod
    def _normalize_construction_mode(construction_mode: str | None) -> str:
        if not construction_mode:
            return "vertices"
        mode = construction_mode.strip().lower()
        if mode not in {"vertices", "diagonal"}:
            raise PolygonCanonicalizationError(f"Unsupported construction_mode '{construction_mode}'.")
        return mode

    @staticmethod
    def _rectangle_from_diagonal(vertices: Sequence[PointLike]) -> List[PointTuple]:
        """Create axis-aligned rectangle from two diagonal corner points."""
        if len(vertices) != 2:
            raise PolygonCanonicalizationError(
                "Diagonal construction mode requires exactly two opposite corner points."
            )
        p1 = point_like_to_tuple(vertices[0])
        p3 = point_like_to_tuple(vertices[1])
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

    @staticmethod
    def _rectangle_from_vertices(
        vertices: Sequence[PointLike],
        tolerance: float,
        enforce_square: bool,
    ) -> List[PointTuple]:
        """Create best-fit rectangle from 4 vertices using PCA."""
        points = [point_like_to_tuple(vertex) for vertex in vertices]
        if len(points) < 4:
            raise PolygonCanonicalizationError("Rectangle canonicalization requires at least four vertices.")

        distinct = QuadrilateralCanonicalizer._rectangle_dedupe_vertices(points, tolerance)
        primary_anchor, opposite_anchor = QuadrilateralCanonicalizer._extract_diagonal_anchors(points)

        ordered_vertices = QuadrilateralCanonicalizer._prioritize_diagonal_vertices(
            distinct,
            primary_anchor,
            opposite_anchor,
            tolerance,
        )

        centroid_x, centroid_y = QuadrilateralCanonicalizer._compute_centroid(ordered_vertices)
        ux, uy, vx, vy = QuadrilateralCanonicalizer._compute_principal_axes(ordered_vertices, centroid_x, centroid_y)
        ux, uy, vx, vy = QuadrilateralCanonicalizer._align_axes_to_diagonal(ux, uy, vx, vy, ordered_vertices)

        projections = [
            QuadrilateralCanonicalizer._project_point(x, y, centroid_x, centroid_y, ux, uy, vx, vy)
            for x, y in ordered_vertices
        ]
        min_alpha, max_alpha, min_beta, max_beta = QuadrilateralCanonicalizer._bounds_from_projections(projections)
        width = max_alpha - min_alpha
        height = max_beta - min_beta
        if width <= 0 or height <= 0:
            raise PolygonCanonicalizationError("Provided vertices collapse to a line; cannot form a rectangle.")

        if enforce_square:
            half_side = (width + height) / 4.0
            mid_alpha = (min_alpha + max_alpha) / 2.0
            mid_beta = (min_beta + max_beta) / 2.0
            min_alpha = mid_alpha - half_side
            max_alpha = mid_alpha + half_side
            min_beta = mid_beta - half_side
            max_beta = mid_beta + half_side

        canonical = QuadrilateralCanonicalizer._build_corners_from_bounds(
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

        return QuadrilateralCanonicalizer._synthesize_rectangle(
            canonical,
            primary_anchor,
            opposite_anchor,
            tolerance,
            ordered_vertices,
        )

    @staticmethod
    def _rectangle_dedupe_vertices(points: Sequence[PointTuple], tolerance: float) -> List[PointTuple]:
        """Dedupe vertices for rectangle (requires exactly 4)."""
        distinct: List[PointTuple] = []
        for point in points:
            if not contains_point(distinct, point, tolerance):
                distinct.append(point)
        if len(distinct) != 4:
            raise PolygonCanonicalizationError(
                "Exactly four distinct vertices are required for rectangle canonicalization."
            )
        return distinct

    @staticmethod
    def _extract_diagonal_anchors(points: Sequence[PointTuple]) -> tuple[PointTuple, PointTuple]:
        primary = points[0]
        opposite = points[2] if len(points) > 2 else points[-1]
        return primary, opposite

    @staticmethod
    def _prioritize_diagonal_vertices(
        distinct: Sequence[PointTuple],
        primary_anchor: PointTuple,
        opposite_anchor: PointTuple,
        tolerance: float,
    ) -> List[PointTuple]:
        if not contains_point(distinct, primary_anchor, tolerance):
            return list(distinct)

        ordered: List[PointTuple] = []
        for candidate in (primary_anchor, opposite_anchor):
            nearest = nearest_point(distinct, candidate)
            if nearest is not None and nearest not in ordered:
                ordered.append(nearest)
        for vertex in distinct:
            if vertex not in ordered:
                ordered.append(vertex)
        return ordered

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _bounds_from_projections(
        projections: Sequence[PointTuple],
    ) -> tuple[float, float, float, float]:
        min_alpha = min(alpha for alpha, _ in projections)
        max_alpha = max(alpha for alpha, _ in projections)
        min_beta = min(beta for _, beta in projections)
        max_beta = max(beta for _, beta in projections)
        return min_alpha, max_alpha, min_beta, max_beta

    @staticmethod
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

    @staticmethod
    def _synthesize_rectangle(
        canonical: List[PointTuple],
        primary_anchor: PointTuple,
        opposite_anchor: PointTuple,
        tolerance: float,
        source_vertices: Sequence[PointTuple],
    ) -> List[PointTuple]:
        start_index = QuadrilateralCanonicalizer._index_of_closest_corner(canonical, [primary_anchor])
        canonical = canonical[start_index:] + canonical[:start_index]

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
            error = QuadrilateralCanonicalizer._rectangle_fit_error(candidate, source_vertices)
            if error < best_error:
                best_error = error
                best_rectangle = candidate

        if best_rectangle is None:
            raise PolygonCanonicalizationError("Failed to approximate rectangle from provided vertices.")
        return best_rectangle

    @staticmethod
    def _index_of_closest_corner(canonical: List[PointTuple], originals: Iterable[PointTuple]) -> int:
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

    @staticmethod
    def _rectangle_fit_error(rectangle: Sequence[PointTuple], originals: Sequence[PointTuple]) -> float:
        total = 0.0
        for ox, oy in originals:
            total += min(math.hypot(ox - rx, oy - ry) for rx, ry in rectangle)
        return total

    # ------------------------------------------------------------------ #
    # Parallelogram canonicalization
    # ------------------------------------------------------------------ #

    @staticmethod
    def _canonicalize_parallelogram(
        ordered: Sequence[PointTuple],
        original: Sequence[PointTuple],
    ) -> List[PointTuple]:
        """Create parallelogram using first 3 vertices, computing 4th.

        For vertices A, B, C, D to form a parallelogram:
        D = A + C - B (ensures AB || DC and AD || BC)
        """
        a, b, c, _ = ordered
        d = (a[0] + c[0] - b[0], a[1] + c[1] - b[1])
        return QuadrilateralCanonicalizer._order_ccw([a, b, c, d])

    # ------------------------------------------------------------------ #
    # Rhombus canonicalization
    # ------------------------------------------------------------------ #

    @staticmethod
    def _canonicalize_rhombus(
        ordered: Sequence[PointTuple],
        original: Sequence[PointTuple],
        tolerance: float,
    ) -> List[PointTuple]:
        """Create rhombus (all sides equal) preserving centroid and first two vertices' orientation."""
        centroid = QuadrilateralCanonicalizer._compute_centroid(ordered)
        a, b = ordered[0], ordered[1]

        side_length = QuadrilateralCanonicalizer._average_side_length(ordered)
        if side_length <= tolerance:
            raise PolygonCanonicalizationError("Quadrilateral side length too small for rhombus canonicalization.")

        vec_ab = (b[0] - a[0], b[1] - a[1])
        len_ab = math.hypot(vec_ab[0], vec_ab[1])
        if len_ab <= tolerance:
            raise PolygonCanonicalizationError("First two vertices are too close for rhombus construction.")

        unit_ab = (vec_ab[0] / len_ab, vec_ab[1] / len_ab)
        perp = (-unit_ab[1], unit_ab[0])

        half_diag1 = side_length * math.sqrt(2) / 2
        half_diag2 = half_diag1

        new_a = (centroid[0] - unit_ab[0] * half_diag1, centroid[1] - unit_ab[1] * half_diag1)
        new_b = (centroid[0] + perp[0] * half_diag2, centroid[1] + perp[1] * half_diag2)
        new_c = (centroid[0] + unit_ab[0] * half_diag1, centroid[1] + unit_ab[1] * half_diag1)
        new_d = (centroid[0] - perp[0] * half_diag2, centroid[1] - perp[1] * half_diag2)

        return QuadrilateralCanonicalizer._order_ccw([new_a, new_b, new_c, new_d])

    # ------------------------------------------------------------------ #
    # Kite canonicalization
    # ------------------------------------------------------------------ #

    @staticmethod
    def _canonicalize_kite(
        ordered: Sequence[PointTuple],
        original: Sequence[PointTuple],
        tolerance: float,
    ) -> List[PointTuple]:
        """Create kite with two pairs of adjacent equal sides.

        Uses first two vertices to define one side and the axis of symmetry.
        The kite has vertices A, B, C, D where AB = AD and CB = CD.
        """
        a, b, c, d = ordered

        vec_ac = (c[0] - a[0], c[1] - a[1])
        len_ac = math.hypot(vec_ac[0], vec_ac[1])
        if len_ac <= tolerance:
            raise PolygonCanonicalizationError("Diagonal vertices too close for kite construction.")

        midpoint_ac = ((a[0] + c[0]) / 2, (a[1] + c[1]) / 2)
        unit_ac = (vec_ac[0] / len_ac, vec_ac[1] / len_ac)
        perp_ac = (-unit_ac[1], unit_ac[0])

        dist_b = abs((b[0] - midpoint_ac[0]) * perp_ac[0] + (b[1] - midpoint_ac[1]) * perp_ac[1])
        dist_d = abs((d[0] - midpoint_ac[0]) * perp_ac[0] + (d[1] - midpoint_ac[1]) * perp_ac[1])
        avg_dist = (dist_b + dist_d) / 2
        if avg_dist <= tolerance:
            avg_dist = len_ac / 4

        sign_b = 1 if (b[0] - midpoint_ac[0]) * perp_ac[0] + (b[1] - midpoint_ac[1]) * perp_ac[1] >= 0 else -1
        sign_d = -sign_b

        new_b = (midpoint_ac[0] + perp_ac[0] * avg_dist * sign_b, midpoint_ac[1] + perp_ac[1] * avg_dist * sign_b)
        new_d = (midpoint_ac[0] + perp_ac[0] * avg_dist * sign_d, midpoint_ac[1] + perp_ac[1] * avg_dist * sign_d)

        return QuadrilateralCanonicalizer._order_ccw([a, new_b, c, new_d])

    # ------------------------------------------------------------------ #
    # Trapezoid canonicalization
    # ------------------------------------------------------------------ #

    @staticmethod
    def _canonicalize_trapezoid(
        ordered: Sequence[PointTuple],
        original: Sequence[PointTuple],
        tolerance: float,
    ) -> List[PointTuple]:
        """Create trapezoid with one pair of parallel sides.

        Uses first two vertices as one parallel side (base), adjusts the opposite side to be parallel.
        """
        a, b, c, d = ordered

        vec_ab = (b[0] - a[0], b[1] - a[1])
        len_ab = math.hypot(vec_ab[0], vec_ab[1])
        if len_ab <= tolerance:
            raise PolygonCanonicalizationError("First two vertices too close for trapezoid construction.")

        unit_ab = (vec_ab[0] / len_ab, vec_ab[1] / len_ab)

        mid_cd = ((c[0] + d[0]) / 2, (c[1] + d[1]) / 2)

        vec_cd = (d[0] - c[0], d[1] - c[1])
        len_cd = math.hypot(vec_cd[0], vec_cd[1])
        if len_cd <= tolerance:
            len_cd = len_ab * 0.6

        new_c = (mid_cd[0] - unit_ab[0] * len_cd / 2, mid_cd[1] - unit_ab[1] * len_cd / 2)
        new_d = (mid_cd[0] + unit_ab[0] * len_cd / 2, mid_cd[1] + unit_ab[1] * len_cd / 2)

        return QuadrilateralCanonicalizer._order_ccw([a, b, new_c, new_d])

    # ------------------------------------------------------------------ #
    # Isosceles trapezoid canonicalization
    # ------------------------------------------------------------------ #

    @staticmethod
    def _canonicalize_isosceles_trapezoid(
        ordered: Sequence[PointTuple],
        original: Sequence[PointTuple],
        tolerance: float,
    ) -> List[PointTuple]:
        """Create isosceles trapezoid with parallel sides and equal legs.

        The non-parallel sides (legs) are made equal length.
        """
        a, b, c, d = ordered

        vec_ab = (b[0] - a[0], b[1] - a[1])
        len_ab = math.hypot(vec_ab[0], vec_ab[1])
        if len_ab <= tolerance:
            raise PolygonCanonicalizationError("First two vertices too close for isosceles trapezoid construction.")

        unit_ab = (vec_ab[0] / len_ab, vec_ab[1] / len_ab)
        perp_ab = (-unit_ab[1], unit_ab[0])

        mid_ab = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        mid_cd = ((c[0] + d[0]) / 2, (c[1] + d[1]) / 2)

        height_vec = (mid_cd[0] - mid_ab[0], mid_cd[1] - mid_ab[1])
        height = height_vec[0] * perp_ab[0] + height_vec[1] * perp_ab[1]
        if abs(height) <= tolerance:
            height = len_ab * 0.5

        vec_cd = (d[0] - c[0], d[1] - c[1])
        len_cd = math.hypot(vec_cd[0], vec_cd[1])
        if len_cd <= tolerance or len_cd >= len_ab:
            len_cd = len_ab * 0.6

        center_top = (mid_ab[0] + perp_ab[0] * height, mid_ab[1] + perp_ab[1] * height)
        new_c = (center_top[0] - unit_ab[0] * len_cd / 2, center_top[1] - unit_ab[1] * len_cd / 2)
        new_d = (center_top[0] + unit_ab[0] * len_cd / 2, center_top[1] + unit_ab[1] * len_cd / 2)

        return QuadrilateralCanonicalizer._order_ccw([a, b, new_d, new_c])

    # ------------------------------------------------------------------ #
    # Right trapezoid canonicalization
    # ------------------------------------------------------------------ #

    @staticmethod
    def _canonicalize_right_trapezoid(
        ordered: Sequence[PointTuple],
        original: Sequence[PointTuple],
        tolerance: float,
    ) -> List[PointTuple]:
        """Create right trapezoid with one pair of parallel sides and one right angle.

        First two vertices define the base, and one leg is perpendicular to the base.
        """
        a, b, c, d = ordered

        vec_ab = (b[0] - a[0], b[1] - a[1])
        len_ab = math.hypot(vec_ab[0], vec_ab[1])
        if len_ab <= tolerance:
            raise PolygonCanonicalizationError("First two vertices too close for right trapezoid construction.")

        unit_ab = (vec_ab[0] / len_ab, vec_ab[1] / len_ab)
        perp_ab = (-unit_ab[1], unit_ab[0])

        leg_ad = math.hypot(d[0] - a[0], d[1] - a[1])
        if leg_ad <= tolerance:
            leg_ad = len_ab * 0.5

        dir_sign = 1
        test_d = (a[0] + perp_ab[0] * leg_ad, a[1] + perp_ab[1] * leg_ad)
        if math.hypot(test_d[0] - d[0], test_d[1] - d[1]) > math.hypot(
            a[0] - perp_ab[0] * leg_ad - d[0], a[1] - perp_ab[1] * leg_ad - d[1]
        ):
            dir_sign = -1

        new_d = (a[0] + perp_ab[0] * leg_ad * dir_sign, a[1] + perp_ab[1] * leg_ad * dir_sign)

        vec_cd = (c[0] - d[0], c[1] - d[1])
        proj_cd_on_ab = vec_cd[0] * unit_ab[0] + vec_cd[1] * unit_ab[1]
        if abs(proj_cd_on_ab) <= tolerance:
            proj_cd_on_ab = len_ab * 0.6

        new_c = (new_d[0] + unit_ab[0] * proj_cd_on_ab, new_d[1] + unit_ab[1] * proj_cd_on_ab)

        return QuadrilateralCanonicalizer._order_ccw([a, b, new_c, new_d])

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_subtype(subtype: QuadrilateralSubtype | str | None) -> QuadrilateralSubtype | None:
        if subtype is None:
            return None
        try:
            return QuadrilateralSubtype.from_value(subtype)
        except ValueError as exc:
            raise PolygonCanonicalizationError(str(exc)) from exc

    @staticmethod
    def _dedupe_vertices(points: Sequence[PointTuple], tolerance: float) -> List[PointTuple]:
        distinct: List[PointTuple] = []
        for point in points:
            if not contains_point(distinct, point, tolerance):
                distinct.append(point)
        if len(distinct) != 4:
            raise PolygonCanonicalizationError(
                "Quadrilateral canonicalization requires exactly four distinct vertices."
            )
        return distinct

    @staticmethod
    def _order_ccw(points: Sequence[PointTuple]) -> List[PointTuple]:
        centroid = QuadrilateralCanonicalizer._compute_centroid(points)

        def angle(point: PointTuple) -> float:
            return math.atan2(point[1] - centroid[1], point[0] - centroid[0])

        ordered = sorted(points, key=angle)
        if QuadrilateralCanonicalizer._signed_area(ordered) < 0:
            ordered = [ordered[0]] + list(reversed(ordered[1:]))
        return ordered

    @staticmethod
    def _compute_centroid(points: Sequence[PointTuple]) -> PointTuple:
        cx = sum(x for x, _ in points) / len(points)
        cy = sum(y for _, y in points) / len(points)
        return cx, cy

    @staticmethod
    def _signed_area(points: Sequence[PointTuple]) -> float:
        n = len(points)
        area = 0.0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    @staticmethod
    def _ensure_non_degenerate(points: Sequence[PointTuple], tolerance: float) -> None:
        area = abs(QuadrilateralCanonicalizer._signed_area(points))
        if area <= tolerance:
            raise PolygonCanonicalizationError(
                "Provided vertices collapse to a line; cannot form a quadrilateral."
            )

    @staticmethod
    def _average_side_length(points: Sequence[PointTuple]) -> float:
        distances = []
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            distances.append(math.hypot(x2 - x1, y2 - y1))
        return sum(distances) / len(distances)

    @staticmethod
    def _align_to_original_order(
        vertices: Sequence[PointTuple],
        original: Sequence[PointTuple],
    ) -> List[PointTuple]:
        if not original:
            return list(vertices)
        first_orig = original[0]
        distances = [math.hypot(v[0] - first_orig[0], v[1] - first_orig[1]) for v in vertices]
        start_index = distances.index(min(distances))
        aligned = list(vertices[start_index:]) + list(vertices[:start_index])

        if (
            QuadrilateralCanonicalizer._signed_area(aligned)
            * QuadrilateralCanonicalizer._signed_area(original)
            < 0
        ):
            aligned = [aligned[0]] + list(reversed(aligned[1:]))
        return aligned


# ------------------------------------------------------------------ #
# Convenience functions
# ------------------------------------------------------------------ #


def canonicalize_quadrilateral(
    vertices: Sequence[PointLike],
    *,
    subtype: str | None = None,
    tolerance: float = 1e-6,
    construction_mode: str | None = None,
    enforce_square: bool = False,
) -> List[PointTuple]:
    """Convenience wrapper for quadrilateral canonicalization."""
    return QuadrilateralCanonicalizer.canonicalize(
        vertices,
        subtype=subtype,
        tolerance=tolerance,
        construction_mode=construction_mode,
        enforce_square=enforce_square,
    )


def canonicalize_rectangle(
    vertices: Sequence[PointLike],
    *,
    construction_mode: str | None = None,
    tolerance: float | None = 1e-6,
    enforce_square: bool = False,
) -> List[PointTuple]:
    """Convenience function for rectangle canonicalization (legacy API)."""
    return QuadrilateralCanonicalizer.canonicalize(
        vertices,
        subtype=QuadrilateralSubtype.RECTANGLE,
        construction_mode=construction_mode,
        tolerance=tolerance if tolerance is not None else 1e-6,
        enforce_square=enforce_square,
    )


__all__ = [
    "QuadrilateralCanonicalizer",
    "canonicalize_quadrilateral",
    "canonicalize_rectangle",
]
