"""Triangle canonicalization with subtype support.

This module provides canonicalization for triangles, transforming arbitrary
vertex sets into well-formed triangles of specified subtypes.

Key Features:
    - Equilateral triangle canonicalization via centroid and average side length
    - Isosceles triangle canonicalization with apex identification
    - Right triangle canonicalization with perpendicular leg enforcement
    - Right isosceles canonicalization combining both constraints
    - CCW vertex ordering with original order alignment
"""

from __future__ import annotations

import math
from typing import List, Sequence

from .common import (
    PointLike,
    PointTuple,
    PolygonCanonicalizationError,
    contains_point,
    point_like_to_tuple,
)
from utils.polygon_subtypes import TriangleSubtype


class TriangleCanonicalizer:
    """Best-fit triangle canonicalization helpers with subtype support."""

    @staticmethod
    def canonicalize(
        vertices: Sequence[PointLike],
        *,
        subtype: TriangleSubtype | str | None = None,
        tolerance: float = 1e-6,
    ) -> List[PointTuple]:
        points = [point_like_to_tuple(vertex) for vertex in vertices]
        distinct = TriangleCanonicalizer._dedupe_vertices(points, tolerance)
        original_order = list(distinct)
        ordered = TriangleCanonicalizer._order_ccw(distinct)

        TriangleCanonicalizer._ensure_non_degenerate(ordered, tolerance)

        normalized_subtype = TriangleCanonicalizer._normalize_subtype(subtype)
        if normalized_subtype is None:
            return TriangleCanonicalizer._align_to_original_order(ordered, original_order)

        if normalized_subtype is TriangleSubtype.EQUILATERAL:
            result = TriangleCanonicalizer._canonicalize_equilateral(ordered, original_order)
        elif normalized_subtype is TriangleSubtype.ISOSCELES:
            result = TriangleCanonicalizer._canonicalize_isosceles(ordered, original_order, tolerance)
        elif normalized_subtype is TriangleSubtype.RIGHT:
            result = TriangleCanonicalizer._canonicalize_right(ordered, original_order, tolerance)
        elif normalized_subtype is TriangleSubtype.RIGHT_ISOSCELES:
            result = TriangleCanonicalizer._canonicalize_right_isosceles(ordered, original_order, tolerance)
        else:
            raise PolygonCanonicalizationError(f"Unsupported triangle subtype '{subtype}'.")

        return TriangleCanonicalizer._align_to_original_order(result, original_order)

    # ------------------------------------------------------------------ #
    # Subtype canonicalization helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _canonicalize_equilateral(
        ordered: Sequence[PointTuple],
        original: Sequence[PointTuple],
    ) -> List[PointTuple]:
        centroid = TriangleCanonicalizer._compute_centroid(ordered)
        first_target = original[0]
        anchor_vec = (first_target[0] - centroid[0], first_target[1] - centroid[1])
        if math.hypot(anchor_vec[0], anchor_vec[1]) <= 1e-12:
            anchor_vec = (ordered[0][0] - centroid[0], ordered[0][1] - centroid[1])
        orientation = math.atan2(anchor_vec[1], anchor_vec[0])

        side_length = TriangleCanonicalizer._average_side_length(ordered)
        if side_length <= 1e-12:
            raise PolygonCanonicalizationError("Triangle side length too small for equilateral canonicalization.")
        radius = side_length / math.sqrt(3.0)

        new_vertices: List[PointTuple] = []
        for k in range(3):
            angle = orientation + k * (2.0 * math.pi / 3.0)
            x = centroid[0] + radius * math.cos(angle)
            y = centroid[1] + radius * math.sin(angle)
            new_vertices.append((x, y))
        return new_vertices

    @staticmethod
    def _canonicalize_isosceles(
        ordered: Sequence[PointTuple],
        original: Sequence[PointTuple],
        tolerance: float,
    ) -> List[PointTuple]:
        apex_index = TriangleCanonicalizer._identify_isosceles_apex(ordered)
        base_indices = [i for i in range(3) if i != apex_index]
        apex = ordered[apex_index]
        base_start = ordered[base_indices[0]]
        base_end = ordered[base_indices[1]]

        base_vec = (base_end[0] - base_start[0], base_end[1] - base_start[1])
        base_len = math.hypot(base_vec[0], base_vec[1])
        if base_len <= tolerance:
            raise PolygonCanonicalizationError("Isosceles base is too small.")

        base_mid = ((base_start[0] + base_end[0]) / 2.0, (base_start[1] + base_end[1]) / 2.0)
        legs = [
            math.hypot(apex[0] - base_start[0], apex[1] - base_start[1]),
            math.hypot(apex[0] - base_end[0], apex[1] - base_end[1]),
        ]
        leg_length = sum(legs) / 2.0

        if leg_length <= tolerance:
            raise PolygonCanonicalizationError("Isosceles legs are too small.")

        half_base = base_len / 2.0
        height_sq = leg_length * leg_length - half_base * half_base
        if height_sq < -tolerance:
            raise PolygonCanonicalizationError("Isosceles legs too short to satisfy geometry.")
        height = math.sqrt(max(height_sq, 0.0))

        unit_base = (base_vec[0] / base_len, base_vec[1] / base_len)
        normal = (-unit_base[1], unit_base[0])
        direction = (apex[0] - base_mid[0], apex[1] - base_mid[1])
        if normal[0] * direction[0] + normal[1] * direction[1] < 0:
            normal = (-normal[0], -normal[1])

        new_base_start = (base_mid[0] - unit_base[0] * half_base, base_mid[1] - unit_base[1] * half_base)
        new_base_end = (base_mid[0] + unit_base[0] * half_base, base_mid[1] + unit_base[1] * half_base)
        new_apex = (base_mid[0] + normal[0] * height, base_mid[1] + normal[1] * height)

        raw = [new_base_start, new_base_end, new_apex]
        return TriangleCanonicalizer._order_ccw(raw)

    @staticmethod
    def _canonicalize_right(
        ordered: Sequence[PointTuple],
        original: Sequence[PointTuple],
        tolerance: float,
    ) -> List[PointTuple]:
        right_index = TriangleCanonicalizer._identify_right_vertex(ordered)
        right_vertex = ordered[right_index]
        leg_indices = [i for i in range(3) if i != right_index]
        leg_a = ordered[leg_indices[0]]
        leg_b = ordered[leg_indices[1]]

        v1 = (leg_a[0] - right_vertex[0], leg_a[1] - right_vertex[1])
        v2 = (leg_b[0] - right_vertex[0], leg_b[1] - right_vertex[1])
        len1 = math.hypot(v1[0], v1[1])
        len2 = math.hypot(v2[0], v2[1])
        if len1 <= tolerance or len2 <= tolerance:
            raise PolygonCanonicalizationError("Right triangle legs too small.")

        u = (v1[0] / len1, v1[1] / len1)
        perp = (-u[1], u[0])
        if perp[0] * v2[0] + perp[1] * v2[1] < 0:
            perp = (-perp[0], -perp[1])

        new_leg_a = (right_vertex[0] + u[0] * len1, right_vertex[1] + u[1] * len1)
        new_leg_b = (right_vertex[0] + perp[0] * len2, right_vertex[1] + perp[1] * len2)
        raw = [right_vertex, new_leg_a, new_leg_b]
        return TriangleCanonicalizer._order_ccw(raw)

    @staticmethod
    def _canonicalize_right_isosceles(
        ordered: Sequence[PointTuple],
        original: Sequence[PointTuple],
        tolerance: float,
    ) -> List[PointTuple]:
        right_index = TriangleCanonicalizer._identify_right_vertex(ordered)
        right_vertex = ordered[right_index]
        leg_indices = [i for i in range(3) if i != right_index]
        leg_a = ordered[leg_indices[0]]
        leg_b = ordered[leg_indices[1]]

        v1 = (leg_a[0] - right_vertex[0], leg_a[1] - right_vertex[1])
        v2 = (leg_b[0] - right_vertex[0], leg_b[1] - right_vertex[1])
        len1 = math.hypot(v1[0], v1[1])
        len2 = math.hypot(v2[0], v2[1])
        leg_len = (len1 + len2) / 2.0
        if leg_len <= tolerance:
            raise PolygonCanonicalizationError("Right isosceles legs too small.")

        u = (v1[0] / len1 if len1 > tolerance else v2[0] / len2, v1[1] / len1 if len1 > tolerance else v2[1] / len2)
        if len1 <= tolerance:
            u = (v2[0] / len2, v2[1] / len2)
        perp = (-u[1], u[0])
        if perp[0] * v2[0] + perp[1] * v2[1] < 0:
            perp = (-perp[0], -perp[1])

        new_leg_a = (right_vertex[0] + u[0] * leg_len, right_vertex[1] + u[1] * leg_len)
        new_leg_b = (right_vertex[0] + perp[0] * leg_len, right_vertex[1] + perp[1] * leg_len)
        raw = [right_vertex, new_leg_a, new_leg_b]
        return TriangleCanonicalizer._order_ccw(raw)

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_subtype(subtype: TriangleSubtype | str | None) -> TriangleSubtype | None:
        if subtype is None:
            return None
        try:
            return TriangleSubtype.from_value(subtype)
        except ValueError as exc:
            raise PolygonCanonicalizationError(str(exc)) from exc

    @staticmethod
    def _dedupe_vertices(points: Sequence[PointTuple], tolerance: float) -> List[PointTuple]:
        distinct: List[PointTuple] = []
        for point in points:
            if not contains_point(distinct, point, tolerance):
                distinct.append(point)
        if len(distinct) != 3:
            raise PolygonCanonicalizationError("Triangle canonicalization requires exactly three distinct vertices.")
        return distinct

    @staticmethod
    def _order_ccw(points: Sequence[PointTuple]) -> List[PointTuple]:
        centroid = TriangleCanonicalizer._compute_centroid(points)

        def angle(point: PointTuple) -> float:
            return math.atan2(point[1] - centroid[1], point[0] - centroid[0])

        ordered = sorted(points, key=angle)
        if TriangleCanonicalizer._orientation(ordered) < 0:
            ordered = [ordered[0], ordered[2], ordered[1]]
        return ordered

    @staticmethod
    def _compute_centroid(points: Sequence[PointTuple]) -> PointTuple:
        cx = sum(x for x, _ in points) / len(points)
        cy = sum(y for _, y in points) / len(points)
        return cx, cy

    @staticmethod
    def _orientation(points: Sequence[PointTuple]) -> float:
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

    @staticmethod
    def _ensure_non_degenerate(points: Sequence[PointTuple], tolerance: float) -> None:
        area = abs(TriangleCanonicalizer._orientation(points)) / 2.0
        if area <= tolerance:
            raise PolygonCanonicalizationError("Provided vertices collapse to a line; cannot form a triangle.")

    @staticmethod
    def _average_side_length(points: Sequence[PointTuple]) -> float:
        distances = []
        for i in range(3):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % 3]
            distances.append(math.hypot(x2 - x1, y2 - y1))
        return sum(distances) / 3.0

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

        if TriangleCanonicalizer._orientation(aligned) * TriangleCanonicalizer._orientation(original) < 0:
            aligned = [aligned[0], aligned[2], aligned[1]]
        return aligned

    @staticmethod
    def _identify_isosceles_apex(points: Sequence[PointTuple]) -> int:
        d01 = math.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1])
        d12 = math.hypot(points[1][0] - points[2][0], points[1][1] - points[2][1])
        d20 = math.hypot(points[2][0] - points[0][0], points[2][1] - points[0][1])
        candidates = [
            (abs(d12 - d20), 0),
            (abs(d01 - d20), 1),
            (abs(d01 - d12), 2),
        ]
        return min(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _identify_right_vertex(points: Sequence[PointTuple]) -> int:
        best_index = 0
        best_dot = float("inf")
        for i in range(3):
            prev_point = points[(i - 1) % 3]
            curr_point = points[i]
            next_point = points[(i + 1) % 3]
            v1 = (prev_point[0] - curr_point[0], prev_point[1] - curr_point[1])
            v2 = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])
            dot = abs(v1[0] * v2[0] + v1[1] * v2[1])
            if dot < best_dot:
                best_dot = dot
                best_index = i
        return best_index


def canonicalize_triangle(
    vertices: Sequence[PointLike],
    *,
    subtype: str | None = None,
    tolerance: float = 1e-6,
) -> List[PointTuple]:
    """Convenience wrapper mirroring the legacy API."""
    return TriangleCanonicalizer.canonicalize(
        vertices,
        subtype=subtype,
        tolerance=tolerance,
    )


__all__ = ["TriangleCanonicalizer", "canonicalize_triangle"]
