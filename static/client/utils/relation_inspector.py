"""Relation Inspector — verify geometric relations between canvas objects.

Pure static class following the ``GraphAnalyzer`` pattern: a single public
``inspect()`` entry-point that dispatches to relation-specific handlers via
a registry dict.  No browser imports, no canvas mutation.

Supported operations:
    parallel, perpendicular, collinear, concyclic, equal_length,
    similar, congruent, tangent, concurrent, point_on_line,
    point_on_circle, auto
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple


class RelationInspector:
    """Check and explain geometric relations between drawable objects."""

    RELATION_TOLERANCE = 1e-6

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    @staticmethod
    def inspect(
        operation: str,
        objects: List[Any],
        object_types: List[str],
    ) -> Dict[str, Any]:
        """Inspect a geometric relation among *objects*.

        Args:
            operation: One of the supported relation names or ``"auto"``.
            objects: Resolved drawable instances (Point, Segment, …).
            object_types: Parallel list of type tags (``"point"``, ``"segment"``, …).

        Returns:
            A result dict with at least ``operation`` and either ``result`` +
            ``explanation`` + ``tolerance_used`` + ``details``, or ``error``.
        """
        if len(objects) != len(object_types):
            return {"error": "Error: objects and object_types must have the same length"}

        # Reject NaN / inf coordinates up-front
        bad = RelationInspector._check_finite(objects, object_types)
        if bad is not None:
            return bad

        handler = RelationInspector._HANDLERS.get(operation)
        if handler is None:
            supported = ", ".join(sorted(RelationInspector._HANDLERS.keys()))
            return {"error": f"Error: unsupported operation '{operation}'. Supported: {supported}"}

        return handler(objects, object_types)

    # ------------------------------------------------------------------
    # NaN / inf guard
    # ------------------------------------------------------------------

    @staticmethod
    def _check_finite(objects: List[Any], object_types: List[str]) -> Optional[Dict[str, Any]]:
        """Return an error dict if any coordinate is non-finite, else ``None``."""
        for obj, otype in zip(objects, object_types):
            coords = RelationInspector._extract_coords(obj, otype)
            for val in coords:
                if not math.isfinite(val):
                    name = getattr(obj, "name", "?")
                    return {"error": f"Error: object '{name}' has non-finite coordinates"}
        return None

    @staticmethod
    def _extract_coords(obj: Any, otype: str) -> List[float]:
        """Pull all numeric coordinates from *obj* based on *otype*."""
        if otype == "point":
            return [float(obj.x), float(obj.y)]
        if otype in ("segment", "vector"):
            seg = RelationInspector._as_segment(obj, otype)
            return [
                float(seg.point1.x),
                float(seg.point1.y),
                float(seg.point2.x),
                float(seg.point2.y),
            ]
        if otype == "circle":
            return [float(obj.center.x), float(obj.center.y), float(obj.radius)]
        if otype == "triangle":
            coords: List[float] = []
            for v in obj.get_vertices():
                coords.extend([float(v.x), float(v.y)])
            return coords
        if otype == "ellipse":
            return [
                float(obj.center.x),
                float(obj.center.y),
                float(obj.radius_x),
                float(obj.radius_y),
            ]
        if otype == "rectangle":
            coords_r: List[float] = []
            for v in obj.get_vertices():
                coords_r.extend([float(v.x), float(v.y)])
            return coords_r
        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_segment(obj: Any, obj_type: str) -> Any:
        """Unwrap a Vector to its underlying Segment."""
        if obj_type == "vector":
            return obj.segment
        return obj

    @staticmethod
    def _direction(seg: Any) -> Tuple[float, float]:
        """Return the direction vector ``(dx, dy)`` of a segment."""
        return (
            float(seg.point2.x) - float(seg.point1.x),
            float(seg.point2.y) - float(seg.point1.y),
        )

    @staticmethod
    def _magnitude(dx: float, dy: float) -> float:
        return math.hypot(dx, dy)

    @staticmethod
    def _relative_tolerance(mag1: float, mag2: float) -> float:
        """Scale-aware tolerance: ``RELATION_TOLERANCE * max(1, max(mag1, mag2))``."""
        scale = max(1.0, mag1, mag2)
        return RelationInspector.RELATION_TOLERANCE * scale

    @staticmethod
    def _seg_length(seg: Any) -> float:
        dx = float(seg.point2.x) - float(seg.point1.x)
        dy = float(seg.point2.y) - float(seg.point1.y)
        return math.hypot(dx, dy)

    @staticmethod
    def _ok(
        operation: str,
        result: bool,
        explanation: str,
        tolerance: float,
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "operation": operation,
            "result": result,
            "explanation": explanation,
            "tolerance_used": tolerance,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Relation handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_parallel(objects: List[Any], object_types: List[str]) -> Dict[str, Any]:
        if len(objects) != 2 or not all(t in ("segment", "vector") for t in object_types):
            return {"error": "Error: 'parallel' requires exactly 2 segments or vectors"}

        s1 = RelationInspector._as_segment(objects[0], object_types[0])
        s2 = RelationInspector._as_segment(objects[1], object_types[1])

        d1 = RelationInspector._direction(s1)
        d2 = RelationInspector._direction(s2)
        m1 = RelationInspector._magnitude(*d1)
        m2 = RelationInspector._magnitude(*d2)

        if m1 < RelationInspector.RELATION_TOLERANCE:
            return {"error": "Error: first segment/vector has zero length"}
        if m2 < RelationInspector.RELATION_TOLERANCE:
            return {"error": "Error: second segment/vector has zero length"}

        cross = abs(d1[0] * d2[1] - d1[1] * d2[0])
        tol = RelationInspector._relative_tolerance(m1, m2)
        # Normalize cross product by magnitudes for a scale-independent test
        sin_angle = cross / (m1 * m2)
        angle_deg = math.degrees(math.asin(min(sin_angle, 1.0)))

        is_parallel = sin_angle < RelationInspector.RELATION_TOLERANCE
        expl = (
            f"Segments are parallel (angle: {angle_deg:.4f}\u00b0)"
            if is_parallel
            else f"Segments are not parallel (angle: {angle_deg:.4f}\u00b0)"
        )
        return RelationInspector._ok("parallel", is_parallel, expl, tol, {"angle_between": angle_deg})

    @staticmethod
    def _check_perpendicular(objects: List[Any], object_types: List[str]) -> Dict[str, Any]:
        if len(objects) != 2 or not all(t in ("segment", "vector") for t in object_types):
            return {"error": "Error: 'perpendicular' requires exactly 2 segments or vectors"}

        s1 = RelationInspector._as_segment(objects[0], object_types[0])
        s2 = RelationInspector._as_segment(objects[1], object_types[1])

        d1 = RelationInspector._direction(s1)
        d2 = RelationInspector._direction(s2)
        m1 = RelationInspector._magnitude(*d1)
        m2 = RelationInspector._magnitude(*d2)

        if m1 < RelationInspector.RELATION_TOLERANCE:
            return {"error": "Error: first segment/vector has zero length"}
        if m2 < RelationInspector.RELATION_TOLERANCE:
            return {"error": "Error: second segment/vector has zero length"}

        dot = d1[0] * d2[0] + d1[1] * d2[1]
        tol = RelationInspector._relative_tolerance(m1, m2)
        cos_angle = dot / (m1 * m2)
        # Clamp for acos safety
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(abs(cos_angle)))

        is_perp = abs(cos_angle) < RelationInspector.RELATION_TOLERANCE
        expl = (
            f"Segments are perpendicular (angle: {angle_deg:.4f}\u00b0)"
            if is_perp
            else f"Segments are not perpendicular (angle: {angle_deg:.4f}\u00b0)"
        )
        return RelationInspector._ok("perpendicular", is_perp, expl, tol, {"angle_between": angle_deg})

    @staticmethod
    def _check_collinear(objects: List[Any], object_types: List[str]) -> Dict[str, Any]:
        if len(objects) < 3 or not all(t == "point" for t in object_types):
            return {"error": "Error: 'collinear' requires 3 or more points"}

        from utils.math_utils import MathUtils

        p0 = objects[0]
        p1 = objects[1]
        tol = RelationInspector.RELATION_TOLERANCE

        for i in range(2, len(objects)):
            pi = objects[i]
            # Use cross-product divided by both magnitudes for a truly
            # dimensionless (scale-invariant) collinearity measure.
            ax = float(p1.x) - float(p0.x)
            ay = float(p1.y) - float(p0.y)
            bx = float(pi.x) - float(p0.x)
            by = float(pi.y) - float(p0.y)
            cross = abs(ax * by - ay * bx)
            mag_a = RelationInspector._magnitude(ax, ay)
            mag_b = RelationInspector._magnitude(bx, by)
            denom = mag_a * mag_b
            # If either vector is near-zero the points are coincident with p0,
            # which is trivially collinear.
            if denom < RelationInspector.RELATION_TOLERANCE:
                continue
            if cross / denom > tol:
                return RelationInspector._ok(
                    "collinear",
                    False,
                    f"Points are not collinear (point at index {i} deviates)",
                    tol,
                    {"first_deviant_index": i},
                )

        return RelationInspector._ok(
            "collinear",
            True,
            "All points are collinear",
            tol,
            {"point_count": len(objects)},
        )

    @staticmethod
    def _check_concyclic(objects: List[Any], object_types: List[str]) -> Dict[str, Any]:
        if len(objects) < 4 or not all(t == "point" for t in object_types):
            return {"error": "Error: 'concyclic' requires 4 or more points"}

        from utils.math_utils import MathUtils

        p0, p1, p2 = objects[0], objects[1], objects[2]
        try:
            cx, cy, r = MathUtils.circumcenter(
                float(p0.x),
                float(p0.y),
                float(p1.x),
                float(p1.y),
                float(p2.x),
                float(p2.y),
            )
        except ValueError:
            return RelationInspector._ok(
                "concyclic",
                False,
                "First three points are collinear — no common circle exists",
                RelationInspector.RELATION_TOLERANCE,
                {"reason": "collinear_first_three"},
            )

        tol = RelationInspector.RELATION_TOLERANCE * max(1.0, r)
        for i in range(3, len(objects)):
            pi = objects[i]
            dist = math.hypot(float(pi.x) - cx, float(pi.y) - cy)
            if abs(dist - r) > tol:
                return RelationInspector._ok(
                    "concyclic",
                    False,
                    f"Point at index {i} is not on the common circle (distance from center: {dist:.6f}, radius: {r:.6f})",
                    tol,
                    {"circumcenter": [cx, cy], "circumradius": r, "first_deviant_index": i},
                )

        return RelationInspector._ok(
            "concyclic",
            True,
            f"All {len(objects)} points lie on a common circle (center: ({cx:.4f}, {cy:.4f}), radius: {r:.4f})",
            tol,
            {"circumcenter": [cx, cy], "circumradius": r},
        )

    @staticmethod
    def _check_equal_length(objects: List[Any], object_types: List[str]) -> Dict[str, Any]:
        if len(objects) != 2 or not all(t in ("segment", "vector") for t in object_types):
            return {"error": "Error: 'equal_length' requires exactly 2 segments or vectors"}

        s1 = RelationInspector._as_segment(objects[0], object_types[0])
        s2 = RelationInspector._as_segment(objects[1], object_types[1])

        len1 = RelationInspector._seg_length(s1)
        len2 = RelationInspector._seg_length(s2)
        tol = RelationInspector._relative_tolerance(len1, len2)
        diff = abs(len1 - len2)
        equal = diff < tol

        expl = (
            f"Segments have equal length ({len1:.6f})"
            if equal
            else f"Segments have different lengths ({len1:.6f} vs {len2:.6f})"
        )
        return RelationInspector._ok(
            "equal_length",
            equal,
            expl,
            tol,
            {"length1": len1, "length2": len2, "difference": diff},
        )

    @staticmethod
    def _triangle_side_lengths(tri: Any) -> List[float]:
        """Return sorted side lengths for a triangle."""
        verts = list(tri.get_vertices())
        if len(verts) != 3:
            return []
        sides = []
        for i in range(3):
            j = (i + 1) % 3
            dx = float(verts[j].x) - float(verts[i].x)
            dy = float(verts[j].y) - float(verts[i].y)
            sides.append(math.hypot(dx, dy))
        sides.sort()
        return sides

    @staticmethod
    def _check_similar(objects: List[Any], object_types: List[str]) -> Dict[str, Any]:
        if len(objects) != 2 or not all(t == "triangle" for t in object_types):
            return {"error": "Error: 'similar' requires exactly 2 triangles"}

        s1 = RelationInspector._triangle_side_lengths(objects[0])
        s2 = RelationInspector._triangle_side_lengths(objects[1])
        if len(s1) != 3 or len(s2) != 3:
            return {"error": "Error: could not extract 3 vertices from triangle"}
        if s1[0] < RelationInspector.RELATION_TOLERANCE or s2[0] < RelationInspector.RELATION_TOLERANCE:
            return {"error": "Error: degenerate triangle (zero-length side)"}

        ratios = [s1[i] / s2[i] for i in range(3)]
        tol = RelationInspector.RELATION_TOLERANCE * max(1.0, max(ratios))
        similar = all(abs(ratios[i] - ratios[0]) < tol for i in range(1, 3))

        expl = (
            f"Triangles are similar (scale factor: {ratios[0]:.6f})"
            if similar
            else f"Triangles are not similar (side ratios: {ratios[0]:.4f}, {ratios[1]:.4f}, {ratios[2]:.4f})"
        )
        return RelationInspector._ok(
            "similar",
            similar,
            expl,
            tol,
            {"side_ratios": ratios, "scale_factor": ratios[0] if similar else None},
        )

    @staticmethod
    def _check_congruent(objects: List[Any], object_types: List[str]) -> Dict[str, Any]:
        if len(objects) != 2 or not all(t == "triangle" for t in object_types):
            return {"error": "Error: 'congruent' requires exactly 2 triangles"}

        s1 = RelationInspector._triangle_side_lengths(objects[0])
        s2 = RelationInspector._triangle_side_lengths(objects[1])
        if len(s1) != 3 or len(s2) != 3:
            return {"error": "Error: could not extract 3 vertices from triangle"}
        if s1[0] < RelationInspector.RELATION_TOLERANCE or s2[0] < RelationInspector.RELATION_TOLERANCE:
            return {"error": "Error: degenerate triangle (zero-length side)"}

        max_side = max(max(s1), max(s2))
        tol = RelationInspector.RELATION_TOLERANCE * max(1.0, max_side)
        cong = all(abs(s1[i] - s2[i]) < tol for i in range(3))

        expl = (
            f"Triangles are congruent (side lengths: {s1[0]:.4f}, {s1[1]:.4f}, {s1[2]:.4f})"
            if cong
            else f"Triangles are not congruent (sides1: {s1}, sides2: {s2})"
        )
        return RelationInspector._ok(
            "congruent",
            cong,
            expl,
            tol,
            {"sides1": s1, "sides2": s2},
        )

    @staticmethod
    def _check_tangent(objects: List[Any], object_types: List[str]) -> Dict[str, Any]:
        if len(objects) != 2:
            return {"error": "Error: 'tangent' requires exactly 2 objects"}

        types_set = set(object_types)
        sorted_types = sorted(object_types)

        # segment/vector + circle
        if ("segment" in types_set or "vector" in types_set) and "circle" in types_set:
            seg_idx = 0 if object_types[0] in ("segment", "vector") else 1
            cir_idx = 1 - seg_idx
            return RelationInspector._tangent_segment_circle(
                objects[seg_idx],
                object_types[seg_idx],
                objects[cir_idx],
            )

        # circle + circle
        if sorted_types == ["circle", "circle"]:
            return RelationInspector._tangent_circle_circle(objects[0], objects[1])

        return {"error": "Error: 'tangent' supports segment+circle or circle+circle"}

    @staticmethod
    def _tangent_segment_circle(seg_obj: Any, seg_type: str, circle: Any) -> Dict[str, Any]:
        from utils.math_utils import MathUtils

        seg = RelationInspector._as_segment(seg_obj, seg_type)
        x1, y1 = float(seg.point1.x), float(seg.point1.y)
        x2, y2 = float(seg.point2.x), float(seg.point2.y)
        if RelationInspector._magnitude(x2 - x1, y2 - y1) < RelationInspector.RELATION_TOLERANCE:
            return {"error": "Error: segment has zero length"}

        cx, cy = float(circle.center.x), float(circle.center.y)
        r = float(circle.radius)

        try:
            fx, fy = MathUtils.perpendicular_foot(cx, cy, x1, y1, x2, y2)
        except ValueError as exc:
            return {"error": f"Error: {exc}"}

        dist = math.hypot(fx - cx, fy - cy)
        tol = RelationInspector.RELATION_TOLERANCE * max(1.0, r)
        line_tangent = abs(dist - r) < tol

        # Verify the tangent point (perpendicular foot) lies on the segment.
        # Compute parameter t: foot = P1 + t*(P2 - P1); on-segment when 0 <= t <= 1.
        dx_seg, dy_seg = x2 - x1, y2 - y1
        len_sq = dx_seg * dx_seg + dy_seg * dy_seg
        t_param = ((fx - x1) * dx_seg + (fy - y1) * dy_seg) / len_sq if len_sq > 0 else 0.0
        foot_on_segment = -tol <= t_param <= 1.0 + tol

        # If the foot is off the segment, check whether either endpoint
        # touches the circle (the segment could still be tangent at an endpoint).
        endpoint_tangent = False
        if not foot_on_segment:
            d1 = math.hypot(x1 - cx, y1 - cy)
            d2 = math.hypot(x2 - cx, y2 - cy)
            endpoint_tangent = abs(d1 - r) < tol or abs(d2 - r) < tol

        is_tangent = line_tangent and (foot_on_segment or endpoint_tangent)

        if is_tangent:
            expl = f"Segment is tangent to circle (distance from center to line: {dist:.6f}, radius: {r:.6f})"
        elif line_tangent and not foot_on_segment:
            expl = (
                f"The line through the segment is tangent to the circle, but the tangent "
                f"point does not lie on the segment (t={t_param:.4f})"
            )
        else:
            expl = f"Segment is not tangent to circle (distance from center to line: {dist:.6f}, radius: {r:.6f})"

        return RelationInspector._ok(
            "tangent",
            is_tangent,
            expl,
            tol,
            {
                "distance_to_line": dist,
                "radius": r,
                "tangent_point": [fx, fy],
                "foot_on_segment": foot_on_segment,
                "t_parameter": t_param,
            },
        )

    @staticmethod
    def _tangent_circle_circle(c1: Any, c2: Any) -> Dict[str, Any]:
        cx1, cy1, r1 = float(c1.center.x), float(c1.center.y), float(c1.radius)
        cx2, cy2, r2 = float(c2.center.x), float(c2.center.y), float(c2.radius)
        d = math.hypot(cx2 - cx1, cy2 - cy1)

        tol = RelationInspector.RELATION_TOLERANCE * max(1.0, r1, r2, d)

        ext_tangent = abs(d - (r1 + r2)) < tol
        int_tangent = abs(d - abs(r1 - r2)) < tol
        is_tangent = ext_tangent or int_tangent

        if ext_tangent:
            kind = "externally tangent"
        elif int_tangent:
            kind = "internally tangent"
        else:
            kind = "not tangent"

        expl = f"Circles are {kind} (center distance: {d:.6f}, r1+r2: {r1 + r2:.6f}, |r1-r2|: {abs(r1 - r2):.6f})"
        return RelationInspector._ok(
            "tangent",
            is_tangent,
            expl,
            tol,
            {
                "center_distance": d,
                "r1": r1,
                "r2": r2,
                "externally_tangent": ext_tangent,
                "internally_tangent": int_tangent,
            },
        )

    @staticmethod
    def _check_concurrent(objects: List[Any], object_types: List[str]) -> Dict[str, Any]:
        if len(objects) < 3 or not all(t in ("segment", "vector") for t in object_types):
            return {"error": "Error: 'concurrent' requires 3 or more segments/vectors"}

        segs = [RelationInspector._as_segment(o, t) for o, t in zip(objects, object_types)]
        # Validate no zero-length segments
        for i, seg in enumerate(segs):
            if RelationInspector._seg_length(seg) < RelationInspector.RELATION_TOLERANCE:
                return {"error": f"Error: segment at index {i} has zero length"}

        # Find intersection of first two extended lines
        maybe_ix, maybe_iy = RelationInspector._line_line_intersection(segs[0], segs[1])
        if maybe_ix is None or maybe_iy is None:
            return RelationInspector._ok(
                "concurrent",
                False,
                "First two lines are parallel — no single point of concurrence",
                RelationInspector.RELATION_TOLERANCE,
                {"reason": "parallel_pair"},
            )

        ix: float = maybe_ix
        iy: float = maybe_iy

        # Check remaining lines pass through the intersection
        tol = RelationInspector.RELATION_TOLERANCE
        for i in range(2, len(segs)):
            dist = RelationInspector._point_to_line_distance(ix, iy, segs[i])
            ref = max(1.0, abs(ix), abs(iy), RelationInspector._seg_length(segs[i]))
            if dist / ref > tol:
                return RelationInspector._ok(
                    "concurrent",
                    False,
                    f"Lines are not concurrent (line at index {i} misses intersection point)",
                    tol,
                    {"intersection_of_first_two": [ix, iy], "first_deviant_index": i},
                )

        return RelationInspector._ok(
            "concurrent",
            True,
            f"All {len(objects)} lines are concurrent at ({ix:.4f}, {iy:.4f})",
            tol,
            {"intersection": [ix, iy]},
        )

    @staticmethod
    def _line_line_intersection(
        s1: Any,
        s2: Any,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute intersection of infinite lines through two segments.

        Returns ``(x, y)`` or ``(None, None)`` if parallel.
        """
        d1 = RelationInspector._direction(s1)
        d2 = RelationInspector._direction(s2)
        det = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(det) < RelationInspector.RELATION_TOLERANCE:
            return None, None

        # Parametric: P = s1.point1 + t * d1
        # s2.point1 + u * d2 = s1.point1 + t * d1
        dx = float(s2.point1.x) - float(s1.point1.x)
        dy = float(s2.point1.y) - float(s1.point1.y)
        t = (dx * d2[1] - dy * d2[0]) / det
        x = float(s1.point1.x) + t * d1[0]
        y = float(s1.point1.y) + t * d1[1]
        return x, y

    @staticmethod
    def _point_to_line_distance(px: float, py: float, seg: Any) -> float:
        """Distance from point to the *infinite* line through ``seg``."""
        x1, y1 = float(seg.point1.x), float(seg.point1.y)
        x2, y2 = float(seg.point2.x), float(seg.point2.y)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < RelationInspector.RELATION_TOLERANCE:
            return math.hypot(px - x1, py - y1)
        return abs(dx * (y1 - py) - dy * (x1 - px)) / length

    @staticmethod
    def _check_point_on_line(objects: List[Any], object_types: List[str]) -> Dict[str, Any]:
        if len(objects) != 2:
            return {"error": "Error: 'point_on_line' requires exactly 2 objects (point + segment/vector)"}

        # Allow either order
        pt_idx: Optional[int] = None
        seg_idx: Optional[int] = None
        for i, t in enumerate(object_types):
            if t == "point" and pt_idx is None:
                pt_idx = i
            elif t in ("segment", "vector") and seg_idx is None:
                seg_idx = i

        if pt_idx is None or seg_idx is None:
            return {"error": "Error: 'point_on_line' requires one point and one segment/vector"}

        pt = objects[pt_idx]
        seg = RelationInspector._as_segment(objects[seg_idx], object_types[seg_idx])

        if RelationInspector._seg_length(seg) < RelationInspector.RELATION_TOLERANCE:
            return {"error": "Error: segment has zero length"}

        dist = RelationInspector._point_to_line_distance(float(pt.x), float(pt.y), seg)
        ref = max(1.0, RelationInspector._seg_length(seg))
        tol = RelationInspector.RELATION_TOLERANCE * ref
        on_line = dist < tol

        expl = (
            f"Point lies on the extended line (distance: {dist:.6f})"
            if on_line
            else f"Point does not lie on the extended line (distance: {dist:.6f})"
        )
        return RelationInspector._ok("point_on_line", on_line, expl, tol, {"distance": dist})

    @staticmethod
    def _check_point_on_circle(objects: List[Any], object_types: List[str]) -> Dict[str, Any]:
        if len(objects) != 2:
            return {"error": "Error: 'point_on_circle' requires exactly 2 objects (point + circle)"}

        pt_idx: Optional[int] = None
        cir_idx: Optional[int] = None
        for i, t in enumerate(object_types):
            if t == "point" and pt_idx is None:
                pt_idx = i
            elif t == "circle" and cir_idx is None:
                cir_idx = i

        if pt_idx is None or cir_idx is None:
            return {"error": "Error: 'point_on_circle' requires one point and one circle"}

        pt = objects[pt_idx]
        c = objects[cir_idx]
        cx, cy, r = float(c.center.x), float(c.center.y), float(c.radius)
        dist = math.hypot(float(pt.x) - cx, float(pt.y) - cy)
        tol = RelationInspector.RELATION_TOLERANCE * max(1.0, r)
        on_circle = abs(dist - r) < tol

        expl = (
            f"Point lies on the circle (distance from center: {dist:.6f}, radius: {r:.6f})"
            if on_circle
            else f"Point does not lie on the circle (distance from center: {dist:.6f}, radius: {r:.6f})"
        )
        return RelationInspector._ok(
            "point_on_circle",
            on_circle,
            expl,
            tol,
            {"distance_from_center": dist, "radius": r, "deviation": abs(dist - r)},
        )

    # ------------------------------------------------------------------
    # Auto-inspect
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_inspect(objects: List[Any], object_types: List[str]) -> Dict[str, Any]:
        """Run all applicable checks for the given type combination."""
        results: List[Dict[str, Any]] = []
        n = len(objects)

        checks: List[str] = []

        # Determine which checks apply based on types
        seg_vec_types = {"segment", "vector"}
        all_seg_vec = all(t in seg_vec_types for t in object_types)
        all_points = all(t == "point" for t in object_types)
        all_triangles = all(t == "triangle" for t in object_types)
        type_set = set(object_types)

        if n == 2 and all_seg_vec:
            checks = ["parallel", "perpendicular", "equal_length"]
        elif n >= 3 and all_seg_vec:
            checks = ["concurrent"]
            if n == 3:
                # Also check pairwise parallel/perpendicular
                pass
        elif n >= 3 and all_points:
            checks = ["collinear"]
            if n >= 4:
                checks.append("concyclic")
        elif n == 2 and all_triangles:
            checks = ["similar", "congruent"]
        elif n == 2 and type_set == {"segment", "circle"} or n == 2 and type_set == {"vector", "circle"}:
            checks = ["tangent"]
        elif n == 2 and type_set == {"circle"}:
            checks = ["tangent"]
        elif n == 2 and ("point" in type_set) and (type_set & seg_vec_types):
            checks = ["point_on_line"]
        elif n == 2 and type_set == {"point", "circle"}:
            checks = ["point_on_circle"]

        if not checks:
            return {
                "operation": "auto",
                "result": None,
                "explanation": "No applicable relation checks for the given object types",
                "tolerance_used": RelationInspector.RELATION_TOLERANCE,
                "details": {"object_types": object_types, "checks_run": []},
            }

        for op in checks:
            handler = RelationInspector._HANDLERS.get(op)
            if handler and handler != RelationInspector._auto_inspect:
                res = handler(objects, object_types)
                results.append(res)

        # Summarize
        true_relations = [r["operation"] for r in results if r.get("result") is True]
        return {
            "operation": "auto",
            "result": bool(true_relations),
            "explanation": (
                f"Detected relations: {', '.join(true_relations)}"
                if true_relations
                else "No geometric relations detected"
            ),
            "tolerance_used": RelationInspector.RELATION_TOLERANCE,
            "details": {"checks_run": [r["operation"] for r in results], "results": results},
        }

    # ------------------------------------------------------------------
    # Handler registry (dict dispatch)
    # ------------------------------------------------------------------

    _HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {}


# Populate handler registry after class body (avoids forward-ref issues)
RelationInspector._HANDLERS = {
    "parallel": RelationInspector._check_parallel,
    "perpendicular": RelationInspector._check_perpendicular,
    "collinear": RelationInspector._check_collinear,
    "concyclic": RelationInspector._check_concyclic,
    "equal_length": RelationInspector._check_equal_length,
    "similar": RelationInspector._check_similar,
    "congruent": RelationInspector._check_congruent,
    "tangent": RelationInspector._check_tangent,
    "concurrent": RelationInspector._check_concurrent,
    "point_on_line": RelationInspector._check_point_on_line,
    "point_on_circle": RelationInspector._check_point_on_circle,
    "auto": RelationInspector._auto_inspect,
}
