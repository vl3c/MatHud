"""
Geometric Construction Manager for MatHud

Manages geometric constructions (midpoints, perpendicular bisectors, angle
bisectors, perpendicular/parallel lines) by computing coordinates and creating
standard Point and Segment drawables via existing managers.

Constructions produce static snapshots — they create ordinary drawables at
computed positions, with no reactive re-computation if source objects move.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from constants import default_color
from utils.math_utils import MathUtils

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.point_manager import PointManager
    from managers.segment_manager import SegmentManager
    from managers.angle_manager import AngleManager
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from name_generator.drawable import DrawableNameGenerator
    from drawables.point import Point
    from drawables.segment import Segment
    from drawables.angle import Angle
    from drawables.circle import Circle
    from drawables.triangle import Triangle

# Default construction line length in math units
DEFAULT_CONSTRUCTION_LENGTH = 6.0


class ConstructionManager:
    """Manages geometric constructions that produce standard Point/Segment drawables.

    Follows the TangentManager pattern: a dedicated manager that computes
    geometry, then delegates to PointManager/SegmentManager for creation.

    Composite constructions (creating multiple primitives) use the
    ``suspend_archiving`` pattern from AngleManager so the entire
    construction collapses into a single undo step.

    Attributes:
        canvas: Reference to the parent Canvas instance
        drawables: Container for all drawable objects
        point_manager: Manager for creating point drawables
        segment_manager: Manager for creating segment drawables
        angle_manager: Manager for looking up angle drawables
        name_generator: Generates unique names for drawables
        dependency_manager: Tracks object dependencies
        proxy: Manager proxy for inter-manager communication
    """

    def __init__(
        self,
        canvas: "Canvas",
        drawables: "DrawablesContainer",
        point_manager: "PointManager",
        segment_manager: "SegmentManager",
        angle_manager: "AngleManager",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        proxy: "DrawableManagerProxy",
    ) -> None:
        self.canvas: "Canvas" = canvas
        self.drawables: "DrawablesContainer" = drawables
        self.point_manager: "PointManager" = point_manager
        self.segment_manager: "SegmentManager" = segment_manager
        self.angle_manager: "AngleManager" = angle_manager
        self.name_generator: "DrawableNameGenerator" = name_generator
        self.dependency_manager: "DrawableDependencyManager" = dependency_manager
        self.proxy: "DrawableManagerProxy" = proxy

    # ------------------- Helpers -------------------

    def _archive_for_undo(self) -> None:
        """Archive current state before making changes for undo support."""
        undo_redo = getattr(self.canvas, "undo_redo_manager", None)
        if undo_redo:
            undo_redo.archive()

    def _get_point(self, name: str) -> "Point":
        """Retrieve a point by name, raising ValueError if not found."""
        for pt in self.drawables.Points:
            if getattr(pt, "name", None) == name:
                return pt  # type: ignore[return-value]
        raise ValueError(f"Point '{name}' not found")

    def _get_segment(self, name: str) -> "Segment":
        """Retrieve a segment by name, raising ValueError if not found."""
        for seg in self.drawables.Segments:
            if getattr(seg, "name", None) == name:
                return seg  # type: ignore[return-value]
        raise ValueError(f"Segment '{name}' not found")

    def _get_angle(self, name: str) -> "Angle":
        """Retrieve an angle by name, raising ValueError if not found."""
        angle = self.angle_manager.get_angle_by_name(name)
        if angle is None:
            raise ValueError(f"Angle '{name}' not found")
        return angle

    def _get_triangle(self, name: str) -> "Triangle":
        """Retrieve a triangle by name, raising ValueError if not found."""
        tri = self.drawables.get_triangle_by_name(name)
        if tri is None:
            raise ValueError(f"Triangle '{name}' not found")
        return tri  # type: ignore[return-value]

    def _segment_direction(self, seg: "Segment") -> Tuple[float, float]:
        """Return the unit vector from a segment's first point to its second.

        Raises:
            ValueError: If the segment has zero length (coincident endpoints)
        """
        dx = seg.point2.x - seg.point1.x
        dy = seg.point2.y - seg.point1.y
        length = math.hypot(dx, dy)
        if length == 0.0 or not math.isfinite(length):
            raise ValueError(f"Degenerate segment '{getattr(seg, 'name', '')}': endpoints coincide")
        return dx / length, dy / length

    @staticmethod
    def _resolve_length(length: Optional[float]) -> float:
        """The requested construction length, or the default; it must be a positive number."""
        if length is None:
            return DEFAULT_CONSTRUCTION_LENGTH
        value = float(length)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"length must be a positive number, got {length}")
        return value

    @staticmethod
    def _centered_endpoints(
        center: Tuple[float, float], direction: Tuple[float, float], length: float
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Endpoints of a segment of the given length centred on a point along a unit direction."""
        half = length / 2
        (cx, cy), (ux, uy) = center, direction
        return (cx - ux * half, cy - uy * half), (cx + ux * half, cy + uy * half)

    @staticmethod
    def _require_point_off_line(pt: "Point", seg: "Segment", foot_x: float, foot_y: float) -> None:
        """Raise when the point lies on the segment's line, relative to the segment's scale."""
        distance = math.hypot(pt.x - foot_x, pt.y - foot_y)
        scale = max(
            math.hypot(seg.point2.x - seg.point1.x, seg.point2.y - seg.point1.y),
            math.hypot(pt.x - seg.point1.x, pt.y - seg.point1.y),
        )
        if distance <= MathUtils.EPSILON * scale:
            raise ValueError(
                f"Point '{pt.name}' lies on the line through segment '{seg.name}', so the perpendicular "
                "from it would have zero length; use a point off that line."
            )

    @staticmethod
    def _bisector_direction(
        vx: float, vy: float, p1x: float, p1y: float, p2x: float, p2y: float
    ) -> Tuple[float, float, bool]:
        """Unit vector from the vertex along the angle bisector, and whether the angle is straight.

        The bisector of a straight angle is perpendicular to its arms, on the side
        counterclockwise from the first arm, where the angle's arc is drawn.

        Raises:
            ValueError: If an arm has zero length
        """
        a1x, a1y = p1x - vx, p1y - vy
        a2x, a2y = p2x - vx, p2y - vy
        len1 = math.hypot(a1x, a1y)
        len2 = math.hypot(a2x, a2y)
        if len1 == 0.0 or len2 == 0.0:
            raise ValueError("An arm of the angle has zero length: its bisector is undefined")
        u1x, u1y = a1x / len1, a1y / len1
        u2x, u2y = a2x / len2, a2y / len2
        cross = u1x * u2y - u1y * u2x
        dot = u1x * u2x + u1y * u2y
        if dot < 0 and abs(cross) <= MathUtils.EPSILON:
            return -u1y, u1x, True
        bx, by = u1x + u2x, u1y + u2y
        blen = math.hypot(bx, by)
        return bx / blen, by / blen, False

    # ------------------- Public Construction Methods -------------------

    def create_midpoint(
        self,
        p1_name: Optional[str] = None,
        p2_name: Optional[str] = None,
        *,
        segment_name: Optional[str] = None,
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> "Point":
        """Create a point at the midpoint of two points or a segment.

        Either provide ``p1_name`` and ``p2_name``, or ``segment_name``.

        Args:
            p1_name: Name of the first point
            p2_name: Name of the second point
            segment_name: Name of the segment (alternative to point names)
            name: Optional name for the created midpoint
            color: Optional color for the midpoint

        Returns:
            The created Point drawable

        Raises:
            ValueError: If neither points nor segment specified, or not found
        """
        if segment_name:
            seg = self._get_segment(segment_name)
            p1, p2 = seg.point1, seg.point2
        elif p1_name and p2_name:
            p1 = self._get_point(p1_name)
            p2 = self._get_point(p2_name)
        else:
            raise ValueError("Provide either 'segment_name' or both 'p1_name' and 'p2_name'")

        mx, my = MathUtils.get_2D_midpoint(p1, p2)

        point = self.point_manager.create_point(
            mx,
            my,
            name=name or "",
            color=color or default_color,
            extra_graphics=False,
        )
        return point

    def create_perpendicular_bisector(
        self,
        segment_name: str,
        *,
        length: Optional[float] = None,
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> "Segment":
        """Create a segment that is the perpendicular bisector of a given segment.

        The resulting segment passes through the midpoint and is perpendicular
        to the original segment.

        Args:
            segment_name: Name of the segment to bisect
            length: Total length of the bisector segment (default: 6.0)
            name: Optional name for the created segment
            color: Optional color for the segment

        Returns:
            The created Segment drawable

        Raises:
            ValueError: If segment not found
        """
        seg = self._get_segment(segment_name)
        length = self._resolve_length(length)
        if color is None:
            color = getattr(seg, "color", default_color)

        midpoint = MathUtils.get_2D_midpoint(seg.point1, seg.point2)
        ux, uy = self._segment_direction(seg)
        (x1, y1), (x2, y2) = self._centered_endpoints(midpoint, (-uy, ux), length)

        self._archive_for_undo()
        segment = self.segment_manager.create_segment(
            x1,
            y1,
            x2,
            y2,
            name=name or "",
            color=color,
            extra_graphics=True,
        )
        return segment

    def create_perpendicular_from_point(
        self,
        point_name: str,
        segment_name: str,
        *,
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Dict[str, Union["Point", "Segment"]]:
        """Drop a perpendicular from a point to a segment.

        Creates the foot point on the segment and a segment from the given
        point to the foot. Both are created as a single undo step.

        Args:
            point_name: Name of the point to project
            segment_name: Name of the target segment
            name: Optional name for the perpendicular segment
            color: Optional color for created drawables

        Returns:
            Dict with keys 'foot' (Point) and 'segment' (Segment)

        Raises:
            ValueError: If point or segment not found, or the point lies on the
                segment's line (the perpendicular would have zero length)
        """
        pt = self._get_point(point_name)
        seg = self._get_segment(segment_name)
        if color is None:
            color = default_color

        foot_x, foot_y = MathUtils.perpendicular_foot(
            pt.x,
            pt.y,
            seg.point1.x,
            seg.point1.y,
            seg.point2.x,
            seg.point2.y,
        )
        self._require_point_off_line(pt, seg, foot_x, foot_y)

        # Use suspend_archiving pattern for composite construction
        undo_manager = self.canvas.undo_redo_manager
        baseline_state = undo_manager.capture_state()
        undo_manager.suspend_archiving()

        try:
            foot_point = self.point_manager.create_point(
                foot_x,
                foot_y,
                name="",
                color=color,
                extra_graphics=False,
            )

            perp_segment = self.segment_manager.create_segment(
                pt.x,
                pt.y,
                foot_x,
                foot_y,
                name=name or "",
                color=color,
                extra_graphics=True,
            )

            undo_manager.push_undo_state(baseline_state)

            if self.canvas.draw_enabled:
                self.canvas.draw()

            return {"foot": foot_point, "segment": perp_segment}
        except Exception:
            undo_manager.restore_state(baseline_state, redraw=self.canvas.draw_enabled)
            raise
        finally:
            undo_manager.resume_archiving()

    def create_angle_bisector(
        self,
        vertex_name: Optional[str] = None,
        p1_name: Optional[str] = None,
        p2_name: Optional[str] = None,
        *,
        angle_name: Optional[str] = None,
        length: Optional[float] = None,
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> "Segment":
        """Create a segment along the bisector of an angle.

        Either provide ``vertex_name``, ``p1_name``, ``p2_name`` directly,
        or ``angle_name`` to look up an existing Angle object.

        Args:
            vertex_name: Name of the vertex point
            p1_name: Name of the first arm endpoint
            p2_name: Name of the second arm endpoint
            angle_name: Name of an existing Angle (alternative to point names)
            length: Total length of the bisector segment (default: 6.0)
            name: Optional name for the created segment
            color: Optional color for the segment

        Returns:
            The created Segment drawable

        Raises:
            ValueError: If inputs not found or angle is degenerate
        """
        if angle_name:
            angle = self._get_angle(angle_name)
            # Extract vertex and arm endpoints from the angle's segments
            seg1 = angle.segment1
            seg2 = angle.segment2
            vertex = angle.vertex_point
            vx, vy = vertex.x, vertex.y

            # Determine which endpoint of each segment is NOT the vertex
            if abs(seg1.point1.x - vx) < MathUtils.EPSILON and abs(seg1.point1.y - vy) < MathUtils.EPSILON:
                p1x, p1y = seg1.point2.x, seg1.point2.y
            else:
                p1x, p1y = seg1.point1.x, seg1.point1.y

            if abs(seg2.point1.x - vx) < MathUtils.EPSILON and abs(seg2.point1.y - vy) < MathUtils.EPSILON:
                p2x, p2y = seg2.point2.x, seg2.point2.y
            else:
                p2x, p2y = seg2.point1.x, seg2.point1.y
        elif vertex_name and p1_name and p2_name:
            v = self._get_point(vertex_name)
            p1 = self._get_point(p1_name)
            p2 = self._get_point(p2_name)
            vx, vy = v.x, v.y
            p1x, p1y = p1.x, p1.y
            p2x, p2y = p2.x, p2.y
        else:
            raise ValueError("Provide either 'angle_name' or all of 'vertex_name', 'p1_name', 'p2_name'")

        length = self._resolve_length(length)
        if color is None:
            color = default_color

        dx, dy, straight = self._bisector_direction(vx, vy, p1x, p1y, p2x, p2y)

        # For reflex angles, negate the direction so the bisector points
        # into the reflex arc instead of the minor arc (a straight angle has none).
        if angle_name and getattr(angle, "is_reflex", False) and not straight:
            dx, dy = -dx, -dy

        # Create segment from vertex along bisector direction
        x1 = vx
        y1 = vy
        x2 = vx + dx * length
        y2 = vy + dy * length

        self._archive_for_undo()
        segment = self.segment_manager.create_segment(
            x1,
            y1,
            x2,
            y2,
            name=name or "",
            color=color,
            extra_graphics=True,
        )
        return segment

    def create_parallel_line(
        self,
        segment_name: str,
        point_name: str,
        *,
        length: Optional[float] = None,
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> "Segment":
        """Create a segment through a point, parallel to a given segment.

        The resulting segment is centered on the given point and has the
        same slope as the reference segment.

        Args:
            segment_name: Name of the reference segment
            point_name: Name of the point the parallel line passes through
            length: Total length of the parallel segment (default: 6.0)
            name: Optional name for the created segment
            color: Optional color for the segment

        Returns:
            The created Segment drawable

        Raises:
            ValueError: If segment or point not found
        """
        seg = self._get_segment(segment_name)
        pt = self._get_point(point_name)

        length = self._resolve_length(length)
        if color is None:
            color = getattr(seg, "color", default_color)

        direction = self._segment_direction(seg)
        (x1, y1), (x2, y2) = self._centered_endpoints((pt.x, pt.y), direction, length)

        self._archive_for_undo()
        segment = self.segment_manager.create_segment(
            x1,
            y1,
            x2,
            y2,
            name=name or "",
            color=color,
            extra_graphics=True,
        )
        return segment

    # ------------------- Circle Construction Methods -------------------

    def _triangle_vertices(
        self,
        triangle_name: Optional[str],
        p1_name: Optional[str],
        p2_name: Optional[str],
        p3_name: Optional[str],
    ) -> Tuple[float, float, float, float, float, float]:
        """Resolve three vertex coordinates from a triangle name or three point names."""
        if triangle_name:
            tri = self._get_triangle(triangle_name)
            verts = list(tri.get_vertices())
            if len(verts) != 3:
                raise ValueError(f"Triangle '{triangle_name}' does not have exactly 3 vertices")
            return (verts[0].x, verts[0].y, verts[1].x, verts[1].y, verts[2].x, verts[2].y)
        elif p1_name and p2_name and p3_name:
            p1 = self._get_point(p1_name)
            p2 = self._get_point(p2_name)
            p3 = self._get_point(p3_name)
            return (p1.x, p1.y, p2.x, p2.y, p3.x, p3.y)
        else:
            raise ValueError("Provide either 'triangle_name' or all of 'p1_name', 'p2_name', 'p3_name'")

    def create_circumcircle(
        self,
        *,
        triangle_name: Optional[str] = None,
        p1_name: Optional[str] = None,
        p2_name: Optional[str] = None,
        p3_name: Optional[str] = None,
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> "Circle":
        """Create the circumscribed circle of a triangle or three points.

        The circumcircle passes through all three vertices.

        Args:
            triangle_name: Name of an existing triangle
            p1_name: Name of the first point (alternative to triangle_name)
            p2_name: Name of the second point
            p3_name: Name of the third point
            name: Optional name for the created circle
            color: Optional color for the circle

        Returns:
            The created Circle drawable

        Raises:
            ValueError: If inputs not found or points are collinear
        """
        x1, y1, x2, y2, x3, y3 = self._triangle_vertices(triangle_name, p1_name, p2_name, p3_name)
        if color is None:
            color = default_color

        cx, cy, radius = MathUtils.circumcenter(x1, y1, x2, y2, x3, y3)

        # Use suspend_archiving since create_circle internally archives
        undo_manager = self.canvas.undo_redo_manager
        baseline_state = undo_manager.capture_state()
        undo_manager.suspend_archiving()

        try:
            circle = self.proxy.create_circle(
                cx,
                cy,
                radius,
                name=name or "",
                color=color,
                extra_graphics=True,
            )
            undo_manager.push_undo_state(baseline_state)
            if self.canvas.draw_enabled:
                self.canvas.draw()
            return circle  # type: ignore[return-value]
        except Exception:
            undo_manager.restore_state(baseline_state, redraw=self.canvas.draw_enabled)
            raise
        finally:
            undo_manager.resume_archiving()

    def create_incircle(
        self,
        triangle_name: str,
        *,
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> "Circle":
        """Create the inscribed circle of a triangle.

        The incircle is tangent to all three sides.

        Args:
            triangle_name: Name of an existing triangle
            name: Optional name for the created circle
            color: Optional color for the circle

        Returns:
            The created Circle drawable

        Raises:
            ValueError: If triangle not found or is degenerate
        """
        tri = self._get_triangle(triangle_name)
        verts = list(tri.get_vertices())
        if len(verts) != 3:
            raise ValueError(f"Triangle '{triangle_name}' does not have exactly 3 vertices")

        if color is None:
            color = default_color

        cx, cy, radius = MathUtils.incenter_and_inradius(
            verts[0].x,
            verts[0].y,
            verts[1].x,
            verts[1].y,
            verts[2].x,
            verts[2].y,
        )

        # Use suspend_archiving since create_circle internally archives
        undo_manager = self.canvas.undo_redo_manager
        baseline_state = undo_manager.capture_state()
        undo_manager.suspend_archiving()

        try:
            circle = self.proxy.create_circle(
                cx,
                cy,
                radius,
                name=name or "",
                color=color,
                extra_graphics=True,
            )
            undo_manager.push_undo_state(baseline_state)
            if self.canvas.draw_enabled:
                self.canvas.draw()
            return circle  # type: ignore[return-value]
        except Exception:
            undo_manager.restore_state(baseline_state, redraw=self.canvas.draw_enabled)
            raise
        finally:
            undo_manager.resume_archiving()
