"""
MatHud Circle Arc Management System

Handles creation, retrieval, and deletion of CircleArc drawables. Supports arcs
tied to existing Circle objects as well as standalone arcs defined by explicit
center/radius values.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

from drawables.circle_arc import CircleArc
from managers.edit_policy import DrawableEditPolicy, EditRule, get_drawable_edit_policy
from utils.math_utils import MathUtils

if TYPE_CHECKING:
    from canvas import Canvas
    from drawables.circle import Circle
    from drawables.drawable import Drawable
    from drawables.point import Point
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from managers.point_manager import PointManager
    from name_generator.drawable import DrawableNameGenerator


class ArcManager:
    """Manager responsible for lifecycle of CircleArc drawables."""

    def __init__(
        self,
        canvas: "Canvas",
        drawables_container: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        point_manager: "PointManager",
        drawable_manager_proxy: "DrawableManagerProxy",
    ) -> None:
        self.canvas = canvas
        self.drawables = drawables_container
        self.name_generator = name_generator
        self.dependency_manager = dependency_manager
        self.point_manager = point_manager
        self.drawable_manager = drawable_manager_proxy
        self.arc_edit_policy: Optional[DrawableEditPolicy] = get_drawable_edit_policy(
            "CircleArc"
        )

    # ------------------------------------------------------------------
    # Creation helpers
    # ------------------------------------------------------------------

    def _resolve_point(
        self,
        label: str,
        point_name: Optional[str],
        x: Optional[float],
        y: Optional[float],
    ) -> Tuple["Point", bool]:
        if point_name:
            existing = self.point_manager.get_point_by_name(point_name)
            if existing:
                return existing, False
            if x is None or y is None:
                raise ValueError(
                    f"{label} name '{point_name}' was not found and coordinates were not provided."
                )
            created_point = self.point_manager.create_point(
                x,
                y,
                name=point_name,
                extra_graphics=False,
            )
            return created_point, True

        if x is None or y is None:
            raise ValueError(f"{label} coordinates are required when no name is provided.")

        created_point = self.point_manager.create_point(
            x,
            y,
            name="",
            extra_graphics=False,
        )
        return created_point, True

    def _resolve_circle_geometry(
        self,
        circle_name: Optional[str],
        center_x: Optional[float],
        center_y: Optional[float],
        radius: Optional[float],
    ) -> Tuple[Optional["Circle"], float, float, float]:
        if circle_name:
            circle = self.drawable_manager.get_circle_by_name(circle_name)
            if not circle:
                raise ValueError(f"Circle '{circle_name}' was not found.")
            return (
                circle,
                float(circle.center.x),
                float(circle.center.y),
                float(circle.radius),
            )

        if center_x is None or center_y is None or radius is None:
            raise ValueError(
                "Circle arc creation requires either an existing circle name or explicit center and radius."
            )
        if radius <= 0:
            raise ValueError("Circle arc radius must be greater than zero.")

        return None, float(center_x), float(center_y), float(radius)

    def _generate_arc_name(
        self,
        proposed_name: Optional[str],
        point1: "Point",
        point2: "Point",
        use_major_arc: bool,
    ) -> str:
        """Generate arc name using the name generator."""
        existing_names = {arc.name for arc in self.drawables.CircleArcs}
        return self.name_generator.generate_arc_name(
            proposed_name, point1.name, point2.name, use_major_arc, existing_names
        )

    def _find_duplicate_arc(
        self,
        point1: "Point",
        point2: "Point",
        center_x: float,
        center_y: float,
        radius: float,
        use_major_arc: bool,
    ) -> Optional[CircleArc]:
        for arc in cast(List[CircleArc], self.drawables.CircleArcs):
            if (
                arc.point1 is point1
                and arc.point2 is point2
                and math.isclose(arc.center_x, center_x, rel_tol=1e-9, abs_tol=1e-9)
                and math.isclose(arc.center_y, center_y, rel_tol=1e-9, abs_tol=1e-9)
                and math.isclose(arc.radius, radius, rel_tol=1e-9, abs_tol=1e-9)
                and arc.use_major_arc == use_major_arc
            ):
                return arc
        return None

    def _resolve_circle_arc_points(
        self,
        point1_name: Optional[str],
        point1_x: Optional[float],
        point1_y: Optional[float],
        point2_name: Optional[str],
        point2_x: Optional[float],
        point2_y: Optional[float],
        point3_name: Optional[str],
        point3_x: Optional[float],
        point3_y: Optional[float],
        arc_name: Optional[str] = None,
    ) -> Tuple["Point", bool, "Point", bool, Optional["Point"], bool]:
        """Resolve up to three points from the supplied parameters.
        
        If point names are not provided, attempts to extract suggested names
        from the arc_name parameter (e.g., 'arc_AB' -> points A and B).
        Point name validation is handled by create_point via generate_point_name.
        """
        # Extract suggested point names from arc_name
        suggested_p1, suggested_p2 = self.name_generator.extract_point_names_from_arc_name(arc_name)
        
        # Resolve point 1
        point1, point1_is_new = self._resolve_arc_point(
            "Point 1", point1_name, point1_x, point1_y, suggested_p1
        )
        
        # Resolve point 2
        point2, point2_is_new = self._resolve_arc_point(
            "Point 2", point2_name, point2_x, point2_y, suggested_p2
        )

        # Resolve optional point 3
        point3: Optional["Point"] = None
        point3_is_new = False
        if any(value is not None for value in (point3_name, point3_x, point3_y)):
            point3, point3_is_new = self._resolve_point("Point 3", point3_name, point3_x, point3_y)

        return point1, point1_is_new, point2, point2_is_new, point3, point3_is_new

    def _resolve_arc_point(
        self,
        label: str,
        explicit_name: Optional[str],
        x: Optional[float],
        y: Optional[float],
        suggested_name: Optional[str],
    ) -> Tuple["Point", bool]:
        """Resolve a point for arc creation.
        
        If explicit_name is provided, uses _resolve_point (lookup by name).
        If only coordinates with suggested_name, uses create_point directly
        (like segments do) to let generate_point_name handle validation.
        """
        # Explicit name provided - use standard resolution (lookup by name first)
        if explicit_name:
            return self._resolve_point(label, explicit_name, x, y)
        
        # Coordinates required when no explicit name
        if x is None or y is None:
            raise ValueError(f"{label} coordinates are required when no name is provided.")
        
        # Use create_point directly with suggested name (like segments do)
        # create_point handles: coordinate lookup, name validation via generate_point_name
        point = self.point_manager.create_point(
            x, y,
            name=suggested_name or "",
            extra_graphics=False,
        )
        # Check if point was newly created or already existed at coordinates
        is_new = not bool(self.point_manager.get_point(x, y) and suggested_name)
        return point, True  # Assume new for arc creation purposes

    def _determine_arc_geometry(
        self,
        point1: "Point",
        point1_is_new: bool,
        point2: "Point",
        point2_is_new: bool,
        *,
        point3: Optional["Point"],
        point3_is_new: bool,
        center_point_choice: Optional[str],
        circle_name: Optional[str],
        center_x: Optional[float],
        center_y: Optional[float],
        radius: Optional[float],
    ) -> Tuple[Optional["Circle"], float, float, float, "Point", bool, "Point", bool]:
        """Determine the circle geometry either from three points or explicit data."""
        normalized_choice = center_point_choice.lower() if center_point_choice else None
        has_explicit_geometry = bool(circle_name) or any(value is not None for value in (center_x, center_y, radius))
        if has_explicit_geometry:
            normalized_choice = None

        if normalized_choice:
            if normalized_choice not in {"point1", "point2", "point3"}:
                raise ValueError("center_point_choice must be 'point1', 'point2', or 'point3'.")
            if normalized_choice == "point3" and point3 is None:
                raise ValueError("center_point_choice='point3' requires coordinates for Point 3.")

            (
                center_x_val,
                center_y_val,
                derived_radius,
                endpoint_one,
                endpoint_one_new,
                endpoint_two,
                endpoint_two_new,
            ) = self._resolve_geometry_from_three_points(
                point1,
                point1_is_new,
                point2,
                point2_is_new,
                point3,
                point3_is_new,
                normalized_choice,
            )
            return (
                None,
                center_x_val,
                center_y_val,
                derived_radius,
                endpoint_one,
                endpoint_one_new,
                endpoint_two,
                endpoint_two_new,
            )

        circle, resolved_center_x, resolved_center_y, resolved_radius = self._resolve_circle_geometry(
            circle_name,
            center_x,
            center_y,
            radius,
        )
        return circle, resolved_center_x, resolved_center_y, resolved_radius, point1, point1_is_new, point2, point2_is_new

    def _resolve_geometry_from_three_points(
        self,
        point1: "Point",
        point1_is_new: bool,
        point2: "Point",
        point2_is_new: bool,
        point3: Optional["Point"],
        point3_is_new: bool,
        center_choice: str,
    ) -> Tuple[float, float, float, "Point", bool, "Point", bool]:
        """Derive center and radius from three supplied points."""
        points_info: Dict[str, Tuple["Point", bool]] = {
            "point1": (point1, point1_is_new),
            "point2": (point2, point2_is_new),
        }
        if point3 is not None:
            points_info["point3"] = (point3, point3_is_new)

        if center_choice not in points_info:
            raise ValueError(f"{center_choice} must be defined before it can be chosen as the center.")

        center_point, _ = points_info[center_choice]
        endpoint_labels = [
            label
            for label in ("point1", "point2", "point3")
            if label != center_choice and label in points_info
        ]
        if len(endpoint_labels) != 2:
            raise ValueError("Creating a circle arc from three points requires exactly three defined points.")

        endpoint_one, endpoint_one_new = points_info[endpoint_labels[0]]
        endpoint_two, endpoint_two_new = points_info[endpoint_labels[1]]

        distances = [
            MathUtils.get_2D_distance(center_point, endpoint_one),
            MathUtils.get_2D_distance(center_point, endpoint_two),
        ]
        resolved_radius = min(distances)
        if resolved_radius <= MathUtils.EPSILON:
            raise ValueError("Circle arc radius must be greater than zero.")

        return (
            float(center_point.x),
            float(center_point.y),
            resolved_radius,
            endpoint_one,
            endpoint_one_new,
            endpoint_two,
            endpoint_two_new,
        )

    def _project_endpoints_on_circle(
        self,
        point1: "Point",
        point2: "Point",
        center_x: float,
        center_y: float,
        radius: float,
    ) -> None:
        """Project both endpoints onto the circle defined by center/radius."""
        MathUtils.project_point_onto_circle(point1, center_x, center_y, radius)
        MathUtils.project_point_onto_circle(point2, center_x, center_y, radius)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_circle_arc(
        self,
        point1_x: Optional[float] = None,
        point1_y: Optional[float] = None,
        point2_x: Optional[float] = None,
        point2_y: Optional[float] = None,
        *,
        point1_name: Optional[str] = None,
        point2_name: Optional[str] = None,
        point3_x: Optional[float] = None,
        point3_y: Optional[float] = None,
        point3_name: Optional[str] = None,
        center_point_choice: Optional[str] = None,
        circle_name: Optional[str] = None,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        radius: Optional[float] = None,
        arc_name: Optional[str] = None,
        color: Optional[str] = None,
        use_major_arc: bool = False,
        extra_graphics: bool = True,
    ) -> Optional[CircleArc]:
        self.canvas.undo_redo_manager.archive()

        (
            point1,
            point1_is_new,
            point2,
            point2_is_new,
            point3,
            point3_is_new,
        ) = self._resolve_circle_arc_points(
            point1_name,
            point1_x,
            point1_y,
            point2_name,
            point2_x,
            point2_y,
            point3_name,
            point3_x,
            point3_y,
            arc_name,
        )

        (
            circle,
            resolved_center_x,
            resolved_center_y,
            resolved_radius,
            point1,
            point1_is_new,
            point2,
            point2_is_new,
        ) = self._determine_arc_geometry(
            point1,
            point1_is_new,
            point2,
            point2_is_new,
            point3=point3,
            point3_is_new=point3_is_new,
            center_point_choice=center_point_choice,
            circle_name=circle_name,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
        )

        self._project_endpoints_on_circle(
            point1,
            point2,
            resolved_center_x,
            resolved_center_y,
            resolved_radius,
        )

        existing_arc = self._find_duplicate_arc(
            point1,
            point2,
            resolved_center_x,
            resolved_center_y,
            resolved_radius,
            use_major_arc,
        )
        if existing_arc:
            return existing_arc

        final_name = self._generate_arc_name(arc_name, point1, point2, use_major_arc)

        new_arc = CircleArc(
            point1,
            point2,
            center_x=resolved_center_x,
            center_y=resolved_center_y,
            radius=resolved_radius,
            circle=circle,
            use_major_arc=use_major_arc,
            color=color,
            name=final_name,
        )

        self.drawables.add(new_arc)
        self.dependency_manager.register_dependency(new_arc, point1)
        self.dependency_manager.register_dependency(new_arc, point2)
        if circle:
            self.dependency_manager.register_dependency(new_arc, circle)

        if extra_graphics:
            self.drawable_manager.create_drawables_from_new_connections()

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return new_arc

    def _get_circle_arc_or_raise(self, name: str) -> CircleArc:
        arc = self.get_circle_arc_by_name(name)
        if not arc:
            raise ValueError(f"Circle arc '{name}' was not found.")
        return arc

    def get_circle_arc_by_name(self, name: str) -> Optional[CircleArc]:
        for arc in cast(List[CircleArc], self.drawables.CircleArcs):
            if arc.name == name:
                return arc
        return None

    def delete_circle_arc(self, name: str) -> bool:
        arc = self.get_circle_arc_by_name(name)
        if not arc:
            return False

        self.canvas.undo_redo_manager.archive()

        self.dependency_manager.unregister_dependency(arc, arc.point1)
        self.dependency_manager.unregister_dependency(arc, arc.point2)
        if arc.circle:
            self.dependency_manager.unregister_dependency(arc, arc.circle)

        removed = self.drawables.remove(arc)
        if removed:
            self.dependency_manager.remove_drawable(arc)

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return bool(removed)

    def update_circle_arc(
        self,
        arc_name: str,
        *,
        new_color: Optional[str] = None,
        use_major_arc: Optional[bool] = None,
    ) -> bool:
        arc = self._get_circle_arc_or_raise(arc_name)

        requested_fields = self._collect_arc_requested_fields(new_color, use_major_arc)

        if self.arc_edit_policy:
            self._validate_arc_policy(requested_fields)

        self.canvas.undo_redo_manager.archive()

        if new_color is not None:
            sanitized = str(new_color).strip()
            if not sanitized:
                raise ValueError("Circle arc color cannot be empty.")
            arc.update_color(sanitized)

        if use_major_arc is not None:
            arc.set_use_major_arc(bool(use_major_arc))

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def _collect_arc_requested_fields(
        self,
        new_color: Optional[str],
        use_major_arc: Optional[bool],
    ) -> List[str]:
        requested_fields: List[str] = []

        if new_color is not None:
            requested_fields.append("color")

        if use_major_arc is not None:
            requested_fields.append("use_major_arc")

        if not requested_fields:
            raise ValueError("Provide at least one property to update.")

        return requested_fields

    def _validate_arc_policy(self, requested_fields: List[str]) -> Dict[str, EditRule]:
        if not self.arc_edit_policy:
            return {}

        validated: Dict[str, EditRule] = {}
        for field in requested_fields:
            rule = self.arc_edit_policy.get_rule(field)
            if not rule:
                raise ValueError(f"Editing field '{field}' is not permitted for circle arcs.")
            validated[field] = rule
        return validated

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def load_circle_arcs(self, arcs_data: List[Dict[str, Any]]) -> None:
        for arc_state in arcs_data:
            args = arc_state.get("args", {})
            point1_name = args.get("point1_name")
            point2_name = args.get("point2_name")
            circle_name = args.get("circle_name")
            point1 = (
                self.point_manager.get_point_by_name(point1_name)
                if point1_name
                else None
            )
            point2 = (
                self.point_manager.get_point_by_name(point2_name)
                if point2_name
                else None
            )
            if not point1 or not point2:
                continue

            try:
                self.create_circle_arc(
                    point1_x=point1.x,
                    point1_y=point1.y,
                    point2_x=point2.x,
                    point2_y=point2.y,
                    point1_name=point1.name,
                    point2_name=point2.name,
                    circle_name=circle_name,
                    center_x=args.get("center_x"),
                    center_y=args.get("center_y"),
                    radius=args.get("radius"),
                    arc_name=arc_state.get("name"),
                    color=args.get("color"),
                    use_major_arc=args.get("use_major_arc", False),
                    extra_graphics=False,
                )
            except ValueError:
                continue

        if self.canvas.draw_enabled:
            self.canvas.draw()

    def handle_circle_removed(self, circle_name: str) -> None:
        arcs_to_remove: List[str] = []
        for arc in cast(List[CircleArc], self.drawables.CircleArcs):
            if arc.circle_name == circle_name:
                arcs_to_remove.append(arc.name)

        for arc_name in arcs_to_remove:
            self.delete_circle_arc(arc_name)

