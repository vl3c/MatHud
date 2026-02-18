"""
MatHud Client-Side Workspace Persistence and State Management System

Client-side workspace manager that runs in the browser using Brython. Manages workspace
operations including saving, loading, listing and deleting workspaces through AJAX
communication with the Flask backend server.

Key Features:
    - Canvas state serialization with complete object preservation
    - AJAX-based communication with backend workspace API
    - Incremental object restoration with dependency resolution
    - Error handling and validation for workspace operations
    - Support for all drawable object types and relationships
    - Computation history preservation and restoration

Workspace Operations:
    - Save: Serializes current canvas state and sends to server via AJAX
    - Load: Requests workspace data from server and restores canvas state
    - List: Retrieves available workspace names from server storage
    - Delete: Removes workspace files from server persistent storage

Object Restoration:
    - Points: Coordinate-based geometric primitives
    - Segments: Line segments with endpoint dependency resolution
    - Vectors: Directed segments with origin/tip point relationships
    - Triangles: Three-vertex polygons with automatic edge detection
    - Rectangles: Canonicalized polygons rebuilt via the unified polygon manager
    - Circles: Circular objects with center point dependencies
    - Ellipses: Elliptical objects with center and rotation parameters
    - Functions: Mathematical function expressions with domain settings
    - Colored Areas: Bounded regions with drawable object relationships
    - Angles: Angular measurements with vertex and arm dependencies

Dependencies:
    - browser: AJAX communication for backend workspace operations
    - utils.math_utils: Geometric calculations for object restoration
    - json: State serialization and deserialization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, cast

import json

from browser import ajax, document
from constants import (
    default_area_fill_color,
    default_area_opacity,
    default_closed_shape_resolution,
)
from drawables.label_render_mode import LabelRenderMode
from drawables.bars_plot import BarsPlot
from drawables.continuous_plot import ContinuousPlot
from drawables.discrete_plot import DiscretePlot
from drawables.plot import Plot
from utils.math_utils import MathUtils
from managers.polygon_type import PolygonType
from utils.polygon_canonicalizer import (
    PolygonCanonicalizationError,
    canonicalize_rectangle,
)

if TYPE_CHECKING:
    from canvas import Canvas
    from drawables.point import Point
    from drawables.segment import Segment


class WorkspaceManager:
    """
    Client-side workspace manager that handles workspace operations via AJAX communication.

    This class handles all workspace-related operations and their associated error handling.
    It works with the canvas to save and restore workspace states, including all geometric
    objects and computations. Runs in the browser using Brython.

    Attributes:
        canvas: The canvas instance to manage workspaces for.
    """

    def __init__(self, canvas: "Canvas") -> None:
        """Initialize workspace manager with canvas reference."""
        self.canvas: "Canvas" = canvas

    def save_workspace(self, name: Optional[str] = None) -> str:
        """
        Save the current workspace state to server via AJAX.

        Serializes the complete canvas state including all geometric objects,
        computations, and settings, then sends it to the Flask backend server
        for persistent storage.

        Args:
            name (str, optional): Name for the workspace. If None, saves as "current".

        Returns:
            str: Success or error message from the save operation.
        """

        def on_complete(req: Any) -> str:
            return self._parse_save_workspace_response(req, name)

        return self._execute_sync_json_request(
            method="POST",
            url="/save_workspace",
            payload=self._build_save_workspace_payload(name),
            on_complete=on_complete,
            error_prefix="Error saving workspace",
        )

    def _parse_save_workspace_response(self, req: Any, name: Optional[str]) -> str:
        try:
            response = self._response_from_request(req)
            if self._response_is_success(response):
                return f'Workspace "{name if name else "current"}" saved successfully.'
            return self._format_workspace_error("saving", response)
        except Exception as e:
            return f"Error saving workspace: {str(e)}"

    def _build_save_workspace_payload(self, name: Optional[str]) -> Dict[str, Any]:
        return {
            "state": self._snapshot_persistable_canvas_state(),
            "name": name,
        }

    def _snapshot_persistable_canvas_state(self) -> Any:
        state = self.canvas.get_canvas_state()
        return self._drop_plot_derived_bars(state)

    def _drop_plot_derived_bars(self, state: Any) -> Any:
        # Bars are derived from DiscretePlot parameters and are rebuilt on load.
        # Do not persist them in saved workspaces.
        try:
            if isinstance(state, dict) and "Bars" in state:
                del state["Bars"]
        except Exception:
            pass
        return state

    def _create_points(self, state: Dict[str, Any]) -> None:
        """Create points from workspace state."""
        if "Points" not in state:
            return
        for item_state in state["Points"]:
            self._restore_point(item_state)

    def _restore_point(self, item_state: Dict[str, Any]) -> None:
        self.canvas.create_point(
            item_state["args"]["position"]["x"],
            item_state["args"]["position"]["y"],
            name=item_state.get("name", ""),
        )

    def _create_labels(self, state: Dict[str, Any]) -> None:
        """Create standalone labels from workspace state."""
        labels_state = state.get("Labels")
        if not labels_state:
            return
        for item_state in labels_state:
            self._restore_label(item_state)

    def _restore_label(self, item_state: Any) -> None:
        args = self._get_label_args(item_state)
        if args is None:
            return
        label_position = self._get_label_position(args)
        if label_position is None:
            return
        x, y = label_position
        label = self._create_label_from_state(item_state, args, x, y)
        if label is None:
            return
        self._apply_label_visibility(label, args)
        self._apply_label_reference_scale(label, args)
        self._apply_label_render_mode(label, args)

    def _get_label_args(self, item_state: Any) -> Optional[Dict[str, Any]]:
        args = item_state.get("args", {}) if isinstance(item_state, dict) else {}
        if not isinstance(args, dict):
            return None
        return args

    def _get_label_position(self, args: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        pos = args.get("position", {}) if isinstance(args.get("position", {}), dict) else {}
        try:
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))
            return x, y
        except Exception:
            return None

    def _create_label_from_state(
        self,
        item_state: Any,
        args: Dict[str, Any],
        x: float,
        y: float,
    ) -> Optional[Any]:
        text = str(args.get("text", "") or "")
        name = str(item_state.get("name", "") or "")
        color = args.get("color", None)
        font_size = args.get("font_size", None)
        rotation_degrees = args.get("rotation_degrees", None)
        try:
            return self.canvas.create_label(
                x,
                y,
                text,
                name=name,
                color=color,
                font_size=font_size,
                rotation_degrees=rotation_degrees,
            )
        except Exception:
            return None

    def _apply_label_visibility(self, label: Any, args: Dict[str, Any]) -> None:
        try:
            label.visible = bool(args.get("visible", True))
        except Exception:
            pass

    def _apply_label_reference_scale(self, label: Any, args: Dict[str, Any]) -> None:
        try:
            if hasattr(label, "update_reference_scale"):
                label.update_reference_scale(args.get("reference_scale_factor", None))
        except Exception:
            pass

    def _apply_label_render_mode(self, label: Any, args: Dict[str, Any]) -> None:
        try:
            render_mode_raw = args.get("render_mode", None)
            if render_mode_raw is not None:
                label.render_mode = LabelRenderMode.from_state(render_mode_raw)
        except Exception:
            pass

    def _create_segments(self, state: Dict[str, Any]) -> None:
        """Create segments from workspace state."""
        if "Segments" not in state:
            return
        for item_state in state["Segments"]:
            self._restore_segment(item_state)

    def _restore_segment(self, item_state: Dict[str, Any]) -> None:
        args = item_state.get("args", {})
        p1, p2 = self._resolve_segment_points(args)
        if not p1 or not p2:
            return

        label_args = self._get_segment_label_args(args)
        segment = self.canvas.create_segment(
            p1.x,
            p1.y,
            p2.x,
            p2.y,
            name=item_state.get("name", ""),
            label_text=str(label_args.get("text", "") or ""),
            label_visible=bool(label_args.get("visible", False)),
        )
        self._restore_segment_label(segment, label_args)

    def _resolve_segment_points(self, args: Dict[str, Any]) -> Tuple[Optional["Point"], Optional["Point"]]:
        p1 = self._get_point_from_state(args.get("p1"), args.get("p1_coords"))
        p2 = self._get_point_from_state(args.get("p2"), args.get("p2_coords"))
        if not p1 or not p2:
            return None, None
        return self._reconcile_segment_endpoints(p1, p2, args)

    def _get_segment_label_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        label_args = args.get("label", {})
        if isinstance(label_args, dict):
            return label_args
        return {}

    def _restore_segment_label(self, segment: Any, label_args: Dict[str, Any]) -> None:
        try:
            embedded_label = self._get_embedded_segment_label(segment)
            if embedded_label is None:
                return

            self._apply_segment_label_font_size(embedded_label, label_args)
            self._apply_segment_label_rotation(embedded_label, label_args)
            self._apply_segment_label_render_mode(embedded_label, label_args)
        except Exception:
            return

    def _get_embedded_segment_label(self, segment: Any) -> Any:
        return getattr(segment, "label", None)

    def _apply_segment_label_font_size(self, embedded_label: Any, label_args: Dict[str, Any]) -> None:
        if "font_size" in label_args and label_args.get("font_size") is not None:
            embedded_label.update_font_size(float(label_args.get("font_size", 0.0)))

    def _apply_segment_label_rotation(self, embedded_label: Any, label_args: Dict[str, Any]) -> None:
        if "rotation_degrees" in label_args and label_args.get("rotation_degrees") is not None:
            embedded_label.update_rotation(float(label_args.get("rotation_degrees", 0.0)))

    def _apply_segment_label_render_mode(self, embedded_label: Any, label_args: Dict[str, Any]) -> None:
        render_mode_raw = label_args.get("render_mode")
        if isinstance(render_mode_raw, dict):
            embedded_label.render_mode = LabelRenderMode.from_state(render_mode_raw)

    def _get_point_from_state(
        self,
        name: Optional[str],
        coords: Optional[List[float]],
    ) -> Optional["Point"]:
        point: Optional["Point"] = self.canvas.get_point_by_name(name) if name else None
        if point:
            return point
        return self._get_point_from_coords(coords)

    def _get_point_from_coords(self, coords: Optional[List[float]]) -> Optional["Point"]:
        if not coords or len(coords) != 2:
            return None
        return self.canvas.get_point(coords[0], coords[1])

    def _reconcile_segment_endpoints(
        self,
        p1: "Point",
        p2: "Point",
        args: Dict[str, Any],
    ) -> Tuple["Point", "Point"]:
        p1_coords: Optional[List[float]] = args.get("p1_coords")
        p2_coords: Optional[List[float]] = args.get("p2_coords")

        if self._point_matches_coords(p1, p1_coords) and self._point_matches_coords(p2, p2_coords):
            return p1, p2

        if self._point_matches_coords(p1, p2_coords) and self._point_matches_coords(p2, p1_coords):
            return p2, p1

        replacement_p1 = self._get_point_from_coords(p1_coords)
        replacement_p2 = self._get_point_from_coords(p2_coords)

        return (
            replacement_p1 if replacement_p1 else p1,
            replacement_p2 if replacement_p2 else p2,
        )

    def _point_matches_coords(self, point: Optional["Point"], coords: Optional[List[float]]) -> bool:
        if not point or not coords or len(coords) != 2:
            return False
        return bool(MathUtils.point_matches_coordinates(point, coords[0], coords[1]))

    def _create_vectors(self, state: Dict[str, Any]) -> None:
        """Create vectors from workspace state."""
        if "Vectors" not in state:
            return
        for item_state in state["Vectors"]:
            self._restore_vector(item_state)

    def _restore_vector(self, item_state: Dict[str, Any]) -> None:
        origin_point_name, tip_point_name = self._get_vector_point_names(item_state)
        if not origin_point_name or not tip_point_name:
            self._warn_vector_missing_point_names(item_state)
            return

        origin_point: Optional["Point"] = self.canvas.get_point_by_name(origin_point_name)
        tip_point: Optional["Point"] = self.canvas.get_point_by_name(tip_point_name)
        if not origin_point or not tip_point:
            self._warn_vector_missing_points(item_state, origin_point_name, tip_point_name)
            return

        self.canvas.create_vector(
            origin_point.x,
            origin_point.y,
            tip_point.x,
            tip_point.y,
            name=item_state.get("name", ""),
        )

    def _warn_vector_missing_point_names(self, item_state: Dict[str, Any]) -> None:
        print(
            f"Warning: Vector '{item_state.get('name', 'Unnamed')}' is missing origin or tip point name in its state. Skipping."
        )

    def _warn_vector_missing_points(
        self,
        item_state: Dict[str, Any],
        origin_point_name: str,
        tip_point_name: str,
    ) -> None:
        print(
            f"Warning: Could not find origin ('{origin_point_name}') or tip ('{tip_point_name}') point for vector '{item_state.get('name', 'Unnamed')}' in the canvas. Skipping."
        )

    def _get_vector_point_names(self, item_state: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        args = item_state.get("args", {})
        if not isinstance(args, dict):
            return None, None
        return args.get("origin"), args.get("tip")

    def _create_triangles(self, state: Dict[str, Any]) -> None:
        """Create triangles from workspace state."""
        if "Triangles" not in state:
            return
        for item_state in state["Triangles"]:
            self._restore_triangle(item_state)

    def _restore_triangle(self, item_state: Dict[str, Any]) -> None:
        p1: Optional["Point"] = self.canvas.get_point_by_name(item_state["args"]["p1"])
        p2: Optional["Point"] = self.canvas.get_point_by_name(item_state["args"]["p2"])
        p3: Optional["Point"] = self.canvas.get_point_by_name(item_state["args"]["p3"])
        if p1 and p2 and p3:
            self.canvas.create_polygon(
                [
                    (p1.x, p1.y),
                    (p2.x, p2.y),
                    (p3.x, p3.y),
                ],
                polygon_type=PolygonType.TRIANGLE,
                name=item_state.get("name", ""),
            )

    def _create_rectangles(self, state: Dict[str, Any]) -> None:
        """Create rectangles from workspace state using the polygon manager."""
        if "Rectangles" not in state:
            return

        for item_state in state["Rectangles"]:
            self._restore_rectangle(item_state)

    def _restore_rectangle(self, item_state: Dict[str, Any]) -> None:
        rect_name: str = item_state.get("name", "UnnamedRectangle")
        arg_point_names = self._rectangle_point_names(item_state)
        if not all(arg_point_names):
            self._warn_rectangle_missing_point_names(rect_name)
            return

        points = [self.canvas.get_point_by_name(name) for name in arg_point_names]
        if not all(points):
            missing_names: List[str] = [str(arg_point_names[i]) for i, p in enumerate(points) if not p]
            self._warn_rectangle_missing_points(rect_name, missing_names)
            return

        resolved_vertices = self._resolve_rectangle_vertices(points, rect_name)
        if resolved_vertices is None:
            return

        self.canvas.create_polygon(
            resolved_vertices,
            polygon_type=PolygonType.RECTANGLE,
            name=rect_name,
        )

    def _warn_rectangle_missing_point_names(self, rect_name: str) -> None:
        print(
            f"Warning: Rectangle '{rect_name}' is missing one or more point names (p1, p2, p3, p4) in its state. Skipping."
        )

    def _warn_rectangle_missing_points(self, rect_name: str, missing_names: List[str]) -> None:
        print(
            f"Warning: Could not find one or more points ({', '.join(missing_names)}) for rectangle '{rect_name}' in the canvas. Skipping."
        )

    def _rectangle_point_names(self, item_state: Dict[str, Any]) -> List[Optional[str]]:
        args = item_state.get("args", {})
        if not isinstance(args, dict):
            return [None, None, None, None]
        return [
            args.get("p1"),
            args.get("p2"),
            args.get("p3"),
            args.get("p4"),
        ]

    def _resolve_rectangle_vertices(
        self,
        points: List[Optional["Point"]],
        rect_name: str,
    ) -> Optional[List[Tuple[float, float]]]:
        try:
            return self._canonicalize_rectangle_vertices(points)
        except PolygonCanonicalizationError:
            return self._canonicalize_rectangle_from_diagonal(points, rect_name)

    def _canonicalize_rectangle_vertices(self, points: List[Optional["Point"]]) -> List[Tuple[float, float]]:
        return cast(
            List[Tuple[float, float]],
            canonicalize_rectangle(
                [(point.x, point.y) for point in points if point is not None],
                construction_mode="vertices",
            ),
        )

    def _canonicalize_rectangle_from_diagonal(
        self,
        points: List[Optional["Point"]],
        rect_name: str,
    ) -> Optional[List[Tuple[float, float]]]:
        p_diag1, p_diag2 = MathUtils.find_diagonal_points(points, rect_name)
        if not p_diag1 or not p_diag2:
            print(f"Warning: Could not determine diagonal points for rectangle '{rect_name}'. Skipping.")
            return None
        try:
            return cast(
                Optional[List[Tuple[float, float]]],
                canonicalize_rectangle(
                    [(p_diag1.x, p_diag1.y), (p_diag2.x, p_diag2.y)],
                    construction_mode="diagonal",
                ),
            )
        except PolygonCanonicalizationError:
            print(f"Warning: Unable to canonicalize rectangle '{rect_name}' from supplied coordinates. Skipping.")
            return None

    def _create_circles(self, state: Dict[str, Any]) -> None:
        """Create circles from workspace state."""
        if "Circles" not in state:
            return
        for item_state in state["Circles"]:
            self._restore_circle(item_state)

    def _restore_circle(self, item_state: Dict[str, Any]) -> None:
        center_point: Optional["Point"] = self.canvas.get_point_by_name(item_state["args"]["center"])
        if center_point:
            self.canvas.create_circle(
                center_point.x,
                center_point.y,
                item_state["args"]["radius"],
                name=item_state.get("name", ""),
            )

    def _create_ellipses(self, state: Dict[str, Any]) -> None:
        """Create ellipses from workspace state."""
        if "Ellipses" not in state:
            return
        for item_state in state["Ellipses"]:
            self._restore_ellipse(item_state)

    def _restore_ellipse(self, item_state: Dict[str, Any]) -> None:
        center_point: Optional["Point"] = self.canvas.get_point_by_name(item_state["args"]["center"])
        if center_point:
            self.canvas.create_ellipse(
                center_point.x,
                center_point.y,
                item_state["args"]["radius_x"],
                item_state["args"]["radius_y"],
                rotation_angle=item_state["args"].get("rotation_angle", 0),
                name=item_state.get("name", ""),
            )

    def _create_functions(self, state: Dict[str, Any]) -> None:
        """Create functions from workspace state."""
        if "Functions" not in state:
            return
        for item_state in state["Functions"]:
            self._restore_function(item_state)

    def _restore_function(self, item_state: Dict[str, Any]) -> None:
        self.canvas.draw_function(
            item_state["args"]["function_string"],
            name=item_state.get("name", ""),
            left_bound=item_state["args"].get("left_bound"),
            right_bound=item_state["args"].get("right_bound"),
            undefined_at=item_state["args"].get("undefined_at"),
        )

    def _create_piecewise_functions(self, state: Dict[str, Any]) -> None:
        """Create piecewise functions from workspace state."""
        if "PiecewiseFunctions" not in state:
            return
        for item_state in state["PiecewiseFunctions"]:
            self._restore_piecewise_function(item_state)

    def _restore_piecewise_function(self, item_state: Dict[str, Any]) -> None:
        pieces = item_state["args"].get("pieces", [])
        self.canvas.draw_piecewise_function(
            pieces,
            name=item_state.get("name", ""),
            color=item_state["args"].get("color"),
        )

    def _create_parametric_functions(self, state: Dict[str, Any]) -> None:
        """Create parametric functions from workspace state."""
        if "ParametricFunctions" not in state:
            return
        for item_state in state["ParametricFunctions"]:
            self._restore_parametric_function(item_state)

    def _restore_parametric_function(self, item_state: Dict[str, Any]) -> None:
        args = item_state.get("args", {})
        self.canvas.draw_parametric_function(
            args.get("x_expression", "t"),
            args.get("y_expression", "t"),
            name=item_state.get("name", ""),
            t_min=args.get("t_min", 0.0),
            t_max=args.get("t_max"),
            color=args.get("color"),
        )

    def _create_colored_areas(self, state: Dict[str, Any]) -> None:
        """Create colored areas from workspace state."""
        handlers = self._colored_area_handlers()
        for area_type, handler in handlers.items():
            self._restore_colored_area_items(state.get(area_type, []), handler)

    def _colored_area_handlers(
        self,
    ) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        return {
            "ClosedShapeColoredAreas": self._restore_closed_shape_area,
            "FunctionsBoundedColoredAreas": self._restore_functions_bounded_area,
            "SegmentsBoundedColoredAreas": self._restore_generic_colored_area,
            "FunctionSegmentBoundedColoredAreas": self._restore_generic_colored_area,
            "ColoredAreas": self._restore_generic_colored_area,
        }

    def _restore_colored_area_items(
        self,
        items: Any,
        handler: Callable[[Dict[str, Any]], Any],
    ) -> None:
        for item_state in items or []:
            self._restore_colored_area_item(item_state, handler)

    def _restore_colored_area_item(
        self,
        item_state: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Any],
    ) -> None:
        try:
            created = handler(item_state)
            self._apply_restored_colored_area_name(created, item_state)
        except Exception as exc:
            self._warn_colored_area_restore_failure(item_state, exc)
            return

    def _apply_restored_colored_area_name(self, created: Any, item_state: Dict[str, Any]) -> None:
        if created and (name := item_state.get("name")):
            created.name = name

    def _warn_colored_area_restore_failure(self, item_state: Dict[str, Any], exc: Exception) -> None:
        print(f"Warning: Could not restore colored area '{item_state.get('name', 'Unnamed')}': {exc}")

    def _create_plots(self, state: Dict[str, Any]) -> None:
        """
        Restore plot composites (ContinuousPlot/DiscretePlot) from workspace state.

        Plot composites are non-renderable bookkeeping objects that reference other drawables.
        They must be restored after polygons, functions, and colored areas.
        """
        plot_items = self._collect_plot_items(state)

        if not plot_items:
            return

        drawables = self._get_drawables_collection()
        if drawables is None or not hasattr(drawables, "add"):
            return

        for kind, item_state in plot_items:
            if not isinstance(item_state, dict):
                continue
            name = str(item_state.get("name", "") or "")

            try:
                plot, name = self._build_plot_from_state(kind, item_state)
            except Exception as exc:
                print(f"Warning: Could not restore plot '{name}': {exc}")
                continue

            try:
                drawables.add(plot)
            except Exception:
                continue

    def _collect_plot_items(self, state: Dict[str, Any]) -> List[Tuple[str, Any]]:
        plot_items: List[Tuple[str, Any]] = []
        for item_state in state.get("ContinuousPlots", []) or []:
            plot_items.append(("continuous", item_state))
        for item_state in state.get("DiscretePlots", []) or []:
            plot_items.append(("discrete", item_state))
        for item_state in state.get("BarsPlots", []) or []:
            plot_items.append(("bars", item_state))
        for item_state in state.get("Plots", []) or []:
            plot_items.append(("legacy", item_state))
        return plot_items

    def _get_drawables_collection(self) -> Any:
        return getattr(getattr(self.canvas, "drawable_manager", None), "drawables", None)

    def _build_plot_from_state(self, kind: str, item_state: Dict[str, Any]) -> Tuple[Any, str]:
        args = item_state.get("args", {}) if isinstance(item_state.get("args", {}), dict) else {}
        name = str(item_state.get("name", "") or "")
        plot_type = str(args.get("plot_type", "") or "distribution")
        distribution_type = args.get("distribution_type", None)
        distribution_params = args.get("distribution_params", None)
        bounds = args.get("bounds", None)
        metadata = args.get("metadata", None)

        if kind == "continuous":
            plot = self._build_continuous_plot(
                name=name,
                args=args,
                plot_type=plot_type,
                distribution_type=distribution_type,
                distribution_params=distribution_params,
                bounds=bounds,
                metadata=metadata,
            )
            return plot, name

        if kind == "discrete":
            plot = self._build_discrete_plot(
                name=name,
                args=args,
                plot_type=plot_type,
                distribution_type=distribution_type,
                distribution_params=distribution_params,
                bounds=bounds,
                metadata=metadata,
            )
            return plot, name

        if kind == "bars":
            plot = self._build_bars_plot(
                name=name,
                args=args,
                plot_type=plot_type,
                bounds=bounds,
                metadata=metadata,
            )
            return plot, name

        return self._build_legacy_plot(
            name=name,
            args=args,
            plot_type=plot_type,
            distribution_type=distribution_type,
            distribution_params=distribution_params,
            bounds=bounds,
            metadata=metadata,
        ), name

    def _build_continuous_plot(
        self,
        *,
        name: str,
        args: Dict[str, Any],
        plot_type: str,
        distribution_type: Any,
        distribution_params: Any,
        bounds: Any,
        metadata: Any,
    ) -> ContinuousPlot:
        return ContinuousPlot(
            name,
            plot_type=plot_type,
            distribution_type=distribution_type,
            function_name=args.get("function_name"),
            fill_area_name=args.get("fill_area_name"),
            distribution_params=distribution_params,
            bounds=bounds,
            metadata=metadata,
        )

    def _build_discrete_plot(
        self,
        *,
        name: str,
        args: Dict[str, Any],
        plot_type: str,
        distribution_type: Any,
        distribution_params: Any,
        bounds: Any,
        metadata: Any,
    ) -> DiscretePlot:
        return DiscretePlot(
            name,
            plot_type=plot_type,
            distribution_type=distribution_type,
            bar_count=args.get("bar_count"),
            bar_labels=args.get("bar_labels"),
            curve_color=args.get("curve_color"),
            fill_color=args.get("fill_color"),
            fill_opacity=args.get("fill_opacity"),
            rectangle_names=args.get("rectangle_names"),
            fill_area_names=args.get("fill_area_names"),
            distribution_params=distribution_params,
            bounds=bounds,
            metadata=metadata,
        )

    def _build_bars_plot(
        self,
        *,
        name: str,
        args: Dict[str, Any],
        plot_type: str,
        bounds: Any,
        metadata: Any,
    ) -> BarsPlot:
        return BarsPlot(
            name,
            plot_type=plot_type,
            values=args.get("values") or [],
            labels_below=args.get("labels_below") or [],
            labels_above=args.get("labels_above"),
            bar_spacing=args.get("bar_spacing"),
            bar_width=args.get("bar_width"),
            x_start=args.get("x_start"),
            y_base=args.get("y_base"),
            stroke_color=args.get("stroke_color"),
            fill_color=args.get("fill_color"),
            fill_opacity=args.get("fill_opacity"),
            bounds=bounds,
            metadata=metadata,
        )

    def _build_legacy_plot(
        self,
        *,
        name: str,
        args: Dict[str, Any],
        plot_type: str,
        distribution_type: Any,
        distribution_params: Any,
        bounds: Any,
        metadata: Any,
    ) -> Any:
        # Legacy Plot saved before subclasses existed.
        if self._is_legacy_discrete_plot_args(args):
            return self._build_discrete_plot(
                name=name,
                args=args,
                plot_type=plot_type,
                distribution_type=distribution_type,
                distribution_params=distribution_params,
                bounds=bounds,
                metadata=metadata,
            )
        if self._is_legacy_continuous_plot_args(args):
            return self._build_continuous_plot(
                name=name,
                args=args,
                plot_type=plot_type,
                distribution_type=distribution_type,
                distribution_params=distribution_params,
                bounds=bounds,
                metadata=metadata,
            )
        return self._build_base_plot(
            name=name,
            plot_type=plot_type,
            distribution_type=distribution_type,
            distribution_params=distribution_params,
            bounds=bounds,
            metadata=metadata,
        )

    def _is_legacy_discrete_plot_args(self, args: Dict[str, Any]) -> bool:
        return bool(args.get("rectangle_names") or args.get("fill_area_names"))

    def _is_legacy_continuous_plot_args(self, args: Dict[str, Any]) -> bool:
        return bool(args.get("function_name") or args.get("fill_area_name"))

    def _build_base_plot(
        self,
        *,
        name: str,
        plot_type: str,
        distribution_type: Any,
        distribution_params: Any,
        bounds: Any,
        metadata: Any,
    ) -> Plot:
        return Plot(
            name,
            plot_type=plot_type,
            distribution_type=distribution_type,
            distribution_params=distribution_params,
            bounds=bounds,
            metadata=metadata,
        )

    def _restore_closed_shape_area(self, item_state: Dict[str, Any]) -> None:
        shape_args = item_state.get("args", {})
        color, opacity, resolution = self._closed_shape_style_args(shape_args)
        shape_type = shape_args.get("shape_type")
        expression = shape_args.get("expression")

        if shape_type == "region" and expression:
            self.canvas.create_region_colored_area(
                expression=expression,
                resolution=resolution,
                color=color,
                opacity=opacity,
            )
            return

        polygon_names, chord_name = self._closed_shape_polygon_and_chord(shape_args, shape_type)
        circle_name = shape_args.get("circle")
        ellipse_name = shape_args.get("ellipse")
        arc_clockwise = shape_args.get("arc_clockwise", False)

        self.canvas.create_region_colored_area(
            polygon_segment_names=polygon_names,
            circle_name=circle_name,
            ellipse_name=ellipse_name,
            chord_segment_name=chord_name,
            arc_clockwise=arc_clockwise,
            resolution=resolution,
            color=color,
            opacity=opacity,
        )

    def _closed_shape_style_args(self, shape_args: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        color = shape_args.get("color", default_area_fill_color)
        opacity = shape_args.get("opacity", default_area_opacity)
        resolution = shape_args.get("resolution", default_closed_shape_resolution)
        return color, opacity, resolution

    def _closed_shape_polygon_and_chord(
        self,
        shape_args: Dict[str, Any],
        shape_type: Any,
    ) -> Tuple[Any, Any]:
        polygon_names = None
        chord_name = shape_args.get("chord_segment")
        if shape_type == "polygon":
            polygon_names = shape_args.get("segments")
        elif shape_type == "circle":
            chord_name = None
        elif shape_type == "ellipse":
            chord_name = None
        elif shape_type not in ("circle_segment", "ellipse_segment"):
            polygon_names = shape_args.get("segments")
        return polygon_names, chord_name

    def _restore_functions_bounded_area(self, item_state: Dict[str, Any]) -> None:
        args = item_state.get("args", {})
        self.canvas.create_colored_area(
            drawable1_name=args.get("func1"),
            drawable2_name=args.get("func2"),
            left_bound=args.get("left_bound"),
            right_bound=args.get("right_bound"),
            color=args.get("color", default_area_fill_color),
            opacity=args.get("opacity", default_area_opacity),
        )

    def _restore_generic_colored_area(self, item_state: Dict[str, Any]) -> None:
        args = item_state.get("args", {})
        drawable1_name, drawable2_name = self._generic_colored_area_drawable_names(args)
        self.canvas.create_colored_area(
            drawable1_name=drawable1_name,
            drawable2_name=drawable2_name,
            left_bound=args.get("left_bound"),
            right_bound=args.get("right_bound"),
            color=args.get("color", default_area_fill_color),
            opacity=args.get("opacity", default_area_opacity),
        )

    def _generic_colored_area_drawable_names(self, args: Dict[str, Any]) -> Tuple[Any, Any]:
        drawable1_name = args.get("drawable1_name") or args.get("segment1") or args.get("func1")
        drawable2_name = args.get("drawable2_name") or args.get("segment2") or args.get("func2")
        return drawable1_name, drawable2_name

    def _create_angles(self, state: Dict[str, Any]) -> None:
        """Create angles from workspace state."""
        if "Angles" not in state:
            return

        angle_manager = self._get_angle_manager_for_restore()
        if angle_manager is None:
            print("Warning: Angle manager not available for loading angles")
            return

        try:
            angle_manager.load_angles(state["Angles"])
        except Exception as e:
            print(f"Warning: Could not restore angles: {e}")

    def _get_angle_manager_for_restore(self) -> Any:
        if not hasattr(self.canvas, "drawable_manager"):
            return None
        if not hasattr(self.canvas.drawable_manager, "angle_manager"):
            return None
        return self.canvas.drawable_manager.angle_manager

    def _create_circle_arcs(self, state: Dict[str, Any]) -> None:
        """Create circle arcs from workspace state."""
        arcs_data = state.get("CircleArcs")
        if not arcs_data:
            return

        for arc_state in arcs_data:
            self._restore_circle_arc(arc_state)

    def _restore_circle_arc(self, arc_state: Dict[str, Any]) -> None:
        args = arc_state.get("args", {})
        point1, point2 = self._resolve_circle_arc_points(args)
        if not point1 or not point2:
            return

        try:
            self.canvas.create_circle_arc(**self._build_restored_circle_arc_kwargs(arc_state, args, point1, point2))
        except Exception as exc:
            self._warn_circle_arc_restore_failure(arc_state, exc)

    def _resolve_circle_arc_points(self, args: Dict[str, Any]) -> Tuple[Any, Any]:
        point1_name = args.get("point1_name")
        point2_name = args.get("point2_name")
        point1 = self.canvas.get_point_by_name(point1_name) if point1_name else None
        point2 = self.canvas.get_point_by_name(point2_name) if point2_name else None
        return point1, point2

    def _build_restored_circle_arc_kwargs(
        self,
        arc_state: Dict[str, Any],
        args: Dict[str, Any],
        point1: Any,
        point2: Any,
    ) -> Dict[str, Any]:
        return {
            "point1_x": point1.x,
            "point1_y": point1.y,
            "point2_x": point2.x,
            "point2_y": point2.y,
            "point1_name": point1.name,
            "point2_name": point2.name,
            "point3_x": None,
            "point3_y": None,
            "point3_name": None,
            "center_point_choice": None,
            "circle_name": args.get("circle_name"),
            "center_x": args.get("center_x"),
            "center_y": args.get("center_y"),
            "radius": args.get("radius"),
            "arc_name": arc_state.get("name"),
            "color": args.get("color"),
            "use_major_arc": args.get("use_major_arc", False),
            "extra_graphics": False,
        }

    def _warn_circle_arc_restore_failure(self, arc_state: Dict[str, Any], exc: Exception) -> None:
        print(f"Warning: Could not restore circle arc '{arc_state.get('name', '')}': {exc}")

    def _restore_computations(self, state: Dict[str, Any]) -> None:
        """Restore computations from workspace state."""
        if "computations" not in state:
            return
        for comp in state["computations"]:
            self._restore_computation_if_applicable(comp)

    def _restore_computation_if_applicable(self, comp: Dict[str, Any]) -> None:
        # Skip workspace management functions
        if self._is_workspace_management_expression(comp["expression"]):
            return
        self.canvas.add_computation(comp["expression"], comp["result"])

    def _is_workspace_management_expression(self, expression: str) -> bool:
        return (
            expression.startswith("list_workspaces")
            or expression.startswith("save_workspace")
            or expression.startswith("load_workspace")
        )

    def _restore_workspace_state(self, state: Dict[str, Any]) -> None:
        """
        Main restoration orchestrator for complete workspace state.

        Restores all geometric objects and computations in the correct dependency
        order to ensure proper relationships between objects. Clears the canvas
        first, then creates objects from points to complex shapes.

        Args:
            state (dict): Workspace state dictionary containing all object data.
        """
        self._run_restore_phases(state)

    def _run_restore_phases(self, state: Dict[str, Any]) -> None:
        for phase in self._restore_phases():
            phase(state)

    def _restore_phases(self) -> List[Callable[[Dict[str, Any]], None]]:
        return [
            self._phase_clear_canvas_for_restore,
            self._restore_coordinate_system_state,
            self._restore_drawables_in_dependency_order,
            self._phase_draw_canvas_if_enabled,
            self._restore_post_draw_dependencies,
            self._restore_computations,
        ]

    def _clear_canvas_for_restore(self) -> None:
        self.canvas.clear()

    def _phase_clear_canvas_for_restore(self, _: Dict[str, Any]) -> None:
        self._clear_canvas_for_restore()

    def _restore_coordinate_system_state(self, state: Dict[str, Any]) -> None:
        # Always reset to cartesian first (for legacy workspaces without coordinate_system)
        # Then restore the saved mode if present
        if hasattr(self.canvas, "coordinate_system_manager"):
            try:
                self._reset_coordinate_system_to_cartesian()
                self._apply_saved_coordinate_system_state(state.get("coordinate_system"))
            except Exception:
                pass

    def _reset_coordinate_system_to_cartesian(self) -> None:
        # Default to cartesian
        self.canvas.coordinate_system_manager.set_state({"mode": "cartesian"})

    def _apply_saved_coordinate_system_state(self, coord_system_state: Any) -> None:
        # Then apply saved state if present
        if coord_system_state:
            self.canvas.coordinate_system_manager.set_state(coord_system_state)

    def _restore_drawables_in_dependency_order(self, state: Dict[str, Any]) -> None:
        # Create objects in the correct dependency order
        self._run_restore_steps(self._pre_plot_restore_steps(), state)
        # Create colored areas after functions since they may depend on functions
        self._create_colored_areas(state)
        # Restore plot composites after functions, polygons, and colored areas.
        self._create_plots(state)
        self._materialize_discrete_plot_bars()
        self._materialize_bars_plot_bars()

    def _pre_plot_restore_steps(self) -> List[Callable[[Dict[str, Any]], None]]:
        return [
            self._create_points,
            self._create_labels,
            self._create_segments,
            self._create_vectors,
            self._create_triangles,
            self._create_rectangles,
            self._create_circles,
            self._create_circle_arcs,
            self._create_ellipses,
            self._create_functions,
            self._create_piecewise_functions,
            self._create_parametric_functions,
        ]

    def _run_restore_steps(
        self,
        steps: List[Callable[[Dict[str, Any]], None]],
        state: Dict[str, Any],
    ) -> None:
        for step in steps:
            step(state)

    def _draw_canvas_if_enabled(self) -> None:
        if getattr(self.canvas, "draw_enabled", False):
            try:
                self.canvas.draw()
            except Exception:
                pass

    def _phase_draw_canvas_if_enabled(self, _: Dict[str, Any]) -> None:
        self._draw_canvas_if_enabled()

    def _restore_post_draw_dependencies(self, state: Dict[str, Any]) -> None:
        # Create angles after segments since they depend on segments
        self._create_angles(state)

    def _materialize_discrete_plot_bars(self) -> None:
        """
        Build Bar drawables for restored DiscretePlot objects.

        Bars are derived from DiscretePlot params and are not persisted in saved workspaces.
        """
        self._materialize_plot_family(
            class_name="DiscretePlot",
            materializer_name="materialize_discrete_plot",
        )

    def _materialize_bars_plot_bars(self) -> None:
        """
        Build Bar drawables for restored BarsPlot objects.

        Bars are derived from BarsPlot params and are not persisted in saved workspaces.
        """
        self._materialize_plot_family(
            class_name="BarsPlot",
            materializer_name="materialize_bars_plot",
        )

    def _materialize_plot_family(self, class_name: str, materializer_name: str) -> None:
        stats_manager = self._get_statistics_manager_for_materialization(materializer_name)
        if stats_manager is None:
            return

        drawables = self._get_drawables_for_materialization()
        if drawables is None:
            return

        plots = self._get_plots_by_class_name(drawables, class_name)
        materializer = self._get_plot_materializer(stats_manager, materializer_name)
        if materializer is None:
            return

        self._materialize_plots(plots, materializer)

    def _get_statistics_manager_for_materialization(self, materializer_name: str) -> Any:
        stats_manager = getattr(getattr(self.canvas, "drawable_manager", None), "statistics_manager", None)
        if stats_manager is None or not hasattr(stats_manager, materializer_name):
            return None
        return stats_manager

    def _get_drawables_for_materialization(self) -> Any:
        drawables = getattr(getattr(self.canvas, "drawable_manager", None), "drawables", None)
        if drawables is None or not hasattr(drawables, "get_by_class_name"):
            return None
        return drawables

    def _get_plots_by_class_name(self, drawables: Any, class_name: str) -> List[Any]:
        try:
            return list(drawables.get_by_class_name(class_name))
        except Exception:
            return []

    def _get_plot_materializer(self, stats_manager: Any, materializer_name: str) -> Optional[Callable[[Any], None]]:
        materializer = getattr(stats_manager, materializer_name, None)
        if not callable(materializer):
            return None
        return cast(Optional[Callable[[Any], None]], materializer)

    def _materialize_plots(self, plots: List[Any], materializer: Callable[[Any], None]) -> None:
        for plot in plots:
            try:
                materializer(plot)
            except Exception:
                continue

    def load_workspace(self, name: Optional[str] = None) -> str:
        """
        Load and restore workspace state from server.

        Requests workspace data from the Flask backend server via AJAX and
        restores the complete canvas state including all geometric objects
        and computations in the correct dependency order.

        Args:
            name (str, optional): Name of the workspace to load. If None, loads default.

        Returns:
            str: Success or error message from the load operation.
        """

        def on_complete(req: Any) -> str:
            return self._parse_load_workspace_response(req, name)

        url: str = f"/load_workspace?name={name}" if name else "/load_workspace"
        return self._execute_sync_request(
            method="GET",
            url=url,
            on_complete=on_complete,
            error_prefix="Error loading workspace",
        )

    def _parse_load_workspace_response(self, req: Any, name: Optional[str]) -> str:
        return self._parse_workspace_response(
            req=req,
            action_gerund="loading",
            on_success=lambda response: self._build_load_workspace_success_message(response, name),
            exception_prefix="Error loading workspace",
        )

    def _build_load_workspace_success_message(
        self,
        response: Dict[str, Any],
        name: Optional[str],
    ) -> str:
        state = self._workspace_state_from_response(response)
        if not state:
            return "Error loading workspace: No state data found in response"
        self._restore_workspace_state(state)
        return f'Workspace "{name if name else "current"}" loaded successfully.'

    def list_workspaces(self) -> str:
        """
        Retrieve list of available workspace names from server storage.

        Sends an AJAX request to the Flask backend to get all available
        workspace names that can be loaded.

        Returns:
            str: Comma-separated list of workspace names, or 'None' if empty.
        """

        def on_complete(req: Any) -> str:
            return self._parse_list_workspaces_response(req)

        return self._execute_sync_request(
            method="GET",
            url="/list_workspaces",
            on_complete=on_complete,
            error_prefix="Error listing workspaces",
        )

    def _parse_list_workspaces_response(self, req: Any) -> str:
        return self._parse_workspace_response(
            req=req,
            action_gerund="listing",
            on_success=self._build_list_workspaces_success_message,
            exception_prefix="Error listing workspaces",
        )

    def _build_list_workspaces_success_message(self, response: Dict[str, Any]) -> str:
        workspaces = self._workspace_list_from_response(response)
        return ", ".join(workspaces) if workspaces else "None"

    def delete_workspace(self, name: str) -> str:
        """
        Remove workspace from server persistent storage.

        Sends an AJAX request to the Flask backend to permanently delete
        the specified workspace from storage.

        Args:
            name (str): Name of the workspace to delete.

        Returns:
            str: Success or error message from the delete operation.
        """

        def on_complete(req: Any) -> str:
            return self._parse_delete_workspace_response(req, name)

        url: str = f"/delete_workspace?name={name}"
        return self._execute_sync_request(
            method="GET",
            url=url,
            on_complete=on_complete,
            error_prefix="Error deleting workspace",
        )

    def _parse_delete_workspace_response(self, req: Any, name: str) -> str:
        return self._parse_workspace_response(
            req=req,
            action_gerund="deleting",
            on_success=lambda _response: f'Workspace "{name}" deleted successfully.',
            exception_prefix="Error deleting workspace",
        )

    def _parse_workspace_response(
        self,
        req: Any,
        action_gerund: str,
        on_success: Callable[[Dict[str, Any]], str],
        exception_prefix: str,
    ) -> str:
        try:
            response = self._response_from_request(req)
            if self._response_is_success(response):
                return on_success(response)
            return self._format_workspace_error(action_gerund, response)
        except Exception as e:
            return f"{exception_prefix}: {str(e)}"

    def _response_from_request(self, req: Any) -> Dict[str, Any]:
        return cast(Dict[str, Any], json.loads(req.text))

    def _response_is_success(self, response: Dict[str, Any]) -> bool:
        return bool(response.get("status") == "success")

    def _workspace_state_from_response(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return cast(Optional[Dict[str, Any]], response.get("data", {}).get("state"))

    def _workspace_list_from_response(self, response: Dict[str, Any]) -> List[str]:
        return cast(List[str], response.get("data", []))

    def _format_workspace_error(self, action_gerund: str, response: Dict[str, Any]) -> str:
        return f"Error {action_gerund} workspace: {response.get('message')}"

    def _execute_sync_request(
        self,
        method: str,
        url: str,
        on_complete: Callable[[Any], str],
        error_prefix: str,
    ) -> str:
        req = self._build_sync_request(on_complete, error_prefix)
        self._open_and_send_sync_request(req, method, url)
        return self._finalize_sync_request(req, on_complete)

    def _execute_sync_json_request(
        self,
        method: str,
        url: str,
        payload: Dict[str, Any],
        on_complete: Callable[[Any], str],
        error_prefix: str,
    ) -> str:
        req = self._build_sync_request(on_complete, error_prefix)
        self._open_and_send_sync_json_request(req, method, url, payload)
        return self._finalize_sync_request(req, on_complete)

    def _build_sync_request(
        self,
        on_complete: Callable[[Any], str],
        error_prefix: str,
    ) -> Any:
        req: Any = ajax.Ajax()
        req.bind("complete", on_complete)
        req.bind("error", lambda e: f"{error_prefix}: {e.text}")
        return req

    def _open_and_send_sync_request(self, req: Any, method: str, url: str) -> None:
        req.open(method, url, False)  # Set to synchronous
        req.send()

    def _open_and_send_sync_json_request(
        self,
        req: Any,
        method: str,
        url: str,
        payload: Dict[str, Any],
    ) -> None:
        req.open(method, url, False)  # Set to synchronous
        req.set_header("Content-Type", "application/json")
        req.send(json.dumps(payload))

    def _finalize_sync_request(self, req: Any, on_complete: Callable[[Any], str]) -> str:
        return on_complete(req)
