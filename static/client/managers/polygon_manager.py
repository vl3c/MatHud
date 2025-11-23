from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

from drawables.hexagon import Hexagon
from drawables.pentagon import Pentagon
from drawables.quadrilateral import Quadrilateral
from drawables.rectangle import Rectangle
from drawables.triangle import Triangle
from drawables.position import Position
from managers.polygon_type import PolygonType
from managers.edit_policy import EditRule, get_drawable_edit_policy
from utils.geometry_utils import GeometryUtils
from utils.polygon_canonicalizer import (
    PolygonCanonicalizationError,
    canonicalize_rectangle,
)

if TYPE_CHECKING:
    from canvas import Canvas
    from drawables.drawable import Drawable
    from drawables.point import Point
    from drawables.segment import Segment
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from managers.drawables_container import DrawablesContainer
    from managers.point_manager import PointManager
    from managers.segment_manager import SegmentManager
    from name_generator.drawable import DrawableNameGenerator


Coordinate = Tuple[float, float]
SegmentList = List["Segment"]


class PolygonManager:
    """Manages polygonal drawables with shared create/update/delete flows."""

    _TYPE_TO_SIDE_COUNT: Dict[PolygonType, int] = {
        PolygonType.TRIANGLE: 3,
        PolygonType.QUADRILATERAL: 4,
        PolygonType.RECTANGLE: 4,
        PolygonType.SQUARE: 4,
        PolygonType.PENTAGON: 5,
        PolygonType.HEXAGON: 6,
    }

    _TYPE_TO_CLASSES: Dict[PolygonType, Tuple[str, ...]] = {
        PolygonType.TRIANGLE: ("Triangle",),
        PolygonType.QUADRILATERAL: ("Quadrilateral", "Rectangle"),
        PolygonType.RECTANGLE: ("Rectangle",),
        PolygonType.SQUARE: ("Rectangle",),
        PolygonType.PENTAGON: ("Pentagon",),
        PolygonType.HEXAGON: ("Hexagon",),
    }

    _ALL_POLYGON_CLASSES: Tuple[str, ...] = ("Triangle", "Quadrilateral", "Rectangle", "Pentagon", "Hexagon")

    def __init__(
        self,
        canvas: "Canvas",
        drawables_container: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        point_manager: "PointManager",
        segment_manager: "SegmentManager",
        drawable_manager_proxy: "DrawableManagerProxy",
    ) -> None:
        self.canvas = canvas
        self.drawables = drawables_container
        self.name_generator = name_generator
        self.dependency_manager = dependency_manager
        self.point_manager = point_manager
        self.segment_manager = segment_manager
        self.drawable_manager = drawable_manager_proxy

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def create_polygon(
        self,
        vertices: Sequence[Any],
        *,
        polygon_type: Optional[Union[str, PolygonType]] = None,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
    ) -> "Drawable":
        normalized_vertices = self._sanitize_vertices(vertices)
        normalized_type, constraints = self._resolve_polygon_type(normalized_vertices, polygon_type)

        if constraints.get("require_rectangle"):
            mode = "diagonal" if len(normalized_vertices) == 2 else "vertices"
            try:
                normalized_vertices = canonicalize_rectangle(
                    normalized_vertices,
                    construction_mode=mode,
                )
            except PolygonCanonicalizationError as exc:
                raise ValueError(str(exc)) from exc

        self._validate_polygon_coordinates(normalized_vertices, normalized_type, constraints)

        existing = self.get_polygon_by_vertices(normalized_vertices, normalized_type)
        if existing:
            return existing

        self.canvas.undo_redo_manager.archive()

        point_names = self._build_point_names(name, len(normalized_vertices))
        points = self._create_points(normalized_vertices, point_names)
        segment_kwargs = self._build_segment_kwargs(color)
        segments = self._create_segments(points, segment_kwargs)

        polygon = self._instantiate_polygon(normalized_type, segments, color)
        self.drawables.add(polygon)
        self.dependency_manager.analyze_drawable_for_dependencies(polygon)

        if extra_graphics:
            self.drawable_manager.create_drawables_from_new_connections()

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return polygon

    def update_polygon(
        self,
        polygon_name: str,
        *,
        polygon_type: Optional[Union[str, PolygonType]] = None,
        new_color: Optional[str] = None,
    ) -> bool:
        polygon = self.get_polygon_by_name(polygon_name, polygon_type)
        if not polygon:
            raise ValueError(f"Polygon '{polygon_name}' was not found.")

        pending_fields = self._collect_requested_fields(new_color)
        policy = self._resolve_edit_policy(polygon, pending_fields)
        self._validate_color_request(pending_fields, new_color)

        self.canvas.undo_redo_manager.archive()
        self._apply_updates(polygon, pending_fields, new_color)

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def delete_polygon(
        self,
        *,
        polygon_type: Optional[Union[str, PolygonType]] = None,
        name: Optional[str] = None,
        vertices: Optional[Sequence[Any]] = None,
    ) -> bool:
        target = None
        if name:
            target = self.get_polygon_by_name(name, polygon_type)
        if target is None and vertices is not None:
            normalized_vertices = self._sanitize_vertices(vertices)
            target = self.get_polygon_by_vertices(normalized_vertices, polygon_type)

        if target is None:
            return False

        self.canvas.undo_redo_manager.archive()

        removed = self.drawables.remove(target)
        if removed:
            for segment in self._iter_polygon_segments(target):
                self.segment_manager.delete_segment(
                    segment.point1.x,
                    segment.point1.y,
                    segment.point2.x,
                    segment.point2.y,
                )

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return bool(removed)

    def get_polygon_by_name(
        self,
        name: str,
        polygon_type: Optional[Union[str, PolygonType]] = None,
    ) -> Optional["Drawable"]:
        allowed_classes = self._allowed_classes_for_type(polygon_type)
        return self.drawables.get_polygon_by_name(name, allowed_classes)

    def get_polygon_by_vertices(
        self,
        vertices: Sequence[Coordinate],
        polygon_type: Optional[Union[str, PolygonType]] = None,
    ) -> Optional["Drawable"]:
        allowed_classes = self._allowed_classes_for_type(polygon_type, len(vertices))
        vertex_signature = self._build_vertex_signature(vertices)

        for polygon in self.drawables.iter_polygons(allowed_classes):
            if self._build_vertex_signature(self._extract_polygon_vertices(polygon)) == vertex_signature:
                return polygon
        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _sanitize_vertices(self, vertices: Sequence[Any]) -> List[Coordinate]:
        if not vertices:
            raise ValueError("Provide at least three vertices.")

        normalized: List[Coordinate] = []
        for vertex in vertices:
            if isinstance(vertex, dict):
                x = vertex.get("x")
                y = vertex.get("y")
            else:
                try:
                    x, y = vertex  # type: ignore[misc]
                except Exception as exc:
                    raise ValueError("Vertices must be pairs of numeric coordinates.") from exc
            if x is None or y is None:
                raise ValueError("Each vertex must include both x and y coordinates.")
            normalized.append((float(x), float(y)))

        if len(normalized) < 3:
            raise ValueError("Provide at least three vertices.")

        return normalized

    def _coerce_type(
        self,
        polygon_type: Optional[Union[str, PolygonType]],
    ) -> Optional[PolygonType]:
        if polygon_type is None:
            return None
        if isinstance(polygon_type, PolygonType):
            return polygon_type
        return PolygonType.coerce(polygon_type)

    def _resolve_polygon_type(
        self,
        vertices: Sequence[Coordinate],
        explicit_type: Optional[Union[str, PolygonType]],
    ) -> Tuple[PolygonType, Dict[str, bool]]:
        coerced = self._coerce_type(explicit_type)
        if coerced is not None:
            self._validate_vertex_count(len(vertices), coerced)
            constraints = self._type_constraints(coerced)
            resolved_type = PolygonType.RECTANGLE if coerced is PolygonType.SQUARE else coerced
            return resolved_type, constraints

        inferred = self._infer_type_from_vertex_count(len(vertices))
        return inferred, {}

    def _validate_polygon_coordinates(
        self,
        vertices: Sequence[Coordinate],
        polygon_type: PolygonType,
        constraints: Dict[str, bool],
    ) -> None:
        if constraints.get("require_rectangle") or constraints.get("require_square"):
            positions = [Position(x, y) for x, y in vertices]
            if not GeometryUtils.is_rectangle(positions):
                raise ValueError("Provided vertices do not form a rectangle.")
            if constraints.get("require_square") and not GeometryUtils.is_square(positions):
                raise ValueError("Provided vertices do not form a square.")

    def _build_point_names(self, name: str, count: int) -> List[str]:
        if not name:
            return [""] * count
        return self.name_generator.split_point_names(name, count)

    def _create_points(
        self,
        vertices: Sequence[Coordinate],
        point_names: Sequence[str],
    ) -> List["Point"]:
        points: List["Point"] = []
        for index, (x, y) in enumerate(vertices):
            preferred_name = point_names[index] if index < len(point_names) else ""
            point = self.point_manager.create_point(
                x,
                y,
                name=preferred_name,
                extra_graphics=False,
            )
            points.append(point)
        return points

    def _build_segment_kwargs(self, color: Optional[str]) -> Dict[str, str]:
        color_value = str(color).strip() if color is not None else ""
        if not color_value:
            return {}
        return {"color": color_value}

    def _create_segments(
        self,
        points: Sequence["Point"],
        segment_kwargs: Dict[str, str],
    ) -> SegmentList:
        total = len(points)
        segments: SegmentList = []
        for index in range(total):
            start = points[index]
            end = points[(index + 1) % total]
            segment = self.segment_manager.create_segment(
                start.x,
                start.y,
                end.x,
                end.y,
                extra_graphics=False,
                **segment_kwargs,
            )
            segments.append(segment)
        return segments

    def _instantiate_polygon(
        self,
        polygon_type: PolygonType,
        segments: SegmentList,
        color: Optional[str],
    ) -> "Drawable":
        color_value = str(color).strip() if color is not None else ""

        if polygon_type is PolygonType.TRIANGLE:
            s1, s2, s3 = segments
            polygon = Triangle(s1, s2, s3, color=color_value) if color_value else Triangle(s1, s2, s3)
            if color_value:
                polygon.update_color(color_value)
            return polygon

        if polygon_type is PolygonType.RECTANGLE:
            s1, s2, s3, s4 = segments
            polygon = Rectangle(s1, s2, s3, s4, color=color_value) if color_value else Rectangle(s1, s2, s3, s4)
            if color_value:
                polygon.update_color(color_value)
            return polygon

        if polygon_type is PolygonType.QUADRILATERAL:
            s1, s2, s3, s4 = segments
            polygon = Quadrilateral(s1, s2, s3, s4, color=color_value) if color_value else Quadrilateral(s1, s2, s3, s4)
            if color_value:
                polygon.update_color(color_value)
            return polygon

        if polygon_type is PolygonType.PENTAGON:
            polygon = Pentagon(segments, color=color_value) if color_value else Pentagon(segments)
            if color_value:
                polygon.update_color(color_value)
            return polygon

        if polygon_type is PolygonType.HEXAGON:
            polygon = Hexagon(segments, color=color_value) if color_value else Hexagon(segments)
            if color_value:
                polygon.update_color(color_value)
            return polygon

        raise ValueError(f"Unsupported polygon type '{polygon_type.value}'.")

    def _collect_requested_fields(self, new_color: Optional[str]) -> Dict[str, str]:
        pending: Dict[str, str] = {}
        if new_color is not None:
            pending["color"] = "color"
        if not pending:
            raise ValueError("Provide at least one property to update.")
        return pending

    def _resolve_edit_policy(self, polygon: "Drawable", pending_fields: Dict[str, str]) -> Dict[str, EditRule]:
        class_name = polygon.get_class_name() if hasattr(polygon, "get_class_name") else polygon.__class__.__name__
        policy = get_drawable_edit_policy(class_name)
        if not policy:
            raise ValueError(f"Edit policy for {class_name} is not configured.")

        validated: Dict[str, EditRule] = {}
        for field in pending_fields:
            rule = policy.get_rule(field)
            if not rule:
                raise ValueError(f"Editing field '{field}' is not permitted for {class_name.lower()}s.")
            validated[field] = rule
        return validated

    def _validate_color_request(self, pending_fields: Dict[str, str], new_color: Optional[str]) -> None:
        if "color" in pending_fields:
            sanitized = str(new_color).strip() if new_color is not None else ""
            if not sanitized:
                raise ValueError("Polygon color cannot be empty.")

    def _apply_updates(
        self,
        polygon: "Drawable",
        pending_fields: Dict[str, str],
        new_color: Optional[str],
    ) -> None:
        if "color" in pending_fields and new_color is not None:
            if hasattr(polygon, "update_color") and callable(getattr(polygon, "update_color")):
                polygon.update_color(str(new_color))
            else:
                polygon.color = str(new_color)

    def _allowed_classes_for_type(
        self,
        polygon_type: Optional[Union[str, PolygonType]],
        vertex_count: Optional[int] = None,
    ) -> Iterable[str]:
        normalized = self._coerce_type(polygon_type)
        if normalized:
            return self._TYPE_TO_CLASSES.get(normalized, self._ALL_POLYGON_CLASSES)
        if vertex_count is not None:
            inferred = self._infer_type_from_vertex_count(vertex_count)
            return self._TYPE_TO_CLASSES.get(inferred, self._ALL_POLYGON_CLASSES)
        return self._ALL_POLYGON_CLASSES

    def _infer_type_from_vertex_count(self, count: int) -> PolygonType:
        for polygon_type, expected in self._TYPE_TO_SIDE_COUNT.items():
            if polygon_type in (PolygonType.RECTANGLE, PolygonType.SQUARE):
                continue
            if expected == count:
                return polygon_type
        raise ValueError(f"Unsupported polygon with {count} vertices.")

    def _validate_vertex_count(self, count: int, polygon_type: PolygonType) -> None:
        expected = self._TYPE_TO_SIDE_COUNT.get(polygon_type)
        if expected is not None and expected != count:
            raise ValueError(
                f"{polygon_type.value.capitalize()} requires exactly {expected} vertices, received {count}."
            )

    def _type_constraints(self, polygon_type: PolygonType) -> Dict[str, bool]:
        if polygon_type is PolygonType.SQUARE:
            return {"require_rectangle": True, "require_square": True}
        if polygon_type is PolygonType.RECTANGLE:
            return {"require_rectangle": True}
        return {}

    def _iter_polygon_segments(self, polygon: "Drawable") -> Iterable["Segment"]:
        if hasattr(polygon, "get_segments") and callable(getattr(polygon, "get_segments")):
            segments = getattr(polygon, "get_segments")()
            if segments:
                return cast(Iterable["Segment"], list(segments))

        explicit_segments: List["Segment"] = []
        for attribute in ("segment1", "segment2", "segment3", "segment4", "segment5", "segment6"):
            segment = getattr(polygon, attribute, None)
            if segment is not None:
                explicit_segments.append(segment)
        return explicit_segments

    def _extract_polygon_vertices(self, polygon: "Drawable") -> List[Coordinate]:
        if hasattr(polygon, "get_vertices") and callable(getattr(polygon, "get_vertices")):
            vertices = getattr(polygon, "get_vertices")()
            return [(point.x, point.y) for point in vertices]  # type: ignore[attr-defined]
        vertices: List[Coordinate] = []
        for segment in self._iter_polygon_segments(polygon):
            vertices.append((segment.point1.x, segment.point1.y))
            vertices.append((segment.point2.x, segment.point2.y))
        return vertices

    def _build_vertex_signature(self, vertices: Sequence[Coordinate]) -> Tuple[Tuple[float, float], ...]:
        return tuple(sorted({(float(x), float(y)) for x, y in vertices}))

