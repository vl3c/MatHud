"""
MatHud Geometric Transformations Management System

Handles geometric transformations of drawable objects including translation,
rotation, reflection, scaling, and shearing.
Provides coordinated transformation operations with proper state management and canvas integration.

Transformation Types:
    - Translation: Moving objects by specified x and y offsets
    - Rotation: Rotating objects around their centers or an arbitrary point
    - Reflection: Mirroring objects across x-axis, y-axis, or an arbitrary line
    - Scaling (dilation): Uniform or non-uniform scaling from a center point
    - Shearing: Horizontal or vertical shear from a center point

Operation Coordination:
    - State Archiving: Automatic undo/redo state capture before transformations
    - Object Validation: Ensures target objects exist before transformation
    - Method Delegation: Calls transformation methods on drawable objects
    - Canvas Integration: Automatic redrawing after successful transformations

Error Handling:
    - Object Existence Validation: Checks for drawable presence before operations
    - Transformation Validation: Ensures objects support required transformation methods
    - Exception Management: Graceful error handling with descriptive messages
    - State Consistency: Maintains proper canvas state even if transformations fail

Integration Points:
    - DrawableManager: Object lookup and validation
    - UndoRedoManager: State preservation for transformation operations
    - Canvas: Visual updates after transformations
    - Drawable Objects: Delegation to object-specific transformation methods
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Set, Tuple

from drawables.segment import Segment

if TYPE_CHECKING:
    from canvas import Canvas

# Types that do not support geometric transforms via AI tools.
_EXCLUDE_TRANSFORM: Tuple[str, ...] = (
    "Function",
    "ParametricFunction",
    "PiecewiseFunction",
    "Graph",
    "Angle",
    "CircleArc",
    "ColoredArea",
    "Label",
    "Bar",
    "Plot",
)


class TransformationsManager:
    """Manages geometric transformations of drawable objects on a Canvas.

    Coordinates translation, rotation, reflection, scaling, and shearing
    operations with proper state management, object validation,
    and canvas integration.
    """

    def __init__(self, canvas: "Canvas") -> None:
        """
        Initialize the TransformationsManager.

        Args:
            canvas: The Canvas object this manager is responsible for
        """
        self.canvas: "Canvas" = canvas

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _find_drawable_by_name(
        self,
        name: str,
        exclude_types: Tuple[str, ...] = (),
    ) -> Any:
        """Look up a drawable by name, optionally skipping certain class names.

        Returns:
            The matching drawable.

        Raises:
            ValueError: If no drawable with the given name is found.
        """
        for d in self.canvas.drawable_manager.get_drawables():
            if exclude_types:
                cn = d.get_class_name() if hasattr(d, "get_class_name") else d.__class__.__name__
                if cn in exclude_types:
                    continue
            if d.name == name:
                return d
        raise ValueError(f"No drawable found with name '{name}'")

    def _get_class_name(self, drawable: Any) -> str:
        getter = getattr(drawable, "get_class_name", None)
        name: str = getter() if callable(getter) else drawable.__class__.__name__
        return name

    def _gather_moved_points(self, drawable: Any) -> List[Any]:
        get_vertices = getattr(drawable, "get_vertices", None)
        if callable(get_vertices):
            try:
                return list(get_vertices())
            except Exception:
                pass
        return []

    def _validate_shear_support(self, drawable: Any) -> None:
        """Raise before archiving if the drawable cannot be sheared."""
        cn = self._get_class_name(drawable)
        if cn == "Circle":
            raise ValueError("Shearing a circle is not supported; convert to an ellipse first")
        if cn == "Ellipse":
            raise ValueError("Shearing an ellipse is not supported")

    def _validate_scale_support(self, drawable: Any, sx: float, sy: float) -> None:
        """Raise before archiving if the drawable cannot be scaled with these factors."""
        cn = self._get_class_name(drawable)
        uniform = abs(sx - sy) < 1e-9
        if cn == "Circle" and not uniform:
            raise ValueError(
                "Non-uniform scaling of a circle is not supported; convert to an ellipse first or use equal sx and sy"
            )
        if cn == "Ellipse" and not uniform:
            rot = getattr(drawable, "rotation_angle", 0)
            if (rot % 180) > 1e-9:
                raise ValueError("Non-uniform scaling of a rotated ellipse is not supported")

    def _refresh_dependencies_after_transform(
        self,
        drawable: Any,
        moved_points: List[Any],
    ) -> None:
        """Refresh formulas, names, and caches after a transform."""
        class_name = self._get_class_name(drawable)

        # moved_points is non-empty only for drawables with get_vertices()
        # (all Polygon subclasses: Triangle, Rectangle, Quadrilateral,
        # Pentagon, Hexagon, etc.) — no hard-coded class name set needed.
        if moved_points:
            self._refresh_polygon_dependencies(drawable, moved_points)
        elif class_name == "Circle":
            self._refresh_circle_dependencies(drawable)
        elif class_name == "Ellipse":
            self._refresh_ellipse_dependencies(drawable)
        else:
            self._invalidate_drawables([drawable])

    def _redraw(self) -> None:
        if self.canvas.draw_enabled:
            self.canvas.draw()

    # ------------------------------------------------------------------
    # Public transform operations
    # ------------------------------------------------------------------

    def translate_object(self, name: str, x_offset: float, y_offset: float) -> bool:
        """
        Translates a drawable object by the specified offset.

        Args:
            name: Name of the drawable to translate
            x_offset: Horizontal offset to apply
            y_offset: Vertical offset to apply

        Returns:
            bool: True if the translation was successful

        Raises:
            ValueError: If no drawable with the given name is found
        """
        # Find the drawable first to validate it exists
        drawable = None
        for drawable in self.canvas.drawable_manager.get_drawables():
            if drawable.name == name:
                break

        if not drawable or drawable.name != name:
            raise ValueError(f"No drawable found with name '{name}'")

        # Archive current state for undo/redo AFTER finding the object but BEFORE modifying it
        self.canvas.undo_redo_manager.archive()

        moved_points: List[Any] = []
        get_vertices = getattr(drawable, "get_vertices", None)
        if callable(get_vertices):
            try:
                moved_points = list(get_vertices())
            except Exception:
                moved_points = []

        # Apply translation using the drawable's translate method
        # (All drawable objects should implement this method)
        try:
            drawable.translate(x_offset, y_offset)
        except Exception as e:
            # Raise an error to be handled by the AI interface
            raise ValueError(f"Error translating drawable: {str(e)}")

        self._refresh_dependencies_after_transform(drawable, moved_points)

        # If we got here, the translation was successful
        # Redraw the canvas
        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def rotate_object(
        self,
        name: str,
        angle: float,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
    ) -> bool:
        """
        Rotates a drawable object by the specified angle.

        When *center_x* and *center_y* are both provided the rotation is
        performed around that arbitrary point (all drawable types with a
        ``rotate_around`` method are eligible, including Point and Circle).

        When both are ``None`` the existing center-of-object rotation is used
        (Points and Circles are excluded since they are rotationally invariant
        around their own center).

        Args:
            name: Name of the drawable to rotate
            angle: Angle in degrees to rotate the object
            center_x: Optional x-coordinate of the rotation center
            center_y: Optional y-coordinate of the rotation center

        Returns:
            bool: True if the rotation was successful

        Raises:
            ValueError: If no drawable with the given name is found or if rotation fails
        """
        arbitrary_center = center_x is not None and center_y is not None
        if (center_x is None) != (center_y is None):
            raise ValueError("Both center_x and center_y must be provided for rotation around an arbitrary center")

        if arbitrary_center:
            drawable = self._find_drawable_by_name(name, exclude_types=_EXCLUDE_TRANSFORM)
        else:
            # Original behaviour: skip Point/Circle which are no-ops
            drawable = self._find_drawable_by_name(
                name,
                exclude_types=_EXCLUDE_TRANSFORM + ("Point", "Circle"),
            )

        self.canvas.undo_redo_manager.archive()

        moved_points = self._gather_moved_points(drawable)

        try:
            if arbitrary_center:
                drawable.rotate_around(angle, center_x, center_y)
            else:
                drawable.rotate(angle)
        except Exception as e:
            raise ValueError(f"Error rotating drawable: {str(e)}")

        if arbitrary_center:
            self._refresh_dependencies_after_transform(drawable, moved_points)

        self._redraw()
        return True

    def reflect_object(
        self,
        name: str,
        axis: str,
        line_a: float = 0,
        line_b: float = 0,
        line_c: float = 0,
        segment_name: str = "",
    ) -> bool:
        """Reflect a drawable across an axis or line.

        Args:
            name: Name of the drawable to reflect
            axis: One of 'x_axis', 'y_axis', 'line', or 'segment'
            line_a, line_b, line_c: Coefficients for ``ax + by + c = 0`` (axis='line')
            segment_name: Named segment to use as reflection axis (axis='segment')

        Raises:
            ValueError: On invalid axis, degenerate line, or missing segment.
        """
        if axis not in ("x_axis", "y_axis", "line", "segment"):
            raise ValueError(f"Invalid reflection axis '{axis}'; use x_axis, y_axis, line, or segment")

        a, b, c = float(line_a), float(line_b), float(line_c)

        if axis == "segment":
            a, b, c = self._resolve_segment_to_line(segment_name)
            axis = "line"
        elif axis == "line":
            if a * a + b * b < 1e-18:
                raise ValueError("Line coefficients a and b must not both be zero")

        drawable = self._find_drawable_by_name(name, exclude_types=_EXCLUDE_TRANSFORM)
        self.canvas.undo_redo_manager.archive()
        moved_points = self._gather_moved_points(drawable)

        try:
            drawable.reflect(axis, a, b, c)
        except Exception as e:
            raise ValueError(f"Error reflecting drawable: {str(e)}")

        self._refresh_dependencies_after_transform(drawable, moved_points)
        self._redraw()
        return True

    def scale_object(
        self,
        name: str,
        sx: float,
        sy: float,
        cx: float,
        cy: float,
    ) -> bool:
        """Scale (dilate) a drawable from center (cx, cy).

        Args:
            name: Name of the drawable to scale
            sx: Horizontal scale factor
            sy: Vertical scale factor
            cx, cy: Center of scaling

        Raises:
            ValueError: On zero scale factor or unsupported type.
        """
        if abs(sx) < 1e-18 or abs(sy) < 1e-18:
            raise ValueError("Scale factors must not be zero")

        drawable = self._find_drawable_by_name(name, exclude_types=_EXCLUDE_TRANSFORM)
        self._validate_scale_support(drawable, sx, sy)
        self.canvas.undo_redo_manager.archive()
        moved_points = self._gather_moved_points(drawable)

        try:
            drawable.scale(sx, sy, cx, cy)
        except Exception as e:
            raise ValueError(f"Error scaling drawable: {str(e)}")

        self._refresh_dependencies_after_transform(drawable, moved_points)
        self._redraw()
        return True

    def shear_object(
        self,
        name: str,
        axis: str,
        factor: float,
        cx: float,
        cy: float,
    ) -> bool:
        """Shear a drawable along an axis from center (cx, cy).

        Args:
            name: Name of the drawable to shear
            axis: 'horizontal' or 'vertical'
            factor: Shear factor
            cx, cy: Center of shear

        Raises:
            ValueError: On invalid axis or unsupported type.
        """
        if axis not in ("horizontal", "vertical"):
            raise ValueError(f"Invalid shear axis '{axis}'; use 'horizontal' or 'vertical'")

        drawable = self._find_drawable_by_name(name, exclude_types=_EXCLUDE_TRANSFORM)
        self._validate_shear_support(drawable)
        self.canvas.undo_redo_manager.archive()
        moved_points = self._gather_moved_points(drawable)

        try:
            drawable.shear(axis, factor, cx, cy)
        except Exception as e:
            raise ValueError(f"Error shearing drawable: {str(e)}")

        self._refresh_dependencies_after_transform(drawable, moved_points)
        self._redraw()
        return True

    # ------------------------------------------------------------------
    # Segment resolution
    # ------------------------------------------------------------------

    def _resolve_segment_to_line(self, segment_name: str) -> Tuple[float, float, float]:
        """Convert a named segment to line coefficients (a, b, c).

        Raises:
            ValueError: If the segment is not found or is degenerate (zero-length).
        """
        if not segment_name:
            raise ValueError("segment_name is required when axis is 'segment'")

        segment: Optional[Segment] = None
        for d in self.canvas.drawable_manager.get_drawables():
            if d.name == segment_name and isinstance(d, Segment):
                segment = d
                break

        if segment is None:
            raise ValueError(f"No segment found with name '{segment_name}'")

        dx = segment.point2.x - segment.point1.x
        dy = segment.point2.y - segment.point1.y
        if dx * dx + dy * dy < 1e-18:
            raise ValueError(f"Segment '{segment_name}' has zero length and cannot define a reflection axis")

        # Line through two points: a = dy, b = -dx, c = -(dy*x1 - dx*y1)
        a = dy
        b = -dx
        c_val = -(a * segment.point1.x + b * segment.point1.y)
        return a, b, c_val

    # ------------------------------------------------------------------
    # Dependency refresh helpers
    # ------------------------------------------------------------------

    def _refresh_polygon_dependencies(self, polygon: Any, points: Iterable[Any]) -> None:
        dependency_manager = getattr(self.canvas, "dependency_manager", None)

        touched_point_ids: Set[int] = {id(point) for point in points}
        related_drawables: Set[Any] = set()

        related_drawables |= self._collect_segments_from_polygon(polygon, touched_point_ids)
        related_drawables |= self._collect_segments_from_canvas(points, touched_point_ids)
        related_drawables |= self._gather_dependency_children(related_drawables | {polygon}, dependency_manager)

        if not related_drawables:
            self._invalidate_drawables([polygon])
            return

        updated = self._refresh_segment_formulas(related_drawables, touched_point_ids)
        updated.append(polygon)
        self._invalidate_drawables(updated)

    def _refresh_circle_dependencies(self, circle: Any) -> None:
        dependency_manager = getattr(self.canvas, "dependency_manager", None)
        drawables = self._gather_dependency_children({circle}, dependency_manager)
        circle.circle_formula = circle._calculate_circle_algebraic_formula()
        circle.regenerate_name()
        self._invalidate_drawables([circle] + list(drawables))

    def _refresh_ellipse_dependencies(self, ellipse: Any) -> None:
        dependency_manager = getattr(self.canvas, "dependency_manager", None)
        drawables = self._gather_dependency_children({ellipse}, dependency_manager)
        ellipse.ellipse_formula = ellipse._calculate_ellipse_algebraic_formula()
        ellipse.regenerate_name()
        self._invalidate_drawables([ellipse] + list(drawables))

    def _collect_segments_from_polygon(self, polygon: Any, touched_point_ids: Set[int]) -> Set[Segment]:
        segments: Set[Segment] = set()

        for value in vars(polygon).values():
            self._harvest_segments(value, touched_point_ids, segments)

        return segments

    def _collect_segments_from_canvas(
        self,
        points: Iterable[Any],
        touched_point_ids: Set[int],
    ) -> Set[Segment]:
        segments: Set[Segment] = set()
        dependency_manager = getattr(self.canvas, "dependency_manager", None)

        if dependency_manager and hasattr(dependency_manager, "get_children"):
            for point in points:
                try:
                    children = dependency_manager.get_children(point)
                except Exception:
                    children = set()
                for child in children or []:
                    if isinstance(child, Segment) and self._segment_touches_points(child, touched_point_ids):
                        segments.add(child)

        drawable_manager = getattr(self.canvas, "drawable_manager", None)
        drawables = getattr(drawable_manager, "drawables", None) if drawable_manager else None
        if drawables and hasattr(drawables, "Segments"):
            for segment in getattr(drawables, "Segments", []):
                if self._segment_touches_points(segment, touched_point_ids):
                    segments.add(segment)

        return segments

    def _harvest_segments(self, candidate: Any, touched_point_ids: Set[int], segments: Set[Segment]) -> None:
        if candidate is None:
            return
        if isinstance(candidate, Segment) and self._segment_touches_points(candidate, touched_point_ids):
            segments.add(candidate)
            return
        if isinstance(candidate, (list, tuple, set)):
            for item in candidate:
                self._harvest_segments(item, touched_point_ids, segments)

    def _segment_touches_points(self, segment: Segment, touched_point_ids: Set[int]) -> bool:
        try:
            point1 = segment.point1
            point2 = segment.point2
        except Exception:
            return False
        return (point1 and id(point1) in touched_point_ids) or (point2 and id(point2) in touched_point_ids)

    def _gather_dependency_children(self, drawables: Set[Any], dependency_manager: Any) -> Set[Any]:
        if not dependency_manager or not hasattr(dependency_manager, "get_children"):
            return set()

        collected: Set[Any] = set()
        queue = list(drawables)
        visited: Set[Any] = set()

        while queue:
            current = queue.pop()
            if current in visited or current is None:
                continue
            visited.add(current)
            try:
                children = dependency_manager.get_children(current)
            except Exception:
                children = set()
            for child in children or []:
                if child is None or child in visited or child in collected:
                    continue
                collected.add(child)
                queue.append(child)

        return collected

    def _refresh_segment_formulas(self, drawables: Set[Any], touched_point_ids: Set[int]) -> List[Any]:
        updated: List[Any] = []
        for drawable in drawables:
            if drawable is None:
                continue
            class_name_getter = getattr(drawable, "get_class_name", None)
            class_name = class_name_getter() if callable(class_name_getter) else drawable.__class__.__name__
            if class_name == "Segment" and self._segment_touches_points(drawable, touched_point_ids):
                try:
                    drawable.line_formula = drawable._calculate_line_algebraic_formula()
                except Exception:
                    continue
            updated.append(drawable)
        return updated

    def _invalidate_drawables(self, drawables: Iterable[Any]) -> None:
        renderer = getattr(self.canvas, "renderer", None)
        invalidate = getattr(renderer, "invalidate_drawable_cache", None) if renderer else None
        if not callable(invalidate):
            return
        for drawable in drawables:
            if drawable is None:
                continue
            try:
                invalidate(drawable)
            except Exception:
                continue
