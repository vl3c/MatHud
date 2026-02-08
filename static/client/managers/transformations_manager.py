"""
MatHud Geometric Transformations Management System

Handles geometric transformations of drawable objects including translation and rotation.
Provides coordinated transformation operations with proper state management and canvas integration.

Transformation Types:
    - Translation: Moving objects by specified x and y offsets
    - Rotation: Rotating objects around specified points or their centers

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

from typing import TYPE_CHECKING, Any, Iterable, List, Set

from drawables.segment import Segment

if TYPE_CHECKING:
    from canvas import Canvas

class TransformationsManager:
    """Manages geometric transformations of drawable objects on a Canvas.

    Coordinates translation and rotation operations with proper state management,
    object validation, and canvas integration.
    """

    def __init__(self, canvas: "Canvas") -> None:
        """
        Initialize the TransformationsManager.

        Args:
            canvas: The Canvas object this manager is responsible for
        """
        self.canvas: "Canvas" = canvas

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

        class_name_getter = getattr(drawable, "get_class_name", None)
        class_name = class_name_getter() if callable(class_name_getter) else drawable.__class__.__name__

        if moved_points and class_name in {"Triangle", "Rectangle", "Polygon"}:
            self._refresh_polygon_dependencies(drawable, moved_points)
        elif class_name == "Circle":
            self._refresh_circle_dependencies(drawable)
        elif class_name == "Ellipse":
            self._refresh_ellipse_dependencies(drawable)

        # If we got here, the translation was successful
        # Redraw the canvas
        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def rotate_object(self, name: str, angle: float) -> bool:
        """
        Rotates a drawable object by the specified angle.

        Args:
            name: Name of the drawable to rotate
            angle: Angle in degrees to rotate the object

        Returns:
            bool: True if the rotation was successful

        Raises:
            ValueError: If no drawable with the given name is found or if rotation fails
        """
        # Find the drawable first to validate it exists
        drawable = None
        # Get all drawables except Points, Functions, and Circles which don't support rotation
        for d in self.canvas.drawable_manager.get_drawables():
            if d.get_class_name() in ['Function', 'Point', 'Circle']:
                continue
            if d.name == name:
                drawable = d
                break

        if not drawable:
            raise ValueError(f"No drawable found with name '{name}'")

        # Archive current state for undo/redo AFTER finding the object but BEFORE modifying it
        self.canvas.undo_redo_manager.archive()

        # Apply rotation using the drawable's rotate method
        try:
            drawable.rotate(angle)
        except Exception as e:
            # Raise an error to be handled by the AI interface
            raise ValueError(f"Error rotating drawable: {str(e)}")

        # If we got here, the rotation was successful
        # Redraw the canvas
        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

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
