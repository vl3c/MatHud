"""
MatHud Drawable Storage and Organization System

Centralized container for all drawable objects with type-based organization and efficient access patterns.
Provides clean separation of storage concerns from Canvas operations and supports layered rendering.

Storage Architecture:
    - Type-Based Organization: Groups drawables by class name for efficient access
    - Property Access: Convenient attribute-style access to drawable collections
    - Dictionary Interface: Supports both object-oriented and dictionary-style access
    - Layered Storage: Separates colored areas from geometric objects for proper z-ordering

Supported Drawable Types:
    - Points: Coordinate-based geometric primitives
    - Segments: Line segments connecting two points
    - Vectors: Directed line segments with origin and tip
    - Triangles: Three-sided polygons with vertex tracking
    - Rectangles: Four-sided polygons with diagonal point definition
    - Circles: Circular objects with center and radius
    - Ellipses: Elliptical objects with center, radii, and rotation
    - Functions: Mathematical function plots and curves
    - Angles: Angular measurements between line segments
    - ColoredAreas: Various bounded colored regions

Rendering Support:
    - Z-Order Management: Colored areas rendered behind geometric objects
    - Layered Access: get_all_with_layering() provides proper rendering order
    - Background/Foreground Separation: Efficient separation for rendering pipeline

State Management:
    - State Serialization: get_state() for undo/redo functionality
    - Clear Operations: Bulk removal for canvas reset
    - Container Introspection: Type checking and content validation

Access Patterns:
    - Property Style: container.Points, container.Segments
    - Dictionary Style: container['Point'], container['Segment']
    - Bulk Operations: get_all(), get_colored_areas(), get_non_colored_areas()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

if TYPE_CHECKING:
    from drawables.drawable import Drawable

class DrawablesContainer:
    """
    A container for storing and accessing drawable objects by their class names.
    
    This class extracts the drawable storage functionality from Canvas,
    providing a cleaner separation of concerns.
    """
    
    def __init__(self) -> None:
        """Initialize an empty drawables container."""
        self._drawables: Dict[str, List["Drawable"]] = {}
        self._renderables: Dict[str, List["Drawable"]] = {}
        
    def add(self, drawable: "Drawable") -> None:
        """
        Add a drawable to the container.
        
        Args:
            drawable: The drawable object to add
        """
        category = drawable.get_class_name()
        if category not in self._drawables:
            self._drawables[category] = []
        self._drawables[category].append(drawable)
        self._sync_renderable_entry(drawable)

    def _is_renderable(self, drawable: "Drawable") -> bool:
        renderable_attr = getattr(drawable, "is_renderable", True)
        try:
            return bool(renderable_attr)
        except Exception:
            return True

    def _apply_layering(self, colored: List["Drawable"], others: List["Drawable"]) -> List["Drawable"]:
        circles: List["Drawable"] = []
        circle_arcs: List["Drawable"] = []
        remaining: List["Drawable"] = []

        for drawable in others:
            class_name = (
                drawable.get_class_name()
                if hasattr(drawable, "get_class_name")
                else drawable.__class__.__name__
            )
            if class_name == "Circle":
                circles.append(drawable)
            elif class_name == "CircleArc":
                circle_arcs.append(drawable)
            else:
                remaining.append(drawable)

        return colored + remaining + circles + circle_arcs
        
    def _add_to_renderables(self, drawable: "Drawable") -> None:
        category = drawable.get_class_name()
        if category not in self._renderables:
            self._renderables[category] = []
        bucket = self._renderables[category]
        if drawable not in bucket:
            bucket.append(drawable)

    def _remove_from_renderables(self, drawable: "Drawable") -> None:
        category = drawable.get_class_name()
        bucket = self._renderables.get(category)
        if not bucket:
            return
        if drawable in bucket:
            bucket.remove(drawable)
            if not bucket:
                del self._renderables[category]

    def _sync_renderable_entry(self, drawable: "Drawable") -> None:
        if self._is_renderable(drawable):
            self._add_to_renderables(drawable)
        else:
            self._remove_from_renderables(drawable)

    def remove(self, drawable: "Drawable") -> bool:
        """
        Remove a drawable from the container.
        
        Args:
            drawable: The drawable object to remove
            
        Returns:
            bool: True if the drawable was removed, False otherwise
        """
        category = drawable.get_class_name()
        if category in self._drawables and drawable in self._drawables[category]:
            self._drawables[category].remove(drawable)
            if not self._drawables[category]:
                del self._drawables[category]
            self._remove_from_renderables(drawable)
            return True
        return False
        
    def get_by_class_name(self, class_name: str) -> List["Drawable"]:
        """
        Get all drawables of a specific class name (private method).
        
        Args:
            class_name: The name of the class to get drawables for
            
        Returns:
            list: List of drawables of the specified class
        """
        return self._drawables.get(class_name, [])
        
    def get_all(self) -> List["Drawable"]:
        """
        Get all drawables as a flat list.
        
        Returns:
            list: All drawables in the container
        """
        all_drawables = []
        for drawable_type in self._drawables:
            all_drawables.extend(self._drawables[drawable_type])
        return all_drawables
    
    def get_colored_areas(self) -> List["Drawable"]:
        """
        Get all colored area drawables (for background rendering).
        
        Returns:
            list: All colored area drawables in the container
        """
        colored_areas = []
        for drawable_type in self._drawables:
            if 'ColoredArea' in drawable_type:
                colored_areas.extend(self._drawables[drawable_type])
        return colored_areas
    
    def get_non_colored_areas(self) -> List["Drawable"]:
        """
        Get all non-colored area drawables (for foreground rendering).
        
        Returns:
            list: All non-colored area drawables in the container
        """
        other_drawables = []
        for drawable_type in self._drawables:
            if 'ColoredArea' not in drawable_type:
                other_drawables.extend(self._drawables[drawable_type])
        return other_drawables
    
    def get_all_with_layering(self) -> List["Drawable"]:
        """
        Get all drawables with proper layering (colored areas first, then others).
        
        Returns:
            list: All drawables with colored areas first for proper z-ordering
        """
        colored = self.get_colored_areas()
        others = self.get_non_colored_areas()
        return self._apply_layering(colored, others)

    def get_renderables_with_layering(self) -> List["Drawable"]:
        """
        Get renderable drawables with proper layering, excluding abstract objects.
        """
        colored: List["Drawable"] = []
        others: List["Drawable"] = []
        for class_name, bucket in self._renderables.items():
            if 'ColoredArea' in class_name:
                colored.extend(bucket)
            else:
                others.extend(bucket)
        return self._apply_layering(list(colored), list(others))
        
    def clear(self) -> None:
        """Remove all drawables from the container."""
        self._drawables.clear()
        self._renderables.clear()
        
    def get_state(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the state of all drawables in the container.
        
        Returns:
            dict: Dictionary of drawable states by class name
        """
        state_dict = {}
        for category, drawables in self._drawables.items():
            state_dict[category + 's'] = [drawable.get_state() for drawable in drawables]
        return state_dict

    def rebuild_renderables(self) -> None:
        """Rebuild the renderables index from the current drawable storage."""
        self._renderables = {}
        for bucket in self._drawables.values():
            for drawable in bucket:
                self._sync_renderable_entry(drawable)
        
    # Property-style access for specific drawable types (for convenience)
    @property
    def Points(self) -> List["Drawable"]:
        """Get all Point objects."""
        return self.get_by_class_name('Point')
        
    @property
    def Segments(self) -> List["Drawable"]:
        """Get all Segment objects."""
        return self.get_by_class_name('Segment')
        
    @property
    def Vectors(self) -> List["Drawable"]:
        """Get all Vector objects."""
        return self.get_by_class_name('Vector')
        
    @property
    def Triangles(self) -> List["Drawable"]:
        """Get all Triangle objects."""
        return self.get_by_class_name('Triangle')
    
    def get_triangle_by_name(self, name: str) -> Optional["Drawable"]:
        """Retrieve a triangle by name."""
        if not name:
            return None
        for triangle in self.Triangles:
            if getattr(triangle, "name", "") == name:
                return triangle
        return None
        
    @property
    def Rectangles(self) -> List["Drawable"]:
        """Get all Rectangle objects."""
        return self.get_by_class_name('Rectangle')
    
    def get_rectangle_by_name(self, name: str) -> Optional["Drawable"]:
        """Retrieve a rectangle by name."""
        if not name:
            return None
        for rectangle in self.Rectangles:
            if getattr(rectangle, "name", "") == name:
                return rectangle
        return None
    
    @property
    def Quadrilaterals(self) -> List["Drawable"]:
        """Get all Quadrilateral objects."""
        return self.get_by_class_name('Quadrilateral')

    def get_quadrilateral_by_name(self, name: str) -> Optional["Drawable"]:
        """Retrieve a quadrilateral by name."""
        if not name:
            return None
        for quadrilateral in self.Quadrilaterals:
            if getattr(quadrilateral, "name", "") == name:
                return quadrilateral
        return None

    @property
    def Pentagons(self) -> List["Drawable"]:
        """Get all Pentagon objects."""
        return self.get_by_class_name('Pentagon')

    def get_pentagon_by_name(self, name: str) -> Optional["Drawable"]:
        """Retrieve a pentagon by name."""
        if not name:
            return None
        for pentagon in self.Pentagons:
            if getattr(pentagon, "name", "") == name:
                return pentagon
        return None

    @property
    def Hexagons(self) -> List["Drawable"]:
        """Get all Hexagon objects."""
        return self.get_by_class_name('Hexagon')

    def get_hexagon_by_name(self, name: str) -> Optional["Drawable"]:
        """Retrieve a hexagon by name."""
        if not name:
            return None
        for hexagon in self.Hexagons:
            if getattr(hexagon, "name", "") == name:
                return hexagon
        return None

    def iter_polygons(self, allowed_classes: Optional[Iterable[str]] = None) -> Iterable["Drawable"]:
        """Iterate over stored polygon drawables, optionally filtered by class name."""
        polygon_classes = ("Triangle", "Quadrilateral", "Rectangle", "Pentagon", "Hexagon")
        target_classes = tuple(allowed_classes) if allowed_classes else polygon_classes
        for class_name in target_classes:
            for drawable in self.get_by_class_name(class_name):
                yield drawable

    def get_polygon_by_name(self, name: str, allowed_classes: Optional[Iterable[str]] = None) -> Optional["Drawable"]:
        """Retrieve the first polygon matching the provided name."""
        if not name:
            return None
        for polygon in self.iter_polygons(allowed_classes):
            if getattr(polygon, "name", "") == name:
                return polygon
        return None
        
    @property
    def Circles(self) -> List["Drawable"]:
        """Get all Circle objects."""
        return self.get_by_class_name('Circle')
        
    @property
    def Ellipses(self) -> List["Drawable"]:
        """Get all Ellipse objects."""
        return self.get_by_class_name('Ellipse')
        
    @property
    def Functions(self) -> List["Drawable"]:
        """Get all Function objects."""
        return self.get_by_class_name('Function')

    @property
    def PiecewiseFunctions(self) -> List["Drawable"]:
        """Get all PiecewiseFunction objects."""
        return self.get_by_class_name('PiecewiseFunction')

    @property
    def ParametricFunctions(self) -> List["Drawable"]:
        """Get all ParametricFunction objects."""
        return self.get_by_class_name('ParametricFunction')

    @property
    def Labels(self) -> List["Drawable"]:
        """Get all Label objects."""
        return self.get_by_class_name('Label')
        
    @property
    def ColoredAreas(self) -> List["Drawable"]:
        """Get all ColoredArea objects."""
        return self.get_by_class_name('ColoredArea')
        
    @property
    def FunctionsBoundedColoredAreas(self) -> List["Drawable"]:
        """Get all FunctionsBoundedColoredArea objects."""
        return self.get_by_class_name('FunctionsBoundedColoredArea')
        
    @property
    def Angles(self) -> List["Drawable"]:
        """Get all Angle objects."""
        return self.get_by_class_name('Angle')

    @property
    def CircleArcs(self) -> List["Drawable"]:
        """Get all CircleArc objects."""
        return self.get_by_class_name('CircleArc')
        
    @property
    def SegmentsBoundedColoredAreas(self) -> List["Drawable"]:
        """Get all SegmentsBoundedColoredArea objects."""
        return self.get_by_class_name('SegmentsBoundedColoredArea')
        
    @property
    def FunctionSegmentBoundedColoredAreas(self) -> List["Drawable"]:
        """Get all FunctionSegmentBoundedColoredArea objects."""
        return self.get_by_class_name('FunctionSegmentBoundedColoredArea')

    @property
    def ClosedShapeColoredAreas(self) -> List["Drawable"]:
        """Get all ClosedShapeColoredArea objects."""
        return self.get_by_class_name('ClosedShapeColoredArea')
        
    # Direct dictionary-like access
    def __getitem__(self, key: str) -> List["Drawable"]:
        """
        Allow dictionary-like access to drawable types.
        
        Args:
            key: The class name
            
        Returns:
            list: List of drawables of the specified class
        """
        return self.get_by_class_name(key)
        
    def __contains__(self, key: str) -> bool:
        """
        Check if the container has drawables of a specific class.
        
        Args:
            key: The class name
            
        Returns:
            bool: True if drawables of the specified class exist
        """
        return key in self._drawables 