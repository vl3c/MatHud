"""
MatHud Mathematical Canvas System

Core SVG-based canvas for interactive mathematical visualization and geometric construction.
Serves as the central coordinator for all drawable objects, coordinate systems, and user interactions.

Key Features:
    - SVG viewport management with zoom/pan capabilities
    - Geometric object creation (points, segments, vectors, shapes, functions)
    - Coordinate system with Cartesian grid visualization
    - Undo/redo state management for user actions
    - Mathematical computation history tracking
    - Colored area visualization between objects
    - Angle measurement and display
    - Object transformations (translate, rotate)

Architecture:
    - Canvas: Central coordinator and state manager
    - DrawableManager: Handles all geometric object lifecycle
    - UndoRedoManager: Provides state archiving and restoration
    - TransformationsManager: Manages object positioning and rotation
    - Cartesian2Axis: Coordinate system visualization

Dependencies:
    - browser.document: DOM manipulation for SVG rendering
    - geometry: Geometric object definitions (Point, Position, etc.)
    - cartesian_system_2axis: Coordinate grid system
    - coordinate_mapper: Coordinate transformation management
    - utils.*: Mathematical, styling, and geometry utilities
    - managers.*: Specialized management components
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from geometry import Point
from cartesian_system_2axis import Cartesian2Axis
from coordinate_mapper import CoordinateMapper
from utils.math_utils import MathUtils
from utils.style_utils import StyleUtils
from utils.geometry_utils import GeometryUtils
from utils.computation_utils import ComputationUtils
from managers.undo_redo_manager import UndoRedoManager
from managers.drawable_manager import DrawableManager
from managers.drawable_dependency_manager import DrawableDependencyManager
from managers.transformations_manager import TransformationsManager
from constants import DEFAULT_RENDERER_MODE
from rendering.factory import create_renderer
from rendering.interfaces import RendererProtocol

if TYPE_CHECKING:
    from drawables.drawable import Drawable


class Canvas:
    """Central mathematical visualization canvas coordinating all drawable objects and interactions.
    
    Manages the SVG viewport, coordinate system, geometric objects, and user interactions.
    Provides the main interface for creating, manipulating, and visualizing mathematical content.
    
    Attributes:
        width (float): Canvas viewport width in pixels
        height (float): Canvas viewport height in pixels
        coordinate_mapper (CoordinateMapper): Centralized coordinate transformation service
        computations (list): History of mathematical computations performed
        cartesian2axis (Cartesian2Axis): Coordinate grid system
        drawable_manager (DrawableManager): Manages all geometric objects
        dependency_manager (DrawableDependencyManager): Tracks drawable relationships
        undo_redo_manager (UndoRedoManager): Handles state archiving/restoration
        transformations_manager (TransformationsManager): Manages object transformations
        
    Legacy Properties (delegated to coordinate_mapper):
        center (Position): Current viewport center point
        scale_factor (float): Current zoom level (1.0 = normal)
        offset (Position): Current pan offset
        zoom_point (Position): Center point for zoom operations
        zoom_direction (int): Current zoom direction (-1=in, 1=out, 0=none)
        zoom_step (float): Zoom increment per step
    """
    def __init__(self, width: float, height: float, draw_enabled: bool = True, renderer: Optional[RendererProtocol] = None) -> None:
        """Initialize the mathematical canvas with specified dimensions.
        
        Sets up the coordinate system, managers, and initial state for mathematical visualization.
        
        Args:
            width (float): Canvas viewport width in pixels
            height (float): Canvas viewport height in pixels
            draw_enabled (bool): Whether to enable immediate drawing operations
        """
        self.computations: List[Dict[str, Any]] = []  # Store computation history
        self.width: float = width
        self.height: float = height
        
        # Initialize CoordinateMapper for coordinate transformation management
        self.coordinate_mapper: CoordinateMapper = CoordinateMapper(width, height)
        
        # Legacy properties for backward compatibility - delegate to coordinate_mapper
        self.dragging: bool = False
        self.draw_enabled: bool = draw_enabled
        
        # Initialize coordinate system and managers
        self.cartesian2axis: Cartesian2Axis = Cartesian2Axis(self.coordinate_mapper)
        
        # Add managers
        self.undo_redo_manager: UndoRedoManager = UndoRedoManager(self)
        self.drawable_manager: DrawableManager = DrawableManager(self)
        self.dependency_manager: DrawableDependencyManager = self.drawable_manager.dependency_manager
        self.transformations_manager: TransformationsManager = TransformationsManager(self)

        # Initialize renderer lazily to avoid hard dependency in non-browser tests
        if renderer is not None:
            self.renderer: Optional[RendererProtocol] = renderer
        else:
            self.renderer = cast(Optional[RendererProtocol], create_renderer(DEFAULT_RENDERER_MODE))
        
        if self.draw_enabled and self.renderer is not None:
            try:
                self.renderer.render_cartesian(self.cartesian2axis, self.coordinate_mapper)
            except Exception:
                pass
        self._register_renderer_handlers()

    def add_drawable(self, drawable: "Drawable") -> None:
        drawable.canvas = self  # Set the drawable's canvas reference
        self.drawable_manager.drawables.add(drawable)

    def _register_renderer_handlers(self) -> None:
        """Register renderer handlers for all drawable types."""
        if self.renderer is None:
            return
        try:
            self.renderer.register_default_drawables()
        except AttributeError:
            self._register_renderer_handlers_legacy()
        except Exception:
            self._register_renderer_handlers_legacy()

    def _register_renderer_handlers_legacy(self) -> None:
        if self.renderer is None:
            return
        try:
            from drawables.point import Point as _Point
            if hasattr(self.renderer, 'register_point'):
                self.renderer.register_point(_Point)
        except Exception:
            pass
        try:
            from drawables.segment import Segment as _Segment
            if hasattr(self.renderer, 'register_segment'):
                self.renderer.register_segment(_Segment)
        except Exception:
            pass
        try:
            from drawables.circle import Circle as _Circle
            if hasattr(self.renderer, 'register_circle'):
                self.renderer.register_circle(_Circle)
        except Exception:
            pass
        try:
            from drawables.ellipse import Ellipse as _Ellipse
            if hasattr(self.renderer, 'register_ellipse'):
                self.renderer.register_ellipse(_Ellipse)
        except Exception:
            pass
        try:
            from drawables.vector import Vector as _Vector
            if hasattr(self.renderer, 'register_vector'):
                self.renderer.register_vector(_Vector)
        except Exception:
            pass
        try:
            from drawables.angle import Angle as _Angle
            if hasattr(self.renderer, 'register_angle'):
                self.renderer.register_angle(_Angle)
        except Exception:
            pass
        try:
            from drawables.function import Function as _Function
            if hasattr(self.renderer, 'register_function'):
                self.renderer.register_function(_Function)
        except Exception:
            pass
        try:
            from drawables.triangle import Triangle as _Triangle
            if hasattr(self.renderer, 'register_triangle'):
                self.renderer.register_triangle(_Triangle)
        except Exception:
            pass
        try:
            from drawables.rectangle import Rectangle as _Rectangle
            if hasattr(self.renderer, 'register_rectangle'):
                self.renderer.register_rectangle(_Rectangle)
        except Exception:
            pass
        try:
            from drawables.functions_bounded_colored_area import FunctionsBoundedColoredArea as _FBCA
            if hasattr(self.renderer, 'register_functions_bounded_colored_area'):
                self.renderer.register_functions_bounded_colored_area(_FBCA)
        except Exception:
            pass
        try:
            from drawables.function_segment_bounded_colored_area import FunctionSegmentBoundedColoredArea as _FSBCA
            if hasattr(self.renderer, 'register_function_segment_bounded_colored_area'):
                self.renderer.register_function_segment_bounded_colored_area(_FSBCA)
        except Exception:
            pass
        try:
            from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea as _SBCA
            if hasattr(self.renderer, 'register_segments_bounded_colored_area'):
                self.renderer.register_segments_bounded_colored_area(_SBCA)
        except Exception:
            pass
        try:
            from drawables.label import Label as _Label
            if hasattr(self.renderer, 'register_label'):
                self.renderer.register_label(_Label)
        except Exception:
            pass
        try:
            from drawables.circle_arc import CircleArc as _CircleArc
            if hasattr(self.renderer, 'register_circle_arc'):
                self.renderer.register_circle_arc(_CircleArc)
        except Exception:
            pass

    def draw(self, apply_zoom: bool = False) -> None:
        if not self.draw_enabled:
            return
        renderer = self.renderer
        renderer_begin = getattr(renderer, "begin_frame", None) if renderer is not None else None
        renderer_end = getattr(renderer, "end_frame", None) if renderer is not None else None

        if callable(renderer_begin):
            try:
                renderer_begin()
            except Exception:
                pass

        try:
            if renderer is not None:
                skip_clear = bool(getattr(renderer, "SKIP_AUTO_CLEAR", False))
                if not skip_clear:
                    try:
                        renderer.clear()
                    except Exception:
                        pass
                try:
                    if apply_zoom and hasattr(self.cartesian2axis, '_invalidate_cache_on_zoom'):
                        self.cartesian2axis._invalidate_cache_on_zoom()
                    renderer.render_cartesian(self.cartesian2axis, self.coordinate_mapper)
                except Exception:
                    pass

            # Draw all drawables - coordinate transformations handled by CoordinateMapper
            for drawable in self.drawable_manager.get_drawables():
                if apply_zoom and hasattr(drawable, '_invalidate_cache_on_zoom'):
                    drawable._invalidate_cache_on_zoom()
                if renderer is not None:
                    try:
                        renderer.render(drawable, self.coordinate_mapper)
                    except Exception:
                        pass
        finally:
            if callable(renderer_end):
                try:
                    renderer_end()
                except Exception:
                    pass

    def _is_drawable_visible(self, drawable: "Drawable") -> bool:
        """Best-effort visibility check to avoid rendering off-canvas objects.

        Mirrors prior behavior for segments and points; other types default to visible
        because they manage their own bounds or are inexpensive.
        """
        try:
            class_name = drawable.get_class_name() if hasattr(drawable, 'get_class_name') else drawable.__class__.__name__
        except Exception:
            class_name = drawable.__class__.__name__

        try:
            if class_name == 'Point':
                # Use screen coordinates if available, else compute
                # Math-only point; map via CoordinateMapper
                x, y = self.coordinate_mapper.math_to_screen(drawable.x, drawable.y)
                return self.is_point_within_canvas_visible_area(x, y)

            if class_name == 'Segment':
                p1 = drawable.point1
                p2 = drawable.point2
                x1, y1 = self.coordinate_mapper.math_to_screen(p1.x, p1.y)
                x2, y2 = self.coordinate_mapper.math_to_screen(p2.x, p2.y)
                return (
                    self.is_point_within_canvas_visible_area(x1, y1) or
                    self.is_point_within_canvas_visible_area(x2, y2) or
                    self.any_segment_part_visible_in_canvas_area(x1, y1, x2, y2)
                )

            if class_name == 'Vector':
                seg = getattr(drawable, 'segment', None)
                if seg is None:
                    return True
                p1 = seg.point1
                p2 = seg.point2
                x1, y1 = self.coordinate_mapper.math_to_screen(p1.x, p1.y)
                x2, y2 = self.coordinate_mapper.math_to_screen(p2.x, p2.y)
                return (
                    self.is_point_within_canvas_visible_area(x1, y1) or
                    self.is_point_within_canvas_visible_area(x2, y2) or
                    self.any_segment_part_visible_in_canvas_area(x1, y1, x2, y2)
                )

            # Default: visible
            return True
        except Exception:
            return True
    
    # Removed legacy zoom displacement; zoom handled via CoordinateMapper
    
    def _apply_cartesian_zoom_displacement(self, zoom_point: Point, zoom_direction: int, zoom_step: float) -> None:
        """Apply zoom displacement to the cartesian coordinate system origin.
        
        This ensures the coordinate grid participates in the zoom-towards-point effect
        by adjusting the CoordinateMapper's offset based on the distance from the 
        cartesian origin to the zoom point.
        """
        try:
            # Get current cartesian origin screen coordinates
            cartesian_origin = self.cartesian2axis.origin
            
            # Calculate distance from cartesian origin to zoom point
            dx = zoom_point.x - cartesian_origin.x
            dy = zoom_point.y - cartesian_origin.y
            distance = math.sqrt(dx * dx + dy * dy)
            
            # Calculate displacement magnitude
            displacement = distance * zoom_step * zoom_direction
            
            # Normalize direction vector and apply displacement to coordinate mapper offset
            if distance > 0:
                dx /= distance
                dy /= distance
                
                # Apply displacement to CoordinateMapper offset to move the entire coordinate system
                self.coordinate_mapper.offset.x += displacement * dx
                self.coordinate_mapper.offset.y += displacement * dy
                
        except Exception as e:
            print(f"Error applying cartesian zoom displacement: {str(e)}")
    
    # _apply_point_zoom_displacement removed (legacy)

    def _draw_cartesian(self, apply_zoom: bool = False) -> None:
        # Handle cartesian system cache invalidation if needed
        if apply_zoom and hasattr(self.cartesian2axis, '_invalidate_cache_on_zoom'):
            self.cartesian2axis._invalidate_cache_on_zoom()
        self.cartesian2axis.draw()

    def _fix_drawable_canvas_references(self) -> None:
        for drawable in self.drawable_manager.get_drawables():
            drawable.canvas = self

    def clear(self) -> None:
        """Clear all drawables"""
        self.archive()
        self.drawable_manager.drawables.clear()
        
        # Reset the name generator state
        if hasattr(self.drawable_manager, 'name_generator') and hasattr(self.drawable_manager.name_generator, 'reset_state'):
            self.drawable_manager.name_generator.reset_state()
        
        self.reset()

    def reset(self) -> None:
        """Reset the canvas to its initial state"""
        # Reset coordinate transformations using CoordinateMapper
        self.coordinate_mapper.reset_transformations()
        
        # Reset other canvas state
        self.dragging = False
        
        # Reset cartesian system and drawables
        self.cartesian2axis.reset()
        for drawable in self.get_drawables():
            drawable.reset()
        self.draw()

    def archive(self) -> None:
        """Archive the current state for undo functionality"""
        self.undo_redo_manager.archive()

    def undo(self) -> bool:
        """Restores the last archived state from the undo stack"""
        return bool(self.undo_redo_manager.undo())

    def redo(self) -> bool:
        """Restores the last undone state from the redo stack"""
        return bool(self.undo_redo_manager.redo())

    def get_drawables(self) -> List["Drawable"]:
        """Get all drawables as a flat list"""
        return cast(List["Drawable"], self.drawable_manager.get_drawables())

    def get_drawables_state(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.drawable_manager.drawables.get_state())

    def get_drawables_by_class_name(self, class_name: str) -> List["Drawable"]:
        """Get drawables of a specific class name"""
        return cast(List["Drawable"], self.drawable_manager.drawables.get_by_class_name(class_name))

    def get_cartesian2axis_state(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.cartesian2axis.get_state())

    def get_canvas_state(self) -> Dict[str, Any]:
        state = self.get_drawables_state()
        cartesian_state = self.get_cartesian2axis_state()
        if cartesian_state is not None:
            state.update(cartesian_state)
        if self.computations:  # Add computations to state if they exist
            state["computations"] = self.computations
        return state

    def get_point(self, x: float, y: float) -> Optional[Point]:
        """Get a point at the specified coordinates"""
        return self.drawable_manager.get_point(x, y)

    def get_point_by_name(self, name: str) -> Optional[Point]:
        """Get a point by its name"""
        return self.drawable_manager.get_point_by_name(name)

    def get_label_by_name(self, name: str) -> Optional["Drawable"]:
        """Get a label by its name."""
        return cast(Optional["Drawable"], self.drawable_manager.get_label_by_name(name))

    def create_point(self, x: float, y: float, name: str = "", extra_graphics: bool = True) -> Point:
        """Create a point at the specified coordinates"""
        return self.drawable_manager.create_point(x, y, name, extra_graphics)

    def create_label(
        self,
        x: float,
        y: float,
        text: str,
        *,
        name: str = "",
        color: Optional[str] = None,
        font_size: Optional[float] = None,
        rotation_degrees: Optional[float] = None,
    ) -> "Drawable":
        """Create a label at the specified coordinates."""
        return self.drawable_manager.create_label(
            x,
            y,
            text,
            name=name,
            color=color,
            font_size=font_size,
            rotation_degrees=rotation_degrees,
        )

    def delete_point(self, x: float, y: float) -> bool:
        """Delete a point at the specified coordinates"""
        return bool(self.drawable_manager.delete_point(x, y))

    def delete_label(self, name: str) -> bool:
        """Delete a label by its name."""
        return bool(self.drawable_manager.delete_label(name))
    
    def update_label(
        self,
        name: str,
        new_text: Optional[str] = None,
        new_x: Optional[float] = None,
        new_y: Optional[float] = None,
        new_color: Optional[str] = None,
        new_font_size: Optional[float] = None,
        new_rotation_degrees: Optional[float] = None,
    ) -> bool:
        """Update editable properties of a label."""
        return bool(
            self.drawable_manager.update_label(
                name,
                new_text=new_text,
                new_x=new_x,
                new_y=new_y,
                new_color=new_color,
                new_font_size=new_font_size,
                new_rotation_degrees=new_rotation_degrees,
            )
        )

    def delete_point_by_name(self, name: str) -> bool:
        """Delete a point by its name"""
        return bool(self.drawable_manager.delete_point_by_name(name))

    def update_point(
        self,
        point_name: str,
        new_name: Optional[str] = None,
        new_x: Optional[float] = None,
        new_y: Optional[float] = None,
        new_color: Optional[str] = None,
    ) -> bool:
        """Update properties of a solitary point."""
        return bool(
            self.drawable_manager.point_manager.update_point(
                point_name,
                new_name=new_name,
                new_x=new_x,
                new_y=new_y,
                new_color=new_color,
            )
        )

    def is_point_within_canvas_visible_area(self, x: float, y: float) -> bool:
        """Check if a point is within the visible area of the canvas"""
        return (0 <= x <= self.width) and (0 <= y <= self.height)

    def get_segment_by_coordinates(self, x1: float, y1: float, x2: float, y2: float) -> Optional["Drawable"]:
        """Get a segment by its endpoint coordinates"""
        return self.drawable_manager.get_segment_by_coordinates(x1, y1, x2, y2)

    def get_segment_by_name(self, name: str) -> Optional["Drawable"]:
        """Get a segment by its name"""
        return self.drawable_manager.get_segment_by_name(name)

    def get_segment_by_points(self, p1: Point, p2: Point) -> Optional["Drawable"]:
        """Get a segment by its endpoint points"""
        return self.drawable_manager.get_segment_by_points(p1, p2)

    def create_segment(self, x1: float, y1: float, x2: float, y2: float, name: str = "", extra_graphics: bool = True) -> "Drawable":
        """Create a segment between two points"""
        return self.drawable_manager.create_segment(x1, y1, x2, y2, name, extra_graphics)

    def delete_segment(self, x1: float, y1: float, x2: float, y2: float, delete_children: bool = True, delete_parents: bool = False) -> bool:
        """Delete a segment by its endpoint coordinates"""
        return bool(self.drawable_manager.delete_segment(x1, y1, x2, y2, delete_children, delete_parents))

    def delete_segment_by_name(self, name: str, delete_children: bool = True, delete_parents: bool = False) -> bool:
        """Delete a segment by its name"""
        return bool(self.drawable_manager.delete_segment_by_name(name, delete_children, delete_parents))

    def update_segment(
        self,
        name: str,
        new_color: Optional[str] = None,
    ) -> bool:
        """Update editable properties of a segment."""
        return bool(
            self.drawable_manager.update_segment(
                name,
                new_color=new_color,
            )
        )

    def any_segment_part_visible_in_canvas_area(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Check if any part of a segment is visible in the canvas area"""
        intersect_top = MathUtils.segments_intersect(x1, y1, x2, y2, 0, 0, self.width, 0)
        intersect_right = MathUtils.segments_intersect(x1, y1, x2, y2, self.width, 0, self.width, self.height)
        intersect_bottom = MathUtils.segments_intersect(x1, y1, x2, y2, self.width, self.height, 0, self.height)
        intersect_left = MathUtils.segments_intersect(x1, y1, x2, y2, 0, self.height, 0, 0)
        point1_visible: bool = self.is_point_within_canvas_visible_area(x1, y1)
        point2_visible: bool = self.is_point_within_canvas_visible_area(x2, y2)
        return bool(intersect_top or intersect_right or intersect_bottom or intersect_left or point1_visible or point2_visible)

    def get_vector(self, x1: float, y1: float, x2: float, y2: float) -> Optional["Drawable"]:
        """Get a vector by its origin and tip coordinates"""
        return self.drawable_manager.get_vector(x1, y1, x2, y2)

    def create_vector(self, origin_x: float, origin_y: float, tip_x: float, tip_y: float, name: str = "", extra_graphics: bool = True) -> "Drawable":
        """Create a vector from origin to tip"""
        return self.drawable_manager.create_vector(origin_x, origin_y, tip_x, tip_y, name, extra_graphics)

    def delete_vector(self, origin_x: float, origin_y: float, tip_x: float, tip_y: float) -> bool:
        """Delete a vector by its origin and tip coordinates"""
        return bool(self.drawable_manager.delete_vector(origin_x, origin_y, tip_x, tip_y))
    
    def update_vector(
        self,
        name: str,
        new_color: Optional[str] = None,
    ) -> bool:
        """Update editable properties of a vector."""
        return bool(
            self.drawable_manager.update_vector(
                name,
                new_color=new_color,
            )
        )

    def get_triangle(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> Optional["Drawable"]:
        """Get a triangle by its vertex coordinates"""
        return self.drawable_manager.get_triangle(x1, y1, x2, y2, x3, y3)

    def create_triangle(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, name: str = "", extra_graphics: bool = True) -> "Drawable":
        """Create a triangle with the specified vertices"""
        return self.drawable_manager.create_triangle(x1, y1, x2, y2, x3, y3, name, extra_graphics)

    def delete_triangle(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> bool:
        """Delete a triangle by its vertex coordinates"""
        return bool(self.drawable_manager.delete_triangle(x1, y1, x2, y2, x3, y3))
    
    def update_triangle(
        self,
        name: str,
        new_color: Optional[str] = None,
    ) -> bool:
        """Update editable properties of a triangle."""
        return bool(
            self.drawable_manager.update_triangle(
                name,
                new_color=new_color,
            )
        )

    def get_rectangle_by_diagonal_points(self, px: float, py: float, opposite_px: float, opposite_py: float) -> Optional["Drawable"]:
        """Get a rectangle by its diagonal points"""
        return self.drawable_manager.get_rectangle_by_diagonal_points(px, py, opposite_px, opposite_py)

    def get_rectangle_by_name(self, name: str) -> Optional["Drawable"]:
        """Get a rectangle by its name"""
        return self.drawable_manager.get_rectangle_by_name(name)

    def create_rectangle(self, px: float, py: float, opposite_px: float, opposite_py: float, name: str = "", extra_graphics: bool = True) -> "Drawable":
        """Create a rectangle with the specified diagonal points"""
        return self.drawable_manager.create_rectangle(px, py, opposite_px, opposite_py, name, extra_graphics)

    def delete_rectangle(self, name: str) -> bool:
        """Delete a rectangle by its name"""
        return bool(self.drawable_manager.delete_rectangle(name))
    
    def update_rectangle(
        self,
        name: str,
        new_color: Optional[str] = None,
    ) -> bool:
        """Update editable properties of a rectangle."""
        return bool(
            self.drawable_manager.update_rectangle(
                name,
                new_color=new_color,
            )
        )

    def get_circle(self, center_x: float, center_y: float, radius: float) -> Optional["Drawable"]:
        """Get a circle by its center coordinates and radius"""
        return self.drawable_manager.get_circle(center_x, center_y, radius)

    def get_circle_by_name(self, name: str) -> Optional["Drawable"]:
        """Get a circle by its name"""
        return self.drawable_manager.get_circle_by_name(name)

    def create_circle(self, center_x: float, center_y: float, radius: float, name: str = "", extra_graphics: bool = True) -> "Drawable":
        """Create a circle with the specified center and radius"""
        return self.drawable_manager.create_circle(center_x, center_y, radius, name, extra_graphics)

    def delete_circle(self, name: str) -> bool:
        """Delete a circle by its name"""
        return bool(self.drawable_manager.delete_circle(name))
    
    def update_circle(
        self,
        name: str,
        new_color: Optional[str] = None,
        new_center_x: Optional[float] = None,
        new_center_y: Optional[float] = None,
    ) -> bool:
        """Update editable properties of a circle."""
        return bool(
            self.drawable_manager.update_circle(
                name,
                new_color=new_color,
                new_center_x=new_center_x,
                new_center_y=new_center_y,
            )
        )

    def get_ellipse(self, center_x: float, center_y: float, radius_x: float, radius_y: float) -> Optional["Drawable"]:
        """Get an ellipse by its center coordinates and radii"""
        return self.drawable_manager.get_ellipse(center_x, center_y, radius_x, radius_y)

    def get_ellipse_by_name(self, name: str) -> Optional["Drawable"]:
        """Get an ellipse by its name"""
        return self.drawable_manager.get_ellipse_by_name(name)

    def create_ellipse(self, center_x: float, center_y: float, radius_x: float, radius_y: float, rotation_angle: float = 0, name: str = "", extra_graphics: bool = True) -> "Drawable":
        """Create an ellipse with the specified center, radii, and rotation"""
        return self.drawable_manager.create_ellipse(center_x, center_y, radius_x, radius_y, rotation_angle, name, extra_graphics)

    def delete_ellipse(self, name: str) -> bool:
        """Delete an ellipse by its name"""
        return bool(self.drawable_manager.delete_ellipse(name))
    
    def update_ellipse(
        self,
        name: str,
        new_color: Optional[str] = None,
        new_radius_x: Optional[float] = None,
        new_radius_y: Optional[float] = None,
        new_rotation_angle: Optional[float] = None,
        new_center_x: Optional[float] = None,
        new_center_y: Optional[float] = None,
    ) -> bool:
        """Update editable properties of an ellipse."""
        return bool(
            self.drawable_manager.update_ellipse(
                name,
                new_color=new_color,
                new_radius_x=new_radius_x,
                new_radius_y=new_radius_y,
                new_rotation_angle=new_rotation_angle,
                new_center_x=new_center_x,
                new_center_y=new_center_y,
            )
        )

    def get_function(self, name: str) -> Optional["Drawable"]:
        """Get a function by its name"""
        return self.drawable_manager.get_function(name)

    def draw_function(self, function_string: str, name: str, left_bound: Optional[float] = None, right_bound: Optional[float] = None) -> "Drawable":
        """Draw a function on the canvas"""
        return self.drawable_manager.draw_function(function_string, name, left_bound, right_bound)

    def delete_function(self, name: str) -> bool:
        """Delete a function by its name"""
        return bool(self.drawable_manager.delete_function(name))
    
    def update_function(
        self,
        name: str,
        new_color: Optional[str] = None,
        new_left_bound: Optional[float] = None,
        new_right_bound: Optional[float] = None,
    ) -> bool:
        """Update editable properties of a plotted function."""
        return bool(
            self.drawable_manager.update_function(
                name,
                new_color=new_color,
                new_left_bound=new_left_bound,
                new_right_bound=new_right_bound,
            )
        )

    def translate_object(self, name: str, x_offset: float, y_offset: float) -> bool:
        """Translates a drawable object by the specified offset"""
        return bool(self.transformations_manager.translate_object(name, x_offset, y_offset))
        
    def rotate_object(self, name: str, angle: float) -> bool:
        """Rotates a drawable object by the specified angle"""
        return bool(self.transformations_manager.rotate_object(name, angle))

    def has_computation(self, expression: str) -> bool:
        """Check if a computation with the given expression already exists."""
        return bool(ComputationUtils.has_computation(self.computations, expression))

    def add_computation(self, expression: str, result: Any) -> None:
        """Add a computation to the history if it doesn't already exist."""
        self.computations = ComputationUtils.add_computation(self.computations, expression, result)

    def zoom_to_bounds(self, left_bound: float, right_bound: float, top_bound: float, bottom_bound: float) -> bool:
        """Fit the viewport to the supplied math bounds and refresh the grid."""
        self.coordinate_mapper.set_visible_bounds(left_bound, right_bound, top_bound, bottom_bound)
        if hasattr(self.cartesian2axis, '_invalidate_cache_on_zoom'):
            self.cartesian2axis._invalidate_cache_on_zoom()
        self.draw(apply_zoom=True)
        return True

    def find_largest_connected_shape(self, shape: "Drawable") -> tuple[Optional["Drawable"], Optional[str]]:
        """Find the largest shape that shares segments with the given shape.
        Returns a tuple (largest_parent_shape, shape_type) where shape_type is the class name
        or None if no larger shape is found."""
        if not shape:
            return None, None

        # Get all shapes from the canvas
        rectangles = self.drawable_manager.drawables.Rectangles
        triangles = self.drawable_manager.drawables.Triangles

        # If the shape is a rectangle, don't check for parent shapes
        if shape.get_class_name() == 'Rectangle':
            return None, None

        # Check rectangles first as they are larger
        for rect in rectangles:
            if rect != shape:  # Don't compare with itself
                shared_segs = self.get_shared_segments(shape, rect)
                if shared_segs:  # If any segments are shared with a rectangle, return it
                    return rect, rect.get_class_name()

        # Only check triangles if no rectangle was found and the shape isn't a triangle
        if shape.get_class_name() == 'Triangle':
            return None, None

        largest_parent_shape = None
        max_segments = 0

        for tri in triangles:
            if tri != shape:  # Don't compare with itself
                shared_segs = self.get_shared_segments(shape, tri)
                if shared_segs and len(shared_segs) > max_segments:
                    largest_parent_shape = tri
                    max_segments = len(shared_segs)

        return largest_parent_shape, largest_parent_shape.get_class_name() if largest_parent_shape else None

    def get_shared_segments(self, shape1: "Drawable", shape2: "Drawable") -> List["Drawable"]:
        """Check if two shapes share any segments.
        Returns a list of shared segments."""
        shape1_segments = []
        shape2_segments = []

        # Get segments from shape1
        if hasattr(shape1, 'segment1'):
            shape1_segments.append(shape1.segment1)
        if hasattr(shape1, 'segment2'):
            shape1_segments.append(shape1.segment2)
        if hasattr(shape1, 'segment3'):
            shape1_segments.append(shape1.segment3)
        if hasattr(shape1, 'segment4'):
            shape1_segments.append(shape1.segment4)

        # Get segments from shape2
        if hasattr(shape2, 'segment1'):
            shape2_segments.append(shape2.segment1)
        if hasattr(shape2, 'segment2'):
            shape2_segments.append(shape2.segment2)
        if hasattr(shape2, 'segment3'):
            shape2_segments.append(shape2.segment3)
        if hasattr(shape2, 'segment4'):
            shape2_segments.append(shape2.segment4)

        # Find shared segments
        shared_segments = []
        for s1 in shape1_segments:
            for s2 in shape2_segments:
                if s1 == s2:
                    shared_segments.append(s1)

        return shared_segments

    def create_colored_area(self, drawable1_name: str, drawable2_name: Optional[str] = None, left_bound: Optional[float] = None, right_bound: Optional[float] = None, color: str = "lightblue", opacity: float = 0.3) -> "Drawable":
        """Creates a colored area between two functions, two segments, or a function and a segment"""
        return self.drawable_manager.create_colored_area(drawable1_name, drawable2_name, left_bound, right_bound, color, opacity)
        
    def delete_colored_area(self, name: str) -> bool:
        """Deletes a colored area with the given name"""
        return bool(self.drawable_manager.delete_colored_area(name))
        
    def delete_colored_areas_for_function(self, func: "Drawable") -> None:
        """Deletes all colored areas associated with a function"""
        self.drawable_manager.delete_colored_areas_for_function(func)
        
    def delete_colored_areas_for_segment(self, segment: "Drawable") -> None:
        """Deletes all colored areas associated with a segment"""
        self.drawable_manager.delete_colored_areas_for_segment(segment)
        
    def get_colored_areas_for_drawable(self, drawable: "Drawable") -> List["Drawable"]:
        """Gets all colored areas associated with a drawable (function or segment)"""
        return cast(List["Drawable"], self.drawable_manager.get_colored_areas_for_drawable(drawable))
        
    def update_colored_area(
        self,
        name: str,
        new_color: Optional[str] = None,
        new_opacity: Optional[float] = None,
        new_left_bound: Optional[float] = None,
        new_right_bound: Optional[float] = None,
    ) -> bool:
        """Update editable properties of a colored area."""
        return bool(
            self.drawable_manager.update_colored_area(
                name,
                new_color=new_color,
                new_opacity=new_opacity,
                new_left_bound=new_left_bound,
                new_right_bound=new_right_bound,
            )
        )

    def _validate_color_and_opacity(self, color: Optional[str], opacity: Optional[float]) -> bool:
        """Validates both color and opacity values"""
        return bool(StyleUtils.validate_color_and_opacity(color, opacity))

    def _is_valid_css_color(self, color: str) -> bool:
        """Validates if a string is a valid CSS color."""
        return bool(StyleUtils.is_valid_css_color(color))

    def _validate_opacity(self, opacity: float) -> bool:
        """Validates if an opacity value is between 0 and 1"""
        return bool(StyleUtils.validate_opacity(opacity))

    @property
    def name_generator(self) -> Any:  # NameGenerator
        """Property to access the name generator from the drawable manager"""
        return self.drawable_manager.name_generator

    # ------------------- Angle Methods -------------------

    def create_angle(self, vx: float, vy: float, p1x: float, p1y: float, p2x: float, p2y: float, color: Optional[str] = None, angle_name: Optional[str] = None, is_reflex: bool = False, extra_graphics: bool = True) -> Optional["Drawable"]:
        """Create an angle defined by three points via AngleManager."""
        if self.drawable_manager.angle_manager:
            return self.drawable_manager.angle_manager.create_angle(
                vx, vy, p1x, p1y, p2x, p2y, 
                color=color, 
                angle_name=angle_name, 
                is_reflex=is_reflex,
                extra_graphics=extra_graphics
            )
        return None

    def delete_angle(self, name: str) -> bool:
        """Remove an angle by its name via AngleManager."""
        if self.drawable_manager.angle_manager:
            return bool(self.drawable_manager.angle_manager.delete_angle(name))
        return False

    def update_angle(self, name: str, new_color: Optional[str] = None) -> bool:
        """Update editable angle properties via AngleManager."""
        if self.drawable_manager.angle_manager:
            return bool(self.drawable_manager.angle_manager.update_angle(
                name, new_color=new_color
            ))
        return False

    # ------------------- Circle Arc Methods -------------------

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
    ) -> Optional["Drawable"]:
        """Create a circle arc drawable via ArcManager."""
        if self.drawable_manager.arc_manager:
            return self.drawable_manager.arc_manager.create_circle_arc(
                point1_x=point1_x,
                point1_y=point1_y,
                point2_x=point2_x,
                point2_y=point2_y,
                point1_name=point1_name,
                point2_name=point2_name,
                point3_x=point3_x,
                point3_y=point3_y,
                point3_name=point3_name,
                center_point_choice=center_point_choice,
                circle_name=circle_name,
                center_x=center_x,
                center_y=center_y,
                radius=radius,
                arc_name=arc_name,
                color=color,
                use_major_arc=use_major_arc,
                extra_graphics=extra_graphics,
            )
        return None

    def delete_circle_arc(self, name: str) -> bool:
        """Delete a circle arc via ArcManager."""
        if self.drawable_manager.arc_manager:
            return bool(self.drawable_manager.arc_manager.delete_circle_arc(name))
        return False

    def update_circle_arc(
        self,
        name: str,
        *,
        new_color: Optional[str] = None,
        use_major_arc: Optional[bool] = None,
        point1_name: Optional[str] = None,
        point1_x: Optional[float] = None,
        point1_y: Optional[float] = None,
        point2_name: Optional[str] = None,
        point2_x: Optional[float] = None,
        point2_y: Optional[float] = None,
    ) -> bool:
        """Update editable circle arc properties."""
        if self.drawable_manager.arc_manager:
            return bool(
                self.drawable_manager.arc_manager.update_circle_arc(
                    name,
                    new_color=new_color,
                    use_major_arc=use_major_arc,
                    point1_name=point1_name,
                    point1_x=point1_x,
                    point1_y=point1_y,
                    point2_name=point2_name,
                    point2_x=point2_x,
                    point2_y=point2_y,
                )
            )
        return False

    # Property delegations to CoordinateMapper for backward compatibility
    @property
    def center(self) -> Point:
        """Current viewport center point - delegates to coordinate_mapper.origin"""
        return self.coordinate_mapper.origin
    
    @center.setter
    def center(self, value: Point) -> None:
        """Set viewport center point - delegates to coordinate_mapper.origin"""
        self.coordinate_mapper.origin = value
    
    @property
    def scale_factor(self) -> float:
        """Current zoom level - delegates to coordinate_mapper.scale_factor"""
        return float(self.coordinate_mapper.scale_factor)
    
    @scale_factor.setter
    def scale_factor(self, value: float) -> None:
        """Set zoom level - delegates to coordinate_mapper.scale_factor"""
        self.coordinate_mapper.scale_factor = value
    
    @property
    def offset(self) -> Point:
        """Current pan offset - delegates to coordinate_mapper.offset"""
        return self.coordinate_mapper.offset
    
    @offset.setter
    def offset(self, value: Point) -> None:
        """Set pan offset - delegates to coordinate_mapper.offset"""
        self.coordinate_mapper.offset = value
    
    @property
    def zoom_point(self) -> Point:
        """Current zoom center point - delegates to coordinate_mapper.zoom_point"""
        return self.coordinate_mapper.zoom_point
    
    @zoom_point.setter
    def zoom_point(self, value: Point) -> None:
        """Set zoom center point - delegates to coordinate_mapper.zoom_point"""
        self.coordinate_mapper.zoom_point = value
    
    @property
    def zoom_direction(self) -> int:
        """Current zoom direction - delegates to coordinate_mapper.zoom_direction"""
        return int(self.coordinate_mapper.zoom_direction)
    
    @zoom_direction.setter
    def zoom_direction(self, value: int) -> None:
        """Set zoom direction - delegates to coordinate_mapper.zoom_direction"""
        self.coordinate_mapper.zoom_direction = value
    
    @property
    def zoom_step(self) -> float:
        """Zoom step size - delegates to coordinate_mapper.zoom_step"""
        return float(self.coordinate_mapper.zoom_step)
    
    @zoom_step.setter
    def zoom_step(self, value: float) -> None:
        """Set zoom step size - delegates to coordinate_mapper.zoom_step"""
        self.coordinate_mapper.zoom_step = value

