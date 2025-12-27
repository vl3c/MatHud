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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union, cast

from constants import (
    default_area_fill_color,
    default_area_opacity,
    default_closed_shape_resolution,
)
from drawables_aggregator import Point
from cartesian_system_2axis import Cartesian2Axis
from coordinate_mapper import CoordinateMapper
from utils.math_utils import MathUtils
from utils.style_utils import StyleUtils
from utils.geometry_utils import GeometryUtils
from utils.graph_analyzer import GraphAnalyzer
from utils.computation_utils import ComputationUtils
from geometry.graph_state import (
    GraphEdgeDescriptor,
    GraphState,
    GraphVertexDescriptor,
    TreeState,
)
from managers.undo_redo_manager import UndoRedoManager
from managers.drawable_manager import DrawableManager
from managers.drawable_dependency_manager import DrawableDependencyManager
from managers.transformations_manager import TransformationsManager
from managers.polygon_type import PolygonType
from constants import DEFAULT_RENDERER_MODE
from rendering.factory import create_renderer
from rendering.interfaces import RendererProtocol

if TYPE_CHECKING:
    from drawables.drawable import Drawable
    from geometry.graph_state import GraphState


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
        self.renderer_mode: str = self._resolve_renderer_mode(self.renderer)
        
        if self.draw_enabled and self.renderer is not None:
            try:
                self.renderer.render_cartesian(self.cartesian2axis, self.coordinate_mapper)
            except Exception:
                pass
        self._register_renderer_handlers()

    def add_drawable(self, drawable: "Drawable") -> None:
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
            for drawable in self.drawable_manager.get_renderable_drawables():
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

    def create_polygon(
        self,
        vertices: Sequence[Any],
        *,
        polygon_type: Optional[Union[str, PolygonType]] = None,
        name: str = "",
        color: Optional[str] = None,
        subtype: Optional[str] = None,
        extra_graphics: bool = True,
    ) -> "Drawable":
        """Create a polygon with ordered vertices."""
        return self.drawable_manager.create_polygon(
            vertices,
            polygon_type=polygon_type,
            name=name,
            color=color,
            subtype=subtype,
            extra_graphics=extra_graphics,
        )

    def update_polygon(
        self,
        polygon_name: str,
        *,
        polygon_type: Optional[Union[str, PolygonType]] = None,
        new_color: Optional[str] = None,
    ) -> bool:
        """Update editable properties of an existing polygon."""
        return bool(
            self.drawable_manager.update_polygon(
                polygon_name,
                polygon_type=polygon_type,
                new_color=new_color,
            )
        )

    def delete_polygon(
        self,
        *,
        polygon_type: Optional[Union[str, PolygonType]] = None,
        name: Optional[str] = None,
        vertices: Optional[Sequence[Any]] = None,
    ) -> bool:
        """Delete a polygon by name or by vertex coordinates."""
        return bool(
            self.drawable_manager.delete_polygon(
                polygon_type=polygon_type,
                name=name,
                vertices=vertices,
            )
        )

    def get_polygon_by_name(
        self,
        polygon_name: str,
        polygon_type: Optional[Union[str, PolygonType]] = None,
    ) -> Optional["Drawable"]:
        """Retrieve a polygon by name."""
        return cast(
            Optional["Drawable"],
            self.drawable_manager.get_polygon_by_name(polygon_name, polygon_type),
        )

    def get_polygon_by_vertices(
        self,
        vertices: Sequence[Any],
        polygon_type: Optional[Union[str, PolygonType]] = None,
    ) -> Optional["Drawable"]:
        """Retrieve a polygon by its vertex coordinates."""
        return cast(
            Optional["Drawable"],
            self.drawable_manager.get_polygon_by_vertices(vertices, polygon_type),
        )

    def get_cartesian2axis_state(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.cartesian2axis.get_state())

    def get_canvas_state(self) -> Dict[str, Any]:
        state = self.get_drawables_state()
        self._prune_plot_derived_bars_from_state(state)
        cartesian_state = self.get_cartesian2axis_state()
        if cartesian_state is not None:
            state.update(cartesian_state)
        if self.computations:  # Add computations to state if they exist
            state["computations"] = self.computations
        return state

    def _prune_plot_derived_bars_from_state(self, state: Dict[str, Any]) -> None:
        """
        Remove derived Bar drawables that can be rebuilt from plot composites.

        Plot tools like plot_distribution (discrete) and plot_bars create many Bar
        drawables for rendering, but those are derived from DiscretePlot/BarsPlot
        parameters and make serialized canvas state extremely verbose.
        """
        try:
            if not isinstance(state, dict):
                return

            bars_state = state.get("Bars")
            if not isinstance(bars_state, list) or not bars_state:
                return

            prefixes: List[str] = []

            discrete_plots = state.get("DiscretePlots")
            if isinstance(discrete_plots, list):
                for item in discrete_plots:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    if isinstance(name, str) and name:
                        prefixes.append(f"{name}_bar_")

            bars_plots = state.get("BarsPlots")
            if isinstance(bars_plots, list):
                for item in bars_plots:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    if isinstance(name, str) and name:
                        prefixes.append(f"{name}_bar_")

            if not prefixes:
                return

            kept: List[Any] = []
            for item in bars_state:
                if not isinstance(item, dict):
                    kept.append(item)
                    continue
                bar_name = item.get("name")
                if not isinstance(bar_name, str):
                    kept.append(item)
                    continue
                if any(bar_name.startswith(prefix) for prefix in prefixes):
                    continue
                kept.append(item)

            if kept:
                state["Bars"] = kept
            else:
                state.pop("Bars", None)
        except Exception:
            # Best-effort pruning only; never break callers.
            return

    def get_point(self, x: float, y: float) -> Optional[Point]:
        """Get a point at the specified coordinates"""
        return self.drawable_manager.get_point(x, y)

    def get_point_by_name(self, name: str) -> Optional[Point]:
        """Get a point by its name"""
        return self.drawable_manager.get_point_by_name(name)

    def get_label_by_name(self, name: str) -> Optional["Drawable"]:
        """Get a label by its name."""
        return cast(Optional["Drawable"], self.drawable_manager.get_label_by_name(name))

    def create_point(
        self,
        x: float,
        y: float,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
    ) -> Point:
        """Create a point at the specified coordinates"""
        return self.drawable_manager.create_point(
            x,
            y,
            name,
            color=color,
            extra_graphics=extra_graphics,
        )

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

    def create_segment(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
        label_text: Optional[str] = None,
        label_visible: Optional[bool] = None,
    ) -> "Drawable":
        """Create a segment between two points"""
        return self.drawable_manager.create_segment(
            x1,
            y1,
            x2,
            y2,
            name,
            color=color,
            extra_graphics=extra_graphics,
            label_text=label_text,
            label_visible=label_visible,
        )

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
        new_label_text: Optional[str] = None,
        new_label_visible: Optional[bool] = None,
    ) -> bool:
        """Update editable properties of a segment."""
        return bool(
            self.drawable_manager.update_segment(
                name,
                new_color=new_color,
                new_label_text=new_label_text,
                new_label_visible=new_label_visible,
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

    def create_vector(
        self,
        origin_x: float,
        origin_y: float,
        tip_x: float,
        tip_y: float,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
    ) -> "Drawable":
        """Create a vector from origin to tip"""
        return self.drawable_manager.create_vector(
            origin_x,
            origin_y,
            tip_x,
            tip_y,
            name,
            color=color,
            extra_graphics=extra_graphics,
        )

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

    # ------------------- Plot Methods -------------------
    def plot_distribution(
        self,
        *,
        name: Optional[str] = None,
        representation: str = "continuous",
        distribution_type: str = "normal",
        distribution_params: Optional[Dict[str, Any]] = None,
        plot_bounds: Optional[Dict[str, Any]] = None,
        shade_bounds: Optional[Dict[str, Any]] = None,
        curve_color: Optional[str] = None,
        fill_color: Optional[str] = None,
        fill_opacity: Optional[float] = None,
        bar_count: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self.drawable_manager.plot_distribution(
            name=name,
            representation=representation,
            distribution_type=distribution_type,
            distribution_params=distribution_params,
            plot_bounds=plot_bounds,
            shade_bounds=shade_bounds,
            curve_color=curve_color,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            bar_count=bar_count,
        )

    def plot_bars(
        self,
        *,
        name: Optional[str] = None,
        values: Optional[List[float]] = None,
        labels_below: Optional[List[str]] = None,
        labels_above: Optional[List[str]] = None,
        bar_spacing: Optional[float] = None,
        bar_width: Optional[float] = None,
        stroke_color: Optional[str] = None,
        fill_color: Optional[str] = None,
        fill_opacity: Optional[float] = None,
        x_start: Optional[float] = None,
        y_base: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self.drawable_manager.plot_bars(
            name=name,
            values=values or [],
            labels_below=labels_below or [],
            labels_above=labels_above,
            bar_spacing=bar_spacing,
            bar_width=bar_width,
            stroke_color=stroke_color,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            x_start=x_start,
            y_base=y_base,
        )

    def delete_plot(self, name: str) -> bool:
        return bool(self.drawable_manager.delete_plot(name))

    # ------------------- Graph Methods -------------------
    def create_graph(self, graph_state: "GraphState") -> "Drawable":
        return self.drawable_manager.create_graph(graph_state)

    def delete_graph(self, name: str) -> bool:
        return bool(self.drawable_manager.delete_graph(name))

    def get_graph(self, name: str) -> Optional["Drawable"]:
        return self.drawable_manager.get_graph(name)

    def capture_graph_state(self, name: str):
        return self.drawable_manager.capture_graph_state(name)

    def generate_graph(
        self,
        *,
        name: str = "Graph",
        graph_type: str = "graph",
        vertices: Optional[List[Dict[str, Any]]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        adjacency_matrix: Optional[List[List[float]]] = None,
        directed: Optional[bool] = None,
        root: Optional[str] = None,
        layout: Optional[str] = None,
        placement_box: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = self.drawable_manager.build_graph_state(
            name=name,
            graph_type=graph_type,
            vertices=vertices or [],
            edges=edges or [],
            adjacency_matrix=adjacency_matrix,
            directed=directed,
            root=root,
            layout=layout,
            placement_box=placement_box,
            metadata=metadata,
        )
        graph = self.drawable_manager.create_graph(state)
        return graph.get_state()

    def analyze_graph(
        self,
        *,
        graph_name: str,
        operation: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = self.drawable_manager.capture_graph_state(graph_name)
        if state is None:
            return {"error": "Graph not found or spec missing"}
        return GraphAnalyzer.analyze(state, operation, params)

    # ------------------- Circle Methods -------------------
    def get_circle(self, center_x: float, center_y: float, radius: float) -> Optional["Drawable"]:
        """Get a circle by its center coordinates and radius"""
        return self.drawable_manager.get_circle(center_x, center_y, radius)

    def get_circle_by_name(self, name: str) -> Optional["Drawable"]:
        """Get a circle by its name"""
        return self.drawable_manager.get_circle_by_name(name)

    def create_circle(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
    ) -> "Drawable":
        """Create a circle with the specified center and radius"""
        return self.drawable_manager.create_circle(
            center_x,
            center_y,
            radius,
            name,
            color=color,
            extra_graphics=extra_graphics,
        )

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

    def create_ellipse(
        self,
        center_x: float,
        center_y: float,
        radius_x: float,
        radius_y: float,
        rotation_angle: float = 0,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
    ) -> "Drawable":
        """Create an ellipse with the specified center, radii, and rotation"""
        return self.drawable_manager.create_ellipse(
            center_x,
            center_y,
            radius_x,
            radius_y,
            rotation_angle,
            name,
            color=color,
            extra_graphics=extra_graphics,
        )

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

    def draw_function(
        self,
        function_string: str,
        name: str,
        left_bound: Optional[float] = None,
        right_bound: Optional[float] = None,
        color: Optional[str] = None,
        undefined_at: Optional[List[float]] = None,
    ) -> "Drawable":
        """Draw a function on the canvas"""
        return self.drawable_manager.draw_function(
            function_string,
            name,
            left_bound,
            right_bound,
            color=color,
            undefined_at=undefined_at,
        )

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

    def draw_piecewise_function(
        self,
        pieces: List[Dict[str, Any]],
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> "Drawable":
        """Draw a piecewise function on the canvas."""
        return self.drawable_manager.draw_piecewise_function(
            pieces,
            name,
            color=color,
        )

    def delete_piecewise_function(self, name: str) -> bool:
        """Delete a piecewise function by its name."""
        return bool(self.drawable_manager.delete_piecewise_function(name))

    def update_piecewise_function(
        self,
        name: str,
        new_color: Optional[str] = None,
    ) -> bool:
        """Update editable properties of a piecewise function."""
        return bool(
            self.drawable_manager.update_piecewise_function(
                name,
                new_color=new_color,
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

    def zoom(self, center_x: float, center_y: float, range_val: float, range_axis: str) -> bool:
        """Center viewport on (center_x, center_y) with specified range on one axis.
        
        The range applies to the axis specified by range_axis. The other axis
        scales according to the canvas aspect ratio.
        
        Args:
            center_x: X coordinate to center the viewport on
            center_y: Y coordinate to center the viewport on
            range_val: Half-size for the specified axis
            range_axis: 'x' or 'y' - which axis the range applies to
        """
        if range_axis == "x":
            left = center_x - range_val
            right = center_x + range_val
            aspect = self.height / self.width
            y_range = range_val * aspect
            top = center_y + y_range
            bottom = center_y - y_range
        else:
            top = center_y + range_val
            bottom = center_y - range_val
            aspect = self.width / self.height
            x_range = range_val * aspect
            left = center_x - x_range
            right = center_x + x_range
        self.coordinate_mapper.set_visible_bounds(left, right, top, bottom)
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

    def create_colored_area(self, drawable1_name: str, drawable2_name: Optional[str] = None, left_bound: Optional[float] = None, right_bound: Optional[float] = None, color: str = default_area_fill_color, opacity: float = default_area_opacity) -> "Drawable":
        """Creates a vertical bounded colored area between two functions, two segments, or a function and a segment"""
        return self.drawable_manager.create_colored_area(drawable1_name, drawable2_name, left_bound, right_bound, color, opacity)

    def create_region_colored_area(
        self,
        *,
        expression: Optional[str] = None,
        triangle_name: Optional[str] = None,
        rectangle_name: Optional[str] = None,
        polygon_segment_names: Optional[List[str]] = None,
        circle_name: Optional[str] = None,
        ellipse_name: Optional[str] = None,
        chord_segment_name: Optional[str] = None,
        arc_clockwise: bool = False,
        resolution: int = default_closed_shape_resolution,
        color: str = default_area_fill_color,
        opacity: float = default_area_opacity,
    ) -> "Drawable":
        """Creates a region colored area from expression or closed shape."""
        return self.drawable_manager.create_region_colored_area(
            expression=expression,
            triangle_name=triangle_name,
            rectangle_name=rectangle_name,
            polygon_segment_names=polygon_segment_names,
            circle_name=circle_name,
            ellipse_name=ellipse_name,
            chord_segment_name=chord_segment_name,
            arc_clockwise=arc_clockwise,
            resolution=resolution,
            color=color,
            opacity=opacity,
        )
        
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

    def get_renderer_mode(self) -> str:
        """Return a lowercase string describing the active renderer backend."""
        return self.renderer_mode

    def _resolve_renderer_mode(self, renderer: Optional[RendererProtocol]) -> str:
        if renderer is None:
            return "none"
        name = renderer.__class__.__name__.lower()
        if "canvas2d" in name:
            return "canvas2d"
        if "svg" in name:
            return "svg"
        if "webgl" in name:
            return "webgl"
        module = getattr(renderer, "__module__", "")
        module_lower = module.lower()
        if "canvas2d" in module_lower:
            return "canvas2d"
        if "svg" in module_lower:
            return "svg"
        if "webgl" in module_lower:
            return "webgl"
        return "unknown"
