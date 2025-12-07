"""
MatHud Drawable Management System

Central orchestration hub for all drawable objects in the mathematical visualization system.
Coordinates specialized managers for each drawable type and handles object lifecycle management.

Core Architecture:
    - Proxy Pattern: Uses DrawableManagerProxy to resolve circular dependencies during initialization
    - Specialized Managers: Delegates type-specific operations to dedicated manager classes
    - Dependency Tracking: Maintains hierarchical relationships between geometric objects
    - Container Management: Organizes drawables by type with efficient access patterns
    - Name Generation: Automatic naming system for drawable objects

Manager Hierarchy:
    - PointManager: Point creation, retrieval, and deletion operations
    - SegmentManager: Line segment operations with endpoint validation
    - VectorManager: Vector operations with origin and tip management
    - CircleManager: Circle operations with center and radius validation
    - EllipseManager: Ellipse operations with center, radii, and rotation
    - AngleManager: Angle operations with vertex and arm management
    - FunctionManager: Mathematical function plotting and analysis
    - ColoredAreaManager: Bounded area creation with function/segment boundaries

Dependency System:
    - DrawableDependencyManager: Tracks parent-child relationships between objects
    - Propagates changes through dependency chains (e.g., moving a point updates dependent segments)
    - Handles cascading deletions while preserving object integrity

Storage Organization:
    - DrawablesContainer: Type-based storage with property access patterns
    - Layered rendering support (colored areas behind geometric objects)
    - State serialization for undo/redo functionality

Integration Points:
    - Canvas drawing system for visual representation
    - UndoRedoManager for state persistence
    - TransformationsManager for geometric transformations
    - Mathematical computation engine for geometric analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union, cast

from constants import (
    default_area_fill_color,
    default_area_opacity,
    default_closed_shape_resolution,
)
from drawables_aggregator import Point
from managers.point_manager import PointManager
from managers.segment_manager import SegmentManager
from managers.vector_manager import VectorManager
from managers.function_manager import FunctionManager
from managers.piecewise_function_manager import PiecewiseFunctionManager
from managers.circle_manager import CircleManager
from managers.ellipse_manager import EllipseManager
from managers.polygon_manager import PolygonManager
from managers.polygon_type import PolygonType
from managers.colored_area_manager import ColoredAreaManager
from managers.drawable_manager_proxy import DrawableManagerProxy
from name_generator.drawable import DrawableNameGenerator
from managers.drawable_dependency_manager import DrawableDependencyManager
from managers.drawables_container import DrawablesContainer
from managers.angle_manager import AngleManager
from managers.label_manager import LabelManager
from managers.arc_manager import ArcManager
from managers.graph_manager import GraphManager
from drawables.closed_shape_colored_area import ClosedShapeColoredArea

if TYPE_CHECKING:
    from canvas import Canvas
    from drawables.drawable import Drawable
    from drawables.point import Point
    from drawables.segment import Segment
    from drawables.vector import Vector
    from drawables.function import Function
    from drawables.circle import Circle
    from drawables.ellipse import Ellipse
    from drawables.rectangle import Rectangle
    from drawables.angle import Angle
    from drawables.colored_area import ColoredArea
    from geometry.graph_state import GraphState

class DrawableManager:
    """
    Manages drawable objects for a Canvas.
    
    This class coordinates specialized managers for each drawable type:
    - Points
    - Segments
    - Vectors
    - Functions
    - Circles
    - Ellipses
    - Rectangles
    - Colored Areas
    """
    
    def __init__(self, canvas: "Canvas") -> None:
        """
        Initialize the DrawableManager.
        
        Args:
            canvas: The Canvas object this manager is responsible for
        """
        self.canvas: "Canvas" = canvas
        self.name_generator: DrawableNameGenerator = DrawableNameGenerator(canvas)
        self.drawables: DrawablesContainer = DrawablesContainer()
        
        # Create a proxy BEFORE dependency manager
        self.proxy: DrawableManagerProxy = DrawableManagerProxy(self)
        
        # Instantiate DependencyManager with just the proxy
        self.dependency_manager: DrawableDependencyManager = DrawableDependencyManager(drawable_manager_proxy=self.proxy)
        
        # Initialize specialized managers with the proxy
        self.point_manager: PointManager = PointManager(
            canvas, self.drawables, self.name_generator, self.dependency_manager, self.proxy
        )
        
        self.segment_manager: SegmentManager = SegmentManager(
            canvas, self.drawables, self.name_generator, self.dependency_manager, 
            self.point_manager, self.proxy
        )
        
        self.vector_manager: VectorManager = VectorManager(
            canvas, self.drawables, self.name_generator, self.dependency_manager, 
            self.point_manager, self.proxy
        )

        self.polygon_manager: PolygonManager = PolygonManager(
            canvas,
            self.drawables,
            self.name_generator,
            self.dependency_manager,
            self.point_manager,
            self.segment_manager,
            self.proxy,
        )
        
        self.function_manager: FunctionManager = FunctionManager(
            canvas, self.drawables, self.name_generator, self.dependency_manager, self.proxy
        )
        
        self.piecewise_function_manager: PiecewiseFunctionManager = PiecewiseFunctionManager(
            canvas, self.drawables, self.name_generator, self.dependency_manager, self.proxy
        )
        
        self.circle_manager: CircleManager = CircleManager(
            canvas, self.drawables, self.name_generator, self.dependency_manager, 
            self.point_manager, self.proxy
        )
        
        self.ellipse_manager: EllipseManager = EllipseManager(
            canvas, self.drawables, self.name_generator, self.dependency_manager, 
            self.point_manager, self.proxy
        )
        
        self.colored_area_manager: ColoredAreaManager = ColoredAreaManager(
            canvas, self.drawables, self.name_generator, self.dependency_manager, self.proxy
        )
        
        self.angle_manager: AngleManager = AngleManager(
            canvas, self.drawables, self.name_generator, self.dependency_manager,
            self.point_manager, self.segment_manager, self.proxy
        )

        self.label_manager: LabelManager = LabelManager(
            canvas, self.drawables, self.name_generator, self.dependency_manager, self.proxy
        )

        self.arc_manager: ArcManager = ArcManager(
            canvas,
            self.drawables,
            self.name_generator,
            self.dependency_manager,
            self.point_manager,
            self.proxy,
        )

        self.graph_manager: GraphManager = GraphManager(
            canvas,
            self.drawables,
            self.name_generator,
            self.dependency_manager,
            self.point_manager,
            self.segment_manager,
            self.vector_manager,
            self.label_manager,
            self.proxy,
        )
        
        # No need for the loop that sets drawable_manager anymore
        # The proxy handles forwarding calls to the appropriate managers
        
    # ------------------- General Drawable Methods -------------------
    
    def get_drawables(self) -> List["Drawable"]:
        """Get all drawables as a flat list, with colored areas first (behind other elements)"""
        return cast(List["Drawable"], self.drawables.get_all_with_layering())
    
    def get_renderable_drawables(self) -> List["Drawable"]:
        """Get only the drawables that should be rendered, preserving layering."""
        return cast(List["Drawable"], self.drawables.get_renderables_with_layering())
    
    # ------------------- Point Methods -------------------

    # ------------------- Polygon Methods -------------------

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
        """Create a polygon specified by ordered vertices."""
        return self.polygon_manager.create_polygon(
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
            self.polygon_manager.update_polygon(
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
        """Delete a polygon either by name or by its vertices."""
        return bool(
            self.polygon_manager.delete_polygon(
                polygon_type=polygon_type,
                name=name,
                vertices=vertices,
            )
        )

    def create_new_triangles_from_connected_segments(self) -> None:
        """Detect and create triangles from connected segments using the polygon pipeline."""
        self.polygon_manager.create_triangles_from_segments()

    def get_polygon_by_name(
        self,
        polygon_name: str,
        polygon_type: Optional[Union[str, PolygonType]] = None,
    ) -> Optional["Drawable"]:
        """Retrieve a polygon by name."""
        return cast(
            Optional["Drawable"],
            self.polygon_manager.get_polygon_by_name(polygon_name, polygon_type),
        )

    def get_polygon_by_vertices(
        self,
        vertices: Sequence[Any],
        polygon_type: Optional[Union[str, PolygonType]] = None,
    ) -> Optional["Drawable"]:
        """Retrieve a polygon by its vertex coordinates."""
        normalized_vertices = list(vertices)
        return cast(
            Optional["Drawable"],
            self.polygon_manager.get_polygon_by_vertices(normalized_vertices, polygon_type),
        )
    
    def get_point(self, x: float, y: float) -> Optional["Point"]:
        """Get a point at the specified coordinates"""
        return self.point_manager.get_point(x, y)
        
    def get_point_by_name(self, name: str) -> Optional["Point"]:
        """Get a point by its name"""
        return self.point_manager.get_point_by_name(name)

    def get_label_by_name(self, name: str) -> Optional["Drawable"]:
        """Get a label by its name."""
        return self.label_manager.get_label_by_name(name)

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
        """Create a label with the specified properties."""
        return self.label_manager.create_label(
            x,
            y,
            text,
            name=name,
            color=color,
            font_size=font_size,
            rotation_degrees=rotation_degrees,
        )

    def delete_label(self, name: str) -> bool:
        """Delete a label by its name."""
        return bool(self.label_manager.delete_label(name))
    
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
            self.label_manager.update_label(
                name,
                new_text=new_text,
                new_x=new_x,
                new_y=new_y,
                new_color=new_color,
                new_font_size=new_font_size,
                new_rotation_degrees=new_rotation_degrees,
            )
        )
        
    def create_point(
        self,
        x: float,
        y: float,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
    ) -> "Point":
        """Create a new point at the specified coordinates"""
        return self.point_manager.create_point(
            x,
            y,
            name,
            color=color,
            extra_graphics=extra_graphics,
        )
        
    def delete_point(self, x: float, y: float) -> bool:
        """Delete a point at the specified coordinates"""
        return bool(self.point_manager.delete_point(x, y))
        
    def delete_point_by_name(self, name: str) -> bool:
        """Delete a point by its name"""
        return bool(self.point_manager.delete_point_by_name(name))
    
    # ------------------- Segment Methods -------------------
    
    def get_segment_by_coordinates(self, x1: float, y1: float, x2: float, y2: float) -> Optional["Segment"]:
        """Get a segment by its endpoint coordinates"""
        return self.segment_manager.get_segment_by_coordinates(x1, y1, x2, y2)
        
    def get_segment_by_name(self, name: str) -> Optional["Segment"]:
        """Get a segment by its name"""
        return self.segment_manager.get_segment_by_name(name)
        
    def get_segment_by_points(self, p1: Point, p2: Point) -> Optional["Segment"]:
        """Get a segment by its endpoint points"""
        return self.segment_manager.get_segment_by_points(p1, p2)
        
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
    ) -> "Segment":
        """Create a new segment between the specified points"""
        return self.segment_manager.create_segment(
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
        """Delete a segment at the specified coordinates"""
        return bool(self.segment_manager.delete_segment(x1, y1, x2, y2, delete_children, delete_parents))
        
    def delete_segment_by_name(self, name: str, delete_children: bool = True, delete_parents: bool = False) -> bool:
        """Delete a segment by its name"""
        return bool(self.segment_manager.delete_segment_by_name(name, delete_children, delete_parents))

    def update_segment(
        self,
        name: str,
        new_color: Optional[str] = None,
        new_label_text: Optional[str] = None,
        new_label_visible: Optional[bool] = None,
    ) -> bool:
        """Update editable properties of an existing segment."""
        return bool(
            self.segment_manager.update_segment(
                name,
                new_color=new_color,
                new_label_text=new_label_text,
                new_label_visible=new_label_visible,
            )
        )
    
    # ------------------- Vector Methods -------------------
    
    def get_vector(self, origin_x: float, origin_y: float, tip_x: float, tip_y: float) -> Optional["Vector"]:
        """Get a vector by its origin and tip coordinates"""
        return self.vector_manager.get_vector(origin_x, origin_y, tip_x, tip_y)
        
    def create_vector(
        self,
        origin_x: float,
        origin_y: float,
        tip_x: float,
        tip_y: float,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
    ) -> "Vector":
        """Create a new vector with the specified origin and tip"""
        return self.vector_manager.create_vector(
            origin_x,
            origin_y,
            tip_x,
            tip_y,
            name,
            color=color,
            extra_graphics=extra_graphics,
        )
        
    def delete_vector(self, origin_x: float, origin_y: float, tip_x: float, tip_y: float) -> bool:
        """Delete a vector with the specified origin and tip"""
        return bool(self.vector_manager.delete_vector(origin_x, origin_y, tip_x, tip_y))
    
    def update_vector(
        self,
        name: str,
        new_color: Optional[str] = None,
    ) -> bool:
        """Update editable properties of a vector."""
        return bool(
            self.vector_manager.update_vector(
                name,
                new_color=new_color,
            )
        )
    
    # ------------------- Function Methods -------------------
    
    def get_function(self, name: str) -> Optional["Function"]:
        """Get a function by its name"""
        return self.function_manager.get_function(name)
        
    def draw_function(
        self,
        function_string: str,
        name: str,
        left_bound: Optional[float] = None,
        right_bound: Optional[float] = None,
        color: Optional[str] = None,
        undefined_at: Optional[List[float]] = None,
    ) -> "Function":
        """Create a new function with the specified expression"""
        return self.function_manager.draw_function(
            function_string,
            name,
            left_bound,
            right_bound,
            color=color,
            undefined_at=undefined_at,
        )
        
    def delete_function(self, name: str) -> bool:
        """Delete a function by its name"""
        return bool(self.function_manager.delete_function(name))
    
    def update_function(
        self,
        name: str,
        new_color: Optional[str] = None,
        new_left_bound: Optional[float] = None,
        new_right_bound: Optional[float] = None,
    ) -> bool:
        """Update editable properties of a function."""
        return bool(
            self.function_manager.update_function(
                name,
                new_color=new_color,
                new_left_bound=new_left_bound,
                new_right_bound=new_right_bound,
            )
        )

    # ------------------- Piecewise Function Methods -------------------

    def get_piecewise_function(self, name: str) -> Optional["Drawable"]:
        """Get a piecewise function by its name."""
        return self.piecewise_function_manager.get_piecewise_function(name)

    def draw_piecewise_function(
        self,
        pieces: List[Any],
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> "Drawable":
        """Create a new piecewise function with the specified pieces."""
        return self.piecewise_function_manager.draw_piecewise_function(
            pieces,
            name,
            color=color,
        )

    def delete_piecewise_function(self, name: str) -> bool:
        """Delete a piecewise function by its name."""
        return bool(self.piecewise_function_manager.delete_piecewise_function(name))

    def update_piecewise_function(
        self,
        name: str,
        new_color: Optional[str] = None,
    ) -> bool:
        """Update editable properties of a piecewise function."""
        return bool(
            self.piecewise_function_manager.update_piecewise_function(
                name,
                new_color=new_color,
            )
        )
    
    # ------------------- Circle Methods -------------------
    
    def get_circle(self, center_x: float, center_y: float, radius: float) -> Optional["Circle"]:
        """Get a circle by its center coordinates and radius"""
        return self.circle_manager.get_circle(center_x, center_y, radius)
        
    def get_circle_by_name(self, name: str) -> Optional["Circle"]:
        """Get a circle by its name"""
        return self.circle_manager.get_circle_by_name(name)
        
    def create_circle(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        name: str = "",
        color: Optional[str] = None,
        extra_graphics: bool = True,
    ) -> "Circle":
        """Create a new circle with the specified center and radius"""
        return self.circle_manager.create_circle(
            center_x,
            center_y,
            radius,
            name,
            color=color,
            extra_graphics=extra_graphics,
        )
        
    def delete_circle(self, name: str) -> bool:
        """Delete a circle by its name"""
        return bool(self.circle_manager.delete_circle(name))
    
    def update_circle(
        self,
        name: str,
        new_color: Optional[str] = None,
        new_center_x: Optional[float] = None,
        new_center_y: Optional[float] = None,
    ) -> bool:
        """Update editable properties of a circle."""
        return bool(
            self.circle_manager.update_circle(
                name,
                new_color=new_color,
                new_center_x=new_center_x,
                new_center_y=new_center_y,
            )
        )
    
    # ------------------- Ellipse Methods -------------------
    
    def get_ellipse(self, center_x: float, center_y: float, radius_x: float, radius_y: float) -> Optional["Ellipse"]:
        """Get an ellipse by its center coordinates and radii"""
        return self.ellipse_manager.get_ellipse(center_x, center_y, radius_x, radius_y)
        
    def get_ellipse_by_name(self, name: str) -> Optional["Ellipse"]:
        """Get an ellipse by its name"""
        return self.ellipse_manager.get_ellipse_by_name(name)
        
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
    ) -> "Ellipse":
        """Create a new ellipse with the specified center, radii, and rotation"""
        return self.ellipse_manager.create_ellipse(
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
        return bool(self.ellipse_manager.delete_ellipse(name))
    
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
            self.ellipse_manager.update_ellipse(
                name,
                new_color=new_color,
                new_radius_x=new_radius_x,
                new_radius_y=new_radius_y,
                new_rotation_angle=new_rotation_angle,
                new_center_x=new_center_x,
                new_center_y=new_center_y,
            )
        )
    
    # ------------------- Colored Area Methods -------------------
    
    def create_colored_area(
        self,
        drawable1_name: str,
        drawable2_name: Optional[str] = None,
        left_bound: Optional[float] = None,
        right_bound: Optional[float] = None,
        color: str = default_area_fill_color,
        opacity: float = default_area_opacity,
    ) -> "ColoredArea":
        """Create a new colored area between drawables"""
        return self.colored_area_manager.create_colored_area(
            drawable1_name,
            drawable2_name,
            left_bound,
            right_bound,
            color,
            opacity,
        )

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
    ) -> "ClosedShapeColoredArea":
        """Create a region colored area from expression or closed shape."""
        return self.colored_area_manager.create_region_colored_area(
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
        """Delete a colored area by its name"""
        return bool(self.colored_area_manager.delete_colored_area(name))
        
    def delete_colored_areas_for_function(self, func: "Drawable") -> None:
        """Delete all colored areas associated with a function"""
        self.colored_area_manager.delete_colored_areas_for_function(func)
        
    def delete_colored_areas_for_segment(self, segment: "Drawable") -> None:
        """Delete all colored areas associated with a segment"""
        self.colored_area_manager.delete_colored_areas_for_segment(segment)
        
    def get_colored_areas_for_drawable(self, drawable: "Drawable") -> List["Drawable"]:
        """Get all colored areas associated with a drawable"""
        return cast(List["Drawable"], self.colored_area_manager.get_colored_areas_for_drawable(drawable))
        
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
            self.colored_area_manager.update_colored_area(
                name,
                new_color=new_color,
                new_opacity=new_opacity,
                new_left_bound=new_left_bound,
                new_right_bound=new_right_bound,
            )
        )

    def create_drawables_from_new_connections(self) -> None:
        self.polygon_manager.create_triangles_from_segments()

    # ------------------- Graph Methods -------------------
    def create_graph(self, graph_state: "GraphState") -> "Drawable":
        return self.graph_manager.create_graph(graph_state)

    def build_graph_state(
        self,
        *,
        name: str,
        graph_type: str,
        vertices: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        adjacency_matrix: Optional[List[List[float]]],
        directed: Optional[bool],
        root: Optional[str],
        layout: Optional[str],
        placement_box: Optional[Dict[str, float]],
        metadata: Optional[Dict[str, Any]],
    ) -> "GraphState":
        return self.graph_manager.build_graph_state(
            name=name,
            graph_type=graph_type,
            vertices=vertices,
            edges=edges,
            adjacency_matrix=adjacency_matrix,
            directed=directed,
            root=root,
            layout=layout,
            placement_box=placement_box,
            metadata=metadata,
        )

    def delete_graph(self, name: str) -> bool:
        return self.graph_manager.delete_graph(name)

    def get_graph(self, name: str) -> Optional["Drawable"]:
        return self.graph_manager.get_graph(name)

    def capture_graph_state(self, name: str):
        return self.graph_manager.capture_state(name)

    # ------------------- Angle Methods -------------------
    def create_angle(self, vx: float, vy: float, p1x: float, p1y: float, p2x: float, p2y: float, color: Optional[str] = None, angle_name: Optional[str] = None, is_reflex: bool = False, extra_graphics: bool = True) -> Optional["Angle"]:
        """Creates an angle defined by three points."""
        return self.angle_manager.create_angle(
            vx, vy, p1x, p1y, p2x, p2y, 
            color=color,
            angle_name=angle_name,
            is_reflex=is_reflex,
            extra_graphics=extra_graphics
        )

    def delete_angle(self, name: str) -> bool:
        """Removes an angle by its name."""
        return bool(self.angle_manager.delete_angle(name))

    def update_angle(self, name: str, new_color: Optional[str] = None) -> bool:
        """Updates editable properties of an existing angle (currently color)."""
        return bool(self.angle_manager.update_angle(name, new_color=new_color))

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
    ):
        """Create a circle arc drawable."""
        return self.arc_manager.create_circle_arc(
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

    def delete_circle_arc(self, name: str) -> bool:
        """Delete a circle arc by name."""
        return bool(self.arc_manager.delete_circle_arc(name))

    def update_circle_arc(
        self,
        name: str,
        *,
        new_color: Optional[str] = None,
        use_major_arc: Optional[bool] = None,
    ) -> bool:
        """Update editable properties of an existing circle arc."""
        return bool(
            self.arc_manager.update_circle_arc(
                name,
                new_color=new_color,
                use_major_arc=use_major_arc,
            )
        )

    # ------------------- Region-Capable Drawable Lookup -------------------

    def get_region_capable_drawable_by_name(self, name: str) -> Optional["Drawable"]:
        """Get a drawable that can be converted to a Region by its name.
        
        Searches polygons, circles, ellipses, arcs, and segments in sequence.
        Segments are treated as half-planes for region operations.
        
        Args:
            name: The name of the drawable to find
            
        Returns:
            The drawable if found, None otherwise
        """
        if not name:
            return None
        
        polygon = self.polygon_manager.get_polygon_by_name(name)
        if polygon is not None:
            return polygon
        
        circle = self.circle_manager.get_circle_by_name(name)
        if circle is not None:
            return circle
        
        ellipse = self.ellipse_manager.get_ellipse_by_name(name)
        if ellipse is not None:
            return ellipse
        
        arc = self.arc_manager.get_circle_arc_by_name(name)
        if arc is not None:
            return arc
        
        segment = self.segment_manager.get_segment_by_name(name)
        if segment is not None:
            return segment
        
        return None