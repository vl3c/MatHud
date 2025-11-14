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
    - TriangleManager: Triangle operations with vertex coordinate matching
    - RectangleManager: Rectangle operations with diagonal point handling
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

from typing import TYPE_CHECKING, List, Optional, cast

from geometry import Point
from managers.point_manager import PointManager
from managers.segment_manager import SegmentManager
from managers.vector_manager import VectorManager
from managers.triangle_manager import TriangleManager
from managers.function_manager import FunctionManager
from managers.circle_manager import CircleManager
from managers.ellipse_manager import EllipseManager
from managers.rectangle_manager import RectangleManager
from managers.colored_area_manager import ColoredAreaManager
from managers.drawable_manager_proxy import DrawableManagerProxy
from name_generator.drawable import DrawableNameGenerator
from managers.drawable_dependency_manager import DrawableDependencyManager
from managers.drawables_container import DrawablesContainer
from managers.angle_manager import AngleManager
from managers.label_manager import LabelManager

if TYPE_CHECKING:
    from canvas import Canvas
    from drawables.drawable import Drawable
    from drawables.point import Point
    from drawables.segment import Segment
    from drawables.vector import Vector
    from drawables.triangle import Triangle
    from drawables.function import Function
    from drawables.circle import Circle
    from drawables.ellipse import Ellipse
    from drawables.rectangle import Rectangle
    from drawables.angle import Angle
    from drawables.colored_area import ColoredArea

class DrawableManager:
    """
    Manages drawable objects for a Canvas.
    
    This class coordinates specialized managers for each drawable type:
    - Points
    - Segments
    - Vectors
    - Triangles
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
        
        self.triangle_manager: TriangleManager = TriangleManager(
            canvas, self.drawables, self.name_generator, self.dependency_manager, 
            self.point_manager, self.segment_manager, self.proxy
        )
        
        self.function_manager: FunctionManager = FunctionManager(
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
        
        self.rectangle_manager: RectangleManager = RectangleManager(
            canvas, self.drawables, self.name_generator, self.dependency_manager,
            self.point_manager, self.segment_manager, self.proxy
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
        
        # No need for the loop that sets drawable_manager anymore
        # The proxy handles forwarding calls to the appropriate managers
        
    # ------------------- General Drawable Methods -------------------
    
    def get_drawables(self) -> List["Drawable"]:
        """Get all drawables as a flat list, with colored areas first (behind other elements)"""
        return cast(List["Drawable"], self.drawables.get_all_with_layering())
    
    # ------------------- Point Methods -------------------
    
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
        
    def create_point(self, x: float, y: float, name: str = "", extra_graphics: bool = True) -> "Point":
        """Create a new point at the specified coordinates"""
        return self.point_manager.create_point(x, y, name, extra_graphics)
        
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
        
    def create_segment(self, x1: float, y1: float, x2: float, y2: float, name: str = "", extra_graphics: bool = True) -> "Segment":
        """Create a new segment between the specified points"""
        return self.segment_manager.create_segment(x1, y1, x2, y2, name, extra_graphics)
        
    def delete_segment(self, x1: float, y1: float, x2: float, y2: float, delete_children: bool = True, delete_parents: bool = False) -> bool:
        """Delete a segment at the specified coordinates"""
        return bool(self.segment_manager.delete_segment(x1, y1, x2, y2, delete_children, delete_parents))
        
    def delete_segment_by_name(self, name: str, delete_children: bool = True, delete_parents: bool = False) -> bool:
        """Delete a segment by its name"""
        return bool(self.segment_manager.delete_segment_by_name(name, delete_children, delete_parents))
    
    # ------------------- Vector Methods -------------------
    
    def get_vector(self, origin_x: float, origin_y: float, tip_x: float, tip_y: float) -> Optional["Vector"]:
        """Get a vector by its origin and tip coordinates"""
        return self.vector_manager.get_vector(origin_x, origin_y, tip_x, tip_y)
        
    def create_vector(self, origin_x: float, origin_y: float, tip_x: float, tip_y: float, name: str = "", extra_graphics: bool = True) -> "Vector":
        """Create a new vector with the specified origin and tip"""
        return self.vector_manager.create_vector(origin_x, origin_y, tip_x, tip_y, name, extra_graphics)
        
    def delete_vector(self, origin_x: float, origin_y: float, tip_x: float, tip_y: float) -> bool:
        """Delete a vector with the specified origin and tip"""
        return bool(self.vector_manager.delete_vector(origin_x, origin_y, tip_x, tip_y))
    
    # ------------------- Triangle Methods -------------------
    
    def get_triangle(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> Optional["Triangle"]:
        """Get a triangle by its vertex coordinates"""
        return self.triangle_manager.get_triangle(x1, y1, x2, y2, x3, y3)
        
    def create_triangle(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, name: str = "", extra_graphics: bool = True) -> "Triangle":
        """Create a new triangle with the specified vertices"""
        return self.triangle_manager.create_triangle(x1, y1, x2, y2, x3, y3, name, extra_graphics)
        
    def delete_triangle(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> bool:
        """Delete a triangle with the specified vertices"""
        return bool(self.triangle_manager.delete_triangle(x1, y1, x2, y2, x3, y3))
    
    # ------------------- Function Methods -------------------
    
    def get_function(self, name: str) -> Optional["Function"]:
        """Get a function by its name"""
        return self.function_manager.get_function(name)
        
    def draw_function(self, function_string: str, name: str, left_bound: Optional[float] = None, right_bound: Optional[float] = None) -> "Function":
        """Create a new function with the specified expression"""
        return self.function_manager.draw_function(function_string, name, left_bound, right_bound)
        
    def delete_function(self, name: str) -> bool:
        """Delete a function by its name"""
        return bool(self.function_manager.delete_function(name))
    
    # ------------------- Circle Methods -------------------
    
    def get_circle(self, center_x: float, center_y: float, radius: float) -> Optional["Circle"]:
        """Get a circle by its center coordinates and radius"""
        return self.circle_manager.get_circle(center_x, center_y, radius)
        
    def get_circle_by_name(self, name: str) -> Optional["Circle"]:
        """Get a circle by its name"""
        return self.circle_manager.get_circle_by_name(name)
        
    def create_circle(self, center_x: float, center_y: float, radius: float, name: str = "", extra_graphics: bool = True) -> "Circle":
        """Create a new circle with the specified center and radius"""
        return self.circle_manager.create_circle(center_x, center_y, radius, name, extra_graphics)
        
    def delete_circle(self, name: str) -> bool:
        """Delete a circle by its name"""
        return bool(self.circle_manager.delete_circle(name))
    
    # ------------------- Ellipse Methods -------------------
    
    def get_ellipse(self, center_x: float, center_y: float, radius_x: float, radius_y: float) -> Optional["Ellipse"]:
        """Get an ellipse by its center coordinates and radii"""
        return self.ellipse_manager.get_ellipse(center_x, center_y, radius_x, radius_y)
        
    def get_ellipse_by_name(self, name: str) -> Optional["Ellipse"]:
        """Get an ellipse by its name"""
        return self.ellipse_manager.get_ellipse_by_name(name)
        
    def create_ellipse(self, center_x: float, center_y: float, radius_x: float, radius_y: float, rotation_angle: float = 0, name: str = "", extra_graphics: bool = True) -> "Ellipse":
        """Create a new ellipse with the specified center, radii, and rotation"""
        return self.ellipse_manager.create_ellipse(center_x, center_y, radius_x, radius_y, rotation_angle, name, extra_graphics)
        
    def delete_ellipse(self, name: str) -> bool:
        """Delete an ellipse by its name"""
        return bool(self.ellipse_manager.delete_ellipse(name))
    
    # ------------------- Rectangle Methods -------------------
    
    def get_rectangle_by_diagonal_points(self, px: float, py: float, opposite_px: float, opposite_py: float) -> Optional["Rectangle"]:
        """Get a rectangle by its diagonal points"""
        return self.rectangle_manager.get_rectangle_by_diagonal_points(px, py, opposite_px, opposite_py)
        
    def get_rectangle_by_name(self, name: str) -> Optional["Rectangle"]:
        """Get a rectangle by its name"""
        return self.rectangle_manager.get_rectangle_by_name(name)
        
    def create_rectangle(self, px: float, py: float, opposite_px: float, opposite_py: float, name: str = "", extra_graphics: bool = True) -> "Rectangle":
        """Create a new rectangle with the specified diagonal points"""
        return self.rectangle_manager.create_rectangle(px, py, opposite_px, opposite_py, name, extra_graphics)
        
    def delete_rectangle(self, name: str) -> bool:
        """Delete a rectangle by its name"""
        return bool(self.rectangle_manager.delete_rectangle(name))
    
    # ------------------- Colored Area Methods -------------------
    
    def create_colored_area(self, drawable1_name: str, drawable2_name: Optional[str] = None, left_bound: Optional[float] = None, right_bound: Optional[float] = None, color: str = "lightblue", opacity: float = 0.3) -> "ColoredArea":
        """Create a new colored area between drawables"""
        return self.colored_area_manager.create_colored_area(drawable1_name, drawable2_name, left_bound, right_bound, color, opacity)
        
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
        
    def update_colored_area_style(self, name: str, color: Optional[str] = None, opacity: Optional[float] = None) -> bool:
        """Update the style of a colored area"""
        return bool(self.colored_area_manager.update_colored_area_style(name, color, opacity))

    def create_drawables_from_new_connections(self) -> None:
        # Call the method on the TriangleManager
        self.triangle_manager.create_new_triangles_from_connected_segments()

    # The transformation methods have been moved to TransformationsManager
    # def translate_object(self, name, x_offset, y_offset):
    #     ...
    # def rotate_object(self, name, angle):
    #     ...

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