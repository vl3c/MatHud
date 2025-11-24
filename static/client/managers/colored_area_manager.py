"""
MatHud Colored Area Management System

Manages colored area creation, deletion, and style management for mathematical region visualization.
Handles areas bounded by functions, segments, or mixed boundaries with automatic type detection.

Core Responsibilities:
    - Area Creation: Creates colored regions between mathematical boundaries
    - Area Deletion: Safe removal with cleanup of dependencies
    - Type Detection: Automatically determines area type based on boundary objects
    - Style Management: Handles color and opacity customization

Supported Area Types:
    - Function-Function Areas: Regions between two mathematical functions
    - Segment-Segment Areas: Regions between line segments and axes
    - Function-Segment Areas: Mixed regions between functions and segments
    - Axis Integration: Areas between objects and coordinate axes

Advanced Features:
    - Automatic Type Detection: Determines appropriate area class based on boundaries
    - Smart Boundary Handling: Manages x-axis references and null boundaries
    - Intersection Calculation: Creates boundary points for accurate area representation
    - Color Validation: Ensures valid color and opacity values

Integration Points:
    - FunctionManager: Retrieves function objects for boundary definition
    - SegmentManager: Retrieves segment objects and creates intersection points
    - StyleUtils: Validates color and opacity parameters
    - Canvas: Handles area rendering and visual updates

Mathematical Context:
    - Integration Visualization: Represents definite integrals as colored areas
    - Geometric Analysis: Shows relationships between mathematical objects
    - Boundary Calculations: Handles complex boundary intersections
    - Domain Management: Respects function domains and segment ranges

State Management:
    - Undo/Redo: Complete state archiving for area operations
    - Dependency Tracking: Maintains relationships with boundary objects
    - Canvas Integration: Immediate visual updates after modifications
    - Cleanup Logic: Intelligent removal of areas when boundaries are deleted
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Union

from constants import (
    closed_shape_resolution_minimum,
    default_area_fill_color,
    default_area_opacity,
    default_closed_shape_resolution,
)
from drawables.closed_shape_colored_area import ClosedShapeColoredArea
from drawables.circle import Circle
from drawables.ellipse import Ellipse
from drawables.function import Function
from drawables.function_segment_bounded_colored_area import FunctionSegmentBoundedColoredArea
from drawables.functions_bounded_colored_area import FunctionsBoundedColoredArea
from drawables.segment import Segment
from drawables.segments_bounded_colored_area import SegmentsBoundedColoredArea
from managers.edit_policy import DrawableEditPolicy, EditRule, get_drawable_edit_policy
from managers.polygon_type import PolygonType
from utils.geometry_utils import GeometryUtils
from utils.style_utils import StyleUtils

if TYPE_CHECKING:
    from drawables.drawable import Drawable
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from name_generator.drawable import DrawableNameGenerator
    from drawables.rectangle import Rectangle

class ColoredAreaManager:
    """
    Manages colored area drawables for a Canvas.
    
    This class is responsible for:
    - Creating colored area objects
    - Retrieving colored area objects
    - Deleting colored area objects
    """
    
    def __init__(
        self,
        canvas: "Canvas",
        drawables_container: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        drawable_manager_proxy: "DrawableManagerProxy",
    ) -> None:
        """
        Initialize the ColoredAreaManager.
        
        Args:
            canvas: The Canvas object this manager is responsible for
            drawables_container: The container for storing drawables
            name_generator: Generator for drawable names
            dependency_manager: Manager for drawable dependencies
            drawable_manager_proxy: Proxy to the main DrawableManager
        """
        self.canvas: "Canvas" = canvas
        self.drawables: "DrawablesContainer" = drawables_container
        self.name_generator: "DrawableNameGenerator" = name_generator
        self.dependency_manager: "DrawableDependencyManager" = dependency_manager
        self.drawable_manager: "DrawableManagerProxy" = drawable_manager_proxy
        
    def create_colored_area(
        self,
        drawable1_name: Optional[str],
        drawable2_name: Optional[str] = None,
        left_bound: Optional[float] = None,
        right_bound: Optional[float] = None,
        color: str = default_area_fill_color,
        opacity: float = default_area_opacity,
    ) -> Union[FunctionsBoundedColoredArea, SegmentsBoundedColoredArea, FunctionSegmentBoundedColoredArea]:
        """
        Creates a colored area between two functions, two segments, or a function and a segment.
        Automatically determines the type of colored area based on the inputs.
        
        Args:
            drawable1_name: Name of first function/segment (or None for x-axis)
            drawable2_name: Name of second function/segment (or None for x-axis)
            left_bound: Optional left bound for function areas
            right_bound: Optional right bound for function areas
            color: Color of the area (default: lightblue)
            opacity: Opacity of the area (default: 0.3)
            
        Returns:
            The created colored area object
            
        Raises:
            ValueError: If color or opacity values are invalid or if drawables not found
        """
        # Validate color and opacity before proceeding
        self.canvas._validate_color_and_opacity(color, opacity)
        
        # Archive for undo
        self.canvas.undo_redo_manager.archive()
        
        # Get the first drawable
        drawable1: Optional[Union[Function, Segment]] = None
        if drawable1_name is not None and drawable1_name != "x_axis":
            drawable1 = self.drawable_manager.get_segment_by_name(drawable1_name)
            if drawable1 is None:
                drawable1 = self.drawable_manager.get_function(drawable1_name)
            if drawable1 is None:
                raise ValueError(f"Could not find drawable with name {drawable1_name}")
        
        # Get the second drawable if provided
        drawable2: Optional[Union[Function, Segment]] = None
        if drawable2_name is not None and drawable2_name != "x_axis":
            drawable2 = self.drawable_manager.get_segment_by_name(drawable2_name)
            if drawable2 is None:
                drawable2 = self.drawable_manager.get_function(drawable2_name)
            if drawable2 is None:
                raise ValueError(f"Could not find drawable with name {drawable2_name}")
        
        if isinstance(drawable1, Segment) and isinstance(drawable2, Function):
            # Swap so the function is treated as the primary drawable
            drawable1, drawable2 = drawable2, drawable1

        if isinstance(drawable1, Segment) and (drawable2 is None or isinstance(drawable2, Segment)):
            # Segment-segment or segment-xaxis case
            if drawable2:  # Segment-segment case
                # Create points at overlap boundaries
                def get_y_at_x(segment: Segment, x: float) -> float:
                    # Linear interpolation to find y value at x using math coordinates
                    x1: float = segment.point1.x
                    y1: float = segment.point1.y
                    x2: float = segment.point2.x
                    y2: float = segment.point2.y
                    if x2 == x1:
                        return y1  # Vertical segment
                    t: float = (x - x1) / (x2 - x1)
                    return y1 + t * (y2 - y1)

                # Get x-ranges of both segments using math coordinates
                x1_min = min(drawable1.point1.x, drawable1.point2.x)
                x1_max = max(drawable1.point1.x, drawable1.point2.x)
                x2_min = min(drawable2.point1.x, drawable2.point2.x)
                x2_max = max(drawable2.point1.x, drawable2.point2.x)

                # Check if segment1 endpoints create points on segment2
                if x1_min >= x2_min and x1_min <= x2_max:
                    y = get_y_at_x(drawable2, x1_min)
                    self.drawable_manager.create_point(x1_min, y)
                if x1_max >= x2_min and x1_max <= x2_max:
                    y = get_y_at_x(drawable2, x1_max)
                    self.drawable_manager.create_point(x1_max, y)

                # Check if segment2 endpoints create points on segment1
                if x2_min >= x1_min and x2_min <= x1_max:
                    y = get_y_at_x(drawable1, x2_min)
                    self.drawable_manager.create_point(x2_min, y)
                if x2_max >= x1_min and x2_max <= x1_max:
                    y = get_y_at_x(drawable1, x2_max)
                    self.drawable_manager.create_point(x2_max, y)

            colored_area: Union[SegmentsBoundedColoredArea, FunctionSegmentBoundedColoredArea, FunctionsBoundedColoredArea]
            colored_area = SegmentsBoundedColoredArea(drawable1, drawable2, color=color, opacity=opacity)
        elif isinstance(drawable2, Segment):
            # Function-segment case (we know drawable1 is not a segment due to the swap above)
            colored_area = FunctionSegmentBoundedColoredArea(drawable1, drawable2, color=color, opacity=opacity)
        else:
            # Function-function case
            colored_area = FunctionsBoundedColoredArea(drawable1, drawable2, 
                                                     left_bound=left_bound, right_bound=right_bound,
                                                     color=color, opacity=opacity)

        # Add to drawables
        self.drawables.add(colored_area)
        
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return colored_area

    def create_closed_shape_colored_area(
        self,
        *,
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
    ) -> ClosedShapeColoredArea:
        """
        Creates a closed-shape colored area from existing triangles, rectangles,
        polygonal segments, circles, ellipses, or simple round-shape chords.
        """
        self.canvas._validate_color_and_opacity(color, opacity)
        self.canvas.undo_redo_manager.archive()

        shape_type: Optional[str] = None
        segments: List[Segment] = []
        circle: Optional[Circle] = None
        ellipse: Optional[Ellipse] = None
        chord_segment: Optional[Segment] = None

        if triangle_name:
            triangle = self.drawable_manager.get_polygon_by_name(
                triangle_name,
                polygon_type=PolygonType.TRIANGLE,
            )
            if not triangle:
                raise ValueError(f"Triangle '{triangle_name}' was not found.")
            segments = [
                triangle.segment1,
                triangle.segment2,
                triangle.segment3,
            ]
            shape_type = "polygon"
        elif rectangle_name:
            rectangle = self.drawable_manager.get_polygon_by_name(
                rectangle_name,
                polygon_type=PolygonType.RECTANGLE,
            )
            if not rectangle:
                raise ValueError(f"Rectangle '{rectangle_name}' was not found.")
            segments = [
                rectangle.segment1,
                rectangle.segment2,
                rectangle.segment3,
                rectangle.segment4,
            ]
            shape_type = "polygon"
        elif polygon_segment_names:
            if len(polygon_segment_names) < 3:
                raise ValueError("At least three segments are required to form a polygon.")
            for name in polygon_segment_names:
                segment = self.drawable_manager.get_segment_by_name(name)
                if not segment:
                    raise ValueError(f"Segment '{name}' was not found.")
                segments.append(segment)
            if not GeometryUtils.segments_form_closed_loop(segments):
                raise ValueError("Provided segments do not form a closed loop.")
            shape_type = "polygon"
        elif circle_name and chord_segment_name:
            circle = self.drawable_manager.get_circle_by_name(circle_name)
            chord_segment = self.drawable_manager.get_segment_by_name(chord_segment_name)
            if not circle or not chord_segment:
                raise ValueError("Circle or chord segment could not be resolved.")
            segments = [chord_segment]
            shape_type = "circle_segment"
        elif circle_name:
            circle = self.drawable_manager.get_circle_by_name(circle_name)
            if not circle:
                raise ValueError(f"Circle '{circle_name}' was not found.")
            shape_type = "circle"
        elif ellipse_name and chord_segment_name:
            ellipse = self.drawable_manager.get_ellipse_by_name(ellipse_name)
            chord_segment = self.drawable_manager.get_segment_by_name(chord_segment_name)
            if not ellipse or not chord_segment:
                raise ValueError("Ellipse or chord segment could not be resolved.")
            segments = [chord_segment]
            shape_type = "ellipse_segment"
        elif ellipse_name:
            ellipse = self.drawable_manager.get_ellipse_by_name(ellipse_name)
            if not ellipse:
                raise ValueError(f"Ellipse '{ellipse_name}' was not found.")
            shape_type = "ellipse"
        else:
            raise ValueError(
                "Specify a triangle, rectangle, polygon segments, circle, or ellipse to create a closed shape area."
            )

        closed_area = ClosedShapeColoredArea(
            shape_type=shape_type,
            segments=segments,
            circle=circle,
            ellipse=ellipse,
            chord_segment=chord_segment,
            arc_clockwise=bool(arc_clockwise),
            resolution=max(
                closed_shape_resolution_minimum,
                int(resolution if resolution is not None else default_closed_shape_resolution),
            ),
            color=color,
            opacity=opacity,
        )

        self.drawables.add(closed_area)
        self.dependency_manager.analyze_drawable_for_dependencies(closed_area)

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return closed_area
        
    def delete_colored_area(self, name: str) -> bool:
        """
        Delete a colored area by its name.
        
        Searches through all colored area categories to find and remove the area
        with the specified name. Archives the state for undo functionality.
        
        Args:
            name (str): The name of the colored area to delete
            
        Returns:
            bool: True if the colored area was found and deleted, False otherwise
        """
        # Find the colored area in all categories
        colored_area: Optional["Drawable"] = None
        for category_property in [
            self.drawables.FunctionsBoundedColoredAreas,
            self.drawables.SegmentsBoundedColoredAreas,
            self.drawables.FunctionSegmentBoundedColoredAreas,
            self.drawables.ClosedShapeColoredAreas,
        ]:
            for area in category_property:
                if area.name == name:
                    colored_area = area
                    break
            if colored_area:
                break
                
        if not colored_area:
            return False
            
        # Archive before deletion
        self.canvas.undo_redo_manager.archive()
        
        # Remove the colored area
        self.drawables.remove(colored_area)
        
        # Redraw
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return True
        
    def delete_colored_areas_for_function(self, func: Union[str, Function]) -> bool:
        """
        Deletes all colored areas associated with a function
        
        Args:
            func: The function whose colored areas should be deleted
            
        Returns:
            bool: True if any areas were deleted, False otherwise
        """
        # Check if function is a string (name) or object
        if isinstance(func, str):
            func = self.drawable_manager.get_function(func)
            
        if not func:
            return False
            
        # First check if there are any areas to delete
        areas_to_delete = []
        
        # Check FunctionsBoundedColoredArea
        for area in self.drawables.FunctionsBoundedColoredAreas:
            if area.func1 == func or area.func2 == func:
                areas_to_delete.append(area)

        # Check FunctionSegmentBoundedColoredArea
        for area in self.drawables.FunctionSegmentBoundedColoredAreas:
            if area.func == func:
                areas_to_delete.append(area)
        
        if areas_to_delete:
            # Archive for undo
            self.canvas.undo_redo_manager.archive()
            
            # Now delete the areas
            for area in areas_to_delete:
                self.drawables.remove(area)
                
            if self.canvas.draw_enabled:
                self.canvas.draw()
                
            return True
            
        return False
        
    def delete_colored_areas_for_segment(self, segment: Union[str, Segment]) -> bool:
        """
        Deletes all colored areas associated with a segment
        
        Args:
            segment: The segment whose colored areas should be deleted
            
        Returns:
            bool: True if any areas were deleted, False otherwise
        """
        # Check if segment is a string (name) or object
        if isinstance(segment, str):
            segment = self.drawable_manager.get_segment_by_name(segment)
            
        if not segment:
            return False
            
        # First check if there are any areas to delete
        areas_to_delete = []
        
        # Check SegmentsBoundedColoredArea
        for area in self.drawables.SegmentsBoundedColoredAreas:
            if area.uses_segment(segment):
                areas_to_delete.append(area)
                
        # Check FunctionSegmentBoundedColoredArea
        for area in self.drawables.FunctionSegmentBoundedColoredAreas:
            if area.uses_segment(segment):
                areas_to_delete.append(area)
        for area in getattr(self.drawables, "ClosedShapeColoredAreas", []):
            if hasattr(area, "uses_segment") and area.uses_segment(segment):
                areas_to_delete.append(area)
        
        if areas_to_delete:
            # Archive for undo
            self.canvas.undo_redo_manager.archive()
            
            # Now delete the areas
            for area in areas_to_delete:
                self.drawables.remove(area)
                
            if self.canvas.draw_enabled:
                self.canvas.draw()
                
            return True
            
        return False
        
    def get_colored_areas_for_drawable(self, drawable: Union[Function, Segment]) -> List["Drawable"]:
        """
        Gets all colored areas associated with a drawable (function or segment)
        
        Args:
            drawable: The function or segment to find colored areas for
            
        Returns:
            list: List of colored areas that use the drawable
        """
        areas: List["Drawable"] = []
        
        if isinstance(drawable, Function):
            # Check FunctionsBoundedColoredArea
            for area in self.drawables.FunctionsBoundedColoredAreas:
                if area.func1 == drawable or area.func2 == drawable:
                    areas.append(area)

            # Check FunctionSegmentBoundedColoredArea
            for area in self.drawables.FunctionSegmentBoundedColoredAreas:
                if area.func == drawable:
                    areas.append(area)
                    
        elif isinstance(drawable, Segment):
            # Check SegmentsBoundedColoredArea
            for area in self.drawables.SegmentsBoundedColoredAreas:
                if area.uses_segment(drawable):
                    areas.append(area)

            # Check FunctionSegmentBoundedColoredArea
            for area in self.drawables.FunctionSegmentBoundedColoredAreas:
                if area.uses_segment(drawable):
                    areas.append(area)

            for area in self.drawables.ClosedShapeColoredAreas:
                if hasattr(area, "uses_segment") and area.uses_segment(drawable):
                    areas.append(area)

        elif isinstance(drawable, Circle):
            for area in self.drawables.ClosedShapeColoredAreas:
                if hasattr(area, "uses_circle") and area.uses_circle(drawable):
                    areas.append(area)

        elif isinstance(drawable, Ellipse):
            for area in self.drawables.ClosedShapeColoredAreas:
                if hasattr(area, "uses_ellipse") and area.uses_ellipse(drawable):
                    areas.append(area)
                    
        return areas

    def _get_colored_area_by_name(self, name: str) -> Optional["Drawable"]:
        collections: List[List["Drawable"]] = [
            self.drawables.FunctionsBoundedColoredAreas,
            self.drawables.SegmentsBoundedColoredAreas,
            self.drawables.FunctionSegmentBoundedColoredAreas,
            self.drawables.ColoredAreas,
            self.drawables.ClosedShapeColoredAreas,
        ]
        for collection in collections:
            for area in collection:
                if area.name == name:
                    return area
        return None

    def update_colored_area(
        self,
        name: str,
        new_color: Optional[str] = None,
        new_opacity: Optional[float] = None,
        new_left_bound: Optional[float] = None,
        new_right_bound: Optional[float] = None,
    ) -> bool:
        """Update editable properties of a colored area."""
        area = self._get_colored_area_by_name(name)
        if not area:
            raise ValueError(f"Colored area '{name}' not found")

        pending_fields = self._collect_colored_area_fields(
            area,
            new_color,
            new_opacity,
            new_left_bound,
            new_right_bound,
        )

        policy = self._get_policy_for_area(area)
        self._validate_policy(policy, list(pending_fields.keys()))
        self._validate_colored_area_payload(area, pending_fields, new_color, new_opacity, new_left_bound, new_right_bound)

        self.canvas.undo_redo_manager.archive()
        self._apply_colored_area_updates(area, pending_fields, new_color, new_opacity, new_left_bound, new_right_bound)

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def _collect_colored_area_fields(
        self,
        area: "Drawable",
        new_color: Optional[str],
        new_opacity: Optional[float],
        new_left_bound: Optional[float],
        new_right_bound: Optional[float],
    ) -> Dict[str, str]:
        pending_fields: Dict[str, str] = {}

        if new_color is not None:
            pending_fields["color"] = str(new_color)

        if new_opacity is not None:
            pending_fields["opacity"] = "opacity"

        supports_bounds = area.get_class_name() == "FunctionsBoundedColoredArea"

        if (new_left_bound is not None or new_right_bound is not None) and not supports_bounds:
            raise ValueError("Only functions-bounded colored areas support editing left/right bounds.")

        if new_left_bound is not None:
            pending_fields["left_bound"] = "left_bound"

        if new_right_bound is not None:
            pending_fields["right_bound"] = "right_bound"

        if not pending_fields:
            raise ValueError("Provide at least one property to update.")

        return pending_fields

    def _get_policy_for_area(self, area: "Drawable") -> DrawableEditPolicy:
        area_type = area.get_class_name()
        policy = get_drawable_edit_policy(area_type)
        if not policy:
            raise ValueError(f"Edit policy for {area_type} is not configured.")
        return policy

    def _validate_policy(self, policy: DrawableEditPolicy, requested_fields: List[str]) -> Dict[str, EditRule]:
        validated_rules: Dict[str, EditRule] = {}
        for field in requested_fields:
            rule = policy.get_rule(field)
            if not rule:
                raise ValueError(f"Editing field '{field}' is not permitted for this colored area.")
            validated_rules[field] = rule
        return validated_rules

    def _validate_colored_area_payload(
        self,
        area: "Drawable",
        pending_fields: Dict[str, str],
        new_color: Optional[str],
        new_opacity: Optional[float],
        new_left_bound: Optional[float],
        new_right_bound: Optional[float],
    ) -> None:
        if "color" in pending_fields and (new_color is None or not StyleUtils.is_valid_css_color(new_color)):
            raise ValueError(f"Invalid CSS color: {new_color}")

        if "opacity" in pending_fields:
            if new_opacity is None or not StyleUtils.validate_opacity(new_opacity):
                raise ValueError("Opacity must be between 0 and 1.")

        if "left_bound" in pending_fields and new_left_bound is None:
            raise ValueError("left_bound requires a numeric value.")

        if "right_bound" in pending_fields and new_right_bound is None:
            raise ValueError("right_bound requires a numeric value.")

        if any(field in pending_fields for field in ("left_bound", "right_bound")):
            if not hasattr(area, "left_bound") or not hasattr(area, "right_bound"):
                raise ValueError("Bounds can only be set on functions bounded colored areas.")

            updated_left = area.left_bound
            updated_right = area.right_bound

            if "left_bound" in pending_fields and new_left_bound is not None:
                updated_left = float(new_left_bound)

            if "right_bound" in pending_fields and new_right_bound is not None:
                updated_right = float(new_right_bound)

            if (
                updated_left is not None
                and updated_right is not None
                and updated_left >= updated_right
            ):
                raise ValueError("left_bound must be less than right_bound.")

    def _apply_colored_area_updates(
        self,
        area: "Drawable",
        pending_fields: Dict[str, str],
        new_color: Optional[str],
        new_opacity: Optional[float],
        new_left_bound: Optional[float],
        new_right_bound: Optional[float],
    ) -> None:
        if "color" in pending_fields and new_color is not None:
            area.update_color(str(new_color))

        if "opacity" in pending_fields and new_opacity is not None:
            area.update_opacity(float(new_opacity))

        if isinstance(area, FunctionsBoundedColoredArea):
            if "left_bound" in pending_fields and new_left_bound is not None:
                area.update_left_bound(float(new_left_bound))
            if "right_bound" in pending_fields and new_right_bound is not None:
                area.update_right_bound(float(new_right_bound))
        return True 