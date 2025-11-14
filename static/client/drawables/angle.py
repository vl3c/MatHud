"""
MatHud Angle Geometric Object

Represents an angle formed by two intersecting line segments with arc and label visualization.
Provides angle measurement and support for both standard and reflex angles.

Key Features:
    - Two-segment angle construction and validation
    - Angle measurement in degrees with arc visualization
    - Support for standard (0-180°) and reflex (180-360°) angles
    - Arc parameter generation for renderer consumption

Geometric Properties:
    - Vertex point identification from segment intersection
    - Arm point extraction for angle calculation
    - Angle sweep calculation and mid-arc label positioning support

Dependencies:
    - constants: Angle defaults
    - drawables.drawable: Base class interface
    - utils.math_utils: Angle calculation and geometric validation
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, cast

from constants import (
    DEFAULT_ANGLE_COLOR,
    DEFAULT_ANGLE_ARC_SCREEN_RADIUS,
    DEFAULT_ANGLE_TEXT_ARC_RADIUS_FACTOR,
    point_label_font_size,
)
from drawables.drawable import Drawable
from drawables.point import Point
from drawables.segment import Segment
import utils.math_utils as math_utils

class Angle(Drawable):
    """Represents an angle formed by two intersecting line segments with arc visualization.
    
    Validates that two segments form a proper angle and provides arc rendering
    with angle measurement display for both standard and reflex angles.
    
    Attributes:
        segment1 (Segment): First segment forming the angle
        segment2 (Segment): Second segment forming the angle
        is_reflex (bool): Whether to display the reflex (outer) angle
        vertex_point (Point): Common vertex where segments intersect
        arm1_point (Point): End point of first segment arm
        arm2_point (Point): End point of second segment arm
        raw_angle_degrees (float): Fundamental angle measurement (0-360°)
        angle_degrees (float): Display angle (small or reflex based on is_reflex)
         (arc radius is provided by the renderer; a default constant is used when not specified)
    """
    def __init__(
        self,
        arg1: Segment | Point,
        arg2: Segment | Point,
        arg3: Optional[Point] = None,
        *,
        color: str = DEFAULT_ANGLE_COLOR,
        is_reflex: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Initialize an angle from two intersecting line segments.
        
        Validates segment intersection and extracts angle properties for visualization.
        
        Args:
            segment1 (Segment): First segment forming the angle
            segment2 (Segment): Second segment forming the angle
            color (str): CSS color value for angle visualization
            is_reflex (bool): False for inner/standard angle, True for outer/reflex angle
            
        Raises:
            ValueError: If segments do not form a valid angle (must share exactly one vertex)
        """
        if arg3 is None:
            # Accept any segment-like objects (including test doubles) that expose point1/point2.
            segment1 = cast(Segment, arg1)
            segment2 = cast(Segment, arg2)
        elif arg3 is not None and isinstance(arg1, Point) and isinstance(arg2, Point) and isinstance(arg3, Point):
            segment1 = Segment(arg2, arg1)
            segment2 = Segment(arg2, arg3)
        else:
            raise TypeError(
                "Angle requires either two segment-like objects or three Point instances (arm1, vertex, arm2)."
            )

        if not self._segments_form_angle(segment1, segment2):
            raise ValueError("The segments do not form a valid angle (must share exactly one vertex and have distinct arms).")

        self.segment1: Segment = segment1
        self.segment2: Segment = segment2
        self.is_reflex: bool = is_reflex
        
        self.vertex_point: Optional[Point]
        self.arm1_point: Optional[Point]
        self.arm2_point: Optional[Point]
        self.vertex_point, self.arm1_point, self.arm2_point = self._extract_defining_points(self.segment1, self.segment2)

        # Name: prefer provided name; otherwise compute deterministically from segment endpoint names
        computed_name: Optional[str] = None
        try:
            vertex_name: str = self.vertex_point.name
            arm_names: list[str] = sorted([self.arm1_point.name, self.arm2_point.name])
            computed_name = f"angle_{arm_names[0]}{vertex_name}{arm_names[1]}"
            if self.is_reflex:
                computed_name += "_reflex"
        except Exception:
            computed_name = None
        final_name: str = name if name is not None else computed_name if computed_name is not None else "angle"
        
        super().__init__(name=final_name, color=color) 
        
        self.raw_angle_degrees: Optional[float] = None # To store the fundamental CCW angle (0-360)
        self.angle_degrees: Optional[float] = None     # To store the display angle (small or reflex)
        
        self._initialize()

    def _get_common_vertex(self, s1: Segment, s2: Segment) -> Optional[Point]:
        """Identifies and returns the common vertex point object between two segments."""
        if s1.point1 == s2.point1 or s1.point1 == s2.point2:
            return s1.point1
        if s1.point2 == s2.point1 or s1.point2 == s2.point2:
            return s1.point2
        return None

    def _segments_form_angle(self, s1: Segment, s2: Segment) -> bool:
        """
        Validates if two segments can form an angle.
        They must share exactly one common point (vertex) and form distinct, non-degenerate arms.
        """
        if not s1 or not s2: return False # Segments must exist
        if not hasattr(s1, 'point1') or not hasattr(s1, 'point2') or \
           not hasattr(s2, 'point1') or not hasattr(s2, 'point2'):
            return False # Segments must have point attributes

        common_vertex_point_obj: Optional[Point] = self._get_common_vertex(s1, s2) # This returns a Point object or None
        if common_vertex_point_obj is None:
            return False # Segments do not share exactly one common vertex

        # Identify the Point objects for the arms
        arm1_point_obj: Point = s1.point1 if s1.point1 != common_vertex_point_obj else s1.point2
        arm2_point_obj: Point = s2.point1 if s2.point1 != common_vertex_point_obj else s2.point2
        
        return cast(bool, math_utils.MathUtils.are_points_valid_for_angle_geometry(
            (common_vertex_point_obj.x, common_vertex_point_obj.y),
            (arm1_point_obj.x, arm1_point_obj.y),
            (arm2_point_obj.x, arm2_point_obj.y)
        ))

    def _extract_defining_points(self, s1: Segment, s2: Segment) -> Tuple[Point, Point, Point]:
        """Extracts vertex, arm1, and arm2 points. Assumes segments form a valid angle."""
        vertex_p: Optional[Point] = self._get_common_vertex(s1, s2)
        # _segments_form_angle should have already ensured vertex_p is not None
        if vertex_p is None:
            raise ValueError("Segments do not share a common vertex")
        arm1_p: Point = s1.point1 if s1.point1 != vertex_p else s1.point2
        arm2_p: Point = s2.point1 if s2.point1 != vertex_p else s2.point2
        return vertex_p, arm1_p, arm2_p

    def _calculate_display_angle(self, raw_angle_degrees: Optional[float], is_reflex: bool, epsilon: float) -> Optional[float]:
        """Helper function to calculate the display angle based on raw angle and reflex state."""
        if raw_angle_degrees is None:
            return None

        display_angle: Optional[float] = None
        if is_reflex:
            # Calculate reflex angle for display
            if abs(raw_angle_degrees) < epsilon: # Raw angle is 0
                display_angle = 360.0
            elif raw_angle_degrees > epsilon and raw_angle_degrees < (180.0 - epsilon): # Raw is (0, 180)
                display_angle = 360.0 - raw_angle_degrees
            else: # Raw is [180, 360)
                display_angle = raw_angle_degrees
        else:
            # Calculate non-reflex (small) angle for display
            if raw_angle_degrees > (180.0 + epsilon): # Raw is (180, 360)
                display_angle = 360.0 - raw_angle_degrees
            else: # Raw is [0, 180]
                display_angle = raw_angle_degrees
        return display_angle

    def _initialize(self) -> None:
        """Calculates raw_angle_degrees and angle_degrees based on geometry and is_reflex state."""
        if not (self.vertex_point and self.arm1_point and self.arm2_point):
            self.raw_angle_degrees = None
            self.angle_degrees = None
            return

        # Use math-space coordinates for fundamental angle calculation to match tests and model semantics
        vertex_coords: Tuple[float, float] = (self.vertex_point.x, self.vertex_point.y)
        arm1_coords: Tuple[float, float] = (self.arm1_point.x, self.arm1_point.y)
        arm2_coords: Tuple[float, float] = (self.arm2_point.x, self.arm2_point.y)

        # Calculate the fundamental CCW angle from arm1 to arm2 (0-360 degrees)
        self.raw_angle_degrees = math_utils.MathUtils.calculate_angle_degrees(
            vertex_coords, arm1_coords, arm2_coords
        )
        
        self.angle_degrees = self._calculate_display_angle(self.raw_angle_degrees, self.is_reflex, math_utils.MathUtils.EPSILON)
        
        # Arc radius comes from renderer (or default constant when not provided)


    def get_class_name(self) -> str:
        return 'Angle'

    # Removed unused _get_drawing_references (renderer derives directly via mapper)

    def _calculate_arc_parameters(self, vx: float, vy: float, p1x: float, p1y: float, p2x: float, p2y: float, arc_radius: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Calculates SVG path parameters for the arc using screen coordinates for positioning
        and a fixed self.drawn_arc_radius for size.
        vx, vy, p1x, p1y, p2x, p2y are screen coordinates.
        Assumes self.raw_angle_degrees and self.angle_degrees have been set by _initialize.
        """
        if self.raw_angle_degrees is None or self.angle_degrees is None:
            return None
        
        # Degenerate case check (arm points at vertex)
        if (p1x == vx and p1y == vy) or (p2x == vx and p2y == vy):
            return None
        
        current_arc_radius: float = arc_radius if arc_radius is not None else DEFAULT_ANGLE_ARC_SCREEN_RADIUS
        if current_arc_radius <= 0: 
             return None
        
        epsilon: float = math_utils.MathUtils.EPSILON # Use MathUtils.EPSILON

        # Calculate angles in a y-up frame to preserve mathematical CCW orientation
        angle_v_p1_rad: float = math.atan2(vy - p1y, p1x - vx)
        angle_v_p2_rad: float = math.atan2(vy - p2y, p2x - vx)

        # Calculate arc start and end points on screen (invert y to map back to screen coords)
        arc_start_x: float = vx + current_arc_radius * math.cos(angle_v_p1_rad)
        arc_start_y: float = vy - current_arc_radius * math.sin(angle_v_p1_rad)
        arc_end_x: float = vx + current_arc_radius * math.cos(angle_v_p2_rad)
        arc_end_y: float = vy - current_arc_radius * math.sin(angle_v_p2_rad)
        
        final_sweep_flag: str
        final_large_arc_flag: str
        final_sweep_flag, final_large_arc_flag = self._get_arc_flags(
            self.angle_degrees, self.raw_angle_degrees, epsilon
        )
            
        return {
            "arc_radius_on_screen": current_arc_radius,
            "angle_v_p1_rad": angle_v_p1_rad, # For text positioning
            "final_sweep_flag": final_sweep_flag,
            "final_large_arc_flag": final_large_arc_flag,
            "arc_start_x": arc_start_x,
            "arc_start_y": arc_start_y,
            "arc_end_x": arc_end_x,
            "arc_end_y": arc_end_y,
        }

    def _get_arc_flags(self, display_angle_degrees: Optional[float], raw_angle_degrees: Optional[float], epsilon: float) -> Tuple[str, str]:
        """Determines sweep and large-arc flags for drawing the angle arc.

        Note: SVG uses a y-down coordinate system; sweep-flag=1 draws in the positive-angle (clockwise) direction.
        We compute math-space angles CCW, so when display angle follows the raw CCW direction, we use sweep=0;
        when display angle goes the opposite direction (to get the small or reflex complement), use sweep=1.
        """
        if display_angle_degrees is None or raw_angle_degrees is None:
            return '0', '0'
        # Large-arc flag: 1 if display angle > 180 (or ~360)
        if abs(display_angle_degrees - 360.0) < epsilon:
            large_arc_flag: str = '1'
        elif display_angle_degrees > 180.0 + epsilon:
            large_arc_flag = '1'
        else:
            large_arc_flag = '0'

        # Determine if display follows raw CCW direction (math-space)
        same_direction: bool = abs(display_angle_degrees - raw_angle_degrees) < epsilon
        # Special equivalences at 0/360
        if abs(display_angle_degrees) < epsilon and abs(raw_angle_degrees - 360.0) < epsilon:
            same_direction = True
        if abs(display_angle_degrees - 360.0) < epsilon and abs(raw_angle_degrees) < epsilon:
            same_direction = False

        # In SVG, sweep=0 gives CCW visually (since y-down)
        sweep_flag: str = '0' if same_direction else '1'

        return sweep_flag, large_arc_flag

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a serializable dictionary of the angle's state.
        """
        return {
            "name": self.name, 
            "type": "angle",   
            "args": {
                "segment1_name": self.segment1.name,
                "segment2_name": self.segment2.name,
                "color": self.color,
                "is_reflex": self.is_reflex
            }
        }

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        if id(self) in memo:
            return cast(Angle, memo[id(self)])
        
        new_segment1 = deepcopy(self.segment1, memo)
        new_segment2 = deepcopy(self.segment2, memo)
        new_angle: Angle = Angle(
            new_segment1,
            new_segment2,
            color=self.color,
            is_reflex=self.is_reflex
        )
        memo[id(self)] = new_angle
        return new_angle

    def update_points_based_on_segments(self) -> bool:
        """
        Re-evaluates vertex and arm points if segments might have changed.
        Then re-calculates the angle. Returns True if valid, False otherwise.
        """
        if not self._segments_form_angle(self.segment1, self.segment2):
            self.angle_degrees = None
            self.vertex_point = None
            self.arm1_point = None
            self.arm2_point = None
            return False

        # Re-extract defining points as segments might have changed their internal point references
        self.vertex_point, self.arm1_point, self.arm2_point = self._extract_defining_points(self.segment1, self.segment2)
        
        self._initialize()
        return True 

    def reset(self) -> None:
        """Resets the angle to its initial state based on its segments."""
        # The Drawable base class reset calls _initialize.
        # update_points_based_on_segments also calls _initialize and ensures points are current.
        self.update_points_based_on_segments() 

    def update_color(self, color: str) -> None:
        """Update the visual color metadata for the angle."""
        self.color = color