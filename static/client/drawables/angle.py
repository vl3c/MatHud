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

from constants import point_label_font_size, \
    DEFAULT_ANGLE_COLOR, DEFAULT_ANGLE_ARC_SCREEN_RADIUS, DEFAULT_ANGLE_TEXT_ARC_RADIUS_FACTOR
from drawables.drawable import Drawable
import utils.math_utils as math_utils
from copy import deepcopy
import math

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
    def __init__(self, segment1, segment2, canvas, color=DEFAULT_ANGLE_COLOR, is_reflex: bool = False):
        """Initialize an angle from two intersecting line segments.
        
        Validates segment intersection and extracts angle properties for visualization.
        
        Args:
            segment1 (Segment): First segment forming the angle
            segment2 (Segment): Second segment forming the angle
            canvas (Canvas): Parent canvas for coordinate transformations
            color (str): CSS color value for angle visualization
            is_reflex (bool): False for inner/standard angle, True for outer/reflex angle
            
        Raises:
            ValueError: If segments do not form a valid angle (must share exactly one vertex)
        """
        if not self._segments_form_angle(segment1, segment2):
            raise ValueError("The segments do not form a valid angle (must share exactly one vertex and have distinct arms).")

        self.segment1 = segment1
        self.segment2 = segment2
        self.is_reflex = is_reflex
        
        self.vertex_point, self.arm1_point, self.arm2_point = self._extract_defining_points(self.segment1, self.segment2)

        # Name generation might need adjustment later if manager needs to distinguish reflex/non-reflex for same segments
        name_suffix = "_reflex" if self.is_reflex else ""
        base_name = canvas.drawable_manager.name_generator.generate_angle_name_from_segments(
            segment1.name, segment2.name
        )
        name = f"{base_name}{name_suffix}"
        
        super().__init__(name=name, color=color, canvas=canvas) 
        
        self.raw_angle_degrees = None # To store the fundamental CCW angle (0-360)
        self.angle_degrees = None     # To store the display angle (small or reflex)
        
        self._initialize()

    def _get_common_vertex(self, s1, s2):
        """Identifies and returns the common vertex point object between two segments."""
        if s1.point1 == s2.point1 or s1.point1 == s2.point2:
            return s1.point1
        if s1.point2 == s2.point1 or s1.point2 == s2.point2:
            return s1.point2
        return None

    def _segments_form_angle(self, s1, s2):
        """
        Validates if two segments can form an angle.
        They must share exactly one common point (vertex) and form distinct, non-degenerate arms.
        """
        if not s1 or not s2: return False # Segments must exist
        if not hasattr(s1, 'point1') or not hasattr(s1, 'point2') or \
           not hasattr(s2, 'point1') or not hasattr(s2, 'point2'):
            return False # Segments must have point attributes

        common_vertex_point_obj = self._get_common_vertex(s1, s2) # This returns a Point object or None
        if common_vertex_point_obj is None:
            return False # Segments do not share exactly one common vertex

        # Identify the Point objects for the arms
        arm1_point_obj = s1.point1 if s1.point1 != common_vertex_point_obj else s1.point2
        arm2_point_obj = s2.point1 if s2.point1 != common_vertex_point_obj else s2.point2
        
        return math_utils.MathUtils.are_points_valid_for_angle_geometry(
            (common_vertex_point_obj.x, common_vertex_point_obj.y),
            (arm1_point_obj.x, arm1_point_obj.y),
            (arm2_point_obj.x, arm2_point_obj.y)
        )

    def _extract_defining_points(self, s1, s2):
        """Extracts vertex, arm1, and arm2 points. Assumes segments form a valid angle."""
        vertex_p = self._get_common_vertex(s1, s2)
        # _segments_form_angle should have already ensured vertex_p is not None
        arm1_p = s1.point1 if s1.point1 != vertex_p else s1.point2
        arm2_p = s2.point1 if s2.point1 != vertex_p else s2.point2
        return vertex_p, arm1_p, arm2_p

    def _calculate_display_angle(self, raw_angle_degrees, is_reflex, epsilon):
        """Helper function to calculate the display angle based on raw angle and reflex state."""
        if raw_angle_degrees is None:
            return None

        display_angle = None
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

    def _initialize(self):
        """Calculates raw_angle_degrees and angle_degrees based on geometry and is_reflex state."""
        if not (self.vertex_point and self.arm1_point and self.arm2_point):
            self.raw_angle_degrees = None
            self.angle_degrees = None
            return

        # Use screen coordinates when available; fall back to x/y for test mocks
        def _pt_screen_xy(pt):
            sx = getattr(pt, 'screen_x', None)
            sy = getattr(pt, 'screen_y', None)
            if sx is None or sy is None:
                return (pt.x, pt.y)
            return (sx, sy)

        vertex_coords = _pt_screen_xy(self.vertex_point)
        arm1_coords = _pt_screen_xy(self.arm1_point)
        arm2_coords = _pt_screen_xy(self.arm2_point)

        # Calculate the fundamental CCW angle from arm1 to arm2 (0-360 degrees)
        self.raw_angle_degrees = math_utils.MathUtils.calculate_angle_degrees(
            vertex_coords, arm1_coords, arm2_coords
        )
        
        self.angle_degrees = self._calculate_display_angle(self.raw_angle_degrees, self.is_reflex, math_utils.MathUtils.EPSILON)
        
        # Arc radius comes from renderer (or default constant when not provided)

    @property
    def canvas(self): 
        return self._canvas

    @canvas.setter
    def canvas(self, value):
        self._canvas = value
        if hasattr(self.segment1, 'canvas'): self.segment1.canvas = value
        if hasattr(self.segment2, 'canvas'): self.segment2.canvas = value

    def get_class_name(self):
        return 'Angle'

    def _get_drawing_references(self):
        """Returns screen coordinates of vertex, arm1 point, and arm2 point, or None if any are missing."""
        if not (self.vertex_point and self.arm1_point and self.arm2_point):
            return None
        def _pt_screen_xy(pt):
            sx = getattr(pt, 'screen_x', None)
            sy = getattr(pt, 'screen_y', None)
            if sx is None or sy is None:
                return (pt.x, pt.y)
            return (sx, sy)
        vx, vy = _pt_screen_xy(self.vertex_point)
        a1x, a1y = _pt_screen_xy(self.arm1_point)
        a2x, a2y = _pt_screen_xy(self.arm2_point)
        return (vx, vy, a1x, a1y, a2x, a2y)

    def _calculate_arc_parameters(self, vx, vy, p1x, p1y, p2x, p2y, arc_radius=None):
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
        
        current_arc_radius = arc_radius if arc_radius is not None else DEFAULT_ANGLE_ARC_SCREEN_RADIUS
        if current_arc_radius <= 0: 
             return None
        
        epsilon = math_utils.MathUtils.EPSILON # Use MathUtils.EPSILON

        # Calculate geometric angles of the arms relative to the positive x-axis
        angle_v_p1_rad = math.atan2(p1y - vy, p1x - vx) # Radians angle of arm1 vector
        angle_v_p2_rad = math.atan2(p2y - vy, p2x - vx) # Radians angle of arm2 vector

        # Calculate arc start and end points on screen
        arc_start_x = vx + current_arc_radius * math.cos(angle_v_p1_rad)
        arc_start_y = vy + current_arc_radius * math.sin(angle_v_p1_rad)
        arc_end_x = vx + current_arc_radius * math.cos(angle_v_p2_rad)
        arc_end_y = vy + current_arc_radius * math.sin(angle_v_p2_rad)
        
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

    def _get_arc_flags(self, display_angle_degrees, raw_angle_degrees, epsilon):
        """Determines sweep and large-arc flags for drawing the angle arc."""
        is_effectively_raw_angle = False
        diff_abs = abs(display_angle_degrees - raw_angle_degrees)

        if diff_abs < epsilon or abs(diff_abs - 360.0) < epsilon:
            if abs(display_angle_degrees - 360.0) < epsilon and abs(raw_angle_degrees) < epsilon:
                is_effectively_raw_angle = False # Displaying 360 from raw 0 is CW path
            elif abs(display_angle_degrees) < epsilon and abs(raw_angle_degrees - 360.0) < epsilon:
                is_effectively_raw_angle = True  # Displaying 0 from raw 360 is CCW path (of 0)
            else:
                is_effectively_raw_angle = True
        else:
            is_effectively_raw_angle = False

        sweep_flag = '1' if is_effectively_raw_angle else '0'

        large_arc_flag = '0'
        if abs(display_angle_degrees) < epsilon:
            large_arc_flag = '0'
        elif abs(display_angle_degrees - 360.0) < epsilon:
            large_arc_flag = '1'
        elif display_angle_degrees > 180.0 + epsilon:
            large_arc_flag = '1'

        # Special handling for full circle drawing if start/end points are identical
        raw_is_zero_or_360 = abs(raw_angle_degrees) < epsilon or abs(raw_angle_degrees - 360.0) < epsilon
        if abs(display_angle_degrees - 360.0) < epsilon and raw_is_zero_or_360:
            large_arc_flag = '1'
            sweep_flag = '1' # Prefer CCW for canonical full circle SVG representation
        
        return sweep_flag, large_arc_flag

    # Removed SVG element creation helpers; rendering handled by the renderer

    def draw(self):
        # Rendering handled by renderer; no-op to preserve interface
        return None

    def get_state(self):
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

    @classmethod 
    def from_state(cls, state_data, canvas):
        """
        Creates an Angle instance from a state dictionary.
        """
        args = state_data["args"]
        segment1 = canvas.drawable_manager.get_segment_by_name(args["segment1_name"])
        segment2 = canvas.drawable_manager.get_segment_by_name(args["segment2_name"])

        if not segment1 or not segment2:
            return None 

        return cls(
            segment1=segment1, segment2=segment2, canvas=canvas,
            color=args.get("color", DEFAULT_ANGLE_COLOR),
            is_reflex=args.get("is_reflex", False)
        )

    def __deepcopy__(self, memo):
        if id(self) in memo: return memo[id(self)]
        
        new_angle = Angle(
            segment1=deepcopy(self.segment1, memo),
            segment2=deepcopy(self.segment2, memo),
            canvas=self.canvas, 
            color=self.color,
            is_reflex=self.is_reflex
        )
        memo[id(self)] = new_angle
        return new_angle

    def update_points_based_on_segments(self):
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
        
        self._initialize() # This recalculates angle_degrees and ensures drawn_arc_radius is set
        return True 

    def reset(self):
        """Resets the angle to its initial state based on its segments."""
        # The Drawable base class reset calls _initialize.
        # update_points_based_on_segments also calls _initialize and ensures points are current.
        self.update_points_based_on_segments() 