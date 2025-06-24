"""
MatHud Segments Bounded Colored Area

Represents a colored area bounded by line segments with overlap detection and linear interpolation.
Provides area visualization between two segments or between a segment and the x-axis.

Key Features:
    - Two-segment or segment-to-axis area visualization
    - Overlap region detection and calculation
    - Linear interpolation for boundary generation
    - Geometric validation and path optimization

Dependencies:
    - drawables.colored_area: Base class for area visualization
    - drawables.segment: Segment objects for boundary definitions
    - copy: Deep copying capabilities for state management
"""

from drawables.colored_area import ColoredArea
from drawables.segment import Segment
import copy

class SegmentsBoundedColoredArea(ColoredArea):
    """Creates a colored area bounded by line segments with geometric overlap detection.
    
    This class creates a visual representation of the area between two line segments
    or between a segment and the x-axis, using linear interpolation for smooth boundaries.
    
    Attributes:
        segment1 (Segment): The first bounding segment
        segment2 (Segment or None): The second bounding segment (None means x-axis)
    """

    def __init__(self, segment1, segment2=None, canvas=None, color="lightblue", opacity=0.3):
        """Initialize a segments bounded colored area.
        
        Args:
            segment1 (Segment): The first bounding segment
            segment2 (Segment or None): The second bounding segment (None means x-axis)
            canvas (Canvas): Parent canvas for coordinate system access
            color (str): CSS color value for area fill
            opacity (float): Opacity value between 0.0 and 1.0
        """
        name = self._generate_name(segment1, segment2)
        super().__init__(name=name, canvas=canvas, color=color, opacity=opacity)
        self.segment1 = segment1
        self.segment2 = segment2

    def _generate_name(self, segment1, segment2):
        """Generate a descriptive name for the colored area based on segment names."""
        s1_name = segment1.name if segment1 else 'x_axis'
        s2_name = segment2.name if segment2 else 'x_axis'
        return f"area_between_{s1_name}_and_{s2_name}"

    def get_class_name(self):
        """Return the class name 'SegmentsBoundedColoredArea'."""
        return 'SegmentsBoundedColoredArea'

    def draw(self):
        """Draw the colored area between the segments on the canvas."""
        if not self.segment2:
            # Handle segment-xaxis case as before
            points = [(self.segment1.point1.x, self.segment1.point1.y),
                     (self.segment1.point2.x, self.segment1.point2.y)]

            if self.segment1.point1.y > 0 and self.segment1.point2.y > 0:
                # Both points above x-axis, color below
                reverse_points = [(self.segment1.point2.x, self.canvas.cartesian2axis.origin.y),
                                (self.segment1.point1.x, self.canvas.cartesian2axis.origin.y)]
            else:
                # Both points below x-axis or crossing x-axis, color above
                reverse_points = [(self.segment1.point2.x, self.canvas.cartesian2axis.origin.y),
                                (self.segment1.point1.x, self.canvas.cartesian2axis.origin.y)]
        else:
            # Handle segment-segment case with vertical bounds
            # Get x-ranges of both segments
            x1_min = min(self.segment1.point1.x, self.segment1.point2.x)
            x1_max = max(self.segment1.point1.x, self.segment1.point2.x)
            x2_min = min(self.segment2.point1.x, self.segment2.point2.x)
            x2_max = max(self.segment2.point1.x, self.segment2.point2.x)

            # Find overlap range
            overlap_min = max(x1_min, x2_min)
            overlap_max = min(x1_max, x2_max)

            if overlap_max <= overlap_min:
                # No overlap, don't draw anything
                return

            def get_y_at_x(segment, x):
                # Linear interpolation to find y value at x
                x1, y1 = segment.point1.x, segment.point1.y
                x2, y2 = segment.point2.x, segment.point2.y
                if x2 == x1:
                    return y1  # Vertical segment
                t = (x - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)

            # Create path points for the overlapping region
            y1_start = get_y_at_x(self.segment1, overlap_min)
            y1_end = get_y_at_x(self.segment1, overlap_max)
            y2_start = get_y_at_x(self.segment2, overlap_min)
            y2_end = get_y_at_x(self.segment2, overlap_max)

            # Forward path along segment1
            points = [(overlap_min, y1_start),
                     (overlap_max, y1_end)]

            # Reverse path along segment2
            reverse_points = [(overlap_max, y2_end),
                            (overlap_min, y2_start)]

        # Create SVG path using base class method
        self._create_svg_path(points, reverse_points)

    def uses_segment(self, segment):
        """Check if this colored area uses a specific segment for dependency tracking."""
        def segments_match(s1, s2):
            return (s1.point1.x == s2.point1.x and 
                   s1.point1.y == s2.point1.y and
                   s1.point2.x == s2.point2.x and
                   s1.point2.y == s2.point2.y)

        return segments_match(self.segment1, segment) or (self.segment2 and segments_match(self.segment2, segment))

    def get_state(self):
        """Serialize segments bounded area state for persistence."""
        state = super().get_state()
        state["args"].update({
            "segment1": self.segment1.name,
            "segment2": self.segment2.name if self.segment2 else "x_axis"
        })
        return state

    def __deepcopy__(self, memo):
        """Create a deep copy for undo/redo functionality."""
        if id(self) in memo:
            return memo[id(self)]
            
        # Create new instance using __init__
        new_area = SegmentsBoundedColoredArea(
            segment1=self.segment1,  # Segments will be properly deep copied by their own __deepcopy__
            segment2=self.segment2,
            canvas=self.canvas,  # Canvas reference is not deep copied
            color=self.color,
            opacity=self.opacity
        )
        memo[id(self)] = new_area
        return new_area 