"""
MatHud Colored Area Base Class

Abstract base class for all colored area visualizations between geometric objects.
Provides SVG path creation and common area rendering functionality.

Key Features:
    - SVG path generation from boundary points
    - Color and opacity customization
    - Forward and reverse path construction for closed areas
    - Base state management for all area types

Area Types Supported:
    - Functions bounded areas (between two functions)
    - Segment bounded areas (between segments and axes)
    - Function-segment bounded areas (between function and segment)

Dependencies:
    - drawables.drawable: Base class interface
    - constants: Default styling values
"""

from drawables.drawable import Drawable
from constants import default_color

class ColoredArea(Drawable):
    """Abstract base class for all colored area visualizations between geometric objects.
    
    Provides the foundation for area fill operations with SVG path generation
    and common styling capabilities for opacity and color.
    
    Attributes:
        opacity (float): Fill opacity value between 0.0 and 1.0
        color (str): CSS color value for area fill
    """
    def __init__(self, name, canvas=None, color="lightblue", opacity=0.3):
        """Initialize a colored area with basic properties.
        
        Args:
            name (str): Unique identifier for the colored area
            canvas (Canvas): Parent canvas for rendering
            color (str): CSS color value for area fill
            opacity (float): Fill opacity between 0.0 and 1.0
        """
        super().__init__(name=name, color=color, canvas=canvas)
        self.opacity = opacity

    def draw(self):
        """
        Draw the colored area. Each subclass must implement its own draw method
        that calls _create_svg_path with appropriate points.
        """
        raise NotImplementedError("Subclasses must implement draw()")

    def _create_svg_path(self, forward_points, reverse_points):
        """
        Create an SVG path from the given points.
        forward_points: List of (x,y) tuples for the forward path
        reverse_points: List of (x,y) tuples for the reverse path
        """
        if not forward_points or not reverse_points:
            return

        # Create SVG path
        path_d = f"M {forward_points[0][0]},{forward_points[0][1]}"
        
        # Add forward path
        for x, y in forward_points[1:]:
            path_d += f" L {x},{y}"
            
        # Add reverse path
        for x, y in reverse_points:
            path_d += f" L {x},{y}"
            
        # Close path
        path_d += " Z"
        
        # Create SVG path element
        self.create_svg_element('path', d=path_d, stroke="none", 
                              fill=self.color, fill_opacity=str(self.opacity))

    def zoom(self):
        pass  # Zooming is handled by the objects defining the area

    def pan(self):
        pass  # Panning is handled by the objects defining the area

    def get_class_name(self):
        return 'ColoredArea'

    def get_state(self):
        """Base state that all colored areas share"""
        return {
            "name": self.name,
            "args": {
                "color": self.color,
                "opacity": self.opacity
            }
        }

    def __deepcopy__(self, memo):
        """
        Base deepcopy implementation. Subclasses should override this and call
        their own constructor with the appropriate arguments.
        """
        if id(self) in memo:
            return memo[id(self)]
        
        # This will be overridden by subclasses
        raise NotImplementedError("Subclasses must implement __deepcopy__") 