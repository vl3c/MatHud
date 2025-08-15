"""
MatHud Colored Area Base Class

Abstract base class for all colored area visualizations between geometric objects.
Defines math-only properties and state; rendering is handled by renderer modules.

Key Features:
    - Math-space parameters for bounded areas
    - Color and opacity metadata (consumed by renderer)
    - Base state management for all area types

Area Types Supported:
    - Functions bounded areas (between two functions)
    - Segment bounded areas (between segments and axes)
    - Function-segment bounded areas (between function and segment)

Dependencies:
    - drawables.drawable: Base class interface
"""

from drawables.drawable import Drawable

class ColoredArea(Drawable):
    """Abstract base class for all colored area visualizations between geometric objects.
    
    Provides the foundation for area fill operations with SVG path generation
    and common styling capabilities for opacity and color.
    
    Attributes:
        opacity (float): Fill opacity value between 0.0 and 1.0
        color (str): CSS color value for area fill
    """
    def __init__(self, name, color="lightblue", opacity=0.3):
        """Initialize a colored area with basic properties.
        
        Args:
            name (str): Unique identifier for the colored area
            canvas (Canvas): Parent canvas for rendering
            color (str): CSS color value for area fill
            opacity (float): Fill opacity between 0.0 and 1.0
        """
        super().__init__(name=name, color=color)
        self.opacity = opacity

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