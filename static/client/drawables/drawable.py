"""
MatHud Base Drawable Class

Abstract base class for all mathematical objects in math space.
Defines the core interface for geometric objects including state management and transformations.

Key Features:
    - Color and naming system
    - Canvas-agnostic: no view or rendering dependencies
    - State serialization for persistence
    - Abstract interface for drawing and transformations

Core Interface:
    - draw(): No-op in math models; rendering handled by renderer
    - get_state(): Serialize object state for persistence
    - rotate(): Apply rotation transformation

Dependencies:
    - constants: Default styling values
"""

from constants import default_color


class Drawable:
    """Abstract base class for math-space geometric objects.
    
    Provides the fundamental interface and common functionality for geometric objects,
    including state serialization and transformation hooks. Rendering is handled by
    pluggable renderers and is not part of this class.
    
    Attributes:
        name (str): Identifier for the object
        color (str): Color metadata (used by renderers)
    """
    def __init__(self, name="", color=default_color):
        """Initialize a drawable object with basic properties.
        
        Args:
            name (str): Identifier for the object
            color (str): Color metadata for renderers
        """
        self.name = name
        self.color = color
    
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

    def get_class_name(self):
        raise NotImplementedError("Subclasses must implement class_name method")
    
    def get_name(self):
        return self.name
    
    def reset(self):
        # No-op: legacy initializer removed
        return None
    
    def get_state(self):
        raise NotImplementedError("Subclasses must implement get_state method")
    
    def rotate(self, angle):
        raise NotImplementedError("Subclasses must implement rotate method")