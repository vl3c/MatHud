"""
MatHud Base Drawable Class

Abstract base class for all mathematical objects that can be visualized on the canvas.
Defines the core interface for geometric objects including state management and transformations.

Key Features:
    - Color and naming system
    - Canvas integration for coordinate transformations (legacy; to be removed)
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
    """Abstract base class for all mathematical objects that can be visualized on the canvas.
    
    Provides the fundamental interface and common functionality for geometric objects,
    including SVG rendering capabilities, state management, and coordinate system integration.
    
    Attributes:
        name (str): Unique identifier for the object within the canvas
        color (str): CSS color value for object visualization
        canvas (Canvas): Reference to the parent canvas for coordinate transformations
    """
    def __init__(self, name="", color=default_color, canvas=None):
        """Initialize a drawable object with basic properties.
        
        Args:
            name (str): Unique identifier for the object
            color (str): CSS color value for rendering
            canvas (Canvas): Parent canvas for coordinate system integration
        """
        self.name = name
        self.color = color
        self.canvas = canvas
    
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

    @property
    def canvas(self):
        return self._canvas

    @canvas.setter
    def canvas(self, value):
        self._canvas = value

    def get_class_name(self):
        raise NotImplementedError("Subclasses must implement class_name method")
    
    def get_name(self):
        return self.name
    
    def draw(self):
        raise NotImplementedError("Subclasses must implement draw method")

    def reset(self):
        self._initialize()

    
    def get_state(self):
        raise NotImplementedError("Subclasses must implement get_state method")
    
    def rotate(self, angle):
        raise NotImplementedError("Subclasses must implement rotate method")