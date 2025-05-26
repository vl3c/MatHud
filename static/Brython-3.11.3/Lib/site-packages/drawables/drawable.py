"""
MatHud Base Drawable Class

Abstract base class for all mathematical objects that can be visualized on the canvas.
Defines the core interface for geometric objects including drawing, state management, and transformations.

Key Features:
    - SVG element creation and management
    - Color and naming system
    - Canvas integration for coordinate transformations
    - State serialization for persistence
    - Abstract interface for zoom, pan, and draw operations

Core Interface:
    - draw(): Render the object to SVG canvas
    - zoom(): Update object for scale factor changes
    - pan(): Update object for viewport translation
    - get_state(): Serialize object state for persistence
    - rotate(): Apply rotation transformation

Dependencies:
    - browser: DOM manipulation for SVG rendering
    - constants: Default styling values
"""

from browser import document, svg
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
    
    def zoom(self):
        raise NotImplementedError("Subclasses must implement zoom method")

    def pan(self):
        raise NotImplementedError("Subclasses must implement pan method")

    def draw(self):
        raise NotImplementedError("Subclasses must implement draw method")

    def reset(self):
        self._initialize()

    def create_svg_element(self, element_name, **attributes):
        # If text_content is present in attributes, store it separately and remove from attributes
        text_content = attributes.pop('text_content', None)
        text_font_size = attributes.pop('text_font_size', None)
        # If there was a font size, add it to the attributes
        if text_font_size is not None:
            attributes['font-size'] = f'{text_font_size}px'
        
        svg_element = getattr(svg, element_name)(**attributes)
        # If there was text content, add it to the svg element
        if text_content is not None:
            svg_element.text = text_content
            # Add a style to prevent text selection
            svg_element.style['user-select'] = 'none'
            svg_element.style['-webkit-user-select'] = 'none'
            svg_element.style['-moz-user-select'] = 'none'
            svg_element.style['-ms-user-select'] = 'none'
        document["math-svg"] <= svg_element
             
        return svg_element
    
    def get_state(self):
        raise NotImplementedError("Subclasses must implement get_state method")
    
    def rotate(self, angle):
        raise NotImplementedError("Subclasses must implement rotate method")