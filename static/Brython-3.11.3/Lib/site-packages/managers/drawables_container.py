"""
MatHud Drawable Storage and Organization System

Centralized container for all drawable objects with type-based organization and efficient access patterns.
Provides clean separation of storage concerns from Canvas operations and supports layered rendering.

Storage Architecture:
    - Type-Based Organization: Groups drawables by class name for efficient access
    - Property Access: Convenient attribute-style access to drawable collections
    - Dictionary Interface: Supports both object-oriented and dictionary-style access
    - Layered Storage: Separates colored areas from geometric objects for proper z-ordering

Supported Drawable Types:
    - Points: Coordinate-based geometric primitives
    - Segments: Line segments connecting two points
    - Vectors: Directed line segments with origin and tip
    - Triangles: Three-sided polygons with vertex tracking
    - Rectangles: Four-sided polygons with diagonal point definition
    - Circles: Circular objects with center and radius
    - Ellipses: Elliptical objects with center, radii, and rotation
    - Functions: Mathematical function plots and curves
    - Angles: Angular measurements between line segments
    - ColoredAreas: Various bounded colored regions

Rendering Support:
    - Z-Order Management: Colored areas rendered behind geometric objects
    - Layered Access: get_all_with_layering() provides proper rendering order
    - Background/Foreground Separation: Efficient separation for rendering pipeline

State Management:
    - State Serialization: get_state() for undo/redo functionality
    - Clear Operations: Bulk removal for canvas reset
    - Container Introspection: Type checking and content validation

Access Patterns:
    - Property Style: container.Points, container.Segments
    - Dictionary Style: container['Point'], container['Segment']
    - Bulk Operations: get_all(), get_colored_areas(), get_non_colored_areas()
"""

class DrawablesContainer:
    """
    A container for storing and accessing drawable objects by their class names.
    
    This class extracts the drawable storage functionality from Canvas,
    providing a cleaner separation of concerns.
    """
    
    def __init__(self):
        """Initialize an empty drawables container."""
        self._drawables = {}
        
    def add(self, drawable):
        """
        Add a drawable to the container.
        
        Args:
            drawable: The drawable object to add
        """
        category = drawable.get_class_name()
        if category not in self._drawables:
            self._drawables[category] = []
        self._drawables[category].append(drawable)
        
    def remove(self, drawable):
        """
        Remove a drawable from the container.
        
        Args:
            drawable: The drawable object to remove
            
        Returns:
            bool: True if the drawable was removed, False otherwise
        """
        category = drawable.get_class_name()
        if category in self._drawables and drawable in self._drawables[category]:
            self._drawables[category].remove(drawable)
            return True
        return False
        
    def get_by_class_name(self, class_name):
        """
        Get all drawables of a specific class name (private method).
        
        Args:
            class_name: The name of the class to get drawables for
            
        Returns:
            list: List of drawables of the specified class
        """
        return self._drawables.get(class_name, [])
        
    def get_all(self):
        """
        Get all drawables as a flat list.
        
        Returns:
            list: All drawables in the container
        """
        all_drawables = []
        for drawable_type in self._drawables:
            all_drawables.extend(self._drawables[drawable_type])
        return all_drawables
    
    def get_colored_areas(self):
        """
        Get all colored area drawables (for background rendering).
        
        Returns:
            list: All colored area drawables in the container
        """
        colored_areas = []
        for drawable_type in self._drawables:
            if 'ColoredArea' in drawable_type:
                colored_areas.extend(self._drawables[drawable_type])
        return colored_areas
    
    def get_non_colored_areas(self):
        """
        Get all non-colored area drawables (for foreground rendering).
        
        Returns:
            list: All non-colored area drawables in the container
        """
        other_drawables = []
        for drawable_type in self._drawables:
            if 'ColoredArea' not in drawable_type:
                other_drawables.extend(self._drawables[drawable_type])
        return other_drawables
    
    def get_all_with_layering(self):
        """
        Get all drawables with proper layering (colored areas first, then others).
        
        Returns:
            list: All drawables with colored areas first for proper z-ordering
        """
        return self.get_colored_areas() + self.get_non_colored_areas()
        
    def clear(self):
        """Remove all drawables from the container."""
        self._drawables.clear()
        
    def get_state(self):
        """
        Get the state of all drawables in the container.
        
        Returns:
            dict: Dictionary of drawable states by class name
        """
        state_dict = {}
        for category, drawables in self._drawables.items():
            state_dict[category + 's'] = [drawable.get_state() for drawable in drawables]
        return state_dict
        
    # Property-style access for specific drawable types (for convenience)
    @property
    def Points(self):
        """Get all Point objects."""
        return self.get_by_class_name('Point')
        
    @property
    def Segments(self):
        """Get all Segment objects."""
        return self.get_by_class_name('Segment')
        
    @property
    def Vectors(self):
        """Get all Vector objects."""
        return self.get_by_class_name('Vector')
        
    @property
    def Triangles(self):
        """Get all Triangle objects."""
        return self.get_by_class_name('Triangle')
        
    @property
    def Rectangles(self):
        """Get all Rectangle objects."""
        return self.get_by_class_name('Rectangle')
        
    @property
    def Circles(self):
        """Get all Circle objects."""
        return self.get_by_class_name('Circle')
        
    @property
    def Ellipses(self):
        """Get all Ellipse objects."""
        return self.get_by_class_name('Ellipse')
        
    @property
    def Functions(self):
        """Get all Function objects."""
        return self.get_by_class_name('Function')
        
    @property
    def ColoredAreas(self):
        """Get all ColoredArea objects."""
        return self.get_by_class_name('ColoredArea')
        
    @property
    def FunctionsBoundedColoredAreas(self):
        """Get all FunctionsBoundedColoredArea objects."""
        return self.get_by_class_name('FunctionsBoundedColoredArea')
        
    @property
    def Angles(self):
        """Get all Angle objects."""
        return self.get_by_class_name('Angle')
        
    @property
    def SegmentsBoundedColoredAreas(self):
        """Get all SegmentsBoundedColoredArea objects."""
        return self.get_by_class_name('SegmentsBoundedColoredArea')
        
    @property
    def FunctionSegmentBoundedColoredAreas(self):
        """Get all FunctionSegmentBoundedColoredArea objects."""
        return self.get_by_class_name('FunctionSegmentBoundedColoredArea')
        
    # Direct dictionary-like access
    def __getitem__(self, key):
        """
        Allow dictionary-like access to drawable types.
        
        Args:
            key: The class name
            
        Returns:
            list: List of drawables of the specified class
        """
        return self.get_by_class_name(key)
        
    def __contains__(self, key):
        """
        Check if the container has drawables of a specific class.
        
        Args:
            key: The class name
            
        Returns:
            bool: True if drawables of the specified class exist
        """
        return key in self._drawables 