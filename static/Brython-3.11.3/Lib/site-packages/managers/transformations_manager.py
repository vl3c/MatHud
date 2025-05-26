"""
MatHud Geometric Transformations Management System

Handles geometric transformations of drawable objects including translation and rotation.
Provides coordinated transformation operations with proper state management and canvas integration.

Transformation Types:
    - Translation: Moving objects by specified x and y offsets
    - Rotation: Rotating objects around specified points or their centers

Operation Coordination:
    - State Archiving: Automatic undo/redo state capture before transformations
    - Object Validation: Ensures target objects exist before transformation
    - Method Delegation: Calls transformation methods on drawable objects
    - Canvas Integration: Automatic redrawing after successful transformations

Error Handling:
    - Object Existence Validation: Checks for drawable presence before operations
    - Transformation Validation: Ensures objects support required transformation methods
    - Exception Management: Graceful error handling with descriptive messages
    - State Consistency: Maintains proper canvas state even if transformations fail

Integration Points:
    - DrawableManager: Object lookup and validation
    - UndoRedoManager: State preservation for transformation operations
    - Canvas: Visual updates after transformations
    - Drawable Objects: Delegation to object-specific transformation methods
"""

class TransformationsManager:
    """Manages geometric transformations of drawable objects on a Canvas.
    
    Coordinates translation and rotation operations with proper state management,
    object validation, and canvas integration.
    """
    
    def __init__(self, canvas):
        """
        Initialize the TransformationsManager.
        
        Args:
            canvas: The Canvas object this manager is responsible for
        """
        self.canvas = canvas
    
    def translate_object(self, name, x_offset, y_offset):
        """
        Translates a drawable object by the specified offset.
        
        Args:
            name: Name of the drawable to translate
            x_offset: Horizontal offset to apply
            y_offset: Vertical offset to apply
            
        Returns:
            bool: True if the translation was successful
            
        Raises:
            ValueError: If no drawable with the given name is found
        """
        # Find the drawable first to validate it exists
        drawable = None
        for drawable in self.canvas.drawable_manager.get_drawables():
            if drawable.name == name:
                break
            
        if not drawable or drawable.name != name:
            raise ValueError(f"No drawable found with name '{name}'")
            
        # Archive current state for undo/redo AFTER finding the object but BEFORE modifying it
        self.canvas.undo_redo_manager.archive()
        
        # Apply translation using the drawable's translate method
        # (All drawable objects should implement this method)
        try:
            drawable.translate(x_offset, y_offset)
        except Exception as e:
            # Raise an error to be handled by the AI interface
            raise ValueError(f"Error translating drawable: {str(e)}")
            
        # If we got here, the translation was successful
        # Redraw the canvas
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return True
    
    def rotate_object(self, name, angle):
        """
        Rotates a drawable object by the specified angle.
        
        Args:
            name: Name of the drawable to rotate
            angle: Angle in degrees to rotate the object
            
        Returns:
            bool: True if the rotation was successful
            
        Raises:
            ValueError: If no drawable with the given name is found or if rotation fails
        """
        # Find the drawable first to validate it exists
        drawable = None
        # Get all drawables except Points, Functions, and Circles which don't support rotation
        for d in self.canvas.drawable_manager.get_drawables():
            if d.get_class_name() in ['Function', 'Point', 'Circle']:
                continue
            if d.name == name:
                drawable = d
                break
        
        if not drawable:
            raise ValueError(f"No drawable found with name '{name}'")
            
        # Archive current state for undo/redo AFTER finding the object but BEFORE modifying it
        self.canvas.undo_redo_manager.archive()
        
        # Apply rotation using the drawable's rotate method
        try:
            drawable.rotate(angle)
        except Exception as e:
            # Raise an error to be handled by the AI interface
            raise ValueError(f"Error rotating drawable: {str(e)}")
            
        # If we got here, the rotation was successful
        # Redraw the canvas
        if self.canvas.draw_enabled:
            self.canvas.draw()
            
        return True 