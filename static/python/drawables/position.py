"""
MatHud Geometric Position Container

Fundamental coordinate representation for all mathematical objects in the canvas system.
Provides simple x, y coordinate storage with state serialization capabilities.

Key Features:
    - Immutable coordinate pair storage
    - State serialization for persistence and undo/redo
    - String representation for debugging and display

Dependencies:
    - None (pure data container)
"""

class Position:
    """Represents a 2D coordinate position in the mathematical coordinate system.
    
    Fundamental building block for all geometric objects, providing x,y coordinate
    storage with serialization capabilities for state management.
    
    Attributes:
        x (float): X-coordinate in the mathematical coordinate system
        y (float): Y-coordinate in the mathematical coordinate system
    """
    def __init__(self, x, y):
        """Initialize a position with x and y coordinates.
        
        Args:
            x (float): X-coordinate in the mathematical coordinate system
            y (float): Y-coordinate in the mathematical coordinate system
        """
        self.x = x
        self.y = y

    def __str__(self):
        return f'Position: {self.x}, {self.y}'
    
    def get_state(self):
        state = {"Position": {"x": self.x, "y": self.y}}
        return state 