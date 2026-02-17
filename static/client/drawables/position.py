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

from __future__ import annotations

from typing import Dict, Any


class Position:
    """Represents a 2D coordinate position in the mathematical coordinate system.

    Fundamental building block for all geometric objects, providing x,y coordinate
    storage with serialization capabilities for state management.

    Attributes:
        x (float): X-coordinate in the mathematical coordinate system
        y (float): Y-coordinate in the mathematical coordinate system
    """

    def __init__(self, x: float, y: float) -> None:
        """Initialize a position with x and y coordinates.

        Args:
            x (float): X-coordinate in the mathematical coordinate system
            y (float): Y-coordinate in the mathematical coordinate system
        """
        self.x: float = x
        self.y: float = y

    def __str__(self) -> str:
        return f"Position: {self.x}, {self.y}"

    def get_state(self) -> Dict[str, Dict[str, float]]:
        state: Dict[str, Dict[str, float]] = {"Position": {"x": self.x, "y": self.y}}
        return state
