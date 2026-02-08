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

from __future__ import annotations

from typing import Any, Dict

from drawables.drawable import Drawable

class ColoredArea(Drawable):
    """Abstract base class for all colored area visualizations between geometric objects.

    Provides the foundation for area fill operations with SVG path generation
    and common styling capabilities for opacity and color.

    Attributes:
        opacity (float): Fill opacity value between 0.0 and 1.0
        color (str): CSS color value for area fill
    """
    def __init__(self, name: str, color: str = "lightblue", opacity: float = 0.3) -> None:
        """Initialize a colored area with basic properties.

        Args:
            name (str): Unique identifier for the colored area
            color (str): CSS color value for area fill
            opacity (float): Fill opacity between 0.0 and 1.0
        """
        super().__init__(name=name, color=color)
        self.opacity: float = opacity

    def get_class_name(self) -> str:
        return 'ColoredArea'

    def get_state(self) -> Dict[str, Any]:
        """Base state that all colored areas share"""
        return {
            "name": self.name,
            "args": {
                "color": self.color,
                "opacity": self.opacity
            }
        }

    def update_color(self, color: str) -> None:
        """Update the area fill color."""
        self.color = str(color)

    def update_opacity(self, opacity: float) -> None:
        """Update the area fill opacity."""
        self.opacity = float(opacity)

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        """
        Base deepcopy implementation. Subclasses should override this and call
        their own constructor with the appropriate arguments.
        """
        if id(self) in memo:
            return memo[id(self)]

        # This will be overridden by subclasses
        raise NotImplementedError("Subclasses must implement __deepcopy__")
