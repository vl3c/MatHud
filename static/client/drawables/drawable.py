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

from __future__ import annotations

from typing import Any, Dict

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
    def __init__(self, name: str = "", color: str = default_color, *, is_renderable: bool = True) -> None:
        """Initialize a drawable object with basic properties.

        Args:
            name (str): Identifier for the object
            color (str): Color metadata for renderers
            is_renderable (bool): Indicates if the object should be rendered directly
        """
        self._name: str = name
        self._color: str = color
        self._is_renderable: bool = bool(is_renderable)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, value: str) -> None:
        self._color = value

    @property
    def is_renderable(self) -> bool:
        return self._is_renderable

    @is_renderable.setter
    def is_renderable(self, value: bool) -> None:
        self._is_renderable = bool(value)

    def get_class_name(self) -> str:
        raise NotImplementedError("Subclasses must implement class_name method")

    def get_name(self) -> str:
        return self.name

    def reset(self) -> None:
        # No-op: legacy initializer removed
        return None

    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement get_state method")

    def rotate(self, angle: float) -> None:
        raise NotImplementedError("Subclasses must implement rotate method")
