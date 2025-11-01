"""
Rendering interfaces for MatHud.

Defines a minimal renderer interface used by the Canvas to delegate all
rendering operations. Concrete implementations (e.g., SVG) are responsible
for translating math-space models into on-screen graphics using a provided
CoordinateMapper.

Notes:
- Renderers should avoid importing drawable classes to keep loose coupling.
- Dispatch is expected to be registry-based: the renderer maps model types
  (or class names) to handler functions.
"""

from __future__ import annotations

from typing import Any


class Renderer:
    """Abstract renderer interface.

    Concrete implementations should override the methods below.
    """

    def clear(self) -> None:
        """Clear the drawing surface for a new frame."""
        raise NotImplementedError("Renderer.clear must be implemented by subclasses")

    def render(self, drawable: Any, coordinate_mapper: Any) -> bool:
        """Render a single drawable using the provided CoordinateMapper.

        Args:
            drawable: A math-space model instance (Point, Segment, etc.)
            coordinate_mapper: The CoordinateMapper instance to convert math
                coordinates to screen coordinates
            
        Returns:
            bool: True if the drawable was rendered, False otherwise
        """
        raise NotImplementedError("Renderer.render must be implemented by subclasses")


