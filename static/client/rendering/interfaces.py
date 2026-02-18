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

from typing import Any, Callable, Protocol


class RendererProtocol(Protocol):
    """Minimal renderer contract consumed by the canvas."""

    def clear(self) -> None: ...

    def render(self, drawable: Any, coordinate_mapper: Any) -> bool: ...

    def render_cartesian(self, cartesian: Any, coordinate_mapper: Any) -> None: ...

    def render_polar(self, polar_grid: Any, coordinate_mapper: Any) -> None: ...

    def register(self, cls: type, handler: Callable[[Any, Any], None]) -> None: ...

    def register_default_drawables(self) -> None: ...

    def begin_frame(self) -> None: ...

    def end_frame(self) -> None: ...
