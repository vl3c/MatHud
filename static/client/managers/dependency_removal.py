"""Shared helpers for dependency-aware drawable removal.

This module centralizes common patterns used across managers:
remove a drawable from the drawables container and, if successful, also
remove its dependency-graph entries. Keeping this logic in one place helps
prevent stale dependency edges and preserves graph invariants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from drawables.drawable import Drawable
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager


def remove_drawable_with_dependencies(
    drawables: "DrawablesContainer",
    dependency_manager: "DrawableDependencyManager",
    drawable: "Drawable",
) -> bool:
    """Remove drawable from container and dependency graph in one place."""
    removed = drawables.remove(drawable)
    if removed and hasattr(dependency_manager, "remove_drawable"):
        dependency_manager.remove_drawable(drawable)
    return bool(removed)


def get_polygon_segments(polygon: Any) -> List[Any]:
    """Extract segments from any polygon type.

    Prefers ``get_segments()`` (used by Pentagon through GenericPolygon),
    falling back to named ``segment1``..``segment4`` attributes (used by
    Triangle, Rectangle, and Quadrilateral).
    """
    if hasattr(polygon, "get_segments") and callable(getattr(polygon, "get_segments")):
        return polygon.get_segments()
    segments: List[Any] = []
    for attr in ("segment1", "segment2", "segment3", "segment4"):
        seg = getattr(polygon, attr, None)
        if seg is not None:
            segments.append(seg)
    return segments
