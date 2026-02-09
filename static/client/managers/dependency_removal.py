"""Shared helpers for dependency-aware drawable removal.

This module centralizes a common removal pattern used across managers:
remove a drawable from the drawables container and, if successful, also
remove its dependency-graph entries. Keeping this logic in one place helps
prevent stale dependency edges and preserves graph invariants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
