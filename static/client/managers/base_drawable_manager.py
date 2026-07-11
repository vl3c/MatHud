"""
Base class for drawable managers.

Captures the common constructor pattern, attribute storage, edit-policy
lookup, and name-based retrieval shared by every specialized manager.
Subclasses supply the drawable-type name and any additional constructor
parameters they need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from managers.edit_policy import DrawableEditPolicy, get_drawable_edit_policy

if TYPE_CHECKING:
    from drawables.drawable import Drawable
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from name_generator.drawable import DrawableNameGenerator


class BaseDrawableManager:
    """Shared foundation for specialized drawable managers.

    Stores the five common dependencies every manager receives and
    automatically resolves the edit policy for the declared drawable type.

    Subclasses must set ``drawable_type`` (a class attribute) to the
    drawable class name string (e.g. ``"Point"``, ``"Label"``).
    """

    drawable_type: str = ""
    """Override in subclasses with the drawable class name (e.g. ``"Point"``)."""

    def __init__(
        self,
        canvas: "Canvas",
        drawables_container: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        drawable_manager_proxy: "DrawableManagerProxy",
    ) -> None:
        self.canvas: "Canvas" = canvas
        self.drawables: "DrawablesContainer" = drawables_container
        self.name_generator: "DrawableNameGenerator" = name_generator
        self.dependency_manager: "DrawableDependencyManager" = dependency_manager
        self.drawable_manager: "DrawableManagerProxy" = drawable_manager_proxy
        self.edit_policy: Optional[DrawableEditPolicy] = (
            get_drawable_edit_policy(self.drawable_type) if self.drawable_type else None
        )

    # ------------------------------------------------------------------
    # Common lookup
    # ------------------------------------------------------------------

    def _get_by_name(self, name: str) -> Optional["Drawable"]:
        """Look up a drawable by *name* inside the container.

        Iterates over drawables whose class name matches ``drawable_type``.
        Returns ``None`` when *name* is empty or no match is found.
        """
        if not name:
            return None
        for drawable in self.drawables.get_by_class_name(self.drawable_type):
            if getattr(drawable, "name", None) == name:
                return drawable
        return None
