"""
MatHud Label Management System

Handles creation, retrieval, and deletion of label drawables. Labels store
math-space anchored annotations and participate in undo/redo flows like other
drawables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from constants import default_color, default_label_font_size
from drawables.label import Label
from utils.math_utils import MathUtils

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from name_generator.drawable import DrawableNameGenerator


class LabelManager:
    """Manages label drawables for a Canvas."""

    def __init__(
        self,
        canvas: "Canvas",
        drawables_container: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
        dependency_manager: "DrawableDependencyManager",
        drawable_manager_proxy: "DrawableManagerProxy",
    ) -> None:
        self.canvas = canvas
        self.drawables = drawables_container
        self.name_generator = name_generator
        self.dependency_manager = dependency_manager
        self.drawable_manager = drawable_manager_proxy

    def get_label_by_name(self, name: str) -> Optional[Label]:
        if not name:
            return None
        for label in self.drawables.Labels:
            if label.name == name:
                return label
        return None

    def get_labels_at_position(self, x: float, y: float) -> List[Label]:
        matches: List[Label] = []
        for label in self.drawables.Labels:
            if MathUtils.point_matches_coordinates(label.position, x, y):
                matches.append(label)
        return matches

    def create_label(
        self,
        x: float,
        y: float,
        text: str,
        *,
        name: str = "",
        color: Optional[str] = None,
        font_size: Optional[float] = None,
        rotation_degrees: Optional[float] = None,
    ) -> Label:
        self.canvas.undo_redo_manager.archive()
        sanitized_name = name.strip() if isinstance(name, str) else ""
        resolved_color = str(color) if color else default_color
        resolved_font_size = float(font_size) if font_size is not None else float(default_label_font_size)
        label = Label(
            x,
            y,
            text,
            name=sanitized_name,
            color=resolved_color,
            font_size=resolved_font_size,
            rotation_degrees=rotation_degrees,
        )
        label.canvas = self.canvas
        self.drawables.add(label)
        if self.canvas.draw_enabled:
            self.canvas.draw()
        return label

    def delete_label(self, name: str) -> bool:
        label = self.get_label_by_name(name)
        if label is None:
            return False
        self.canvas.undo_redo_manager.archive()
        removed = self.drawables.remove(label)
        if removed and self.canvas.draw_enabled:
            self.canvas.draw()
        return bool(removed)

