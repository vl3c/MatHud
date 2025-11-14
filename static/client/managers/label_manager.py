"""
MatHud Label Management System

Handles creation, retrieval, and deletion of label drawables. Labels store
math-space anchored annotations and participate in undo/redo flows like other
drawables.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, cast

from constants import default_color, default_label_font_size
from drawables.label import Label
from utils.math_utils import MathUtils
from managers.edit_policy import DrawableEditPolicy, EditRule, get_drawable_edit_policy

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
        self.label_edit_policy: Optional[DrawableEditPolicy] = get_drawable_edit_policy("Label")

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
        mapper = getattr(self.canvas, "coordinate_mapper", None)
        scale_factor = getattr(mapper, "scale_factor", None) if mapper is not None else None
        label.update_reference_scale(scale_factor)
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

    def update_label(
        self,
        label_name: str,
        new_text: Optional[str] = None,
        new_x: Optional[float] = None,
        new_y: Optional[float] = None,
        new_color: Optional[str] = None,
        new_font_size: Optional[float] = None,
        new_rotation_degrees: Optional[float] = None,
    ) -> bool:
        label = self.get_label_by_name(label_name)
        if not label:
            raise ValueError(f"Label '{label_name}' was not found.")

        pending_fields = self._collect_label_requested_fields(
            new_text,
            new_color,
            new_x,
            new_y,
            new_font_size,
            new_rotation_degrees,
        )
        self._validate_label_policy(list(pending_fields.keys()))

        normalized_text = self._normalize_label_text(pending_fields, new_text)
        new_coordinates = self._compute_label_coordinates(pending_fields, new_x, new_y)
        sanitized_color = self._normalize_color_value(pending_fields, new_color)
        normalized_font_size = self._normalize_font_size(pending_fields, new_font_size)
        normalized_rotation = self._normalize_rotation(pending_fields, new_rotation_degrees)

        self.canvas.undo_redo_manager.archive()

        if normalized_text is not None:
            label.update_text(normalized_text)

        if new_coordinates is not None:
            label.update_position(*new_coordinates)

        if sanitized_color is not None:
            label.update_color(sanitized_color)

        if normalized_font_size is not None:
            label.update_font_size(normalized_font_size)

        if normalized_rotation is not None:
            label.update_rotation(normalized_rotation)

        if self.canvas.draw_enabled:
            self.canvas.draw()

        return True

    def _collect_label_requested_fields(
        self,
        new_text: Optional[str],
        new_color: Optional[str],
        new_x: Optional[float],
        new_y: Optional[float],
        new_font_size: Optional[float],
        new_rotation_degrees: Optional[float],
    ) -> Dict[str, str]:
        pending_fields: Dict[str, str] = {}

        if new_text is not None:
            pending_fields["text"] = "text"
        if new_color is not None:
            pending_fields["color"] = "color"
        if new_x is not None or new_y is not None:
            if new_x is None or new_y is None:
                raise ValueError("Updating a label position requires both x and y coordinates.")
            pending_fields["position"] = "position"
        if new_font_size is not None:
            pending_fields["font_size"] = "font_size"
        if new_rotation_degrees is not None:
            pending_fields["rotation"] = "rotation"

        if not pending_fields:
            raise ValueError("Provide at least one property to update.")

        return pending_fields

    def _validate_label_policy(self, requested_fields: List[str]) -> Dict[str, EditRule]:
        if not self.label_edit_policy:
            raise ValueError("Edit policy for labels is not configured.")

        validated_rules: Dict[str, EditRule] = {}
        for field in requested_fields:
            rule = self.label_edit_policy.get_rule(field)
            if not rule:
                raise ValueError(f"Editing field '{field}' is not permitted for labels.")
            validated_rules[field] = rule
        return validated_rules

    def _normalize_label_text(
        self,
        pending_fields: Dict[str, str],
        new_text: Optional[str],
    ) -> Optional[str]:
        if "text" not in pending_fields:
            return None
        return Label.validate_text("" if new_text is None else new_text)

    def _compute_label_coordinates(
        self,
        pending_fields: Dict[str, str],
        new_x: Optional[float],
        new_y: Optional[float],
    ) -> Optional[tuple[float, float]]:
        if "position" not in pending_fields:
            return None
        return (float(cast(float, new_x)), float(cast(float, new_y)))

    def _normalize_color_value(
        self,
        pending_fields: Dict[str, str],
        new_color: Optional[str],
    ) -> Optional[str]:
        if "color" not in pending_fields:
            return None
        sanitized = str(new_color).strip() if new_color is not None else ""
        if not sanitized:
            raise ValueError("Label color cannot be empty.")
        return sanitized

    def _normalize_font_size(
        self,
        pending_fields: Dict[str, str],
        new_font_size: Optional[float],
    ) -> Optional[float]:
        if "font_size" not in pending_fields:
            return None
        numeric = float(cast(float, new_font_size))
        if numeric <= 0:
            raise ValueError("Label font size must be positive.")
        return numeric

    def _normalize_rotation(
        self,
        pending_fields: Dict[str, str],
        new_rotation_degrees: Optional[float],
    ) -> Optional[float]:
        if "rotation" not in pending_fields:
            return None
        numeric = float(cast(float, new_rotation_degrees))
        if not math.isfinite(numeric):
            raise ValueError("Label rotation must be a finite number.")
        return numeric

