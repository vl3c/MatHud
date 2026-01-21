"""Bar manager for creating and managing bar chart elements.

This module provides the BarManager class which handles creation,
update, and deletion of Bar drawables used in bar charts and histograms.

Key Features:
    - Bar creation with configurable bounds and labels
    - Automatic name generation for unnamed bars
    - Update-in-place for existing bars with same name
    - Stroke color, fill color, and opacity configuration
    - Above and below label text positioning
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from drawables.bar import Bar

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawable_dependency_manager import DrawableDependencyManager
    from managers.drawable_manager_proxy import DrawableManagerProxy
    from managers.drawables_container import DrawablesContainer
    from name_generator.drawable import DrawableNameGenerator


class BarManager:
    """Create and delete Bar drawables."""

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

    def get_bar_by_name(self, name: str) -> Optional[Bar]:
        if not name:
            return None
        for bar in self.drawables.get_by_class_name("Bar"):
            if getattr(bar, "name", None) == name:
                return bar  # type: ignore[return-value]
        return None

    def create_bar(
        self,
        *,
        name: str = "",
        x_left: float,
        x_right: float,
        y_top: float,
        y_bottom: float = 0.0,
        stroke_color: Optional[str] = None,
        fill_color: Optional[str] = None,
        fill_opacity: Optional[float] = None,
        label_above_text: Optional[str] = None,
        label_below_text: Optional[str] = None,
        archive: bool = True,
        redraw: bool = True,
    ) -> Bar:
        if archive:
            try:
                self.canvas.archive()
            except Exception:
                pass

        sanitized_name = ""
        if isinstance(name, str):
            sanitized_name = self.name_generator.filter_string(name.strip())

        if sanitized_name:
            existing = self.get_bar_by_name(sanitized_name)
            if existing is not None:
                self._update_existing_bar(
                    existing,
                    x_left=x_left,
                    x_right=x_right,
                    y_bottom=y_bottom,
                    y_top=y_top,
                    stroke_color=stroke_color,
                    fill_color=fill_color,
                    fill_opacity=fill_opacity,
                    label_above_text=label_above_text,
                    label_below_text=label_below_text,
                )
                if redraw and getattr(self.canvas, "draw_enabled", False):
                    try:
                        self.canvas.draw()
                    except Exception:
                        pass
                return existing
            resolved_name = sanitized_name
        else:
            resolved_name = self._generate_unique_name("bar")

        x_left_f = float(x_left)
        x_right_f = float(x_right)
        y_bottom_f = float(y_bottom)
        y_top_f = float(y_top)
        if not math.isfinite(x_left_f) or not math.isfinite(x_right_f):
            raise ValueError("x_left and x_right must be finite")
        if not math.isfinite(y_bottom_f) or not math.isfinite(y_top_f):
            raise ValueError("y_bottom and y_top must be finite")
        if x_left_f == x_right_f:
            raise ValueError("Bar requires x_left != x_right")
        if y_bottom_f == y_top_f:
            raise ValueError("Bar requires y_bottom != y_top")
        if x_left_f > x_right_f:
            x_left_f, x_right_f = x_right_f, x_left_f

        bar = Bar(
            name=resolved_name,
            x_left=x_left_f,
            x_right=x_right_f,
            y_bottom=y_bottom_f,
            y_top=y_top_f,
            stroke_color=stroke_color,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            label_above_text=label_above_text,
            label_below_text=label_below_text,
        )
        self.drawables.add(bar)

        try:
            self.dependency_manager.analyze_drawable_for_dependencies(bar)
        except Exception:
            pass

        if redraw and getattr(self.canvas, "draw_enabled", False):
            try:
                self.canvas.draw()
            except Exception:
                pass

        return bar

    def delete_bar(self, name: str, *, archive: bool = True, redraw: bool = True) -> bool:
        bar = self.get_bar_by_name(name)
        if bar is None:
            return False

        if archive:
            try:
                self.canvas.archive()
            except Exception:
                pass

        removed = self.drawables.remove(bar)
        if removed:
            try:
                self.dependency_manager.remove_drawable(bar)
            except Exception:
                pass

        if removed and redraw and getattr(self.canvas, "draw_enabled", False):
            try:
                self.canvas.draw()
            except Exception:
                pass

        return bool(removed)

    def _generate_unique_name(self, base: str) -> str:
        base = str(base or "").strip()
        if not base:
            base = "bar"

        existing_names = set()
        try:
            for drawable in self.drawables.get_by_class_name("Bar"):
                dname = getattr(drawable, "name", "")
                if dname:
                    existing_names.add(str(dname))
        except Exception:
            existing_names = set()

        if base not in existing_names:
            return base

        idx = 1
        while True:
            candidate = f"{base}_{idx}"
            if candidate not in existing_names:
                return candidate
            idx += 1

    def _update_existing_bar(
        self,
        bar: Bar,
        *,
        x_left: float,
        x_right: float,
        y_bottom: float,
        y_top: float,
        stroke_color: Optional[str],
        fill_color: Optional[str],
        fill_opacity: Optional[float],
        label_above_text: Optional[str],
        label_below_text: Optional[str],
    ) -> None:
        try:
            bar.x_left = float(x_left)
            bar.x_right = float(x_right)
            if bar.x_left > bar.x_right:
                bar.x_left, bar.x_right = bar.x_right, bar.x_left
        except Exception:
            pass

        try:
            bar.y_bottom = float(y_bottom)
            bar.y_top = float(y_top)
        except Exception:
            pass

        try:
            bar.color = "" if stroke_color is None else str(stroke_color)
        except Exception:
            pass

        try:
            bar.fill_color = None if fill_color is None else str(fill_color)
        except Exception:
            pass

        try:
            bar.fill_opacity = None if fill_opacity is None else float(fill_opacity)
        except Exception:
            pass

        try:
            bar.label_above_text = None if label_above_text is None else str(label_above_text)
            bar.label_below_text = None if label_below_text is None else str(label_below_text)
            bar.label_text = bar.label_above_text
        except Exception:
            pass


