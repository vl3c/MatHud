from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from drawables.drawable import Drawable


class Bar(Drawable):
    """
    A simple filled rectangle bar in math space.

    This is intended for discrete plots (histograms / bar charts) and does not
    create or depend on points/segments.
    """

    def __init__(
        self,
        *,
        name: str,
        x_left: float,
        x_right: float,
        y_top: float,
        y_bottom: float = 0.0,
        stroke_color: Optional[str] = None,
        fill_color: Optional[str] = None,
        fill_opacity: Optional[float] = None,
        label_above_text: Optional[str] = None,
        label_below_text: Optional[str] = None,
        label_text: Optional[str] = None,
        is_renderable: bool = True,
    ) -> None:
        super().__init__(name=name, color=str(stroke_color) if stroke_color is not None else "", is_renderable=is_renderable)
        self.x_left: float = float(x_left)
        self.x_right: float = float(x_right)
        self.y_bottom: float = float(y_bottom)
        self.y_top: float = float(y_top)
        self.fill_color: Optional[str] = None if fill_color is None else str(fill_color)
        self.fill_opacity: Optional[float] = None if fill_opacity is None else float(fill_opacity)
        resolved_above_text = label_above_text
        if resolved_above_text is None:
            resolved_above_text = label_text
        self.label_above_text: Optional[str] = None if resolved_above_text is None else str(resolved_above_text)
        self.label_below_text: Optional[str] = None if label_below_text is None else str(label_below_text)
        # Backward-compatible alias used by older code paths.
        self.label_text: Optional[str] = self.label_above_text

    def get_class_name(self) -> str:
        return "Bar"

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "args": {
                "x_left": self.x_left,
                "x_right": self.x_right,
                "y_bottom": self.y_bottom,
                "y_top": self.y_top,
                "stroke_color": self.color,
                "fill_color": self.fill_color,
                "fill_opacity": self.fill_opacity,
                "label_above_text": self.label_above_text,
                "label_below_text": self.label_below_text,
                "label_text": self.label_text,
            },
        }

    def translate(self, x_offset: float, y_offset: float) -> None:
        self.x_left += float(x_offset)
        self.x_right += float(x_offset)
        self.y_bottom += float(y_offset)
        self.y_top += float(y_offset)

    def rotate(self, angle: float) -> None:
        # A Bar is axis-aligned; rotation is not supported.
        return None

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Bar":
        if id(self) in memo:
            return memo[id(self)]  # type: ignore[return-value]
        copied = Bar(
            name=self.name,
            x_left=deepcopy(self.x_left, memo),
            x_right=deepcopy(self.x_right, memo),
            y_bottom=deepcopy(self.y_bottom, memo),
            y_top=deepcopy(self.y_top, memo),
            stroke_color=deepcopy(self.color, memo),
            fill_color=deepcopy(self.fill_color, memo),
            fill_opacity=deepcopy(self.fill_opacity, memo),
            label_above_text=deepcopy(self.label_above_text, memo),
            label_below_text=deepcopy(self.label_below_text, memo),
            label_text=deepcopy(self.label_text, memo),
            is_renderable=self.is_renderable,
        )
        memo[id(self)] = copied
        return copied


