"""
MatHud Label Drawable

Provides a math-space anchored text label that can be rendered on the canvas.
Labels support configurable text content, color, font size, and rotation.
Text is automatically wrapped when it exceeds the configured line length
threshold and validated against the maximum allowed length.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, cast

from constants import (
    default_color,
    default_label_font_size,
    default_label_rotation_degrees,
    label_line_wrap_threshold,
    label_text_max_length,
)
from drawables.drawable import Drawable
from drawables.position import Position


class Label(Drawable):
    """Represents a text annotation anchored at a math-space coordinate."""

    def __init__(
        self,
        x: float,
        y: float,
        text: str,
        *,
        name: str = "",
        color: Optional[str] = None,
        font_size: Optional[float] = None,
        rotation_degrees: Optional[float] = None,
    ) -> None:
        self._position: Position = Position(float(x), float(y))
        super().__init__(name=name, color=color or default_color)
        self._font_size: float = float(font_size) if font_size is not None else float(default_label_font_size)
        self._text: str = ""
        self._lines: List[str] = []
        self._rotation_degrees: float = (
            float(rotation_degrees)
            if rotation_degrees is not None
            else float(default_label_rotation_degrees)
        )
        self.set_text(text)

    def get_class_name(self) -> str:
        return "Label"

    @property
    def position(self) -> Position:
        return self._position

    @property
    def text(self) -> str:
        return self._text

    def set_text(self, value: str) -> None:
        normalized = self._normalize_text(value)
        if len(normalized) > label_text_max_length:
            raise ValueError(f"Label text exceeds maximum length of {label_text_max_length} characters")
        self._text = normalized
        self._lines = self._wrap_text(normalized)

    @property
    def lines(self) -> List[str]:
        return list(self._lines)

    @property
    def font_size(self) -> float:
        return self._font_size

    @font_size.setter
    def font_size(self, value: float) -> None:
        numeric = float(value)
        if numeric <= 0:
            raise ValueError("Label font size must be positive")
        self._font_size = numeric

    @property
    def rotation_degrees(self) -> float:
        return self._rotation_degrees

    @rotation_degrees.setter
    def rotation_degrees(self, value: float) -> None:
        numeric = float(value)
        self._rotation_degrees = numeric

    def translate(self, x_offset: float, y_offset: float) -> None:
        self._position.x += float(x_offset)
        self._position.y += float(y_offset)

    def rotate(self, angle: float) -> None:
        delta = float(angle)
        if not math.isfinite(delta):
            raise ValueError("Rotation angle must be finite")
        self._rotation_degrees += delta

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "args": {
                "position": {"x": self._position.x, "y": self._position.y},
                "text": self._text,
                "color": self.color,
                "font_size": self._font_size,
                "rotation_degrees": self._rotation_degrees,
            },
        }

    def __deepcopy__(self, memo: Dict[int, Any]) -> Label:
        if id(self) in memo:
            return cast(Label, memo[id(self)])
        clone = Label(
            self._position.x,
            self._position.y,
            self._text,
            name=self.name,
            color=self.color,
            font_size=self._font_size,
            rotation_degrees=self._rotation_degrees,
        )
        memo[id(self)] = clone
        return clone

    def _normalize_text(self, text: str) -> str:
        cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        return cleaned.strip()

    def _wrap_text(self, text: str) -> List[str]:
        if not text:
            return [""]
        raw_lines = text.split("\n")
        wrapped: List[str] = []
        for raw_line in raw_lines:
            wrapped.extend(self._wrap_line(raw_line))
        return wrapped

    def _wrap_line(self, line: str) -> List[str]:
        if len(line) <= label_line_wrap_threshold:
            return [line]
        words = line.split()
        if not words:
            return [line]
        current = words[0]
        lines: List[str] = []
        for word in words[1:]:
            tentative = f"{current} {word}"
            if len(tentative) <= label_line_wrap_threshold:
                current = tentative
                continue
            lines.append(current)
            current = word
        lines.append(current)
        flattened: List[str] = []
        for entry in lines:
            flattened.extend(self._split_long_word(entry))
        return flattened

    def _split_long_word(self, chunk: str) -> List[str]:
        if len(chunk) <= label_line_wrap_threshold:
            return [chunk]
        segments: List[str] = []
        start = 0
        while start < len(chunk):
            end = start + label_line_wrap_threshold
            segments.append(chunk[start:end])
            start = end
        return segments

