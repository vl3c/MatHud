"""Label render mode configuration for different label rendering strategies.

This module provides serializable configuration objects that control how
labels are positioned and rendered relative to the coordinate system.

Key Features:
    - World mode: Labels scale with zoom and vanish at extreme zoom-out
    - Screen-offset mode: Fixed pixel offset from anchor point
    - Configurable text format with optional coordinate display
    - Font size sourcing from label or style dictionary
    - Serialization and deserialization for workspace persistence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type


class LabelRenderMode:
    """Serializable label rendering configuration.

    This is a data/config object (not a renderer). Rendering helpers inspect
    the mode and choose an appropriate drawing strategy.
    """

    kind: str = "base"

    def to_state(self) -> Dict[str, Any]:
        return {"kind": self.kind}

    @classmethod
    def from_state(cls, raw: Any) -> "LabelRenderMode":
        if not isinstance(raw, dict):
            return _WorldLabelMode()
        kind = raw.get("kind")
        mode_cls = _KIND_TO_MODE.get(kind)
        if mode_cls is None:
            return _WorldLabelMode()
        return mode_cls.from_state(raw)


@dataclass(frozen=True)
class _WorldLabelMode(LabelRenderMode):
    """World-anchored label mode (default): zoom-out scales font and can vanish."""

    kind: str = "world"

    @classmethod
    def from_state(cls, raw: Any) -> "_WorldLabelMode":
        return cls()


@dataclass(frozen=True)
class _ScreenOffsetLabelMode(LabelRenderMode):
    """Screen-offset label mode: fixed pixel offset and fixed font size on zoom.

    This is used to match the existing point label behavior and can be embedded
    into other drawables.
    """

    # Keep string values stable: these are persisted.
    kind: str = "screen_offset"

    # text_format:
    # - "text_only": draw label.text as-is (possibly multi-line)
    # - "text_with_anchor_coords": draw f"{text}(x, y)" with rounded anchor coords
    text_format: str = "text_only"

    # rounding precision for text_with_anchor_coords
    coord_precision: int = 3

    # if True, base offset uses (radius, -radius) where radius is style["point_radius"]
    offset_from_point_radius: bool = True

    # apply CSS style overrides to prevent text selection
    non_selectable: bool = False

    # font sizing options:
    # - "label": use label.font_size
    # - "style": use style[font_size_key]
    font_size_source: str = "label"
    font_size_key: str = "point_label_font_size"

    # font family style key to use (eg "point_label_font_family" or "label_font_family")
    font_family_key: str = "label_font_family"

    # explicit offset when offset_from_point_radius is False
    offset_px_x: float = 0.0
    offset_px_y: float = 0.0

    def to_state(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "text_format": self.text_format,
            "coord_precision": int(self.coord_precision),
            "offset_from_point_radius": bool(self.offset_from_point_radius),
            "non_selectable": bool(self.non_selectable),
            "font_size_source": self.font_size_source,
            "font_size_key": self.font_size_key,
            "font_family_key": self.font_family_key,
            "offset_px_x": float(self.offset_px_x),
            "offset_px_y": float(self.offset_px_y),
        }

    @classmethod
    def from_state(cls, raw: Any) -> "_ScreenOffsetLabelMode":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            text_format=str(raw.get("text_format", "text_only") or "text_only"),
            coord_precision=int(raw.get("coord_precision", 3) or 3),
            offset_from_point_radius=bool(raw.get("offset_from_point_radius", True)),
            non_selectable=bool(raw.get("non_selectable", False)),
            font_size_source=str(raw.get("font_size_source", "label") or "label"),
            font_size_key=str(raw.get("font_size_key", "point_label_font_size") or "point_label_font_size"),
            font_family_key=str(raw.get("font_family_key", "label_font_family") or "label_font_family"),
            offset_px_x=float(raw.get("offset_px_x", 0.0) or 0.0),
            offset_px_y=float(raw.get("offset_px_y", 0.0) or 0.0),
        )


_KIND_TO_MODE: Dict[Optional[str], Type[LabelRenderMode]] = {
    "world": _WorldLabelMode,
    "screen_offset": _ScreenOffsetLabelMode,
}
