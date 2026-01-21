"""Attached label drawable for labels embedded in other drawables.

This module provides the AttachedLabel class for labels that are owned
and rendered by their parent drawable rather than the main container.

Key Features:
    - Non-renderable by default to prevent duplicate rendering
    - Screen-offset mode for constant pixel offset positioning
    - Configurable text format (text only or with coordinates)
    - Font size sourced from label or style dictionary
    - Pixel offset control relative to point radius
"""

from __future__ import annotations

from typing import Optional

from drawables.label import Label
from drawables.label_render_mode import LabelRenderMode, _ScreenOffsetLabelMode


class AttachedLabel(Label):
    """Label intended to be embedded in another drawable.

    Attached labels are not stored in the drawables container and are rendered
    by their owning drawable. They default to a screen-offset mode (point-like
    constant size/offset behavior).
    """

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
        reference_scale_factor: Optional[float] = None,
        visible: bool = True,
        # attached-mode knobs (kept simple; they map directly to the mode state)
        text_format: str = "text_only",
        coord_precision: int = 3,
        offset_from_point_radius: bool = True,
        non_selectable: bool = False,
        font_size_source: str = "label",
        font_size_key: str = "point_label_font_size",
        font_family_key: str = "label_font_family",
        offset_px_x: float = 0.0,
        offset_px_y: float = 0.0,
        # allow explicit mode override for advanced cases / restore paths
        render_mode: Optional[LabelRenderMode] = None,
    ) -> None:
        if render_mode is None:
            render_mode = _ScreenOffsetLabelMode(
                text_format=text_format,
                coord_precision=coord_precision,
                offset_from_point_radius=offset_from_point_radius,
                non_selectable=non_selectable,
                font_size_source=font_size_source,
                font_size_key=font_size_key,
                font_family_key=font_family_key,
                offset_px_x=offset_px_x,
                offset_px_y=offset_px_y,
            )

        super().__init__(
            x,
            y,
            text,
            name=name,
            color=color,
            font_size=font_size,
            rotation_degrees=rotation_degrees,
            reference_scale_factor=reference_scale_factor,
            visible=visible,
            render_mode=render_mode,
        )

        # Safety rail: attached labels should not be rendered as standalone if they
        # are accidentally added to the drawables container.
        try:
            self.is_renderable = False
        except Exception:
            pass


