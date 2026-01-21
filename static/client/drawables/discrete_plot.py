"""Discrete plot drawable for bar-based distribution plots.

This module provides the DiscretePlot class for statistical plots
that display discrete probability distributions using bar elements.

Key Features:
    - Bar count and label configuration for discrete values
    - Customizable bar colors and opacity
    - Legacy support for rectangle and fill area component names
    - Distribution parameters for plot reconstruction
    - Deep copy support for undo/redo operations
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from drawables.plot import Plot


class DiscretePlot(Plot):
    """
    Non-renderable plot composite for discrete plots (bars).

    New-style plots store rebuild parameters and generate Bar drawables on demand.
    Legacy fields (rectangle_names/fill_area_names) are kept optional for backward compatibility.
    """

    def __init__(
        self,
        name: str,
        *,
        plot_type: str,
        distribution_type: Optional[str] = None,
        bar_count: Optional[int] = None,
        bar_labels: Optional[List[str]] = None,
        curve_color: Optional[str] = None,
        fill_color: Optional[str] = None,
        fill_opacity: Optional[float] = None,
        rectangle_names: Optional[List[str]] = None,
        fill_area_names: Optional[List[str]] = None,
        distribution_params: Optional[Dict[str, Any]] = None,
        bounds: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_renderable: bool = True,
    ) -> None:
        super().__init__(
            name,
            plot_type=plot_type,
            distribution_type=distribution_type,
            distribution_params=distribution_params,
            bounds=bounds,
            metadata=metadata,
        )
        self.bar_count: Optional[int] = None if bar_count is None else int(bar_count)
        self.bar_labels: Optional[List[str]] = None if bar_labels is None else list(bar_labels)
        self.curve_color: Optional[str] = None if curve_color is None else str(curve_color)
        self.fill_color: Optional[str] = None if fill_color is None else str(fill_color)
        self.fill_opacity: Optional[float] = None if fill_opacity is None else float(fill_opacity)

        # Legacy component tracking (old discrete implementation).
        self.rectangle_names: List[str] = list(rectangle_names or [])
        self.fill_area_names: List[str] = list(fill_area_names or [])
        self.is_renderable = bool(is_renderable)

    def get_class_name(self) -> str:
        return "DiscretePlot"

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["args"].update(
            {
                "bar_count": self.bar_count,
                "bar_labels": list(self.bar_labels) if self.bar_labels is not None else None,
                "curve_color": self.curve_color,
                "fill_color": self.fill_color,
                "fill_opacity": self.fill_opacity,
                "rectangle_names": list(self.rectangle_names),
                "fill_area_names": list(self.fill_area_names),
            }
        )
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> "DiscretePlot":
        if id(self) in memo:
            return memo[id(self)]  # type: ignore[return-value]
        copied = DiscretePlot(
            name=self.name,
            plot_type=self.plot_type,
            distribution_type=deepcopy(self.distribution_type, memo),
            bar_count=deepcopy(self.bar_count, memo),
            bar_labels=deepcopy(self.bar_labels, memo),
            curve_color=deepcopy(self.curve_color, memo),
            fill_color=deepcopy(self.fill_color, memo),
            fill_opacity=deepcopy(self.fill_opacity, memo),
            rectangle_names=deepcopy(self.rectangle_names, memo),
            fill_area_names=deepcopy(self.fill_area_names, memo),
            distribution_params=deepcopy(self.distribution_params, memo),
            bounds=deepcopy(self.bounds, memo),
            metadata=deepcopy(self.metadata, memo),
            is_renderable=self.is_renderable,
        )
        memo[id(self)] = copied
        return copied


