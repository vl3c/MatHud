from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from drawables.plot import Plot


class BarsPlot(Plot):
    """
    Non-renderable plot composite for bar charts.

    Stores parameters needed to rebuild derived Bar drawables.
    """

    def __init__(
        self,
        name: str,
        *,
        plot_type: str,
        values: List[float],
        labels_below: List[str],
        labels_above: Optional[List[str]] = None,
        bar_spacing: Optional[float] = None,
        bar_width: Optional[float] = None,
        x_start: Optional[float] = None,
        y_base: Optional[float] = None,
        stroke_color: Optional[str] = None,
        fill_color: Optional[str] = None,
        fill_opacity: Optional[float] = None,
        bounds: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_renderable: bool = True,
    ) -> None:
        super().__init__(
            name,
            plot_type=plot_type,
            distribution_type=None,
            distribution_params=None,
            bounds=bounds,
            metadata=metadata,
        )
        self.values: List[float] = [float(v) for v in list(values or [])]
        self.labels_below: List[str] = [str(v) for v in list(labels_below or [])]
        self.labels_above: Optional[List[str]] = None if labels_above is None else [str(v) for v in list(labels_above)]
        self.bar_spacing: Optional[float] = None if bar_spacing is None else float(bar_spacing)
        self.bar_width: Optional[float] = None if bar_width is None else float(bar_width)
        self.x_start: Optional[float] = None if x_start is None else float(x_start)
        self.y_base: Optional[float] = None if y_base is None else float(y_base)
        self.stroke_color: Optional[str] = None if stroke_color is None else str(stroke_color)
        self.fill_color: Optional[str] = None if fill_color is None else str(fill_color)
        self.fill_opacity: Optional[float] = None if fill_opacity is None else float(fill_opacity)
        self.is_renderable = bool(is_renderable)

    def get_class_name(self) -> str:
        return "BarsPlot"

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["args"].update(
            {
                "values": list(self.values),
                "labels_below": list(self.labels_below),
                "labels_above": None if self.labels_above is None else list(self.labels_above),
                "bar_spacing": self.bar_spacing,
                "bar_width": self.bar_width,
                "x_start": self.x_start,
                "y_base": self.y_base,
                "stroke_color": self.stroke_color,
                "fill_color": self.fill_color,
                "fill_opacity": self.fill_opacity,
            }
        )
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> "BarsPlot":
        if id(self) in memo:
            return memo[id(self)]  # type: ignore[return-value]
        copied = BarsPlot(
            name=self.name,
            plot_type=self.plot_type,
            values=deepcopy(self.values, memo),
            labels_below=deepcopy(self.labels_below, memo),
            labels_above=deepcopy(self.labels_above, memo),
            bar_spacing=deepcopy(self.bar_spacing, memo),
            bar_width=deepcopy(self.bar_width, memo),
            x_start=deepcopy(self.x_start, memo),
            y_base=deepcopy(self.y_base, memo),
            stroke_color=deepcopy(self.stroke_color, memo),
            fill_color=deepcopy(self.fill_color, memo),
            fill_opacity=deepcopy(self.fill_opacity, memo),
            bounds=deepcopy(self.bounds, memo),
            metadata=deepcopy(self.metadata, memo),
            is_renderable=self.is_renderable,
        )
        memo[id(self)] = copied
        return copied


