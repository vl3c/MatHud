"""Continuous plot drawable for function-based distribution plots.

This module provides the ContinuousPlot class for statistical plots
that display continuous probability distributions using function curves.

Key Features:
    - References a Function drawable for the distribution curve
    - Optional filled area reference for shaded regions
    - Distribution type and parameters for reconstruction
    - Extends base Plot with continuous-specific attributes
    - Deep copy support for undo/redo operations
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from drawables.plot import Plot


class ContinuousPlot(Plot):
    """Non-renderable plot composite backed by a function and a filled area."""

    def __init__(
        self,
        name: str,
        *,
        plot_type: str,
        distribution_type: Optional[str] = None,
        function_name: Optional[str] = None,
        fill_area_name: Optional[str] = None,
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
        self.function_name: Optional[str] = function_name
        self.fill_area_name: Optional[str] = fill_area_name
        self.is_renderable = bool(is_renderable)

    def get_class_name(self) -> str:
        return "ContinuousPlot"

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["args"].update(
            {
                "function_name": self.function_name,
                "fill_area_name": self.fill_area_name,
            }
        )
        return state

    def __deepcopy__(self, memo: Dict[int, Any]) -> "ContinuousPlot":
        if id(self) in memo:
            return memo[id(self)]  # type: ignore[return-value]
        copied = ContinuousPlot(
            name=self.name,
            plot_type=self.plot_type,
            distribution_type=deepcopy(self.distribution_type, memo),
            function_name=deepcopy(self.function_name, memo),
            fill_area_name=deepcopy(self.fill_area_name, memo),
            distribution_params=deepcopy(self.distribution_params, memo),
            bounds=deepcopy(self.bounds, memo),
            metadata=deepcopy(self.metadata, memo),
            is_renderable=self.is_renderable,
        )
        memo[id(self)] = copied
        return copied
