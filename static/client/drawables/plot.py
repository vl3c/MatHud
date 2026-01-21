"""Base plot drawable for statistical and distribution plots.

This module provides the Plot class as a non-renderable metadata container
for statistical plots composed of other drawable components.

Key Features:
    - Non-renderable composite pattern for plot metadata
    - Distribution type and parameter storage
    - Configurable plot bounds for axis control
    - Metadata dictionary for extensible properties
    - Serialization support for workspace persistence
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from drawables.drawable import Drawable


class Plot(Drawable):
    """
    Non-renderable composite representing a plot composed of other drawables.

    Base plot metadata container. This object is not rendered directly.

    Subclasses track the concrete drawable components created for the plot.
    """

    def __init__(
        self,
        name: str,
        *,
        plot_type: str,
        distribution_type: Optional[str] = None,
        distribution_params: Optional[Dict[str, Any]] = None,
        bounds: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, is_renderable=False)
        self.plot_type: str = str(plot_type)
        self.distribution_type: Optional[str] = distribution_type
        self.distribution_params: Dict[str, Any] = dict(distribution_params or {})
        self.bounds: Dict[str, float] = dict(bounds or {})
        self.metadata: Dict[str, Any] = dict(metadata or {})

    def get_class_name(self) -> str:
        return "Plot"

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "args": {
                "plot_type": self.plot_type,
                "distribution_type": self.distribution_type,
                "distribution_params": self.distribution_params,
                "bounds": self.bounds,
                "metadata": self.metadata,
            },
        }

    def translate(self, x_offset: float, y_offset: float) -> None:
        # Plot is a composite reference object; translating it has no meaning.
        return None

    def rotate(self, angle: float) -> None:
        # Plot is a composite reference object; rotating it has no meaning.
        return None

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Plot":
        if id(self) in memo:
            return memo[id(self)]  # type: ignore[return-value]
        copied = Plot(
            name=self.name,
            plot_type=self.plot_type,
            distribution_type=deepcopy(self.distribution_type, memo),
            distribution_params=deepcopy(self.distribution_params, memo),
            bounds=deepcopy(self.bounds, memo),
            metadata=deepcopy(self.metadata, memo),
        )
        memo[id(self)] = copied
        return copied

