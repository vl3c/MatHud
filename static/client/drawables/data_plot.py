"""Base class of the plots drawn from a data list: histograms, box plots and scatter plots.

Unlike the distribution and bar plots, which are records of the drawables they create, a
data plot draws itself: it is renderable and has its own plan helper
(``rendering/cached_render_plan._HELPERS``). It keeps its data as tuples and computes its
summary once, when built. Nothing changes a data plot after that (there is no update tool
and transforms refuse plots), so a copy shares the data and summary, and
``render_signature`` is fixed, which keeps undo snapshots and per-frame renderer checks
cheap for plots of up to 1000 values.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from drawables.plot import Plot


class DataPlot(Plot):
    """A plot that draws itself from data it keeps (see the module docstring)."""

    # Attributes a copy shares: built once and never mutated.
    _SHARED_ATTRIBUTES: Tuple[str, ...] = ("summary",)

    def __init__(self, name: str, *, plot_type: str, color: Optional[str] = None) -> None:
        super().__init__(name, plot_type=plot_type)
        self.is_renderable = True
        self.plot_color: Optional[str] = None if color is None else str(color)
        self._signature: Optional[Tuple[Any, ...]] = None

    def _plot_args(self) -> Dict[str, Any]:
        """The saved args beyond ``plot_type``; subclasses list their data and options."""
        raise NotImplementedError

    def get_state(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {"plot_type": self.plot_type}
        args.update(self._plot_args())
        if self.plot_color is not None:
            args["color"] = self.plot_color
        return {"name": self.name, "args": args}

    def extent_points(self) -> List[Tuple[float, float]]:
        """Two corners of the box around what the plot draws (lower left, upper right)."""
        raise NotImplementedError

    def render_signature(self) -> Tuple[Any, ...]:
        """A fixed token for the renderers' plan cache: the plot never changes once built."""
        if self._signature is None:
            self._signature = (self.get_class_name(), repr(self._plot_args()), self.plot_color)
        return (self.name,) + self._signature

    def __deepcopy__(self, memo: Dict[int, Any]) -> "DataPlot":
        if id(self) in memo:
            return memo[id(self)]  # type: ignore[no-any-return]
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        for key, value in self.__dict__.items():
            # Data and summaries are tuples or never mutated: share them instead of copying
            shared = (
                key in self._SHARED_ATTRIBUTES or value is None or isinstance(value, (tuple, str, int, float, bool))
            )
            copied.__dict__[key] = value if shared else deepcopy(value, memo)
        return copied
