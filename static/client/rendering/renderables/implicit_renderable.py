"""
Implicit Renderable for MatHud

Plots an implicit curve (an equation in x and y) or an inequality region over the view
plus a margin of the screen on each side (utils.implicit_plotter), caches the result in
math coordinates, and maps it to the screen on each call.

The cache key is the zoom, the canvas size, the drawable's relation and the pan bucket:
pans within a bucket (the margin's size) stay inside the plotted area, so only a pan out
of it, a zoom or a new relation plots again. While a wheel or pinch zoom is under way
(``coordinate_mapper.zoom_in_progress``) the last plot is mapped to the new view instead,
and the plot is redone when the zoom settles. Plotting works in math coordinates on a grid
anchored to the plane, so plots of neighbouring buckets agree where they overlap.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

from utils.implicit_plotter import PlotResult, plot

Point = Tuple[float, float]

# Grid cell in screen pixels; a coarse block is BLOCK cells
CELL_PIXELS: float = 3.0
BLOCK: int = 16
# Evaluation budget of one plot (a denser curve is plotted on a coarser grid), and a time
# limit as a safety net; a circle takes about 25 ms per 1000 evaluations in Brython
MAX_EVALS: int = 25000
DEADLINE_SECONDS: float = 1.0


class ImplicitRenderable:
    """Plot cache and screen mapping for one implicit curve or inequality region."""

    def __init__(self, model: Any, coordinate_mapper: Any) -> None:
        self.model: Any = model
        self.mapper: Any = coordinate_mapper
        self._key: Optional[Tuple[Any, ...]] = None
        self._result: Optional[PlotResult] = None

    def invalidate_cache(self) -> None:
        """Forget the last plot."""
        self._key = None
        self._result = None

    def math_result(self, view_margin: float, *, fill: bool) -> PlotResult:
        """The plot of the view plus view_margin of a screen on each side (cached)."""
        mapper = self.mapper
        scale = float(getattr(mapper, "scale_factor", 1.0) or 1.0)
        width = float(getattr(mapper, "canvas_width", 0) or 0)
        height = float(getattr(mapper, "canvas_height", 0) or 0)
        offset = getattr(mapper, "offset", None)
        offset_x = float(getattr(offset, "x", 0.0) or 0.0)
        offset_y = float(getattr(offset, "y", 0.0) or 0.0)
        bucket = _bucket(offset_x, offset_y, width, height, view_margin)
        key = (scale, width, height, view_margin, bucket, fill, self._relation_signature())
        if self._result is not None and key == self._key:
            return self._result
        if self._result is not None and self._key is not None and key[5:] == self._key[5:]:
            if getattr(mapper, "zoom_in_progress", False):
                # Mid-zoom: map the last plot (math coordinates) to the new view, and plot
                # again when the zoom settles; the margin covers zooming out up to 2x
                return self._result
        self._result = self._plot(scale, width, height, view_margin, fill)
        self._key = key
        return self._result

    def screen_paths(self, view_margin: float, *, fill: bool) -> Tuple[List[List[Point]], List[int]]:
        """The plotted curves (or a region's boundary stretches) in screen coordinates,
        with the index of the inequality each stretch belongs to."""
        result = self.math_result(view_margin, fill=fill)
        to_screen = self._screen_mapping()
        return [to_screen(path) for path in result.curves], list(result.curve_atoms)

    def screen_rings(self, view_margin: float) -> List[List[Point]]:
        """A region's boundary rings in screen coordinates."""
        result = self.math_result(view_margin, fill=True)
        to_screen = self._screen_mapping()
        return [to_screen(ring) for ring in result.rings]

    # ------------------------------------------------------------------

    def _relation_signature(self) -> str:
        text = getattr(self.model, "equation", None)
        if text is None:
            text = getattr(self.model, "inequality", "")
        return str(text)

    def _plot(self, scale: float, width: float, height: float, view_margin: float, fill: bool) -> PlotResult:
        if scale <= 0 or width <= 0 or height <= 0:
            return PlotResult()
        bounds = self.mapper.get_visible_bounds()
        left, right = float(bounds["left"]), float(bounds["right"])
        bottom, top = float(bounds["bottom"]), float(bounds["top"])
        margin_x = (right - left) * view_margin
        margin_y = (top - bottom) * view_margin
        rect = (left - margin_x, bottom - margin_y, right + margin_x, top + margin_y)
        cell = CELL_PIXELS / scale
        if fill:
            relation = self.model.relation
            return plot(
                relation.value,
                rect,
                cell,
                fill=True,
                block=BLOCK,
                max_evals=MAX_EVALS,
                clock=time.time,
                deadline=DEADLINE_SECONDS,
                atom_of=relation.atom_of if len(relation.atoms) > 1 else None,
            )
        return plot(
            self.model.g,
            rect,
            cell,
            block=BLOCK,
            max_evals=MAX_EVALS,
            clock=time.time,
            deadline=DEADLINE_SECONDS,
            factors=self.model.factors,
        )

    def _screen_mapping(self) -> Callable[[Sequence[Point]], List[Point]]:
        mapper = self.mapper
        scale = float(getattr(mapper, "scale_factor", 1.0) or 1.0)
        origin = getattr(mapper, "origin", None)
        offset = getattr(mapper, "offset", None)
        shift_x = float(getattr(origin, "x", 0.0) or 0.0) + float(getattr(offset, "x", 0.0) or 0.0)
        shift_y = float(getattr(origin, "y", 0.0) or 0.0) + float(getattr(offset, "y", 0.0) or 0.0)

        def to_screen(points: Sequence[Point]) -> List[Point]:
            return [(shift_x + x * scale, shift_y - y * scale) for x, y in points]

        return to_screen


def _bucket(offset_x: float, offset_y: float, width: float, height: float, view_margin: float) -> Tuple[int, int]:
    """The pan bucket, as Canvas2DRenderer buckets function plans (half a screen by default)."""
    if view_margin <= 0 or width <= 0 or height <= 0:
        return (int(round(offset_x * 1000)), int(round(offset_y * 1000)))
    return (int(math.floor(offset_x / (width * view_margin))), int(math.floor(offset_y / (height * view_margin))))
