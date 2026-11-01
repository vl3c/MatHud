"""Cobweb (web) diagram of the orbit x0, g(x0), g(g(x0)), ... of a function g.

It draws y = g(x) and y = x over ``x_range`` and the web (x0, 0) -> (x0, x1) -> (x1, x1)
-> (x1, x2) -> ...; ``utils.sequences`` computes the orbit and says what it does. The plot
saves its inputs and a ``summary`` for the canvas text; a restore recomputes the orbit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from drawables.data_plot import DataPlot
from utils.sequences import IterationFunction, drawn_orbit, orbit_until_settled, orbit_x_range


class CobwebCurve:
    """The curve y = g(x) over the plot's x range, in the parametric interface its renderer reads.

    ``evaluate_x(t) = t`` and ``evaluate_y(t) = g(t)``; it has no name, so it gets no label.
    """

    def __init__(self, g: Any, x_min: float, x_max: float) -> None:
        self._g = g
        self.t_min = float(x_min)
        self.t_max = float(x_max)
        self.name = ""

    def evaluate_x(self, t: float) -> float:
        return float(t)

    def evaluate_y(self, t: float) -> float:
        return float(self._g(t))

    def evaluate(self, t: float) -> Tuple[float, float]:
        return float(t), float(self._g(t))


class CobwebPlot(DataPlot):
    """The cobweb of ``steps`` iterations of g from ``x0``.

    Attributes:
        function: The parsed g (``expression``, ``evaluate``).
        x0: The starting value.
        steps: The iterations asked for. The web stops earlier when g becomes undefined, a
            value escapes, or a value passes ``DRAW_FACTOR * max(1, |x0|)``.
        orbit: x0, x1, ... as drawn (a tuple).
        x_range: The interval the curve and the diagonal are drawn on.
        curve: The ``CobwebCurve`` the renderer draws g with.
        summary: ``orbit``, ``x_range``, ``computed_steps`` (the iterations computed for the
            drawing, before the draw bound) and the ``classify_orbit`` fields, which come from
            up to ``CLASSIFY_STEPS`` iterations.
    """

    _SHARED_ATTRIBUTES = ("summary", "function", "curve", "orbit")

    def __init__(
        self,
        name: str,
        *,
        function: IterationFunction,
        x0: float,
        steps: int,
        color: Optional[str] = None,
    ) -> None:
        super().__init__(name, plot_type="cobweb", color=color)
        self.function = function
        self.x0 = float(x0)
        self.steps = int(steps)
        computed, behaviour = orbit_until_settled(function.evaluate, self.x0, self.steps)
        self.orbit: Tuple[float, ...] = tuple(drawn_orbit(computed.values))
        extra = [behaviour.get("fixed_point")] + list(behaviour.get("cycle") or [])
        self.x_range: Tuple[float, float] = orbit_x_range(self.orbit, [v for v in extra if v is not None])
        self.curve = CobwebCurve(function.evaluate, self.x_range[0], self.x_range[1])
        summary: Dict[str, Any] = {
            "orbit": self.orbit,
            "x_range": self.x_range,
            "computed_steps": len(computed.values) - 1,
        }
        for key, value in behaviour.items():
            summary[key] = tuple(value) if isinstance(value, list) else value
        self.summary = summary

    def get_class_name(self) -> str:
        return "CobwebPlot"

    @property
    def expression(self) -> str:
        return self.function.expression

    def web_points(self) -> List[Tuple[float, float]]:
        """The web's corners: (x0, 0), (x0, x1), (x1, x1), (x1, x2), ..."""
        points = [(self.orbit[0], 0.0)]
        for current, following in zip(self.orbit, self.orbit[1:]):
            points.append((current, following))
            points.append((following, following))
        return points

    def extent_points(self) -> List[Tuple[float, float]]:
        x_min, x_max = self.x_range
        ys = [0.0, x_min, x_max] + list(self.orbit)
        for x in (x_min, x_max):
            try:
                y = float(self.function.evaluate(x))
            except Exception:
                continue
            if y == y and abs(y) != float("inf"):
                ys.append(y)
        return [(x_min, min(ys)), (x_max, max(ys))]

    def _plot_args(self) -> Dict[str, Any]:
        summary = {key: list(value) if isinstance(value, tuple) else value for key, value in self.summary.items()}
        return {"function": self.function.expression, "x0": self.x0, "steps": self.steps, "summary": summary}
