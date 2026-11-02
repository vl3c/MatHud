"""The sequence tools: plot_sequence and plot_cobweb.

Each tool parses its expression (``utils.sequences``), builds one self-drawing plot
(``drawables/sequence_plot.py``, ``drawables/cobweb_plot.py``), adds it as one undo step
and returns the numbers the plot shows. Like the data plots, the tools are value tools
(``tool_kinds.VALUE_TOOLS``) that archive their own undo step; the function registry
binds them to this manager directly, and ``delete_plot`` deletes the plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from drawables.cobweb_plot import CobwebPlot
from drawables.data_plot import DataPlot
from drawables.sequence_plot import SequencePlot
from utils.sequences import (
    MAX_LISTED_UNDEFINED,
    check_index_range,
    check_initial_terms,
    check_steps,
    finite_number,
    parse_iteration_function,
    parse_sequence,
)
from utils.statistics.data_plots import round_significant

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.drawables_container import DrawablesContainer
    from name_generator.drawable import DrawableNameGenerator

# Values listed in full in a result; longer lists give their first and last values.
MAX_LISTED_VALUES = 50
LISTED_HEAD = 20
LISTED_TAIL = 10


class SequencePlotManager:
    """Creates sequence plots and cobweb diagrams (see the module docstring)."""

    def __init__(
        self,
        canvas: "Canvas",
        drawables: "DrawablesContainer",
        name_generator: "DrawableNameGenerator",
    ) -> None:
        self.canvas = canvas
        self.drawables = drawables
        self.name_generator = name_generator

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def plot_sequence(
        self,
        *,
        expression: str,
        start: Any,
        end: Any,
        initial_terms: Optional[List[float]] = None,
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Draw u(n) for n from start to end as dots; return the terms."""
        rule = parse_sequence(expression)
        first, last = check_index_range(start, end)
        initial = check_initial_terms(rule, initial_terms)
        plot = SequencePlot(
            self._plot_name(name, "sequence"),
            rule=rule,
            start=first,
            end=last,
            initial_terms=initial,
            color=_optional_text(color),
        )
        self._add(plot)
        return self._sequence_result(plot)

    def plot_cobweb(
        self,
        *,
        function: str,
        x0: Any,
        steps: Any = None,
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Iterate g from x0 and draw the cobweb; return the orbit and what it does."""
        parsed = parse_iteration_function(function)
        start = finite_number(x0, "x0")
        count = check_steps(steps)
        plot = CobwebPlot(
            self._plot_name(name, "cobweb"),
            function=parsed,
            x0=start,
            steps=count,
            color=_optional_text(color),
        )
        if plot.summary["computed_steps"] == 0:
            raise ValueError(
                f"g(x) = {parsed.expression} is undefined (or too large) at x0 = {start}, so there is nothing to iterate"
            )
        self._add(plot)
        return self._cobweb_result(plot)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    @staticmethod
    def _sequence_result(plot: SequencePlot) -> Dict[str, Any]:
        summary = plot.summary
        result: Dict[str, Any] = {
            "plot_name": plot.name,
            "plot_type": "sequence",
            "kind": "recursive" if plot.rule.order else "explicit",
        }
        if plot.rule.order:
            result["order"] = plot.rule.order
        result["start"] = plot.start
        result["end"] = plot.end
        result.update(_listed("terms", summary["terms"]))
        if len(summary["terms"]) > MAX_LISTED_VALUES:
            result["first_n"] = summary["n_values"][0]
        result["min"] = _rounded(summary["min"])
        result["max"] = _rounded(summary["max"])
        result["trend"] = summary["trend"]
        notes: List[str] = []
        undefined = list(summary["undefined_at"])
        if undefined:
            result["undefined_at"] = undefined[:MAX_LISTED_UNDEFINED]
            result["undefined_count"] = len(undefined)
            notes.append(
                f"u(n) is undefined (or too large) at {len(undefined)} index(es); those terms are left out "
                "and not drawn"
            )
        if summary["stopped_at"] is not None:
            result["stopped_at"] = summary["stopped_at"]
            notes.append(
                f"the recursion stopped at n = {summary['stopped_at']}: that term is undefined or too large "
                "(overflow or division by zero), so it and the later terms are not drawn"
            )
        if len(summary["terms"]) > MAX_LISTED_VALUES:
            notes.append(f"{len(summary['terms'])} terms: the first {LISTED_HEAD} and last {LISTED_TAIL} are listed")
        if notes:
            result["note"] = "; ".join(notes)
        return result

    @staticmethod
    def _cobweb_result(plot: CobwebPlot) -> Dict[str, Any]:
        summary = plot.summary
        drawn = len(plot.orbit) - 1
        result: Dict[str, Any] = {
            "plot_name": plot.name,
            "plot_type": "cobweb",
            "function": plot.expression,
            "x0": plot.x0,
            "steps": drawn,
        }
        result.update(_listed("orbit", plot.orbit))
        result["behaviour"] = summary["behaviour"]
        notes = [_behaviour_note(summary, result)]
        if drawn < plot.steps and summary["computed_steps"] > drawn:
            notes.append(
                f"the web is drawn for {drawn} of the {plot.steps} steps: later values pass "
                f"{_rounded(1000.0 * max(1.0, abs(plot.x0)), 3)} in size"
            )
        elif drawn < plot.steps:
            notes.append(f"the orbit stops after {drawn} of the {plot.steps} steps")
        if len(plot.orbit) > MAX_LISTED_VALUES:
            notes.append(f"{len(plot.orbit)} orbit values: the first {LISTED_HEAD} and last {LISTED_TAIL} are listed")
        result["note"] = "; ".join(notes)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _plot_name(self, name: Optional[str], default: str) -> str:
        base = self.name_generator.filter_string(name or "") or default
        return str(self.drawables.unique_name(base))

    def _add(self, plot: DataPlot) -> None:
        """Add the plot as one undo step and redraw."""
        self.canvas.undo_redo_manager.archive()
        self.drawables.add(plot)
        if getattr(self.canvas, "draw_enabled", False):
            self.canvas.draw()


def _rounded(value: Optional[float], digits: int = 12) -> Optional[float]:
    return None if value is None else round_significant(float(value), digits) + 0.0  # no -0.0


def _listed(key: str, values: Sequence[float]) -> Dict[str, List[Optional[float]]]:
    if len(values) <= MAX_LISTED_VALUES:
        return {key: [_rounded(v) for v in values]}
    return {
        f"first_{key}": [_rounded(v) for v in values[:LISTED_HEAD]],
        f"last_{key}": [_rounded(v) for v in values[-LISTED_TAIL:]],
    }


def _stability_text(stable: Optional[bool], quantity: str, known: bool = True) -> str:
    if not known:
        return (
            "; the slope of g cannot be computed there (g is not defined on both sides), so it does not "
            "tell whether nearby orbits are attracted"
        )
    if stable is None:
        return f"; |{quantity}| is about 1 there, so it does not tell whether nearby orbits are attracted"
    if stable:
        return f"; |{quantity}| < 1, so it attracts nearby orbits"
    return f"; |{quantity}| > 1, so it repels nearby orbits"


def _behaviour_note(summary: Dict[str, Any], result: Dict[str, Any]) -> str:
    """Fill the behaviour's fields into ``result`` and describe it, saying how many iterations decided it."""
    behaviour = summary["behaviour"]
    iterations = summary["iterations"]
    after = f"after {iterations} iterations"
    if behaviour == "fixed_point":
        result["fixed_point"] = _rounded(summary["fixed_point"])
        result["slope"] = _rounded(summary["slope"], 6)
        result["stable"] = summary["stable"]
        return f"{after} the orbit has settled on x = {result['fixed_point']}, where g(x) = x" + _stability_text(
            summary["stable"], f"g'(x) = {result['slope']}", summary["slope"] is not None
        )
    if behaviour == "cycle":
        result["period"] = summary["period"]
        result["cycle"] = [_rounded(v) for v in summary["cycle"]]
        result["multiplier"] = _rounded(summary["multiplier"], 6)
        result["stable"] = summary["stable"]
        return f"{after} the orbit repeats with period {summary['period']}" + _stability_text(
            summary["stable"],
            f"product of g' over the cycle = {result['multiplier']}",
            summary["multiplier"] is not None,
        )
    if behaviour == "escapes":
        result["escaped_at"] = summary["escaped_at"]
        return f"iteration {summary['escaped_at']} is infinite or too large in size, so the orbit escapes (grows without bound)"
    if behaviour == "undefined":
        result["undefined_at"] = summary["undefined_at"]
        result["undefined_input"] = _rounded(summary["undefined_input"])
        return (
            f"g is undefined at x = {result['undefined_input']} (iteration {summary['undefined_at']}): "
            "the orbit leaves the domain of g and stops there"
        )
    result["last_change"] = _rounded(summary["last_change"])
    if behaviour == "diverges":
        return f"{after} the values are still growing in size by steady or larger steps: the orbit diverges"
    pattern = summary.get("pattern")
    result["pattern"] = pattern
    if pattern == "monotone":
        return f"{after} the values still move one way (last change {result['last_change']}): it may converge or diverge slowly"
    if pattern == "shrinking" and summary.get("period"):
        result["period"] = summary["period"]
        return (
            f"{after} the gaps between values {summary['period']} steps apart are still shrinking: the orbit "
            f"may be settling slowly into a cycle of period {summary['period']}"
        )
    if pattern == "shrinking":
        return f"{after} the changes are still shrinking (last change {result['last_change']}): it may be converging slowly"
    return f"no fixed point or cycle of period up to 12 {after}: the orbit may be chaotic"


def _optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
