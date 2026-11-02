"""Sequence plot: the terms u(n) of a sequence, drawn as dots at (n, u(n)).

The term is explicit (``1/n``) or recursive (``u(n-1) + u(n-2)`` from ``initial_terms``);
``utils.sequences`` parses it and computes the terms. The plot saves its inputs and a
``summary`` of the terms for the canvas text; a restore recomputes the terms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from drawables.data_plot import DataPlot
from utils.sequences import SequenceRule, saved_summary, sequence_summary, sequence_terms


class SequencePlot(DataPlot):
    """Dots at (n, u(n)) for n from ``start`` to ``end`` where u(n) is defined.

    Attributes:
        rule: The parsed term (``expression``, ``order``).
        start, end: The index range.
        initial_terms: u(start), ... for a recursion (a tuple), empty for an explicit term.
        summary: ``n_values``, ``terms``, ``min``, ``max``, ``trend``, ``undefined_at``,
            ``stopped_at`` (lists as tuples).
    """

    _SHARED_ATTRIBUTES = ("summary", "rule")

    def __init__(
        self,
        name: str,
        *,
        rule: SequenceRule,
        start: int,
        end: int,
        initial_terms: Optional[Sequence[float]] = None,
        color: Optional[str] = None,
    ) -> None:
        super().__init__(name, plot_type="sequence", color=color)
        self.rule = rule
        self.start = int(start)
        self.end = int(end)
        self.initial_terms: Tuple[float, ...] = tuple(float(v) for v in (initial_terms or ()))
        terms = sequence_terms(rule, self.start, self.end, list(self.initial_terms) if rule.order else None)
        summary = sequence_summary(terms)
        self.summary: Dict[str, Any] = {
            key: tuple(value) if isinstance(value, list) else value for key, value in summary.items()
        }

    def get_class_name(self) -> str:
        return "SequencePlot"

    @property
    def expression(self) -> str:
        return self.rule.expression

    def points(self) -> List[Tuple[float, float]]:
        """The (n, u(n)) pairs drawn."""
        return [(float(n), value) for n, value in zip(self.summary["n_values"], self.summary["terms"])]

    def extent_points(self) -> List[Tuple[float, float]]:
        summary = self.summary
        if not summary["terms"]:
            return [(float(self.start), 0.0), (float(self.end), 0.0)]
        return [
            (float(summary["n_values"][0]), float(summary["min"])),
            (float(summary["n_values"][-1]), float(summary["max"])),
        ]

    def _plot_args(self) -> Dict[str, Any]:
        summary = saved_summary(self.summary)
        return {
            "expression": self.rule.expression,
            "start": self.start,
            "end": self.end,
            "initial_terms": list(self.initial_terms) if self.rule.order else None,
            "summary": summary,
        }
