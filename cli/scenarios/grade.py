"""Grading a scenario step by step: invariants, checks, snapshots and bindings.

The runner feeds the grader each step's view as it runs; ``--regrade`` feeds it
the stored views from a results file. Both go through ``ScenarioGrader`` so a
regrade is exactly a rerun of the checks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

from cli.scenarios.checks import BatchData, CheckContext, CheckResult, StepData, evaluate_checks, run_invariants
from cli.scenarios.geometry import CanvasView, SampleRequests
from cli.scenarios.model import Scenario, Step


@dataclass
class StepRecordData:
    """The raw data one executed step produced (what results.jsonl stores)."""

    state: dict[str, Any]
    inspection: Optional[dict[str, Any]] = None
    calls: Optional[list[dict[str, Any]]] = None
    undoable: Optional[list[bool]] = None
    undo_before: Optional[int] = None
    undo_after: Optional[int] = None
    redo_before: Optional[int] = None
    redo_after: Optional[int] = None
    final_text: Optional[str] = None
    # Per-batch records of a live turn or its retrace: calls, undo depths, and the
    # trace delta (live) or the canvas after the batch (retrace: state, inspection).
    batches: Optional[list[dict[str, Any]]] = None

    def view(self) -> CanvasView:
        return CanvasView(self.state, self.inspection)

    def step_data(self, mode: str) -> Optional[StepData]:
        if self.calls is None:
            return None
        return StepData(
            calls=list(self.calls),
            undoable=list(self.undoable or []),
            undo_before=self.undo_before,
            undo_after=self.undo_after,
            redo_before=self.redo_before,
            redo_after=self.redo_after,
            final_text=self.final_text,
            mode=mode,
            batches=[batch_data(batch) for batch in self.batches] if self.batches else None,
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "StepRecordData":
        return cls(
            state=record.get("state") or {},
            inspection=record.get("inspection"),
            calls=record.get("calls"),
            undoable=record.get("undoable"),
            undo_before=record.get("undo_before"),
            undo_after=record.get("undo_after"),
            redo_before=record.get("redo_before"),
            redo_after=record.get("redo_after"),
            final_text=record.get("final_text"),
            batches=record.get("batches"),
        )


def fitted_view_fields(data: StepRecordData) -> dict[str, Any]:
    """What a display fit changed (the view's fields), as a step record stores it for ``rebase_view``."""
    inspection = data.inspection or {}
    return {
        "state": {key: value for key, value in data.state.items() if not isinstance(value, list)},
        "inspection": {key: inspection[key] for key in ("polar_radial_spacing",) if key in inspection},
    }


def batch_data(record: dict[str, Any]) -> BatchData:
    """A stored batch record as the invariants see it."""
    state = record.get("state")
    return BatchData(
        calls=list(record.get("calls") or []),
        undo_before=record.get("undo_before"),
        undo_after=record.get("undo_after"),
        redo_before=record.get("redo_before"),
        redo_after=record.get("redo_after"),
        view=CanvasView(state, record.get("inspection")) if isinstance(state, dict) else None,
        delta=record.get("delta"),
    )


class ScenarioGrader:
    """Holds a scenario's snapshots and bindings while its steps are graded in order."""

    def __init__(self, scenario: Scenario, waivers: dict[str, str], mode: str = "replay") -> None:
        self.scenario = scenario
        self.waivers = waivers
        self.mode = mode
        self.snapshots: dict[str, CanvasView] = {}
        self.bindings: dict[str, tuple[str, str]] = {}
        self.previous: Optional[CanvasView] = None

    def start(self, data: StepRecordData) -> None:
        """The reset canvas, before setup: snapshot ``start``."""
        view = data.view()
        self.snapshots["start"] = view
        self.previous = view

    def needed_samples(self, step: Optional[Step], data: StepRecordData) -> SampleRequests:
        """Function samples the step's checks will need that ``data`` lacks (a dry run)."""
        if step is None or not step.checks:
            return SampleRequests()
        view = data.view()
        ctx = CheckContext(
            view,
            data.step_data(self.mode) or StepData(mode=self.mode),
            dict(self.snapshots),
            copy.deepcopy(self.bindings),
            self.scenario.tolerance,
        )
        evaluate_checks(step.checks, ctx, "dry-run")
        return view.requests

    def grade(self, step_id: str, step: Optional[Step], data: StepRecordData) -> list[CheckResult]:
        """Invariants (for steps that ran calls) plus the step's checks; then take its snapshot."""
        view = data.view()
        step_data = data.step_data(self.mode)
        results: list[CheckResult] = []
        if step_data is not None and self.previous is not None:
            results.extend(
                run_invariants(
                    self.previous,
                    view,
                    step_data,
                    step_id,
                    self.waivers,
                    allow_large=self.scenario.allow_large,
                )
            )
        if step is not None and step.checks:
            ctx = CheckContext(
                view,
                step_data or StepData(mode=self.mode),
                self.snapshots,
                self.bindings,
                self.scenario.tolerance,
            )
            results.extend(evaluate_checks(step.checks, ctx, step_id))
        limit = step.limits.get("max_tool_calls") if step is not None and self.mode == "live" else None
        if isinstance(limit, int) and step_data is not None:
            # The turn's max_tool_calls limit is a model-quality check, graded only live.
            check = {"check": "max_tool_calls", "max": limit, "id": f"{step_id}.limit"}
            ctx = CheckContext(view, step_data, self.snapshots, self.bindings, self.scenario.tolerance)
            results.extend(evaluate_checks([check], ctx, step_id))
        if step is not None and step.kind == "snapshot" and step.snapshot:
            self.snapshots[step.snapshot] = view
        if step_id == "setup":
            self.snapshots["setup"] = view
        self.previous = view
        return results

    def rebase_view(self, data: StepRecordData) -> None:
        """Replace the last graded canvas with ``data``: the same drawables after a display-only fit.

        Attach mode zooms the view to the content after a step is graded. The
        next step then starts from the fitted view, so the canvas it is compared
        with (the invariants' previous view, and a snapshot taken at the step
        just graded) must be the fitted one too. Only the view fields are taken
        from ``data`` (the bounds, tick spacing, polar ring spacing); the
        drawables and any function samples stay those of the graded canvas.
        """
        old = self.previous
        if old is None:
            return
        state = dict(old.state)
        state.update({key: value for key, value in data.state.items() if not isinstance(value, list)})
        inspection = dict(old.inspection)
        if "polar_radial_spacing" in (data.inspection or {}):
            inspection["polar_radial_spacing"] = (data.inspection or {})["polar_radial_spacing"]
        new = StepRecordData(state=state, inspection=inspection).view()
        for name, view in list(self.snapshots.items()):
            if view is old:
                self.snapshots[name] = new
        self.previous = new
