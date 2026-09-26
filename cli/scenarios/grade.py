"""Grading a scenario step by step: invariants, checks, snapshots and bindings.

The runner feeds the grader each step's view as it runs; ``--regrade`` feeds it
the stored views from a results file. Both go through ``ScenarioGrader`` so a
regrade is exactly a rerun of the checks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

from cli.scenarios.checks import CheckContext, CheckResult, StepData, evaluate_checks, run_invariants
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
        if step is not None and step.kind == "snapshot" and step.snapshot:
            self.snapshots[step.snapshot] = view
        if step_id == "setup":
            self.snapshots["setup"] = view
        self.previous = view
        return results
