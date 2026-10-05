"""Classifying scenario failures (section 4.7 of the design doc), and the live run's signals.

Pure functions over stored step records, so live runs, ``--mode retrace`` and
``--regrade`` classify the same way:

- ``app``: an invariant failed (any mode), or an outcome check failed in replay.
- ``model``: an outcome check failed in a live turn, and re-executing the live
  turn's calls on a fresh session (retrace) reproduced the live canvas.
- ``nondeterministic``: the retrace did not reproduce the live canvas.
- ``known``: an expected failure (a ``known`` check or a waived invariant).
- ``infra``: the turn ended in an error or a timeout, or the scenario could not
  finish (browser, server, request cap).

Also here: which calls a retrace re-executes, comparing a retrace with its live
run, and the per-turn efficiency signals (dropped calls, calls against the
reference, recovered tool errors).
"""

from __future__ import annotations

from typing import Any, Optional

from cli.scenarios.checks import UNCOUNTED_TOOLS, call_is_error, diff_views
from cli.scenarios.geometry import CanvasView, Tolerance

CLASSES = ("app", "model", "nondeterministic", "known", "infra")
# Classes that make a live or retrace run exit non-zero: the app, not the model, is at fault.
FAILING_CLASSES = frozenset({"app", "nondeterministic"})
# Turn outcomes that are not the model's doing.
INFRA_TURN_OUTCOMES = frozenset({"error", "timeout", "request_cap", "not_started", "trace_error"})
# Calls a retrace leaves out: search_tools runs the server's tool search (which may
# call the model) and changes nothing on the canvas.
NOT_RETRACED = frozenset({"search_tools"})
# The tolerance a retrace must reproduce the live canvas within.
REPRODUCE_TOL = Tolerance(1e-9, 1e-9)


def classify_result(
    result: dict[str, Any],
    mode: str,
    turn_outcome: Optional[str] = None,
    reproduced: Optional[bool] = None,
) -> Optional[str]:
    """The class of one check result, or None when it did not fail.

    ``turn_outcome`` is how the step's live turn ended; ``reproduced`` whether a
    retrace reproduced the live canvas up to this step (None: not retraced, and
    an outcome failure then counts as ``model``).
    """
    status = result.get("status")
    if status == "xfail":
        return "known"
    if status not in ("fail", "error"):
        return None
    if result.get("kind") == "invariant" or mode == "replay":
        return "app"
    if turn_outcome in INFRA_TURN_OUTCOMES:
        return "infra"
    if reproduced is False:
        return "nondeterministic"
    return "model"


def first_difference(retrace: Optional[dict[str, Any]]) -> Optional[str]:
    """The first step at which a retrace differed from its live run, if any."""
    if not retrace:
        return None
    value = retrace.get("first_difference")
    return str(value) if value else None


def annotate_steps(
    steps: list[dict[str, Any]], mode: str, retrace: Optional[dict[str, Any]] = None
) -> list[dict[str, Any]]:
    """Set ``class`` on every failing or expected-failing result of ``steps`` (in place); returns them.

    A step at or after the retrace's first difference counts as not reproduced.
    """
    differed_at = first_difference(retrace)
    # Without a retrace (or when it could not finish) an outcome failure counts as the model's.
    reproduced: Optional[bool] = None if not retrace or retrace.get("reproduced") is None else True
    for step in steps:
        if differed_at is not None and step.get("step") == differed_at:
            reproduced = False
        turn = step.get("turn") or {}
        # Without the turn's calls (unreadable action traces) its checks cannot be judged.
        outcome = "trace_error" if turn.get("trace_error") else turn.get("outcome")
        for result in step.get("results", []):
            klass = classify_result(result, mode, outcome, reproduced)
            if klass is None:
                result.pop("class", None)
            else:
                result["class"] = klass
    return steps


def step_classes(steps: list[dict[str, Any]]) -> set[str]:
    return {str(r["class"]) for step in steps for r in step.get("results", []) if r.get("class")}


# ----------------------------------------------------------------------
# Retrace
# ----------------------------------------------------------------------


def retrace_batches(record: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """The batches a retrace re-executes for one live step record, in order.

    Each batch keeps the calls the client executed (``function_name`` and
    ``arguments``) minus ``search_tools``; batches left empty are dropped. A
    record without batches (a scripted step) gives its calls as one batch.
    """
    raw_batches = record.get("batches")
    if raw_batches is None:
        raw_batches = [{"calls": record.get("calls") or []}]
    batches: list[list[dict[str, Any]]] = []
    for batch in raw_batches:
        calls = []
        for call in batch.get("calls") or []:
            name = call.get("function_name")
            if not isinstance(name, str) or not name or name in NOT_RETRACED:
                continue
            arguments = call.get("arguments")
            calls.append({"function_name": name, "arguments": arguments if isinstance(arguments, dict) else {}})
        if calls:
            batches.append(calls)
    return batches


def _view(record: dict[str, Any]) -> CanvasView:
    return CanvasView(record.get("state") or {}, record.get("inspection"))


def compare_runs(live_steps: list[dict[str, Any]], retrace_steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Whether a retrace reproduced its live run: the canvas after every step, in order.

    Returns ``{"reproduced", "first_difference", "differences": {step: [...]}}``;
    a step missing from the retrace counts as a difference.
    """
    retraced = {str(record.get("step")): record for record in retrace_steps}
    differences: dict[str, list[str]] = {}
    first: Optional[str] = None
    for live in live_steps:
        step_id = str(live.get("step"))
        if step_id == "start":
            continue
        other = retraced.get(step_id)
        if other is None:
            found = ["step not retraced"]
        else:
            found = diff_views(_view(live), _view(other), REPRODUCE_TOL, inspect=True, include_view=True)
        if found:
            differences[step_id] = found[:20]
            first = first or step_id
    return {"reproduced": first is None, "first_difference": first, "differences": differences}


# ----------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------


def efficiency_signals(
    calls: list[dict[str, Any]], reference_calls: int, metrics: Optional[dict[str, Any]]
) -> dict[str, Any]:
    """Per-turn signals: calls against the reference, tool errors and recoveries, dropped calls.

    ``executed_calls`` leaves out ``search_tools``. A tool error counts as
    recovered when a later call in the turn succeeded. ``dropped_calls`` is the
    model's emitted calls minus the client's executions (turn metrics), i.e. the
    calls the server dropped for using tools not loaded through ``search_tools``.
    """
    counted = [call for call in calls if call.get("function_name") not in UNCOUNTED_TOOLS]
    errors = [call_is_error(call) for call in counted]
    recovered = sum(1 for index, failed in enumerate(errors) if failed and not all(errors[index + 1 :] or [True]))
    signals: dict[str, Any] = {
        "executed_calls": len(counted),
        "reference_calls": reference_calls,
        "extra_calls": len(counted) - reference_calls,
        "tool_errors": sum(errors),
        "recovered_errors": recovered,
        "search_calls": len(calls) - len(counted),
        "dropped_calls": None,
    }
    if metrics:
        emitted, executed = metrics.get("tool_calls"), metrics.get("tool_executions")
        if isinstance(emitted, int) and isinstance(executed, int):
            signals["dropped_calls"] = max(emitted - executed, 0)
    return signals
