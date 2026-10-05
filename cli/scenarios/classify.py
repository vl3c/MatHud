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
# "infra" stands for a turn after one of those in the same run.
INFRA_TURN_OUTCOMES = frozenset({"error", "timeout", "request_cap", "not_started", "trace_error", "infra"})
# Where a turn's error came from (turn metrics ``error_source``): an exception in the
# browser while handling the reply, or in the server route, is the app's failure.
APP_ERROR_SOURCES = frozenset({"client", "server"})
# The share of runs with an infrastructure failure above which a live run fails anyway:
# an app regression that breaks every turn must not pass as "infra".
DEFAULT_MAX_INFRA_RATE = 0.5
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
    if result.get("kind") == "invariant" or mode == "replay" or turn_outcome == "app_error":
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

    A step at or after the retrace's first difference counts as not reproduced,
    and every outcome failure from an infrastructure turn on is ``infra``: the
    scenario's later turns build on a turn that did not finish.
    """
    differed_at = first_difference(retrace)
    # Without a retrace (or when it could not finish) an outcome failure counts as the model's.
    reproduced: Optional[bool] = None if not retrace or retrace.get("reproduced") is None else True
    infra_seen = False
    for step in steps:
        if differed_at is not None and step.get("step") == differed_at:
            reproduced = False
        infra_seen = infra_seen or turn_is_infra(step)
        if turn_app_error(step):
            outcome: Optional[str] = "app_error"
        else:
            outcome = "infra" if infra_seen else (step.get("turn") or {}).get("outcome")
        for result in step.get("results", []):
            klass = classify_result(result, mode, outcome, reproduced)
            if klass is None:
                result.pop("class", None)
            else:
                result["class"] = klass
    return steps


def step_classes(steps: list[dict[str, Any]]) -> set[str]:
    """Failure classes of the steps' results; ``infra`` when a turn did not finish properly."""
    found = {str(r["class"]) for step in steps for r in step.get("results", []) if r.get("class")}
    if has_infra_turn(steps):
        found.add("infra")
    if any(turn_app_error(step) for step in steps):
        found.add("app")
    return found


def turn_app_error(step: dict[str, Any]) -> bool:
    """True for a live turn that ended in an error raised by the app (browser or server route)."""
    turn = step.get("turn") or {}
    return turn.get("outcome") == "error" and turn.get("error_source") in APP_ERROR_SOURCES


def turn_is_infra(step: dict[str, Any]) -> bool:
    """True for a live turn that ended in an infrastructure outcome or whose calls could not be read.

    An error the app raised itself (``turn_app_error``) is not infrastructure.
    """
    turn = step.get("turn") or {}
    if turn_app_error(step):
        return False
    return bool(turn.get("trace_error")) or turn.get("outcome") in INFRA_TURN_OUTCOMES


def has_infra_turn(steps: list[dict[str, Any]]) -> bool:
    return any(turn_is_infra(step) for step in steps)


def counted_batches(step: dict[str, Any]) -> int:
    """How many of a step's batches ran calls other than ``search_tools``."""
    return sum(
        1
        for batch in step.get("batches") or []
        if any(call.get("function_name") not in UNCOUNTED_TOOLS for call in batch.get("calls") or [])
    )


def needs_retrace(steps: list[dict[str, Any]], infra_error: Optional[str]) -> bool:
    """Whether a live run is retraced: it failed, or a turn ran several batches.

    A turn of several batches is judged batch by batch only by its retrace (I4,
    I5). Runs that did not finish, or with a turn that ended in an infrastructure
    outcome (its calls may be incomplete), are not retraced: they are ``infra``.
    """
    if infra_error or has_infra_turn(steps):
        return False
    failed = any(r.get("status") in ("fail", "error") for step in steps for r in step.get("results", []))
    return failed or any(counted_batches(step) > 1 for step in steps)


BATCH_INVARIANTS = ("I4", "I5")


def apply_batch_verdicts(live_steps: list[dict[str, Any]], retrace: Optional[dict[str, Any]]) -> None:
    """Replace the turn-level I4 and I5 of multi-batch live turns by the retrace's per-batch verdicts.

    Only when the retrace reproduced the live canvas: then its batches are the
    live batches, each judged on its own canvas and undo depths.
    """
    if not retrace or retrace.get("reproduced") is not True:
        return
    retraced = {str(record.get("step")): record for record in retrace.get("steps") or []}
    for step in live_steps:
        other = retraced.get(str(step.get("step")))
        if other is None or counted_batches(step) <= 1:
            continue
        verdicts = {r.get("name"): r for r in other.get("results", []) if r.get("kind") == "invariant"}
        results = step.get("results", [])
        for index, result in enumerate(results):
            replacement = verdicts.get(result.get("name"))
            if result.get("kind") == "invariant" and result.get("name") in BATCH_INVARIANTS and replacement:
                updated = {k: v for k, v in replacement.items() if k != "class"}
                updated["judged_by"] = "retrace"
                results[index] = updated


def retrace_invariant_failures(steps: list[dict[str, Any]]) -> list[str]:
    """Ids of the invariants a retrace broke (each batch judged on its own canvas)."""
    return [
        str(r.get("id"))
        for step in steps
        for r in step.get("results", [])
        if r.get("kind") == "invariant" and r.get("status") in ("fail", "error")
    ]


def summarize_retrace(live_steps: list[dict[str, Any]], retrace_steps: list[dict[str, Any]]) -> dict[str, Any]:
    """The comparison with the live run, the retrace's invariant failures, and its step records."""
    summary = compare_runs(live_steps, retrace_steps)
    summary["invariant_failures"] = retrace_invariant_failures(retrace_steps)
    summary["steps"] = retrace_steps
    return summary


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
            found += _undo_difference(live, other)
        if found:
            differences[step_id] = found[:20]
            first = first or step_id
    return {"reproduced": first is None, "first_difference": first, "differences": differences}


# ----------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------


def _undo_delta(record: dict[str, Any]) -> Optional[int]:
    before, after = record.get("undo_before"), record.get("undo_after")
    if isinstance(before, int) and isinstance(after, int):
        return after - before
    return None


def _undo_difference(live: dict[str, Any], retrace: dict[str, Any]) -> list[str]:
    """The undo stack must move the same way in both: the retrace's I5 verdict stands in for the live one."""
    live_delta, retrace_delta = _undo_delta(live), _undo_delta(retrace)
    if live_delta is None or retrace_delta is None or live_delta == retrace_delta:
        return []
    return [f"undo depth changed by {live_delta} live and by {retrace_delta} in the retrace"]


def infra_rate(outcomes_classes: list[set[str]]) -> Optional[float]:
    """The share of runs with an infrastructure failure (None without runs)."""
    if not outcomes_classes:
        return None
    return round(sum(1 for classes in outcomes_classes if "infra" in classes) / len(outcomes_classes), 4)


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
