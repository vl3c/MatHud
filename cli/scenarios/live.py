"""Live and retrace runners (section 4.6 of the design doc).

Live mode sends each turn's ``user`` prompt to a model through the app's own
chat path (``sendMatHudMessage``), waits for the turn while enforcing the turn
limits (``stopMatHudTurn`` on a timeout or the request cap), then grades the
canvas with the same checks and invariants as replay. Setup and scripted
(``do``) steps still run as replay.

Retrace mode re-executes a live run's executed calls, batch by batch from its
action traces, on a fresh session with no model, and compares the canvas with
the live one after every step: the tool that tells a model mistake (the
retrace reproduces the failing canvas) from app non-determinism (it does not).
A live run retraces each failing scenario right away unless told not to.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from cli.scenarios.checks import UNCOUNTED_TOOLS
from cli.scenarios.classify import (
    apply_batch_verdicts,
    efficiency_signals,
    retrace_batches,
    summarize_retrace,
)
from cli.scenarios.classify import needs_retrace as classify_needs_retrace
from cli.scenarios.grade import ScenarioGrader, StepRecordData
from cli.scenarios.model import Catalogue, Scenario, Step
from cli.scenarios.report import ResultSink, ScenarioOutcome
from cli.scenarios.runner import BrowserSession, HookError, ReplayOptions, ReplayRunner, RunStopped
from static.client.constants import MAX_RESULT_STR_LEN

DEFAULT_TURN_TIMEOUT_S = 300.0
# The client's response timeout is set this much above the turn timeout, so the harness stops first.
CLIENT_TIMEOUT_MARGIN_S = 30.0
# Client turn outcomes after which the turn's last request may still run in the server.
UNFINISHED_OUTCOMES = frozenset({"timeout", "error", "stopped"})
DEFAULT_TURN_MAX_REQUESTS = 8
POLL_INTERVAL_S = 0.2
# A turn that never starts processing within this many seconds is recorded as not started.
TURN_START_GRACE_S = 10.0
# How long to wait for the client to settle after stopMatHudTurn.
STOP_SETTLE_S = 10.0

# The text of every assistant message in the chat panel, in order (the raw
# Markdown the app keeps for its copy action, else the rendered text).
_ASSISTANT_TEXTS_JS = """
var out = [];
document.querySelectorAll('#chat-history .chat-message').forEach(function (m) {
  var sender = m.querySelector(':scope > .chat-sender');
  if (sender && sender.classList.contains('ai')) {
    out.push(typeof m._raw_message_text === 'string' ? m._raw_message_text : (m.innerText || ''));
  }
});
return out;
"""
_TRACES_JS = "return JSON.stringify(window.getActionTraces ? window.getActionTraces() : []);"
_CLEAR_TRACES_JS = "if (window.clearActionTraces) { window.clearActionTraces(); } return true;"
_LAST_TURN_JS = "return window.getMatHudLastTurnMetrics ? window.getMatHudLastTurnMetrics() : null;"


@dataclass
class LiveOptions:
    turn_timeout_s: float = DEFAULT_TURN_TIMEOUT_S
    turn_max_requests: int = DEFAULT_TURN_MAX_REQUESTS
    poll_interval_s: float = POLL_INTERVAL_S
    retrace_failures: bool = True


def turn_limits(step: Step, options: LiveOptions) -> tuple[float, int]:
    """``(timeout_s, max_requests)`` for a turn: the scenario's limits, else the run's defaults."""
    timeout = step.limits.get("timeout_s") or options.turn_timeout_s
    max_requests = step.limits.get("max_requests") or options.turn_max_requests
    return float(timeout), int(max_requests)


def turn_request_ceiling(max_requests: int) -> int:
    """The most requests a turn with cap ``max_requests`` can send.

    The harness hands the cap to the client (``sendMatHudMessage`` option
    ``max_requests``), which ends the turn instead of sending a request beyond
    it, so the cap is hard; the harness's own polling is only a backstop.
    """
    return max_requests


def planned_requests(scenarios: list[Scenario], models: int, repeats: int, options: LiveOptions) -> int:
    """The most requests a live run may send: every turn's ceiling, per model and repeat."""
    per_pass = sum(
        turn_request_ceiling(turn_limits(step, options)[1])
        for s in scenarios
        for step in s.steps
        if step.kind == "user"
    )
    return per_pass * models * repeats


class RequestBudget:
    """The running total of model requests a run has sent, against an optional hard cap.

    The cap is hard because the client enforces it: each turn is sent with a
    request limit no larger than what the cap leaves (``room``), and the client
    ends the turn instead of sending a request beyond it. A turn that ends
    unfinished (stopped, timed out, failed) counts one request more than it
    completed, since its last request may still be running in the server.
    ``reached`` is only the harness's polling backstop for a client that does
    not stop at its limit.
    """

    def __init__(self, cap: Optional[int] = None) -> None:
        self.cap = cap
        self.sent = 0

    @property
    def exhausted(self) -> bool:
        """True when no further request may be sent."""
        return self.cap is not None and self.sent >= self.cap

    def room(self) -> Optional[int]:
        """How many more requests the cap allows (None: no cap)."""
        return None if self.cap is None else max(self.cap - self.sent, 0)

    def reached(self, in_turn: int) -> bool:
        """True when ``in_turn`` requests of the running turn reach the cap."""
        return self.cap is not None and self.sent + in_turn >= self.cap

    def add(self, count: int) -> None:
        self.sent += max(count, 0)


class NullSink(ResultSink):
    """A sink that keeps nothing: an inline retrace must not add records to the live run's files."""

    def __init__(self) -> None:  # no output directory, no results.jsonl
        self.out_dir = Path(".")
        self.failures_dir = Path(".")
        self.config = {}
        self.started = time.time()
        self.outcomes = []
        self._jsonl = None


def _last_non_empty(texts: list[str]) -> Optional[str]:
    for text in reversed(texts):
        if text and text.strip():
            return text
    return None


class LiveRunner(ReplayRunner):
    """Runs scenarios against a model and records every turn."""

    mode = "live"

    def __init__(
        self,
        catalogue: Catalogue,
        session: BrowserSession,
        sink: ResultSink,
        options: ReplayOptions,
        live: LiveOptions,
        *,
        provider: str,
        reset_conversation: Callable[[], None],
        budget: Optional[RequestBudget] = None,
        wait_idle: Callable[[], float] = lambda: 0.0,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(catalogue, session, sink, options, log)
        self.live = live
        self.provider = provider
        self.reset_conversation = reset_conversation
        # Waits until the server runs no model request (a stopped turn's request goes on
        # there for a while); returns the seconds waited.
        self.wait_idle = wait_idle
        self.budget = budget or RequestBudget()
        self._turn_requests_seen = 0
        self.model: Optional[str] = None
        self.repeat = 1
        self._clock = clock
        self._sleep = sleep
        self._stop_after_record: Optional[str] = None

    def run_live(
        self,
        scenarios: list[Scenario],
        models: list[str],
        repeats: int,
        skipped: Optional[dict[str, str]] = None,
    ) -> list[ScenarioOutcome]:
        """Every scenario for every model, ``repeats`` times; serial, and stopped by the request cap."""
        outcomes: list[ScenarioOutcome] = []
        for model in models:
            for repeat in range(1, repeats + 1):
                if self.stopped:
                    return outcomes
                self.model, self.repeat = model, repeat
                if len(models) > 1 or repeats > 1:
                    self.log(f"Model {model}, repeat {repeat}/{repeats}")
                outcomes.extend(self.run(scenarios, skipped))
        return outcomes

    # ------------------------------------------------------------------
    # Scenario hooks of the replay runner
    # ------------------------------------------------------------------

    def run_scenario(self, scenario: Scenario) -> ScenarioOutcome:
        outcome = super().run_scenario(scenario)
        if self.live.retrace_failures and not self.stopped and needs_retrace(outcome):
            outcome.retrace = self.retrace(outcome)
            apply_batch_verdicts(outcome.steps, outcome.retrace)
        return outcome

    def _new_outcome(self, scenario: Scenario, attempts: int) -> ScenarioOutcome:
        return ScenarioOutcome(
            scenario, attempts=attempts, provider=self.provider, model=self.model, repeat=self.repeat
        )

    def _reset(self, scenario: Scenario) -> None:
        try:
            self.wait_idle()
            self.reset_conversation()
        except Exception as exc:
            raise HookError(f"could not reset the server conversation: {exc}") from exc
        super()._reset(scenario)

    def _execute_step(
        self, scenario: Scenario, step: Step, grader: ScenarioGrader
    ) -> tuple[StepRecordData, dict[str, Any]]:
        if step.kind != "user":
            return super()._execute_step(scenario, step, grader)
        return self._run_turn(step, grader)

    def _record(
        self,
        outcome: ScenarioOutcome,
        grader: ScenarioGrader,
        step_id: str,
        kind: str,
        step: Optional[Step],
        data: StepRecordData,
        duration: float,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        super()._record(outcome, grader, step_id, kind, step, data, duration, extra)
        if self._stop_after_record:
            reason, self._stop_after_record = self._stop_after_record, None
            raise RunStopped(reason)

    # ------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------

    def _run_turn(self, step: Step, grader: ScenarioGrader) -> tuple[StepRecordData, dict[str, Any]]:
        if self.budget.exhausted:
            raise RunStopped(f"the request cap ({self.budget.cap}) is reached")
        timeout_s, max_requests = turn_limits(step, self.live)
        # The client enforces the turn's request limit: the turn's own cap, or what the run's cap leaves.
        room = self.budget.room()
        limit = max_requests if room is None else min(max_requests, room)
        options = {
            "max_requests": limit,
            # The client's own timeouts (60 s for a first reply) must not end the turn before the harness does.
            "response_timeout_ms": int((timeout_s + CLIENT_TIMEOUT_MARGIN_S) * 1000),
        }
        before = (self.session.hook("getMatHudCanvasState", json.dumps({"inspect": True})).get("inspection")) or {}
        texts_before = len(self._assistant_texts())
        self._js(_CLEAR_TRACES_JS)
        completed_before = int(self.session.hook("getMatHudTurnStatus").get("completed_turns") or 0)
        self._turn_requests_seen = 0
        accounted = False
        try:
            # From here on the turn may have sent requests: whatever happens, they are counted.
            self.session.hook("sendMatHudMessage", step.user or "", self.model or "", json.dumps(options))
            started = self._clock()
            stop_reason = self._wait_for_turn(completed_before, started, timeout_s, limit)
            wall = self._clock() - started
            metrics = self._last_turn_metrics(completed_before)
            client_outcome = (metrics or {}).get("outcome")
            # A stopped, timed-out or failed turn may leave its last request running in the server.
            unfinished = bool(stop_reason) or client_outcome in UNFINISHED_OUTCOMES
            drain_s = self.wait_idle() if unfinished else 0.0
            completed = max(int((metrics or {}).get("requests") or 0), self._turn_requests_seen)
            requests_sent = completed + (1 if unfinished else 0)
            self.budget.add(requests_sent)
            accounted = True
            if stop_reason is None and client_outcome == "max_requests" and limit < max_requests:
                stop_reason = "request_cap"  # the client stopped at what the run's cap left, not the turn's cap
        finally:
            if not accounted:
                # The turn broke off (hook error, browser hang): count what was seen plus one in flight.
                self.budget.add(self._turn_requests_seen + 1)

        traces, trace_error = self._traces()
        texts = self._assistant_texts()[texts_before:]
        batches: list[dict[str, Any]] = [
            {"calls": mark_truncated(trace.get("tool_calls") or []), "delta": trace.get("state_delta"),
             "duration_ms": trace.get("total_duration_ms")}
            for trace in traces
        ]  # fmt: skip
        calls: list[dict[str, Any]] = [call for batch in batches for call in batch["calls"]]
        data = self._with_samples(grader, step, None)
        after = data.inspection or {}
        data.calls = calls
        data.undo_before, data.redo_before = before.get("undo_depth"), before.get("redo_depth")
        data.undo_after, data.redo_after = after.get("undo_depth"), after.get("redo_depth")
        data.final_text = _last_non_empty(texts)
        data.batches = batches
        turn = {
            "outcome": stop_reason or (metrics or {}).get("outcome") or "unknown",
            "stop_reason": stop_reason,
            "wall_time_s": round(wall, 3),
            "requests_sent": requests_sent,
            "drain_s": round(drain_s, 3),
            "limits": {"timeout_s": timeout_s, "max_requests": max_requests, "client_max_requests": limit},
            "metrics": metrics,
        }
        error_source = (metrics or {}).get("error_source")
        if error_source:
            turn["error_source"] = error_source
        if trace_error:
            # The turn's calls are unknown, so its checks cannot be trusted (classified infra).
            turn["trace_error"] = trace_error
            self.log(f"  {step.id}: could not read the action traces: {trace_error}")
        extra = {
            "provider": self.provider,
            "model": self.model,
            "repeat": self.repeat,
            "assistant_texts": texts,
            "turn": turn,
            "signals": efficiency_signals(calls, len(step.calls), metrics),
        }
        if stop_reason == "request_cap":
            self._stop_after_record = f"the request cap ({self.budget.cap}) was reached"
        counted = sum(1 for c in calls if c.get("function_name") not in UNCOUNTED_TOOLS)
        self.log(f"  {step.id}: {turn['outcome']} in {wall:.1f} s, {requests_sent} requests, {counted} calls")
        return data, extra

    def _wait_for_turn(
        self, completed_before: int, started: float, timeout_s: float, max_requests: int
    ) -> Optional[str]:
        """Poll the turn until it ends; returns why the harness stopped it, or None when it finished."""
        seen_processing = False
        while True:
            status = self.session.hook("getMatHudTurnStatus")
            processing = bool(status.get("processing"))
            completed = int(status.get("completed_turns") or 0)
            seen_processing = seen_processing or processing
            if not processing and completed > completed_before:
                return None
            requests = int(status.get("requests") or 0)
            self._turn_requests_seen = max(self._turn_requests_seen, requests)
            elapsed = self._clock() - started
            reason: Optional[str] = None
            # Backstops: the client ends the turn at its request limit itself, so a turn still
            # running once that many requests completed has a client that did not.
            if processing and self.budget.reached(requests):
                reason = "request_cap"
            elif processing and requests >= max_requests:
                reason = "max_requests"
            elif elapsed > timeout_s:
                reason = "timeout"
            elif not seen_processing and elapsed > TURN_START_GRACE_S:
                reason = "not_started"
            if reason:
                self._stop_turn()
                return reason
            self._sleep(self.live.poll_interval_s)

    def _stop_turn(self) -> None:
        self.session.hook("stopMatHudTurn")
        settle_until = self._clock() + STOP_SETTLE_S
        while self._clock() < settle_until:
            if not self.session.hook("getMatHudTurnStatus").get("processing"):
                return
            self._sleep(self.live.poll_interval_s)

    def _js(self, script: str) -> Any:
        browser = self.session.browser
        return self.session.call(lambda: browser.execute_js(script, timeout=int(self.session.timeout_s)))

    def _assistant_texts(self) -> list[str]:
        texts = self._js(_ASSISTANT_TEXTS_JS)
        return [str(text) for text in texts] if isinstance(texts, list) else []

    def _traces(self) -> tuple[list[dict[str, Any]], Optional[str]]:
        """The turn's action traces, and an error when the app could not export them."""
        raw = self._js(_TRACES_JS)
        traces = json.loads(raw) if isinstance(raw, str) else None
        if isinstance(traces, dict) and "error" in traces:
            return [], str(traces["error"])
        if not isinstance(traces, list):
            return [], f"unexpected getActionTraces() reply: {str(raw)[:120]}"
        return [trace for trace in traces if isinstance(trace, dict)], None

    def _last_turn_metrics(self, completed_before: int) -> Optional[dict[str, Any]]:
        raw = self._js(_LAST_TURN_JS)
        metrics = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(metrics, dict) or int(metrics.get("turn_id") or 0) <= completed_before:
            return None
        return metrics

    # ------------------------------------------------------------------
    # Retrace of a failing scenario
    # ------------------------------------------------------------------

    def retrace(self, live: ScenarioOutcome) -> dict[str, Any]:
        """Re-execute ``live``'s calls on a fresh session and compare the canvases."""
        runner = RetraceRunner(self.catalogue, self.session, NullSink(), self.options, log=lambda _line: None)
        runner.save_artifacts = False
        retraced = runner.retrace_outcome(live.scenario, live.steps, live.provider, live.model, live.repeat)
        return retrace_summary(live.steps, retraced)


def retrace_summary(live_steps: list[dict[str, Any]], retraced: ScenarioOutcome) -> dict[str, Any]:
    """What a retrace showed: whether it reproduced the live canvas, its invariant failures and its steps."""
    if retraced.infra_error:
        return {"reproduced": None, "error": retraced.infra_error}
    return summarize_retrace(live_steps, retraced.steps)


def needs_retrace(outcome: ScenarioOutcome) -> bool:
    return classify_needs_retrace(outcome.steps, outcome.infra_error)


def mark_truncated(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag results the action-trace export cut short (I4's naming rule cannot judge those)."""
    for call in calls:
        result = call.get("result")
        if isinstance(result, str) and len(result) == MAX_RESULT_STR_LEN + 3 and result.endswith("..."):
            call["result_truncated"] = True
    return calls


class RetraceRunner(ReplayRunner):
    """Re-executes a live run's calls batch by batch, with no model.

    A turn's batches run in one undo group, as the chat turn ran them, so the
    retrace's undo stack moves as the live one did.
    """

    mode = "retrace"

    def __init__(
        self,
        catalogue: Catalogue,
        session: BrowserSession,
        sink: ResultSink,
        options: Optional[ReplayOptions] = None,
        log: Callable[[str], None] = print,
    ) -> None:
        super().__init__(catalogue, session, sink, options, log)
        self.save_artifacts = True
        self._source: dict[str, dict[str, Any]] = {}
        self._meta: tuple[Optional[str], Optional[str], int] = (None, None, 1)

    def retrace_outcome(
        self,
        scenario: Scenario,
        live_steps: list[dict[str, Any]],
        provider: Optional[str],
        model: Optional[str],
        repeat: int,
    ) -> ScenarioOutcome:
        """Retrace one live run of ``scenario`` (its stored step records)."""
        self._source = {str(record.get("step")): record for record in live_steps}
        self._meta = (provider, model, repeat)
        return self.run_scenario(scenario)

    def _new_outcome(self, scenario: Scenario, attempts: int) -> ScenarioOutcome:
        provider, model, repeat = self._meta
        return ScenarioOutcome(scenario, attempts=attempts, provider=provider, model=model, repeat=repeat)

    def _save_artifacts(self, scenario_id: str, step_id: str, data: StepRecordData) -> dict[str, str]:
        return super()._save_artifacts(scenario_id, step_id, data) if self.save_artifacts else {}

    def _execute_step(
        self, scenario: Scenario, step: Step, grader: ScenarioGrader
    ) -> tuple[StepRecordData, dict[str, Any]]:
        if step.kind != "user":
            return super()._execute_step(scenario, step, grader)
        live = self._source.get(step.id)
        if live is None:
            raise HookError(f"the live run has no record of step {step.id}")
        batches = self._run_turn_batches(retrace_batches(live))
        data = self._turn_data(grader, step, batches)
        data.final_text = live.get("final_text")
        extra = {
            "provider": live.get("provider"),
            "model": live.get("model"),
            "repeat": live.get("repeat"),
            "turn": live.get("turn"),
            "retraced_batches": len(batches),
        }
        return data, extra
