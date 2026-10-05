"""Tests for live mode, retrace and failure classification, with a fake chat (no browser, no model).

Nothing here reaches a provider or a llama-server: the browser is a fake whose
"model" runs scripted tool batches, and the command's server and HTTP helpers
are patched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest
from click.testing import CliRunner

from cli.main import cli
from cli.scenarios import command as command_module
from cli.scenarios.checks import BatchData, CheckResult, StepData, run_invariants
from cli.scenarios.classify import (
    annotate_steps,
    apply_batch_verdicts,
    needs_retrace,
    classify_result,
    compare_runs,
    efficiency_signals,
    retrace_batches,
)
from cli.scenarios.geometry import CanvasView
from cli.scenarios.live import (
    LiveOptions,
    LiveRunner,
    RequestBudget,
    RetraceRunner,
    mark_truncated,
    planned_requests,
    retrace_summary,
)
from cli.scenarios.live_config import (
    GuardError,
    LiveSettings,
    check_models,
    check_request_cap,
    live_plan,
    recorded_env,
    server_env,
)
from cli.scenarios.model import Catalogue, ScenarioError, load_catalogue
from cli.scenarios.report import ResultSink, ScenarioOutcome, regrade
from cli.scenarios.runner import BrowserSession, ReplayOptions

from server_tests.test_cli.scenario_states import point, state
from server_tests.test_cli.test_scenario_runner import FakeBrowser, write_catalogue

PROMPT = "Point P at (1, 2)."


def create(x: float, y: float, name: str = "P") -> dict[str, Any]:
    return {"tool": "create_point", "args": {"x": x, "y": y, "name": name}}


SEARCH = {"tool": "search_tools", "args": {"query": "point"}}


class FakeChatBrowser(FakeBrowser):
    """FakeBrowser plus a scripted chat: each status poll completes one model request.

    ``script`` maps a prompt to the tool batches the "model" sends; after the
    last batch one more request gives the final answer. ``loop`` keeps sending
    search batches forever; ``hang`` never completes a request.
    """

    def __init__(
        self,
        script: Optional[dict[str, list[list[dict[str, Any]]]]] = None,
        final_text: str = "Done: P is at (1, 2).",
        loop: bool = False,
        hang: bool = False,
        dropped: int = 0,
        drift_after_turn: float = 0.0,
    ) -> None:
        super().__init__()
        self.script = script if script is not None else {PROMPT: [[SEARCH], [create(1, 2)]]}
        self.final_text = final_text
        self.loop = loop
        self.hang = hang
        # The status hook fails once this many requests completed (a page that broke mid-turn).
        self.fail_after: Optional[int] = None
        # Every tool batch runs, but the final answer never arrives.
        self.stall_before_answer = False
        self.options: list[dict[str, Any]] = []
        self.limit: Optional[int] = None
        # False: a client that ignores the request limit (the harness backstop must stop it).
        self.honour_limit = True
        # (outcome, error source): the turn ends that way after one request (e.g. a client timeout).
        self.end_with: Optional[tuple[str, Optional[str]]] = None
        self.dropped = dropped
        self.drift_after_turn = drift_after_turn
        self.processing = False
        self.completed = 0
        self.requests = 0
        self.pending: list[list[dict[str, Any]]] = []
        self.traces: list[dict[str, Any]] = []
        self.texts: list[str] = []
        self.metrics: Optional[dict[str, Any]] = None
        self.sent: list[tuple[str, str]] = []
        self.stops = 0
        self.executions = 0

    def call_hook(self, name: str, *args: Any, timeout: int = 30) -> dict[str, Any]:
        if name == "sendMatHudMessage":
            if self.processing:
                return {"status": "busy"}
            self.sent.append((args[0], args[1]))
            self.options.append(json.loads(args[2]) if len(args) > 2 else {})
            # Like the client: no request beyond the turn's limit (None: the client ignores limits).
            self.limit = self.options[-1].get("max_requests") if self.honour_limit else None
            self.processing, self.requests, self.executions = True, 0, 0
            if self.end_with is not None:
                self.requests = 1
                self._run_batch(self.pending.pop(0)) if self.pending else None
                self._finish(*self.end_with)
            self.pending = [list(batch) for batch in self.script.get(args[0], [])]
            return {"status": "started"}
        if name == "getMatHudTurnStatus":
            if self.processing and self.fail_after is not None and self.requests >= self.fail_after:
                return {"status": "error", "error": "the page broke"}
            if self.processing and not self.hang:
                self._advance()
            return {"processing": self.processing, "completed_turns": self.completed, "requests": self.requests}
        if name == "stopMatHudTurn":
            self.stops += 1
            if self.processing:
                self._finish("stopped")
            return {"status": "stopped"}
        if name == "runMatHudToolCalls" and self.processing:
            return {"status": "busy", "error": "a chat turn is running"}
        if name == "resetMatHudSession":
            self.texts, self.traces = [], []
        return super().call_hook(name, *args, timeout=timeout)

    def _advance(self) -> None:
        self.requests += 1
        if self.loop:
            self._run_batch([SEARCH])
        elif self.pending:
            self._run_batch(self.pending.pop(0))
        else:
            self._answer()
            return
        if self.limit is not None and self.requests >= self.limit:
            self._finish("max_requests")  # the client ends the turn instead of sending one more

    def _answer(self) -> None:
        if self.stall_before_answer:
            self.requests -= 1  # the final answer never comes
        else:
            self.texts.append(self.final_text)
            self._finish("stop")

    def _run_batch(self, batch: list[dict[str, Any]]) -> None:
        before = {p["name"] for p in self.points}
        traced = [self._run(call["tool"], call["args"]) for call in batch]
        after = {p["name"] for p in self.points}
        self.executions += len(traced)
        self.traces.append(
            {
                "tool_calls": traced,
                "state_delta": {"added": sorted(after - before), "removed": sorted(before - after), "modified": []},
                "total_duration_ms": 1.0,
            }
        )

    def _finish(self, outcome: str, error_source: Optional[str] = None) -> None:
        self.processing = False
        self.completed += 1
        self.metrics = {
            "turn_id": self.completed,
            "outcome": outcome,
            "requests": self.requests,
            "tool_calls": self.executions + self.dropped,
            "tool_executions": self.executions,
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "time_to_first_token_s": 0.5,
        }
        if error_source is not None:
            self.metrics["error_source"] = error_source
        self.moved_by_bug = self.drift_after_turn

    def execute_js(self, script: str, *args: Any, timeout: int = 30) -> Any:
        if "JSON.stringify(window.getActionTraces" in script:
            return json.dumps(self.traces)
        if "clearActionTraces" in script:
            self.traces = []
            return True
        if "getMatHudLastTurnMetrics" in script:
            return json.dumps(self.metrics) if self.metrics else None
        if "chat-history" in script:
            return list(self.texts)
        raise AssertionError(f"unexpected script {script[:60]}")


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def live_runner(
    catalogue: Catalogue,
    browser: FakeChatBrowser,
    out: Path,
    *,
    live: Optional[LiveOptions] = None,
    budget: Optional[RequestBudget] = None,
    resets: Optional[list[int]] = None,
    idle_waits: Optional[list[int]] = None,
) -> tuple[LiveRunner, ResultSink]:
    sink = ResultSink(out, {"mode": "live", "provider": "local"})
    session = BrowserSession(lambda: browser, timeout_s=5)
    clock = FakeClock()
    calls = resets if resets is not None else []
    runner = LiveRunner(
        catalogue,
        session,
        sink,
        ReplayOptions(step_timeout_s=5, retries=0),
        live or LiveOptions(turn_timeout_s=30, poll_interval_s=1),
        provider="local",
        reset_conversation=lambda: calls.append(1),
        budget=budget,
        wait_idle=lambda: float(len(idle_waits.append(1) or idle_waits)) if idle_waits is not None else 0.0,
        log=lambda _line: None,
        clock=clock.time,
        sleep=clock.sleep,
    )
    return runner, sink


@pytest.fixture
def catalogue(tmp_path: Path) -> Catalogue:
    directory = tmp_path / "scenarios"
    directory.mkdir()
    return write_catalogue(directory)


def geo90(catalogue: Catalogue) -> list[Any]:
    return catalogue.select(ids=["GEO-90"])


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------


class TestClassification:
    def test_result_classes(self) -> None:
        fail = {"status": "fail", "kind": "check"}
        assert classify_result({"status": "pass", "kind": "check"}, "live") is None
        assert classify_result({"status": "xfail", "kind": "check"}, "live") == "known"
        assert classify_result({"status": "fail", "kind": "invariant"}, "live") == "app"
        assert classify_result(fail, "replay") == "app"
        assert classify_result(fail, "live") == "model"
        assert classify_result(fail, "live", reproduced=True) == "model"
        assert classify_result(fail, "live", reproduced=False) == "nondeterministic"
        assert classify_result(fail, "live", turn_outcome="timeout") == "infra"
        assert classify_result(fail, "live", turn_outcome="error") == "infra"
        assert classify_result(fail, "live", turn_outcome="max_requests") == "model"
        assert classify_result({"status": "error", "kind": "invariant"}, "retrace") == "app"
        assert classify_result(fail, "retrace") == "model"

    def test_annotate_marks_steps_from_the_first_difference(self) -> None:
        steps = [
            {"step": "setup", "results": []},
            {"step": "t1", "turn": {"outcome": "stop"}, "results": [{"status": "fail", "kind": "check"}]},
            {"step": "t2", "turn": {"outcome": "stop"}, "results": [{"status": "fail", "kind": "check"}]},
        ]
        annotate_steps(steps, "live", {"reproduced": False, "first_difference": "t2"})
        assert steps[1]["results"][0]["class"] == "model"
        assert steps[2]["results"][0]["class"] == "nondeterministic"
        # A retrace that could not finish leaves the failures to the model.
        annotate_steps(steps, "live", {"reproduced": None, "error": "boom"})
        assert steps[2]["results"][0]["class"] == "model"

    def test_outcome_classes_include_retrace_findings(self, catalogue: Catalogue) -> None:
        outcome = ScenarioOutcome(geo90(catalogue)[0], model="m")
        outcome.retrace = {"reproduced": False, "first_difference": "t1", "invariant_failures": ["t1.I5"]}
        assert outcome.classes() == {"nondeterministic", "app"}
        assert outcome.app_failure()
        outcome.retrace, outcome.infra_error = None, "browser died"
        assert outcome.classes() == {"infra"} and not outcome.app_failure()


class TestRetraceExtraction:
    def test_batches_drop_search_tools_and_empty_batches(self) -> None:
        record = {
            "batches": [
                {"calls": [{"function_name": "search_tools", "arguments": {"query": "x"}, "result": []}]},
                {
                    "calls": [
                        {"function_name": "create_point", "arguments": {"x": 1}, "result": "ok"},
                        {"function_name": "search_tools", "arguments": {}},
                        {"function_name": "zoom", "arguments": {"_raw": "bad"}, "is_error": True},
                    ]
                },
                {"calls": [{"function_name": "undo", "arguments": None}]},
            ]
        }
        assert retrace_batches(record) == [
            [
                {"function_name": "create_point", "arguments": {"x": 1}},
                {"function_name": "zoom", "arguments": {"_raw": "bad"}},
            ],
            [{"function_name": "undo", "arguments": {}}],
        ]

    def test_record_without_batches_is_one_batch(self) -> None:
        record = {"calls": [{"function_name": "undo", "arguments": {}}]}
        assert retrace_batches(record) == [[{"function_name": "undo", "arguments": {}}]]
        assert retrace_batches({"calls": []}) == []

    def test_compare_runs(self) -> None:
        live = [
            {"step": "start", "state": state()},
            {"step": "t1", "state": state(point("P", 1, 2))},
            {"step": "t2", "state": state(point("P", 1, 2))},
        ]
        same = [dict(record) for record in live]
        assert compare_runs(live, same) == {"reproduced": True, "first_difference": None, "differences": {}}
        moved = [live[0], {"step": "t1", "state": state(point("P", 1.5, 2))}, live[2]]
        result = compare_runs(live, moved)
        assert result["reproduced"] is False and result["first_difference"] == "t1"
        assert result["differences"]["t1"] == ["Points P changed"]
        missing = compare_runs(live, live[:2])
        assert missing["first_difference"] == "t2" and missing["differences"]["t2"] == ["step not retraced"]


class TestSignals:
    def test_efficiency_signals(self) -> None:
        calls = [
            {"function_name": "search_tools", "result": []},
            {"function_name": "update_point", "result": "Error: cannot move", "is_error": True},
            {"function_name": "translate_object", "result": "ok"},
            {"function_name": "delete_point", "result": "Error: no", "is_error": True},
        ]
        signals = efficiency_signals(calls, 1, {"tool_calls": 6, "tool_executions": 4})
        assert signals == {
            "executed_calls": 3,
            "reference_calls": 1,
            "extra_calls": 2,
            "tool_errors": 2,
            "recovered_errors": 1,
            "search_calls": 1,
            "dropped_calls": 2,
        }
        assert efficiency_signals([], 2, None)["dropped_calls"] is None


# ----------------------------------------------------------------------
# Invariants over several batches
# ----------------------------------------------------------------------


def call(tool: str, result: Any = "Call successful!", is_error: bool = False, **arguments: Any) -> dict[str, Any]:
    return {"function_name": tool, "arguments": arguments, "result": result, "is_error": is_error}


def by_id(results: list[CheckResult], invariant: str) -> CheckResult:
    return next(r for r in results if r.name == invariant)


class TestBatchedInvariants:
    empty = CanvasView(state())
    one = CanvasView(state(point("A", 0, 0)))
    two = CanvasView(state(point("A", 0, 0), point("B", 1, 0)))

    def turn(self, undo_after: int, *batches: BatchData) -> StepData:
        calls = [c for batch in batches for c in batch.calls]
        return StepData(calls=calls, undo_before=0, undo_after=undo_after, mode="live", batches=list(batches))

    def test_live_turn_undo_range(self) -> None:
        first = BatchData(calls=[call("create_point", x=0, y=0)], delta={"added": ["A"]})
        second = BatchData(calls=[call("create_point", x=1, y=0)], delta={"added": ["B"]})
        search = BatchData(calls=[call("search_tools", [])], delta={})
        assert by_id(run_invariants(self.empty, self.two, self.turn(2, search, first, second), "t1"), "I5").passed
        low = by_id(run_invariants(self.empty, self.two, self.turn(1, first, second), "t1"), "I5")
        assert low.status == "fail" and "expected 2 to 2" in low.message
        assert by_id(run_invariants(self.empty, self.two, self.turn(3, first, second), "t1"), "I5").status == "fail"
        # A batch whose trace names no change may or may not have archived (colours are not in the delta).
        recolour = BatchData(calls=[call("update_point", name="A", new_color="red")], delta={"modified": []})
        for added in (1, 2):
            step = self.turn(added, first, recolour)
            assert by_id(run_invariants(self.empty, self.one, step, "t1"), "I5").passed

    def test_live_turn_leaves_success_claims_to_the_retrace(self) -> None:
        made = BatchData(calls=[call("create_point", x=0, y=0, name="A")], delta={"added": ["A"]})
        removed = BatchData(calls=[call("delete_point", x=0, y=0)], delta={"removed": ["A"]})
        step = self.turn(2, made, removed)
        assert by_id(run_invariants(self.empty, self.empty, step, "t1"), "I4").passed
        # Batches that cancel out (zoom, then undo) leave the canvas as it was: not a false claim.
        zoom = BatchData(calls=[call("zoom", center_x=0, center_y=0, range_val=5)], delta={})
        undo = BatchData(calls=[call("undo")], delta={})
        step = self.turn(0, zoom, undo)
        assert by_id(run_invariants(self.one, self.one, step, "t1"), "I4").passed
        recolour = BatchData(calls=[call("update_point", name="A", new_color="red")], delta={})
        step = self.turn(1, recolour, BatchData(calls=[call("undo")], delta={}))
        assert by_id(run_invariants(self.one, self.one, step, "t1"), "I4").passed
        # Every call failing while the drawables changed is still caught for the turn as a whole.
        failed = [BatchData(calls=[call("create_point", "Error: no", is_error=True)], delta={}) for _ in range(2)]
        result = by_id(run_invariants(self.empty, self.one, self.turn(1, *failed), "t1"), "I4")
        assert result.status == "fail" and "every call failed" in result.message

    def test_retraced_batches_are_judged_one_by_one(self) -> None:
        first = BatchData(
            calls=[call("create_point", x=0, y=0)], undo_before=0, undo_after=1, view=self.one, delta=None
        )
        double = BatchData(calls=[call("create_point", x=1, y=0)], undo_before=1, undo_after=3, view=self.two)
        step = self.turn(3, first, double)
        result = by_id(run_invariants(self.empty, self.two, step, "t1"), "I5")
        assert result.status == "fail" and result.message.startswith("batch 2: ")
        double.undo_after = 2
        assert by_id(run_invariants(self.empty, self.two, self.turn(2, first, double), "t1"), "I5").passed


# ----------------------------------------------------------------------
# Loader limits
# ----------------------------------------------------------------------


def test_turn_limits_are_validated(tmp_path: Path) -> None:
    scenario = {
        "id": "GEO-92",
        "title": "limits",
        "steps": [
            {
                "user": "x",
                "reference": [create(1, 2)],
                "limits": {"max_tool_calls": 2, "timeout_s": 0, "max_requests": 2.5, "retries": 1},
            }
        ],
    }
    (tmp_path / "geometry.json").write_text(json.dumps({"schema": 1, "area": "GEO", "scenarios": [scenario]}))
    (tmp_path / "known_bugs.json").write_text(json.dumps({"bugs": {}, "invariant_waivers": {}}))
    with pytest.raises(ScenarioError) as raised:
        load_catalogue(tmp_path)
    problems = "\n".join(raised.value.problems)
    assert "unknown limit 'retries'" in problems
    assert "limit timeout_s must be a positive number" in problems
    assert "limit max_requests must be a whole number" in problems
    assert "max_tool_calls" not in problems


# ----------------------------------------------------------------------
# Guards, pinned environment and plan
# ----------------------------------------------------------------------


AVAILABLE = {
    "local_agent": [{"id": "qwen-local"}],
    "openai": [],
    "openrouter_paid": [{"id": "deepseek/deepseek-v4.1-flash"}],
    "openrouter_free": [{"id": "some/model:free"}],
}


class TestGuards:
    def test_unregistered_model_aborts(self) -> None:
        with pytest.raises(GuardError, match="falls back to a paid provider"):
            check_models(AVAILABLE, "local", ["stale-model"])
        with pytest.raises(GuardError, match="not registered under openrouter_paid or openrouter_free"):
            check_models(AVAILABLE, "openrouter", ["qwen-local"])
        with pytest.raises(GuardError, match="no local_agent model"):
            check_models({"local_agent": []}, "local", [])
        with pytest.raises(GuardError, match="needs --models"):
            check_models(AVAILABLE, "openrouter", [])

    def test_registered_models(self) -> None:
        assert check_models(AVAILABLE, "local", []) == ["qwen-local"]
        assert check_models(AVAILABLE, "local", ["qwen-local", "qwen-local"]) == ["qwen-local"]
        assert check_models(AVAILABLE, "openrouter", ["some/model:free"]) == ["some/model:free"]

    def test_request_cap(self) -> None:
        check_request_cap(10, 10)
        with pytest.raises(GuardError, match="nothing was sent"):
            check_request_cap(11, 10)

    def test_openrouter_refuses_uncounted_search_requests(self) -> None:
        LiveSettings(provider="openrouter").validate()
        with pytest.raises(GuardError, match="cannot count"):
            LiveSettings(provider="openrouter", tool_search_mode="hybrid").validate()
        LiveSettings(provider="local", tool_search_mode="api").validate()

    def test_server_env_pins_and_blanks_keys(self) -> None:
        env = server_env(LiveSettings(local_reasoning_effort="high", canvas_budget=0), "/tmp/ws")
        assert env["MATHUD_TOOL_EXPOSURE"] == "search"
        assert env["MATHUD_CANVAS_FORMAT"] == "text"
        assert env["MATHUD_CANVAS_BUDGET_TOKENS"] == "0"
        assert env["TOOL_SEARCH_MODE"] == "hybrid"
        assert env["MATHUD_WORKSPACES_DIR"] == "/tmp/ws"
        assert env["REQUIRE_AUTH"] == "false"
        assert env["MATHUD_LOCAL_REASONING_EFFORT"] == "high"
        assert env["MATHUD_OPENROUTER_MAX_RETRIES"] == "0"
        assert all(env[key] == "" for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"))
        remote = server_env(LiveSettings(provider="openrouter"), "/tmp/ws")
        assert "OPENROUTER_API_KEY" not in remote and remote["OPENAI_API_KEY"] == ""
        assert remote["TOOL_SEARCH_MODE"] == "local" and remote["MATHUD_CANVAS_BUDGET_TOKENS"] == ""
        assert recorded_env(env)["OPENAI_API_KEY"] == "<blank>"

    def test_planned_requests_use_scenario_limits(self, catalogue: Catalogue) -> None:
        scenarios = geo90(catalogue)
        options = LiveOptions(turn_max_requests=4)
        # The client enforces the cap, so a turn capped at N requests sends at most N.
        assert planned_requests(scenarios, 2, 3, options) == 4 * 2 * 3
        scenarios[0].steps[0].limits = {"max_requests": 2}
        assert planned_requests(scenarios, 1, 1, options) == 2

    def test_plan_cost_estimate(self) -> None:
        settings = LiveSettings(provider="openrouter")
        plan = live_plan(
            scenarios=11,
            turns=18,
            models=["deepseek/deepseek-v4.1-flash", "unpriced/model"],
            repeats=1,
            planned=288,
            settings=settings,
            max_requests=250,
        )
        assert plan["within_cap"] is False and plan["turns"] == 36
        assert plan["estimated_prompt_tokens_per_request"] > 4000
        assert plan["estimated_cost_usd"]["deepseek/deepseek-v4.1-flash"] > 0
        assert plan["estimated_cost_usd"]["unpriced/model"] is None
        local = live_plan(
            scenarios=1, turns=1, models=["m"], repeats=1, planned=8, settings=LiveSettings(), max_requests=None
        )
        assert "estimated_cost_usd" not in local and local["within_cap"]

    def test_budget(self) -> None:
        budget = RequestBudget(5)
        budget.add(3)
        assert not budget.exhausted and not budget.reached(1) and budget.reached(2)
        budget.add(2)
        assert budget.exhausted
        assert not RequestBudget(None).reached(10_000)


# ----------------------------------------------------------------------
# Live runner with a fake chat
# ----------------------------------------------------------------------


class TestLiveRunner:
    def test_turn_is_recorded_and_graded(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser(dropped=1)
        resets: list[int] = []
        runner, sink = live_runner(catalogue, browser, tmp_path / "out", resets=resets)
        outcomes = runner.run_live(geo90(catalogue), ["qwen-local"], repeats=2)
        summary = sink.close(catalogue)
        assert [o.status for o in outcomes] == ["pass", "pass"] and [o.repeat for o in outcomes] == [1, 2]
        assert browser.sent == [(PROMPT, "qwen-local")] * 2 and len(resets) == 2
        turn = next(step for step in outcomes[0].steps if step["step"] == "t1")
        assert turn["mode"] == "live" and turn["model"] == "qwen-local" and turn["user"] == PROMPT
        assert turn["turn"]["outcome"] == "stop" and turn["turn"]["requests_sent"] == 3
        assert turn["final_text"] == "Done: P is at (1, 2)."
        assert [c["function_name"] for c in turn["calls"]] == ["search_tools", "create_point"]
        assert len(turn["batches"]) == 2 and turn["undo_before"] == 0 and turn["undo_after"] == 1
        assert turn["signals"]["dropped_calls"] == 1 and turn["signals"]["executed_calls"] == 1
        assert summary["exit_code"] == 0 and summary["models"]["qwen-local"]["runs"] == 2
        assert summary["models"]["qwen-local"]["requests_sent"] == 6
        text = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
        assert "## By model" in text and "## Passes per scenario" in text and "| GEO-90 | 2/2 |" in text

    def test_model_mistake_is_retraced_and_classified(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser(script={PROMPT: [[create(5, 5)]]})
        runner, sink = live_runner(catalogue, browser, tmp_path / "out")
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        summary = sink.close(catalogue)
        assert outcome.status == "fail" and outcome.retrace and outcome.retrace["reproduced"] is True
        failing = [r for r in outcome.results() if r["status"] == "fail"]
        assert failing and all(r["class"] == "model" for r in failing)
        # Model mistakes do not fail the run.
        assert summary["exit_code"] == 0 and summary["classes"]["model"] == 1

    def test_irreproducible_canvas_is_nondeterministic(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser(script={PROMPT: [[create(5, 5)]]}, drift_after_turn=0.25)
        runner, sink = live_runner(catalogue, browser, tmp_path / "out")
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        summary = sink.close(catalogue)
        assert outcome.retrace and outcome.retrace["first_difference"] == "t1"
        assert "nondeterministic" in outcome.classes() and summary["exit_code"] == 1

    def test_timeout_stops_the_turn(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser(hang=True)
        live = LiveOptions(turn_timeout_s=3, poll_interval_s=1, retrace_failures=False)
        runner, sink = live_runner(catalogue, browser, tmp_path / "out", live=live)
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        sink.close(catalogue)
        turn = next(step for step in outcome.steps if step["step"] == "t1")
        assert browser.stops == 1 and turn["turn"]["outcome"] == "timeout" and turn["turn"]["stop_reason"] == "timeout"
        assert turn["turn"]["requests_sent"] == 1  # the aborted request counts as sent
        assert {r["class"] for r in turn["results"] if r["status"] == "fail"} == {"infra"}

    def test_request_cap_per_turn(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser(loop=True)
        live = LiveOptions(turn_max_requests=3, poll_interval_s=1, retrace_failures=False)
        runner, sink = live_runner(catalogue, browser, tmp_path / "out", live=live)
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        sink.close(catalogue)
        turn = next(step for step in outcome.steps if step["step"] == "t1")
        # The client ends the turn at its limit: no request beyond it is sent.
        assert turn["turn"]["outcome"] == "max_requests" and turn["turn"]["requests_sent"] == 3
        assert turn["turn"]["stop_reason"] is None and browser.options[0]["max_requests"] == 3
        assert {r["class"] for r in turn["results"] if r["status"] == "fail"} == {"model"}

    def test_request_cap_backstop_stops_a_client_that_ignores_it(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser(loop=True)
        browser.honour_limit = False
        live = LiveOptions(turn_max_requests=3, poll_interval_s=1, retrace_failures=False)
        runner, sink = live_runner(catalogue, browser, tmp_path / "out", live=live)
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        turn = next(step for step in outcome.steps if step["step"] == "t1")
        assert turn["turn"]["stop_reason"] == "max_requests" and browser.stops == 1
        assert turn["turn"]["requests_sent"] == 4  # three completed and the one in flight

    def test_client_timeouts_follow_the_turn_timeout(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser()
        live = LiveOptions(turn_timeout_s=120, turn_max_requests=6, poll_interval_s=1)
        runner, sink = live_runner(catalogue, browser, tmp_path / "out", live=live)
        runner.run_live(geo90(catalogue), ["m"], 1)
        assert browser.options == [{"max_requests": 6, "response_timeout_ms": 150_000}]

    def test_client_timeout_counts_the_request_left_running(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser(script={PROMPT: [[create(1, 2)]]})
        browser.end_with = ("timeout", None)
        idle: list[int] = []
        budget = RequestBudget(50)
        runner, sink = live_runner(catalogue, browser, tmp_path / "out", budget=budget, idle_waits=idle)
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        turn = next(step for step in outcome.steps if step["step"] == "t1")
        assert turn["turn"]["outcome"] == "timeout" and turn["turn"]["requests_sent"] == 2 and idle
        assert budget.sent == 2

    def test_run_request_cap_stops_everything(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser(loop=True)
        live = LiveOptions(poll_interval_s=1, retrace_failures=False)
        runner, sink = live_runner(catalogue, browser, tmp_path / "out", live=live, budget=RequestBudget(3))
        sink.config["stopped"] = None
        outcomes = runner.run_live(geo90(catalogue), ["m"], repeats=3)
        assert len(outcomes) == 1 and runner.stopped and "request cap (3)" in runner.stopped
        turn = next(step for step in outcomes[0].steps if step["step"] == "t1")
        assert turn["turn"]["outcome"] == "request_cap" and browser.options[0]["max_requests"] == 3
        assert runner.budget.sent == 3  # the client stopped at what the cap left: never more than the cap
        sink.config["stopped"] = runner.stopped
        summary = sink.close(catalogue)
        assert summary["exit_code"] == 1 and "infra" in outcomes[0].classes()

    def test_unknown_turn_metrics_are_tolerated(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser()
        runner, sink = live_runner(catalogue, browser, tmp_path / "out")
        original = browser.execute_js

        def no_metrics(script: str, *args: Any, timeout: int = 30) -> Any:
            return None if "getMatHudLastTurnMetrics" in script else original(script, *args, timeout=timeout)

        browser.execute_js = no_metrics  # type: ignore[method-assign]
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        turn = next(step for step in outcome.steps if step["step"] == "t1")
        assert turn["turn"]["metrics"] is None and turn["turn"]["outcome"] == "unknown"

    def test_unreadable_traces_make_the_turn_infra(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser()
        runner, sink = live_runner(catalogue, browser, tmp_path / "out", live=LiveOptions(retrace_failures=False))
        original = browser.execute_js

        def broken_traces(script: str, *args: Any, timeout: int = 30) -> Any:
            if "JSON.stringify(window.getActionTraces" in script:
                return json.dumps({"error": "Object of type tuple is not JSON serializable"})
            return original(script, *args, timeout=timeout)

        browser.execute_js = broken_traces  # type: ignore[method-assign]
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        sink.close(catalogue)
        turn = next(step for step in outcome.steps if step["step"] == "t1")
        assert turn["turn"]["trace_error"].startswith("Object of type tuple") and turn["calls"] == []
        failing = [r for r in turn["results"] if r["status"] == "fail" and r["kind"] != "invariant"]
        assert all(r["class"] == "infra" for r in failing)


class TestRetraceRunner:
    def live_steps(self, catalogue: Catalogue, tmp_path: Path, browser: FakeChatBrowser) -> list[dict[str, Any]]:
        runner, _sink = live_runner(
            catalogue, browser, tmp_path / "live", live=LiveOptions(poll_interval_s=1, retrace_failures=False)
        )
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        return outcome.steps

    def test_retrace_reproduces_and_judges_batches(self, catalogue: Catalogue, tmp_path: Path) -> None:
        script = {PROMPT: [[SEARCH], [create(1, 2)], [create(3, 3, "Q")]]}
        steps = self.live_steps(catalogue, tmp_path, FakeChatBrowser(script=script))
        retrace_browser = FakeChatBrowser()
        sink = ResultSink(tmp_path / "retrace", {"mode": "retrace"})
        runner = RetraceRunner(catalogue, BrowserSession(lambda: retrace_browser, 5), sink, log=lambda _l: None)
        outcome = runner.retrace_outcome(geo90(catalogue)[0], steps, "local", "m", 1)
        turn = next(step for step in outcome.steps if step["step"] == "t1")
        assert turn["mode"] == "retrace" and turn["retraced_batches"] == 2 and len(turn["batches"]) == 2
        assert turn["batches"][0]["state"]["Points"][0]["name"] == "P"
        assert retrace_summary(steps, outcome)["reproduced"] is True
        assert not [r for r in outcome.results() if r.get("kind") == "invariant" and r["status"] != "pass"]

    def test_missing_live_step_is_an_infra_error(self, catalogue: Catalogue, tmp_path: Path) -> None:
        sink = ResultSink(tmp_path / "retrace", {"mode": "retrace"})
        runner = RetraceRunner(catalogue, BrowserSession(lambda: FakeChatBrowser(), 5), sink, log=lambda _l: None)
        outcome = runner.retrace_outcome(geo90(catalogue)[0], [], None, None, 1)
        assert outcome.infra_error and "no record of step t1" in outcome.infra_error
        assert retrace_summary([], outcome) == {"reproduced": None, "error": outcome.infra_error}


def test_regrade_keeps_live_fields(catalogue: Catalogue, tmp_path: Path) -> None:
    browser = FakeChatBrowser(script={PROMPT: [[create(5, 5)]]})
    runner, sink = live_runner(catalogue, browser, tmp_path / "out")
    runner.run_live(geo90(catalogue), ["m"], 1)
    sink.close(catalogue)
    summary, target = regrade(tmp_path / "out" / "results.json", catalogue)
    stored = json.loads(target.read_text(encoding="utf-8"))["scenarios"][0]
    assert stored["model"] == "m" and stored["retrace"]["reproduced"] is True
    assert summary["classes"]["model"] == 1 and summary["exit_code"] == 0
    # Answer checks run on the stored final text in a live regrade.
    assert "## By model" in (tmp_path / "out" / "summary_regraded.md").read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Command
# ----------------------------------------------------------------------


class FakeManager:
    port = 5079
    base_url = "http://127.0.0.1:5079"

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> tuple[bool, str]:
        self.stopped = True
        return True, "stopped"


def no_server(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("no server may start")


class TestLiveCommand:
    def invoke(self, *args: str) -> Any:
        return CliRunner().invoke(cli, ["test", "scenarios", *args])

    def test_local_dry_run_needs_no_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(command_module, "_start_own_server", no_server)
        result = self.invoke("--mode", "live", "--smoke", "--dry-run", "--json")
        assert result.exit_code == 0, result.output
        plan = json.loads(result.output)
        assert plan["provider"] == "local" and plan["scenarios"] == 11 and plan["max_requests"] is None

    def test_openrouter_needs_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(command_module, "_start_own_server", no_server)
        result = self.invoke("--mode", "live", "--provider", "openrouter", "--smoke")
        assert result.exit_code == 2 and "needs --models" in result.output

    def test_openrouter_dry_run_estimates_cost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(command_module, "_start_own_server", no_server)
        args = ["--mode", "live", "--provider", "openrouter", "--models", "deepseek/deepseek-v4.1-flash", "--smoke"]
        result = self.invoke(*args, "--dry-run", "--max-requests", "10")
        assert result.exit_code == 0, result.output
        assert "Estimated cost for deepseek/deepseek-v4.1-flash" in result.output
        assert "would abort" in result.output

    def test_openrouter_over_the_cap_sends_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(command_module, "_start_own_server", no_server)
        args = ["--mode", "live", "--provider", "openrouter", "--models", "x/y", "--smoke", "--max-requests", "10"]
        result = self.invoke(*args)
        assert result.exit_code == 2 and "nothing was sent" in result.output

    def test_openrouter_hybrid_search_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(command_module, "_start_own_server", no_server)
        args = ["--mode", "live", "--provider", "openrouter", "--models", "x/y", "--tool-search-mode", "hybrid"]
        result = self.invoke(*args)
        assert result.exit_code == 2 and "cannot count" in result.output

    def test_unregistered_model_aborts_before_the_browser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = FakeManager()
        started: list[dict[str, str]] = []

        def start(port: int, env: dict[str, str]) -> tuple[FakeManager, str]:
            started.append(env)
            return manager, ""

        monkeypatch.setattr(command_module, "_start_own_server", start)
        monkeypatch.setattr(command_module, "_available_models", lambda _url: AVAILABLE)
        monkeypatch.setattr("cli.browser.BrowserAutomation", no_server)
        result = self.invoke("--mode", "live", "--models", "stale-model", "--ids", "GEO-01")
        assert result.exit_code == 2 and "falls back to a paid provider" in result.output
        assert manager.stopped and started[0]["OPENAI_API_KEY"] == ""

    def test_retrace_needs_a_live_results_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(command_module, "_start_own_server", no_server)
        assert self.invoke("--mode", "retrace").exit_code == 2
        results = tmp_path / "results.json"
        results.write_text(json.dumps({"config": {"mode": "replay"}, "scenarios": []}))
        result = self.invoke("--mode", "retrace", str(results))
        assert result.exit_code == 2 and "not a live run" in result.output


# ----------------------------------------------------------------------
# Review follow-ups: budget on errors, infra handling, multi-batch retrace, regrade
# ----------------------------------------------------------------------


class TestBudgetOnErrors:
    def test_hook_error_mid_turn_still_counts_its_requests(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser(loop=True)
        browser.fail_after = 3
        budget = RequestBudget(100)
        runner, sink = live_runner(catalogue, browser, tmp_path / "out", budget=budget)
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        sink.close(catalogue)
        assert outcome.infra_error and "hook error" in outcome.infra_error
        assert budget.sent == 3 + 1  # the requests seen, plus the one in flight

    def test_no_turn_starts_when_its_first_request_would_reach_the_cap(
        self, catalogue: Catalogue, tmp_path: Path
    ) -> None:
        budget = RequestBudget(5)
        budget.add(5)
        browser = FakeChatBrowser()
        runner, sink = live_runner(catalogue, browser, tmp_path / "out", budget=budget)
        outcomes = runner.run_live(geo90(catalogue), ["m"], 1)
        assert browser.sent == [] and runner.stopped and "request cap (5)" in runner.stopped
        assert outcomes[0].infra_error

    def test_last_turn_gets_only_the_requests_the_cap_leaves(self, catalogue: Catalogue, tmp_path: Path) -> None:
        budget = RequestBudget(5)
        budget.add(4)
        browser = FakeChatBrowser(loop=True)
        runner, sink = live_runner(catalogue, browser, tmp_path / "out", budget=budget)
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        assert browser.options[0]["max_requests"] == 1 and budget.sent == 5
        assert runner.stopped and "request cap (5)" in runner.stopped


class TestInfraHandling:
    def test_timeout_with_passing_checks_is_infra_and_not_retraced(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser(script={PROMPT: [[create(1, 2)]]})
        browser.stall_before_answer = True
        idle: list[int] = []
        live = LiveOptions(turn_timeout_s=3, poll_interval_s=1)
        runner, sink = live_runner(catalogue, browser, tmp_path / "out", live=live, idle_waits=idle)
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        summary = sink.close(catalogue)
        turn = next(step for step in outcome.steps if step["step"] == "t1")
        assert turn["turn"]["outcome"] == "timeout" and turn["turn"]["drain_s"] > 0 and idle
        assert outcome.status == "pass" and outcome.retrace is None and outcome.classes() == {"infra"}
        assert summary["models"]["m"]["scenario_pass_rate"] is None  # infra runs are left out of the rates
        # Every run had an infrastructure failure: the run fails rather than passing on nothing.
        assert summary["exit_code"] == 1 and summary["infra_rate"] == 1.0

    def test_steps_after_an_infra_turn_are_infra(self) -> None:
        steps = [
            {"step": "t1", "turn": {"outcome": "timeout"}, "results": []},
            {"step": "t2", "turn": {"outcome": "stop"}, "results": [{"status": "fail", "kind": "check"}]},
            {"step": "chk1", "results": [{"status": "fail", "kind": "check"}, {"status": "fail", "kind": "invariant"}]},
        ]
        annotate_steps(steps, "live")
        assert steps[1]["results"][0]["class"] == "infra"
        assert [r["class"] for r in steps[2]["results"]] == ["infra", "app"]
        assert not needs_retrace(steps, None)

    def test_unreadable_traces_with_retrace_on_are_infra(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser()
        runner, sink = live_runner(catalogue, browser, tmp_path / "out")
        original = browser.execute_js

        def broken_traces(script: str, *args: Any, timeout: int = 30) -> Any:
            if "JSON.stringify(window.getActionTraces" in script:
                return json.dumps({"error": "Circular reference detected"})
            return original(script, *args, timeout=timeout)

        browser.execute_js = broken_traces  # type: ignore[method-assign]
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        summary = sink.close(catalogue)
        assert outcome.retrace is None and "infra" in outcome.classes()
        assert not outcome.classes() & {"app", "nondeterministic"}
        assert summary["exit_code"] == 1 and summary["infra_rate_exceeded"] == 0.5  # the only run is infra


class TestMultiBatchRetrace:
    def test_every_multi_batch_turn_is_retraced_and_judged_per_batch(
        self, catalogue: Catalogue, tmp_path: Path
    ) -> None:
        script = {PROMPT: [[create(1, 2)], [create(3, 3, "Q")], [{"tool": "undo", "args": {}}]]}
        runner, sink = live_runner(catalogue, FakeChatBrowser(script=script), tmp_path / "out")
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        sink.close(catalogue)
        assert outcome.retrace and outcome.retrace["reproduced"] is True
        turn = next(step for step in outcome.steps if step["step"] == "t1")
        judged = {r["name"]: r for r in turn["results"] if r.get("judged_by") == "retrace"}
        assert set(judged) == {"I4", "I5"} and all(r["status"] == "pass" for r in judged.values())
        assert outcome.status == "pass"

    def test_single_batch_passing_turn_is_not_retraced(self, catalogue: Catalogue, tmp_path: Path) -> None:
        runner, sink = live_runner(catalogue, FakeChatBrowser(script={PROMPT: [[create(1, 2)]]}), tmp_path / "out")
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        assert outcome.retrace is None

    def test_verdicts_apply_only_when_reproduced(self) -> None:
        live = [
            {
                "step": "t1",
                "batches": [{"calls": [{"function_name": "zoom"}]}, {"calls": [{"function_name": "undo"}]}],
                "results": [{"id": "t1.I4", "kind": "invariant", "name": "I4", "status": "fail"}],
            }
        ]
        retraced = [{"step": "t1", "results": [{"id": "t1.I4", "kind": "invariant", "name": "I4", "status": "pass"}]}]
        apply_batch_verdicts(live, {"reproduced": False, "steps": retraced})
        assert live[0]["results"][0]["status"] == "fail"
        apply_batch_verdicts(live, {"reproduced": True, "steps": retraced})
        assert live[0]["results"][0] == {
            "id": "t1.I4",
            "kind": "invariant",
            "name": "I4",
            "status": "pass",
            "judged_by": "retrace",
        }


class TestRegradeRetrace:
    def test_stored_retrace_steps_are_regraded(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser(script={PROMPT: [[create(5, 5)]]})
        runner, sink = live_runner(catalogue, browser, tmp_path / "out")
        runner.run_live(geo90(catalogue), ["m"], 1)
        sink.close(catalogue)
        results = tmp_path / "out" / "results.json"
        data = json.loads(results.read_text(encoding="utf-8"))
        retrace = data["scenarios"][0]["retrace"]
        assert retrace["steps"] and retrace["invariant_failures"] == []
        # A stale verdict stored in the file is replaced by the regraded one.
        retrace["invariant_failures"] = ["t1.I5"]
        results.write_text(json.dumps(data), encoding="utf-8")
        summary, target = regrade(results, catalogue)
        stored = json.loads(target.read_text(encoding="utf-8"))["scenarios"][0]
        assert stored["retrace"]["invariant_failures"] == [] and summary["classes"]["app"] == 0

    def test_retrace_without_steps_is_marked_stale(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser(script={PROMPT: [[create(5, 5)]]})
        runner, sink = live_runner(catalogue, browser, tmp_path / "out")
        runner.run_live(geo90(catalogue), ["m"], 1)
        sink.close(catalogue)
        results = tmp_path / "out" / "results.json"
        data = json.loads(results.read_text(encoding="utf-8"))
        data["scenarios"][0]["retrace"].pop("steps")
        data["scenarios"][0]["retrace"]["invariant_failures"] = ["t1.I5"]
        results.write_text(json.dumps(data), encoding="utf-8")
        summary, target = regrade(results, catalogue)
        stored = json.loads(target.read_text(encoding="utf-8"))["scenarios"][0]["retrace"]
        assert "invariant_failures" not in stored and stored["stale"]
        assert summary["classes"]["app"] == 0


def test_truncated_results_skip_the_naming_rule() -> None:
    long_result = "x" * 500 + "..."
    calls = mark_truncated([{"function_name": "create_point", "arguments": {"name": "P"}, "result": long_result}])
    assert calls[0]["result_truncated"] is True
    before, after = CanvasView(state()), CanvasView(state(point("A", 0, 0)))
    step = StepData(calls=calls, undo_before=0, undo_after=1)
    result = next(r for r in run_invariants(before, after, step, "t1") if r.name == "I4")
    assert result.passed
    calls[0].pop("result_truncated")
    assert not next(r for r in run_invariants(before, after, step, "t1") if r.name == "I4").passed


class TestCommandWiring:
    def test_openrouter_cap_and_no_retries_reach_the_runner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class CapturingRunner:
            def __init__(self, catalogue: Any, session: Any, sink: Any, options: Any, live: Any, **kwargs: Any) -> None:
                captured.update(options=options, live=live, **kwargs)
                self.budget = kwargs["budget"]
                self.stopped = None

            def run_live(self, *args: Any) -> list[Any]:
                return []

        manager = FakeManager()
        available = {"openrouter_paid": [{"id": "deepseek/deepseek-v4.1-flash"}]}
        monkeypatch.setattr(command_module, "_start_own_server", lambda port, env: (manager, ""))
        monkeypatch.setattr(command_module, "_available_models", lambda _url: available)
        monkeypatch.setattr(command_module, "LiveRunner", CapturingRunner)
        args = ["--mode", "live", "--provider", "openrouter", "--models", "deepseek/deepseek-v4.1-flash"]
        result = CliRunner().invoke(cli, ["test", "scenarios", *args, "--ids", "GEO-01", "--max-requests", "40"])
        assert result.exit_code == 0, result.output
        assert captured["budget"].cap == 40 and captured["options"].retries == 0
        assert captured["provider"] == "openrouter" and manager.stopped

    def test_local_run_has_no_cap_but_never_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class CapturingRunner:
            def __init__(self, catalogue: Any, session: Any, sink: Any, options: Any, live: Any, **kwargs: Any) -> None:
                captured.update(options=options, **kwargs)
                self.budget = kwargs["budget"]
                self.stopped = None

            def run_live(self, *args: Any) -> list[Any]:
                return []

        monkeypatch.setattr(command_module, "_start_own_server", lambda port, env: (FakeManager(), ""))
        monkeypatch.setattr(command_module, "_available_models", lambda _url: AVAILABLE)
        monkeypatch.setattr(command_module, "LiveRunner", CapturingRunner)
        result = CliRunner().invoke(cli, ["test", "scenarios", "--mode", "live", "--ids", "GEO-01"])
        assert result.exit_code == 0, result.output
        assert captured["budget"].cap is None and captured["options"].retries == 0

    def test_retrace_dry_run_starts_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(command_module, "_start_own_server", no_server)
        live = {"step": "t1", "turn": {"outcome": "stop"}, "batches": [{"calls": [{"function_name": "undo"}]}]}
        timed_out = dict(live, turn={"outcome": "timeout"})
        results = tmp_path / "results.json"
        scenarios = [
            {"id": "GEO-01", "model": "m", "repeat": 1, "steps": [live]},
            {"id": "GEO-02", "model": "m", "repeat": 1, "steps": [timed_out]},
        ]
        results.write_text(json.dumps({"config": {"mode": "live"}, "scenarios": scenarios}))
        result = CliRunner().invoke(
            cli, ["test", "scenarios", "--mode", "retrace", str(results), "--dry-run", "--json"]
        )
        assert result.exit_code == 0, result.output
        plan = json.loads(result.output)
        assert plan["runs"] == ["GEO-01 [m #1]"] and plan["left_out_infra"] == 1 and plan["batches"] == 1


class TestErrorSources:
    def test_client_error_is_an_app_failure(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser(script={PROMPT: [[create(1, 2)]]})
        browser.end_with = ("error", "client")
        runner, sink = live_runner(catalogue, browser, tmp_path / "out")
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        summary = sink.close(catalogue)
        turn = next(step for step in outcome.steps if step["step"] == "t1")
        assert turn["turn"]["error_source"] == "client"
        assert "app" in outcome.classes() and "infra" not in outcome.classes()
        assert summary["exit_code"] == 1

    def test_provider_error_is_infra(self, catalogue: Catalogue, tmp_path: Path) -> None:
        browser = FakeChatBrowser(script={PROMPT: [[create(1, 2)]]})
        browser.end_with = ("error", "provider")
        runner, sink = live_runner(catalogue, browser, tmp_path / "out")
        [outcome] = runner.run_live(geo90(catalogue), ["m"], 1)
        assert outcome.classes() == {"infra"}

    def test_route_error_steps_are_app(self) -> None:
        steps = [{"step": "t1", "turn": {"outcome": "error", "error_source": "server"},
                  "results": [{"status": "fail", "kind": "check"}]}]  # fmt: skip
        annotate_steps(steps, "live")
        assert steps[0]["results"][0]["class"] == "app"


class TestInfraRate:
    def outcome(self, catalogue: Catalogue, turn_outcome: str) -> ScenarioOutcome:
        outcome = ScenarioOutcome(geo90(catalogue)[0], model="m")
        outcome.steps = [{"step": "t1", "turn": {"outcome": turn_outcome}, "results": []}]
        return outcome

    def test_threshold(self, catalogue: Catalogue) -> None:
        from cli.scenarios.report import summarize

        half = [self.outcome(catalogue, "timeout"), self.outcome(catalogue, "stop")]
        assert summarize(half, mode="live")["exit_code"] == 0
        most = half + [self.outcome(catalogue, "error")]
        result = summarize(most, mode="live")
        assert result["exit_code"] == 1 and result["infra_rate"] == 0.6667
        assert summarize(most, mode="live", max_infra_rate=0.9)["exit_code"] == 0
        every = [self.outcome(catalogue, "timeout")]
        assert summarize(every, mode="live", max_infra_rate=1.0)["exit_code"] == 1  # all runs infra
        assert summarize(every, mode="replay")["exit_code"] == 0

    def test_command_records_the_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(command_module, "_start_own_server", no_server)
        result = CliRunner().invoke(cli, ["test", "scenarios", "--mode", "live", "--max-infra-rate", "2"])
        assert result.exit_code == 2  # outside 0..1


class TestUndoReproduction:
    def test_retrace_with_another_undo_delta_is_not_reproduced(self) -> None:
        live = [{"step": "t1", "state": state(point("P", 1, 2)), "undo_before": 0, "undo_after": 2}]
        same = [dict(live[0])]
        assert compare_runs(live, same)["reproduced"] is True
        other = [dict(live[0], undo_after=1)]
        result = compare_runs(live, other)
        assert result["reproduced"] is False and "undo depth" in result["differences"]["t1"][0]

    def test_failing_live_i5_kept_when_undo_depths_differ(self) -> None:
        live = [
            {
                "step": "t1",
                "state": state(),
                "undo_before": 0,
                "undo_after": 2,
                "batches": [{"calls": [{"function_name": "zoom"}]}, {"calls": [{"function_name": "zoom"}]}],
                "results": [{"id": "t1.I5", "kind": "invariant", "name": "I5", "status": "fail"}],
            }
        ]
        retraced = [
            {
                "step": "t1",
                "state": state(),
                "undo_before": 0,
                "undo_after": 1,
                "results": [{"id": "t1.I5", "kind": "invariant", "name": "I5", "status": "pass"}],
            }
        ]
        summary = compare_runs(live, retraced)
        summary["steps"] = retraced
        apply_batch_verdicts(live, summary)
        assert live[0]["results"][0]["status"] == "fail"


# ----------------------------------------------------------------------
# Abandoned turns and --regrade's infra limit
# ----------------------------------------------------------------------


def test_abandoned_turn_is_an_app_failure(catalogue: Catalogue) -> None:
    # The harness's tab is the only client of its server: nothing should abandon its turns.
    steps = [{"step": "t1", "turn": {"outcome": "abandoned"}, "results": [{"status": "fail", "kind": "check"}]}]
    annotate_steps(steps, "live")
    assert steps[0]["results"][0]["class"] == "app"
    passing = ScenarioOutcome(geo90(catalogue)[0], model="m")
    passing.steps = [{"step": "t1", "turn": {"outcome": "abandoned"}, "results": [{"status": "pass", "kind": "check"}]}]
    assert passing.classes() == {"app"} and passing.app_failure()


class TestRegradeInfraLimit:
    """--regrade honours an explicit --max-infra-rate and never reports a failing run in green."""

    def half_infra_results(self, catalogue: Catalogue, tmp_path: Path) -> Path:
        browser = FakeChatBrowser(script={PROMPT: [[create(5, 5)]]})
        runner, sink = live_runner(catalogue, browser, tmp_path / "out")
        runner.run_live(geo90(catalogue), ["m"], 2)
        sink.close(catalogue)
        results = tmp_path / "out" / "results.json"
        data = json.loads(results.read_text(encoding="utf-8"))
        turn = next(step for step in data["scenarios"][1]["steps"] if step["step"] == "t1")
        turn["turn"]["outcome"] = "timeout"  # one of the two runs timed out
        results.write_text(json.dumps(data), encoding="utf-8")
        return results

    def test_regrade_takes_the_given_limit(self, catalogue: Catalogue, tmp_path: Path) -> None:
        results = self.half_infra_results(catalogue, tmp_path)
        summary, _ = regrade(results, catalogue)
        assert summary["infra_rate"] == 0.5 and summary["exit_code"] == 0  # the stored limit is 0.5
        summary, target = regrade(results, catalogue, max_infra_rate=0.4)
        assert summary["infra_rate_exceeded"] == 0.4 and summary["exit_code"] == 1
        assert json.loads(target.read_text(encoding="utf-8"))["config"]["max_infra_rate"] == 0.4

    def test_command_passes_an_explicit_limit_and_reports_the_failure(
        self, catalogue: Catalogue, tmp_path: Path
    ) -> None:
        results = self.half_infra_results(catalogue, tmp_path)
        base = ["test", "scenarios", "--regrade", str(results), "--scenarios-dir", str(tmp_path / "scenarios")]
        passing = CliRunner().invoke(cli, base)
        assert passing.exit_code == 0, passing.output
        assert "No unexpected failures." in passing.output
        failing = CliRunner().invoke(cli, [*base, "--max-infra-rate", "0.4"])
        assert failing.exit_code == 1, failing.output
        assert "Too many infrastructure failures" in failing.output
        assert "No unexpected failures." not in failing.output
        assert "the run fails" in failing.output
