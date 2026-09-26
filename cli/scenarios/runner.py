"""Replay runner: run each scenario's reference tool calls in the real app, with no model.

The runner drives headless Chrome through ``cli.browser.BrowserAutomation`` and
the app's scenario hooks (``static/client/scenario_hooks.py``). Per scenario it
resets the session, runs the setup calls, then every step's reference or
scripted calls through ``runMatHudToolCalls`` (the path a model's batch takes),
reads the canvas with ``getMatHudCanvasState`` and grades the step.

Every browser call runs under a per-step timeout. A step that times out kills
the browser; the runner starts a new one and retries the scenario once. A hook
error reloads the page before the next scenario.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from cli.scenarios.geometry import SampleRequests
from cli.scenarios.grade import ScenarioGrader, StepRecordData
from cli.scenarios.model import Catalogue, Scenario, Step, ToolCall
from cli.scenarios.report import ResultSink, ScenarioOutcome

T = TypeVar("T")

DEFAULT_STEP_TIMEOUT_S = 60
# A check stops at its first missing function sample, so samples may take a few rounds.
MAX_SAMPLE_ROUNDS = 5


class StepTimeout(Exception):
    """A browser call did not return within the step timeout."""


class HookError(Exception):
    """A scenario hook reported an error."""


@dataclass
class ReplayOptions:
    step_timeout_s: float = DEFAULT_STEP_TIMEOUT_S
    known_artifacts: bool = False
    retries: int = 1


class BrowserSession:
    """One headless Chrome on the app, restartable after a hang."""

    def __init__(self, factory: Callable[[], Any], timeout_s: float) -> None:
        self._factory = factory
        self.timeout_s = timeout_s
        self.browser: Any = None
        self.restarts = 0

    def open(self) -> None:
        browser = self._factory()
        self.browser = browser
        self.call(browser.setup, timeout=120)
        if not self.call(browser.navigate_to_app, timeout=90):
            raise HookError("could not open the app")
        if not self.call(browser.wait_for_app_ready, timeout=120):
            raise HookError("the app did not become ready")

    def reload(self) -> None:
        browser = self.browser
        if not self.call(browser.navigate_to_app, timeout=90) or not self.call(browser.wait_for_app_ready, timeout=120):
            raise HookError("the app did not become ready after a reload")

    def restart(self) -> None:
        self.kill()
        self.restarts += 1
        self.open()

    def kill(self) -> None:
        """Stop the browser, killing chromedriver and Chrome if they hang."""
        browser = self.browser
        self.browser = None
        if browser is None:
            return
        driver = getattr(browser, "driver", None)
        pid = getattr(getattr(getattr(driver, "service", None), "process", None), "pid", None)
        if pid is not None:
            _kill_process_tree(int(pid))
        finished = threading.Event()

        def cleanup() -> None:
            try:
                browser.cleanup()
            finally:
                finished.set()

        threading.Thread(target=cleanup, daemon=True).start()
        finished.wait(15)

    def call(self, fn: Callable[..., T], *args: Any, timeout: Optional[float] = None) -> T:
        """Run ``fn(*args)`` in a worker thread; raise StepTimeout if it does not return in time."""
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["value"] = fn(*args)
            except BaseException as exc:  # re-raised in the caller's thread
                box["error"] = exc

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        worker.join(timeout if timeout is not None else self.timeout_s)
        if worker.is_alive():
            raise StepTimeout(
                f"{getattr(fn, '__name__', 'browser call')} did not return in {timeout or self.timeout_s:.0f} s"
            )
        if "error" in box:
            raise box["error"]
        return box["value"]  # type: ignore[no-any-return]

    def hook(self, name: str, *args: Any) -> dict[str, Any]:
        browser = self.browser
        script_timeout = int(self.timeout_s) + 5
        reply: dict[str, Any] = self.call(lambda: browser.call_hook(name, *args, timeout=script_timeout))
        if reply.get("status") == "error":
            raise HookError(f"{name}: {reply.get('error')}")
        return reply

    def screenshot(self, path: Path) -> bool:
        try:
            return bool(self.call(self.browser.capture_screenshot, str(path), timeout=30))
        except Exception:
            return False


def _kill_process_tree(pid: int) -> None:
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is in every requirements file
        return
    try:
        parent = psutil.Process(pid)
        processes = parent.children(recursive=True) + [parent]
    except psutil.Error:
        return
    for process in processes:
        try:
            process.kill()
        except psutil.Error:
            continue


def _artifact_name(scenario_id: str, step_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{scenario_id}__{step_id}")


class ReplayRunner:
    """Runs scenarios in replay mode and records every step."""

    def __init__(
        self,
        catalogue: Catalogue,
        session: BrowserSession,
        sink: ResultSink,
        options: Optional[ReplayOptions] = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.catalogue = catalogue
        self.session = session
        self.sink = sink
        self.options = options or ReplayOptions()
        self.log = log

    def run(self, scenarios: list[Scenario], skipped: Optional[dict[str, str]] = None) -> list[ScenarioOutcome]:
        skipped = skipped or {}
        outcomes = []
        for number, scenario in enumerate(scenarios, start=1):
            if scenario.id in skipped:
                outcome = ScenarioOutcome(scenario, skipped_reason=skipped[scenario.id])
            else:
                outcome = self.run_scenario(scenario)
            self.sink.add(outcome)
            outcomes.append(outcome)
            extra = f" ({outcome.infra_error})" if outcome.infra_error else ""
            self.log(f"[{number}/{len(scenarios)}] {scenario.id} {outcome.status} {outcome.duration_s:.1f}s{extra}")
        return outcomes

    def run_scenario(self, scenario: Scenario) -> ScenarioOutcome:
        attempts = 0
        while True:
            attempts += 1
            started = time.time()
            outcome = ScenarioOutcome(scenario, attempts=attempts)
            try:
                if self.session.browser is None:
                    self.session.open()
                self._run_steps(scenario, outcome)
            except StepTimeout as exc:
                outcome.infra_error = f"timeout: {exc}"
                self._recover(restart=True)
                if attempts <= self.options.retries:
                    self.log(f"  {scenario.id}: {exc}; restarting the browser and retrying")
                    continue
            except HookError as exc:
                outcome.infra_error = f"hook error: {exc}"
                self._recover(restart=False)
            except Exception as exc:  # the browser died or returned garbage
                outcome.infra_error = f"{type(exc).__name__}: {exc}"
                self._recover(restart=True)
            outcome.duration_s = time.time() - started
            return outcome

    def _recover(self, restart: bool) -> None:
        try:
            if restart or self.session.browser is None:
                self.session.restart()
            else:
                self.session.reload()
        except Exception as exc:
            self.log(f"  browser recovery failed: {exc}")
            self.session.kill()

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _run_steps(self, scenario: Scenario, outcome: ScenarioOutcome) -> None:
        grader = ScenarioGrader(scenario, self.catalogue.waivers_for(scenario), mode="replay")
        reset_options: dict[str, Any] = {"chat": True}
        if scenario.fixture_state is not None:
            reset_options["fixture"] = scenario.fixture_state
        self.session.hook("resetMatHudSession", json.dumps(reset_options))
        start = self._snapshot(None)
        grader.start(start)
        self._record(outcome, grader, "start", "start", None, start, 0.0)

        t0 = time.time()
        setup = self._run_calls(scenario.setup_calls) if scenario.setup_calls else None
        data = self._with_samples(grader, None, setup)
        self._record(outcome, grader, "setup", "setup", None, data, time.time() - t0)

        for step in scenario.steps:
            t0 = time.time()
            batch = self._run_calls(step.calls) if step.runs_calls else None
            data = self._with_samples(grader, step, batch)
            self._record(outcome, grader, step.id, step.kind, step, data, time.time() - t0)

    def _run_calls(self, calls: list[ToolCall]) -> dict[str, Any]:
        return self.session.hook("runMatHudToolCalls", json.dumps([call.payload() for call in calls]))

    def _snapshot(self, batch: Optional[dict[str, Any]], options: Optional[dict[str, Any]] = None) -> StepRecordData:
        request = {"inspect": True}
        request.update(options or {})
        snapshot = self.session.hook("getMatHudCanvasState", json.dumps(request))
        data = StepRecordData(state=snapshot.get("state") or {}, inspection=snapshot.get("inspection"))
        if batch is not None:
            data.calls = list(batch.get("traced") or [])
            data.undoable = list(batch.get("undoable") or [])
            data.undo_before = batch.get("undo_depth_before")
            data.undo_after = batch.get("undo_depth_after")
            data.redo_before = batch.get("redo_depth_before")
            data.redo_after = batch.get("redo_depth_after")
        return data

    def _with_samples(
        self, grader: ScenarioGrader, step: Optional[Step], batch: Optional[dict[str, Any]]
    ) -> StepRecordData:
        """Read the canvas after a step, sampling functions at every x its checks need."""
        data = self._snapshot(batch)
        wanted = SampleRequests()
        for _ in range(MAX_SAMPLE_ROUNDS):
            requests = grader.needed_samples(step, data)
            if not requests:
                break
            wanted.merge(requests)
            data = self._snapshot(batch, wanted.as_options())
        return data

    def _record(
        self,
        outcome: ScenarioOutcome,
        grader: ScenarioGrader,
        step_id: str,
        kind: str,
        step: Optional[Step],
        data: StepRecordData,
        duration: float,
    ) -> None:
        if step_id == "start":
            results = []
        else:
            results = grader.grade(step_id, step, data)
        record: dict[str, Any] = {
            "step": step_id,
            "kind": kind,
            "mode": "replay",
            "user": step.user if step is not None else None,
            "calls": data.calls,
            "undoable": data.undoable,
            "undo_before": data.undo_before,
            "undo_after": data.undo_after,
            "redo_before": data.redo_before,
            "redo_after": data.redo_after,
            "final_text": data.final_text,
            "state": data.state,
            "inspection": data.inspection,
            "results": [result.to_dict() for result in results],
            "duration_s": round(duration, 3),
        }
        statuses = {result.status for result in results}
        if statuses & {"fail", "error"} or (self.options.known_artifacts and "xfail" in statuses):
            record["artifacts"] = self._save_artifacts(outcome.scenario.id, step_id, data)
        outcome.steps.append(record)
        self.sink.record_step(outcome.scenario.id, record)

    def _save_artifacts(self, scenario_id: str, step_id: str, data: StepRecordData) -> dict[str, str]:
        failures = self.sink.failures_dir
        failures.mkdir(parents=True, exist_ok=True)
        base = failures / _artifact_name(scenario_id, step_id)
        artifacts: dict[str, str] = {}
        state_path = base.with_suffix(".json")
        state_path.write_text(
            json.dumps({"state": data.state, "inspection": data.inspection}, indent=1), encoding="utf-8"
        )
        artifacts["state"] = str(state_path)
        png = base.with_suffix(".png")
        if self.session.screenshot(png):
            artifacts["screenshot"] = str(png)
        return artifacts
