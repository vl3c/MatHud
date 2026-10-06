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
from cli.scenarios.grade import ScenarioGrader, StepRecordData, fitted_view_fields
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


class RunStopped(Exception):
    """The whole run must stop after this step (live mode: the request cap was reached)."""


@dataclass
class ReplayOptions:
    step_timeout_s: float = DEFAULT_STEP_TIMEOUT_S
    known_artifacts: bool = False
    retries: int = 1
    # Seconds to pause after each step, so someone watching an attached window sees every canvas.
    pace_s: float = 0.0
    # Attach mode: zoom the view to the content after each graded step (fitMatHudView), for display.
    fit_view: bool = False


# Tools that set the view, and check fields that read it: a scenario using either is never fitted.
VIEW_TOOLS = frozenset({"zoom", "set_coordinate_system", "set_grid_visible"})
_VIEW_CHECK_PATHS = ("polar_radial_spacing", "grid_visible", "coordinate_mode", "left_bound", "right_bound",
                     "top_bound", "bottom_bound", "reference_scale_factor")  # fmt: skip


def view_sensitive(scenario: Scenario) -> bool:
    """True when ``scenario`` sets or checks the view, so a display fit could change its outcome.

    Such scenarios (the ``view`` tag, a view tool in any call, or a check that
    reads the view) run unfitted in attach mode and grade exactly as headless.
    """
    if "view" in scenario.tags:
        return True
    step_calls = [call for step in scenario.steps for call in step.calls]
    if any(call.tool in VIEW_TOOLS for call in scenario.setup_calls + step_calls):
        return True
    # Tools whose result depends on the view when they run after a fit (setup runs before any).
    if any(_uses_the_view(call) for call in step_calls):
        return True
    return any(_reads_view(check) for step in scenario.steps for check in step.checks)


def _uses_the_view(call: ToolCall) -> bool:
    """``find_function_features`` without bounds searches the visible x range; ``generate_graph``
    without a ``placement_box`` places vertices with no coordinates inside the visible view."""
    args = call.args or {}
    if call.tool == "find_function_features":
        return args.get("left_bound") is None or args.get("right_bound") is None
    if call.tool == "generate_graph" and not args.get("placement_box"):
        vertices = args.get("vertices") or []
        return any(not isinstance(v, dict) or v.get("x") is None or v.get("y") is None for v in vertices)
    return False


def _reads_view(check: Any) -> bool:
    if isinstance(check, dict):
        if check.get("target") == "view" or check.get("view") is True:
            return True
        if any(field in str(check.get("path", "")) for field in _VIEW_CHECK_PATHS):
            return True
        return any(_reads_view(value) for value in check.values())
    if isinstance(check, list):
        return any(_reads_view(item) for item in check)
    return False


class BrowserSession:
    """One browser on the app (``cli.browser_backend.AppBrowser``), restartable after a hang.

    The default backend is a headless Chrome of the run's own; in attach mode it
    is the desktop window over CDP, which is never navigated on open and only
    disconnected on close.
    """

    def __init__(self, factory: Callable[[], Any], timeout_s: float) -> None:
        self._factory = factory
        self.timeout_s = timeout_s
        self.browser: Any = None
        self.restarts = 0

    def open(self) -> None:
        browser = self._factory()
        self.browser = browser
        self.call(browser.setup, timeout=120)
        # An attached window already shows the app; it is reloaded only to recover from a hang.
        if not getattr(browser, "attached", False) or self.restarts:
            if not self.call(browser.reload, timeout=90):
                raise HookError("could not open the app")
        if not self.call(browser.wait_for_app_ready, timeout=120):
            raise HookError("the app did not become ready")

    def reload(self) -> None:
        browser = self.browser
        if not self.call(browser.reload, timeout=90) or not self.call(browser.wait_for_app_ready, timeout=120):
            raise HookError("the app did not become ready after a reload")

    def restart(self) -> None:
        self.kill()
        self.restarts += 1
        self.open()

    def kill(self) -> None:
        """Stop the browser, killing chromedriver and Chrome if they hang (an attached window only disconnects)."""
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
                browser.close()
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
        if reply.get("status") in ("error", "busy"):
            raise HookError(f"{name}: {reply.get('status')}: {reply.get('error')}")
        return reply

    def screenshot(self, path: Path) -> bool:
        try:
            return bool(self.call(self.browser.screenshot, str(path), timeout=30))
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
    """Runs scenarios in replay mode and records every step.

    Subclasses (``cli.scenarios.live``) change how a step's calls are produced by
    overriding ``_reset`` and ``_execute_step``; grading and recording are shared.
    """

    mode = "replay"

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
        # Why the run stopped early (RunStopped), or None.
        self.stopped: Optional[str] = None

    def run(self, scenarios: list[Scenario], skipped: Optional[dict[str, str]] = None) -> list[ScenarioOutcome]:
        skipped = skipped or {}
        outcomes = []
        for number, scenario in enumerate(scenarios, start=1):
            if self.stopped:
                break
            if scenario.id in skipped:
                outcome = ScenarioOutcome(scenario, skipped_reason=skipped[scenario.id])
            else:
                outcome = self.run_scenario(scenario)
            self.sink.add(outcome)
            outcomes.append(outcome)
            extra = f" ({outcome.infra_error})" if outcome.infra_error else ""
            self.log(f"[{number}/{len(scenarios)}] {outcome.label} {outcome.status} {outcome.duration_s:.1f}s{extra}")
        return outcomes

    def run_scenario(self, scenario: Scenario) -> ScenarioOutcome:
        attempts = 0
        while True:
            attempts += 1
            started = time.time()
            outcome = self._new_outcome(scenario, attempts)
            try:
                if self.session.browser is None:
                    self.session.open()
                self._run_steps(scenario, outcome)
            except RunStopped as exc:
                # The step was recorded; the run ends here (e.g. the request cap was reached).
                outcome.infra_error = f"run stopped: {exc}"
                self.stopped = str(exc)
            except StepTimeout as exc:
                outcome.infra_error = f"timeout: {exc}"
                self._recover(restart=True)
                if attempts <= self.options.retries:
                    self.log(f"  {scenario.id}: {exc}; restarting the browser and retrying")
                    self.sink.discard_attempt(scenario.id, attempts, str(exc))
                    continue
            except HookError as exc:
                outcome.infra_error = f"hook error: {exc}"
                self._recover(restart=False)
            except Exception as exc:  # the browser died or returned garbage
                outcome.infra_error = f"{type(exc).__name__}: {exc}"
                self._recover(restart=True)
            outcome.duration_s = time.time() - started
            return outcome

    def _new_outcome(self, scenario: Scenario, attempts: int) -> ScenarioOutcome:
        return ScenarioOutcome(scenario, attempts=attempts)

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
        grader = ScenarioGrader(scenario, self.catalogue.waivers_for(scenario), mode=self.mode)
        self._reset(scenario)
        start = self._snapshot(None)
        grader.start(start)
        self._record(outcome, grader, "start", "start", None, start, 0.0)

        t0 = time.time()
        setup = self._run_calls(scenario.setup_calls) if scenario.setup_calls else None
        data = self._with_samples(grader, None, setup)
        self._record(outcome, grader, "setup", "setup", None, data, time.time() - t0)
        self._present(scenario, grader, outcome, paced=setup is not None or scenario.fixture_state is not None)

        for step in scenario.steps:
            t0 = time.time()
            data, extra = self._execute_step(scenario, step, grader)
            self._record(outcome, grader, step.id, step.kind, step, data, time.time() - t0, extra)
            self._present(scenario, grader, outcome)

    def _present(
        self, scenario: Scenario, grader: ScenarioGrader, outcome: ScenarioOutcome, paced: bool = True
    ) -> None:
        """After a step is graded and recorded: fit the view for display (attach mode), then pause."""
        if self.options.fit_view and not view_sensitive(scenario):
            try:
                reply = self.session.hook("fitMatHudView")
            except HookError as exc:
                self.log(f"  could not fit the view: {exc}")
            else:
                if reply.get("fitted"):
                    # Later steps start from the fitted view, so it is what they are compared with;
                    # the step's record keeps it so --regrade rebases the same way.
                    fitted = self._snapshot(None)
                    grader.rebase_view(fitted)
                    if outcome.steps:
                        outcome.steps[-1]["fitted_view"] = fitted_view_fields(fitted)
        if paced and self.options.pace_s > 0:
            time.sleep(self.options.pace_s)

    def _reset(self, scenario: Scenario) -> None:
        """Reset the session for ``scenario``, restoring its fixture."""
        reset_options: dict[str, Any] = {"chat": True}
        if scenario.fixture_state is not None:
            reset_options["fixture"] = scenario.fixture_state
        self.session.hook("resetMatHudSession", json.dumps(reset_options))

    def _execute_step(
        self, scenario: Scenario, step: Step, grader: ScenarioGrader
    ) -> tuple[StepRecordData, dict[str, Any]]:
        """Run one step; returns its data and any extra fields for its record."""
        batch = self._run_calls(step.calls) if step.runs_calls else None
        return self._with_samples(grader, step, batch), {}

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
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        if step_id == "start":
            results = []
        else:
            results = grader.grade(step_id, step, data)
        record: dict[str, Any] = {
            "step": step_id,
            "kind": kind,
            "mode": self.mode,
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
            "attempt": outcome.attempts,
        }
        if data.batches is not None:
            record["batches"] = data.batches
        record.update(extra or {})
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
        # Paths relative to the output directory, so the reports still resolve after a move.
        artifacts["state"] = state_path.relative_to(self.sink.out_dir).as_posix()
        png = base.with_suffix(".png")
        if self.session.screenshot(png):
            artifacts["screenshot"] = png.relative_to(self.sink.out_dir).as_posix()
        return artifacts
