"""Tests for attach mode (`test scenarios --attach-desktop N`), with fake windows (no browser, no model).

The CDP backend is replaced by the runner tests' fake browsers marked as
attached; the desktop app's HTTP endpoints are patched. Nothing reaches a
window, a server or a provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest
from click.testing import CliRunner

from cli.main import cli
from cli.scenarios import attach as attach_module
from cli.scenarios import command as command_module
from cli.scenarios import runner as runner_module
from cli.scenarios.attach import (
    ATTACHED_PACE_S,
    UNPINNED_SETTINGS,
    confirm,
    confirmation_text,
    default_pace,
    pin_conflicts,
)
from cli.scenarios.report import ResultSink
from cli.scenarios.runner import BrowserSession, ReplayOptions, ReplayRunner

from server_tests.test_cli.scenario_states import VIEW
from server_tests.test_cli.test_scenario_live import PROMPT, FakeChatBrowser
from server_tests.test_cli.test_scenario_runner import FakeBrowser, write_catalogue

AVAILABLE = {"local_agent": [{"id": "qwen-local"}], "openai": [{"id": "gpt-5"}]}


class AttachedWindow(FakeChatBrowser):
    """A fake chat browser that behaves like the CDP backend: attached, with the app's URL."""

    attached = True
    base_url = "http://127.0.0.1:5110"

    def __init__(self, busy: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.busy = busy
        self.closes = 0
        self.view = dict(VIEW)
        self.fits = 0
        self.guards: list[bool] = []

    def _state(self) -> dict[str, Any]:
        state = super()._state()
        state["Cartesian_System_Visibility"] = dict(self.view)
        return state

    def execute_js(self, script: str, *args: Any, timeout: int = 30) -> Any:
        if "ai-model-selector" in script:
            return True  # every model is in the window's dropdown
        return super().execute_js(script, *args, timeout=timeout)

    def call_hook(self, name: str, *args: Any, timeout: int = 30) -> dict[str, Any]:
        if name == "getMatHudTurnStatus" and self.busy:
            return {"processing": True, "completed_turns": 0, "requests": 0}
        if name == "fitMatHudView":
            return self._fit()
        if name == "setMatHudAutomationGuards":
            blocked = bool(json.loads(args[0])["block_workspace_tools"])
            self.guards.append(blocked)
            return {"status": "ok", "workspace_tools_blocked": blocked}
        return super().call_hook(name, *args, timeout=timeout)

    def _fit(self) -> dict[str, Any]:
        """Like the app: a display-only zoom to the points (no undo entry); nothing to fit leaves the view."""
        self.fits += 1
        if not self.points:
            return {"status": "ok", "fitted": False}
        xs = [p["args"]["position"]["x"] for p in self.points]
        ys = [p["args"]["position"]["y"] for p in self.points]
        self.view = {
            "left_bound": min(xs) - 1,
            "right_bound": max(xs) + 1,
            "top_bound": max(ys) + 1,
            "bottom_bound": min(ys) - 1,
        }
        return {"status": "ok", "fitted": True, "view": dict(self.view)}

    def close(self) -> None:
        self.closes += 1


@pytest.fixture
def scenarios_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "scenarios"
    directory.mkdir()
    write_catalogue(directory)
    return directory


@pytest.fixture
def window(monkeypatch: pytest.MonkeyPatch) -> AttachedWindow:
    fake = AttachedWindow()
    monkeypatch.setattr(attach_module, "_connect", lambda port: fake)
    monkeypatch.setattr(attach_module, "attached_browser", lambda port: fake)
    monkeypatch.setattr(command_module, "_available_models", lambda base_url: AVAILABLE)
    monkeypatch.setattr(command_module, "_conversation_resetter", lambda base_url: lambda: None)
    monkeypatch.setattr(command_module, "_idle_waiter", lambda base_url: lambda: 0.0)
    monkeypatch.setattr(command_module, "_start_own_server", _no_server)
    monkeypatch.setattr(command_module.ServerManager, "is_server_running", _no_server)
    return fake


def _no_server(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("attach mode must not start or probe a server of its own")


@pytest.fixture
def pauses(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    class FakeTime:
        def __getattr__(self, name: str) -> Any:
            import time

            return getattr(time, name)

        @staticmethod
        def sleep(seconds: float) -> None:
            recorded.append(seconds)

    monkeypatch.setattr(runner_module, "time", FakeTime())
    return recorded


def invoke(scenarios_dir: Path, out: Path, *args: str, stdin: Optional[str] = None) -> Any:
    command = ["test", "scenarios", "--scenarios-dir", str(scenarios_dir), "--out", str(out), *args]
    return CliRunner().invoke(cli, command, input=stdin)


class TestHelpers:
    def test_default_pace(self) -> None:
        assert default_pace(None, attached=True) == ATTACHED_PACE_S
        assert default_pace(None, attached=False) == 0.0
        assert default_pace(0.4, attached=True) == 0.4
        assert default_pace(0.4, attached=False) == 0.4

    def test_pin_conflicts(self) -> None:
        assert pin_conflicts([]) is None
        message = pin_conflicts(["tool_exposure", "canvas_budget"])
        assert message is not None
        assert "--tool-exposure, --canvas-budget cannot be applied with --attach-desktop" in message

    def test_confirm(self) -> None:
        def ask(text: str) -> bool:
            raise AssertionError("must not ask")

        assert confirm("x", yes=True, ask=ask, interactive=False)
        assert not confirm("x", yes=False, ask=ask, interactive=False)
        assert confirm("x", yes=False, ask=lambda text: True, interactive=True)
        assert not confirm("x", yes=False, ask=lambda text: False, interactive=True)

    def test_confirmation_text_names_the_resets_and_openrouter_caveats(self) -> None:
        replay = confirmation_text("http://127.0.0.1:5110", "replay", 3, None)
        assert "resets its canvas, undo history and chat." in replay
        assert "The server conversation is reset once first." in replay
        live = confirmation_text("http://127.0.0.1:5110", "live", 3, "openrouter")
        assert "server conversation" in live and "TOOL_SEARCH_MODE must be local" in live


class TestSessionAttached:
    def test_open_does_not_reload_an_attached_window_but_a_restart_does(self) -> None:
        browser = FakeBrowser()
        browser.attached = True
        session = BrowserSession(lambda: browser, 5)
        session.open()
        assert browser.reloads == 0
        session.restart()
        assert browser.reloads == 1
        session.kill()
        assert browser.cleaned  # close() only; an attached window just disconnects

    def test_a_browser_of_its_own_is_loaded_on_open(self) -> None:
        browser = FakeBrowser()
        BrowserSession(lambda: browser, 5).open()
        assert browser.reloads == 1

    def test_pace_pauses_after_each_step(self, tmp_path: Path, scenarios_dir: Path, pauses: list[float]) -> None:
        from cli.scenarios.model import load_catalogue

        catalogue = load_catalogue(scenarios_dir)
        sink = ResultSink(tmp_path / "out", {"mode": "replay"})
        runner = ReplayRunner(
            catalogue, BrowserSession(FakeBrowser, 5), sink, ReplayOptions(pace_s=0.7), log=lambda line: None
        )
        runner.run(catalogue.select(ids=["GEO-90"]))
        assert pauses == [0.7, 0.7]  # GEO-90 has no setup calls and two steps


class TestAttachedReplay:
    def test_runs_in_the_window_and_records_the_attachment(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, pauses: list[float]
    ) -> None:
        out = tmp_path / "out"
        result = invoke(scenarios_dir, out, "--attach-desktop", "9301", "--yes")

        assert result.exit_code == 0, result.output
        assert "Attached to the desktop window at http://127.0.0.1:5110" in result.output
        data = json.loads((out / "results.json").read_text())
        config = data["config"]
        assert config["attached_desktop"] is True and config["automation_port"] == 9301
        assert config["port"] == 5110 and config["server_started"] is False and config["pins_applied"] is False
        assert config["pace_s"] == ATTACHED_PACE_S
        statuses = {item["id"]: item.get("skipped_reason") or item["status"] for item in data["scenarios"]}
        assert statuses["GEO-90"] == "pass"
        assert "uses workspaces" in statuses["GEO-91"]
        assert data["summary"]["attached_desktop"]["url"] == "http://127.0.0.1:5110"
        assert "Attached to the desktop app at http://127.0.0.1:5110" in (out / "summary.md").read_text()
        assert pauses and set(pauses) == {ATTACHED_PACE_S}
        assert window.closes >= 1  # the session disconnected; nothing closed the window

    def test_workspace_writes_are_refused_when_attached(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow
    ) -> None:
        out = tmp_path / "out"
        result = invoke(scenarios_dir, out, "--attach-desktop", "9301", "--yes", "--allow-workspace-writes")
        assert result.exit_code == 2 and "cannot be used with --attach-desktop" in result.output
        assert not (out / "results.json").exists()

    def test_workspace_tools_are_blocked_during_the_run_and_unblocked_after(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, pauses: list[float]
    ) -> None:
        from cli.scenarios.attach import AttachedBrowser

        # The session's browser blocks them whenever the app is ready (AttachedBrowser); the run unblocks them.
        result = invoke(scenarios_dir, tmp_path / "out", "--attach-desktop", "9301", "--yes")
        assert result.exit_code == 0, result.output
        assert window.guards[-1] is False
        assert issubclass(AttachedBrowser, attach_module.CDPBrowser)

    def test_replay_resets_the_server_conversation_once(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, monkeypatch: pytest.MonkeyPatch,
        pauses: list[float],
    ) -> None:  # fmt: skip
        resets: list[str] = []
        monkeypatch.setattr(command_module, "_conversation_resetter", lambda url: lambda: resets.append(url))
        result = invoke(scenarios_dir, tmp_path / "out", "--attach-desktop", "9301", "--yes")
        assert result.exit_code == 0, result.output
        assert resets == ["http://127.0.0.1:5110"]
        assert "(1 skipped)" in result.output  # GEO-91 saves a workspace

    def test_ctrl_c_stops_the_turn_in_the_window(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def interrupted(self: Any, *args: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(attach_module.LiveRunner, "run_live", interrupted)
        result = invoke(scenarios_dir, tmp_path / "out", "--mode", "live", "--attach-desktop", "9301", "--yes")
        assert result.exit_code == 130
        assert window.stops == 1
        assert window.guards[-1] is False

    def test_live_attached_run_never_retries(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        real = attach_module.LiveRunner

        def capturing(*args: Any, **kwargs: Any) -> Any:
            captured["options"] = args[3]
            return real(*args, **kwargs)

        monkeypatch.setattr(attach_module, "LiveRunner", capturing)
        args = ["--mode", "live", "--attach-desktop", "9301", "--yes", "--ids", "GEO-90", "--pace", "0"]
        result = invoke(scenarios_dir, tmp_path / "out", *args)
        assert result.exit_code == 0, result.output
        assert captured["options"].retries == 0

    def test_openrouter_is_refused_unless_the_app_searches_locally_without_retries(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cli.desktop_automation as desktop_automation

        available = {"openrouter_paid": [{"id": "x/y"}]}
        monkeypatch.setattr(command_module, "_available_models", lambda base_url: available)
        reported = {"tool_search_mode": "hybrid", "openrouter_max_retries": 0}
        monkeypatch.setattr(desktop_automation, "automation_settings", lambda base_url: reported)
        args = ["--mode", "live", "--attach-desktop", "9301", "--yes", "--provider", "openrouter", "--models", "x/y"]
        result = invoke(scenarios_dir, tmp_path / "out", *args, "--ids", "GEO-90")
        assert result.exit_code == 2 and "TOOL_SEARCH_MODE is 'hybrid'" in result.output
        assert window.sent == []

    def test_attached_browser_blocks_workspace_tools_whenever_the_app_is_ready(self) -> None:
        from cli.scenarios.attach import AttachedBrowser
        from server_tests.test_cli.test_cdp import PAGE, FakePage, FakeSocket, _value

        class GuardPage(FakePage):
            def __init__(self) -> None:
                super().__init__()
                self.guards: list[str] = []

            def __call__(self, message: dict[str, Any]) -> dict[str, Any]:
                expression = (message.get("params") or {}).get("expression", "")
                if "setMatHudAutomationGuards" in expression:
                    self.guards.append(expression)
                    reply = json.dumps({"status": "ok", "workspace_tools_blocked": True})
                    return {"id": message["id"], "result": _value(reply)}
                if "resetMatHudSession" in expression or "getMatHudCanvasState" in expression:
                    return {"id": message["id"], "result": _value(json.dumps({"status": "ok"}))}
                return super().__call__(message)

        page = GuardPage()
        browser = AttachedBrowser(9301, connect=lambda url, timeout: FakeSocket(page), targets=lambda p, h: [PAGE])
        browser.setup()
        assert browser.wait_for_app_ready(2) is True
        assert len(page.guards) == 1 and "block_workspace_tools" in page.guards[0]
        browser.reload()  # a reload lifts the block in the page, so readiness applies it again
        assert browser.wait_for_app_ready(2) is True
        assert len(page.guards) == 2
        # A reload from outside is not seen, so hooks that can run tools re-apply it first.
        browser.call_hook("resetMatHudSession", "{}")
        assert len(page.guards) == 3
        browser.call_hook("getMatHudCanvasState", "{}")  # reads only: no extra round trip
        assert len(page.guards) == 3

    def test_a_window_that_cannot_block_its_workspace_tools_stops_the_run(self) -> None:
        from cli.scenarios.attach import AttachedBrowser
        from cli.scenarios.runner import RunStopped
        from server_tests.test_cli.test_cdp import PAGE, FakePage, FakeSocket, _value

        class OldPage(FakePage):
            def __call__(self, message: dict[str, Any]) -> dict[str, Any]:
                expression = (message.get("params") or {}).get("expression", "")
                if "setMatHudAutomationGuards" in expression:
                    reply = json.dumps({"status": "ok", "workspace_tools_blocked": False})
                    return {"id": message["id"], "result": _value(reply)}
                return super().__call__(message)

        page = OldPage()
        browser = AttachedBrowser(9301, connect=lambda url, timeout: FakeSocket(page), targets=lambda p, h: [PAGE])
        browser.setup()
        with pytest.raises(RunStopped, match="did not block its workspace tools"):
            browser.wait_for_app_ready(2)

    def test_a_closed_window_stops_the_run(self) -> None:
        from cli.cdp import CDPUnavailable
        from cli.scenarios.attach import AttachedBrowser
        from cli.scenarios.runner import RunStopped

        def gone(port: int, host: str) -> Any:
            raise CDPUnavailable("nothing listens")

        with pytest.raises(RunStopped, match="window is gone"):
            AttachedBrowser(9301, targets=gone).setup()

    def test_asks_before_resetting_the_window(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: list[str] = []

        def ask(text: str) -> bool:
            asked.append(text)
            return False

        monkeypatch.setattr(attach_module, "_ask", ask)
        monkeypatch.setattr(attach_module, "_stdin_is_terminal", lambda: True)
        out = tmp_path / "out"
        result = invoke(scenarios_dir, out, "--attach-desktop", "9301")
        assert result.exit_code == 2
        assert "Not confirmed" in result.output
        assert asked == ["Continue?"]
        assert "resets its canvas, undo history and chat" in result.output
        assert not (out / "results.json").exists()

    def test_the_warning_is_shown_with_yes_too(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, pauses: list[float]
    ) -> None:
        result = invoke(scenarios_dir, tmp_path / "out", "--attach-desktop", "9301", "--yes")
        assert "resets its canvas, undo history and chat" in result.output

    def test_refuses_without_a_terminal_or_yes(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow
    ) -> None:
        result = invoke(scenarios_dir, tmp_path / "out", "--attach-desktop", "9301")
        assert result.exit_code == 2 and "Pass --yes" in result.output

    def test_refuses_while_a_turn_runs(self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow) -> None:
        window.busy = True
        result = invoke(scenarios_dir, tmp_path / "out", "--attach-desktop", "9301", "--yes")
        assert result.exit_code == 2 and "A turn is running" in result.output

    def test_reports_a_missing_window(
        self, tmp_path: Path, scenarios_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing(port: int) -> Any:
            raise RuntimeError("no DevTools endpoint at http://127.0.0.1:9301/json")

        monkeypatch.setattr(attach_module, "_connect", missing)
        result = invoke(scenarios_dir, tmp_path / "out", "--attach-desktop", "9301", "--yes")
        assert result.exit_code == 2 and "Could not attach" in result.output

    @pytest.mark.parametrize(
        "args, message",
        [
            (["--start-server"], "drop --start-server"),
            (["--mode", "retrace", "results.json"], "not retrace"),
            (["--mode", "live", "--tool-exposure", "full"], "--tool-exposure cannot be applied"),
            (["--mode", "live", "--local-reasoning-effort", "high"], "--local-reasoning-effort cannot be applied"),
        ],
    )
    def test_refused_combinations(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, args: list[str], message: str
    ) -> None:
        if "results.json" in args:
            (tmp_path / "results.json").write_text("{}")
            args = [str(tmp_path / "results.json") if a == "results.json" else a for a in args]
        result = invoke(scenarios_dir, tmp_path / "out", "--attach-desktop", "9301", "--yes", *args)
        assert result.exit_code == 2, result.output
        assert message in result.output


class TestAttachedLive:
    def test_runs_the_prompt_in_the_window_with_an_explicit_model(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, pauses: list[float]
    ) -> None:
        out = tmp_path / "out"
        result = invoke(scenarios_dir, out, "--mode", "live", "--attach-desktop", "9301", "--yes", "--ids", "GEO-90")

        assert result.exit_code == 0, result.output
        assert window.sent == [(PROMPT, "qwen-local")]
        config = json.loads((out / "results.json").read_text())["config"]
        assert config["attached_desktop"] is True and config["models"] == ["qwen-local"]
        assert config["unpinned_settings"] == list(UNPINNED_SETTINGS)
        assert "server_env" not in config  # nothing was pinned

    def test_unregistered_model_aborts_before_any_message(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow
    ) -> None:
        result = invoke(scenarios_dir, tmp_path / "out", "--mode", "live", "--attach-desktop", "9301", "--yes",
                        "--models", "gpt-5")  # fmt: skip
        assert result.exit_code == 2 and "Aborting before the first message" in result.output
        assert window.sent == []

    def test_openrouter_keeps_its_guards(self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow) -> None:
        base = ["--mode", "live", "--attach-desktop", "9301", "--yes", "--provider", "openrouter"]
        no_models = invoke(scenarios_dir, tmp_path / "a", *base)
        assert no_models.exit_code == 2 and "needs --models" in no_models.output
        over_cap = invoke(scenarios_dir, tmp_path / "b", *base, "--models", "x/y", "--max-requests", "1")
        assert over_cap.exit_code == 2 and "nothing was sent" in over_cap.output
        dry = invoke(scenarios_dir, tmp_path / "c", *base, "--models", "x/y", "--dry-run", "--json")
        assert dry.exit_code == 0, dry.output
        assert json.loads(dry.output)["attached_desktop"] == 9301
        assert window.sent == []


def write_fit_catalogue(directory: Path) -> None:
    """GEO-80: a point, then a batch that changes nothing (I5 compares it with the canvas before it); CV-80: a zoom."""
    scenario = {
        "id": "GEO-80",
        "title": "A no-op after a point",
        "tags": ["points"],
        "steps": [
            {
                "user": "Point P at (1, 2).",
                "reference": [{"tool": "create_point", "args": {"x": 1, "y": 2, "name": "P"}}],
                "checks": [{"check": "count", "select": {"type": "Point"}, "eq": 1}],
            },
            {
                "do": [{"tool": "evaluate_expression", "args": {"expression": "1+1"}}],
                "checks": [{"check": "count", "select": {"type": "Point"}, "eq": 1}],
            },
        ],
    }
    zoom = {"tool": "zoom", "args": {"center_x": 0, "center_y": 0, "range_val": 2, "range_axis": "x"}}
    view = {"id": "CV-80", "title": "Zoom", "tags": ["points"], "steps": [{"user": "Zoom in.", "reference": [zoom]}]}
    (directory / "geometry.json").write_text(json.dumps({"schema": 1, "area": "GEO", "scenarios": [scenario]}))
    (directory / "canvas.json").write_text(json.dumps({"schema": 1, "area": "CV", "scenarios": [view]}))
    (directory / "known_bugs.json").write_text(json.dumps({"bugs": {}, "invariant_waivers": {}}))


class TestFitView:
    @pytest.fixture
    def fit_dir(self, tmp_path: Path) -> Path:
        directory = tmp_path / "fit_scenarios"
        directory.mkdir()
        write_fit_catalogue(directory)
        return directory

    def test_view_sensitive_scenarios_in_the_catalogue(self) -> None:
        from cli.scenarios.model import load_catalogue
        from cli.scenarios.runner import view_sensitive

        by_id = {s.id: s for s in load_catalogue().scenarios}
        for scenario_id in ("CV-01", "CV-03", "CV-05", "CV-06", "GEO-05"):
            assert view_sensitive(by_id[scenario_id]), scenario_id
        for scenario_id in ("GEO-01", "CV-02", "TR-01"):
            assert not view_sensitive(by_id[scenario_id]), scenario_id

    def test_rebase_view_replaces_the_last_graded_canvas_only(self) -> None:
        from cli.scenarios.grade import ScenarioGrader, StepRecordData
        from cli.scenarios.model import load_catalogue

        scenario = next(s for s in load_catalogue().scenarios if s.id == "CV-02")
        grader = ScenarioGrader(scenario, {})
        grader.start(StepRecordData(state={"Cartesian_System_Visibility": dict(VIEW)}))
        point = {"name": "A", "args": {"position": {"x": 0.5, "y": 0.5}}}
        samples = {"drawables": [{"class": "Point", "name": "A", "samples": [[1, 2]]}], "polar_radial_spacing": 50}
        graded = {"Points": [point], "Cartesian_System_Visibility": dict(VIEW), "current_tick_spacing": 100}
        grader.grade("setup", None, StepRecordData(state=graded, inspection=samples))
        start, setup = grader.snapshots["start"], grader.snapshots["setup"]
        fitted = {"left_bound": -1, "right_bound": 1, "top_bound": 1, "bottom_bound": -1}
        after_fit = {"Points": [point], "Cartesian_System_Visibility": fitted, "current_tick_spacing": 0.2}

        grader.rebase_view(StepRecordData(state=after_fit, inspection={"drawables": [], "polar_radial_spacing": 0.2}))

        rebased = grader.snapshots["setup"]
        assert grader.snapshots["start"] is start
        assert rebased is not setup and grader.previous is rebased
        assert rebased.view_bounds["left_bound"] == -1
        assert rebased.state["current_tick_spacing"] == 0.2
        assert rebased.state["Points"] == [point]
        # The graded canvas's inspection (function samples) is kept; only the view's fields change.
        assert rebased.inspection["drawables"] == samples["drawables"]
        assert rebased.inspection["polar_radial_spacing"] == 0.2

    def test_attached_replay_fits_after_each_step_and_grades_against_the_fitted_view(
        self, tmp_path: Path, fit_dir: Path, window: AttachedWindow, pauses: list[float]
    ) -> None:
        out = tmp_path / "out"
        result = invoke(fit_dir, out, "--attach-desktop", "9301", "--yes", "--ids", "GEO-80")

        assert result.exit_code == 0, result.output
        assert window.fits == 3  # after setup, t1 and do1
        data = json.loads((out / "results.json").read_text())
        assert data["scenarios"][0]["status"] == "pass"
        assert data["config"]["fit_view"] is True and data["summary"]["attached_desktop"]["fit_view"] is True
        t1 = next(step for step in data["scenarios"][0]["steps"] if step["step"] == "t1")
        # The step was graded and recorded before the fit.
        assert t1["state"]["Cartesian_System_Visibility"] == VIEW
        assert "View fitted to the drawings" in (out / "summary.md").read_text()

    def test_without_the_rebase_the_fit_would_fail_the_next_step(
        self,
        tmp_path: Path,
        fit_dir: Path,
        window: AttachedWindow,
        pauses: list[float],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cli.scenarios.grade import ScenarioGrader

        monkeypatch.setattr(ScenarioGrader, "rebase_view", lambda self, data: None)
        out = tmp_path / "out"
        result = invoke(fit_dir, out, "--attach-desktop", "9301", "--yes", "--ids", "GEO-80")
        assert result.exit_code == 1
        assert json.loads((out / "results.json").read_text())["scenarios"][0]["status"] == "fail"

    def test_view_sensitive_scenarios_and_no_fit_view_are_not_fitted(
        self, tmp_path: Path, fit_dir: Path, window: AttachedWindow, pauses: list[float]
    ) -> None:
        invoke(fit_dir, tmp_path / "a", "--attach-desktop", "9301", "--yes", "--ids", "CV-80")
        assert window.fits == 0  # (the fake does not zoom, so its grading is beside the point here)
        off = invoke(fit_dir, tmp_path / "b", "--attach-desktop", "9301", "--yes", "--ids", "GEO-80", "--no-fit-view")
        assert off.exit_code == 0, off.output
        assert window.fits == 0
        assert json.loads((tmp_path / "b" / "results.json").read_text())["config"]["fit_view"] is False

    def test_live_attached_run_fits_after_the_turn(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, pauses: list[float]
    ) -> None:
        args = ["--mode", "live", "--attach-desktop", "9301", "--yes", "--ids", "GEO-90", "--no-retrace"]
        result = invoke(scenarios_dir, tmp_path / "out", *args)
        assert result.exit_code == 0, result.output
        assert window.sent == [(PROMPT, "qwen-local")]
        assert window.fits >= 2

    def test_fit_view_needs_attach_mode(self, tmp_path: Path, fit_dir: Path) -> None:
        result = invoke(fit_dir, tmp_path / "out", "--fit-view", "--dry-run")
        assert result.exit_code == 2 and "only with --attach-desktop" in result.output
