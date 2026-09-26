"""Tests for the replay runner, the reports and --regrade, with a fake browser (no Chrome)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import pytest
from click.testing import CliRunner

from cli.main import cli
from cli.scenarios.model import Catalogue, load_catalogue
from cli.scenarios.report import ResultSink, regrade
from cli.scenarios.runner import BrowserSession, ReplayOptions, ReplayRunner, StepTimeout

from server_tests.test_cli.scenario_states import VIEW


class FakeBrowser:
    """A tiny canvas behind the scenario hooks: create_point, undo and a point-moving bug."""

    def __init__(self, hang_on: Optional[str] = None, moved_by_bug: float = 0.0) -> None:
        self.points: list[dict[str, Any]] = []
        self.undo: list[list[dict[str, Any]]] = []
        self.hang_on = hang_on
        self.moved_by_bug = moved_by_bug
        self.cleaned = False
        self.screenshots: list[str] = []
        self.driver = None

    def setup(self) -> None:
        pass

    def navigate_to_app(self) -> bool:
        return True

    def wait_for_app_ready(self) -> bool:
        return True

    def cleanup(self) -> None:
        self.cleaned = True

    def capture_screenshot(self, path: str) -> bool:
        Path(path).write_bytes(b"png")
        self.screenshots.append(path)
        return True

    def _state(self) -> dict[str, Any]:
        return {
            "Points": [dict(p) for p in self.points],
            "Cartesian_System_Visibility": dict(VIEW),
            "coordinate_system": {"mode": "cartesian"},
        }

    def call_hook(self, name: str, *args: Any, timeout: int = 30) -> dict[str, Any]:
        if name == "resetMatHudSession":
            self.points, self.undo = [], []
            return {"status": "ok"}
        if name == "getMatHudCanvasState":
            return {"state": self._state(), "inspection": {"drawables": [], "undo_depth": len(self.undo)}}
        if name == "runMatHudToolCalls":
            calls = json.loads(args[0])
            if self.hang_on and any(c["function_name"] == self.hang_on for c in calls):
                time.sleep(5)
            before = len(self.undo)
            traced = []
            for call in calls:
                traced.append(self._run(call["function_name"], call["arguments"]))
            return {
                "status": "ok",
                "traced": traced,
                "undoable": [c["function_name"] == "create_point" for c in calls],
                "state": self._state(),
                "undo_depth_before": before,
                "undo_depth_after": len(self.undo),
                "redo_depth_before": 0,
                "redo_depth_after": 0,
            }
        raise AssertionError(f"unexpected hook {name}")

    def _run(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool == "create_point":
            self.undo.append([dict(p) for p in self.points])
            name = args.get("name") or "ABCDEFG"[len(self.points)]
            self.points.append(
                {"name": name, "args": {"position": {"x": args["x"] + self.moved_by_bug, "y": args["y"]}}}
            )
        elif tool == "undo" and self.undo:
            self.points = self.undo.pop()
        return {"function_name": tool, "arguments": args, "result": "Call successful!", "is_error": False}


def write_catalogue(directory: Path, known_check: bool = False) -> Catalogue:
    at_check: dict[str, Any] = {"check": "point_at", "select": {"type": "Point", "name": "P"}, "at": [1, 2]}
    scenario: dict[str, Any] = {
        "id": "GEO-90",
        "title": "Point and undo",
        "tags": ["points"],
        "smoke": True,
        "steps": [
            {
                "user": "Point P at (1, 2).",
                "reference": [{"tool": "create_point", "args": {"x": 1, "y": 2, "name": "P"}}],
                "checks": [at_check],
            },
            {"do": [{"tool": "undo"}], "checks": [{"check": "state_equals", "snapshot": "setup"}]},
        ],
    }
    if known_check:
        at_check["known"] = "K9"
        scenario["known"] = ["K9"]
    second = {
        "id": "GEO-91",
        "title": "Workspace",
        "tags": ["workspaces"],
        "steps": [
            {
                "user": "Save.",
                "reference": [{"tool": "save_workspace", "args": {"name": "w"}}],
                "checks": [{"check": "no_tool_errors"}],
            }
        ],
    }
    (directory / "geometry.json").write_text(json.dumps({"schema": 1, "area": "GEO", "scenarios": [scenario, second]}))
    (directory / "known_bugs.json").write_text(json.dumps({"bugs": {"K9": "points move"}, "invariant_waivers": {}}))
    return load_catalogue(directory)


class TestReplayRunner:
    def _run(self, tmp_path: Path, browser: FakeBrowser, known_check: bool = False) -> tuple[Any, Path, Catalogue]:
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir(exist_ok=True)
        catalogue = write_catalogue(scenarios_dir, known_check)
        out = tmp_path / "out"
        sink = ResultSink(out, {"mode": "replay"})
        session = BrowserSession(lambda: browser, 30)
        runner = ReplayRunner(catalogue, session, sink, ReplayOptions(), log=lambda line: None)
        outcomes = runner.run(catalogue.select(ids=["GEO-90"]))
        sink.close(catalogue)
        return outcomes, out, catalogue

    def test_passing_run_writes_reports(self, tmp_path: Path) -> None:
        outcomes, out, _ = self._run(tmp_path, FakeBrowser())
        assert outcomes[0].status == "pass"
        assert [step["step"] for step in outcomes[0].steps] == ["start", "setup", "t1", "do1"]
        lines = (out / "results.jsonl").read_text().splitlines()
        assert len(lines) == 4 and json.loads(lines[2])["scenario"] == "GEO-90"
        results = json.loads((out / "results.json").read_text())
        assert results["summary"]["exit_code"] == 0
        assert results["scenarios"][0]["counts"]["pass"] > 0
        summary = (out / "summary.md").read_text()
        assert "GEO-90 Point and undo (smoke)" in summary and "no unexpected failures" in summary

    def test_unexpected_failure_saves_artifacts(self, tmp_path: Path) -> None:
        browser = FakeBrowser(moved_by_bug=1.0)
        outcomes, out, _ = self._run(tmp_path, browser)
        assert outcomes[0].status == "fail"
        step = next(s for s in outcomes[0].steps if s["step"] == "t1")
        assert set(step["artifacts"]) == {"state", "screenshot"}
        assert (out / "failures" / "GEO-90__t1.json").exists()
        summary = (out / "summary.md").read_text()
        assert "## Unexpected failures" in summary and "failures/GEO-90__t1.png" in summary
        assert json.loads((out / "results.json").read_text())["summary"]["exit_code"] == 1

    def test_known_failure_is_xfail_without_artifacts(self, tmp_path: Path) -> None:
        outcomes, out, _ = self._run(tmp_path, FakeBrowser(moved_by_bug=1.0), known_check=True)
        assert outcomes[0].status == "xfail"
        assert not (out / "failures").exists()
        results = json.loads((out / "results.json").read_text())
        assert results["summary"]["exit_code"] == 0
        assert "| K9 | 1 | points move |" in (out / "summary.md").read_text()

    def test_known_check_that_passes_is_xpass(self, tmp_path: Path) -> None:
        outcomes, out, _ = self._run(tmp_path, FakeBrowser(), known_check=True)
        assert outcomes[0].status == "xpass"
        results = json.loads((out / "results.json").read_text())
        assert results["summary"]["xpass"] == {"GEO-90": ["K9"]}
        assert results["summary"]["exit_code"] == 0
        assert "fixed? K9" in (out / "summary.md").read_text()

    def test_timeout_restarts_the_browser_and_retries(self, tmp_path: Path) -> None:
        browsers = [FakeBrowser(hang_on="create_point"), FakeBrowser()]
        created: list[FakeBrowser] = []

        def factory() -> FakeBrowser:
            browser = browsers[len(created)]
            created.append(browser)
            return browser

        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        catalogue = write_catalogue(scenarios_dir)
        sink = ResultSink(tmp_path / "out", {})
        session = BrowserSession(factory, 1.0)
        runner = ReplayRunner(catalogue, session, sink, ReplayOptions(step_timeout_s=1.0), log=lambda line: None)
        outcome = runner.run_scenario(catalogue.select(ids=["GEO-90"])[0])
        sink.close(catalogue)
        assert outcome.status == "pass" and outcome.attempts == 2
        assert len(created) == 2 and browsers[0].cleaned

    def test_session_call_times_out(self) -> None:
        session = BrowserSession(FakeBrowser, 0.2)
        with pytest.raises(StepTimeout):
            session.call(time.sleep, 2)

    def test_regrade_reproduces_and_follows_scenario_edits(self, tmp_path: Path) -> None:
        _, out, catalogue = self._run(tmp_path, FakeBrowser(moved_by_bug=1.0))
        summary, target = regrade(out / "results.json", catalogue)
        assert summary["exit_code"] == 1 and target.name == "results_regraded.json"
        assert (out / "summary_regraded.md").exists()
        edited = write_catalogue(tmp_path / "scenarios", known_check=True)
        summary, _ = regrade(out / "results.json", edited)
        assert summary["exit_code"] == 0 and summary["scenario_counts"]["xfail"] == 1


class TestScenariosCommand:
    def test_live_mode_is_refused(self) -> None:
        result = CliRunner().invoke(cli, ["test", "scenarios", "--mode", "live"])
        assert result.exit_code == 2
        assert "not implemented yet" in result.output

    def test_dry_run_lists_the_catalogue(self) -> None:
        result = CliRunner().invoke(cli, ["test", "scenarios", "--dry-run", "--smoke"])
        assert result.exit_code == 0, result.output
        assert "GEO-01 (smoke)" in result.output
        assert "[skipped: uses workspaces" in result.output

    def test_dry_run_json(self) -> None:
        result = CliRunner().invoke(cli, ["test", "scenarios", "--dry-run", "--ids", "GEO-04,CV", "--json"])
        plan = json.loads(result.output)
        assert plan["ids"][0] == "GEO-04" and all(i == "GEO-04" or i.startswith("CV-") for i in plan["ids"])

    def test_no_match(self) -> None:
        result = CliRunner().invoke(cli, ["test", "scenarios", "--dry-run", "--tags", "no-such-tag"])
        assert result.exit_code == 2

    def test_server_must_be_running_without_start_server(self) -> None:
        result = CliRunner().invoke(cli, ["test", "scenarios", "--ids", "GEO-01", "--port", "1"])
        assert result.exit_code == 2
        assert "not running" in result.output

    def test_regrade_cli(self, tmp_path: Path) -> None:
        results = tmp_path / "results.json"
        results.write_text(json.dumps({"config": {"mode": "replay"}, "scenarios": []}))
        result = CliRunner().invoke(cli, ["test", "scenarios", "--regrade", str(results), "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["scenarios"] == 0
