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

    def __init__(self, hang_on: Optional[str] = None, moved_by_bug: float = 0.0, double_archive: bool = False) -> None:
        self.double_archive = double_archive
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
            if self.double_archive:  # like K1: one extra undo entry per create
                self.undo.append([dict(p) for p in self.points])
            name = args.get("name") or "ABCDEFG"[len(self.points)]
            self.points.append(
                {"name": name, "args": {"position": {"x": args["x"] + self.moved_by_bug, "y": args["y"]}}}
            )
        elif tool == "undo" and self.undo:
            self.points = self.undo.pop()
        return {"function_name": tool, "arguments": args, "result": "Call successful!", "is_error": False}


def write_catalogue(
    directory: Path,
    known_check: bool = False,
    global_waivers: Optional[dict[str, str]] = None,
    scenario_waivers: Optional[dict[str, str]] = None,
) -> Catalogue:
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
    if scenario_waivers:
        scenario["invariants"] = scenario_waivers
        scenario["known"] = sorted(set(scenario.get("known", [])) | set(scenario_waivers.values()))
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
    bugs = {"K1": "undo entries", "K2": "results", "K9": "points move"}
    (directory / "known_bugs.json").write_text(json.dumps({"bugs": bugs, "invariant_waivers": global_waivers or {}}))
    return load_catalogue(directory)


class TestReplayRunner:
    def _run(
        self, tmp_path: Path, browser: FakeBrowser, known_check: bool = False, **waivers: Any
    ) -> tuple[Any, Path, Catalogue]:
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir(exist_ok=True)
        catalogue = write_catalogue(scenarios_dir, known_check, **waivers)
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

    def test_retry_marks_the_abandoned_attempt_in_the_jsonl(self, tmp_path: Path) -> None:
        browsers = [FakeBrowser(hang_on="create_point"), FakeBrowser()]
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        catalogue = write_catalogue(scenarios_dir)
        sink = ResultSink(tmp_path / "out", {})
        runner = ReplayRunner(
            catalogue,
            BrowserSession(lambda: browsers.pop(0), 1.0),
            sink,
            ReplayOptions(step_timeout_s=1.0),
            log=lambda line: None,
        )
        runner.run(catalogue.select(ids=["GEO-90"]))
        sink.close(catalogue)
        records = [json.loads(line) for line in (tmp_path / "out" / "results.jsonl").read_text().splitlines()]
        kinds = [(r["kind"], r.get("attempt")) for r in records]
        assert ("attempt_discarded", 1) in kinds
        after_marker = records[kinds.index(("attempt_discarded", 1)) + 1 :]
        assert after_marker and all(r["attempt"] == 2 for r in after_marker)
        stored = json.loads((tmp_path / "out" / "results.json").read_text())["scenarios"][0]
        assert stored["attempts"] == 2 and all(step["attempt"] == 2 for step in stored["steps"])

    def test_jsonl_starts_fresh_for_each_run(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "results.jsonl").write_text('{"scenario": "OLD-01"}\n')
        self._run(tmp_path, FakeBrowser())
        assert "OLD-01" not in (out / "results.jsonl").read_text()

    def test_unused_global_and_scenario_waivers_are_reported(self, tmp_path: Path) -> None:
        outcomes, out, _ = self._run(
            tmp_path, FakeBrowser(), global_waivers={"I5": "K1"}, scenario_waivers={"I4": "K2"}
        )
        assert outcomes[0].status == "pass"
        summary = json.loads((out / "results.json").read_text())["summary"]
        assert summary["unused_waivers"] == ["I5:K1 (global)", "I4:K2 (GEO-90)"]
        assert summary["exit_code"] == 0
        assert "waiver I5:K1 (global) excused nothing in this run" in (out / "summary.md").read_text()

    def test_used_waiver_makes_the_scenario_waived(self, tmp_path: Path) -> None:
        outcomes, out, _ = self._run(tmp_path, FakeBrowser(double_archive=True), global_waivers={"I5": "K1"})
        assert outcomes[0].status == "waived"
        summary = json.loads((out / "results.json").read_text())["summary"]
        assert summary["unused_waivers"] == [] and summary["scenario_counts"]["waived"] == 1

    def test_known_check_beats_waiver_for_scenario_status(self, tmp_path: Path) -> None:
        outcomes, _, _ = self._run(
            tmp_path, FakeBrowser(double_archive=True, moved_by_bug=1.0), known_check=True, global_waivers={"I5": "K1"}
        )
        assert outcomes[0].status == "xfail"

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


class TestRegradeFromElsewhere:
    def test_regrade_with_a_relative_path_from_another_directory(self, tmp_path: Path, monkeypatch: Any) -> None:
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        catalogue = write_catalogue(scenarios_dir)
        out = tmp_path / "runs" / "one"
        sink = ResultSink(out, {"mode": "replay"})
        runner = ReplayRunner(
            catalogue, BrowserSession(lambda: FakeBrowser(moved_by_bug=1.0), 30), sink, log=lambda line: None
        )
        runner.run(catalogue.select(ids=["GEO-90"]))
        sink.close(catalogue)
        step = json.loads((out / "results.json").read_text())["scenarios"][0]["steps"][2]
        assert step["artifacts"]["state"] == "failures/GEO-90__t1.json"

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        summary, target = regrade(Path("..") / "runs" / "one" / "results.json", catalogue)
        assert summary["exit_code"] == 1
        assert "(failures/GEO-90__t1.png)" in (target.parent / "summary_regraded.md").read_text()

    def test_old_absolute_artifact_paths_still_render(self, tmp_path: Path) -> None:
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        catalogue = write_catalogue(scenarios_dir)
        results = {
            "config": {"mode": "replay"},
            "scenarios": [
                {
                    "id": "GEO-90",
                    "steps": [
                        {"step": "start", "state": {}},
                        {"step": "setup", "state": {}},
                        {"step": "t1", "state": {}, "artifacts": {"state": "Z:/nowhere/GEO-90__t1.json"}},
                    ],
                }
            ],
        }
        path = tmp_path / "results.json"
        path.write_text(json.dumps(results))
        regrade(path, catalogue)
        assert "Z:/nowhere/GEO-90__t1.json" in (tmp_path / "summary_regraded.md").read_text()

    def test_regrade_honours_filters(self, tmp_path: Path) -> None:
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        catalogue = write_catalogue(scenarios_dir)
        results = {"config": {}, "scenarios": [{"id": "GEO-90", "steps": []}, {"id": "GEO-91", "steps": []}]}
        path = tmp_path / "results.json"
        path.write_text(json.dumps(results))
        summary, _ = regrade(path, catalogue, {"GEO-91"})
        assert summary["scenarios"] == 1

    def test_regrade_reports_unrecorded_samples(self, tmp_path: Path) -> None:
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir()
        scenario = {
            "id": "FN-90",
            "title": "f",
            "tags": ["functions"],
            "steps": [
                {
                    "user": "Plot x^2.",
                    "reference": [{"tool": "draw_function", "args": {"function_string": "x^2", "name": "f"}}],
                    "checks": [
                        {
                            "check": "relation",
                            "relation": "function_value",
                            "select": [{"type": "Function", "name": "f"}],
                            "x": 2,
                            "y": 4,
                        }
                    ],
                }
            ],
        }
        (scenarios_dir / "functions.json").write_text(json.dumps({"schema": 1, "area": "FN", "scenarios": [scenario]}))
        catalogue = load_catalogue(scenarios_dir)
        function_state = {"Functions": [{"name": "f", "args": {"function_string": "x^2"}}]}
        results = {
            "config": {},
            "scenarios": [
                {
                    "id": "FN-90",
                    "steps": [
                        {"step": "start", "state": {}},
                        {"step": "setup", "state": {}},
                        {"step": "t1", "state": function_state, "inspection": {"drawables": []}},
                    ],
                }
            ],
        }
        path = tmp_path / "results.json"
        path.write_text(json.dumps(results))
        summary, _ = regrade(path, catalogue)
        assert summary["check_counts"]["unrecorded"] == 1 and summary["unrecorded"] == ["FN-90"]
        assert summary["exit_code"] == 0
        assert "Not evaluated (data not recorded)" in (tmp_path / "summary_regraded.md").read_text()


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

    def test_regrade_cli_with_filters(self, tmp_path: Path) -> None:
        results = tmp_path / "results.json"
        results.write_text(
            json.dumps(
                {
                    "config": {"mode": "replay"},
                    "scenarios": [{"id": "GEO-01", "steps": []}, {"id": "CV-01", "steps": []}],
                }
            )
        )
        result = CliRunner().invoke(cli, ["test", "scenarios", "--regrade", str(results), "--ids", "CV", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["scenarios"] == 1

    def test_regrade_cli(self, tmp_path: Path) -> None:
        results = tmp_path / "results.json"
        results.write_text(json.dumps({"config": {"mode": "replay"}, "scenarios": []}))
        result = CliRunner().invoke(cli, ["test", "scenarios", "--regrade", str(results), "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["scenarios"] == 0
