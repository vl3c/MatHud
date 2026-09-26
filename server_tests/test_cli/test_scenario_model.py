"""Tests for the scenario loader (cli/scenarios/model.py) and the step grader (grade.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cli.scenarios.grade import ScenarioGrader, StepRecordData
from cli.scenarios.model import ScenarioError, complete_call, load_catalogue, parse_scenario

from server_tests.test_cli.scenario_states import point, state


def scenario(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "GEO-90",
        "title": "Test scenario",
        "tags": ["points"],
        "steps": [
            {
                "user": "Put a point at (1, 2).",
                "reference": [{"tool": "create_point", "args": {"x": 1, "y": 2}}],
                "checks": [{"check": "exists", "select": {"type": "Point", "at": [1, 2]}}],
            }
        ],
    }
    data.update(overrides)
    return data


def parse(data: dict[str, Any], area: str = "GEO") -> tuple[Any, list[str]]:
    problems: list[str] = []
    return parse_scenario(data, area, "geometry.json", Path("."), problems), problems


def write_catalogue(directory: Path, scenarios: list[dict[str, Any]], bugs: Any = None) -> None:
    (directory / "geometry.json").write_text(json.dumps({"schema": 1, "area": "GEO", "scenarios": scenarios}))
    if bugs is not None:
        (directory / "known_bugs.json").write_text(json.dumps(bugs))


class TestCompleteCall:
    def test_fills_required_arguments_with_null(self) -> None:
        problems: list[str] = []
        call = complete_call({"tool": "create_point", "args": {"x": 1, "y": 2}}, "here", problems)
        assert problems == []
        assert call is not None
        assert call.args["x"] == 1 and call.args["name"] is None and "color" in call.args
        assert call.payload() == {"function_name": "create_point", "arguments": call.args}

    def test_unknown_tool_and_argument(self) -> None:
        problems: list[str] = []
        assert complete_call({"tool": "make_point", "args": {}}, "here", problems) is None
        assert complete_call({"tool": "create_point", "args": {"x": 1, "y": 2, "z": 3}}, "there", problems) is None
        assert any("unknown tool 'make_point'" in p for p in problems)
        assert any("has no argument 'z'" in p for p in problems)

    def test_invalid_value(self) -> None:
        problems: list[str] = []
        assert (
            complete_call(
                {"tool": "zoom", "args": {"center_x": 0, "center_y": 0, "range_val": 2, "range_axis": "z"}},
                "w",
                problems,
            )
            is None
        )
        assert problems

    def test_missing_non_nullable_argument(self) -> None:
        problems: list[str] = []
        assert complete_call({"tool": "create_point", "args": {"x": 1}}, "w", problems) is None
        assert problems


class TestParseScenario:
    def test_valid_scenario(self) -> None:
        parsed, problems = parse(scenario())
        assert problems == []
        assert parsed.id == "GEO-90" and parsed.turns == 1 and parsed.steps[0].id == "t1"
        assert parsed.steps[0].calls[0].tool == "create_point"
        assert not parsed.uses_workspaces

    def test_step_kinds_and_ids(self) -> None:
        steps = [
            {"snapshot": "before"},
            {"user": "Undo.", "reference": [{"tool": "undo"}]},
            {"do": [{"tool": "redo"}], "checks": [{"check": "state_equals", "snapshot": "before"}]},
            {"checks": [{"check": "no_tool_errors"}]},
        ]
        parsed, problems = parse(scenario(steps=steps))
        assert problems == []
        assert [(s.kind, s.id) for s in parsed.steps] == [
            ("snapshot", "snap1"),
            ("user", "t1"),
            ("do", "do1"),
            ("checks", "chk1"),
        ]

    def test_snapshot_used_before_taken(self) -> None:
        steps = [
            {"do": [{"tool": "undo"}], "checks": [{"check": "state_equals", "snapshot": "later"}]},
            {"snapshot": "later"},
        ]
        _, problems = parse(scenario(steps=steps))
        assert any("snapshot 'later' is used before it is taken" in p for p in problems)

    def test_auto_snapshots_are_available(self) -> None:
        steps = [
            {
                "do": [{"tool": "undo"}],
                "checks": [
                    {"check": "state_equals", "snapshot": "setup"},
                    {"check": "state_equals", "snapshot": "start"},
                ],
            }
        ]
        _, problems = parse(scenario(steps=steps))
        assert problems == []

    def test_binding_used_before_bound(self) -> None:
        steps = [
            {
                "checks": [
                    {"check": "point_at", "select": "$M", "at": [0, 0]},
                    {"bind": "M", "select": {"type": "Point"}},
                ]
            }
        ]
        _, problems = parse(scenario(steps=steps))
        assert any("$M is used before it is bound" in p for p in problems)

    def test_check_known_must_be_listed_on_the_scenario(self) -> None:
        steps = [{"checks": [{"check": "no_tool_errors", "known": "K7"}]}]
        _, problems = parse(scenario(steps=steps))
        assert any("not listed in known: ['K7']" in p for p in problems)
        _, problems = parse(scenario(steps=steps, known="K7"))
        assert problems == []

    def test_invariant_waivers(self) -> None:
        parsed, problems = parse(scenario(known=["K3"], invariants={"I4": "K3"}))
        assert problems == [] and parsed.invariant_waivers == {"I4": "K3"}
        _, problems = parse(scenario(known=["K3"], invariants={"I9": "K3"}))
        assert any("bad invariant waiver I9" in p for p in problems)

    def test_bad_fields(self) -> None:
        _, problems = parse(scenario(id="GEO-1"))
        assert any("id must look like AREA-NN" in p for p in problems)
        _, problems = parse(scenario(), area="CON")
        assert any("does not belong to area CON" in p for p in problems)
        _, problems = parse(scenario(colour="red"))
        assert any("unknown scenario keys ['colour']" in p for p in problems)
        _, problems = parse(scenario(steps=[{"user": "hi"}]))
        assert any("needs a reference call list" in p for p in problems)
        _, problems = parse(scenario(steps=[{"do": [{"tool": "undo"}], "user": "x"}]))
        assert any("one of snapshot, user or do" in p for p in problems)
        _, problems = parse(scenario(steps=[{"checks": [{"check": "bogus"}]}]))
        assert any("unknown check type 'bogus'" in p for p in problems)

    def test_workspace_detection_and_tolerance(self) -> None:
        steps = [{"user": "Save.", "reference": [{"tool": "save_workspace", "args": {"name": "x"}}]}]
        parsed, problems = parse(scenario(steps=steps, tol=1e-3))
        assert problems == [] and parsed.uses_workspaces
        assert parsed.tolerance.abs == 1e-3

    def test_fixture_loading(self, tmp_path: Path) -> None:
        (tmp_path / "fx.json").write_text(json.dumps({"state": state(point("A", 0, 0))}))
        problems: list[str] = []
        parsed = parse_scenario(scenario(setup={"fixture": "fx.json"}), "GEO", "g.json", tmp_path, problems)
        assert problems == [] and parsed is not None
        assert parsed.fixture_state is not None and parsed.fixture_state["Points"][0]["name"] == "A"
        parse_scenario(scenario(setup={"fixture": "missing.json"}), "GEO", "g.json", tmp_path, problems)
        assert any("fixture missing.json not found" in p for p in problems)


class TestLoadCatalogue:
    def test_loads_and_selects(self, tmp_path: Path) -> None:
        write_catalogue(
            tmp_path,
            [scenario(), scenario(id="GEO-91", smoke=True, tags=["undo"])],
            {"bugs": {"K1": "undo"}, "invariant_waivers": {"I5": "K1"}},
        )
        catalogue = load_catalogue(tmp_path)
        assert [s.id for s in catalogue.scenarios] == ["GEO-90", "GEO-91"]
        assert catalogue.invariant_waivers == {"I5": "K1"}
        assert [s.id for s in catalogue.select(smoke=True)] == ["GEO-91"]
        assert [s.id for s in catalogue.select(tags=["undo"])] == ["GEO-91"]
        assert [s.id for s in catalogue.select(ids=["GEO-90"])] == ["GEO-90"]
        assert len(catalogue.select(ids=["GEO"])) == 2
        assert catalogue.waivers_for(catalogue.scenarios[0]) == {"I5": "K1"}

    def test_scenario_waiver_overrides_global(self, tmp_path: Path) -> None:
        write_catalogue(
            tmp_path,
            [scenario(known=["K2"], invariants={"I5": "K2"})],
            {"bugs": {"K1": "a", "K2": "b"}, "invariant_waivers": {"I5": "K1"}},
        )
        catalogue = load_catalogue(tmp_path)
        assert catalogue.waivers_for(catalogue.scenarios[0]) == {"I5": "K2"}

    def test_marks_and_waivers_of_fixed_bugs_are_refused(self, tmp_path: Path) -> None:
        steps = [{"checks": [{"check": "no_tool_errors", "known": "K1"}]}]
        write_catalogue(
            tmp_path,
            [scenario(steps=steps, known=["K1"])],
            {"bugs": {"K1": "undo"}, "fixed": {"K1": "vl3c/MatHud#73"}, "invariant_waivers": {"I5": "K1"}},
        )
        with pytest.raises(ScenarioError) as info:
            load_catalogue(tmp_path)
        text = "\n".join(info.value.problems)
        assert "marks bugs listed as fixed: ['K1']" in text
        assert "waivers name fixed bugs ['K1']" in text

    def test_duplicate_ids_and_unknown_bugs(self, tmp_path: Path) -> None:
        write_catalogue(tmp_path, [scenario(), scenario(known=["K9"])], {"bugs": {"K1": "a"}})
        with pytest.raises(ScenarioError) as info:
            load_catalogue(tmp_path)
        text = "\n".join(info.value.problems)
        assert "duplicate scenario id GEO-90" in text
        assert "known bugs not in known_bugs.json: ['K9']" in text

    def test_bad_file(self, tmp_path: Path) -> None:
        (tmp_path / "broken.json").write_text("{not json")
        with pytest.raises(ScenarioError, match="not valid JSON"):
            load_catalogue(tmp_path)


class TestScenarioGrader:
    def test_snapshots_bindings_and_invariants(self, tmp_path: Path) -> None:
        steps = [
            {
                "user": "Put a point at (1, 2).",
                "reference": [{"tool": "create_point", "args": {"x": 1, "y": 2}}],
                "checks": [{"bind": "P", "select": {"type": "Point", "at": [1, 2]}}],
            },
            {"snapshot": "one"},
            {
                "do": [{"tool": "undo"}],
                "checks": [{"check": "state_equals", "snapshot": "setup"}, {"check": "absent", "select": "$P"}],
            },
        ]
        parsed, problems = parse(scenario(steps=steps))
        assert problems == []
        grader = ScenarioGrader(parsed, {"I5": "K1"})
        empty = StepRecordData(state=state())
        grader.start(empty)
        assert grader.grade("setup", None, empty) == []
        created = StepRecordData(
            state=state(point("A", 1, 2)),
            calls=[
                {
                    "function_name": "create_point",
                    "arguments": {"x": 1, "y": 2},
                    "result": "Call successful!",
                    "is_error": False,
                }
            ],
            undoable=[True],
            undo_before=0,
            undo_after=2,
        )
        results = grader.grade("t1", parsed.steps[0], created)
        statuses = {r.name: r.status for r in results}
        assert statuses["I5"] == "xfail" and statuses["bind"] == "pass"
        assert grader.grade("snap1", parsed.steps[1], StepRecordData(state=created.state)) == []
        assert "one" in grader.snapshots
        undone = StepRecordData(
            state=state(),
            calls=[{"function_name": "undo", "arguments": {}, "result": "Call successful!", "is_error": False}],
            undoable=[False],
            undo_before=2,
            undo_after=1,
            redo_before=0,
        )
        results = grader.grade("do1", parsed.steps[2], undone)
        assert [r.status for r in results if r.kind == "check"] == ["pass", "pass"]

    def test_needed_samples_dry_run_does_not_bind(self) -> None:
        steps = [
            {
                "user": "Plot x^2.",
                "reference": [{"tool": "draw_function", "args": {"function_string": "x^2", "name": "f"}}],
                "checks": [
                    {"bind": "F", "select": {"type": "Function", "name": "f"}},
                    {"check": "relation", "relation": "function_value", "select": ["$F"], "x": 2, "y": 4},
                ],
            }
        ]
        parsed, problems = parse(scenario(steps=steps))
        assert problems == []
        grader = ScenarioGrader(parsed, {})
        grader.start(StepRecordData(state=state()))
        data = StepRecordData(state=state(("Functions", {"name": "f", "args": {"function_string": "x^2"}})))
        assert grader.needed_samples(parsed.steps[0], data).as_options() == {"samples": {"f": [2.0]}}
        assert grader.bindings == {}
