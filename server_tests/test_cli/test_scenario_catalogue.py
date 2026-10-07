"""Gate for the scenario catalogue in scenarios/: every file loads and every reference call is valid.

The loader fills omitted arguments with null and validates each call with the
server's ToolArgumentValidator, so this test fails on an unknown tool, an
unknown argument or an invalid value anywhere in the catalogue.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cli.scenarios.model import AREA_ORDER, load_catalogue

# The design doc's catalogue (section 5): 72 scenarios, 11 of them in the smoke subset.
DOC_SMOKE = {"GEO-01", "CON-01", "FN-01", "TR-01", "GR-01", "ST-01", "MC-01", "CV-02", "WS-01", "NM-03", "MT-01"}
DOC_COUNTS = {
    "GEO": 16,
    "CON": 6,
    "FN": 9,
    "AR": 4,
    "TR": 6,
    "GR": 6,
    "ST": 4,
    "MC": 4,
    "CV": 4,
    "WS": 3,
    "NM": 6,
    "MT": 4,
}


def test_catalogue_loads_and_validates() -> None:
    catalogue = load_catalogue()
    assert catalogue.scenarios


def test_every_doc_scenario_is_ported() -> None:
    catalogue = load_catalogue()
    ids = {scenario.id for scenario in catalogue.scenarios}
    doc_ids = {f"{area}-{number:02d}" for area, count in DOC_COUNTS.items() for number in range(1, count + 1)}
    assert len(doc_ids) == 72
    assert doc_ids <= ids
    assert {s.id for s in catalogue.scenarios if s.smoke} == DOC_SMOKE
    assert sum(s.turns for s in catalogue.scenarios if s.id in doc_ids) == 103


def test_areas_are_in_report_order() -> None:
    catalogue = load_catalogue()
    areas = [scenario.area for scenario in catalogue.scenarios]
    ranks = [AREA_ORDER.index(area) for area in areas]
    assert ranks == sorted(ranks)


def test_every_scenario_checks_something() -> None:
    catalogue = load_catalogue()
    for scenario in catalogue.scenarios:
        assert any(step.checks for step in scenario.steps), scenario.id
        assert scenario.tags, scenario.id


def test_known_bugs_are_declared() -> None:
    catalogue = load_catalogue()
    assert catalogue.bugs, "scenarios/known_bugs.json lists the known bugs"
    for scenario in catalogue.scenarios:
        assert scenario.all_known() <= set(catalogue.bugs), scenario.id
    assert set(catalogue.invariant_waivers.values()) <= set(catalogue.bugs)


def test_fixed_bugs_are_not_marked() -> None:
    """Bugs fixed on main (vl3c/MatHud#72, #73) are regression guards, never expected failures."""
    catalogue = load_catalogue()
    assert {"K1", "K2", "K6", "K7", "K18", "K21"} <= set(catalogue.fixed)
    for scenario in catalogue.scenarios:
        assert not scenario.all_known() & set(catalogue.fixed), scenario.id
    assert not catalogue.invariant_waivers, "the global I5:K1 waiver went away with the K1 fix"


def test_null_string_arguments_load_as_null() -> None:
    """GEO-19 sends "null", "None" and "undefined" as a local model does; the loader applies the server's rule."""
    catalogue = load_catalogue()
    [scenario] = [s for s in catalogue.scenarios if s.id == "GEO-19"]
    polygon, function = scenario.steps[0].calls
    assert (polygon.args["color"], polygon.args["name"], polygon.args["subtype"]) == (None, None, None)
    assert (function.args["name"], function.args["left_bound"], function.args["color"]) == (None, None, None)
    assert function.args["function_string"] == "x^2/4"


_CREATING_PREFIXES = ("create_", "draw_", "construct_", "plot_", "fit_", "generate_")


def _created_names(scenario: Any) -> set[str]:
    """Names a user turn's reference gives to the objects it creates (the model may choose others)."""
    return {
        str(call.args["name"])
        for step in scenario.steps
        if step.kind == "user"
        for call in step.calls
        if isinstance(call.args.get("name"), str) and call.tool.startswith(_CREATING_PREFIXES)
    }


def _selected_names(value: Any) -> list[str]:
    """Names a check selects by: a selector's ``name`` and the string values of its ``where: {"args.*": ...}``."""
    if isinstance(value, dict):
        own = [value["name"]] if isinstance(value.get("type"), str) and isinstance(value.get("name"), str) else []
        where = value.get("where")
        if isinstance(where, dict):
            own += [v for k, v in where.items() if str(k).startswith("args.") and isinstance(v, str)]
        return own + [name for item in value.values() for name in _selected_names(item)]
    if isinstance(value, list):
        return [name for item in value for name in _selected_names(item)]
    return []


def _name_problems(scenarios: list[Any]) -> list[str]:
    """Checks that select a name only a reference chose, before (or without) a prompt giving it."""
    problems = []
    for scenario in scenarios:
        chosen = _created_names(scenario)
        prompts = ""
        for step in scenario.steps:
            # Only prompts the model has seen by this step can have given the name.
            prompts += " " + (step.user or "")
            for name in _selected_names(step.checks):
                named = re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", prompts)
                if name in chosen and not named:
                    problems.append(f"{scenario.id} {step.id}: selects {name!r}, a name only the reference chose")
    return problems


def test_checks_select_only_names_the_prompt_gives() -> None:
    """A live model names what the prompt does not name as it likes, so checks must not select those names.

    MC-01 selected the parabola as "p", a name only its reference chose, and failed a correct live run.
    """
    problems = _name_problems(load_catalogue().scenarios)
    assert not problems, problems


def test_the_name_gate_reads_where_clauses_and_prompt_order(tmp_path: Path) -> None:
    draw = {"tool": "draw_function", "args": {"function_string": "x", "name": "f"}}
    area = {"check": "exists", "select": {"type": "FunctionsBoundedColoredArea", "where": {"args.func1": "f"}}}
    scenario = {
        "id": "FN-90",
        "title": "names",
        "tags": ["functions"],
        "steps": [
            {"user": "Plot y = x.", "reference": [draw], "checks": [area]},
            {
                "user": "Call it f.",
                "reference": [draw],
                "checks": [{"check": "exists", "select": {"type": "Function", "name": "f"}}],
            },
        ],
    }
    (tmp_path / "functions.json").write_text(json.dumps({"schema": 1, "area": "FN", "scenarios": [scenario]}))
    (tmp_path / "known_bugs.json").write_text(json.dumps({"bugs": {}, "invariant_waivers": {}}))
    problems = _name_problems(load_catalogue(tmp_path).scenarios)
    # The where clause in t1 is caught (the prompt naming f comes only in t2); t2's selector is fine.
    assert problems == ["FN-90 t1: selects 'f', a name only the reference chose"]
