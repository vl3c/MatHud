"""Gate for the scenario catalogue in scenarios/: every file loads and every reference call is valid.

The loader fills omitted arguments with null and validates each call with the
server's ToolArgumentValidator, so this test fails on an unknown tool, an
unknown argument or an invalid value anywhere in the catalogue.
"""

from __future__ import annotations

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
