"""Scenario files: loading, null filling and validation against the tool schemas.

Scenario files are plain JSON under ``scenarios/``, one per area (see section 4.4
of documentation/development/agentic_scenario_testing.md)::

    {"schema": 1, "area": "GEO", "scenarios": [
        {"id": "GEO-04", "title": "...", "tags": [...], "smoke": false, "known": ["K1"],
         "setup": {"fixture": null, "calls": [{"tool": "create_segment", "args": {...}}]},
         "steps": [
            {"snapshot": "before"},
            {"user": "prompt", "reference": [calls], "limits": {...}, "checks": [...]},
            {"do": [calls], "checks": [...]},
            {"checks": [...]}
         ]}
    ]}

A call lists only the arguments that matter; the loader fills every other
required argument with ``null``, as a strict-schema model sends it, and then
validates the call with the server's ``ToolArgumentValidator``. ``known_bugs.json``
next to the scenario files lists the known bugs and the invariant waivers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from cli.config import PROJECT_ROOT
from cli.scenarios.checks import INVARIANT_IDS, KNOWN_BUG_PATTERN, referenced_snapshots, validate_check
from cli.scenarios.geometry import Tolerance
from static.tool_argument_validator import ToolArgumentValidator

SCHEMA_VERSION = 1
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
KNOWN_BUGS_FILE = "known_bugs.json"
# Snapshots every scenario gets for free: the reset canvas, and the canvas after setup.
AUTO_SNAPSHOTS = ("start", "setup")
# Tools that read or write the server's workspace directory.
WORKSPACE_TOOLS = frozenset({"save_workspace", "load_workspace", "delete_workspace", "list_workspaces"})
_ID_PATTERN = re.compile(r"^[A-Z]+-\d{2,}$")
# Report order of the areas (section 5 of the design doc).
AREA_ORDER = ("GEO", "CON", "FN", "AR", "TR", "GR", "ST", "MC", "CV", "WS", "NM", "MT")
_STEP_KINDS = ("snapshot", "user", "do", "checks")


class ScenarioError(Exception):
    """One or more scenario files are invalid."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class ToolCall:
    """A reference or scripted call with every required argument present."""

    tool: str
    args: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        """The client's call format (what the server sends to the browser)."""
        return {"function_name": self.tool, "arguments": self.args}


@dataclass
class Step:
    index: int
    id: str
    kind: str  # snapshot, user, do or checks
    user: Optional[str] = None
    calls: list[ToolCall] = field(default_factory=list)
    snapshot: Optional[str] = None
    checks: list[dict[str, Any]] = field(default_factory=list)
    limits: dict[str, Any] = field(default_factory=dict)

    @property
    def runs_calls(self) -> bool:
        return self.kind in ("user", "do")


@dataclass
class Scenario:
    id: str
    title: str
    area: str
    file: str
    tags: list[str] = field(default_factory=list)
    smoke: bool = False
    known: list[str] = field(default_factory=list)
    targets: str = ""
    tolerance: Tolerance = field(default_factory=Tolerance)
    invariant_waivers: dict[str, str] = field(default_factory=dict)
    allow_large: bool = False
    fixture: Optional[str] = None
    fixture_state: Optional[dict[str, Any]] = None
    setup_calls: list[ToolCall] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)

    @property
    def turns(self) -> int:
        return sum(1 for step in self.steps if step.kind == "user")

    @property
    def uses_workspaces(self) -> bool:
        calls = self.setup_calls + [call for step in self.steps for call in step.calls]
        return any(call.tool in WORKSPACE_TOOLS for call in calls)

    def all_known(self) -> set[str]:
        """Every known-bug id this scenario mentions (scenario, checks and waivers)."""
        found = set(self.known) | set(self.invariant_waivers.values())
        for step in self.steps:
            for check in step.checks:
                if isinstance(check.get("known"), str):
                    found.add(check["known"])
        return found


@dataclass
class Catalogue:
    scenarios: list[Scenario]
    bugs: dict[str, str] = field(default_factory=dict)
    invariant_waivers: dict[str, str] = field(default_factory=dict)

    def select(
        self,
        *,
        smoke: bool = False,
        tags: Optional[Iterable[str]] = None,
        ids: Optional[Iterable[str]] = None,
    ) -> list[Scenario]:
        """Scenarios matching every given filter (ids match exactly or by area prefix)."""
        wanted_tags = {t for t in (tags or []) if t}
        wanted_ids = [i for i in (ids or []) if i]
        chosen = []
        for scenario in self.scenarios:
            if smoke and not scenario.smoke:
                continue
            if wanted_tags and not wanted_tags & set(scenario.tags):
                continue
            if wanted_ids and not any(
                scenario.id == wanted or scenario.id.startswith(wanted + "-") or scenario.area == wanted
                for wanted in wanted_ids
            ):
                continue
            chosen.append(scenario)
        return chosen

    def waivers_for(self, scenario: Scenario) -> dict[str, str]:
        waivers = dict(self.invariant_waivers)
        waivers.update(scenario.invariant_waivers)
        return waivers


# ----------------------------------------------------------------------
# Calls
# ----------------------------------------------------------------------


def complete_call(raw: Any, where: str, problems: list[str]) -> Optional[ToolCall]:
    """Fill a call's missing required arguments with null and validate it."""
    if not isinstance(raw, dict) or not isinstance(raw.get("tool"), str):
        problems.append(f"{where}: a call needs a tool name")
        return None
    tool = raw["tool"]
    args = raw.get("args", {})
    if not isinstance(args, dict):
        problems.append(f"{where}: args of {tool} must be an object")
        return None
    extra = set(raw) - {"tool", "args"}
    if extra:
        problems.append(f"{where}: unknown call keys {sorted(extra)}")
    schema = ToolArgumentValidator.get_schema(tool)
    if schema is None:
        problems.append(f"{where}: unknown tool {tool!r}")
        return None
    properties = schema.get("properties", {}) or {}
    for key in args:
        if key not in properties:
            problems.append(f"{where}: {tool} has no argument {key!r}")
    full = {key: args.get(key) for key in schema.get("required", [])}
    full.update(args)
    result = ToolArgumentValidator.validate(tool, full)
    if not result["valid"]:
        problems.extend(f"{where}: {error}" for error in result["errors"])
        return None
    return ToolCall(tool, result["arguments"])


def _calls(raw: Any, where: str, problems: list[str]) -> list[ToolCall]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        problems.append(f"{where}: calls must be a list")
        return []
    calls = []
    for index, item in enumerate(raw, start=1):
        call = complete_call(item, f"{where} call {index}", problems)
        if call is not None:
            calls.append(call)
    return calls


# ----------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------


def _known_list(value: Any, where: str, problems: list[str]) -> list[str]:
    if value is None:
        return []
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(i, str) and KNOWN_BUG_PATTERN.match(i) for i in items):
        problems.append(f"{where}: known must be K<n> or a list of them, got {value!r}")
        return []
    return list(items)


def _parse_step(raw: Any, index: int, counters: dict[str, int], where: str, problems: list[str]) -> Optional[Step]:
    if not isinstance(raw, dict):
        problems.append(f"{where}: a step must be an object")
        return None
    kinds = [kind for kind in ("snapshot", "user", "do") if kind in raw]
    if len(kinds) > 1:
        problems.append(f"{where}: a step has one of snapshot, user or do, found {kinds}")
        return None
    kind = kinds[0] if kinds else "checks"
    if kind == "checks" and "checks" not in raw:
        problems.append(f"{where}: empty step")
        return None
    allowed = {"id", "checks", "limits", "reference", "snapshot", "user", "do", "note"}
    extra = set(raw) - allowed
    if extra:
        problems.append(f"{where}: unknown step keys {sorted(extra)}")
    counters[kind] = counters.get(kind, 0) + 1
    default_id = {"user": "t", "do": "do", "checks": "chk", "snapshot": "snap"}[kind] + str(counters[kind])
    step = Step(index=index, id=str(raw.get("id") or default_id), kind=kind)
    if kind == "snapshot":
        if not isinstance(raw["snapshot"], str) or not raw["snapshot"]:
            problems.append(f"{where}: snapshot needs a name")
        step.snapshot = str(raw["snapshot"])
    elif kind == "user":
        if not isinstance(raw["user"], str) or not raw["user"].strip():
            problems.append(f"{where}: user needs the prompt text")
        step.user = str(raw["user"])
        if not raw.get("reference"):
            problems.append(f"{where}: a user turn needs a reference call list")
        step.calls = _calls(raw.get("reference"), f"{where} reference", problems)
        limits = raw.get("limits") or {}
        if not isinstance(limits, dict):
            problems.append(f"{where}: limits must be an object")
            limits = {}
        step.limits = limits
    elif kind == "do":
        if not raw["do"]:
            problems.append(f"{where}: do needs calls")
        step.calls = _calls(raw["do"], f"{where} do", problems)
    checks = raw.get("checks") or []
    if not isinstance(checks, list):
        problems.append(f"{where}: checks must be a list")
        checks = []
    for number, check in enumerate(checks, start=1):
        problems.extend(f"{where} check {number}: {problem}" for problem in validate_check(check))
    step.checks = [check for check in checks if isinstance(check, dict)]
    return step


def _load_fixture(name: str, base: Path, where: str, problems: list[str]) -> Optional[dict[str, Any]]:
    for candidate in (base / name, PROJECT_ROOT / name):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except ValueError as exc:
                problems.append(f"{where}: fixture {name} is not JSON: {exc}")
                return None
            if isinstance(data, dict) and isinstance(data.get("state"), dict):
                data = data["state"]
            if not isinstance(data, dict):
                problems.append(f"{where}: fixture {name} is not a canvas state")
                return None
            return data
    problems.append(f"{where}: fixture {name} not found")
    return None


def parse_scenario(raw: Any, area: str, file: str, base: Path, problems: list[str]) -> Optional[Scenario]:
    """Build a Scenario from its JSON object, appending any problems."""
    if not isinstance(raw, dict):
        problems.append(f"{file}: a scenario must be an object")
        return None
    scenario_id = str(raw.get("id", ""))
    where = f"{file} {scenario_id or '?'}"
    if not _ID_PATTERN.match(scenario_id):
        problems.append(f"{where}: id must look like AREA-NN")
    elif not scenario_id.startswith(area + "-"):
        problems.append(f"{where}: id does not belong to area {area}")
    allowed = {
        "id", "title", "tags", "smoke", "known", "targets", "tol",
        "invariants", "allow_large", "setup", "steps", "note",
    }  # fmt: skip
    extra = set(raw) - allowed
    if extra:
        problems.append(f"{where}: unknown scenario keys {sorted(extra)}")
    try:
        tolerance = Tolerance().merged(raw.get("tol"))
    except ValueError as exc:
        problems.append(f"{where}: {exc}")
        tolerance = Tolerance()
    scenario = Scenario(
        id=scenario_id,
        title=str(raw.get("title", "")),
        area=area,
        file=file,
        tags=[str(t) for t in raw.get("tags", []) or []],
        smoke=bool(raw.get("smoke", False)),
        known=_known_list(raw.get("known"), where, problems),
        targets=str(raw.get("targets", "")),
        tolerance=tolerance,
        allow_large=bool(raw.get("allow_large", False)),
    )
    waivers = raw.get("invariants") or {}
    if not isinstance(waivers, dict):
        problems.append(f"{where}: invariants must map I<n> to K<n>")
        waivers = {}
    for invariant, bug in waivers.items():
        if invariant not in INVARIANT_IDS or not isinstance(bug, str) or not KNOWN_BUG_PATTERN.match(bug):
            problems.append(f"{where}: bad invariant waiver {invariant}: {bug!r}")
        else:
            scenario.invariant_waivers[invariant] = bug
    setup = raw.get("setup") or {}
    if not isinstance(setup, dict):
        problems.append(f"{where}: setup must be an object")
        setup = {}
    if setup.get("fixture"):
        scenario.fixture = str(setup["fixture"])
        scenario.fixture_state = _load_fixture(scenario.fixture, base, where, problems)
    scenario.setup_calls = _calls(setup.get("calls"), f"{where} setup", problems)
    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        problems.append(f"{where}: steps must be a non-empty list")
        raw_steps = []
    counters: dict[str, int] = {}
    for index, raw_step in enumerate(raw_steps):
        step = _parse_step(raw_step, index, counters, f"{where} step {index + 1}", problems)
        if step is not None:
            scenario.steps.append(step)
    _check_references(scenario, where, problems)
    return scenario


def _check_references(scenario: Scenario, where: str, problems: list[str]) -> None:
    """Snapshots and step ids are defined before use; check-level known bugs are listed on the scenario."""
    snapshots = set(AUTO_SNAPSHOTS)
    step_ids: set[str] = set()
    bindings: set[str] = set()
    for step in scenario.steps:
        if step.id in step_ids:
            problems.append(f"{where}: duplicate step id {step.id}")
        step_ids.add(step.id)
        if step.kind == "snapshot" and step.snapshot:
            snapshots.add(step.snapshot)
        for check in step.checks:
            for name in referenced_snapshots(check):
                if name not in snapshots:
                    problems.append(f"{where} step {step.id}: snapshot {name!r} is used before it is taken")
            for ref in _binding_refs(check):
                if ref not in bindings:
                    problems.append(f"{where} step {step.id}: ${ref} is used before it is bound")
            if isinstance(check.get("bind"), str):
                bindings.add(check["bind"])
    missing = sorted(scenario.all_known() - set(scenario.known))
    if missing:
        problems.append(f"{where}: known bugs used by checks or waivers but not listed in known: {missing}")


def _binding_refs(check: dict[str, Any]) -> list[str]:
    refs: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str) and value.startswith("$"):
            refs.append(value[1:])
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(check.get("select"))
    walk(check.get("except"))
    return refs


def load_file(path: Path, problems: list[str]) -> list[Scenario]:
    """Scenarios of one area file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        problems.append(f"{path.name}: not valid JSON: {exc}")
        return []
    if not isinstance(data, dict) or data.get("schema") != SCHEMA_VERSION:
        problems.append(f"{path.name}: expected an object with schema {SCHEMA_VERSION}")
        return []
    area = data.get("area")
    if not isinstance(area, str) or not area:
        problems.append(f"{path.name}: needs an area")
        return []
    scenarios = []
    for raw in data.get("scenarios", []) or []:
        scenario = parse_scenario(raw, area, path.name, path.parent, problems)
        if scenario is not None:
            scenarios.append(scenario)
    return scenarios


def load_known_bugs(directory: Path, problems: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """``(bugs, invariant_waivers)`` from ``known_bugs.json`` (empty when the file is absent)."""
    path = directory / KNOWN_BUGS_FILE
    if not path.is_file():
        return {}, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        problems.append(f"{KNOWN_BUGS_FILE}: not valid JSON: {exc}")
        return {}, {}
    bugs = data.get("bugs", {}) if isinstance(data, dict) else {}
    waivers = data.get("invariant_waivers", {}) if isinstance(data, dict) else {}
    if not isinstance(bugs, dict) or not all(KNOWN_BUG_PATTERN.match(str(k)) for k in bugs):
        problems.append(f"{KNOWN_BUGS_FILE}: bugs must map K<n> to a title")
        bugs = {}
    if not isinstance(waivers, dict) or not all(k in INVARIANT_IDS for k in waivers):
        problems.append(f"{KNOWN_BUGS_FILE}: invariant_waivers must map I<n> to K<n>")
        waivers = {}
    return {str(k): str(v) for k, v in bugs.items()}, {str(k): str(v) for k, v in waivers.items()}


def _catalogue_order(scenario: Scenario) -> tuple[int, str, int]:
    """Areas in the design doc's order (unknown areas last), then by scenario number."""
    area_rank = AREA_ORDER.index(scenario.area) if scenario.area in AREA_ORDER else len(AREA_ORDER)
    number = scenario.id.rsplit("-", 1)[-1]
    return area_rank, scenario.area, int(number) if number.isdigit() else 0


def load_catalogue(directory: Optional[Path] = None) -> Catalogue:
    """Load and validate every ``*.json`` scenario file in ``directory``.

    Raises:
        ScenarioError: listing every problem found.
    """
    directory = directory or SCENARIOS_DIR
    problems: list[str] = []
    bugs, waivers = load_known_bugs(directory, problems)
    scenarios: list[Scenario] = []
    for path in sorted(directory.glob("*.json")):
        if path.name == KNOWN_BUGS_FILE:
            continue
        scenarios.extend(load_file(path, problems))
    seen: set[str] = set()
    for scenario in scenarios:
        if scenario.id in seen:
            problems.append(f"{scenario.file}: duplicate scenario id {scenario.id}")
        seen.add(scenario.id)
        if bugs:
            unknown = sorted(scenario.all_known() - set(bugs))
            if unknown:
                problems.append(f"{scenario.file} {scenario.id}: known bugs not in {KNOWN_BUGS_FILE}: {unknown}")
    scenarios.sort(key=_catalogue_order)
    if bugs:
        unknown_waivers = sorted(set(waivers.values()) - set(bugs))
        if unknown_waivers:
            problems.append(f"{KNOWN_BUGS_FILE}: waivers name unknown bugs {unknown_waivers}")
    if problems:
        raise ScenarioError(problems)
    return Catalogue(scenarios=scenarios, bugs=bugs, invariant_waivers=waivers)
