"""Scenario checks: selectors, outcome checks and the always-on invariants.

Checks are pure functions of stored data: the canvas views after each step
(state plus inspection view), the tool calls the step executed and, in live
mode, the final assistant text. See section 4.5 of
documentation/development/agentic_scenario_testing.md for the language.

A check is a JSON object. ``{"bind": "M", "select": ...}`` names an object for
later checks (``"$M"``); every other check has a ``"check"`` type from
``CHECK_TYPES``. Any check may carry ``"known": "K<n>"`` (an expected failure,
reported as xfail, or xpass when it passes) and ``"tol"`` (a number, or
``{"abs": ..., "rel": ...}``).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from cli.scenarios.geometry import (
    COLORED_AREA_TYPES,
    FUNCTION_TYPES,
    GRAPH_TYPES,
    PLOT_TYPES,
    POLYGON_TYPES,
    CanvasObject,
    CanvasView,
    Coord,
    MissingSample,
    Tolerance,
    ccw_angle_degrees,
    cross,
    direction,
    distance,
    dot,
    iter_numbers,
    line_distance,
    point_in_polygon,
    polygon_area,
    segment_distance,
    triangle_flags,
    unit_angle_between,
)

CHECK_TYPES = frozenset(
    {
        "exists",
        "absent",
        "count",
        "point_at",
        "moved",
        "relation",
        "attribute",
        "state_equals",
        "unchanged_except",
        "tool_called",
        "tool_not_called",
        "max_tool_calls",
        "no_tool_errors",
        "tool_error",
        "tool_result",
        "answer_mentions",
    }
)
RELATIONS = frozenset(
    {
        "point_on_circle",
        "point_on_segment",
        "point_on_line",
        "point_on_function",
        "collinear",
        "parallel",
        "perpendicular",
        "midpoint_of",
        "equal_length",
        "equal_angles",
        "tangent_to",
        "distance",
        "length",
        "angle_deg",
        "area",
        "function_value",
        "inside",
        "slope",
        "direction",
    }
)
COMPARISON_OPS = frozenset(
    {
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "contains",
        "not_contains",
        "set_eq",
        "in",
        "matches",
        "has_numbers",
        "len",
        "is_null",
        "not_null",
    }
)
SELECTOR_KEYS = frozenset(
    {
        "type",
        "name",
        "at",
        "ends",
        "origin",
        "tip",
        "contains",
        "through",
        "vertices",
        "center",
        "radius",
        "radius_x",
        "radius_y",
        "samples",
        "only",
        "new_since",
        "where",
        "tol",
        "coords",
        "slope",
        "within",
    }
)
INVARIANT_IDS = ("I1", "I2", "I3", "I4", "I5", "I6", "I7")
KNOWN_BUG_PATTERN = re.compile(r"^K\d+$")
# Tools the harness never counts as the model's work.
UNCOUNTED_TOOLS = frozenset({"search_tools"})
# Magnitude above which a coordinate counts as runaway (I7).
MAX_SANE_MAGNITUDE = 1e12
# |f| beyond which a probe next to a listed vertical asymptote counts as blowing up (I3).
ASYMPTOTE_BLOWUP = 1e4
# Mutating tools whose success must change something (I4); set_* and reset_canvas can be no-ops.
_MUTATING_PREFIXES = ("create_", "delete_", "update_", "construct_", "draw_", "plot_")
_MUTATING_TOOLS = frozenset(
    {
        "translate_object",
        "rotate_object",
        "reflect_object",
        "scale_object",
        "shear_object",
        "generate_graph",
        "fit_regression",
        "clear_canvas",
        "undo",
        "redo",
        "zoom",
        "load_workspace",
    }
)
# Results that claim success without saying what happened (static/client/constants.py successful_call_message).
GENERIC_SUCCESS_MESSAGES = frozenset({"Call successful!", ""})
# Tools matching a mutating prefix that do not change the canvas.
_NON_CANVAS_TOOLS = frozenset({"delete_workspace"})
# Tools whose "name" argument is a hint for a new object's name (I4 naming rule).
_NAMING_PREFIXES = ("create_", "construct_", "draw_", "generate_", "plot_", "fit_")
_NAME_HINT_KEYS = ("name", "angle_name")


class SelectorError(Exception):
    """A selector matched the wrong number of objects or is malformed."""


class CheckFailure(Exception):
    """A check's condition does not hold."""

    def __init__(self, message: str, expected: Any = None, actual: Any = None) -> None:
        super().__init__(message)
        self.expected = expected
        self.actual = actual


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------


@dataclass
class CheckResult:
    """Outcome of one check or invariant."""

    id: str
    kind: str  # "check", "bind" or "invariant"
    name: str  # the check type, or the invariant id
    passed: Optional[bool]  # None: skipped (e.g. an answer check in replay mode)
    known: Optional[str] = None
    warning: bool = False
    error: bool = False
    unrecorded: bool = False
    message: str = ""
    expected: Any = None
    actual: Any = None
    spec: Optional[dict[str, Any]] = None

    @property
    def status(self) -> str:
        """pass, fail, xfail, xpass, warn, skip, unrecorded or error.

        A known bug turns a failure into xfail and a pass into xpass, except for
        invariant waivers, which never report xpass (an invariant can hold in one
        step and break in the next). A check that could not be evaluated is an
        error even when it is marked known: a crash is not the bug's failure.
        ``unrecorded`` means the check needs data the run did not store (a
        function sample, when regrading); rerun the replay to evaluate it.
        """
        if self.error:
            return "error"
        if self.unrecorded:
            return "unrecorded"
        if self.passed is None:
            return "skip"
        if self.known:
            if self.passed:
                return "pass" if self.kind == "invariant" else "xpass"
            return "xfail"
        if self.warning and not self.passed:
            return "warn"
        return "pass" if self.passed else "fail"

    @property
    def unexpected(self) -> bool:
        return self.status in ("fail", "error")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "status": self.status,
            "passed": self.passed,
        }
        for key in ("known", "message", "expected", "actual", "spec"):
            value = getattr(self, key)
            if value not in (None, ""):
                data[key] = _jsonable(value)
        return data


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


# ----------------------------------------------------------------------
# Step data and context
# ----------------------------------------------------------------------


@dataclass
class StepData:
    """What a step left behind for its checks: executed calls, undo depths, final text."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    undoable: list[bool] = field(default_factory=list)
    undo_before: Optional[int] = None
    undo_after: Optional[int] = None
    redo_before: Optional[int] = None
    redo_after: Optional[int] = None
    final_text: Optional[str] = None
    mode: str = "replay"

    @property
    def counted_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c.get("function_name") not in UNCOUNTED_TOOLS]


@dataclass
class CheckContext:
    """Everything a check can look at."""

    view: CanvasView
    step: StepData = field(default_factory=StepData)
    snapshots: dict[str, CanvasView] = field(default_factory=dict)
    bindings: dict[str, tuple[str, str]] = field(default_factory=dict)
    tolerance: Tolerance = field(default_factory=Tolerance)

    def snapshot(self, name: Any) -> CanvasView:
        if not isinstance(name, str) or name not in self.snapshots:
            raise SelectorError(f"unknown snapshot {name!r}")
        return self.snapshots[name]


def call_is_error(call: dict[str, Any]) -> bool:
    """A call failed if it is flagged, returned an ``Error...`` string, or an ``{"error": ...}`` dict (K21)."""
    result = call.get("result")
    if call.get("is_error"):
        return True
    if isinstance(result, str) and result.startswith("Error"):
        return True
    return isinstance(result, dict) and ("error" in result or result.get("status") == "error")


# ----------------------------------------------------------------------
# Validation (used by the loader)
# ----------------------------------------------------------------------


# Keys every check may carry, and the keys each check type accepts on top of them.
_COMMON_CHECK_KEYS = frozenset({"check", "known", "tol", "id", "note"})
_RELATION_PARAMS = frozenset({"value", "x", "y", "values", "h", "slope_tol", "parallel_to", "perpendicular_to"})
_CHECK_KEYS: dict[str, frozenset[str]] = {
    "exists": frozenset({"select"}),
    "absent": frozenset({"select"}),
    "count": frozenset({"select", "mod"}) | COMPARISON_OPS,
    "point_at": frozenset({"select", "at"}),
    "moved": frozenset({"select", "since", "by"}),
    "relation": frozenset({"relation", "select"}) | _RELATION_PARAMS,
    "attribute": frozenset({"select", "target", "path", "same_as", "mod"}) | COMPARISON_OPS,
    "state_equals": frozenset({"snapshot", "ignore", "inspect", "view", "match_names"}),
    "unchanged_except": frozenset({"since", "except"}),
    "tool_called": frozenset({"tool"}),
    "tool_not_called": frozenset({"tool"}),
    "max_tool_calls": frozenset({"max"}),
    "no_tool_errors": frozenset(),
    "tool_error": frozenset({"tool", "index"}),
    "tool_result": frozenset({"tool", "path", "index", "mod"}) | COMPARISON_OPS,
    "answer_mentions": frozenset({"numbers", "words", "names"}),
}
_BIND_KEYS = frozenset({"bind", "select", "known", "tol", "id", "note"})
# Keys each check type must carry.
_REQUIRED_CHECK_KEYS: dict[str, tuple[str, ...]] = {
    "exists": ("select",),
    "absent": ("select",),
    "count": ("select",),
    "point_at": ("select", "at"),
    "moved": ("select", "since", "by"),
    "relation": ("relation", "select"),
    "attribute": (),
    "state_equals": ("snapshot",),
    "unchanged_except": ("since",),
    "tool_called": ("tool",),
    "tool_not_called": ("tool",),
    "max_tool_calls": ("max",),
    "no_tool_errors": (),
    "tool_error": ("tool",),
    "tool_result": ("tool",),
    "answer_mentions": (),
}
# Required keys whose absence the checks below already report with a specific message.
_KEYS_WITH_OWN_MESSAGES = frozenset({"select", "snapshot", "since", "tool", "max", "relation"})
# Number of selectors each relation takes: (minimum, maximum or None for no limit).
_RELATION_ARITY: dict[str, tuple[int, Optional[int]]] = {
    "point_on_circle": (2, None),
    "point_on_segment": (2, None),
    "point_on_line": (2, None),
    "point_on_function": (2, None),
    "collinear": (2, None),
    "parallel": (2, 2),
    "perpendicular": (2, 2),
    "midpoint_of": (2, 2),
    "equal_length": (2, None),
    "equal_angles": (3, 3),
    "tangent_to": (2, 2),
    "distance": (2, 2),
    "length": (1, 1),
    "angle_deg": (1, 3),
    "area": (1, 1),
    "function_value": (1, 1),
    "inside": (2, 2),
    "slope": (1, 1),
    "direction": (1, 1),
}
# Relations that compare one measured value with "value".
_VALUE_RELATIONS = frozenset({"distance", "length", "slope", "area", "angle_deg"})


def validate_check(check: Any) -> list[str]:
    """Structural problems with one check definition (empty when fine)."""
    if not isinstance(check, dict):
        return ["check must be an object"]
    problems: list[str] = []
    known = check.get("known")
    if known is not None and (not isinstance(known, str) or not KNOWN_BUG_PATTERN.match(known)):
        problems.append(f"known must look like K<n>, got {known!r}")
    if "bind" in check:
        problems.extend(f"unknown key {key!r} for bind" for key in sorted(set(check) - _BIND_KEYS))
        if not isinstance(check["bind"], str) or not check["bind"]:
            problems.append("bind needs a name")
        problems.extend(_validate_selector(check.get("select")))
        return problems
    kind = check.get("check")
    if kind not in CHECK_TYPES:
        return problems + [f"unknown check type {kind!r}"]
    allowed = _COMMON_CHECK_KEYS | _CHECK_KEYS[str(kind)]
    problems.extend(f"unknown key {key!r} for {kind}" for key in sorted(set(check) - allowed))
    problems.extend(
        f"{kind} needs {key}"
        for key in _REQUIRED_CHECK_KEYS[str(kind)]
        if key not in check and key not in _KEYS_WITH_OWN_MESSAGES
    )
    if kind == "point_at" and "at" in check and not _is_pair(check["at"]):
        problems.append("point_at needs at as [x, y]")
    if kind == "moved" and "by" in check and not _is_pair(check["by"]):
        problems.append("moved needs by as [dx, dy]")
    if kind in ("exists", "absent", "count", "point_at", "moved", "attribute") and "select" in check:
        problems.extend(_validate_selector(check["select"]))
    if kind in ("exists", "absent", "count", "point_at", "moved") and "select" not in check:
        problems.append(f"{kind} needs a select")
    if kind == "relation":
        if check.get("relation") not in RELATIONS:
            problems.append(f"unknown relation {check.get('relation')!r}")
        selectors = check.get("select")
        if not isinstance(selectors, list) or not selectors:
            problems.append("relation needs a list of selectors")
        else:
            for selector in selectors:
                problems.extend(_validate_selector(selector))
        relation = check.get("relation")
        arity = _RELATION_ARITY.get(str(relation))
        if arity is not None and isinstance(selectors, list) and selectors:
            low, high = arity
            if len(selectors) < low or (high is not None and len(selectors) > high):
                expected_count = str(low) if low == high else f"{low} or more" if high is None else f"{low} to {high}"
                problems.append(f"relation {relation} takes {expected_count} selectors, got {len(selectors)}")
        if relation == "tangent_to" and isinstance(selectors, list) and len(selectors) == 2:
            target = selectors[1]
            if isinstance(target, dict) and target.get("type") in FUNCTION_TYPES | {"AnyFunction"} and "x" not in check:
                problems.append("relation tangent_to a function needs x")
        if relation == "direction" and not ("parallel_to" in check or "perpendicular_to" in check):
            problems.append("relation direction needs parallel_to or perpendicular_to")
        if relation in _VALUE_RELATIONS and "value" not in check:
            problems.append(f"relation {relation} needs a value")
        if relation == "function_value" and not ("values" in check or ("x" in check and "y" in check)):
            problems.append("relation function_value needs x and y, or values")
    if kind == "unchanged_except":
        for selector in check.get("except") or []:
            problems.extend(_validate_selector(selector))
    if kind == "state_equals" and not isinstance(check.get("snapshot"), str):
        problems.append("state_equals needs a snapshot name")
    if kind in ("moved", "unchanged_except") and not isinstance(check.get("since"), str):
        problems.append(f"{kind} needs since (a snapshot name)")
    if kind in ("tool_called", "tool_not_called", "tool_error", "tool_result") and not isinstance(
        check.get("tool"), str
    ):
        problems.append(f"{kind} needs a tool")
    if kind == "max_tool_calls" and not isinstance(check.get("max"), int):
        problems.append("max_tool_calls needs max (an integer)")
    if kind in ("attribute", "tool_result", "count") and not (
        any(op in check for op in COMPARISON_OPS) or (kind == "attribute" and isinstance(check.get("same_as"), str))
    ):
        problems.append(f"{kind} needs a comparison ({', '.join(sorted(COMPARISON_OPS))})")
    if kind == "attribute" and "select" not in check and check.get("target") not in ("view", "state", "inspection"):
        problems.append("attribute needs a select or target (view, state or inspection)")
    return problems


def _validate_selector(selector: Any) -> list[str]:
    if isinstance(selector, str):
        return [] if selector.startswith("$") and len(selector) > 1 else [f"bad selector reference {selector!r}"]
    if not isinstance(selector, dict):
        return [f"selector must be an object or $binding, got {selector!r}"]
    unknown = set(selector) - SELECTOR_KEYS
    problems = [f"unknown selector key {key!r}" for key in sorted(unknown)]
    if "coords" not in selector and not isinstance(selector.get("type"), str):
        problems.append("selector needs a type")
    return problems


def referenced_snapshots(check: dict[str, Any]) -> list[str]:
    """Snapshot names a check refers to (for the loader's ordering check)."""
    names: list[str] = []
    for key in ("snapshot", "since", "same_as"):
        if isinstance(check.get(key), str):
            names.append(check[key])

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("new_since"), str):
                names.append(value["new_since"])
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(check.get("select"))
    walk(check.get("except"))
    return names


# ----------------------------------------------------------------------
# Selectors
# ----------------------------------------------------------------------


def _literal(coords: Any) -> CanvasObject:
    if not _is_pair(coords):
        raise SelectorError(f"coords must be [x, y], got {coords!r}")
    return CanvasObject("", "Coords", "", {"coords": [float(coords[0]), float(coords[1])]})


def _is_pair(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )


def select(selector: Any, ctx: CheckContext, view: Optional[CanvasView] = None) -> list[CanvasObject]:
    """All objects in ``view`` (default: the current view) matching ``selector``."""
    view = view or ctx.view
    if isinstance(selector, str):
        if not selector.startswith("$") or selector[1:] not in ctx.bindings:
            raise SelectorError(f"unknown binding {selector!r}")
        bucket, name = ctx.bindings[selector[1:]]
        found = view.find(bucket, name)
        return [found] if found else []
    if not isinstance(selector, dict):
        raise SelectorError(f"bad selector {selector!r}")
    if "coords" in selector:
        return [_literal(selector["coords"])]
    obj_type = selector.get("type")
    if not isinstance(obj_type, str):
        raise SelectorError("selector needs a type")
    tol = ctx.tolerance.merged(selector.get("tol"))
    candidates = view.of_type(obj_type)
    if selector.get("only"):
        if len(candidates) != 1:
            raise SelectorError(f"expected exactly one {obj_type}, found {len(candidates)}")
    if "name" in selector:
        candidates = [obj for obj in candidates if obj.name == selector["name"]]
    if "new_since" in selector:
        before = ctx.snapshot(selector["new_since"])
        candidates = [obj for obj in candidates if before.find(obj.bucket, obj.name) is None]
    for key, matcher in _MATCHERS.items():
        if key in selector:
            candidates = [obj for obj in candidates if matcher(view, obj, selector[key], tol)]
    return candidates


def select_one(selector: Any, ctx: CheckContext, view: Optional[CanvasView] = None) -> CanvasObject:
    matches = select(selector, ctx, view)
    if len(matches) != 1:
        raise SelectorError(f"selector {_describe(selector)} matched {len(matches)} objects, expected 1")
    return matches[0]


def _describe(selector: Any) -> str:
    return selector if isinstance(selector, str) else json.dumps(selector, sort_keys=True)


def _match_at(view: CanvasView, obj: CanvasObject, at: Any, tol: Tolerance) -> bool:
    anchor = view.anchor(obj)
    return anchor is not None and _is_pair(at) and tol.close_coords(anchor, at)


def _match_ends(view: CanvasView, obj: CanvasObject, ends: Any, tol: Tolerance) -> bool:
    got = view.segment_ends(obj)
    if got is None or not isinstance(ends, list) or len(ends) != 2:
        return False
    a, b = ends
    if tol.close_coords(got[0], a) and tol.close_coords(got[1], b):
        return True
    return obj.type != "Vector" and tol.close_coords(got[0], b) and tol.close_coords(got[1], a)


def _match_origin(view: CanvasView, obj: CanvasObject, origin: Any, tol: Tolerance) -> bool:
    got = view.segment_ends(obj)
    return got is not None and tol.close_coords(got[0], origin)


def _match_tip(view: CanvasView, obj: CanvasObject, tip: Any, tol: Tolerance) -> bool:
    got = view.segment_ends(obj)
    return got is not None and tol.close_coords(got[1], tip)


def _match_contains(view: CanvasView, obj: CanvasObject, point: Any, tol: Tolerance) -> bool:
    ends = view.segment_ends(obj)
    return ends is not None and _is_pair(point) and tol.close(segment_distance(tuple(point), *ends), 0.0)


def _match_through(view: CanvasView, obj: CanvasObject, point: Any, tol: Tolerance) -> bool:
    ends = view.segment_ends(obj)
    return ends is not None and _is_pair(point) and tol.close(line_distance(tuple(point), *ends), 0.0)


def _match_vertices(view: CanvasView, obj: CanvasObject, vertices: Any, tol: Tolerance) -> bool:
    got = view.polygon_vertices(obj)
    if got is None or not isinstance(vertices, list) or len(got) != len(vertices):
        return False
    remaining = list(got)
    for wanted in vertices:
        match = next((v for v in remaining if tol.close_coords(v, wanted)), None)
        if match is None:
            return False
        remaining.remove(match)
    return True


def _match_center(view: CanvasView, obj: CanvasObject, center: Any, tol: Tolerance) -> bool:
    got = view.point(obj.args.get("center"))
    if got is None and obj.type == "CircleArc":
        arc = view.arc(obj)
        got = arc["center"] if arc else None
    return got is not None and tol.close_coords(got, center)


def _match_arg(key: str) -> Callable[[CanvasView, CanvasObject, Any, Tolerance], bool]:
    def matcher(view: CanvasView, obj: CanvasObject, value: Any, tol: Tolerance) -> bool:
        got = obj.args.get(key)
        return isinstance(got, (int, float)) and tol.close(float(got), float(value))

    return matcher


def _match_samples(view: CanvasView, obj: CanvasObject, samples: Any, tol: Tolerance) -> bool:
    if obj.type not in FUNCTION_TYPES:
        return False
    for x, y in samples:
        value = view.sample(obj, float(x))
        if value is None or not tol.close(value, float(y)):
            return False
    return True


def _match_slope(view: CanvasView, obj: CanvasObject, slope: Any, tol: Tolerance) -> bool:
    ends = view.segment_ends(obj)
    if ends is None:
        return False
    dx, dy = direction(*ends)
    return dx != 0 and tol.close(dy / dx, float(slope))


def _match_within(view: CanvasView, obj: CanvasObject, box: Any, tol: Tolerance) -> bool:
    """``within: [xmin, ymin, xmax, ymax]``: the object's position lies in the box."""
    anchor = view.anchor(obj)
    if anchor is None or not isinstance(box, list) or len(box) != 4:
        return False
    xmin, ymin, xmax, ymax = (float(v) for v in box)
    x, y = anchor
    return xmin - tol.abs <= x <= xmax + tol.abs and ymin - tol.abs <= y <= ymax + tol.abs


def _match_where(view: CanvasView, obj: CanvasObject, where: Any, tol: Tolerance) -> bool:
    if not isinstance(where, dict):
        raise SelectorError("where must be an object of path: value")
    base = dict(obj.data)
    base["inspect"] = obj.inspect or {}
    for path, expected in where.items():
        found, value = resolve_path(base, path)
        if not found or not values_equal(value, expected, tol):
            return False
    return True


_MATCHERS: dict[str, Callable[[CanvasView, CanvasObject, Any, Tolerance], bool]] = {
    "at": _match_at,
    "ends": _match_ends,
    "origin": _match_origin,
    "tip": _match_tip,
    "contains": _match_contains,
    "through": _match_through,
    "vertices": _match_vertices,
    "center": _match_center,
    "radius": _match_arg("radius"),
    "radius_x": _match_arg("radius_x"),
    "radius_y": _match_arg("radius_y"),
    "samples": _match_samples,
    "where": _match_where,
    "slope": _match_slope,
    "within": _match_within,
}


# ----------------------------------------------------------------------
# Paths and comparisons
# ----------------------------------------------------------------------

_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(-?\d+)\]")


def resolve_path(base: Any, path: str) -> tuple[bool, Any]:
    """Follow a dotted path with ``[i]`` indexes; returns (found, value)."""
    value = base
    if path in ("", "."):
        return True, value
    for key, index in _PATH_TOKEN.findall(path):
        if key:
            if not isinstance(value, dict) or key not in value:
                return False, None
            value = value[key]
        else:
            if not isinstance(value, list):
                return False, None
            i = int(index)
            if not -len(value) <= i < len(value):
                return False, None
            value = value[i]
    return True, value


def values_equal(actual: Any, expected: Any, tol: Tolerance) -> bool:
    """Deep equality with numeric tolerance."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected or actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return tol.close(float(actual), float(expected))
    if isinstance(expected, list) and isinstance(actual, (list, tuple)):
        return len(actual) == len(expected) and all(values_equal(a, e, tol) for a, e in zip(actual, expected))
    if isinstance(expected, dict) and isinstance(actual, dict):
        return set(actual) == set(expected) and all(values_equal(actual[k], expected[k], tol) for k in expected)
    return bool(actual == expected)


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numbers_in(value: Any) -> list[float]:
    """All numbers in a value: JSON numbers, or numbers written in strings."""
    if isinstance(value, str):
        return [float(n) for n in _NUMBER.findall(value)]
    found = [number for _, number in iter_numbers(value)]
    if isinstance(value, (dict, list)):
        for item in _strings_in(value):
            found.extend(float(n) for n in _NUMBER.findall(item))
    return found


def _strings_in(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for item in value.values() for s in _strings_in(item)]
    if isinstance(value, list):
        return [s for item in value for s in _strings_in(item)]
    return []


def compare(actual: Any, spec: dict[str, Any], tol: Tolerance) -> Optional[str]:
    """Apply every comparison op in ``spec``; returns a failure description or None."""
    for op in COMPARISON_OPS:
        if op not in spec:
            continue
        expected = spec[op]
        if not _apply_op(op, actual, expected, spec, tol):
            return f"{op} {json.dumps(expected) if not isinstance(expected, str) else repr(expected)}"
    return None


def _apply_op(op: str, actual: Any, expected: Any, spec: dict[str, Any], tol: Tolerance) -> bool:
    if op == "eq":
        modulus = spec.get("mod")
        if modulus and isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            diff = (float(actual) - float(expected)) % float(modulus)
            return tol.close(diff, 0.0) or tol.close(diff, float(modulus))
        return values_equal(actual, expected, tol)
    if op == "ne":
        return not values_equal(actual, expected, tol)
    if op in ("lt", "le", "gt", "ge"):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False
        a, e = float(actual), float(expected)
        if op == "lt":
            return a < e
        if op == "le":
            return a <= e or tol.close(a, e)
        if op == "gt":
            return a > e
        return a >= e or tol.close(a, e)
    if op == "contains":
        if isinstance(actual, str):
            return str(expected) in actual
        if isinstance(actual, (list, tuple)):
            return any(values_equal(item, expected, tol) for item in actual)
        if isinstance(actual, dict):
            return expected in actual
        return False
    if op == "not_contains":
        return not _apply_op("contains", actual, expected, spec, tol)
    if op == "set_eq":
        if not isinstance(actual, (list, tuple)) or not isinstance(expected, list):
            return False
        return sorted(map(str, actual)) == sorted(map(str, expected))
    if op == "in":
        return isinstance(expected, list) and any(values_equal(actual, item, tol) for item in expected)
    if op == "matches":
        text = actual if isinstance(actual, str) else json.dumps(actual, default=str)
        return re.search(str(expected), text) is not None
    if op == "has_numbers":
        found = numbers_in(actual)
        return all(any(tol.close(n, float(e)) for n in found) for e in expected)
    if op == "len":
        return hasattr(actual, "__len__") and len(actual) == expected
    if op == "is_null":
        return (actual is None) == bool(expected)
    if op == "not_null":
        return (actual is not None) == bool(expected)
    raise ValueError(f"unknown comparison {op}")


# ----------------------------------------------------------------------
# Check evaluation
# ----------------------------------------------------------------------


def evaluate_checks(checks: Sequence[dict[str, Any]], ctx: CheckContext, id_prefix: str) -> list[CheckResult]:
    """Evaluate ``checks`` in order (bindings from earlier checks are visible to later ones)."""
    results: list[CheckResult] = []
    for index, check in enumerate(checks, start=1):
        check_id = str(check.get("id") or f"{id_prefix}.c{index}")
        results.append(evaluate_check(check, ctx, check_id))
    return results


def evaluate_check(check: dict[str, Any], ctx: CheckContext, check_id: str) -> CheckResult:
    known = check.get("known")
    tol = ctx.tolerance.merged(check.get("tol"))
    local = CheckContext(ctx.view, ctx.step, ctx.snapshots, ctx.bindings, tol)
    if "bind" in check:
        kind, name = "bind", "bind"
    else:
        kind, name = "check", str(check.get("check"))
    result = CheckResult(check_id, kind, name, passed=True, known=known, spec=check)
    try:
        if kind == "bind":
            obj = select_one(check.get("select"), local)
            ctx.bindings[str(check["bind"])] = obj.key
            result.actual = obj.name
        elif name == "answer_mentions":
            _check_answer(check, local, result)
        else:
            outcome = _CHECKS[name](check, local)
            if outcome is not None:
                result.actual = outcome
    except MissingSample as exc:
        result.passed = None
        result.unrecorded = True
        result.message = f"needs a function sample that was not recorded: {exc}"
    except CheckFailure as exc:
        result.passed = False
        result.message = str(exc)
        result.expected, result.actual = exc.expected, exc.actual
    except SelectorError as exc:
        result.passed = False
        result.message = str(exc)
    except Exception as exc:  # a malformed check or an unexpected state shape
        result.passed = False
        result.error = True
        result.message = f"{type(exc).__name__}: {exc}"
    return result


def _count_compare(count: int, check: dict[str, Any], tol: Tolerance) -> None:
    failure = compare(count, check, tol)
    if failure:
        raise CheckFailure(f"count {count} does not satisfy {failure}", expected=failure, actual=count)


def _check_exists(check: dict[str, Any], ctx: CheckContext) -> Any:
    matches = select(check["select"], ctx)
    if not matches:
        raise CheckFailure(f"nothing matches {_describe(check['select'])}")
    return [obj.name for obj in matches]


def _check_absent(check: dict[str, Any], ctx: CheckContext) -> Any:
    matches = select(check["select"], ctx)
    if matches:
        raise CheckFailure(
            f"{len(matches)} object(s) match {_describe(check['select'])}", actual=[o.name for o in matches]
        )
    return None


def _check_count(check: dict[str, Any], ctx: CheckContext) -> Any:
    count = len(select(check["select"], ctx))
    _count_compare(count, check, ctx.tolerance)
    return count


def _anchor(ctx: CheckContext, obj: CanvasObject, view: Optional[CanvasView] = None) -> Coord:
    if obj.type == "Coords":
        return (obj.data["coords"][0], obj.data["coords"][1])
    anchor = (view or ctx.view).anchor(obj)
    if anchor is None:
        raise SelectorError(f"{obj.type} {obj.name} has no position")
    return anchor


def _check_point_at(check: dict[str, Any], ctx: CheckContext) -> Any:
    obj = select_one(check["select"], ctx)
    got = _anchor(ctx, obj)
    if not ctx.tolerance.close_coords(got, check["at"]):
        raise CheckFailure(f"{obj.name} is at {list(got)}", expected=check["at"], actual=list(got))
    return list(got)


def _check_moved(check: dict[str, Any], ctx: CheckContext) -> Any:
    obj = select_one(check["select"], ctx)
    before_view = ctx.snapshot(check["since"])
    before = before_view.find(obj.bucket, obj.name)
    if before is None:
        raise CheckFailure(f"{obj.name} did not exist at snapshot {check['since']}")
    start, end = _anchor(ctx, before, before_view), _anchor(ctx, obj)
    moved_by = [end[0] - start[0], end[1] - start[1]]
    if not ctx.tolerance.close_coords(moved_by, check["by"]):
        raise CheckFailure(f"{obj.name} moved by {moved_by}", expected=check["by"], actual=moved_by)
    return moved_by


def _attribute_base(check: dict[str, Any], ctx: CheckContext, view: CanvasView) -> Any:
    target = check.get("target")
    if target == "view":
        base: Any = dict(view.view_bounds)
        base["mode"] = view.coordinate_mode
        return base
    if target == "state":
        return view.state
    if target == "inspection":
        return view.inspection
    obj = select_one(check["select"], ctx, view)
    base = dict(obj.data)
    base["inspect"] = obj.inspect or {}
    base["name"] = obj.name
    return base


def _check_attribute(check: dict[str, Any], ctx: CheckContext) -> Any:
    path = str(check.get("path", ""))
    found, value = resolve_path(_attribute_base(check, ctx, ctx.view), path)
    if not found and not check.get("is_null"):
        raise CheckFailure(f"no value at {path!r}")
    if "same_as" in check:
        snapshot = ctx.snapshot(check["same_as"])
        found_then, then = resolve_path(_attribute_base(check, ctx, snapshot), path)
        if not found_then or not values_equal(value, then, ctx.tolerance):
            raise CheckFailure(
                f"{path} = {value!r}, at snapshot {check['same_as']} it was {then!r}", expected=then, actual=value
            )
    failure = compare(value, check, ctx.tolerance)
    if failure:
        raise CheckFailure(f"{path} = {value!r} does not satisfy {failure}", expected=failure, actual=value)
    return value


def _check_state_equals(check: dict[str, Any], ctx: CheckContext) -> Any:
    snapshot = ctx.snapshot(check["snapshot"])
    differences = diff_views(
        snapshot,
        ctx.view,
        ctx.tolerance,
        ignore=check.get("ignore") or [],
        inspect=bool(check.get("inspect")),
        include_view=bool(check.get("view")),
        match_names=check.get("match_names", True) is not False,
    )
    if differences:
        raise CheckFailure(
            f"state differs from snapshot {check['snapshot']}: " + "; ".join(differences[:8]),
            actual=differences[:20],
        )
    return None


def _check_unchanged_except(check: dict[str, Any], ctx: CheckContext) -> Any:
    before = ctx.snapshot(check["since"])
    excluded: set[tuple[str, str]] = set()
    for selector in check.get("except") or []:
        for view in (before, ctx.view):
            excluded.update(obj.key for obj in select(selector, ctx, view))
    differences = diff_views(before, ctx.view, ctx.tolerance, exclude=excluded)
    if differences:
        raise CheckFailure("unexpected changes: " + "; ".join(differences[:8]), actual=differences[:20])
    return None


def _executed(ctx: CheckContext, tool: str) -> list[dict[str, Any]]:
    return [c for c in ctx.step.counted_calls if c.get("function_name") == tool]


def _check_tool_called(check: dict[str, Any], ctx: CheckContext) -> Any:
    calls = _executed(ctx, check["tool"])
    if not calls:
        raise CheckFailure(f"{check['tool']} was not called")
    return len(calls)


def _check_tool_not_called(check: dict[str, Any], ctx: CheckContext) -> Any:
    calls = _executed(ctx, check["tool"])
    if calls:
        raise CheckFailure(f"{check['tool']} was called {len(calls)} time(s)")
    return None


def _check_max_tool_calls(check: dict[str, Any], ctx: CheckContext) -> Any:
    count = len(ctx.step.counted_calls)
    if count > int(check["max"]):
        raise CheckFailure(f"{count} tool calls, at most {check['max']} allowed", expected=check["max"], actual=count)
    return count


def _check_no_tool_errors(check: dict[str, Any], ctx: CheckContext) -> Any:
    errors = [f"{c.get('function_name')}: {str(c.get('result'))[:160]}" for c in ctx.step.calls if call_is_error(c)]
    if errors:
        raise CheckFailure("tool errors: " + "; ".join(errors), actual=errors)
    return None


def _tool_call(check: dict[str, Any], ctx: CheckContext) -> dict[str, Any]:
    calls = _executed(ctx, check["tool"])
    if not calls:
        raise CheckFailure(f"{check['tool']} was not called")
    index = int(check.get("index", -1))
    try:
        return calls[index]
    except IndexError:
        raise CheckFailure(f"{check['tool']} was called {len(calls)} time(s), no call #{index}") from None


def _check_tool_error(check: dict[str, Any], ctx: CheckContext) -> Any:
    call = _tool_call(check, ctx)
    if not call_is_error(call):
        raise CheckFailure(f"{check['tool']} succeeded: {str(call.get('result'))[:160]}", actual=call.get("result"))
    return call.get("result")


def _check_tool_result(check: dict[str, Any], ctx: CheckContext) -> Any:
    call = _tool_call(check, ctx)
    result = call.get("result")
    if isinstance(result, str) and result[:1] in "[{":
        try:
            result = json.loads(result)
        except ValueError:
            pass
    path = str(check.get("path", ""))
    found, value = resolve_path(result, path)
    if not found:
        raise CheckFailure(f"result has no {path!r}: {str(result)[:200]}", actual=result)
    failure = compare(value, check, ctx.tolerance)
    if failure:
        raise CheckFailure(f"result {path or 'value'} = {value!r} does not satisfy {failure}", actual=value)
    return value


def _check_answer(check: dict[str, Any], ctx: CheckContext, result: CheckResult) -> None:
    text = ctx.step.final_text
    if ctx.step.mode != "live" or text is None:
        result.passed = None
        result.message = "answer checks run in live mode only"
        return
    missing: list[str] = []
    for number in check.get("numbers", []):
        if not any(ctx.tolerance.merged(check.get("tol", 1e-2)).close(n, float(number)) for n in numbers_in(text)):
            missing.append(str(number))
    for word in check.get("names", []) + check.get("words", []):
        if str(word).lower() not in text.lower():
            missing.append(repr(word))
    if missing:
        result.passed = False
        result.message = "answer does not mention " + ", ".join(missing)


# ----------------------------------------------------------------------
# Relations
# ----------------------------------------------------------------------


def _check_relation(check: dict[str, Any], ctx: CheckContext) -> Any:
    relation = check["relation"]
    objects = [select_one(selector, ctx) for selector in check["select"]]
    return _RELATIONS[relation](objects, check, ctx)


def _ends(ctx: CheckContext, obj: CanvasObject) -> tuple[Coord, Coord]:
    ends = ctx.view.segment_ends(obj)
    if ends is None:
        raise SelectorError(f"{obj.type} {obj.name} has no endpoints")
    return ends


def _circle(ctx: CheckContext, obj: CanvasObject) -> tuple[Coord, float]:
    if obj.type == "CircleArc":
        arc = ctx.view.arc(obj)
        if arc is None:
            raise SelectorError(f"arc {obj.name} has no centre and radius")
        return arc["center"], arc["radius"]
    circle = ctx.view.circle(obj)
    if circle is None:
        raise SelectorError(f"{obj.type} {obj.name} is not a circle")
    return circle


def _polygon(ctx: CheckContext, obj: CanvasObject) -> list[Coord]:
    vertices = ctx.view.polygon_vertices(obj)
    if vertices is None:
        raise SelectorError(f"{obj.type} {obj.name} has no resolvable vertices")
    return vertices


def _points_of(ctx: CheckContext, obj: CanvasObject) -> list[Coord]:
    if obj.type in ("Segment", "Vector"):
        return list(_ends(ctx, obj))
    return [_anchor(ctx, obj)]


def _expect(ok: bool, message: str, expected: Any = None, actual: Any = None) -> None:
    if not ok:
        raise CheckFailure(message, expected=expected, actual=actual)


def _value_check(actual: float, check: dict[str, Any], ctx: CheckContext, label: str) -> float:
    expected = float(check["value"])
    _expect(ctx.tolerance.close(actual, expected), f"{label} is {actual}, expected {expected}", expected, actual)
    return actual


def _rel_point_on_circle(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    center, radius = _circle(ctx, objects[-1])
    offsets = [distance(_anchor(ctx, obj), center) - radius for obj in objects[:-1]]
    _expect(all(ctx.tolerance.close(d, 0.0) for d in offsets), f"distances from the circle: {offsets}", 0, offsets)
    return offsets


def _rel_point_on_segment(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    a, b = _ends(ctx, objects[-1])
    gaps = [segment_distance(_anchor(ctx, obj), a, b) for obj in objects[:-1]]
    _expect(all(ctx.tolerance.close(d, 0.0) for d in gaps), f"distances from the segment: {gaps}", 0, gaps)
    return gaps


def _rel_point_on_line(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    a, b = _ends(ctx, objects[-1])
    gaps = [line_distance(_anchor(ctx, obj), a, b) for obj in objects[:-1]]
    _expect(all(ctx.tolerance.close(d, 0.0) for d in gaps), f"distances from the line: {gaps}", 0, gaps)
    return gaps


def _function(ctx: CheckContext, obj: CanvasObject) -> CanvasObject:
    if obj.type not in FUNCTION_TYPES:
        raise SelectorError(f"{obj.type} {obj.name} is not a function")
    return obj


def _rel_point_on_function(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    function = _function(ctx, objects[-1])
    gaps = []
    for obj in objects[:-1]:
        x, y = _anchor(ctx, obj)
        value = ctx.view.sample(function, x)
        gaps.append(float("nan") if value is None else value - y)
    _expect(all(ctx.tolerance.close(g, 0.0) for g in gaps), f"f(x) - y at the points: {gaps}", 0, gaps)
    return gaps


def _rel_collinear(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    points: list[Coord] = [p for obj in objects for p in _points_of(ctx, obj)]
    base = points[0]
    other = next((p for p in points[1:] if distance(p, base) > ctx.tolerance.abs), None)
    if other is None:
        return 0.0
    gaps = [line_distance(p, base, other) for p in points]
    _expect(all(ctx.tolerance.close(g, 0.0) for g in gaps), f"distances from the common line: {gaps}", 0, gaps)
    return max(gaps)


def _line_directions(ctx: CheckContext, objects: list[CanvasObject]) -> list[Coord]:
    return [direction(*_ends(ctx, obj)) for obj in objects]


def _rel_parallel(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    u, v = _line_directions(ctx, objects[:2])
    sine = cross(u, v) / (math.hypot(*u) * math.hypot(*v))
    _expect(ctx.tolerance.close(sine, 0.0), f"sine of the angle between the lines is {sine}", 0, sine)
    return sine


def _rel_perpendicular(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    u, v = _line_directions(ctx, objects[:2])
    cosine = dot(u, v) / (math.hypot(*u) * math.hypot(*v))
    _expect(ctx.tolerance.close(cosine, 0.0), f"cosine of the angle between the lines is {cosine}", 0, cosine)
    return cosine


def _rel_midpoint_of(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    point = _anchor(ctx, objects[0])
    a, b = _ends(ctx, objects[1])
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    _expect(
        ctx.tolerance.close_coords(point, mid), f"point {list(point)}, midpoint {list(mid)}", list(mid), list(point)
    )
    return list(point)


def _rel_equal_length(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    lengths = [distance(*_ends(ctx, obj)) for obj in objects]
    _expect(all(ctx.tolerance.close(length, lengths[0]) for length in lengths), f"lengths {lengths}", None, lengths)
    return lengths


def _line_angle(u: Coord, v: Coord) -> float:
    """Angle between two lines (not rays), 0 to 90 degrees."""
    angle = unit_angle_between(u, v)
    return min(angle, 180 - angle)


def _rel_equal_angles(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    line, first, second = _line_directions(ctx, objects[:3])
    angles = [_line_angle(line, first), _line_angle(line, second)]
    _expect(ctx.tolerance.close(angles[0], angles[1]), f"angles with the two lines: {angles}", None, angles)
    return angles


def _rel_tangent_to(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    a, b = _ends(ctx, objects[0])
    target = objects[1]
    if target.type == "Circle":
        center, radius = _circle(ctx, target)
        gap = line_distance(center, a, b) - radius
        _expect(ctx.tolerance.close(gap, 0.0), f"distance from centre minus radius is {gap}", 0, gap)
        return gap
    function = _function(ctx, target)
    x = float(check["x"])
    h = float(check.get("h", 1e-4))
    y0, y1, y2 = (ctx.view.sample(function, value) for value in (x, x - h, x + h))
    if y0 is None or y1 is None or y2 is None:
        raise CheckFailure(f"{function.name} is undefined near x = {x}")
    slope = (y2 - y1) / (2 * h)
    u = direction(a, b)
    _expect(ctx.tolerance.close(line_distance((x, y0), a, b), 0.0), f"the line misses ({x}, {y0})")
    line_slope = u[1] / u[0] if u[0] else float("inf")
    _expect(
        ctx.tolerance.merged(check.get("slope_tol", 1e-4)).close(line_slope, slope),
        f"line slope {line_slope}, derivative {slope}",
        slope,
        line_slope,
    )
    return line_slope


def _rel_distance(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    a, b = objects[:2]
    if b.type in ("Segment", "Vector"):
        return _value_check(line_distance(_anchor(ctx, a), *_ends(ctx, b)), check, ctx, "distance")
    return _value_check(distance(_anchor(ctx, a), _anchor(ctx, b)), check, ctx, "distance")


def _rel_length(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    return _value_check(distance(*_ends(ctx, objects[0])), check, ctx, "length")


def _rel_angle_deg(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    if len(objects) == 3:
        p1, vertex, p2 = (_anchor(ctx, obj) for obj in objects)
        raw = ccw_angle_degrees(vertex, p1, p2)
        value = min(raw, 360 - raw)
    elif len(objects) == 1 and objects[0].type == "Angle":
        angle = ctx.view.angle_degrees(objects[0])
        if angle is None:
            raise SelectorError(f"angle {objects[0].name} has no resolvable arms")
        value = angle
    else:
        u, v = _line_directions(ctx, objects[:2])
        value = unit_angle_between(u, v)
    return _value_check(value, check, ctx, "angle")


def _rel_area(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    obj = objects[0]
    if obj.type == "Circle":
        area = math.pi * _circle(ctx, obj)[1] ** 2
    else:
        area = polygon_area(_polygon(ctx, obj))
    return _value_check(area, check, ctx, "area")


def _rel_function_value(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    function = _function(ctx, objects[0])
    pairs = check.get("values") or [[check["x"], check["y"]]]
    got = []
    for x, y in pairs:
        value = ctx.view.sample(function, float(x))
        got.append([x, value])
        _expect(
            value is not None and ctx.tolerance.close(value, float(y)),
            f"{function.name}({x}) = {value}, expected {y}",
            y,
            value,
        )
    return got


def _rel_inside(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    point = _anchor(ctx, objects[0])
    container = objects[1]
    if container.type == "Circle":
        center, radius = _circle(ctx, container)
        inside = distance(point, center) < radius
    else:
        inside = point_in_polygon(point, _polygon(ctx, container))
    _expect(inside, f"{list(point)} is not inside {container.name}")
    return list(point)


def _rel_slope(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    u = direction(*_ends(ctx, objects[0]))
    slope = u[1] / u[0] if u[0] else float("inf")
    return _value_check(slope, check, ctx, "slope")


def _rel_direction(objects: list[CanvasObject], check: dict[str, Any], ctx: CheckContext) -> Any:
    u = direction(*_ends(ctx, objects[0]))
    norm = math.hypot(*u)
    if "parallel_to" in check:
        v = check["parallel_to"]
        value = cross(u, v) / (norm * math.hypot(*v))
        _expect(ctx.tolerance.close(value, 0.0), f"direction {list(u)} is not parallel to {v}", v, list(u))
    if "perpendicular_to" in check:
        v = check["perpendicular_to"]
        value = dot(u, v) / (norm * math.hypot(*v))
        _expect(ctx.tolerance.close(value, 0.0), f"direction {list(u)} is not perpendicular to {v}", v, list(u))
    return list(u)


_RELATIONS: dict[str, Callable[[list[CanvasObject], dict[str, Any], CheckContext], Any]] = {
    "point_on_circle": _rel_point_on_circle,
    "point_on_segment": _rel_point_on_segment,
    "point_on_line": _rel_point_on_line,
    "point_on_function": _rel_point_on_function,
    "collinear": _rel_collinear,
    "parallel": _rel_parallel,
    "perpendicular": _rel_perpendicular,
    "midpoint_of": _rel_midpoint_of,
    "equal_length": _rel_equal_length,
    "equal_angles": _rel_equal_angles,
    "tangent_to": _rel_tangent_to,
    "distance": _rel_distance,
    "length": _rel_length,
    "angle_deg": _rel_angle_deg,
    "area": _rel_area,
    "function_value": _rel_function_value,
    "inside": _rel_inside,
    "slope": _rel_slope,
    "direction": _rel_direction,
}

_CHECKS: dict[str, Callable[[dict[str, Any], CheckContext], Any]] = {
    "exists": _check_exists,
    "absent": _check_absent,
    "count": _check_count,
    "point_at": _check_point_at,
    "moved": _check_moved,
    "relation": _check_relation,
    "attribute": _check_attribute,
    "state_equals": _check_state_equals,
    "unchanged_except": _check_unchanged_except,
    "tool_called": _check_tool_called,
    "tool_not_called": _check_tool_not_called,
    "max_tool_calls": _check_max_tool_calls,
    "no_tool_errors": _check_no_tool_errors,
    "tool_error": _check_tool_error,
    "tool_result": _check_tool_result,
}


# ----------------------------------------------------------------------
# State comparison
# ----------------------------------------------------------------------

# Inspection fields compared by state_equals(inspect=true).
_INSPECTION_FIELDS = ("color", "label", "is_reflex")


def diff_views(
    before: CanvasView,
    after: CanvasView,
    tol: Tolerance,
    *,
    ignore: Sequence[str] = (),
    exclude: Optional[set[tuple[str, str]]] = None,
    inspect: bool = False,
    include_view: bool = False,
    match_names: bool = True,
    include_mode: bool = True,
) -> list[str]:
    """Human-readable differences between two views' drawables (empty when equal).

    ``ignore`` lists buckets or types to skip; ``exclude`` lists ``(bucket, name)``
    keys. With ``inspect`` the inspection view's colours, labels, grid
    visibility and coordinate mode are compared too; with ``include_view`` the
    visible bounds are. With ``match_names=False`` objects are compared by
    geometry only.
    """
    skipped = set(ignore)
    excluded = exclude or set()

    def keep(obj: CanvasObject) -> bool:
        return obj.bucket not in skipped and obj.type not in skipped and obj.key not in excluded

    differences: list[str] = []
    if match_names:
        before_map = {obj.key: obj for obj in before.objects if keep(obj)}
        after_map = {obj.key: obj for obj in after.objects if keep(obj)}
        for key in sorted(set(before_map) - set(after_map)):
            differences.append(f"{key[0]} {key[1]} removed")
        for key in sorted(set(after_map) - set(before_map)):
            differences.append(f"{key[0]} {key[1]} added")
        for key in sorted(set(before_map) & set(after_map)):
            old, new = before_map[key], after_map[key]
            if not values_equal(_comparable(new.data), _comparable(old.data), tol):
                differences.append(f"{key[0]} {key[1]} changed")
            elif inspect and _inspect_fields(old) != _inspect_fields(new):
                differences.append(
                    f"{key[0]} {key[1]} inspection changed: {_inspect_fields(old)} -> {_inspect_fields(new)}"
                )
    else:
        old_sigs = sorted(repr(before.signature(obj)) for obj in before.objects if keep(obj))
        new_sigs = sorted(repr(after.signature(obj)) for obj in after.objects if keep(obj))
        for sig in sorted(set(old_sigs) - set(new_sigs)):
            differences.append(f"missing {sig}")
        for sig in sorted(set(new_sigs) - set(old_sigs)):
            differences.append(f"extra {sig}")
    if include_mode and (before.coordinate_mode or "cartesian") != (after.coordinate_mode or "cartesian"):
        differences.append(f"coordinate mode {before.coordinate_mode} -> {after.coordinate_mode}")
    if include_view and not values_equal(after.view_bounds, before.view_bounds, tol):
        differences.append(f"view {before.view_bounds} -> {after.view_bounds}")
    if inspect:
        old_grid = before.inspection.get("grid_visible")
        new_grid = after.inspection.get("grid_visible")
        if old_grid != new_grid:
            differences.append(f"grid visibility {old_grid} -> {new_grid}")
    return differences


def _comparable(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k != "name"}


def _inspect_fields(obj: CanvasObject) -> dict[str, Any]:
    record = obj.inspect or {}
    return {key: record.get(key) for key in _INSPECTION_FIELDS if key in record}


# ----------------------------------------------------------------------
# Invariants
# ----------------------------------------------------------------------


def run_invariants(
    before: CanvasView,
    after: CanvasView,
    step: StepData,
    step_id: str,
    waivers: Optional[dict[str, str]] = None,
    allow_large: bool = False,
) -> list[CheckResult]:
    """Evaluate I1 to I7 for one step that executed tool calls."""
    waivers = waivers or {}
    checks: list[tuple[str, Callable[[], tuple[list[str], list[str]]]]] = [
        ("I1", lambda: _inv_unique_names(after)),
        ("I2", lambda: (_inv_references(after), [])),
        ("I3", lambda: (_inv_derived(after), [])),
        ("I4", lambda: (_inv_truthful_results(before, after, step), [])),
        ("I5", lambda: (_inv_undo_accounting(before, after, step), [])),
        ("I6", lambda: (_inv_errors_flagged(step), [])),
        ("I7", lambda: (_inv_sane_numbers(after, allow_large), [])),
    ]
    results = []
    for invariant_id, run in checks:
        result = CheckResult(
            f"{step_id}.{invariant_id}", "invariant", invariant_id, True, known=waivers.get(invariant_id)
        )
        try:
            problems, warnings = run()
        except Exception as exc:
            problems, warnings = [f"{type(exc).__name__}: {exc}"], []
            result.error = True
        if problems:
            result.passed = False
            result.message = "; ".join(problems[:10])
            result.actual = problems[:30]
        elif warnings:
            result.passed = False
            result.warning = True
            result.known = None
            result.message = "; ".join(warnings[:10])
        results.append(result)
    return results


def _inv_unique_names(view: CanvasView) -> tuple[list[str], list[str]]:
    """I1: names are unique within a bucket; the same name in two buckets is a warning."""
    problems: list[str] = []
    seen: dict[str, set[str]] = {}
    per_bucket: dict[tuple[str, str], int] = {}
    for obj in view.objects:
        per_bucket[obj.key] = per_bucket.get(obj.key, 0) + 1
        seen.setdefault(obj.name, set()).add(obj.bucket)
    for (bucket, name), count in sorted(per_bucket.items()):
        if count > 1:
            problems.append(f"{count} {bucket} named {name}")
    warnings = [
        f"{name} is used by {', '.join(sorted(buckets))}" for name, buckets in sorted(seen.items()) if len(buckets) > 1
    ]
    return problems, warnings


_REFERENCE_AXES = frozenset({"x_axis", "y_axis"})


def _inv_references(view: CanvasView) -> list[str]:
    """I2: every referenced point, segment, function, circle or area exists."""
    problems: list[str] = []
    all_names = {obj.name for obj in view.objects}

    def need(owner: CanvasObject, role: str, name: Any, obj_type: Optional[str] = None) -> None:
        if name in (None, "") or name in _REFERENCE_AXES or isinstance(name, (int, float)):
            return
        exists = view.find_by_type_and_name(obj_type, str(name)) if obj_type else (str(name) in all_names)
        if not exists:
            problems.append(f"{owner.type} {owner.name}: {role} {name} does not exist")

    for obj in view.objects:
        args = obj.args
        if obj.type == "Segment":
            need(obj, "p1", args.get("p1"), "Point")
            need(obj, "p2", args.get("p2"), "Point")
        elif obj.type == "Vector":
            need(obj, "origin", args.get("origin"), "Point")
            need(obj, "tip", args.get("tip"), "Point")
        elif obj.type in POLYGON_TYPES:
            for index, name in enumerate(view.polygon_vertex_names(obj), start=1):
                need(obj, f"p{index}", name, "Point")
        elif obj.type in ("Circle", "Ellipse"):
            need(obj, "center", args.get("center"), "Point")
        elif obj.type == "CircleArc":
            need(obj, "point1", args.get("point1_name"), "Point")
            need(obj, "point2", args.get("point2_name"), "Point")
            need(obj, "circle", args.get("circle_name"), "Circle")
        elif obj.type == "Angle":
            need(obj, "segment1", args.get("segment1_name"), "Segment")
            need(obj, "segment2", args.get("segment2_name"), "Segment")
        elif obj.type in COLORED_AREA_TYPES:
            for key in ("func1", "func2", "segment1", "segment2", "segment", "function", "circle", "ellipse"):
                need(obj, key, args.get(key))
            for name in args.get("segments") or []:
                need(obj, "segment", name, "Segment")
        elif obj.type in GRAPH_TYPES:
            for key, obj_type in (("segments", "Segment"), ("vectors", "Vector"), ("isolated_points", "Point")):
                for name in args.get(key) or []:
                    need(obj, key[:-1], name, obj_type)
        elif obj.type in PLOT_TYPES:
            need(obj, "function", args.get("function_name"))
            need(obj, "fill area", args.get("fill_area_name"))
    return problems


_CIRCLE_FORMULA = re.compile(
    r"^\(x - (?P<h>\(?-?[\d.eE+-]+\)?)\)\*\*2 \+ \(y - (?P<k>\(?-?[\d.eE+-]+\)?)\)\*\*2 = (?P<r>[\d.eE+-]+)\*\*2$"
)
_ELLIPSE_FORMULA = re.compile(
    r"^(?P<a>-?[\d.eE+-]+)\*\(x - (?P<h>-?\s*-?[\d.eE+-]+)\)\*\*2 (?P<bs>[+-]) (?P<b>[\d.eE+-]+)\*\(x - -?[\d.eE+-]+\)"
    r"\*\(y - (?P<k>-?\s*-?[\d.eE+-]+)\) \+ (?P<c>-?[\d.eE+-]+)\*\(y - -?[\d.eE+-]+\)\*\*2 = 1$"
)
_FORMULA_TOL = Tolerance(1e-8, 1e-8)
_DERIVED_TOL = Tolerance(1e-9, 1e-9)
_TYPE_FLAGS = ("equilateral", "isosceles", "scalene", "right")


def _number(text: str) -> float:
    return float(text.replace("(", "").replace(")", "").replace(" ", ""))


def _inv_derived(view: CanvasView) -> list[str]:
    """I3: cached derived fields agree with the geometry."""
    problems: list[str] = []
    for obj in view.objects:
        if obj.type in ("Segment", "Vector"):
            problems.extend(_derived_endpoints(view, obj))
        elif obj.type == "Circle":
            problems.extend(_derived_circle_formula(view, obj))
        elif obj.type == "Ellipse":
            problems.extend(_derived_ellipse_formula(view, obj))
        elif obj.type == "Triangle":
            problems.extend(_derived_triangle_types(view, obj))
        elif obj.type == "Angle":
            problems.extend(_derived_angle(view, obj))
        elif obj.type == "CircleArc":
            problems.extend(_derived_arc(view, obj))
        elif obj.type in FUNCTION_TYPES:
            problems.extend(_derived_function(obj))
    return problems


def _derived_endpoints(view: CanvasView, obj: CanvasObject) -> list[str]:
    keys = ("_origin_coords", "_tip_coords") if obj.type == "Vector" else ("_p1_coords", "_p2_coords")
    names = (
        (obj.args.get("origin"), obj.args.get("tip"))
        if obj.type == "Vector"
        else (obj.args.get("p1"), obj.args.get("p2"))
    )
    problems = []
    for key, name in zip(keys, names):
        cached, actual = obj.data.get(key), view.point(name)
        if cached is not None and actual is not None and not _DERIVED_TOL.close_coords(cached, actual):
            problems.append(f"{obj.type} {obj.name}: {key} {cached} but {name} is at {list(actual)}")
    return problems


def _derived_circle_formula(view: CanvasView, obj: CanvasObject) -> list[str]:
    formula = obj.args.get("circle_formula")
    circle = view.circle(obj)
    if not isinstance(formula, str) or circle is None:
        return []
    match = _CIRCLE_FORMULA.match(formula.strip())
    if not match:
        return []
    (cx, cy), radius = circle
    stated = (_number(match["h"]), _number(match["k"]), _number(match["r"]))
    if not (
        _FORMULA_TOL.close(stated[0], cx)
        and _FORMULA_TOL.close(stated[1], cy)
        and _FORMULA_TOL.close(stated[2], radius)
    ):
        return [f"Circle {obj.name}: formula {formula!r} but centre ({cx}, {cy}), radius {radius}"]
    return []


def _derived_ellipse_formula(view: CanvasView, obj: CanvasObject) -> list[str]:
    formula = obj.args.get("ellipse_formula")
    ellipse = view.ellipse(obj)
    if not isinstance(formula, str) or ellipse is None:
        return []
    match = _ELLIPSE_FORMULA.match(formula.strip())
    if not match:
        return []
    (cx, cy), rx, ry, rotation = ellipse
    theta = math.radians(rotation)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    expected = (
        cos_t**2 / rx**2 + sin_t**2 / ry**2,
        2 * cos_t * sin_t * (1 / rx**2 - 1 / ry**2),
        sin_t**2 / rx**2 + cos_t**2 / ry**2,
    )
    b = float(match["b"]) * (-1 if match["bs"] == "-" else 1)
    stated = (float(match["a"]), b, float(match["c"]))
    centre = (_number(match["h"]), _number(match["k"]))
    tol = Tolerance(1e-8, 1e-6)
    if not all(tol.close(s, e) for s, e in zip(stated, expected)) or not tol.close_coords(centre, (cx, cy)):
        return [f"Ellipse {obj.name}: formula {formula!r} does not match centre, radii and rotation {rotation}"]
    return []


def _derived_triangle_types(view: CanvasView, obj: CanvasObject) -> list[str]:
    types = obj.data.get("types")
    vertices = view.polygon_vertices(obj)
    if not isinstance(types, list) or vertices is None or len(vertices) != 3:
        return []
    computed = triangle_flags(vertices)
    wrong = [flag for flag in _TYPE_FLAGS if (flag in types) != (flag in computed)]
    if wrong:
        return [f"Triangle {obj.name}: types {types} disagree with its geometry on {', '.join(wrong)}"]
    return []


def _derived_angle(view: CanvasView, obj: CanvasObject) -> list[str]:
    cached = (obj.inspect or {}).get("angle_degrees")
    actual = view.angle_degrees(obj)
    if not isinstance(cached, (int, float)) or actual is None:
        return []
    if not Tolerance(1e-6, 1e-9).close(float(cached), actual):
        return [f"Angle {obj.name}: cached {cached} degrees, geometry gives {actual}"]
    return []


def _derived_arc(view: CanvasView, obj: CanvasObject) -> list[str]:
    arc = view.arc(obj)
    if arc is None:
        return []
    problems = []
    for key in ("p1", "p2"):
        point = arc[key]
        if point is not None and not Tolerance(1e-7, 1e-9).close(distance(point, arc["center"]), arc["radius"]):
            problems.append(
                f"CircleArc {obj.name}: endpoint {list(point)} is {distance(point, arc['center'])} from the centre, "
                f"radius {arc['radius']}"
            )
    circle_name = obj.args.get("circle_name")
    circle_obj = view.find("Circles", str(circle_name)) if circle_name else None
    circle = view.circle(circle_obj) if circle_obj is not None else None
    if circle is not None and not (
        _DERIVED_TOL.close_coords(circle[0], arc["center"]) and _DERIVED_TOL.close(circle[1], arc["radius"])
    ):
        problems.append(f"CircleArc {obj.name}: centre and radius disagree with its circle {circle_name}")
    return problems


def _blows_up(value: Any) -> bool:
    return (
        value is None
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or abs(value) > ASYMPTOTE_BLOWUP
    )


def _derived_function(obj: CanvasObject) -> list[str]:
    problems = []
    left, right = obj.args.get("left_bound"), obj.args.get("right_bound")
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not left < right:
        problems.append(f"{obj.type} {obj.name}: left_bound {left} is not below right_bound {right}")
    for probe in (obj.inspect or {}).get("asymptote_probes", []) or []:
        if isinstance(probe, list) and len(probe) == 3 and not (_blows_up(probe[1]) or _blows_up(probe[2])):
            problems.append(
                f"{obj.type} {obj.name}: listed vertical asymptote x = {probe[0]} but f is {probe[1]} and {probe[2]} beside it"
            )
    return problems


def _is_mutating(tool: str) -> bool:
    if tool in _NON_CANVAS_TOOLS:
        return False
    return tool in _MUTATING_TOOLS or tool.startswith(_MUTATING_PREFIXES)


def is_generic_success(result: Any) -> bool:
    """True for results that say only "it worked" (the generic message, True, or nothing).

    A tool that changed nothing but explains why (e.g. "Point 'C' already exists")
    told the truth; only a bare success claim is suspect.
    """
    if result is None or result is True:
        return True
    return isinstance(result, str) and result.strip() in GENERIC_SUCCESS_MESSAGES


def _removes_objects(tool: str) -> bool:
    return tool.startswith("delete_") or tool in ("clear_canvas", "load_workspace", "undo", "redo")


def _result_mentions(result: Any, names: set[str]) -> bool:
    text = json.dumps(result) if not isinstance(result, str) else result
    return any(re.search(rf"(?<![\w']){re.escape(name)}(?![\w'])", text) for name in names)


def _inv_truthful_results(before: CanvasView, after: CanvasView, step: StepData) -> list[str]:
    """I4: tool results match what happened to the canvas."""
    calls = step.counted_calls
    if not calls:
        return []
    problems: list[str] = []
    changed = diff_views(before, after, _DERIVED_TOL, inspect=True, include_view=True)
    drawables_changed = diff_views(before, after, _DERIVED_TOL)
    errors = [call_is_error(call) for call in calls]
    if all(errors) and drawables_changed:
        problems.append("every call failed but the canvas changed: " + "; ".join(drawables_changed[:5]))
    if (
        not any(errors)
        and all(_is_mutating(str(c.get("function_name"))) for c in calls)
        and all(is_generic_success(c.get("result")) for c in calls)
        and not changed
    ):
        names = ", ".join(str(c.get("function_name")) for c in calls)
        problems.append(f"{names} reported success but changed nothing")
    before_keys = {obj.key for obj in before.objects}
    added = {obj.name for obj in after.objects if obj.key not in before_keys}
    touched = added | {
        obj.name
        for obj in after.objects
        if (old := before.find(obj.bucket, obj.name)) is not None and old.data != obj.data
    }
    if any(_removes_objects(str(c.get("function_name"))) for c in calls):
        # An object created and then removed within the batch cannot be traced by names.
        return problems
    for call, failed in zip(calls, errors):
        tool = str(call.get("function_name"))
        if failed or not tool.startswith(_NAMING_PREFIXES):
            continue
        arguments = call.get("arguments") or {}
        for key in _NAME_HINT_KEYS:
            hint = arguments.get(key)
            explained = _result_mentions(call.get("result"), added) or (
                not added and not is_generic_success(call.get("result"))
            )
            if isinstance(hint, str) and hint and hint not in touched and not explained:
                problems.append(
                    f"{tool} asked for {key} {hint!r}, no object got it, and the result "
                    f"({str(call.get('result'))[:80]!r}) does not name what was created ({', '.join(sorted(added)) or 'nothing'})"
                )
    return problems


def _inv_undo_accounting(before: CanvasView, after: CanvasView, step: StepData) -> list[str]:
    """I5: one undo entry per batch that changed the canvas; none for a batch that changed nothing.

    Judged by what happened, not by what the calls reported: a failed call, a
    refused call and a truthful no-op (e.g. creating a point that already
    exists) must all leave the undo stack alone. "Changed" is judged as in I4:
    drawables, the view, the coordinate mode and inspection-only fields
    (colour, label, grid visibility) all count. Undo and redo batches are
    simulated against the stack depths instead.
    """
    if step.undo_before is None or step.undo_after is None:
        return []
    calls = step.counted_calls
    if not calls:
        return []
    tools = [str(c.get("function_name")) for c in calls]
    added = step.undo_after - step.undo_before
    if any(t in ("undo", "redo") for t in tools):
        if not all(t in ("undo", "redo") for t in tools) or step.redo_before is None:
            return []
        undo, redo = step.undo_before, step.redo_before
        for tool in tools:
            if tool == "undo" and undo > 0:
                undo, redo = undo - 1, redo + 1
            elif tool == "redo" and redo > 0:
                undo, redo = undo + 1, redo - 1
        if step.undo_after != undo:
            return [f"undo depth went from {step.undo_before} to {step.undo_after}, expected {undo}"]
        return []
    changed = diff_views(before, after, _DERIVED_TOL, inspect=True, include_view=True)
    expected = 1 if changed else 0
    if added != expected:
        what = "changed the canvas" if changed else "changed nothing"
        return [f"batch ({', '.join(tools)}) {what} and added {added} undo entries, expected {expected}"]
    return []


def _inv_errors_flagged(step: StepData) -> list[str]:
    """I6: a result shaped like an error is flagged as one."""
    problems = []
    for call in step.calls:
        result = call.get("result")
        if (
            isinstance(result, dict)
            and ("error" in result or result.get("status") == "error")
            and not call.get("is_error")
        ):
            problems.append(f"{call.get('function_name')} returned {str(result)[:120]} without is_error")
    return problems


def _inv_sane_numbers(view: CanvasView, allow_large: bool) -> list[str]:
    """I7: no NaN, no infinity and no runaway coordinates."""
    problems = []
    for obj in view.objects:
        for path, value in iter_numbers(obj.data):
            if not math.isfinite(value):
                problems.append(f"{obj.type} {obj.name}: {path} is {value}")
            elif not allow_large and abs(value) > MAX_SANE_MAGNITUDE:
                problems.append(f"{obj.type} {obj.name}: {path} = {value:g}")
    for path, value in iter_numbers(view.view_bounds):
        if not math.isfinite(value):
            problems.append(f"view {path} is {value}")
    return problems
