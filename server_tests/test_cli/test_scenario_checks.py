"""Tests for the scenario check engine (cli/scenarios/geometry.py and checks.py), on fixture states."""

from __future__ import annotations

import math
from typing import Any, Optional

import pytest

from cli.scenarios.checks import (
    CheckContext,
    CheckResult,
    StepData,
    call_is_error,
    compare,
    diff_views,
    evaluate_check,
    evaluate_checks,
    numbers_in,
    resolve_path,
    run_invariants,
    select,
    validate_check,
)
from cli.scenarios.geometry import CanvasView, SampleRequests, Tolerance, triangle_flags

from server_tests.test_cli.scenario_states import (
    angle,
    circle,
    function,
    inspection,
    point,
    points,
    right_triangle,
    segment,
    state,
    triangle,
    vector,
)


def ctx_for(view: CanvasView, **kwargs: Any) -> CheckContext:
    return CheckContext(view=view, **kwargs)


def run(check: dict[str, Any], view: CanvasView, **kwargs: Any) -> CheckResult:
    return evaluate_check(check, ctx_for(view, **kwargs), "t1.c1")


def call(tool: str, result: Any = "Call successful!", is_error: bool = False, **arguments: Any) -> dict[str, Any]:
    return {"function_name": tool, "arguments": arguments, "result": result, "is_error": is_error}


def by_id(results: list[CheckResult], invariant: str) -> CheckResult:
    return next(r for r in results if r.name == invariant)


# ----------------------------------------------------------------------
# Tolerance and geometry view
# ----------------------------------------------------------------------


class TestTolerance:
    def test_absolute_and_relative(self) -> None:
        tol = Tolerance(1e-9, 1e-9)
        assert tol.close(1.0, 1.0 + 5e-10)
        assert not tol.close(0.0, 1e-8)
        assert tol.close(1e9, 1e9 + 0.5)
        assert not tol.close(float("nan"), float("nan"))
        assert tol.close(float("inf"), float("inf"))

    def test_merged(self) -> None:
        base = Tolerance()
        assert base.merged(None) is base
        assert base.merged(1e-3) == Tolerance(1e-3, 1e-3)
        assert base.merged({"abs": 0}) == Tolerance(0.0, 1e-9)
        with pytest.raises(ValueError):
            base.merged("loose")


class TestCanvasView:
    def test_objects_and_points(self) -> None:
        view = CanvasView(state(*right_triangle()))
        assert [obj.type for obj in view.objects].count("Point") == 3
        assert view.point("B") == (4.0, 0.0)
        assert view.of_type("Polygon")[0].name == "ABC"
        assert view.coordinate_mode == "cartesian"

    def test_non_drawable_keys_are_skipped(self) -> None:
        raw = state(point("A", 1, 1))
        raw["current_tick_spacing"] = 100
        raw["computations"] = [{"expression": "1+1", "result": 2}]
        assert [obj.name for obj in CanvasView(raw).objects] == ["A"]

    def test_segment_ends_fall_back_to_cached_coords(self) -> None:
        view = CanvasView(state(segment("A", "B", (1, 2), (3, 4))))
        assert view.segment_ends(view.objects[0]) == ((1.0, 2.0), (3.0, 4.0))

    def test_polygon_vertices_from_names_or_inspection(self) -> None:
        raw = state(*right_triangle())
        view = CanvasView(raw)
        assert sorted(view.polygon_vertices(view.of_type("Triangle")[0]) or []) == [(0, 0), (0, 3), (4, 0)]
        inspected = CanvasView(
            raw, inspection({"class": "Triangle", "name": "ABC", "vertices": [[9, 9], [10, 9], [9, 10]]})
        )
        assert sorted(inspected.polygon_vertices(inspected.of_type("Triangle")[0]) or []) == [(9, 9), (9, 10), (10, 9)]

    def test_angle_degrees_and_reflex(self) -> None:
        raw = state(*right_triangle(), angle("angle_BAC", "AB", "CA"))
        view = CanvasView(raw)
        assert view.angle_degrees(view.of_type("Angle")[0]) == pytest.approx(90)
        reflex = CanvasView(state(*right_triangle(), angle("r", "AB", "BC", is_reflex=True)))
        assert reflex.angle_degrees(reflex.of_type("Angle")[0]) == pytest.approx(360 - math.degrees(math.atan2(3, 4)))

    def test_missing_sample_is_recorded(self) -> None:
        view = CanvasView(state(function("f", "x^2")))
        with pytest.raises(Exception):
            view.sample(view.objects[0], 2.0)
        assert view.requests.x == {"f": {2.0}}
        assert view.requests.as_options() == {"samples": {"f": [2.0]}}

    def test_sample_requests_merge(self) -> None:
        first, second = SampleRequests(), SampleRequests()
        first.add_x("f", 1)
        second.add_x("f", 2)
        second.add_t("p", 0)
        first.merge(second)
        assert first.as_options() == {"samples": {"f": [1.0, 2.0]}, "t_samples": {"p": [0.0]}}

    def test_triangle_flags(self) -> None:
        assert triangle_flags([(0, 0), (4, 0), (0, 3)]) >= {"scalene", "right"}
        assert triangle_flags([(0, 0), (2, 0), (1, math.sqrt(3))]) >= {"equilateral", "isosceles", "acute"}
        assert "obtuse" in triangle_flags([(0, 0), (10, 0), (1, 1)])


# ----------------------------------------------------------------------
# Selectors
# ----------------------------------------------------------------------


class TestSelectors:
    def setup_method(self) -> None:
        raw = state(
            *right_triangle(),
            point("D", 4, 4),
            vector("A", "D", (0, 0), (4, 4)),
            circle("A", 5, 0, 0),
            function("f", "x^2"),
        )
        self.view = CanvasView(raw, inspection({"class": "Circle", "name": "A(5)", "color": "red"}))
        self.ctx = ctx_for(self.view)

    def names(self, selector: Any) -> list[str]:
        return sorted(obj.name for obj in select(selector, self.ctx))

    def test_type_name_and_at(self) -> None:
        assert self.names({"type": "Point", "at": [4, 0]}) == ["B"]
        assert self.names({"type": "Point", "name": "C"}) == ["C"]
        assert self.names({"type": "Point", "at": [4, 0.001]}) == []
        assert self.names({"type": "Point", "at": [4, 0.001], "tol": 0.01}) == ["B"]

    def test_segment_ends_either_order_vector_ordered(self) -> None:
        assert self.names({"type": "Segment", "ends": [[4, 0], [0, 0]]}) == ["AB"]
        assert self.names({"type": "Vector", "ends": [[0, 0], [4, 4]]}) == ["AD"]
        assert self.names({"type": "Vector", "ends": [[4, 4], [0, 0]]}) == []
        assert self.names({"type": "Vector", "origin": [0, 0], "tip": [4, 4]}) == ["AD"]

    def test_contains_through_slope(self) -> None:
        assert self.names({"type": "Segment", "contains": [2, 0]}) == ["AB"]
        assert self.names({"type": "Segment", "contains": [8, 0]}) == []
        assert self.names({"type": "Segment", "through": [8, 0]}) == ["AB"]
        assert self.names({"type": "Segment", "slope": -0.75}) == ["BC"]

    def test_vertices_any_order(self) -> None:
        assert self.names({"type": "Triangle", "vertices": [[0, 3], [0, 0], [4, 0]]}) == ["ABC"]
        assert self.names({"type": "Polygon", "vertices": [[0, 3], [0, 0], [4, 1]]}) == []

    def test_circle_center_radius_and_where(self) -> None:
        assert self.names({"type": "Circle", "center": [0, 0], "radius": 5}) == ["A(5)"]
        assert self.names({"type": "Circle", "radius": 4}) == []
        assert self.names({"type": "Circle", "where": {"inspect.color": "red"}}) == ["A(5)"]
        assert self.names({"type": "Circle", "where": {"args.radius": 5.0}}) == ["A(5)"]

    def test_within_box(self) -> None:
        assert self.names({"type": "Point", "within": [0, 0, 4, 1]}) == ["A", "B"]

    def test_only(self) -> None:
        assert self.names({"type": "Circle", "only": True}) == ["A(5)"]
        with pytest.raises(Exception, match="exactly one Point"):
            select({"type": "Point", "only": True}, self.ctx)

    def test_new_since_snapshot(self) -> None:
        before = CanvasView(state(point("A", 0, 0)))
        ctx = ctx_for(self.view, snapshots={"setup": before})
        assert sorted(o.name for o in select({"type": "Point", "new_since": "setup"}, ctx)) == ["B", "C", "D"]

    def test_bindings_and_literals(self) -> None:
        ctx = ctx_for(self.view, bindings={"M": ("Points", "C")})
        assert [o.name for o in select("$M", ctx)] == ["C"]
        assert select({"coords": [1, 2]}, ctx)[0].data == {"coords": [1.0, 2.0]}
        with pytest.raises(Exception, match="unknown binding"):
            select("$X", ctx)

    def test_function_samples(self) -> None:
        sampled = CanvasView(
            state(function("f", "x^2"), function("g", "x")),
            inspection(
                {"class": "Function", "name": "f", "samples": [[2, 4]]},
                {"class": "Function", "name": "g", "samples": [[2, 2]]},
            ),
        )
        assert [o.name for o in select({"type": "Function", "samples": [[2, 4]]}, ctx_for(sampled))] == ["f"]


# ----------------------------------------------------------------------
# Comparisons and paths
# ----------------------------------------------------------------------


class TestComparisons:
    tol = Tolerance()

    def test_resolve_path(self) -> None:
        data = {"args": {"label": {"text": "diag"}, "list": [1, [2, 3]]}}
        assert resolve_path(data, "args.label.text") == (True, "diag")
        assert resolve_path(data, "args.list[1][0]") == (True, 2)
        assert resolve_path(data, "args.list[-1]") == (True, [2, 3])
        assert resolve_path(data, "args.missing") == (False, None)

    def test_ops(self) -> None:
        assert compare(1.0 + 1e-12, {"eq": 1}, self.tol) is None
        assert compare(270.0, {"eq": 90, "mod": 180}, self.tol) is None
        assert compare(100.0, {"eq": 90, "mod": 180}, self.tol) is not None
        assert compare([1, 2], {"contains": 2}, self.tol) is None
        assert compare("abc", {"matches": "^a"}, self.tol) is None
        assert compare({"text": "reused A"}, {"matches": "reused"}, self.tol) is None
        assert compare(["B", "A"], {"set_eq": ["A", "B"]}, self.tol) is None
        assert compare("x = 2, y = 1", {"has_numbers": [1, 2]}, self.tol) is None
        assert compare(None, {"is_null": True}, self.tol) is None
        assert compare(3, {"le": 3, "gt": 2}, self.tol) is None
        assert compare(3, {"lt": 3}, self.tol) == "lt 3"
        assert compare("Call successful!", {"ne": "Call successful!"}, self.tol) is not None

    def test_numbers_in(self) -> None:
        assert numbers_in("[2, 3]") == [2.0, 3.0]
        assert numbers_in({"mean": 30, "note": "sd 14.14"}) == [30.0, 14.14]

    def test_call_is_error(self) -> None:
        assert call_is_error(call("x", "Error: boom"))
        assert call_is_error(call("x", {"error": "no graph"}))
        assert call_is_error(call("x", "ok", is_error=True))
        assert not call_is_error(call("x", {"path": None}))


# ----------------------------------------------------------------------
# Outcome checks
# ----------------------------------------------------------------------


class TestOutcomeChecks:
    def setup_method(self) -> None:
        self.view = CanvasView(state(*right_triangle(), point("M", 2, 0)))

    def test_exists_absent_count(self) -> None:
        assert run({"check": "exists", "select": {"type": "Point", "at": [2, 0]}}, self.view).status == "pass"
        assert run({"check": "absent", "select": {"type": "Point", "name": "Z"}}, self.view).status == "pass"
        assert run({"check": "count", "select": {"type": "Point"}, "eq": 4}, self.view).status == "pass"
        failed = run({"check": "count", "select": {"type": "Point"}, "eq": 3}, self.view)
        assert failed.status == "fail" and "count 4" in failed.message

    def test_known_turns_failure_into_xfail_and_pass_into_xpass(self) -> None:
        assert run({"check": "count", "select": {"type": "Point"}, "eq": 3, "known": "K1"}, self.view).status == "xfail"
        assert run({"check": "count", "select": {"type": "Point"}, "eq": 4, "known": "K1"}, self.view).status == "xpass"

    def test_bind_then_use(self) -> None:
        ctx = ctx_for(self.view)
        results = evaluate_checks(
            [
                {"bind": "M", "select": {"type": "Point", "at": [2, 0]}},
                {"check": "point_at", "select": "$M", "at": [2, 0]},
                {"check": "attribute", "select": "$M", "path": "name", "eq": "M"},
            ],
            ctx,
            "t1",
        )
        assert [r.status for r in results] == ["pass", "pass", "pass"]
        assert ctx.bindings["M"] == ("Points", "M")

    def test_bind_needs_exactly_one(self) -> None:
        result = run({"bind": "P", "select": {"type": "Point"}}, self.view)
        assert result.status == "fail" and "matched 4 objects" in result.message

    def test_moved(self) -> None:
        before = CanvasView(state(point("A", 0, 0)))
        after = CanvasView(state(point("A", 10, 0)))
        check = {"check": "moved", "select": {"type": "Point", "name": "A"}, "since": "setup", "by": [10, 0]}
        assert run(check, after, snapshots={"setup": before}).status == "pass"
        check["by"] = [0, 10]
        assert run(check, after, snapshots={"setup": before}).status == "fail"

    def test_attribute_targets_and_same_as(self) -> None:
        zoomed = CanvasView(state(view={"left_bound": -2, "right_bound": 2, "top_bound": 1, "bottom_bound": -1}))
        start = CanvasView(state())
        assert run({"check": "attribute", "target": "view", "path": "left_bound", "eq": -2}, zoomed).status == "pass"
        assert run({"check": "attribute", "target": "view", "path": "mode", "eq": "cartesian"}, zoomed).status == "pass"
        same = {"check": "attribute", "target": "view", "path": "left_bound", "same_as": "start"}
        assert run(same, zoomed, snapshots={"start": start}).status == "fail"
        assert run(same, start, snapshots={"start": start}).status == "pass"
        missing = run({"check": "attribute", "target": "state", "path": "nope", "eq": 1}, zoomed)
        assert missing.status == "fail" and "no value" in missing.message
        absent_ok = {"check": "attribute", "target": "state", "path": "nope", "is_null": True}
        assert run(absent_ok, zoomed).status == "pass"

    def test_state_equals(self) -> None:
        a = CanvasView(state(point("A", 0, 0), point("B", 1, 1)))
        reordered = CanvasView(state(point("B", 1, 1 + 1e-12), point("A", 0, 0)))
        moved = CanvasView(state(point("A", 0, 0), point("B", 2, 1)))
        check = {"check": "state_equals", "snapshot": "s"}
        assert run(check, reordered, snapshots={"s": a}).status == "pass"
        failed = run(check, moved, snapshots={"s": a})
        assert failed.status == "fail" and "Points B changed" in failed.message
        assert run({**check, "ignore": ["Point"]}, moved, snapshots={"s": a}).status == "pass"

    def test_state_equals_inspection_and_names(self) -> None:
        raw = state(point("A", 0, 0))
        red = CanvasView(raw, inspection({"class": "Point", "name": "A", "color": "red"}))
        black = CanvasView(raw, inspection({"class": "Point", "name": "A", "color": "black"}))
        check = {"check": "state_equals", "snapshot": "s"}
        assert run(check, black, snapshots={"s": red}).status == "pass"
        assert run({**check, "inspect": True}, black, snapshots={"s": red}).status == "fail"
        renamed = CanvasView(state(point("Q", 0, 0)))
        assert run(check, renamed, snapshots={"s": red}).status == "fail"
        assert run({**check, "match_names": False}, renamed, snapshots={"s": red}).status == "pass"

    def test_unchanged_except(self) -> None:
        before = CanvasView(state(point("A", 0, 0), point("B", 1, 1)))
        after = CanvasView(state(point("A", 5, 0), point("B", 1, 1), point("E", 2, 2)))
        check = {"check": "unchanged_except", "since": "s", "except": [{"type": "Point", "name": "A"}]}
        failed = run(check, after, snapshots={"s": before})
        assert failed.status == "fail" and "Points E added" in failed.message
        check["except"].append({"type": "Point", "at": [2, 2]})
        assert run(check, after, snapshots={"s": before}).status == "pass"

    def test_tool_checks(self) -> None:
        step = StepData(
            calls=[
                {"function_name": "search_tools", "arguments": {}, "result": "x", "is_error": False},
                call("create_point"),
                call("analyze_graph", {"path": ["A", "B"], "cost": 4}),
                call("solve", "[2, 3]"),
            ]
        )
        view = CanvasView(state())
        assert run({"check": "tool_called", "tool": "create_point"}, view, step=step).status == "pass"
        assert run({"check": "tool_not_called", "tool": "undo"}, view, step=step).status == "pass"
        assert run({"check": "max_tool_calls", "max": 3}, view, step=step).status == "pass"
        assert run({"check": "max_tool_calls", "max": 2}, view, step=step).status == "fail"
        assert run({"check": "no_tool_errors"}, view, step=step).status == "pass"
        assert (
            run({"check": "tool_result", "tool": "analyze_graph", "path": "cost", "eq": 4}, view, step=step).status
            == "pass"
        )
        assert (
            run(
                {"check": "tool_result", "tool": "analyze_graph", "path": "path", "eq": ["A", "B"]}, view, step=step
            ).status
            == "pass"
        )
        assert run({"check": "tool_result", "tool": "solve", "eq": [2, 3]}, view, step=step).status == "pass"
        assert run({"check": "tool_result", "tool": "solve", "has_numbers": [2, 3]}, view, step=step).status == "pass"
        missing = run({"check": "tool_result", "tool": "analyze_graph", "path": "order", "eq": 1}, view, step=step)
        assert missing.status == "fail" and "has no 'order'" in missing.message
        assert run({"check": "tool_error", "tool": "create_point"}, view, step=step).status == "fail"

    def test_tool_error_counts_error_dicts(self) -> None:
        step = StepData(calls=[call("analyze_graph", {"error": "Graph not found"})])
        view = CanvasView(state())
        assert run({"check": "tool_error", "tool": "analyze_graph"}, view, step=step).status == "pass"
        assert run({"check": "no_tool_errors"}, view, step=step).status == "fail"

    def test_answer_checks_skip_in_replay_and_run_live(self) -> None:
        view = CanvasView(state())
        check = {"check": "answer_mentions", "numbers": [2.5], "words": ["area"]}
        assert run(check, view, step=StepData(final_text=None)).status == "skip"
        live = StepData(final_text="The area is 2.50.", mode="live")
        assert run(check, view, step=live).status == "pass"
        assert run({**check, "numbers": [3]}, view, step=live).status == "fail"

    def test_known_check_that_crashes_is_an_error_not_xfail(self) -> None:
        broken = {
            "check": "relation",
            "relation": "distance",
            "select": [{"type": "Point", "name": "A"}],
            "known": "K1",
        }
        result = run(broken, self.view)
        assert result.status == "error" and result.unexpected

    def test_malformed_check_is_an_error(self) -> None:
        result = run(
            {"check": "attribute", "select": {"type": "Point", "name": "A"}, "path": "args", "lt": 3}, self.view
        )
        assert result.status == "fail"
        broken = run(
            {"check": "relation", "relation": "distance", "select": [{"type": "Point", "name": "A"}]}, self.view
        )
        assert broken.status == "error"


# ----------------------------------------------------------------------
# Relations
# ----------------------------------------------------------------------


def rel(relation: str, *selectors: Any, **params: Any) -> dict[str, Any]:
    return {"check": "relation", "relation": relation, "select": list(selectors), **params}


class TestRelations:
    def setup_method(self) -> None:
        raw = state(
            *points(A=(0, 0), B=(4, 0), C=(0, 3), M=(2, 0), P=(3, 4)),
            segment("A", "B", (0, 0), (4, 0)),
            segment("M", "A", (2, 0), (0, 0)),
            segment("C", "A", (0, 3), (0, 0)),
            segment("B", "C", (4, 0), (0, 3)),
            triangle("A", "B", "C", ["triangle", "scalene", "right"]),
            circle("A", 5, 0, 0),
            function("f", "x^2"),
        )
        insp = inspection(
            {"class": "Function", "name": "f", "samples": [[1, 1], [0.9999, 0.99980001], [1.0001, 1.00020001], [2, 4]]}
        )
        self.view = CanvasView(raw, insp)

    def status(self, check: dict[str, Any]) -> str:
        return run(check, self.view).status

    def test_point_relations(self) -> None:
        assert (
            self.status(rel("point_on_circle", {"type": "Point", "name": "P"}, {"type": "Circle", "only": True}))
            == "pass"
        )
        assert (
            self.status(rel("point_on_circle", {"type": "Point", "name": "B"}, {"type": "Circle", "only": True}))
            == "fail"
        )
        assert (
            self.status(rel("point_on_segment", {"type": "Point", "name": "M"}, {"type": "Segment", "name": "AB"}))
            == "pass"
        )
        assert self.status(rel("point_on_line", {"coords": [9, 0]}, {"type": "Segment", "name": "AB"})) == "pass"
        assert self.status(rel("point_on_segment", {"coords": [9, 0]}, {"type": "Segment", "name": "AB"})) == "fail"
        assert (
            self.status(rel("midpoint_of", {"type": "Point", "name": "M"}, {"type": "Segment", "name": "AB"})) == "pass"
        )
        assert self.status(rel("inside", {"coords": [1, 1]}, {"type": "Triangle", "only": True})) == "pass"
        assert self.status(rel("inside", {"coords": [3, 3]}, {"type": "Triangle", "only": True})) == "fail"

    def test_line_relations(self) -> None:
        ab, ma, ca = ({"type": "Segment", "name": n} for n in ("AB", "MA", "CA"))
        assert self.status(rel("collinear", ab, ma)) == "pass"
        assert self.status(rel("collinear", ab, ca)) == "fail"
        assert self.status(rel("parallel", ab, ma)) == "pass"
        assert self.status(rel("perpendicular", ab, ca)) == "pass"
        assert self.status(rel("equal_length", ma, {"type": "Segment", "name": "MA"})) == "pass"
        assert self.status(rel("length", {"type": "Segment", "name": "BC"}, value=5)) == "pass"
        assert self.status(rel("slope", {"type": "Segment", "name": "BC"}, value=-0.75)) == "pass"
        assert self.status(rel("direction", ab, parallel_to=[2, 0], perpendicular_to=[0, 1])) == "pass"
        assert self.status(rel("angle_deg", ab, ca, value=90)) == "pass"
        assert (
            self.status(rel("angle_deg", {"coords": [4, 0]}, {"coords": [0, 0]}, {"coords": [0, 3]}, value=90))
            == "pass"
        )
        assert (
            self.status(rel("distance", {"type": "Point", "name": "A"}, {"type": "Point", "name": "P"}, value=5))
            == "pass"
        )
        assert self.status(rel("distance", {"type": "Point", "name": "C"}, ab, value=3)) == "pass"

    def test_equal_angles(self) -> None:
        raw = state(
            *points(A=(0, 0), B=(4, 0), C=(0, 3), D=(1, 1)),
            segment("A", "B", (0, 0), (4, 0)),
            segment("C", "A", (0, 3), (0, 0)),
            segment("A", "D", (0, 0), (1, 1)),
        )
        view = CanvasView(raw)
        check = rel(
            "equal_angles",
            {"type": "Segment", "name": "AD"},
            {"type": "Segment", "name": "AB"},
            {"type": "Segment", "name": "CA"},
        )
        assert run(check, view).status == "pass"

    def test_area(self) -> None:
        assert self.status(rel("area", {"type": "Triangle", "only": True}, value=6)) == "pass"
        assert self.status(rel("area", {"type": "Circle", "only": True}, value=25 * math.pi)) == "pass"

    def test_tangent_to_circle_and_function(self) -> None:
        raw = state(
            *points(A=(0, 0), T=(-5, 5), U=(5, 5), V=(0, -1), W=(2, 3)),
            segment("T", "U", (-5, 5), (5, 5)),
            segment("V", "W", (0, -1), (2, 3)),
            circle("A", 5, 0, 0),
            function("f", "x^2"),
        )
        insp = inspection(
            {"class": "Function", "name": "f", "samples": [[1, 1], [0.9999, 0.99980001], [1.0001, 1.00020001]]}
        )
        view = CanvasView(raw, insp)
        assert (
            run(rel("tangent_to", {"type": "Segment", "name": "TU"}, {"type": "Circle", "only": True}), view).status
            == "pass"
        )
        tangent = rel("tangent_to", {"type": "Segment", "name": "VW"}, {"type": "Function", "name": "f"}, x=1, tol=1e-6)
        assert run(tangent, view).status == "pass"

    def test_function_values_need_samples(self) -> None:
        assert self.status(rel("function_value", {"type": "Function", "name": "f"}, x=2, y=4)) == "pass"
        assert self.status(rel("function_value", {"type": "Function", "name": "f"}, values=[[1, 1], [2, 5]])) == "fail"
        pending = run(rel("function_value", {"type": "Function", "name": "f"}, x=3, y=9), self.view)
        assert pending.status == "unrecorded" and "not recorded" in pending.message
        assert self.view.requests.x == {"f": {3.0}}

    def test_point_on_function(self) -> None:
        raw = state(point("R", 2, 4), function("f", "x^2"))
        view = CanvasView(raw, inspection({"class": "Function", "name": "f", "samples": [[2, 4]]}))
        assert (
            run(
                rel("point_on_function", {"type": "Point", "name": "R"}, {"type": "Function", "name": "f"}), view
            ).status
            == "pass"
        )


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


class TestValidateCheck:
    def test_valid_checks(self) -> None:
        assert validate_check({"check": "exists", "select": {"type": "Point", "at": [0, 0]}}) == []
        assert validate_check({"bind": "M", "select": {"type": "Point"}}) == []
        assert validate_check({"check": "attribute", "target": "view", "path": "left_bound", "same_as": "start"}) == []
        assert validate_check({"check": "no_tool_errors", "known": "K7"}) == []

    def test_problems(self) -> None:
        assert validate_check({"check": "nope"}) == ["unknown check type 'nope'"]
        assert "known must look like K<n>" in validate_check({"check": "no_tool_errors", "known": "bug"})[0]
        assert validate_check({"check": "exists"}) == ["exists needs a select"]
        assert "unknown selector key 'colour'" in validate_check(
            {"check": "exists", "select": {"type": "Point", "colour": 1}}
        )
        assert "count needs a comparison" in validate_check({"check": "count", "select": {"type": "Point"}})[0]
        assert (
            "unknown relation"
            in validate_check({"check": "relation", "relation": "near", "select": [{"type": "Point"}]})[0]
        )
        assert "state_equals needs a snapshot name" in validate_check({"check": "state_equals"})

    def test_unknown_keys_are_rejected(self) -> None:
        assert validate_check({"check": "exists", "select": {"type": "Point"}, "kown": "K1"}) == [
            "unknown key 'kown' for exists"
        ]
        assert "unknown key 'eq' for point_at" in validate_check(
            {"check": "point_at", "select": {"type": "Point"}, "at": [0, 0], "eq": 1}
        )
        assert "unknown key 'names' for bind" in validate_check({"bind": "M", "select": {"type": "Point"}, "names": 1})
        assert "unknown key 'snapshots' for state_equals" in validate_check(
            {"check": "state_equals", "snapshot": "s", "snapshots": "t"}
        )
        assert validate_check({"check": "no_tool_errors", "known": "K1", "tol": 1, "id": "x", "note": "n"}) == []

    def test_required_keys(self) -> None:
        point_sel = {"type": "Point", "name": "A"}
        assert "point_at needs at" in validate_check({"check": "point_at", "select": point_sel})
        assert "point_at needs at as [x, y]" in validate_check({"check": "point_at", "select": point_sel, "at": [1]})
        assert "moved needs by" in validate_check({"check": "moved", "select": point_sel, "since": "setup"})
        assert validate_check({"check": "moved", "select": point_sel, "since": "setup", "by": [1, 0]}) == []
        # Keys with their own messages are reported once.
        assert validate_check({"check": "state_equals"}) == ["state_equals needs a snapshot name"]
        assert validate_check({"check": "tool_error"}) == ["tool_error needs a tool"]

    def test_relation_selector_counts_and_tangent_x(self) -> None:
        seg = {"type": "Segment", "only": True}
        assert "relation parallel takes 2 selectors, got 1" in validate_check(
            {"check": "relation", "relation": "parallel", "select": [seg]}
        )
        assert "relation length takes 1 selectors, got 2" in validate_check(
            {"check": "relation", "relation": "length", "select": [seg, seg], "value": 1}
        )
        assert "relation point_on_circle takes 2 or more selectors, got 1" in validate_check(
            {"check": "relation", "relation": "point_on_circle", "select": [seg]}
        )
        assert "relation angle_deg takes 1 to 3 selectors, got 4" in validate_check(
            {"check": "relation", "relation": "angle_deg", "select": [seg] * 4, "value": 90}
        )
        function = {"type": "Function", "name": "f"}
        assert "relation tangent_to a function needs x" in validate_check(
            {"check": "relation", "relation": "tangent_to", "select": [seg, function]}
        )
        assert validate_check({"check": "relation", "relation": "tangent_to", "select": [seg, function], "x": 1}) == []
        assert (
            validate_check({"check": "relation", "relation": "tangent_to", "select": [seg, {"type": "Circle"}]}) == []
        )

    def test_relation_parameters(self) -> None:
        segment = {"type": "Segment", "only": True}
        assert "relation direction needs parallel_to or perpendicular_to" in validate_check(
            {"check": "relation", "relation": "direction", "select": [segment]}
        )
        assert (
            validate_check({"check": "relation", "relation": "direction", "select": [segment], "parallel_to": [1, 0]})
            == []
        )
        assert "relation length needs a value" in validate_check(
            {"check": "relation", "relation": "length", "select": [segment]}
        )
        assert "relation function_value needs x and y, or values" in validate_check(
            {"check": "relation", "relation": "function_value", "select": [{"type": "Function"}], "x": 1}
        )
        assert "unknown selector key 'at_'" in validate_check(
            {"check": "unchanged_except", "since": "s", "except": [{"type": "Point", "at_": [0, 0]}]}
        )


# ----------------------------------------------------------------------
# Invariants (states copied from section 6's observations)
# ----------------------------------------------------------------------


def invariants(
    before: CanvasView, after: CanvasView, step: Optional[StepData] = None, waivers: Optional[dict[str, str]] = None
) -> list[CheckResult]:
    return run_invariants(before, after, step or StepData(calls=[call("noop_tool")]), "t1", waivers or {})


class TestInvariants:
    empty = CanvasView(state())

    def test_clean_state_passes_all(self) -> None:
        after = CanvasView(state(*right_triangle()))
        step = StepData(calls=[call("create_polygon", name="ABC")], undoable=[True], undo_before=0, undo_after=1)
        results = invariants(self.empty, after, step)
        assert [r.status for r in results] == ["pass"] * 7

    def test_i1_duplicates_fail_and_cross_bucket_warns(self) -> None:
        duplicate = CanvasView(state(point("A", 0, 0), point("A", 1, 1)))
        assert by_id(invariants(self.empty, duplicate), "I1").status == "fail"
        clash = CanvasView(
            state(*points(A=(0, 0), B=(3, 0)), segment("A", "B", (0, 0), (3, 0)), vector("A", "B", (0, 0), (3, 0)))
        )
        result = by_id(invariants(self.empty, clash), "I1")
        assert result.status == "warn" and "AB is used by Segments, Vectors" in result.message

    def test_i2_dangling_references(self) -> None:
        dangling = CanvasView(
            state(
                point("A", 0, 0),
                segment("A", "B", (0, 0), (1, 0)),
                ("ContinuousPlots", {"name": "nd", "args": {"function_name": "nd_pdf", "fill_area_name": "nd_fill"}}),
                ("FunctionsBoundedColoredAreas", {"name": "area", "args": {"func1": "f", "func2": "x_axis"}}),
            )
        )
        result = by_id(invariants(self.empty, dangling), "I2")
        assert result.status == "fail"
        for fragment in ("p2 B does not exist", "nd_pdf", "nd_fill", "func1 f"):
            assert fragment in result.message

    def test_i3_stale_circle_formula_k8(self) -> None:
        moved = state(*points(A=(10, 0)), circle("A", 5, 0, 0))
        result = by_id(invariants(self.empty, CanvasView(moved)), "I3")
        assert result.status == "fail" and "Circle A(5)" in result.message
        fresh = state(*points(A=(10, 0)), circle("A", 5, 10, 0))
        assert by_id(invariants(self.empty, CanvasView(fresh)), "I3").status == "pass"

    def test_i3_stale_segment_coords(self) -> None:
        stale = state(*points(A=(0, 0), B=(5, 0)), segment("A", "B", (0, 0), (4, 0)))
        assert "_p2_coords" in by_id(invariants(self.empty, CanvasView(stale)), "I3").message

    def test_i3_triangle_types_k16(self) -> None:
        sheared = state(
            *points(A=(0, 0), B=(4, 0), C=(5.464101615138, 3.464101615138)),
            triangle("A", "B", "C", ["triangle", "equilateral", "isosceles"]),
        )
        assert "equilateral" in by_id(invariants(self.empty, CanvasView(sheared)), "I3").message

    def test_i3_cached_angle_k17(self) -> None:
        raw = state(*right_triangle(), angle("angle_BAC", "AB", "CA"))
        stale = CanvasView(raw, inspection({"class": "Angle", "name": "angle_BAC", "angle_degrees": 60}))
        assert "cached 60" in by_id(invariants(self.empty, stale), "I3").message
        fresh = CanvasView(raw, inspection({"class": "Angle", "name": "angle_BAC", "angle_degrees": 90}))
        assert by_id(invariants(self.empty, fresh), "I3").status == "pass"

    def test_i3_asymptotes_and_bounds_k11_k12(self) -> None:
        raw = state(
            function("f", "(1/(x - 2)) + 3", vertical_asymptotes=[0]), function("g", "x", left_bound=5, right_bound=-5)
        )
        insp = inspection(
            {"class": "Function", "name": "f", "asymptote_probes": [[0, 2.5000000249, 2.4999999749]]},
            {"class": "Function", "name": "g"},
        )
        message = by_id(invariants(self.empty, CanvasView(raw, insp)), "I3").message
        assert "vertical asymptote x = 0" in message and "left_bound 5" in message
        real = inspection({"class": "Function", "name": "f", "asymptote_probes": [[2, -1e7, 1e7]]})
        ok = state(function("f", "1/(x-2)", vertical_asymptotes=[2]))
        assert by_id(invariants(self.empty, CanvasView(ok, real)), "I3").status == "pass"

    def test_i3_ellipse_formula(self) -> None:
        good = state(
            point("A", -8, 5),
            (
                "Ellipses",
                {
                    "name": "A(3, 1.5)",
                    "args": {
                        "center": "A",
                        "radius_x": 3,
                        "radius_y": 1.5,
                        "rotation_angle": 30,
                        "ellipse_formula": "0.1944444444*(x - -8)**2 - 0.2886751346*(x - -8)*(y - 5) + 0.3611111111*(y - 5)**2 = 1",
                    },
                },
            ),
        )
        assert by_id(invariants(self.empty, CanvasView(good)), "I3").status == "pass"
        good["Ellipses"][0]["args"]["rotation_angle"] = 90
        assert by_id(invariants(self.empty, CanvasView(good)), "I3").status == "fail"

    def test_i4_success_that_changed_nothing_k2(self) -> None:
        step = StepData(calls=[call("delete_circle", name="Z(9)")])
        view = CanvasView(state(point("A", 0, 0)))
        result = by_id(invariants(view, view, step), "I4")
        assert result.status == "fail" and "changed nothing" in result.message

    def test_i4_truthful_no_op_passes(self) -> None:
        """Results that explain why nothing changed are truthful (as returned once K2 and K3 are fixed)."""
        view = CanvasView(state(*right_triangle()))
        occupied = StepData(
            calls=[
                call(
                    "create_point",
                    "Point 'C' already exists at (0, 3); no new point was created. The requested name 'Z' was not applied.",
                    x=0,
                    y=3,
                    name="Z",
                )
            ]
        )
        assert by_id(invariants(view, view, occupied), "I4").status == "pass"
        empty_undo = StepData(calls=[call("undo", "Nothing to undo: the undo history is empty.")])
        assert by_id(invariants(view, view, empty_undo), "I4").status == "pass"
        bare = StepData(calls=[call("undo", True)])
        assert by_id(invariants(view, view, bare), "I4").status == "fail"

    def test_i4_error_that_changed_the_canvas(self) -> None:
        step = StepData(calls=[call("translate_object", "Error: nope", is_error=True, name="x")])
        before, after = CanvasView(state(point("A", 0, 0))), CanvasView(state(point("A", 1, 0)))
        assert "every call failed" in by_id(invariants(before, after, step), "I4").message

    def test_i4_silent_rename_k3(self) -> None:
        before = CanvasView(state(point("A", 0, 0)))
        after = CanvasView(state(point("A", 0, 0), point("B", 7, 7)))
        silent = StepData(calls=[call("create_point", x=7, y=7, name="A")])
        assert "asked for name 'A'" in by_id(invariants(before, after, silent), "I4").message
        reported = StepData(calls=[call("create_point", "Created point B (A was taken)", x=7, y=7, name="A")])
        assert by_id(invariants(before, after, reported), "I4").status == "pass"

    def test_i4_naming_rule_skipped_when_the_batch_deletes(self) -> None:
        before = CanvasView(state())
        after = CanvasView(state(point("Z", 9, 9)))
        step = StepData(
            calls=[call("create_polygon", name="ABC"), call("clear_canvas"), call("create_point", name="Z")]
        )
        assert by_id(invariants(before, after, step), "I4").status == "pass"

    def test_i4_waiver_makes_it_xfail(self) -> None:
        view = CanvasView(state())
        step = StepData(calls=[call("undo")])
        result = by_id(invariants(view, view, step, {"I4": "K2"}), "I4")
        assert result.status == "xfail" and result.known == "K2"

    def test_i5_undo_accounting_k1(self) -> None:
        view = CanvasView(state())
        triangle_view = CanvasView(state(*right_triangle()))
        three = StepData(calls=[call("create_segment")] * 3, undoable=[True] * 3, undo_before=0, undo_after=17)
        result = by_id(invariants(view, triangle_view, three), "I5")
        assert result.status == "fail" and "added 17 undo entries, expected 1" in result.message
        assert by_id(invariants(view, triangle_view, three, {"I5": "K1"}), "I5").status == "xfail"
        one = StepData(calls=[call("create_segment")] * 3, undoable=[True] * 3, undo_before=0, undo_after=1)
        assert by_id(invariants(view, triangle_view, one), "I5").status == "pass"
        failed = StepData(
            calls=[call("scale_object", "Error: no", is_error=True)], undoable=[True], undo_before=3, undo_after=4
        )
        assert "expected 0" in by_id(invariants(view, view, failed), "I5").message
        read_only = StepData(
            calls=[call("calculate_area", {"value": 2})], undoable=[False], undo_before=2, undo_after=2
        )
        assert by_id(invariants(view, view, read_only), "I5").status == "pass"

    def test_i5_judges_by_what_changed_not_by_what_was_reported(self) -> None:
        canvas = CanvasView(state(*right_triangle()))
        # A truthful no-op create (the point already exists and the result says so) must add no entry.
        truthful_no_op = StepData(
            calls=[call("create_point", "C already exists at (0, 3)", x=0, y=3)],
            undoable=[True],
            undo_before=5,
            undo_after=5,
        )
        assert by_id(invariants(canvas, canvas, truthful_no_op), "I5").status == "pass"
        archived_anyway = StepData(
            calls=[call("create_point", "C already exists at (0, 3)", x=0, y=3)],
            undoable=[True],
            undo_before=5,
            undo_after=6,
        )
        result = by_id(invariants(canvas, canvas, archived_anyway), "I5")
        assert result.status == "fail" and "changed nothing" in result.message
        # A non-undoable tool that changes drawables (load, regression) still owes one entry.
        loaded = StepData(calls=[call("load_workspace", name="w")], undoable=[False], undo_before=0, undo_after=0)
        assert by_id(invariants(CanvasView(state()), canvas, loaded), "I5").status == "fail"

    def test_i5_counts_view_mode_and_inspection_changes(self) -> None:
        """A batch that changes only the view, the mode or a colour still changed the canvas (as in I4)."""
        base = CanvasView(state(point("A", 0, 0)), inspection({"class": "Point", "name": "A", "color": "black"}))
        zoomed = CanvasView(
            state(point("A", 0, 0), view={"left_bound": -2, "right_bound": 2, "top_bound": 1, "bottom_bound": -1}),
            inspection({"class": "Point", "name": "A", "color": "black"}),
        )
        polar = CanvasView(
            state(point("A", 0, 0), mode="polar"), inspection({"class": "Point", "name": "A", "color": "black"})
        )
        red = CanvasView(state(point("A", 0, 0)), inspection({"class": "Point", "name": "A", "color": "red"}))
        cases = [
            (zoomed, call("zoom", center_x=0, center_y=0, range_val=2, range_axis="x")),
            (polar, call("set_coordinate_system", mode="polar")),
            (red, call("update_point", point_name="A", new_color="red")),
        ]
        for after, batch_call in cases:
            one_entry = StepData(calls=[batch_call], undoable=[True], undo_before=3, undo_after=4)
            assert by_id(invariants(base, after, one_entry), "I5").status == "pass", batch_call["function_name"]
            no_entry = StepData(calls=[batch_call], undoable=[True], undo_before=3, undo_after=3)
            result = by_id(invariants(base, after, no_entry), "I5")
            assert result.status == "fail" and "changed the canvas" in result.message, batch_call["function_name"]
        unchanged = StepData(calls=[call("zoom")], undoable=[True], undo_before=3, undo_after=4)
        assert "changed nothing" in by_id(invariants(base, base, unchanged), "I5").message

    def test_i5_undo_and_redo_simulation(self) -> None:
        view = CanvasView(state())
        undo = StepData(calls=[call("undo")], undoable=[False], undo_before=2, undo_after=1, redo_before=0)
        assert by_id(invariants(view, view, undo), "I5").status == "pass"
        empty_undo = StepData(calls=[call("undo")], undoable=[False], undo_before=0, undo_after=0, redo_before=0)
        assert by_id(invariants(view, view, empty_undo), "I5").status == "pass"
        bad = StepData(calls=[call("undo")], undoable=[False], undo_before=3, undo_after=3, redo_before=0)
        assert by_id(invariants(view, view, bad), "I5").status == "fail"

    def test_i6_error_dicts_must_be_flagged_k21(self) -> None:
        view = CanvasView(state())
        step = StepData(calls=[call("analyze_graph", {"error": "Graph G not found"})])
        assert by_id(invariants(view, view, step), "I6").status == "fail"
        flagged = StepData(calls=[call("analyze_graph", {"error": "Graph G not found"}, is_error=True)])
        assert by_id(invariants(view, view, flagged), "I6").status == "pass"

    def test_i7_runaway_numbers_k13(self) -> None:
        runaway = CanvasView(state(point("A", 1.5708, 1.63312e16)))
        assert "1.63312e+16" in by_id(invariants(self.empty, runaway), "I7").message
        nan = CanvasView(state(point("A", float("nan"), 0)))
        assert by_id(invariants(self.empty, nan), "I7").status == "fail"
        allowed = run_invariants(self.empty, runaway, StepData(calls=[call("x")]), "t1", {}, allow_large=True)
        assert by_id(allowed, "I7").status == "pass"


class TestDiffViews:
    def test_view_and_mode(self) -> None:
        a = CanvasView(state())
        b = CanvasView(
            state(view={"left_bound": -2, "right_bound": 2, "top_bound": 1, "bottom_bound": -1}, mode="polar")
        )
        tol = Tolerance()
        assert diff_views(a, b, tol) == ["coordinate mode cartesian -> polar"]
        assert len(diff_views(a, b, tol, include_view=True)) == 2
