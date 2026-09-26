#!/usr/bin/env python3
"""
Canvas-format comprehension benchmark.

Asks models factual questions about the canvas-state fixtures in
``server_tests/fixtures/canvas_states/`` and measures how well (and how cheaply)
they answer when the canvas reaches them in each ``MATHUD_CANVAS_FORMAT``
(``json``, ``min_json``, ``text``).

The messages are built by the providers' own code, as the app builds them: the
system prompt (``build_developer_message`` plus the search-mode paragraph), the
user message (prompt JSON in ``json`` format, ``<canvas>`` block plus text
otherwise; only an object-count line for LocalAgent in ``json``), and for the
change scene the continuation request after a tool batch (tool results,
followed in ``text``/``min_json`` by ``[canvas changes]``; in ``json`` the app
sends no canvas at all there, since the state is stripped from history after
the first completion). Unlike the app, requests are not streamed, carry no tools
(so a ``json`` model cannot re-fetch the canvas with get_current_canvas_state)
and send no OpenRouter ``reasoning_details``; LocalAgent requests carry the
provider's own sampling and reasoning-effort parameters.

Ground truths are computed from the fixtures, and answers are graded by the pure
functions below (numbers with a tolerance, sets order-insensitive, names
case-insensitive). Questions whose information a format does not send (see
``provided_info``) are reported as "not provided" instead of counting toward
accuracy; the headline figure is the accuracy on the static scenes, with the
change scene reported separately.

Usage examples:
  python scripts/benchmark_canvas_formats.py --dry-run
  python scripts/benchmark_canvas_formats.py --models deepseek/deepseek-v4.1-flash xiaomi/mimo-v2.6-pro \\
      --formats json text
  python scripts/benchmark_canvas_formats.py --provider local
  python scripts/benchmark_canvas_formats.py --regrade logs/canvas_format_benchmark/<time>/results.json

``--provider openrouter`` (default) needs OPENROUTER_API_KEY (loaded from the
environment or .env like the app does). ``--provider local`` targets the
LocalAgent llama-server (LOCAL_AGENT_BASE_URL, default http://127.0.0.1:8080),
uses every model it serves unless ``--models`` picks one, and compares
``text``, ``min_json`` and ``json`` by default (``min_json`` is the real structured
JSON LocalAgent can send; its ``json`` is only the legacy count line).
``--dry-run`` builds and writes every prompt and prints token and cost
estimates without any network access. Each answer is appended to
``results.jsonl`` as it arrives; ``results.json`` and ``summary.md`` are written
at the end, also when the run is interrupted. ``--regrade`` re-grades a
``results.json`` with the current grader, without network access.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from static import env_config  # noqa: E402
from static.ai_model import AIModel  # noqa: E402
from static.canvas_state_formatter import CANVAS_FORMATS, CanvasFormat  # noqa: E402
from static.client.constants import successful_call_message  # noqa: E402
from static.openai_api_base import CANVAS_BUDGET_ENV, CANVAS_FORMAT_ENV, TOOL_EXPOSURE_ENV, OpenAIAPIBase  # noqa: E402
from static.providers.local import REASONING_EFFORT_ENV, REASONING_EFFORTS  # noqa: E402
from static.providers.local.local_agent_api import LocalAgentAPI  # noqa: E402
from static.providers.openrouter_api import OpenRouterAPI  # noqa: E402
from static.response_metrics import ResponseMetricsTracker, record_chat_completions_usage  # noqa: E402
from static.token_estimation import estimate_tokens_from_text  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "server_tests" / "fixtures" / "canvas_states"

PROVIDERS = ("openrouter", "local")
DEFAULT_OPENROUTER_MODELS = ("deepseek/deepseek-v4.1-flash", "xiaomi/mimo-v2.6-pro")
LOCAL_PLACEHOLDER_MODEL = "local-model"
DEFAULT_FORMATS: Dict[str, Tuple[CanvasFormat, ...]] = {
    "openrouter": ("json", "text"),
    "local": ("text", "min_json", "json"),
}
LOCAL_REASONING_EFFORT_CHOICES = REASONING_EFFORTS + ("default",)
DEFAULT_MAX_REQUESTS = 250
DEFAULT_TIMEOUT_S = {"openrouter": 180.0, "local": 900.0}
DEFAULT_CONCURRENCY = {"openrouter": 4, "local": 1}

# USD per 1M tokens (input, output) on OpenRouter, as of PRICES_AS_OF.
PRICES_AS_OF = "2026-09-25"
PRICES_PER_MTOK: Dict[str, Tuple[float, float]] = {
    "deepseek/deepseek-v4.1-flash": (0.099, 0.60),
    "xiaomi/mimo-v2.6-pro": (0.435, 0.87),
}
# Dry-run cost estimates assume this many completion tokens per answer (reasoning included).
ESTIMATED_COMPLETION_TOKENS = 400

# Provider settings the benchmark pins while it builds prompts, so a local .env
# cannot change them: each provider's defaults apply (search tool mode, hybrid
# json summaries, canvas budget 4000 cloud / 1500 local unless --canvas-budget,
# LocalAgent reasoning effort medium unless --local-reasoning-effort).
_PINNED_ENV_VARS = (
    TOOL_EXPOSURE_ENV,
    CANVAS_BUDGET_ENV,
    REASONING_EFFORT_ENV,
    OpenAIAPIBase.CANVAS_SUMMARY_MODE_ENV,
    OpenAIAPIBase.CANVAS_HYBRID_MAX_BYTES_ENV,
    OpenAIAPIBase.CANVAS_SUMMARY_TELEMETRY_ENV,
)
_OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

ANSWER_INSTRUCTIONS = (
    "No tools are available for this question: answer from the canvas state you were given, "
    "working out anything you need yourself. Give numbers as plain decimals. "
    "End your reply with a single line of the form `Answer: <value>`."
)
CHANGE_TURN_INTRO = "Edit the sketch, then answer the question below about the canvas as it is after your edits."

JsonDict = Dict[str, Any]
Provider = Union[OpenRouterAPI, LocalAgentAPI]
Point = Tuple[float, float]


# --------------------------------------------------------------------------- scenes


@dataclass(frozen=True)
class Scene:
    """A benchmark scene: a fixture state, plus the state after a tool batch for the change scene."""

    name: str
    state: JsonDict
    after: Optional[JsonDict] = None


CHANGE_SCENE = "mixed_medium_change"
SCENE_NAMES = ("triangle_circle", "mixed_medium", "weighted_graph", "regression_duplicates", CHANGE_SCENE)


def load_fixture(name: str) -> JsonDict:
    with open(FIXTURES_DIR / f"{name}.json", encoding="utf-8") as handle:
        data = json.load(handle)
    assert isinstance(data, dict)
    return data


def load_scenes(names: Sequence[str] = SCENE_NAMES) -> List[Scene]:
    scenes: List[Scene] = []
    for name in names:
        if name == CHANGE_SCENE:
            scenes.append(Scene(name, load_fixture("mixed_medium"), load_fixture("mixed_medium_after")))
        else:
            scenes.append(Scene(name, load_fixture(name)))
    return scenes


# --------------------------------------------------------------------------- geometry of a state


def _items(state: Mapping[str, Any], bucket: str) -> List[JsonDict]:
    value = state.get(bucket)
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _args(item: Mapping[str, Any]) -> JsonDict:
    args = item.get("args")
    return args if isinstance(args, dict) else {}


def _find(state: Mapping[str, Any], bucket: str, name: str) -> JsonDict:
    for item in _items(state, bucket):
        if item.get("name") == name:
            return item
    raise KeyError(f"{bucket} {name!r} not in the scene")


def point_list(state: Mapping[str, Any]) -> List[Tuple[str, Point]]:
    """Every point as (name, (x, y)), duplicates included, in state order."""
    points: List[Tuple[str, Point]] = []
    for item in _items(state, "Points"):
        position = _args(item).get("position", {})
        points.append((str(item.get("name")), (float(position["x"]), float(position["y"]))))
    return points


def point_map(state: Mapping[str, Any]) -> Dict[str, Point]:
    """Point name -> coordinates (the first point of a duplicated name wins)."""
    result: Dict[str, Point] = {}
    for name, xy in point_list(state):
        result.setdefault(name, xy)
    return result


def distance(p: Point, q: Point) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def segment_endpoints(state: Mapping[str, Any], name: str) -> Tuple[str, str]:
    args = _args(_find(state, "Segments", name))
    return str(args["p1"]), str(args["p2"])


def segment_length(state: Mapping[str, Any], name: str) -> float:
    points = point_map(state)
    p1, p2 = segment_endpoints(state, name)
    return distance(points[p1], points[p2])


def polygon_area(coords: Sequence[Point]) -> float:
    twice = sum(coords[i][0] * coords[(i + 1) % len(coords)][1] for i in range(len(coords)))
    twice -= sum(coords[(i + 1) % len(coords)][0] * coords[i][1] for i in range(len(coords)))
    return abs(twice) / 2.0


def triangle_vertices(state: Mapping[str, Any], name: str) -> List[str]:
    args = _args(_find(state, "Triangles", name))
    return [str(args["p1"]), str(args["p2"]), str(args["p3"])]


def interior_angles(state: Mapping[str, Any], vertices: Sequence[str]) -> Dict[str, float]:
    """Interior angle in degrees at each vertex of a triangle."""
    points = point_map(state)
    angles: Dict[str, float] = {}
    for index, vertex in enumerate(vertices):
        a = points[vertex]
        b = points[vertices[(index + 1) % 3]]
        c = points[vertices[(index + 2) % 3]]
        angles[vertex] = _angle_between((b[0] - a[0], b[1] - a[1]), (c[0] - a[0], c[1] - a[1]))
    return angles


def _angle_between(u: Point, v: Point) -> float:
    cosine = (u[0] * v[0] + u[1] * v[1]) / (math.hypot(*u) * math.hypot(*v))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def angle_degrees(state: Mapping[str, Any], name: str) -> float:
    """Size of a (non-reflex) angle object defined by two segments sharing a vertex."""
    args = _args(_find(state, "Angles", name))
    first = segment_endpoints(state, str(args["segment1_name"]))
    second = segment_endpoints(state, str(args["segment2_name"]))
    shared = set(first) & set(second)
    assert len(shared) == 1, f"angle {name} has no single shared vertex"
    vertex = shared.pop()
    points = point_map(state)
    arm1 = next(p for p in first if p != vertex)
    arm2 = next(p for p in second if p != vertex)
    origin = points[vertex]
    u = (points[arm1][0] - origin[0], points[arm1][1] - origin[1])
    v = (points[arm2][0] - origin[0], points[arm2][1] - origin[1])
    return _angle_between(u, v)


def circle_center_radius(state: Mapping[str, Any], name: str) -> Tuple[str, float]:
    args = _args(_find(state, "Circles", name))
    return str(args["center"]), float(args["radius"])


def points_on_circle(state: Mapping[str, Any], name: str, relative_tolerance: float = 1e-6) -> Set[str]:
    """Named points (other than the center) whose distance from the center equals the radius."""
    center, radius = circle_center_radius(state, name)
    points = point_map(state)
    origin = points[center]
    return {
        point
        for point, xy in points.items()
        if point != center and abs(distance(origin, xy) - radius) <= relative_tolerance * radius
    }


def unique_extreme(values: Mapping[str, float], largest: bool = True) -> str:
    """Key of the largest (or smallest) value; raises if the extreme is tied."""
    ranked = sorted(values.items(), key=lambda kv: kv[1], reverse=largest)
    if len(ranked) > 1 and math.isclose(ranked[0][1], ranked[1][1], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"tie between {ranked[0][0]} and {ranked[1][0]}")
    return ranked[0][0]


@dataclass(frozen=True)
class Edge:
    name: str
    ends: Tuple[str, str]
    weight: float


def graph_edges(state: Mapping[str, Any], bucket: str, name: str) -> List[Edge]:
    """A graph's edges: its segments with the weight from their label text."""
    edges: List[Edge] = []
    for segment_name in _args(_find(state, bucket, name)).get("segments", []):
        segment = _find(state, "Segments", str(segment_name))
        label = _args(segment).get("label", {})
        edges.append(Edge(str(segment_name), segment_endpoints(state, str(segment_name)), float(label["text"])))
    return edges


def shortest_path_length(edges: Sequence[Edge], start: str, goal: str) -> float:
    """Dijkstra over an undirected weighted edge list."""
    best: Dict[str, float] = {start: 0.0}
    done: Set[str] = set()
    while True:
        frontier = {node: cost for node, cost in best.items() if node not in done}
        if not frontier:
            raise ValueError(f"{goal} is not reachable from {start}")
        node = min(frontier, key=lambda key: frontier[key])
        if node == goal:
            return best[node]
        done.add(node)
        for edge in edges:
            if node in edge.ends:
                other = edge.ends[1] if edge.ends[0] == node else edge.ends[0]
                cost = best[node] + edge.weight
                if cost < best.get(other, math.inf):
                    best[other] = cost


def linear_coefficients(function_string: str) -> Tuple[float, float]:
    """Slope and intercept of a fitted line written as ``(m)*x + b``."""
    match = re.fullmatch(r"\(?\s*(?P<m>[-+\d.eE]+)\s*\)?\s*\*\s*x\s*\+\s*(?P<b>[-+\d.eE]+)", function_string.strip())
    if match is None:
        raise ValueError(f"not a linear function: {function_string!r}")
    return float(match.group("m")), float(match.group("b"))


def view_bounds(state: Mapping[str, Any]) -> Dict[str, float]:
    visibility = state["Cartesian_System_Visibility"]
    return {key: float(visibility[key]) for key in ("left_bound", "right_bound", "top_bound", "bottom_bound")}


def object_keys(state: Mapping[str, Any]) -> Dict[Tuple[str, str], JsonDict]:
    """(bucket, name) -> item for every named drawable of the state."""
    keys: Dict[Tuple[str, str], JsonDict] = {}
    for bucket, value in state.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "name" in item:
                    keys.setdefault((bucket, str(item["name"])), item)
    return keys


@dataclass(frozen=True)
class CanvasDiff:
    added: List[Tuple[str, str]]
    removed: List[Tuple[str, str]]
    moved_points: Dict[str, Tuple[Point, Point]]


def canvas_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> CanvasDiff:
    before_keys, after_keys = object_keys(before), object_keys(after)
    before_points, after_points = point_map(before), point_map(after)
    moved = {
        name: (before_points[name], xy)
        for name, xy in after_points.items()
        if name in before_points and before_points[name] != xy
    }
    return CanvasDiff(
        added=[key for key in after_keys if key not in before_keys],
        removed=[key for key in before_keys if key not in after_keys],
        moved_points=moved,
    )


def object_names(state: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(sorted({name for _, name in object_keys(state)}))


# --------------------------------------------------------------------------- questions

ANSWER_KINDS = ("number", "numbers", "name", "set", "edge", "expression")

# Information a question needs, and which formats send it (see provided_info).
INFO_CANVAS = "canvas"  # the scene's objects with names and coordinates (change scene: before the edits)
INFO_COUNTS = "counts"  # how many objects of each type
INFO_CHANGES = "changes"  # what the tool batch changed, with names ("[canvas changes]")
INFO_TOOL_ARGS = "tool_args"  # the arguments of the tool calls in the history
INFO_KINDS = frozenset({INFO_CANVAS, INFO_COUNTS, INFO_CHANGES, INFO_TOOL_ARGS})
_COUNT_NEEDS = ({INFO_COUNTS}, {INFO_CANVAS})


@dataclass(frozen=True)
class Question:
    """One factual question with a machine-checkable expected answer.

    ``expected`` by kind: number -> float; numbers -> ordered floats; name and
    expression -> accepted spellings; set and edge -> names (order-insensitive).
    ``vocabulary`` holds the names a free-form answer is matched against: the
    scene's object names, or for set questions only the names of the kind asked
    for (points, segments). ``needs`` lists alternative sets of information (see
    ``INFO_*``), any one of which is enough to answer; a format that sends none
    of them leaves the question "not provided". ``after_arrow`` grades only the
    part after an arrow ("old -> new" answers to a new-position question).
    """

    scene: str
    qid: str
    category: str
    text: str
    kind: str
    expected: Any
    vocabulary: Tuple[str, ...] = field(default=(), repr=False)
    needs: Tuple[FrozenSet[str], ...] = field(default=(frozenset({INFO_CANVAS}),), repr=False)
    after_arrow: bool = field(default=False, repr=False)


def _question(
    scene: Scene,
    qid: str,
    category: str,
    text: str,
    kind: str,
    expected: Any,
    needs: Sequence[Set[str]] = (),
    set_of: Optional[str] = None,
    after_arrow: bool = False,
) -> Question:
    """Build a question; ``set_of`` names the state bucket whose names a set answer may contain."""
    assert kind in ANSWER_KINDS, kind
    assert all(option <= INFO_KINDS for option in needs), needs
    states = [scene.state] + ([scene.after] if scene.after is not None else [])
    if set_of is not None:
        vocabulary = {str(item.get("name")) for state in states for item in _items(state, set_of)}
    else:
        vocabulary = {name for state in states for name in object_names(state)}
    return Question(
        scene.name,
        qid,
        category,
        text,
        kind,
        expected,
        tuple(sorted(vocabulary)),
        tuple(frozenset(option) for option in needs) or (frozenset({INFO_CANVAS}),),
        after_arrow,
    )


def _triangle_circle_questions(scene: Scene) -> List[Question]:
    state = scene.state
    points = point_map(state)
    vertices = triangle_vertices(state, "ABC")
    coords = [points[v] for v in vertices]
    _, radius = circle_center_radius(state, "A(3)")
    view = view_bounds(state)
    sides = [distance(coords[i], coords[(i + 1) % 3]) for i in range(3)]
    q = _question
    return [
        q(
            scene,
            "tc_point_c",
            "coordinates",
            "What are the coordinates of point C? Answer as x, y.",
            "numbers",
            list(points["C"]),
        ),
        q(
            scene,
            "tc_len_bc",
            "measurement",
            "What is the length of segment BC?",
            "number",
            segment_length(state, "BC"),
        ),
        q(scene, "tc_area_abc", "measurement", "What is the area of triangle ABC?", "number", polygon_area(coords)),
        q(scene, "tc_perimeter_abc", "multi_hop", "What is the perimeter of triangle ABC?", "number", sum(sides)),
        q(
            scene,
            "tc_largest_angle",
            "multi_hop",
            "Which vertex of triangle ABC has the largest interior angle? Answer with the point name.",
            "name",
            [unique_extreme(interior_angles(state, vertices))],
        ),
        q(
            scene,
            "tc_on_circle",
            "relation",
            "Which named points lie exactly on circle A(3), that is on its circumference (not inside it and not its "
            "center)? List the point names separated by commas, or answer none.",
            "set",
            sorted(points_on_circle(state, "A(3)")),
            set_of="Points",
        ),
        q(scene, "tc_circle_area", "measurement", "What is the area of circle A(3)?", "number", math.pi * radius**2),
        q(
            scene,
            "tc_segment_count",
            "count",
            "How many segments are on the canvas?",
            "number",
            float(len(_items(state, "Segments"))),
            needs=_COUNT_NEEDS,
        ),
        q(
            scene,
            "tc_view_x",
            "view",
            "What are the left and right bounds of the visible x-range of the view? Answer as left, right.",
            "numbers",
            [view["left_bound"], view["right_bound"]],
        ),
    ]


def _mixed_medium_questions(scene: Scene) -> List[Question]:
    state = scene.state
    points = point_map(state)
    f_args = _args(_find(state, "Functions", "f"))
    e1_args = _args(_find(state, "ParametricFunctions", "e1"))
    ac_label = _args(_find(state, "Segments", "AC"))["label"]["text"]
    angles = interior_angles(state, triangle_vertices(state, "ABC"))
    right_vertex = [vertex for vertex, size in angles.items() if math.isclose(size, 90.0, abs_tol=1e-6)]
    assert len(right_vertex) == 1
    to_h = {name: distance(xy, points["H"]) for name, xy in points.items() if name != "H"}
    q = _question
    return [
        q(
            scene,
            "mm_point_j",
            "coordinates",
            "What are the coordinates of point J? Answer as x, y.",
            "numbers",
            list(points["J"]),
        ),
        q(
            scene,
            "mm_len_gh",
            "measurement",
            "What is the length of segment GH?",
            "number",
            segment_length(state, "GH"),
        ),
        q(
            scene,
            "mm_angle_bac",
            "measurement",
            "What is the size of the angle object angle_BAC, in degrees?",
            "number",
            angle_degrees(state, "angle_BAC"),
        ),
        q(
            scene,
            "mm_f_expression",
            "function",
            "What is the expression of function f? Answer with the right-hand side only.",
            "expression",
            [str(f_args["function_string"])],
        ),
        q(
            scene,
            "mm_f_domain",
            "function",
            "Over which x-interval is function f drawn? Answer as left, right.",
            "numbers",
            [float(f_args["left_bound"]), float(f_args["right_bound"])],
        ),
        q(
            scene,
            "mm_point_count",
            "count",
            "How many points are on the canvas?",
            "number",
            float(len(point_list(state))),
            needs=_COUNT_NEEDS,
        ),
        q(scene, "mm_e1_color", "style", "What color is the parametric curve e1?", "name", [str(e1_args["color"])]),
        q(scene, "mm_ac_label", "style", "What is the label text of segment AC?", "name", [str(ac_label)]),
        q(
            scene,
            "mm_right_angle",
            "multi_hop",
            "At which vertex does triangle ABC have its right angle?",
            "name",
            right_vertex,
        ),
        q(
            scene,
            "mm_closest_to_h",
            "multi_hop",
            "Which named point (not label) is closest to point H (not counting H itself)?",
            "name",
            [unique_extreme(to_h, largest=False)],
        ),
    ]


def _weighted_graph_questions(scene: Scene) -> List[Question]:
    state = scene.state
    points = point_map(state)
    edges = graph_edges(state, "UndirectedGraphs", "G1")
    weights = {frozenset(edge.ends): edge.weight for edge in edges}
    touching_d = {edge.name: edge.weight for edge in edges if "D" in edge.ends}
    lightest_d = next(edge for edge in edges if edge.name == unique_extreme(touching_d, largest=False))
    neighbours_f = {end for edge in edges if "F" in edge.ends for end in edge.ends if end != "F"}
    q = _question
    return [
        q(scene, "wg_edge_count", "graph", "How many edges does graph G1 have?", "number", float(len(edges))),
        q(
            scene,
            "wg_weight_bd",
            "graph",
            "What is the weight of the edge between B and D in graph G1?",
            "number",
            weights[frozenset(("B", "D"))],
        ),
        q(
            scene,
            "wg_degree_c",
            "graph",
            "How many edges of graph G1 touch vertex C?",
            "number",
            float(sum(1 for edge in edges if "C" in edge.ends)),
        ),
        q(
            scene,
            "wg_neighbours_f",
            "graph",
            "Which vertices of graph G1 are adjacent to vertex F? List their names separated by commas.",
            "set",
            sorted(neighbours_f),
            set_of="Points",
        ),
        q(
            scene,
            "wg_total_weight",
            "graph",
            "What is the total weight of all edges of graph G1?",
            "number",
            sum(edge.weight for edge in edges),
        ),
        q(
            scene,
            "wg_lightest_edge_d",
            "multi_hop",
            "Which edge of graph G1 touching vertex D has the smallest weight? Answer with its two endpoint names "
            "as X-Y.",
            "edge",
            sorted(lightest_d.ends),
        ),
        q(
            scene,
            "wg_shortest_a_f",
            "multi_hop",
            "What is the length (total edge weight) of the shortest path from A to F in graph G1?",
            "number",
            shortest_path_length(edges, "A", "F"),
        ),
        q(
            scene,
            "wg_point_b",
            "coordinates",
            "What are the coordinates of point B? Answer as x, y.",
            "numbers",
            list(points["B"]),
        ),
        q(
            scene,
            "wg_distance_ac",
            "measurement",
            "What is the Euclidean distance between points A and C on the canvas (not the edge weight)?",
            "number",
            distance(points["A"], points["C"]),
        ),
    ]


def _regression_duplicates_questions(scene: Scene) -> List[Question]:
    state = scene.state
    points = point_list(state)
    fit_args = _args(_find(state, "Functions", "fit1"))
    slope, intercept = linear_coefficients(str(fit_args["function_string"]))
    bars = _args(_find(state, "BarsPlots", "sales"))
    values = [float(v) for v in bars["values"]]
    by_label = dict(zip([str(label) for label in bars["labels_below"]], values))
    highest = max(points, key=lambda item: item[1][1])
    q = _question
    return [
        q(
            scene,
            "rd_point_count",
            "count",
            "How many points are on the canvas?",
            "number",
            float(len(points)),
            needs=_COUNT_NEEDS,
        ),
        q(
            scene,
            "rd_named_f_count",
            "count",
            "How many points have exactly the name F (no prime marks)?",
            "number",
            float(sum(1 for name, _ in points if name == "F")),
        ),
        q(scene, "rd_fit_slope", "function", "What is the slope of the line fit1?", "number", slope),
        q(scene, "rd_fit_intercept", "function", "What is the y-intercept of the line fit1?", "number", intercept),
        q(
            scene,
            "rd_fit_domain",
            "function",
            "Over which x-interval is fit1 drawn? Answer as left, right.",
            "numbers",
            [float(fit_args["left_bound"]), float(fit_args["right_bound"])],
        ),
        q(
            scene,
            "rd_bar_wed",
            "chart",
            "What is the value of the bar labelled Wed in the bar chart sales?",
            "number",
            by_label["Wed"],
        ),
        q(
            scene,
            "rd_bar_max",
            "chart",
            "Which label of the bar chart sales has the largest value?",
            "name",
            [unique_extreme(by_label)],
        ),
        q(scene, "rd_bar_sum", "chart", "What is the sum of all values in the bar chart sales?", "number", sum(values)),
        q(
            scene,
            "rd_highest_point",
            "multi_hop",
            "What are the coordinates of the point with the largest y-coordinate? Answer as x, y.",
            "numbers",
            list(highest[1]),
        ),
        q(
            scene,
            "rd_view_top",
            "view",
            "What is the top bound of the visible y-range of the view?",
            "number",
            view_bounds(state)["top_bound"],
        ),
    ]


def _change_questions(scene: Scene) -> List[Question]:
    assert scene.after is not None
    before, after = scene.state, scene.after
    diff = canvas_diff(before, after)
    assert len(diff.removed) == 1 and len(diff.added) == 1 and len(diff.moved_points) == 1, diff
    removed_name = diff.removed[0][1]
    added_bucket, added_name = diff.added[0]
    assert added_bucket == "Circles"
    moved_name, (old_xy, new_xy) = next(iter(diff.moved_points.items()))
    _, radius = circle_center_radius(after, added_name)
    q = _question
    return [
        q(
            scene,
            "ch_removed",
            "change",
            "Which object did your edits remove? Answer with its name.",
            "name",
            [removed_name],
            needs=[{INFO_CHANGES}, {INFO_CANVAS, INFO_TOOL_ARGS}],
        ),
        q(
            scene,
            "ch_moved_point",
            "change",
            "Which point did your edits move? Answer with its name.",
            "name",
            [moved_name],
            needs=[{INFO_TOOL_ARGS}],
        ),
        q(
            scene,
            "ch_old_position",
            "change",
            f"Where was point {moved_name} before your edits? Answer as x, y.",
            "numbers",
            list(old_xy),
            needs=[{INFO_CANVAS}, {INFO_CHANGES}],
        ),
        q(
            scene,
            "ch_new_position",
            "change",
            f"Where is point {moved_name} now? Answer as x, y.",
            "numbers",
            list(new_xy),
            needs=[{INFO_TOOL_ARGS}],
            after_arrow=True,
        ),
        q(
            scene,
            "ch_added_name",
            "change",
            "What is the name of the circle your edits added?",
            "name",
            [added_name],
            needs=[{INFO_CHANGES}],
        ),
        q(
            scene,
            "ch_on_new_circle",
            "relation",
            "Which named points lie exactly on the circumference of the circle your edits added (not its center)? "
            "List the point names separated by commas, or answer none.",
            "set",
            sorted(points_on_circle(after, added_name)),
            needs=[{INFO_CHANGES}, {INFO_CANVAS, INFO_TOOL_ARGS}],
            set_of="Points",
        ),
        q(
            scene,
            "ch_new_circle_area",
            "measurement",
            "What is the area of the circle your edits added?",
            "number",
            math.pi * radius**2,
            needs=[{INFO_TOOL_ARGS}],
        ),
        q(
            scene,
            "ch_segments_now",
            "count",
            "Which segments are on the canvas after your edits? List their names separated by commas.",
            "set",
            sorted(item["name"] for item in _items(after, "Segments")),
            needs=[{INFO_CANVAS, INFO_TOOL_ARGS}, {INFO_CANVAS, INFO_CHANGES}],
            set_of="Segments",
        ),
    ]


_QUESTION_BUILDERS: Dict[str, Callable[[Scene], List[Question]]] = {
    "triangle_circle": _triangle_circle_questions,
    "mixed_medium": _mixed_medium_questions,
    "weighted_graph": _weighted_graph_questions,
    "regression_duplicates": _regression_duplicates_questions,
    CHANGE_SCENE: _change_questions,
}


def build_questions(scenes: Sequence[Scene]) -> List[Question]:
    questions: List[Question] = []
    for scene in scenes:
        questions.extend(_QUESTION_BUILDERS[scene.name](scene))
    return questions


# --------------------------------------------------------------------------- grading (pure)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_LINE = re.compile(r"^[\s*_>#-]*(?:final\s+)?answer[\s*_]*:[\s*_]*(?P<value>.*?)[\s*_]*$", re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\s*\{")
_ARROW = re.compile(r"->|→|=>|⟶")
# A set answer ends at an explanation: " (", ";", " because", " since" (a name such as A(5) keeps its parentheses).
_SET_ANSWER_END = re.compile(r"\s\(|;|\s(?:because|since)\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![A-Za-z_.\d])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_NAME_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_']*(?:\([^()\s]*\))?")
_NONE_ANSWER = re.compile(r"^\s*(?:none|no points?|nothing|empty|\{\s*\}|∅)\b", re.IGNORECASE)
# Anywhere in an answer that lists no names: "There are none", "No point lies on it", "∅".
_SAYS_NONE = re.compile(r"\b(?:none|no points?|nothing|empty)\b|\{\s*\}|∅", re.IGNORECASE)
_NAME_PREFIX = re.compile(
    r"^(?:at\s+)?(?:the\s+)?(?:(?:point|segment|circle|vertex|object|curve|label|colou?r|day|bar|edge|text)\s+)?",
    re.IGNORECASE,
)
_EXPRESSION_PREFIX = re.compile(r"^(?:[a-z]\w*\(x\)|y)=")
# Numbers: relative 2e-3 or absolute 5e-3. The text format shows 6 significant
# digits, so an answer read off the canvas is within ~5e-6; the relative bound
# still accepts 3-4 significant figures (281.7 for 281.71) and the absolute one
# any rounding to 2 decimals (3.33 for 10/3, values near 0), while whole-number
# guesses such as 20 for 20.14 or 280 for 281.71 fail.
NUMBER_RELATIVE_TOLERANCE = 2e-3
NUMBER_ABSOLUTE_TOLERANCE = 5e-3


def extract_answer(reply: str) -> Tuple[str, bool]:
    """The value of the reply's last ``Answer:`` (or ``Final answer:``) line, and whether one was present.

    Without one, the last non-empty line is used. An answer line with no value
    (``**Final Answer:**`` followed by ``$\\boxed{6}$``) takes the next non-empty
    line. Markdown emphasis, backticks, ``$``, ``\\boxed{...}`` and ``<think>``
    blocks are ignored.
    """
    text = _THINK_BLOCK.sub("", reply or "")
    lines = [line for line in text.splitlines() if line.strip()]
    for index in range(len(lines) - 1, -1, -1):
        match = _ANSWER_LINE.match(lines[index])
        if match:
            value = _clean_answer(match.group("value"))
            if not value and index + 1 < len(lines):
                value = _clean_answer(lines[index + 1])
            return value, True
    return (_clean_answer(lines[-1]) if lines else ""), False


def _clean_answer(value: str) -> str:
    value = _unbox(value)
    for token in ("**", "`", "$", "\\(", "\\)"):
        value = value.replace(token, "")
    return value.replace("\u2212", "-").strip()


def _unbox(value: str) -> str:
    """Replace each ``\\boxed{...}`` with its content (nested braces allowed)."""
    while True:
        match = _BOXED.search(value)
        if match is None:
            return value
        depth, end = 1, match.end()
        while end < len(value) and depth:
            depth += {"{": 1, "}": -1}.get(value[end], 0)
            end += 1
        inner = value[match.end() : end - 1] if depth == 0 else value[match.end() :]
        value = value[: match.start()] + inner + value[end:]


def parse_numbers(text: str) -> List[float]:
    return [float(match) for match in _NUMBER.findall(text.replace("\u2212", "-"))]


def numbers_close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=NUMBER_RELATIVE_TOLERANCE, abs_tol=NUMBER_ABSOLUTE_TOLERANCE)


def _grade_number(answer: str, expected: float) -> bool:
    candidates = parse_numbers(answer)[:1]
    for marker in ("≈", "="):
        if marker in answer:
            candidates += parse_numbers(answer.rsplit(marker, 1)[1])[:1]
    return any(numbers_close(value, expected) for value in candidates)


def _grade_numbers(answer: str, expected: Sequence[float]) -> bool:
    values = parse_numbers(answer)
    if len(values) < len(expected):
        return False
    return all(numbers_close(value, target) for value, target in zip(values, expected))


def _name_candidates(answer: str, keep_parentheses: bool) -> Set[str]:
    text = answer.strip().rstrip(".").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    text = text.strip('"*').strip()
    stripped = _NAME_PREFIX.sub("", text).strip()
    candidates = {text, stripped}
    if not keep_parentheses:
        candidates.add(re.sub(r"\s*\(.*\)\s*$", "", stripped))
    words = re.split(r"[\s,;:]+", stripped)
    if words and words[0]:
        candidates.add(words[0].rstrip("."))
    return {candidate.strip().strip('"').lower() for candidate in candidates if candidate.strip()}


def _grade_name(answer: str, accepted: Sequence[str]) -> bool:
    keep_parentheses = any("(" in name for name in accepted)
    return bool(_name_candidates(answer, keep_parentheses) & {name.lower() for name in accepted})


def parse_name_set(answer: str, vocabulary: Sequence[str]) -> Set[str]:
    """Names of the scene mentioned in a free-form list answer ("none" gives the empty set).

    Multi-letter names match case-insensitively; one-letter names must match
    exactly, so the article "a" is not read as point A.
    """
    if _NONE_ANSWER.match(answer):
        return set()
    by_lower = {name.lower(): name for name in vocabulary}
    found: Set[str] = set()
    for token in _NAME_TOKEN.findall(answer):
        if token in vocabulary:
            found.add(token)
        elif len(token) > 1 and token.lower() in by_lower:
            found.add(by_lower[token.lower()])
        elif len(token) == 2 and token[::-1] in vocabulary:
            found.add(token[::-1])  # segment CA is segment AC
    return found


def _grade_set(answer: str, expected: Sequence[str], vocabulary: Sequence[str]) -> bool:
    end = _SET_ANSWER_END.search(answer)
    listed = answer[: end.start()] if end is not None and end.start() > 0 else answer
    found = parse_name_set(listed, vocabulary)
    if not found and not expected:
        # Naming no point counts as "none" only when the answer says so: "unknown" or a cut-off reply is wrong.
        return bool(_SAYS_NONE.search(listed))
    return {name.lower() for name in found} == {name.lower() for name in expected}


def _grade_edge(answer: str, expected: Sequence[str]) -> bool:
    letters = {letter for token in re.findall(r"\b[A-Z]{1,2}\b", answer) for letter in token}
    return letters == set(expected)


def normalize_expression(expression: str) -> str:
    text = expression.lower().replace("²", "^2").replace("³", "^3").replace("**", "^").replace("\u2212", "-")
    text = re.sub(r"\s+", "", text).rstrip(".")
    return _EXPRESSION_PREFIX.sub("", text)


def _grade_expression(answer: str, accepted: Sequence[str]) -> bool:
    return normalize_expression(answer) in {normalize_expression(expression) for expression in accepted}


def grade(question: Question, answer: str) -> bool:
    """True when ``answer`` (the extracted answer value) matches the question's expected answer."""
    if not answer:
        return False
    if question.kind == "number":
        return _grade_number(answer, float(question.expected))
    if question.kind == "numbers":
        if question.after_arrow:
            answer = _ARROW.split(answer)[-1]
        return _grade_numbers(answer, [float(v) for v in question.expected])
    if question.kind == "name":
        return _grade_name(answer, question.expected)
    if question.kind == "set":
        return _grade_set(answer, question.expected, question.vocabulary)
    if question.kind == "edge":
        return _grade_edge(answer, question.expected)
    if question.kind == "expression":
        return _grade_expression(answer, question.expected)
    raise ValueError(f"unknown answer kind {question.kind!r}")


# --------------------------------------------------------------------------- what each format sends

_STATIC_INFO = frozenset({INFO_CANVAS, INFO_COUNTS})
_CHANGE_INFO = frozenset({INFO_CANVAS, INFO_COUNTS, INFO_CHANGES, INFO_TOOL_ARGS})


def provided_info(provider: str, canvas_format: str, change_scene: bool) -> FrozenSet[str]:
    """The information (``INFO_*``) a request of this provider and format carries.

    text / min_json: the canvas with the question; after a tool batch the tool
    calls plus ``[canvas changes]``. OpenRouter json: the prompt JSON with the
    canvas; after a tool batch only the tool calls (the app strips the state from
    history after the first completion). LocalAgent json: an object-count line
    only, and after a tool batch the tool calls.
    """
    if canvas_format != "json":
        return _CHANGE_INFO if change_scene else _STATIC_INFO
    if change_scene:
        return frozenset({INFO_TOOL_ARGS, INFO_COUNTS}) if provider == "local" else frozenset({INFO_TOOL_ARGS})
    return frozenset({INFO_COUNTS}) if provider == "local" else _STATIC_INFO


def is_provided(provider: str, canvas_format: str, question: Question) -> bool:
    """Whether the format sends enough information to answer ``question``."""
    info = provided_info(provider, canvas_format, question.scene == CHANGE_SCENE)
    return any(option <= info for option in question.needs)


def format_label(provider: str, canvas_format: str) -> str:
    """Format name for reports (LocalAgent's json is only a count line)."""
    if provider == "local" and canvas_format == "json":
        return "json (count line only, legacy LocalAgent)"
    return canvas_format


FORMAT_DESCRIPTIONS: Dict[Tuple[str, str], str] = {
    ("openrouter", "text"): "a `<canvas>` block with one object per line and computed facts in the user message; "
    "after a tool batch, `[canvas changes]` in the last tool result",
    ("openrouter", "min_json"): "the `<canvas>` block as compact JSON; after a tool batch, `[canvas changes]`",
    ("openrouter", "json"): "the whole prompt JSON with `canvas_state`; after a tool batch no canvas at all "
    "(the state is stripped from history), only the tool calls and their results",
    ("local", "text"): "a `<canvas>` block with one object per line and computed facts in front of the user text; "
    "after a tool batch, `[canvas changes]` in the last tool result",
    ("local", "min_json"): "the `<canvas>` block as compact JSON (the real structured JSON LocalAgent can send); "
    "after a tool batch, `[canvas changes]`",
    ("local", "json"): "the user text plus a `[Canvas: 3 Points, ...]` object-count line only (legacy LocalAgent); "
    "after a tool batch, only the tool calls and their results",
}


# --------------------------------------------------------------------------- prompts (the app's own code)


@contextmanager
def prompt_environment(
    canvas_format: CanvasFormat,
    canvas_budget: Optional[int],
    placeholder_key: bool,
    local_reasoning_effort: Optional[str] = None,
) -> Iterator[None]:
    """Pin the provider settings read while prompts are built, restoring the environment afterwards.

    ``placeholder_key`` sets a dummy OPENROUTER_API_KEY when none is set, so a dry run
    can construct the provider without loading any .env file.
    """
    overrides: Dict[str, Optional[str]] = {name: None for name in _PINNED_ENV_VARS}
    overrides[CANVAS_FORMAT_ENV] = canvas_format
    if canvas_budget is not None:
        overrides[CANVAS_BUDGET_ENV] = str(canvas_budget)
    if local_reasoning_effort is not None:
        overrides[REASONING_EFFORT_ENV] = local_reasoning_effort
    if placeholder_key and not os.environ.get(_OPENROUTER_KEY_ENV):
        overrides[_OPENROUTER_KEY_ENV] = "dry-run-placeholder"
    saved = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            _set_env(name, value)
        yield
    finally:
        for name, value in saved.items():
            _set_env(name, value)


def _set_env(name: str, value: Optional[str]) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def create_provider(provider: str, model_id: str) -> Provider:
    """The provider instance the app would use for ``model_id`` (default tool mode, no custom tools)."""
    model = AIModel.from_identifier(model_id)
    if provider == "openrouter":
        return OpenRouterAPI(model=model, tool_mode="search")
    if provider == "local":
        return LocalAgentAPI(model=model)
    raise ValueError(f"unknown provider {provider!r}")


def user_prompt_json(user_message: Optional[str], state: Mapping[str, Any], model_id: str, **extra: Any) -> str:
    """The prompt JSON the browser client sends (static/client/ai_interface.py ``_send_prompt_to_ai``)."""
    prompt: JsonDict = {
        "canvas_state": state,
        "user_message": user_message,
        "tool_call_results": None,
        "use_vision": False,
        "ai_model": model_id,
    }
    prompt.update(extra)
    return json.dumps(prompt)


def _send_prompt(api: Provider, full_prompt: str) -> None:
    """Add a prompt to the history the way the provider's create_chat_completion_stream does."""
    if isinstance(api, OpenRouterAPI):
        api._prepare_messages_for_request(full_prompt)
        return
    message = api._parse_and_prepare_message(full_prompt)
    if message is not None:
        api.messages.append(message)


def change_tool_calls(before: Mapping[str, Any], after: Mapping[str, Any]) -> List[JsonDict]:
    """The tool calls a model would make to turn ``before`` into ``after`` (streamed-call shape)."""
    diff = canvas_diff(before, after)
    calls: List[Tuple[str, JsonDict]] = []
    for name, (_, (x, y)) in diff.moved_points.items():
        calls.append(
            ("update_point", {"point_name": name, "new_name": None, "new_x": x, "new_y": y, "new_color": None})
        )
    for bucket, name in diff.removed:
        assert bucket == "Segments", f"unsupported removal: {bucket}"
        p1, p2 = segment_endpoints(before, name)
        (x1, y1), (x2, y2) = point_map(before)[p1], point_map(before)[p2]
        calls.append(("delete_segment", {"x1": x1, "y1": y1, "x2": x2, "y2": y2}))
    for bucket, name in diff.added:
        assert bucket == "Circles", f"unsupported addition: {bucket}"
        center, radius = circle_center_radius(after, name)
        cx, cy = point_map(after)[center]
        calls.append(("create_circle", {"center_x": cx, "center_y": cy, "radius": radius, "color": None, "name": None}))
    return [
        {"id": f"call_{index}", "function": {"name": function, "arguments": json.dumps(_plain_numbers(arguments))}}
        for index, (function, arguments) in enumerate(calls)
    ]


def _plain_numbers(arguments: Mapping[str, Any]) -> JsonDict:
    """Whole floats as ints (``-1`` rather than ``-1.0``), as a model writes them."""
    return {k: int(v) if isinstance(v, float) and v.is_integer() else v for k, v in arguments.items()}


def tool_results_prompt(calls: Sequence[JsonDict], state_after: Mapping[str, Any], model_id: str) -> str:
    """The prompt the client sends after running a tool batch (per-call results plus the new state).

    Result keys follow the client's ResultProcessor.generate_result_key
    (``name(arg:value, ...)``); every call reports the client's success message.
    """
    entries = []
    for call in calls:
        function = call["function"]
        arguments = json.loads(function["arguments"])
        key = f"{function['name']}({', '.join(f'{k}:{v}' for k, v in arguments.items())})"
        entries.append({"tool_call_id": call["id"], "result": {key: successful_call_message}})
    return json.dumps(
        {
            "canvas_state": state_after,
            "user_message": None,
            "tool_call_results": json.dumps(entries),
            "use_vision": False,
            "ai_model": model_id,
        }
    )


def question_user_text(question: Question, change_turn: bool) -> str:
    text = f"{question.text}\n\n{ANSWER_INSTRUCTIONS}"
    return f"{CHANGE_TURN_INTRO}\n\nQuestion: {text}" if change_turn else text


def build_conversation(api: Provider, scene: Scene, question: Question, model_id: str) -> List[JsonDict]:
    """The messages the app sends for ``question``, built by the provider's own methods.

    Static scenes: one user message carrying the canvas. The change scene: the
    continuation request after a tool batch (user turn with the old canvas, the
    assistant's tool calls, their results, plus ``[canvas changes]`` in the
    non-json formats).
    """
    api.reset_conversation()
    if scene.after is None:
        _send_prompt(api, user_prompt_json(question_user_text(question, False), scene.state, model_id))
        return copy.deepcopy(api.messages)
    _send_prompt(api, user_prompt_json(question_user_text(question, True), scene.state, model_id))
    calls = change_tool_calls(scene.state, scene.after)
    api._finalize_stream("", calls)
    _send_prompt(api, tool_results_prompt(calls, scene.after, model_id))
    return copy.deepcopy(api.messages)


def estimate_prompt_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Estimated prompt tokens (static/token_estimation.py): message text plus tool-call names and arguments."""
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += estimate_tokens_from_text(content)
        elif isinstance(content, list):
            total += sum(
                estimate_tokens_from_text(str(part.get("text", ""))) for part in content if isinstance(part, dict)
            )
        for call in message.get("tool_calls") or []:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            total += estimate_tokens_from_text(f"{function.get('name', '')}{function.get('arguments', '')}")
    return total


@dataclass
class PreparedRequest:
    model: str
    canvas_format: CanvasFormat
    question: Question
    messages: List[JsonDict]
    estimated_prompt_tokens: int
    request_kwargs: JsonDict
    provided: bool = True


def request_kwargs(api: Provider) -> JsonDict:
    """Request parameters the provider sends (no tools).

    LocalAgent: its own ``_completion_options`` (temperature, max_tokens and the
    reasoning effort in ``chat_template_kwargs``). OpenRouter: max_tokens only.
    """
    if isinstance(api, LocalAgentAPI):
        return copy.deepcopy(api._completion_options())
    return {"max_tokens": api.max_tokens}


def build_requests(
    provider: str,
    models: Sequence[str],
    formats: Sequence[CanvasFormat],
    scenes: Sequence[Scene],
    questions: Sequence[Question],
    canvas_budget: Optional[int],
    dry_run: bool,
    local_reasoning_effort: Optional[str] = None,
) -> Tuple[List[PreparedRequest], Dict[Tuple[str, str], Provider]]:
    """Build every (model, format, question) request, and the provider instance per (model, format)."""
    scene_by_name = {scene.name: scene for scene in scenes}
    requests: List[PreparedRequest] = []
    providers: Dict[Tuple[str, str], Provider] = {}
    for model in models:
        for canvas_format in formats:
            with prompt_environment(canvas_format, canvas_budget, dry_run, local_reasoning_effort):
                api = create_provider(provider, model)
                providers[(model, canvas_format)] = api
                for question in questions:
                    messages = build_conversation(api, scene_by_name[question.scene], question, model)
                    requests.append(
                        PreparedRequest(
                            model=model,
                            canvas_format=canvas_format,
                            question=question,
                            messages=messages,
                            estimated_prompt_tokens=estimate_prompt_tokens(messages),
                            request_kwargs=request_kwargs(api),
                            provided=is_provided(provider, canvas_format, question),
                        )
                    )
    return requests, providers


# --------------------------------------------------------------------------- live requests


def send_request(client: Any, provider: str, request: PreparedRequest, repeat: int) -> JsonDict:
    """Send one request (plain chat completion, no tools) and grade the reply. Never raises."""
    question = request.question
    tracker = ResponseMetricsTracker(provider=provider, model=request.model, api="chat_completions", streamed=False)
    reply = ""
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    try:
        response = client.chat.completions.create(
            model=request.model, messages=request.messages, **request.request_kwargs
        )
        record_chat_completions_usage(response, tracker)
        choice = response.choices[0]
        reply = str(getattr(choice.message, "content", None) or "")
        finish_reason = getattr(choice, "finish_reason", None)
    except Exception as exc:  # recorded per request; the run goes on
        error = f"{type(exc).__name__}: {exc}"
    metrics = tracker.finish("error" if error else finish_reason, 0, error=error)
    answer, has_answer_line = extract_answer(reply)
    return {
        "model": request.model,
        "format": request.canvas_format,
        "scene": question.scene,
        "qid": question.qid,
        "category": question.category,
        "repeat": repeat,
        "question": question.text,
        "kind": question.kind,
        "expected": question.expected,
        "provided": request.provided,
        "reply": reply,
        "answer": answer,
        "answer_line": has_answer_line,
        "correct": error is None and grade(question, answer),
        "estimated_prompt_tokens": request.estimated_prompt_tokens,
        "prompt_tokens": metrics.get("prompt_tokens"),
        "completion_tokens": metrics.get("completion_tokens"),
        "reasoning_tokens": metrics.get("reasoning_tokens"),
        "latency_s": metrics.get("total_latency_s"),
        "output_tokens_per_s": metrics.get("output_tokens_per_s"),
        "server_timings": metrics.get("server_timings"),
        "finish_reason": metrics.get("finish_reason"),
        "error": error,
    }


def order_jobs(requests: Sequence[PreparedRequest], repeats: int) -> List[Tuple[PreparedRequest, int]]:
    """Jobs question-major, models and formats interleaved within each question, repeats last.

    Drift over the run (provider load, a server warming up) then spreads over
    every format instead of lining up with one of them.
    """
    positions: Dict[str, Dict[str, int]] = {"qid": {}, "model": {}, "format": {}}
    for request in requests:
        positions["qid"].setdefault(request.question.qid, len(positions["qid"]))
        positions["model"].setdefault(request.model, len(positions["model"]))
        positions["format"].setdefault(request.canvas_format, len(positions["format"]))
    ordered = sorted(
        requests,
        key=lambda r: (
            positions["qid"][r.question.qid],
            positions["model"][r.model],
            positions["format"][r.canvas_format],
        ),
    )
    return [(request, repeat) for repeat in range(repeats) for request in ordered]


class ResultSink:
    """Collects results from worker threads, appending each to a JSON-lines file as it arrives."""

    def __init__(self, jsonl_path: Optional[Path] = None) -> None:
        self._results: List[JsonDict] = []
        self._lock = threading.Lock()
        self._path = jsonl_path
        if jsonl_path is not None:
            jsonl_path.write_text("", encoding="utf-8")

    def add(self, result: JsonDict) -> None:
        with self._lock:
            self._results.append(result)
            if self._path is not None:
                with open(self._path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    def results(self) -> List[JsonDict]:
        """The results so far, in job order."""
        with self._lock:
            return sorted(self._results, key=lambda result: int(result.get("job", 0)))


def run_live(
    provider: str,
    requests: Sequence[PreparedRequest],
    clients: Mapping[Tuple[str, str], Any],
    repeats: int,
    concurrency: int,
    sink: Optional[ResultSink] = None,
    log: Callable[[str], None] = print,
) -> List[JsonDict]:
    """Send every job (see order_jobs) and return the results in job order.

    Each result goes to ``sink`` as soon as it arrives. On an interruption the
    queued jobs are dropped (requests in flight still finish into the sink).
    """
    jobs = order_jobs(requests, repeats)
    sink = sink if sink is not None else ResultSink()

    def run(index: int) -> None:
        request, repeat = jobs[index]
        result = send_request(clients[(request.model, request.canvas_format)], provider, request, repeat)
        result["job"] = index
        sink.add(result)
        status = "ERROR" if result["error"] else ("ok" if result["correct"] else "wrong")
        if not request.provided:
            status += " (not provided)"
        log(
            f"[{index + 1}/{len(jobs)}] {request.model} {request.canvas_format} {request.question.qid}: "
            f"{status} ({result['latency_s']}s)"
        )

    pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
    try:
        for future in as_completed([pool.submit(run, index) for index in range(len(jobs))]):
            future.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return sink.results()


# --------------------------------------------------------------------------- re-grading


def regrade_results(results: Sequence[Mapping[str, Any]], provider: str) -> List[JsonDict]:
    """Stored results re-graded with the current grader (answer extraction, grading, "provided").

    Results whose question no longer exists keep their stored grade.
    """
    questions = {question.qid: question for question in build_questions(load_scenes())}
    regraded: List[JsonDict] = []
    for stored in results:
        result = dict(stored)
        question = questions.get(str(result.get("qid")))
        if question is not None:
            answer, has_answer_line = extract_answer(str(result.get("reply") or ""))
            result["answer"] = answer
            result["answer_line"] = has_answer_line
            result["correct"] = not result.get("error") and grade(question, answer)
            result["provided"] = is_provided(provider, str(result.get("format")), question)
        regraded.append(result)
    return regraded


def regrade_file(results_path: Path, out_dir: Optional[Path]) -> Tuple[Path, str]:
    """Re-grade a results.json; writes results_regraded.json and summary_regraded.md (default: next to it)."""
    data = json.loads(results_path.read_text(encoding="utf-8"))
    config: JsonDict = dict(data.get("config") or {})
    config["regraded_at"] = datetime.now().isoformat(timespec="seconds")
    config["regraded_from"] = str(results_path)
    results = regrade_results(data.get("results") or [], str(config.get("provider", "openrouter")))
    rows = summarize(results, config.get("models"), config.get("formats"))
    markdown = render_markdown(config, rows, results)
    target = out_dir or results_path.parent
    target.mkdir(parents=True, exist_ok=True)
    (target / "results_regraded.json").write_text(
        json.dumps({"config": config, "summary": rows, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (target / "summary_regraded.md").write_text(markdown, encoding="utf-8")
    return target, markdown


# --------------------------------------------------------------------------- summaries


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    numbers = [float(v) for v in values if isinstance(v, (int, float))]
    return round(sum(numbers) / len(numbers), 3) if numbers else None


def result_cost(result: Mapping[str, Any]) -> Optional[float]:
    prices = PRICES_PER_MTOK.get(str(result.get("model")))
    if prices is None:
        return None
    prompt, completion = result.get("prompt_tokens"), result.get("completion_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    return (prompt * prices[0] + completion * prices[1]) / 1_000_000


def _provided(result: Mapping[str, Any]) -> bool:
    return bool(result.get("provided", True))


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def outcome(items: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Accuracy over the questions the format provides; not-provided ones are counted apart.

    ``accuracy`` counts request errors as wrong answers; ``accuracy_excluding_errors``
    leaves errored requests out.
    """
    provided = [item for item in items if _provided(item)]
    not_provided = [item for item in items if not _provided(item)]
    correct = sum(1 for item in provided if item["correct"])
    errors = sum(1 for item in provided if item.get("error"))
    return {
        "correct": correct,
        "provided": len(provided),
        "errors": errors,
        "accuracy": _ratio(correct, len(provided)),
        "accuracy_excluding_errors": _ratio(correct, len(provided) - errors),
        "not_provided": len(not_provided),
        "not_provided_correct": sum(1 for item in not_provided if item["correct"]),
    }


def _order_key(values: Optional[Sequence[str]], value: str) -> Tuple[int, str]:
    listed = list(values or [])
    return (listed.index(value) if value in listed else len(listed), value)


def summarize(
    results: Sequence[Mapping[str, Any]],
    models: Optional[Sequence[str]] = None,
    formats: Optional[Sequence[str]] = None,
) -> List[JsonDict]:
    """Per model x format (in the given order): static-scene and change-scene outcomes, tokens, latency, cost.

    ``accuracy`` is the headline figure: static-scene accuracy over provided questions.
    """
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for result in results:
        groups.setdefault((str(result["model"]), str(result["format"])), []).append(result)
    rows: List[JsonDict] = []
    for model, canvas_format in sorted(
        groups, key=lambda key: (_order_key(models, key[0]), _order_key(formats, key[1]))
    ):
        items = groups[(model, canvas_format)]
        static = [item for item in items if item["scene"] != CHANGE_SCENE]
        costs = [result_cost(item) for item in items]
        static_outcome = outcome(static)
        rows.append(
            {
                "model": model,
                "format": canvas_format,
                "requests": len(items),
                "accuracy": static_outcome["accuracy"],
                "static": static_outcome,
                "change": outcome([item for item in items if item["scene"] == CHANGE_SCENE]),
                "by_scene": _outcome_by(items, "scene"),
                "by_category": _outcome_by(static, "category"),
                "change_by_question": _outcome_by([item for item in items if item["scene"] == CHANGE_SCENE], "qid"),
                "mean_prompt_tokens": _mean([item.get("prompt_tokens") for item in items]),
                "mean_estimated_prompt_tokens": _mean([item.get("estimated_prompt_tokens") for item in items]),
                "mean_completion_tokens": _mean([item.get("completion_tokens") for item in items]),
                "mean_latency_s": _mean([item.get("latency_s") for item in items]),
                "mean_output_tokens_per_s": _mean([item.get("output_tokens_per_s") for item in items]),
                "answer_line_rate": _ratio(sum(1 for item in items if item.get("answer_line")), len(items)),
                "errors": sum(1 for item in items if item.get("error")),
                "cost_usd": round(sum(c for c in costs if c is not None), 6)
                if any(c is not None for c in costs)
                else None,
            }
        )
    return rows


def _outcome_by(items: Sequence[Mapping[str, Any]], key: str) -> Dict[str, JsonDict]:
    buckets: Dict[str, List[Mapping[str, Any]]] = {}
    for item in items:
        buckets.setdefault(str(item[key]), []).append(item)
    return {name: outcome(group) for name, group in buckets.items()}


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _outcome_cell(result: Optional[Mapping[str, Any]]) -> str:
    """``correct/provided``, plus ``n/p`` (and how many of those were right anyway) for not-provided ones."""
    if result is None:
        return "-"
    parts = []
    if result["provided"]:
        parts.append(f"{result['correct']}/{result['provided']}")
    if result["not_provided"]:
        parts.append(f"n/p {result['not_provided_correct']}/{result['not_provided']}")
    return " ".join(parts) or "-"


def _summary_header(config: Mapping[str, Any]) -> List[str]:
    provider = str(config.get("provider", "openrouter"))
    formats = [str(f) for f in config.get("formats", [])]
    lines = [
        "# Canvas format benchmark",
        "",
        f"- Provider: {provider}; models: {', '.join(config.get('models', []))}",
        f"- Formats: {', '.join(format_label(provider, f) for f in formats)}; repeats: {config.get('repeats')}; "
        f"questions: {config.get('questions')}",
        f"- Run at {config.get('started_at')}; prices as of {config.get('prices_as_of', PRICES_AS_OF)} "
        "(USD per 1M tokens in/out)",
    ]
    if config.get("local_reasoning_effort") is not None or provider == "local":
        lines.append(f"- LocalAgent reasoning effort: {config.get('local_reasoning_effort') or 'not sent'}")
    if config.get("completed_requests") is not None and config.get("completed_requests") != config.get(
        "planned_requests"
    ):
        lines.append(
            f"- Interrupted: {config.get('completed_requests')} of {config.get('planned_requests')} requests completed"
        )
    if config.get("regraded_at"):
        lines.append(f"- Re-graded at {config['regraded_at']} with the current grader")
    lines += ["", "What each format sends:", ""]
    for canvas_format in formats:
        description = FORMAT_DESCRIPTIONS.get((provider, canvas_format), "")
        lines.append(f"- `{format_label(provider, canvas_format)}`: {description}")
    lines += [
        "",
        "Questions are asked without tools, one per request. Accuracy counts only questions whose information the "
        "format sends; the others are listed as n/p (not provided), with how many were answered right anyway. "
        "Accuracy counts request errors as wrong; 'excl. errors' leaves errored requests out.",
    ]
    return lines


def render_markdown(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> str:
    provider = str(config.get("provider", "openrouter"))

    def label(row: Mapping[str, Any]) -> str:
        return f"| {row['model']} | {format_label(provider, str(row['format']))} |"

    lines = _summary_header(config)
    for title, key in (("Static scenes (headline)", "static"), ("Change scene (after a tool batch)", "change")):
        lines += [
            "",
            f"## {title}",
            "",
            "| Model | Format | Accuracy | Correct | Excl. errors | Errors | Not provided (right anyway) |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            result = row[key]
            lines.append(
                f"{label(row)} {_pct(result['accuracy'])} | {result['correct']}/{result['provided']} "
                f"| {_pct(result['accuracy_excluding_errors'])} | {result['errors']} "
                f"| {result['not_provided']} ({result['not_provided_correct']}) |"
            )
    change_qids = [question.qid for question in build_questions(load_scenes([CHANGE_SCENE]))]
    if any(row["change_by_question"] for row in rows):
        lines += ["", "Change scene by question (correct/provided; n/p = not provided in the format):", ""]
        lines += ["| Model | Format | " + " | ".join(change_qids) + " |", "|---|---|" + "---|" * len(change_qids)]
        for row in rows:
            cells = [_outcome_cell(row["change_by_question"].get(qid)) for qid in change_qids]
            lines.append(f"{label(row)} " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Tokens, speed and cost (all requests)",
        "",
        "| Model | Format | Requests | Mean prompt tok | Mean est. prompt tok | Mean completion tok | Mean latency s "
        "| Output tok/s | Answer line | Errors | Cost USD |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"{label(row)} {row['requests']} | {_fmt(row['mean_prompt_tokens'])} "
            f"| {_fmt(row['mean_estimated_prompt_tokens'])} | {_fmt(row['mean_completion_tokens'])} "
            f"| {_fmt(row['mean_latency_s'], 2)} | {_fmt(row['mean_output_tokens_per_s'])} "
            f"| {_pct(row['answer_line_rate'])} | {row['errors']} | {_fmt(row['cost_usd'], 4)} |"
        )
    for title, key in (("Accuracy by scene", "by_scene"), ("Static accuracy by category", "by_category")):
        names = sorted({name for row in rows for name in row[key]})
        lines += ["", f"## {title}", "", "| Model | Format | " + " | ".join(names) + " |"]
        lines.append("|---|---|" + "---|" * len(names))
        for row in rows:
            lines.append(f"{label(row)} " + " | ".join(_outcome_cell(row[key].get(name)) for name in names) + " |")
    wrong = [result for result in results if not result["correct"] and _provided(result)]
    lines += ["", f"## Wrong answers to provided questions ({len(wrong)})", ""]
    if wrong:
        lines += ["| Model | Format | Question | Expected | Answer |", "|---|---|---|---|---|"]
        for result in wrong:
            answer = f"ERROR: {result['error']}" if result.get("error") else result["answer"]
            lines.append(
                f"| {result['model']} | {result['format']} | {result['qid']}: {_cell(result['question'])} "
                f"| {_cell(json.dumps(result['expected']))} | {_cell(answer)[:200]} |"
            )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- dry run


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "model"


def write_prompts(requests: Sequence[PreparedRequest], out_dir: Path) -> Path:
    prompts_dir = out_dir / "prompts"
    for request in requests:
        question = request.question
        path = prompts_dir / _slug(request.model) / request.canvas_format / f"{question.scene}__{question.qid}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": request.model,
            "format": request.canvas_format,
            "question": question.text,
            "kind": question.kind,
            "expected": question.expected,
            "provided": request.provided,
            "estimated_prompt_tokens": request.estimated_prompt_tokens,
            "request_kwargs": request.request_kwargs,
            "messages": request.messages,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return prompts_dir


def dry_run_report(requests: Sequence[PreparedRequest], repeats: int) -> Tuple[JsonDict, str]:
    """Token and cost estimates per model x format (and per scene), as data and printable text."""
    by_group: Dict[Tuple[str, str], List[PreparedRequest]] = {}
    for request in requests:
        by_group.setdefault((request.model, request.canvas_format), []).append(request)
    rows: List[JsonDict] = []
    for (model, canvas_format), items in by_group.items():
        tokens = [item.estimated_prompt_tokens for item in items]
        prices = PRICES_PER_MTOK.get(model)
        cost = None
        if prices is not None:
            cost = repeats * sum(t * prices[0] + ESTIMATED_COMPLETION_TOKENS * prices[1] for t in tokens) / 1_000_000
        per_scene: Dict[str, List[int]] = {}
        for item in items:
            per_scene.setdefault(item.question.scene, []).append(item.estimated_prompt_tokens)
        rows.append(
            {
                "model": model,
                "format": canvas_format,
                "requests": len(items) * repeats,
                "mean_prompt_tokens": round(sum(tokens) / len(tokens), 1),
                "min_prompt_tokens": min(tokens),
                "max_prompt_tokens": max(tokens),
                "total_prompt_tokens": sum(tokens) * repeats,
                "mean_prompt_tokens_by_scene": {s: round(sum(v) / len(v), 1) for s, v in per_scene.items()},
                "estimated_cost_usd": None if cost is None else round(cost, 4),
            }
        )
    lines = [
        "| Model | Format | Requests | Mean est. prompt tok | Min | Max | Total est. prompt tok | Est. cost USD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['format']} | {row['requests']} | {row['mean_prompt_tokens']} "
            f"| {row['min_prompt_tokens']} | {row['max_prompt_tokens']} | {row['total_prompt_tokens']} "
            f"| {_fmt(row['estimated_cost_usd'], 4)} |"
        )
    scenes = sorted({scene for row in rows for scene in row["mean_prompt_tokens_by_scene"]})
    lines += ["", "Mean estimated prompt tokens per request by scene:", ""]
    lines += ["| Model | Format | " + " | ".join(scenes) + " |", "|---|---|" + "---|" * len(scenes)]
    for row in rows:
        cells = [_fmt(row["mean_prompt_tokens_by_scene"].get(scene)) for scene in scenes]
        lines.append(f"| {row['model']} | {row['format']} | " + " | ".join(cells) + " |")
    costs = [row["estimated_cost_usd"] for row in rows if row["estimated_cost_usd"] is not None]
    if costs:
        lines += [
            "",
            f"Estimated total cost: ${sum(costs):.4f} (prices as of {PRICES_AS_OF}; assumes "
            f"{ESTIMATED_COMPLETION_TOKENS} completion tokens per request; the token heuristic counts one "
            "token per digit, so it overestimates cloud tokenizers)",
        ]
    return {"rows": rows, "estimated_total_cost_usd": round(sum(costs), 4) if costs else None}, "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canvas-format comprehension benchmark (json vs text canvas state).")
    parser.add_argument("--provider", choices=PROVIDERS, default="openrouter", help="openrouter (default) or local")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model ids. OpenRouter default: "
        + ", ".join(DEFAULT_OPENROUTER_MODELS)
        + "; local default: every model the llama-server serves.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=CANVAS_FORMATS,
        default=None,
        help="Canvas formats (default: json text for OpenRouter; text min_json json for local, where json is "
        "the legacy object-count line and min_json the structured format)",
    )
    parser.add_argument("--scenes", nargs="+", choices=SCENE_NAMES, default=list(SCENE_NAMES))
    parser.add_argument("--repeats", type=int, default=1, help="Times each question is asked (default 1)")
    parser.add_argument("--dry-run", action="store_true", help="Build and write prompts only; no network access")
    parser.add_argument(
        "--regrade",
        type=Path,
        default=None,
        metavar="RESULTS_JSON",
        help="Re-grade a results.json with the current grader and write summary_regraded.md (no network access)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default logs/canvas_format_benchmark/<time>; with --regrade, the results' directory)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=DEFAULT_MAX_REQUESTS,
        help=f"Abort before sending when the run needs more requests (default {DEFAULT_MAX_REQUESTS}); "
        "requests are never retried",
    )
    parser.add_argument(
        "--canvas-budget",
        type=int,
        default=None,
        help="MATHUD_CANVAS_BUDGET_TOKENS for the canvas block (default: provider default, 0 = unlimited)",
    )
    parser.add_argument(
        "--local-reasoning-effort",
        choices=LOCAL_REASONING_EFFORT_CHOICES,
        default=None,
        help="MATHUD_LOCAL_REASONING_EFFORT for LocalAgent requests (default: the app's default, medium; "
        "default sends no effort)",
    )
    parser.add_argument("--timeout", type=float, default=None, help="Per-request timeout in seconds")
    parser.add_argument("--concurrency", type=int, default=None, help="Parallel requests (default 4 cloud, 1 local)")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.local_reasoning_effort is not None and args.provider != "local":
        parser.error("--local-reasoning-effort applies to --provider local only")
    if args.formats is None:
        args.formats = list(DEFAULT_FORMATS[args.provider])
    return args


def resolve_models(args: argparse.Namespace) -> List[str]:
    """Models to benchmark; for a live local run without --models, every model the server serves."""
    if args.models:
        return list(dict.fromkeys(args.models))
    if args.provider == "openrouter":
        return list(DEFAULT_OPENROUTER_MODELS)
    if args.dry_run:
        return [LOCAL_PLACEHOLDER_MODEL]
    served = [str(info["name"]) for info in LocalAgentAPI.fetch_models()]
    if not served:
        raise SystemExit(f"No models served at {LocalAgentAPI.resolve_base_url()}/v1/models; is llama-server running?")
    return served


def _use_utf8_output() -> None:
    """Print answers with any characters on Windows consoles (cp1252 by default)."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def _write_live_results(out_dir: Path, config: JsonDict, results: List[JsonDict]) -> str:
    config["completed_requests"] = len(results)
    rows = summarize(results, config["models"], config["formats"])
    markdown = render_markdown(config, rows, results)
    (out_dir / "results.json").write_text(
        json.dumps({"config": config, "summary": rows, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(markdown, encoding="utf-8")
    return markdown


def main(argv: Optional[Sequence[str]] = None) -> int:
    _use_utf8_output()
    args = parse_args(argv)
    if args.regrade is not None:
        target, markdown = regrade_file(args.regrade, args.out)
        print(markdown)
        print(f"Re-graded summary written to {target}")
        return 0

    started_at = datetime.now()
    out_dir: Path = args.out or REPO_ROOT / "logs" / "canvas_format_benchmark" / started_at.strftime("%Y%m%d-%H%M%S")

    if not args.dry_run:
        # The app loads .env the same way (API key, LOCAL_AGENT_BASE_URL); values are never printed.
        env_config.load_env_files()
        if args.provider == "openrouter" and not os.environ.get(_OPENROUTER_KEY_ENV):
            print(
                f"Error: {_OPENROUTER_KEY_ENV} is not set (environment or .env); use --dry-run to only build prompts."
            )
            return 2

    models = resolve_models(args)
    scenes = load_scenes(args.scenes)
    questions = build_questions(scenes)
    formats: List[CanvasFormat] = list(dict.fromkeys(args.formats))
    planned = len(models) * len(formats) * len(questions) * args.repeats
    config: JsonDict = {
        "provider": args.provider,
        "models": models,
        "formats": formats,
        "scenes": [scene.name for scene in scenes],
        "questions": len(questions),
        "repeats": args.repeats,
        "planned_requests": planned,
        "canvas_budget": args.canvas_budget,
        "started_at": started_at.isoformat(timespec="seconds"),
        "prices_per_mtok": PRICES_PER_MTOK,
        "prices_as_of": PRICES_AS_OF,
    }
    mode = "dry run" if args.dry_run else "live"
    print(f"Canvas format benchmark ({mode}, provider {args.provider})")
    print(f"Models: {', '.join(models)}")
    print(
        f"Formats: {', '.join(format_label(args.provider, f) for f in formats)}; scenes: {len(scenes)}; "
        f"questions: {len(questions)}; repeats: {args.repeats}"
    )
    print(f"Planned requests: {planned} (max {args.max_requests})")
    if planned > args.max_requests and not args.dry_run:
        print(f"Aborting: {planned} requests exceed --max-requests {args.max_requests}. Nothing was sent.")
        return 2

    requests, providers = build_requests(
        args.provider, models, formats, scenes, questions, args.canvas_budget, args.dry_run, args.local_reasoning_effort
    )
    local_apis = [api for api in providers.values() if isinstance(api, LocalAgentAPI)]
    if local_apis:
        config["local_reasoning_effort"] = local_apis[0].reasoning_effort
        print(f"LocalAgent reasoning effort: {config['local_reasoning_effort'] or 'not sent'}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = write_prompts(requests, out_dir)

    if args.dry_run:
        report, text = dry_run_report(requests, args.repeats)
        if planned > args.max_requests:
            print(f"Warning: a live run would abort ({planned} requests > --max-requests {args.max_requests}).")
        print("\nEstimated prompt tokens (heuristic, static/token_estimation.py):\n")
        print(text)
        (out_dir / "dry_run.json").write_text(json.dumps({"config": config, **report}, indent=2), encoding="utf-8")
        print(f"\nPrompts written to {prompts_dir}")
        return 0

    timeout = args.timeout or DEFAULT_TIMEOUT_S[args.provider]
    concurrency = args.concurrency or DEFAULT_CONCURRENCY[args.provider]
    # No SDK retries: every request sent is one counted by --max-requests.
    clients = {key: api.client.with_options(timeout=timeout, max_retries=0) for key, api in providers.items()}
    sink = ResultSink(out_dir / "results.jsonl")
    try:
        run_live(args.provider, requests, clients, args.repeats, concurrency, sink)
    finally:
        # Also on Ctrl+C: keep every answer received so far.
        markdown = _write_live_results(out_dir, config, sink.results())
        print()
        print(markdown)
        print(f"Results written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
