"""Performance benchmark for the rendering and expression-evaluation hot paths.

Run by ``python -m cli.main test perf`` through ``window.startMatHudPerfBenchmark``.
It times draw, pan and zoom on a few fixed scenes (geometry, functions with
asymptotes, parametric curves, shaded areas and statistics plots) and a set of
expression micro-benchmarks (parsing, per-sample evaluation, math.js calls).

Each scene builds its own ``Canvas``, which draws on the page's surfaces over
whatever the page showed, so it is meant for a fresh headless page, not a
window in use.
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Callable, Dict, List, Optional

from browser import window

from client_tests.renderer_performance_tests import run_renderer_performance

_NO_GEOMETRY: Dict[str, Any] = {
    "points": 0,
    "segments": 0,
    "vectors": 0,
    "triangles": 0,
    "rectangles": 0,
    "circles": 0,
    "ellipses": 0,
    "functions": [],
}

SCENES: Dict[str, Dict[str, Any]] = {
    # Smaller than the test's default workload: building it is slow (each new shape scans
    # every triple of segments for triangles), and the build time per shape is measured too.
    "geometry": {
        "points": 30,
        "segments": 12,
        "vectors": 6,
        "triangles": 4,
        "rectangles": 2,
        "circles": 3,
        "ellipses": 2,
        "functions": [],
    },
    "functions": {
        **_NO_GEOMETRY,
        "functions": [
            {"expression": "sin(x)", "name": "f1", "left": -20, "right": 20},
            {"expression": "tan(x)", "name": "f2", "left": -20, "right": 20},
            {"expression": "1/x", "name": "f3", "left": -20, "right": 20},
            {"expression": "sin(1/x)", "name": "f4", "left": -5, "right": 5},
            {"expression": "x^3 - 4*x", "name": "f5", "left": -20, "right": 20},
            {"expression": "exp(x)/x", "name": "f6", "left": -10, "right": 10},
        ],
    },
    "parametric": {
        **_NO_GEOMETRY,
        "parametric": [
            {"x": "5*cos(t)", "y": "5*sin(t)", "name": "c1", "t_min": 0, "t_max": 6.283185307179586},
            {"x": "8*sin(3*t)", "y": "8*sin(2*t)", "name": "c2", "t_min": 0, "t_max": 6.283185307179586},
            {"x": "t*cos(t)", "y": "t*sin(t)", "name": "c3", "t_min": 0, "t_max": 18.84955592153876},
            {"x": "6*cos(4*t)*cos(t)", "y": "6*cos(4*t)*sin(t)", "name": "c4", "t_min": 0, "t_max": 6.283185307179586},
        ],
    },
    "areas": {
        **_NO_GEOMETRY,
        "functions": [
            {"expression": "sin(x)", "name": "g1", "left": -20, "right": 20},
            {"expression": "x^2/10 - 3", "name": "g2", "left": -20, "right": 20},
            {"expression": "1/x", "name": "g3", "left": -20, "right": 20},
        ],
        "colored_areas": [
            {"drawable1": "g1", "drawable2": "g2", "left": -5, "right": 5},
            {"drawable1": "g1", "drawable2": None, "left": -10, "right": 10},
            {"drawable1": "g3", "drawable2": None, "left": 0.5, "right": 8},
        ],
    },
    "statistics": {
        **_NO_GEOMETRY,
        "distributions": [
            {"name": "N1", "representation": "continuous", "mean": 0, "sigma": 1, "shade": [-1, 1]},
            {"name": "N2", "representation": "discrete", "mean": 5, "sigma": 2, "shade": None},
        ],
        "bars": [
            {"name": "B1", "values": [3, 7, 2, 9, 4, 6, 5, 8]},
        ],
    },
}

_WARMUP_SCENE: Dict[str, Any] = {
    "points": 3,
    "segments": 2,
    "vectors": 1,
    "triangles": 1,
    "rectangles": 1,
    "circles": 1,
    "ellipses": 1,
    "functions": [{"expression": "tan(x)", "name": "w1", "left": -5, "right": 5}],
    "parametric": [{"x": "cos(t)", "y": "sin(t)", "name": "w2", "t_min": 0, "t_max": 6.283185307179586}],
    "colored_areas": [{"drawable1": "w1", "drawable2": None, "left": 0, "right": 1}],
    "distributions": [{"name": "W3", "representation": "continuous", "mean": 0, "sigma": 1, "shade": None}],
    "bars": [{"name": "W4", "values": [1, 2]}],
}


def _per_drawable(build_ms: Dict[str, float], created: Dict[str, int]) -> Dict[str, float]:
    """Average build time per drawable of each kind the scene created."""
    return {kind: total / created[kind] for kind, total in build_ms.items() if created.get(kind)}


_FUNCTION_EXPRESSIONS: List[str] = ["sin(x)*x^2", "tan(x)", "1/(x^2 - 1)", "exp(-x^2/2)/sqrt(2*pi)"]
_MATHJS_EXPRESSIONS: List[str] = ["2*sin(1) + sqrt(16)", "log(10)^2 / 3", "gcd(84, 36) + 5!"]
# Denominators and log arguments without a polynomial form: their zeros are found by a scan
_ANALYSIS_EXPRESSIONS: List[str] = ["1/sin(x)", "ln(sin(x))", "1/(e^x-2)"]
# Numbers the uncached function_create appends, so its texts never repeat within a page
_UNCACHED_TEXTS = {"next": 0}


def _time_loop(perf_api: Any, repeats: int, body: Callable[[], Any]) -> Dict[str, float]:
    start = perf_api.now()
    for _ in range(repeats):
        body()
    total = perf_api.now() - start
    return {"total_ms": total, "count": float(repeats), "avg_ms": total / max(repeats, 1)}


def _expression_benchmarks(perf_api: Any, scale: int) -> Dict[str, Dict[str, float]]:
    """Time expression parsing and evaluation, the hot paths the refactors touch."""
    from compiled_expression import compile_expression
    from drawables.function import Function
    from drawables.parametric_function import ParametricFunction
    from expression_validator import ExpressionValidator
    from utils.math_utils import MathUtils

    results: Dict[str, Dict[str, float]] = {}
    expressions = _FUNCTION_EXPRESSIONS

    counter = {"i": 0}

    def fix_expression() -> None:
        ExpressionValidator.fix_math_expression(expressions[counter["i"] % len(expressions)])
        counter["i"] += 1

    results["fix_math_expression"] = _time_loop(perf_api, 200 * scale, fix_expression)

    def create_function() -> None:
        Function(expressions[counter["i"] % len(expressions)], name="perf_f")
        counter["i"] += 1

    results["function_create"] = _time_loop(perf_api, 4 * scale, create_function)

    def create_uncached_function() -> None:
        # A new text each time, so the parse cache misses (function_create measures hits)
        _UNCACHED_TEXTS["next"] += 1
        text = expressions[counter["i"] % len(expressions)] + " + " + str(_UNCACHED_TEXTS["next"])
        Function(text, name="perf_f")
        counter["i"] += 1

    results["function_create_uncached"] = _time_loop(perf_api, 4 * scale, create_uncached_function)

    function = Function("sin(x)*x^2 + 1/(x - 0.123456789)", name="perf_eval")
    samples = 2000 * scale
    xs = [-10.0 + 20.0 * k / samples for k in range(samples)]
    position = {"k": 0}

    def eval_function() -> None:
        function.function(xs[position["k"] % samples])
        position["k"] += 1

    results["function_eval"] = _time_loop(perf_api, samples, eval_function)

    curve = ParametricFunction("t*cos(t)", "t*sin(t)", name="perf_p", t_min=0, t_max=20)
    position["k"] = 0

    def eval_parametric() -> None:
        curve.evaluate(xs[position["k"] % samples] + 10.0)
        position["k"] += 1

    results["parametric_eval"] = _time_loop(perf_api, samples, eval_parametric)

    # A function of two variables, the budget of an implicit curve F(x, y) = 0 on a grid
    implicit = compile_expression("x^2 + y^2 - 1", ("x", "y")).bind()
    grid_samples = 10000 * scale
    position["k"] = 0

    def eval_xy() -> None:
        k = position["k"]
        implicit(xs[k % samples], xs[(k * 7) % samples])
        position["k"] = k + 1

    results["xy_eval"] = _time_loop(perf_api, grid_samples, eval_xy)

    def evaluate_mathjs() -> None:
        MathUtils.evaluate(_MATHJS_EXPRESSIONS[counter["i"] % len(_MATHJS_EXPRESSIONS)])
        counter["i"] += 1

    results["mathutils_evaluate"] = _time_loop(perf_api, 30 * scale, evaluate_mathjs)

    def analyse_function() -> None:
        # Asymptotes and discontinuities of a function drawn without bounds
        MathUtils.calculate_asymptotes_and_discontinuities(
            _ANALYSIS_EXPRESSIONS[counter["i"] % len(_ANALYSIS_EXPRESSIONS)]
        )
        counter["i"] += 1

    results["function_analysis"] = _time_loop(perf_api, 3 * scale, analyse_function)
    return results


def run_benchmark(scene_names: Optional[List[str]] = None, iterations: int = 3, scale: int = 1) -> Dict[str, Any]:
    """Run the scene and expression benchmarks; return plain data for JSON."""
    perf_api = window.performance
    names = scene_names or list(SCENES)
    unknown = [name for name in names if name not in SCENES]
    if unknown:
        raise ValueError(f"Unknown scene(s): {', '.join(unknown)}; known: {', '.join(SCENES)}")
    # One small run first, so first-use imports and caches don't land in the timings.
    run_renderer_performance(scene_spec=_WARMUP_SCENE, iterations=1)

    scenes: Dict[str, Any] = {}
    for name in names:
        result = run_renderer_performance(scene_spec=SCENES[name], iterations=iterations)
        scenes[name] = {
            "renderer": result.get("renderer"),
            "drawables_created": result.get("drawables_created"),
            "build_ms_per_drawable": _per_drawable(result.get("build_ms") or {}, result.get("drawables_created") or {}),
            "metrics": {metric["operation"]: metric["avg_ms"] for metric in result.get("metrics", [])},
            "phases": {
                phase["phase"]: phase.get("avg_ms", phase.get("total_ms"))
                for phase in result.get("phase_metrics", [])
                if phase.get("phase") != "plan_skip"
            },
        }

    return {
        "iterations": iterations,
        "scale": scale,
        "user_agent": str(window.navigator.userAgent),
        "scenes": scenes,
        "expressions": _expression_benchmarks(perf_api, max(1, int(scale))),
    }


_state: Dict[str, Optional[str]] = {"status": "idle", "result": None}


def start_benchmark_json(options_json: str = "{}") -> str:
    """Start a benchmark run in the background; poll ``get_benchmark_result_json`` for the result.

    Exposed as ``window.startMatHudPerfBenchmark``. Options: ``scenes`` (list of names),
    ``iterations`` (draws per operation), ``scale`` (multiplier for the expression loops).
    A run takes longer than a WebDriver call may block, hence start-and-poll.
    """
    if _state["status"] == "running":
        return json.dumps({"status": "already_running"})
    try:
        options = json.loads(options_json or "{}")
    except Exception as exc:
        return json.dumps({"status": "error", "error": f"Bad options: {exc}"})
    _state["status"] = "running"
    _state["result"] = None

    def execute() -> None:
        try:
            result = run_benchmark(
                scene_names=options.get("scenes"),
                iterations=int(options.get("iterations", 3)),
                scale=int(options.get("scale", 1)),
            )
            _state["result"] = json.dumps({"status": "done", **result})
        except Exception as exc:
            _state["result"] = json.dumps(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            )
        finally:
            _state["status"] = "idle"

    window.setTimeout(execute, 10)
    return json.dumps({"status": "started"})


def get_benchmark_result_json() -> str:
    """The finished run as JSON, or ``{"status": "running"}`` / ``{"status": "no_results"}``.

    Exposed as ``window.getMatHudPerfBenchmarkResult``.
    """
    if _state["status"] == "running":
        return json.dumps({"status": "running"})
    return _state["result"] or json.dumps({"status": "no_results"})
