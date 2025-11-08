"""Renderer performance benchmark and test case."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import unittest

from browser import document, window
from constants import DEFAULT_RENDERER_MODE

from canvas import Canvas


# Baseline workload used when no custom scene is supplied
WORKLOAD_SCENE: Dict[str, Any] = {
    "seed": 1,
    "points": 50,
    "segments": 25,
    "vectors": 10,
    "triangles": 10,
    "rectangles": 5,
    "circles": 5,
    "ellipses": 3,
    "functions": [
        {"expression": "sin(x)", "name": "f_sin", "left": -20, "right": 20},
        {"expression": "cos(x)", "name": "f_cos", "left": -20, "right": 20},
        {"expression": "x**2", "name": "f_parabola", "left": -20, "right": 20},
    ],
}


def _populate_scene(canvas: Canvas, spec: Dict[str, Any]) -> Dict[str, int]:
    rng = random.Random(spec.get("seed", 42))

    created: Dict[str, int] = {
        "points": 0,
        "segments": 0,
        "vectors": 0,
        "triangles": 0,
        "rectangles": 0,
        "circles": 0,
        "ellipses": 0,
        "functions": 0,
    }

    points: List[Any] = []

    def pick_point() -> Any:
        if points:
            return rng.choice(points)
        point = canvas.create_point(rng.uniform(-50, 50), rng.uniform(-50, 50))
        points.append(point)
        created["points"] += 1
        return point

    for _ in range(spec.get("points", 0)):
        point = canvas.create_point(rng.uniform(-50, 50), rng.uniform(-50, 50))
        points.append(point)
    created["points"] = len(points)

    for _ in range(spec.get("segments", 0)):
        p1 = pick_point()
        p2 = pick_point()
        if p1 is p2:
            continue
        canvas.create_segment(p1.x, p1.y, p2.x, p2.y)
        created["segments"] += 1

    for _ in range(spec.get("vectors", 0)):
        origin = pick_point()
        tip = pick_point()
        if origin is tip:
            continue
        canvas.create_vector(origin.x, origin.y, tip.x, tip.y)
        created["vectors"] += 1

    for _ in range(spec.get("triangles", 0)):
        p1 = pick_point()
        p2 = pick_point()
        p3 = pick_point()
        if len({id(p1), id(p2), id(p3)}) < 3:
            continue
        canvas.create_triangle(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y)
        created["triangles"] += 1

    for _ in range(spec.get("rectangles", 0)):
        p = pick_point()
        opposite = pick_point()
        if p is opposite:
            continue
        canvas.create_rectangle(p.x, p.y, opposite.x, opposite.y)
        created["rectangles"] += 1

    for _ in range(spec.get("circles", 0)):
        center = pick_point()
        radius = rng.uniform(5, 25)
        canvas.create_circle(center.x, center.y, radius)
        created["circles"] += 1

    for _ in range(spec.get("ellipses", 0)):
        center = pick_point()
        radius_x = rng.uniform(5, 30)
        radius_y = rng.uniform(5, 30)
        rotation = rng.uniform(0, 180)
        canvas.create_ellipse(center.x, center.y, radius_x, radius_y, rotation)
        created["ellipses"] += 1

    for func_spec in spec.get("functions", []):
        expression = func_spec.get("expression", "sin(x)")
        name = func_spec.get("name", f"f_{created['functions']}")
        left = func_spec.get("left")
        right = func_spec.get("right")
        canvas.draw_function(expression, name, left, right)
        created["functions"] += 1

    return created


def run_renderer_performance(
    *,
    scene_spec: Optional[Dict[str, Any]] = None,
    iterations: int = 5,
    render_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the renderer performance benchmark inside the Brython runtime."""

    if "math-svg" not in document:
        raise RuntimeError("math-svg element not available in document")

    perf_api = getattr(window, "performance", None)
    if perf_api is None:
        raise RuntimeError("window.performance API is not available")

    previous_strategy = getattr(window, "MatHudRendererStrategy", None)
    if render_mode:
        try:
            setattr(window, "MatHudRendererStrategy", render_mode)
        except Exception:
            pass

    viewport = document["math-svg"].getBoundingClientRect()
    canvas = Canvas(viewport.width, viewport.height)
    active_renderer = getattr(canvas, "renderer", None)

    spec: Dict[str, Any] = {**WORKLOAD_SCENE, **(scene_spec or {})}

    drawables_created = _populate_scene(canvas, spec)

    metrics: List[Dict[str, float]] = []

    def collect(operation: str, total: float, draws: int) -> None:
        metrics.append(
            {
                "operation": operation,
                "iterations": float(draws),
                "total_ms": total,
                "avg_ms": total / max(draws, 1),
                "dom_nodes": float(len(document["math-svg"].children)),
            }
        )

    start = perf_api.now()
    for _ in range(iterations):
        canvas.draw()
    collect("draw", perf_api.now() - start, iterations)

    mapper = canvas.coordinate_mapper
    start = perf_api.now()
    for _ in range(iterations):
        mapper.apply_pan(20, 0)
        canvas.draw()
        mapper.apply_pan(-20, 0)
        canvas.draw()
    collect("pan", perf_api.now() - start, iterations * 2)

    start = perf_api.now()
    for _ in range(iterations):
        mapper.apply_zoom(1.05)
        canvas.draw(apply_zoom=True)
        mapper.apply_zoom(0.9523809524)
        canvas.draw(apply_zoom=True)
    collect("zoom", perf_api.now() - start, iterations * 2)

    telemetry_snapshot: Optional[Dict[str, Any]] = None
    if active_renderer is not None:
        drain_telemetry = getattr(active_renderer, "drain_telemetry", None)
        if callable(drain_telemetry):
            try:
                telemetry_snapshot = drain_telemetry()
            except Exception:
                telemetry_snapshot = None

    phase_metrics: List[Dict[str, Any]] = []
    if telemetry_snapshot:
        phase_data = telemetry_snapshot.get("phase", {})

        def add_phase(name: str, total_key: str, count_key: str) -> None:
            total_value = phase_data.get(total_key)
            count_value = phase_data.get(count_key)
            if total_value is None or count_value is None:
                return
            try:
                total_ms = float(total_value)
            except Exception:
                return
            try:
                count_int = int(count_value)
            except Exception:
                return
            entry: Dict[str, Any] = {
                "phase": name,
                "total_ms": total_ms,
                "count": count_int,
            }
            if count_int > 0:
                entry["avg_ms"] = total_ms / count_int
            phase_metrics.append(entry)

        add_phase("plan_build", "plan_build_ms", "plan_build_count")
        add_phase("plan_apply", "plan_apply_ms", "plan_apply_count")
        add_phase("legacy_render", "legacy_render_ms", "legacy_render_count")
        add_phase("cartesian_plan_build", "cartesian_plan_build_ms", "cartesian_plan_count")
        add_phase("cartesian_plan_apply", "cartesian_plan_apply_ms", "cartesian_plan_count")
        skip_count = phase_data.get("plan_skip_count")
        if skip_count:
            try:
                skip_int = int(skip_count)
            except Exception:
                skip_int = None
            if skip_int is not None:
                phase_metrics.append(
                    {
                        "phase": "plan_skip",
                        "total_ms": 0.0,
                        "count": skip_int,
                    }
                )

    console = getattr(window, "console", None)
    if console is not None:
        console.groupCollapsed("[RendererPerf] Baseline Results")
        for metric in metrics:
            console.log(
                f"[RendererPerf] {metric['operation']}: avg={metric['avg_ms']:.2f} ms "
                f"over {metric['iterations']} draws (DOM nodes={metric['dom_nodes']})"
            )
        console.log(f"[RendererPerf] Scene spec: {spec}")
        if telemetry_snapshot:
            console.groupCollapsed("[RendererPerf] Canvas2D Telemetry")
            frames = telemetry_snapshot.get("frames")
            if frames is not None:
                console.log(f"[RendererPerf] frames={frames}")
            for summary in phase_metrics:
                phase_name = summary["phase"]
                total_ms = summary["total_ms"]
                count_int = summary["count"]
                avg_ms = summary.get("avg_ms")
                if avg_ms is None:
                    console.log(
                        f"[RendererPerf] {phase_name}: total={total_ms:.2f} ms over {count_int} events"
                    )
                else:
                    console.log(
                        f"[RendererPerf] {phase_name}: avg={avg_ms:.2f} ms "
                        f"over {count_int} events (total={total_ms:.2f} ms)"
                    )
            plan_miss = telemetry_snapshot.get("phase", {}).get("plan_miss_count")
            if plan_miss:
                console.log(f"[RendererPerf] plan_miss_count={int(plan_miss)}")
            adapter_events = telemetry_snapshot.get("adapter_events", {})
            if adapter_events:
                console.groupCollapsed("[RendererPerf] Adapter Events")
                for event_name in sorted(adapter_events):
                    console.log(f"[RendererPerf] adapter.{event_name}={adapter_events[event_name]}")
                console.groupEnd()
            console.groupEnd()
        console.groupEnd()

    result = {
        "scene_spec": spec,
        "metrics": metrics,
        "drawables_created": drawables_created,
        "render_mode": render_mode or "legacy",
    }
    result["telemetry"] = telemetry_snapshot
    result["phase_metrics"] = phase_metrics
    if active_renderer is not None:
        result["renderer"] = getattr(active_renderer, "__class__", type(active_renderer)).__name__
    else:
        result["renderer"] = "None"
    result["default_mode"] = DEFAULT_RENDERER_MODE
    if render_mode is not None:
        if previous_strategy is None:
            try:
                delattr(window, "MatHudRendererStrategy")
            except Exception:
                pass
        else:
            try:
                setattr(window, "MatHudRendererStrategy", previous_strategy)
            except Exception:
                pass

    return result


class TestRendererPerformance(unittest.TestCase):
    """Renderer performance test case executed via the client test suite."""

    def test_renderer_performance_modes(self) -> None:
        # Warm legacy mode once, then capture the measured run
        run_renderer_performance(iterations=1, render_mode="legacy")
        legacy = run_renderer_performance(iterations=1, render_mode="legacy")

        # Warm optimized mode once, then capture the measured run
        run_renderer_performance(iterations=1, render_mode="optimized")
        optimized = run_renderer_performance(iterations=1, render_mode="optimized")

        for label, result in (("legacy", legacy), ("optimized", optimized)):
            self.assertIsInstance(result, dict, f"{label} run did not return a dict")
            self.assertIn("metrics", result, f"{label} run missing metrics")

        legacy_avgs = {metric["operation"]: metric["avg_ms"] for metric in legacy["metrics"]}
        optimized_avgs = {metric["operation"]: metric["avg_ms"] for metric in optimized["metrics"]}

        for operation, legacy_avg in legacy_avgs.items():
            optimized_avg = optimized_avgs.get(operation)
            self.assertIsNotNone(
                optimized_avg,
                f"Missing optimized metric for operation {operation}",
            )
            if optimized_avg is None:
                continue
            if optimized_avg > legacy_avg * 1.25:
                print(
                    "[RendererPerf] Optimized mode slower than legacy for "
                    f"{operation}: optimized={optimized_avg:.2f} ms legacy={legacy_avg:.2f} ms"
                )
            allowed = legacy_avg * 2.5 + 1e-6
            self.assertLessEqual(
                optimized_avg,
                allowed,
                f"Optimized mode slower for {operation}: {optimized_avg:.2f} ms vs {legacy_avg:.2f} ms",
            )

