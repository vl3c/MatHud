"""Tests for the benchmark result handling in cli/perf.py (no browser needed)."""

from __future__ import annotations

from typing import Any, Dict

from cli.perf import compare_metrics, flatten_run, is_slower, median_metrics


def _run(draw: float, plan_build: float, function_eval: float) -> Dict[str, Any]:
    return {
        "scenes": {
            "functions": {
                "metrics": {"draw": draw, "pan": 2 * draw},
                "phases": {"plan_build": plan_build, "plan_apply": None},
                "build_ms_per_drawable": {"functions": 3 * draw},
            }
        },
        "expressions": {"function_eval": {"avg_ms": function_eval, "count": 100.0, "total_ms": 100 * function_eval}},
    }


def test_flatten_run_names_scene_and_expression_metrics() -> None:
    flat = flatten_run(_run(10.0, 4.0, 0.02))
    assert flat == {
        "functions.draw": 10.0,
        "functions.pan": 20.0,
        "functions.plan_build": 4.0,
        "functions.build.functions": 30.0,
        "expr.function_eval": 0.02,
    }


def test_median_metrics_takes_the_middle_run() -> None:
    medians = median_metrics([_run(10.0, 4.0, 0.02), _run(30.0, 1.0, 0.05), _run(20.0, 2.0, 0.01)])
    assert medians["functions.draw"] == 20.0
    assert medians["functions.plan_build"] == 2.0
    assert medians["expr.function_eval"] == 0.02


def test_compare_metrics_covers_both_sides() -> None:
    rows = compare_metrics({"a": 3.0, "b": 1.0}, {"a": 2.0, "c": 5.0})
    assert rows == [("a", 2.0, 3.0, 1.5), ("b", None, 1.0, None), ("c", 5.0, None, None)]


def test_is_slower_needs_both_the_ratio_and_the_absolute_floor() -> None:
    assert is_slower(10.0, 13.0, 1.25, 1.0)
    assert not is_slower(10.0, 12.0, 1.25, 1.0)  # ratio 1.2
    assert not is_slower(0.02, 0.05, 1.25, 1.0)  # 2.5x but only 0.03 ms
    assert not is_slower(None, 5.0, 1.25, 1.0)
    assert not is_slower(0.0, 5.0, 1.25, 1.0)
