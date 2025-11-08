# Renderer Rendering Modes And Benchmarks

MatHud renderers support two execution strategies:

- `legacy` — emits primitives immediately through the historical helper pipeline.
- `optimized` — builds drawable plans, reuses pooled primitives, and batches DOM or context mutations.

Both strategies generate identical geometry; the optimized path targets CPU and DOM efficiency while staying within a tight performance envelope relative to legacy.

## Strategy Selection

Renderers inspect `window.MatHudRendererStrategy` (or `mathud.renderer.strategy` in `localStorage`) when they are constructed. Expected values are `legacy` and `optimized`.

```python
from browser import window
window.MatHudRendererStrategy = "optimized"  # runtime override
window.localStorage.setItem("mathud.renderer.strategy", "optimized")  # persisted
```

If no preference is stored the legacy strategy is used.

## Optimized Path Highlights

- **Plan caching with viewport culling**  
  Canvas2D and SVG renderers capture per-drawable plans using normalized signatures (drawable state plus identifying attributes). Plans retain screen-bounds metadata so off-screen drawables are skipped unless they intersect the viewport.

- **Frame pipeline aware caching**  
  `Canvas.draw` now calls `renderer.begin_frame()` before any clearing or rendering work. Optimized SVG skips `renderer.clear()`, allowing pooled elements and cached plans to persist between frames while legacy renderers still clear the surface.

- **Batched DOM mutations**  
  The SVG primitive adapter stages new nodes inside a `DocumentFragment` and issues attribute updates only when values change. Existing pool entries remain attached, so most frames update style in place without extra DOM churn.

- **Offscreen staging**  
  The Canvas2D renderer optionally composites from an offscreen canvas. The SVG renderer mirrors the behaviour behind a feature flag, rendering into a hidden `<svg>` before cloning staged nodes back into the visible surface. Offscreen rendering is disabled by default; explicitly enable it with:
  - `window.MatHudSvgOffscreen = True`
  - `window.localStorage.setItem("mathud.svg.offscreen", "1")`

- **Telemetry hooks**  
  Both renderers expose `drain_telemetry()` and `peek_telemetry()` describing plan build/apply timing, legacy fallbacks, skip counts, and adapter events. SVG now reports `adapter.fragment_append`, `adapter.direct_append`, and `adapter.svg_clone_copy` so you can confirm whether DOM updates flowed through the staged fragment or the clone path.

## Cache Invalidation Heuristics

- Plan caches are keyed per drawable and persist until the drawable disappears from the frame. When a drawable is no longer rendered, the cache entry is pruned during `end_frame`.
- Map state changes trigger `plan.update_map_state`, avoiding redundant rebuilds. Explicit helpers (`invalidate_drawable_cache`, `invalidate_cartesian_cache`) remain available for complex mutation flows.

## Benchmark Procedure

Use `run_renderer_performance` from `static/client/client_tests/renderer_performance_tests.py` to record timings and DOM counts. Provide the workload specification, iteration count, and desired mode:

```python
from renderer_performance_tests import run_renderer_performance

legacy = run_renderer_performance(iterations=2, render_mode="legacy")
optimized = run_renderer_performance(iterations=2, render_mode="optimized")
```

Every run captures average draw, pan, and zoom times, DOM node totals, and drained telemetry (emitted to the browser console for quick comparison). The automated test suite enforces:
- Optimized timings within 1.5× the legacy average.
- DOM node deltas within a 15% / 5 node window and non-zero node counts for optimized frames.
- Optimized telemetry with zero plan misses and limited plan rebuild totals relative to plan applies.

## Telemetry Checklist

After running the benchmark, inspect `optimized["telemetry"]["phase"]`:
- `plan_build_count` should remain close to zero once caches are warm (threshold enforced by the test).
- `plan_build_ms` should track `plan_apply_ms`; sustained spikes indicate cache churn.
- `adapter.fragment_append`, `adapter.direct_append`, and `adapter.svg_clone_copy` provide a quick sanity check on DOM staging routes.

## Maintaining Baselines

1. Launch the Brython client test harness (for example, run `run tests` in the console helper).
2. Execute `run_renderer_performance` for both strategies against the baseline scene.
3. Record timing, DOM, and telemetry summaries in the rendering metrics tables.
4. Keep `TestRendererPerformance.test_renderer_performance_modes` green; it enforces the timing, DOM, and telemetry thresholds described above.

Refresh the benchmarking tables whenever new optimizations land so Canvas2D and SVG baselines remain aligned.

## Rollout Notes

- Leave the renderer strategy flags (`window.MatHudRendererStrategy`, `window.MatHudSvgOffscreen`) available so teams can fall back to legacy rendering if parity issues surface.
- Do not flip the defaults until optimized SVG meets the timing and DOM thresholds on representative customer scenes in addition to the baseline harness.
- Once parity is confirmed, update the shared benchmark tables with the new optimized metrics and call out the telemetry snapshot that was captured for traceability.
