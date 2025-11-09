# Renderer Pipeline Overview

MatHud renderers now ship a single optimized pipeline that records drawable plans,
reuses pooled primitives, and applies updates in batches. The historical
“legacy” helpers are still used internally for plan generation, but renders are
always executed through the optimized path.

## Runtime Selection

- `Canvas2DRenderer` is the default renderer created by `Canvas`.
- To opt into `SvgRenderer` or `WebGLRenderer`, instantiate them explicitly and
  pass the instance to `Canvas(width, height, renderer=...)`, or call
  `rendering.factory.create_renderer("svg")` / `"webgl"` as needed.
- Optimized behaviour is always enabled; there is no longer a strategy flag such
  as `window.MatHudRendererStrategy`.

### Feature Flags

- **SVG offscreen staging**: enable with `window.MatHudSvgOffscreen = True` or
  `window.localStorage["mathud.svg.offscreen"] = "1"`.
- **Canvas2D layer compositing**: enable with `window.MatHudCanvas2DOffscreen = True`
  or `window.localStorage["mathud.canvas2d.offscreen"] = "1"`.

These toggles remain available as regression escape hatches while preserving the
optimized renderer behaviour.

## Optimized Path Highlights

- **Plan caching with viewport culling**  
  Canvas2D and SVG cache per-drawable plans keyed by normalized signatures and
  reuse them until the drawable changes or leaves the frame. Plans store screen
  bounds so off-screen drawables are skipped quickly.

- **Frame-aware batching**  
  `Canvas.draw` opens each frame with `renderer.begin_frame()`. SVG keeps pooled
  nodes alive between frames, while Canvas2D clears its surface inside the
  renderer to avoid ghosting.

- **Batched DOM mutations**  
  The SVG primitive adapter stages new nodes inside a `DocumentFragment` and
  performs attribute updates only when values change. Canvas2D batches stroke
  and polygon operations to minimize context thrash.

- **Telemetry hooks**  
  Renderers expose `peek_telemetry()` and `drain_telemetry()` covering plan
  build/apply timings, skip counts, adapter events, and batch depth. SVG records
  `adapter.fragment_append` and `adapter.direct_append` so you can see how DOM
  updates were staged.

- **Screen-space primitives**  
  Point markers, vector arrowheads, and angle arcs/labels stay visually stable
  as you zoom. Plans record math-space anchors and recompute screen offsets on
  every map update, preventing oversized tips or drifting labels after scale
  changes. Text labels are included in this group, keeping line wrapping,
  rotation, and color consistent across renderers.

## Cache Invalidation

- Drawable plan caches are pruned at the end of any frame in which a plan is not
  touched.
- Map state changes invoke `plan.update_map_state`, avoiding redundant rebuilds.
- Explicit helpers (`invalidate_drawable_cache`, `invalidate_cartesian_cache`)
  remain available for complex mutation flows.

## Benchmarking

Use `run_renderer_performance` from
`static/client/client_tests/renderer_performance_tests.py` to record timings,
DOM counts, and telemetry:

```python
from renderer_performance_tests import run_renderer_performance

# Default Canvas2D baseline
baseline = run_renderer_performance(iterations=2)

# Explicit SVG baseline
svg_baseline = run_renderer_performance(iterations=2, renderer_name="svg")
```

Each run reports draw/pan/zoom averages, DOM node totals, and telemetry
snapshots. The automated test suite (`TestRendererPerformance`) ensures the
optimized renderer applies plans without misses and emits valid metrics.

## Fallback Options

- Instantiate `create_renderer("svg")` or `create_renderer("webgl")` during
  `Canvas` construction to compare alternative renderers.
- Toggle SVG or Canvas2D offscreen staging (see Feature Flags) to isolate DOM or
  compositing regressions.

Record updated benchmarks whenever optimization work lands so Canvas2D and SVG
baselines stay current. Use telemetry snapshots to verify plan reuse, skip
counts, and adapter event ratios after each change.
