# Renderer Rollout and Verification Plan

The legacy renderer pathways have been removed. Canvas, SVG, and WebGL now run
exclusively through the optimized plan pipeline. This document tracks rollout
gating items, regression toggles, and benchmark expectations.

## Current Status

- `Canvas2DRenderer` is the default renderer instantiated by `Canvas`.
- `SvgRenderer` and `WebGLRenderer` remain available via
  `rendering.factory.create_renderer(...)` for targeted scenarios.
- Optimized telemetry is enforced by `TestRendererPerformance`.

## Regression Toggles

- **Renderer selection**: construct `Canvas(width, height, renderer=create_renderer("svg"))`
  (or `"webgl"`) to investigate issues in an alternative backend.
- **Offscreen staging flags**:
  - `window.MatHudSvgOffscreen` / `localStorage["mathud.svg.offscreen"]`
  - `window.MatHudCanvas2DOffscreen` / `localStorage["mathud.canvas2d.offscreen"]`

Leave these toggles documented for customer support and incident response.

## Smoke Checklist

1. Create and delete each drawable type (points, segments, polygons, angles,
   colored areas, functions).
2. Exercise pan/zoom loops and verify telemetry plan counts remain steady.
3. Confirm screen-space primitives behave correctly:
   - Point labels remain offset by a constant screen radius.
   - Vector arrowheads keep their tip size.
   - Angle arcs and labels stay anchored to the vertex.
4. Run undo/redo cycles on mixed scenes.
5. Drain telemetry and confirm `plan_miss_count == 0`.
5. Capture screenshots for accessibility/visual regressions (text clarity,
   anti-aliasing).

## Benchmark Expectations

- Baseline workload results are recorded in
  `documentation/rendering/performance_baseline.md`.
- Refresh benchmarks with:

  ```python
  from renderer_performance_tests import run_renderer_performance

  canvas2d = run_renderer_performance(iterations=2)
  svg = run_renderer_performance(iterations=2, renderer_name="svg")
  ```

- Watch for increases in `plan_build_avg`, DOM node counts, or adapter event
  spikes. Any non-zero `plan_miss_count` is a regression.

## Rollout Steps

1. **Validation**  
   - Keep the client test suite green (`TestRendererPerformance`,
     `TestOptimizedRendererParity`, geometry coverage).
   - Perform the smoke checklist on representative workspaces (geometry-heavy,
     function-heavy, mixed).

2. **Communication**  
   - Publish the updated baselines and telemetry snapshot.
   - Circulate fallback instructions (`create_renderer("svg")`, offscreen
     toggles) to support teams.

3. **Monitoring**  
   - Add renderer telemetry to the release dashboard (plan counts, adapter
     events).
   - Schedule weekly benchmark refreshes until the release stabilises.

4. **Post-Rollout**  
   - Remove references to the legacy pipeline from product documentation.
   - Review WebGL experimental status once Canvas2D/SVG metrics stay within
     guardrails for two consecutive releases.

