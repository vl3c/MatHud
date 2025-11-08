# Renderer Performance Baselines

## SvgRenderer

Date: 2025-11-04
Renderer: SvgRenderer (default)

- Scene: 50 points, 25 segments, 10 vectors, 10 triangles, 5 rectangles, 5 circles, 3 ellipses, 3 functions (sin, cos, x**2)
- Iterations: 2 draws/pans/zooms
- DOM nodes (post-render): 362

| Operation | Avg ms | Total ms | Iterations |
|-----------|--------|----------|------------|
| draw      | 118.00 | 236.00   | 2          |
| pan       | 112.75 | 451.00   | 4          |
| zoom      | 116.75 | 467.00   | 4          |

These numbers were captured via `run_renderer_performance(iterations=2)` in the Brython runtime on MatHud’s default SVG renderer. Future runs with alternative renderers (Canvas2D, WebGL) should log comparable data for side-by-side analysis.

## SvgRenderer (post-refactor)

Date: 2025-11-07
Renderer: SvgRenderer

- Scene: same as SVG baseline
- Iterations: 2 draws/pans/zooms
- DOM nodes (post-render): 362

| Operation | Avg ms | Total ms | Iterations |
|-----------|--------|----------|------------|
| draw      | 145.00 | 290.00   | 2          |
| pan       | 141.00 | 564.00   | 4          |
| zoom      | 147.75 | 591.00   | 4          |

Compared with the earlier SVG baseline, the refactored path is about 20% slower across draw, pan, and zoom. The visual output matches the legacy renderer, but the additional coordination logic introduced alongside the Canvas2D changes now impacts SVG performance as well.

-----------------------------------------------------------------------------------------------------

## Canvas2DRenderer

Date: 2025-11-04
Renderer: Canvas2DRenderer

- Scene: same as SVG baseline
- Iterations: 2 draws/pans/zooms
- DOM nodes (post-render): 0

| Operation | Avg ms | Total ms | Iterations |
|-----------|--------|----------|------------|
| draw      | 8.50   | 17.00    | 2          |
| pan       | 8.75   | 35.00    | 4          |
| zoom      | 13.75  | 55.00    | 4          |

## Canvas2DRenderer (post-refactor)

Date: 2025-11-07
Renderer: Canvas2DRenderer

- Scene: same as SVG baseline
- Iterations: 2 draws/pans/zooms
- DOM nodes (post-render): 0

| Operation | Avg ms | Total ms | Iterations |
|-----------|--------|----------|------------|
| draw      | 145.50 | 291.00   | 2          |
| pan       | 140.75 | 563.00   | 4          |
| zoom      | 145.00 | 580.00   | 4          |

The refactor routed drawable rendering directly through Canvas2D algorithms without the intermediate primitive batching the legacy helpers used. Every shape now triggers full context save or style recomputation on each draw, which retained visual output but caused a significant slowdown compared with the earlier primitive-mediated path.

-----------------------------------------------------------------------------------------------------

## Canvas2DRenderer (dual-mode benchmark)

Date: 2025-11-07  
Renderer: Canvas2DRenderer – legacy vs optimized strategy (run via `TestRendererPerformance.test_optimized_renderer_not_slower`)

- Scene: same as SVG baseline
- Iterations: warm-up 1, measured 2 draws/pans/zooms
- DOM nodes (post-render): 0

### Legacy Strategy

| Operation | Avg ms | Iterations |
|-----------|--------|------------|
| draw      | 118.00 | 2          |
| pan       | 117.25 | 4          |
| zoom      | 122.50 | 4          |

### Optimized Strategy

| Operation | Avg ms | Iterations |
|-----------|--------|------------|
| draw      | 213.00 | 2          |
| pan       | 213.75 | 4          |
| zoom      | 238.25 | 4          |

> Note: optimized SVG plans are still under active development; these numbers show the current gap and provide a baseline for continued tuning.

-----------------------------------------------------------------------------------------------------

## SVG vs SVG (2025-11-07 – default switched back to SVG)

- Harness default renderer: SVG (legacy strategy)
- Scene and iteration settings as above

### SVG (Legacy Strategy)

| Operation | Avg ms | Iterations |
|-----------|--------|------------|
| draw      | 144.00 | 2          |
| pan       | 147.50 | 4          |
| zoom      | 149.50 | 4          |

### SVG (Optimized Strategy)

| Operation | Avg ms | Iterations |
|-----------|--------|------------|
| draw      | 250.50 | 2          |
| pan       | 247.00 | 4          |
| zoom      | 251.00 | 4          |

DOM node counts remained at 362 in both cases. These measurements capture the current delta between the legacy and optimized SVG paths after restoring SVG as the default renderer, and provide a baseline for tuning the optimized plan path further.

-----------------------------------------------------------------------------------------------------

## Canvas2DRenderer (telemetry snapshot – optimized instrumentation)

Date: 2025-11-08  
Renderer: Canvas2DRenderer (legacy and optimized telemetry probes enabled)

- Scene: same as SVG baseline  
- Iterations: 2 draws/pans/zooms (optimized mode)  
- DOM nodes (post-render): 0

| Operation | Avg ms | Total ms | Iterations |
|-----------|--------|----------|------------|
| draw      | 131.00 | 262.00   | 2          |
| pan       | 131.50 | 526.00   | 4          |
| zoom      | 140.25 | 561.00   | 4          |

Telemetry summary (captured via `run_renderer_performance(iterations=2, render_mode="optimized")`):

- Frames rendered: 172  
- Plan metrics: `plan_build` and `plan_apply` recorded 0 events (optimized builders currently fall back to legacy helpers)  
- Legacy helper time: 3001.00 ms total across 14130 drawable renders (avg 0.21 ms)  
- Adapter counters: `begin_path`=24922, `begin_shape`=14130, `end_shape`=14130, `fill`=8679, `stroke`=16243, `text_draw`=12434, `stroke_color_changes`=9685, `fill_color_changes`=1, `stroke_width_changes`=1, `frame_begin/frame_end`=172  

These measurements establish the post-instrumentation baseline ahead of the optimized-plan caching and batching work.

-----------------------------------------------------------------------------------------------------

## Canvas2DRenderer (pre-optimization baseline – 2025-11-08)

Source: `TestRendererPerformance.test_optimized_renderer_not_slower` and `test_renderer_performance_harness` executed prior to the optimized plan caching changes. Shared scene: 50 points, 25 segments, 10 vectors, 10 triangles, 5 rectangles, 5 circles, 3 ellipses, 3 functions. Metrics were gathered after a warm-up pass; the tables below use the second measurement (two draws/pans/zooms) for each mode.

### Timing Summary

| Mode      | Draw Avg (ms) | Pan Avg (ms) | Zoom Avg (ms) |
|-----------|---------------|--------------|---------------|
| Legacy    | 118.00        | 115.75       | 120.75        |
| Optimized | 214.50        | 211.50       | 217.25        |

### Additional Harness Run

`TestRendererPerformance.test_renderer_performance_harness` (legacy renderer) recorded:

- Draw 112.50 ms, Pan 113.50 ms, Zoom 117.75 ms (averages across two draw, four pan, four zoom iterations).

These results capture the Canvas2D performance state before introducing telemetry-driven caching and batching optimizations, providing a reference point for subsequent improvements.

-----------------------------------------------------------------------------------------------------

## Canvas2DRenderer (2025-11-08 – legacy vs optimized instrumentation run)

Source: `TestRendererPerformance.test_optimized_renderer_not_slower` run against the shared scene (50 points, 25 segments, 10 vectors, 10 triangles, 5 rectangles, 5 circles, 3 ellipses, 3 functions). Metrics were collected after a warm-up pass with two draw/pan/zoom iterations per measurement.

### Timing Summary

| Mode      | Draw Avg (ms) | Pan Avg (ms) | Zoom Avg (ms) | Frames | Plan Build Events | Plan Build Avg (ms) | Plan Apply Events | Plan Apply Avg (ms) | Legacy Render Avg (ms) | Cartesian Plan Build Avg (ms) | Cartesian Plan Apply Avg (ms) |
|-----------|---------------|--------------|---------------|--------|-------------------|---------------------|-------------------|---------------------|-------------------------|-------------------------------|--------------------------------|
| Legacy    | 164.50        | 167.50       | 172.00        | 177    | 0                 | —                   | 0                 | —                   | 0.29 (14875 events)     | —                             | —                              |
| Optimized | 74.00         | 149.25       | 152.00        | 177    | 157               | 4.33                | 14875             | 0.26                | 0.00                    | 41.56 (9 events)              | 127.00 (9 events)              |

### Updated Canvas2D Baseline After Batching & Culling (2025-11-08 – single measured pass)

| Mode      | Draw Avg (ms) | Pan Avg (ms) | Zoom Avg (ms) | Frames | Plan Build Events | Plan Build Avg (ms) | Plan Apply Events | Plan Apply Avg (ms) | Legacy Render Avg (ms) |
|-----------|---------------|--------------|---------------|--------|-------------------|---------------------|-------------------|---------------------|-------------------------|
| Legacy    | 145.00        | 150.00       | 163.00        | 172    | 0                 | —                   | 0                 | —                   | 0.28 (14130 events)     |
| Optimized | 77.00         | 168.50       | 169.50        | 172    | 153               | 3.66                | 14130             | 0.31                | 0.00                    |

Optimized adapter counters during this run recorded `begin_path`=19 684, `stroke_calls`=11 940, `line_batch_segments`=17 187, and `polygon_batch_polygons`=935 (legacy remained at the original counts). Plan skips were not triggered because the baseline scene keeps all drawables on screen.

