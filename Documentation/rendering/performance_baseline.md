# Renderer Performance Baselines

The dual-mode renderer has been retired. These baselines capture the optimized
pipeline for Canvas2D and SVG using the shared benchmark workload:

- 50 points, 25 segments, 10 vectors, 10 triangles, 5 rectangles, 5 circles,
  3 ellipses, 3 functions (`sin`, `cos`, `x**2`)
- Warm-up run followed by a measured pass (two draws, four pans, four zooms)

Use `run_renderer_performance(iterations=2)` to refresh the numbers. Pass
`renderer_name="svg"` to capture the SVG renderer explicitly.

## Canvas2DRenderer (Optimized Pipeline)

- Date: 2025-11-08  
- DOM nodes (post-render): 0  
- Reference run: `run_renderer_performance(iterations=2)`

| Operation | Avg ms | Iterations |
|-----------|--------|------------|
| draw      | 71.00  | 2          |
| pan       | 155.50 | 4          |
| zoom      | 160.00 | 4          |

Telemetry highlights:

- `plan_build_count`: 153 events, `plan_build_avg`: 3.47 ms  
- `plan_apply_count`: 14 130 events, `plan_apply_avg`: 0.29 ms  
- `plan_miss_count`: 0  
- Adapter counters: `begin_path`=19 684, `stroke_calls`=11 940,
  `polygon_batch_polygons`=935

## SvgRenderer (Optimized Pipeline)

- Date: 2025-11-09  
- DOM nodes (post-render): 258  
- Reference run: `run_renderer_performance(iterations=2, renderer_name="svg")`

| Operation | Avg ms | Iterations |
|-----------|--------|------------|
| draw      | 78.00  | 2          |
| pan       | 84.00  | 4          |
| zoom      | 90.00  | 4          |

Telemetry highlights:

- `plan_build_count`: 277 events, `plan_build_avg`: 3.94 ms  
- `plan_apply_count`: 15 038 events, `plan_apply_avg`: 0.42 ms  
- `plan_miss_count`: 0  
- Adapter counters: `fragment_append`=277, `direct_append`=85, `svg_clone_copy`
  emitted once per staged clone

## Refresh Procedure

1. Launch the Brython client harness (`run tests` via the chat helper).
2. Execute `run_renderer_performance(iterations=2)` for Canvas2D.
3. Execute `run_renderer_performance(iterations=2, renderer_name="svg")` for SVG.
4. Record draw/pan/zoom averages, DOM counts, and telemetry summaries in this
   document.

Keep these tables up to date whenever optimization work lands. Zeroth plan
misses and stable adapter ratios are the primary regression checks after each
change.
