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

