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

