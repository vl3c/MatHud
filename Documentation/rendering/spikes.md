# Renderer Spike Summary

## Canvas 2D Prototype (`Canvas2DRenderer`)
1. Creates an overlay `<canvas id="math-canvas-2d">` inside `#math-container` and keeps it hidden until activated.
2. Implements `RendererProtocol` with handlers for points, segments, and circles using familiar 2D drawing primitives.
3. Reuses the existing registry pattern, so `Canvas._register_renderer_handlers()` works without additional glue code.
4. Rendering logic uses the present `CoordinateMapper`, guaranteeing parity for zoom/pan transformations.
5. Missing features: triangles, rectangles, functions, and filled regions still need bespoke handlers, but the API footprint is straightforward.
6. Performance expectations: immediate-mode drawing on Canvas 2D reduces DOM node churn entirely; profiling with the harness should confirm native context throughput improvements on mid-tier hardware.

## WebGL Prototype (`WebGLRenderer`)
1. Allocates `<canvas id="math-webgl">` alongside the SVG surface and boots a minimal shader program for lines and points.
2. Current handlers support points, segments, and circles (approximated via line strip). Axis rendering uses WebGL primitives with orthographic conversion via the `CoordinateMapper`.
3. Geometry uploads happen per drawable, which is acceptable for the spike but would benefit from batching before production rollout.
4. Color handling converts CSS strings through an off-screen Canvas to RGBA, ensuring near-term compatibility with existing styling metadata.
5. Known limitations:
   - WebGL line width support varies by platform; wide strokes or arrowheads need custom geometry.
   - Text (labels, tick marks) requires either a signed distance field pipeline or hybrid Canvas/SVG overlay.
   - Filled regions and complex shapes need indexed meshes to reach parity.

## Target Recommendation
1. Adopt Canvas 2D as the primary next-step renderer. It preserves immediate drawing semantics, drastically cuts DOM node counts, and requires the least migration effort for text and labeling (via `fillText`).
2. Keep the WebGL spike for future experimentation. It promises larger gains for dense function plots and high object counts, but the additional work on text rendering, batching, and cross-platform testing outweighs the short-term benefit.
3. Integration plan:
   - Gate renderer selection behind a feature flag once DOM updates land.
   - Expand Canvas 2D handlers to cover remaining drawable types before enabling public testing.
   - Use the performance harness to compare SVG vs. Canvas 2D across representative scenes, documenting improvements alongside any regressions in label clarity.

