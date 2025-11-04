# Renderer Rollout, Testing, and Benchmark Plan

## Phase 0 – Baseline
1. Use `performance/renderer_performance.py` to record SVG timings on low, medium, and high density scenes.
2. Capture DOM node counts and console logs for each dataset; store results in `/documentation/benchmarks/` for regression tracking.
3. Validate rendering parity manually on representative workspaces (geometry-heavy, function-heavy, mixed shapes).

## Phase 1 – Canvas 2D Shadow Mode
1. Instantiate both SVG and Canvas 2D renderers simultaneously; keep Canvas 2D hidden but feed it the same draw calls to catch runtime errors early.
2. Extend harness to compare frame times between the two renderers per scene, flagging >10% regressions.
3. Add unit coverage that ensures `create_renderer("canvas2d")` returns a functioning instance in Brython (mocking document/canvas APIs with `SimpleMock`).

## Phase 2 – Feature Flag Exposure
1. Ship renderer selector UI for internal users (localStorage flag). Default remains SVG.
2. Define smoke tests:
   - Create/draw/delete each drawable type
   - Zoom/pan loop
   - Undo/redo cycles
3. Collect feedback on text clarity, anti-aliasing, and performance; iterate on Canvas 2D handlers before enabling default use.

## Phase 3 – Default Swap
1. Switch default to Canvas 2D once smoke tests pass and harness shows consistent wins.
2. Keep SVG toggleable for one release cycle to provide escape hatch.
3. Automate harness execution via browser-driven regression job to detect future slowdowns.

## Phase 4 – WebGL Opt-In (Optional)
1. Stabilise WebGL renderer with batching and text overlay allowance.
2. Limit to power users via explicit flag; collect telemetry with harness (points/segments heavy scenes) to assess benefits.
3. Decide on long-term roadmap: either keep WebGL experimental or graduate to default once parity achieved.

