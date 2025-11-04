# Renderer Performance Harness

## Purpose
The harness in `static/client/performance/renderer_performance.py` builds a synthetic scene, runs deterministic redraw, pan, and zoom cycles, and records timings with `window.performance.now()`. This provides a repeatable baseline for profiling the current SVG renderer before integrating alternative backends.

## Running the Benchmark
1. Open MatHud in the browser and ensure the workspace canvas is empty (the harness clears the canvas by default).
2. Open the browser console and execute:
   ```python
   from performance.renderer_performance import run_renderer_performance
   run_renderer_performance()
   ```
3. Results are logged under the `[RendererPerf]` console group and the function returns a dictionary with:
   - `scene_spec`: drawable counts used for the synthetic scene.
   - `metrics`: total and average milliseconds per operation (`draw`, `pan`, `zoom`) plus DOM node counts under `#math-svg`.
   - `drawables_created`: actual counts built during setup.

### Customisation
- Pass a custom scene or iteration count:
  ```python
  run_renderer_performance(
      scene_spec={"points": 100, "segments": 80, "seed": 99},
      iterations=8,
  )
  ```
- To keep the generated drawables on the canvas for manual inspection, call with `cleanup=False`.

## How Measurements Work
- **Scene generation**: deterministic random coordinates populate points, segments, vectors, polygons, circles, ellipses, and a small set of functions.
- **Draw timing**: executes `Canvas.draw()` repeatedly and reports average cost.
- **Pan timing**: alternates horizontal pans while forcing redraws to capture mapper updates plus render cost.
- **Zoom timing**: applies paired zoom in/out steps with `apply_zoom=True` to exercise grid invalidation and drawable zoom caches.
- **DOM node tracking**: counts children under `document['math-svg']` after each phase to correlate load with draw time.

The harness restores the original coordinate mapper state after every run and optionally clears the canvas, allowing successive benchmarks without manual reset.

