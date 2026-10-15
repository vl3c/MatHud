# Performance baselines

`python -m cli.main test perf` (see `cli/README.md`) times rendering and expression evaluation in headless Chrome. Timings depend on the machine, so a saved run is a reference for that machine only. To check a change, record a run on main and compare the branch with it on the same machine (`-o base.json`, then `--compare base.json`).

## `baseline_2026-10-15_cloud.json`

This run measures the app code of main at da72ddf, recorded before the planned expression, renderer and plot refactors. It was recorded in a Claude Code cloud container (headless Chromium, x86_64, no GPU) with 3 runs of 3 draws per operation; values are medians in milliseconds. The `commit` field names the benchmark branch commit it ran from.

What stands out:
- **Creating a shape gets slow as the canvas fills.** The `geometry` scene has 30 points, 12 free segments, a few vectors, triangles and rectangles (about 32 segments with their edges once circles are created), then circles and ellipses. In it, one circle, ellipse or rectangle takes about 440–520 ms, against about 8 ms for a segment.
  - The cause: every shape created with its extra graphics calls `PolygonManager.create_triangles_from_segments`, which checks every triple of segments on the canvas, so the cost grows with the cube of the segment count.
  - In an earlier run with the renderer test's full workload (about 60 segments; not saved here), one circle took about 7 s.
  - `build.*` times only a few drawables per run, so expect about ±25% between runs.
- **A warm geometry frame costs about 190 ms.** The cartesian grid's plans take about 35 ms of it, as in every scene. The drawables' cached plans are cheap, so most of the frame is spent outside them; labels and point drawing are the first suspects.
  - `plan_build` includes the grid's builds, so in this scene it equals `cartesian_plan_build`.
  - The `plan_*` values are averages per build or apply event, not per frame.
- **A first frame costs 120–300 ms** (`first_draw`), because it builds every plan.
- **Expressions:**
  - `Function(...)` construction, which also runs asymptote, discontinuity and periodicity analysis, costs about 70 ms.
  - Evaluating a function costs about 0.02 ms per sample.
  - A `MathUtils.evaluate` call (math.js) costs about 6 ms.
