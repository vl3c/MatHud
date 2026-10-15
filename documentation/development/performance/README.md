# Performance baselines

`python -m cli.main test perf` (see `cli/README.md`) times rendering and expression evaluation in headless Chrome. Timings depend on the machine, so a saved run is a reference for that machine only: to check a change, record a run on main and compare the branch with it on the same machine (`-o base.json`, then `--compare base.json`).

## `baseline_2026-10-15_cloud.json`

Main at da72ddf, before the expression and renderer refactors in the math-completion plan, recorded in a Claude Code cloud container (headless Chromium, x86_64, no GPU), 3 runs with 3 draws per operation, medians in milliseconds.

What stands out:
- **Creating a shape gets slow as the canvas fills.** In the `geometry` scene (30 points, 12 segments, a few vectors, triangles, rectangles, circles and ellipses) one circle, ellipse or rectangle takes about 470–490 ms, against about 8 ms for a segment. Every shape created with its extra graphics calls `PolygonManager.create_triangles_from_segments`, which checks every triple of segments on the canvas, so the cost grows with the cube of the segment count. With the renderer test's full workload (about 60 segments) one circle took about 7 s.
- **A geometry frame costs about 240 ms** while its plan build and apply take under 5 ms, so most of the frame is outside the cached plans (labels and point drawing are the first suspects).
- **Building a function with `Function(...)` costs about 80 ms**, mostly asymptote, discontinuity and periodicity analysis; evaluating it costs about 0.02 ms per sample.
- **A `MathUtils.evaluate` call (math.js) costs about 7 ms.**
