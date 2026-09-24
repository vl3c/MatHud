# MatHud — Roadmap

## Direction

MatHud is a personal workbench for exploring mathematics with an AI assistant: the user describes intent in chat, the model executes tool workflows, and the canvas shows the result. Near-term goals:

1. **Correct math** — tools must not return silently wrong answers.
2. **One stable, unified app** — start one thing, get a window; works offline with a local model.
3. **Model workbench** — run the same explorations against several models (default: local) and compare quality, speed, and tool-use reliability.
4. **Exploration power** — parameters, curve analysis, implicit curves, dynamic constructions.

UI gestures remain secondary unless they directly support AI workflows.

This file has two parts. **Part A** is the committed roadmap for this project, in priority order. **Part B** collects ideas that are interesting but optional — future work, or better suited to a separate app. The previous, longer roadmap (with step-by-step breakdowns of every Part B item) is in git history before this rewrite.

Items marked *(GeoGebra-inspired)* come from a concept review of the GeoGebra source. GeoGebra is GPL-licensed: implement these from the named public algorithms, never by porting its code.

---

## Completed

- Refactor oversized classes/functions into smaller composable units
- `search_tools` as the default tool-discovery path
- AI-facing canvas state summaries for large scenes
- Drawable dependency architecture review
- Deterministic tool-execution logs / action traces
- Strict tool-argument validation and deterministic target resolution
- Static typing rollout (Brython `browser` stubs, mypy)
- Construction toolkit, relation inspector, transform workflows
- CI pipeline (server + client tests, lint)
- Multi-provider support (OpenAI, Anthropic, OpenRouter, LocalAgent as default)
- Text-to-speech read-aloud (Kokoro)
- OpenAI OAuth investigation — blocked (subscription and API billing are separate systems)

---

# Part A — This project

## A0. Stabilization (done, September 2026)

From the September 2026 project review. Conservative fixes, each with a regression test.

- ~~**Math correctness:** scientific-notation parsing, zero results reported as errors, equation-system solving (non-`y=f(x)` forms, nonlinear and cubic systems, nerdamer root parsing and verification), definite integrals across singularities, series convergence tests, vertical-asymptote detection, regression numerics (coefficient formatting, centered/scaled fits, logistic), Newton solver tolerances and non-square systems, polygon canonicalizers (isosceles, rhombus, kite, trapezoid), rotated-ellipse formula, graph algorithms on directed/disconnected/negative-weight graphs, `analyze_graph` wiring, chord and region areas, circle/ellipse intersections, relation checks, scale-aware tolerances, large-integer results from math.js.~~
- ~~**Data safety:** workspace reload restores every drawable type (round-trip test per class); atomic saves, `.bak` on overwrite, trash instead of hard delete; `delete_workspace` via POST; log retention.~~
- ~~**Tool-result plumbing:** one result per tool call for every provider; errors keyed by call; small return values passed back; explicit error when a tool isn't loaded; search-first mode explained to the model; `MATHUD_TOOL_EXPOSURE=search|full`; better local tool-search ranking.~~
- ~~**Canvas state for the model:** compact text format with derived facts (lengths, areas, angles, edge weights), change reports after each tool batch, canvas kept with vision requests, LocalAgent now sees the scene; `MATHUD_CANVAS_FORMAT=json|min_json|text`, `MATHUD_CANVAS_BUDGET_TOKENS`.~~
- ~~**Renderer:** WebGL removed, SVG frozen; Canvas2D cache invalidation/pruning, domain-edge sampling, labels, tick formatting, HiDPI; lean plan recording, JS path tracing, pan reprojection (pan with 4 functions ~250 ms → ~32 ms per frame).~~
- ~~**Test hygiene:** unregistered client test classes registered.~~

Follow-ups:
- **Comprehension benchmark for the canvas format** — once a model is reachable, compare `json` vs `text` answers (lengths, names, graph edges, changes after tool calls) per provider; tune the local budget to the model's context size.
- One user-visible undo step per AI action (nested manager archives currently create several).
- Serialize drawable colors/styles (points, segments, vectors, circles, polygons, graphs, function curves) so they survive reload and reach the model.
- `localStorage` mirror of the canvas so an accidental page reload doesn't lose work.
- Custom names for circles and ellipses (currently always `<center>(<radius>)`; needs a custom-name flag honoured by `regenerate_name()` and `__deepcopy__`).
- Region boolean operations ignore holes and use only outer boundaries when results are combined further.
- Undirected graph analysis collapses parallel edges (a doubled edge is still reported as a bridge).
- Server-side: remaining mypy `no-any-return` warnings in modules outside `mypy.ini`'s file list; a route test can make a live OpenAI call when a key is configured.

## A1. Unified app and model workbench

### Single desktop app
Goal: one command (later one executable) opens a MatHud window; no separate server and browser.
1. **Desktop shell via pywebview** (recommended first step): a launcher starts Flask in a background thread on a free port and opens a native window (Edge WebView2 on Windows). Pure Python, no second runtime, reuses everything. Electron/Tauri only if a concrete need appears — both would still have to bundle Python as a sidecar.
2. **Vendor browser libraries** (Brython, math.js, nerdamer, MathJax) into `static/vendor/` with pinned versions, so the app works fully offline with LocalAgent.
3. ~~**Vision snapshots without Selenium/Firefox:** capture the canvas client-side and send it with the request; drop the headless-Firefox startup dependency.~~ Done: `static/client/canvas_snapshot.py` composites the SVG and Canvas2D layers; `WebDriverManager` and `/init_webdriver` are removed.
4. **Lazy-load optional heavy pieces** (Kokoro/torch) so startup is fast and memory is low when TTS isn't used.
5. **Packaging** (later): PyInstaller one-folder build + Start-menu shortcut.

### Model workbench
1. **Per-response metrics** shown in chat and logged: provider, model, latency, time-to-first-token, tokens/s, prompt/completion tokens, number of tool calls, tool errors.
2. **Benchmark suite (CLI):** a curated set of math prompts with machine-checkable expectations (canvas state or tool results), run against a list of models; outputs a comparison table (accuracy, tool-call validity, speed). Builds on the existing tool-discovery benchmark and action traces.
3. **Side-by-side mode:** send the same prompt to two models (pairs naturally with tabs, A3).
4. **Local-model tuning:** tool descriptions and search-first prompts tuned for small local models; measure with the suite.

### CAS reliability
1. Known-answer audit of `derive`, `integrate`, `limit`, `solve`, `simplify`, `factor`, systems against nerdamer (unmaintained).
2. Numerically verify symbolic results where possible (substitute roots back, compare derivative/integral numerically).
3. Decide on a server-side SymPy fallback for weak areas (limits, non-polynomial solving) based on the audit.

### Conversation persistence
Save and restore the chat transcript with the workspace (truncate older turns on restore).

## A2. Exploration features

Ordered by value for exploring math; cheap, independent items first.

1. **Parameters and sliders** *(GeoGebra-inspired)* — named scalars with range/step (`a ∈ [-5, 5]`) usable in functions, parametric curves, point coordinates, radii; a small slider panel; AI can set or animate them ("sweep a from 0 to 2π"). Animation runs on `requestAnimationFrame` without undo entries per frame.
2. **Roots, extrema, inflection points, intersections of plotted functions** *(GeoGebra-inspired)* — sample at pixel resolution, bracket sign changes, refine with Brent's method; extrema via Brent's minimizer; reject false roots at poles; place labelled points with exact values in the tool result. Also a general `find_intersections(obj1, obj2)` reusing the existing line/circle/ellipse intersection code.
3. **Adaptive plotting with numeric discontinuity detection** *(GeoGebra-inspired)* — Gillam-style bisection on distance + turning angle; continuity test at max depth to split jumps/asymptotes; same sampler for parametric and polar curves. Replaces the string-based asymptote guessing (keep it only as seeds).
4. **Polar plots** `r(θ)` — thin wrapper over the parametric plotter.
5. **Adaptive numeric integration and arc length** *(GeoGebra-inspired)* — Gauss–Kronrod G7/K15 or adaptive Simpson with error estimate; tanh-sinh for endpoint singularities; arc length for functions and parametric curves.
6. **Calculus visualizations** *(GeoGebra-inspired)* — Riemann sums (left/right/mid/upper/lower/trapezoid) as a grouped composite, Taylor polynomial overlay at a point, curvature and osculating circle, slope triangle.
7. **Implicit curves `F(x,y)=0` and inequality shading** *(GeoGebra-inspired)* — marching squares on an adaptive quadtree, segment linking; regions like `y < x^2`, `x^2+y^2 < 4` with AND/OR. Also gives contour lines and level sets.
8. **General conics** *(GeoGebra-inspired)* — conic through 5 points, classification (ellipse/hyperbola/parabola/degenerate), foci, directrix, eccentricity, asymptotes, pole/polar, tangents from a point. Renders via parametric or implicit plotting.
9. **Circle inversion and tangent constructions** *(GeoGebra-inspired)* — reflect in a circle (lines/circles map to lines/circles), tangents from an external point, common tangents of two circles.
10. **Slope fields and ODE solutions** *(GeoGebra-inspired)* — direction field for `y' = f(x,y)`; solution curves via Dormand–Prince RK5(4) with adaptive step; phase portraits for 2D systems; vector fields on the same base.
11. **Dependent (dynamic) constructions** *(GeoGebra-inspired)* — derived objects store `{op, inputs, params}` and recompute in topological order when an input changes; undefined results hide instead of being deleted; definitions saved in workspaces. Builds on `drawable_dependency_manager.resolve_dependency_order()`. Unlocks loci and "is this true in general?" checks.
12. **Loci and points on paths** *(GeoGebra-inspired)* — a point constrained to a curve by parameter; the trace of a dependent point as a driver moves (or a slider varies), with adaptive step control.
13. **Probability distributions** *(GeoGebra-inspired)* — t, χ², F, exponential, gamma, beta, uniform, log-normal, binomial, Poisson, geometric; pdf/cdf/quantile and shaded `P(a < X < b)` (incomplete gamma/beta, Lanczos log-gamma, Brent for quantiles). Then hypothesis tests and confidence intervals.
14. **Sequences, iteration, cobweb diagrams** *(GeoGebra-inspired)* — `sequence(expr(k), k, a, b)` creates many objects in one call (one undo step); `iterate(g, x0, n)` for orbits and cobweb plots.
15. **Linear algebra hardening** — tests for `eigs`, `lup`, `qr`; add `rref`.
16. **Relation checks "in general"** *(GeoGebra-inspired)* — after dependent constructions exist: randomly perturb free objects, recompute, re-check the relation (randomized identity testing).
17. **Point-set geometry** *(GeoGebra-inspired, lower priority)* — Delaunay (Bowyer–Watson), Voronoi as its dual, TSP tour (nearest neighbour + 2-opt).
18. **Scatter plot and histogram with binning** — small additions on existing plot infrastructure.

## A3. UI and workflow

1. **Math symbol input for the chat box** — a symbol button next to the input opening a grouped palette (Greek letters, operators `± × ÷ · ≤ ≥ ≠ ≈`, calculus `∫ ∑ ∏ ∂ ∞ √`, superscripts `² ³ ⁿ`, sets/logic `∈ ∪ ∩ ⊂ ∀ ∃ ¬ ∧ ∨`, arrows), inserting at the cursor; plus `\alpha`-style completion reusing `CommandAutocomplete`. Extend the expression normalizer (which already maps `π`, `√`, `°`) with `² ³ × ÷ · − ≤ ≥ ≠` so the same text works when passed into tools.
2. **Tabs for parallel problems** — each tab holds its own canvas state, chat history, and selected model. Implement by swapping state through the workspace serializer (depends on the A0 round-trip fixes) rather than running multiple `Canvas` instances; AI tools `create_tab`, `switch_tab`, `list_tabs`. Enables side-by-side model comparison (A1).
3. **Export** — canvas to PNG/SVG; session (chat + results) to Markdown/LaTeX.
4. **One `highlight`/`annotate` tool** — lets the AI point at objects and label them temporarily (replaces the earlier list of 11 temporary-drawable types).
5. **Dark mode.**
6. **Speech input** (TTS output already exists).

## A4. Photo to math (real-world problems)

Goal: take a photo and let the AI model a real problem in it mathematically — measurements, trajectories, areas, angles, perspective — not just describe it.

1. **Photo as a canvas underlay** — place an image behind the math canvas with an adjustable transform (scale, rotation, offset); it pans/zooms with the canvas.
2. **Calibration** — the user or AI marks a known length (reference object) to set real-world scale; optional 4-point perspective rectification (homography) for photos of planar surfaces.
3. **AI-assisted extraction** — a vision-capable model proposes points/segments/curves on the calibrated image as normal drawables; the user confirms or adjusts; then regular tools compute (areas, angles, curve fits for trajectories, etc.).
4. **Worked-problem flow** — "how tall is this tree", "what's the area of this room", "fit the ball's path" as benchmark prompts in the model workbench.

Heavier segmentation (SAM) and OCR pipelines are in Part B; modern vision models cover most of this without them.

---

# Part B — Optional / future / separate projects

Not scheduled. Worth revisiting once Part A is solid, or better built as a separate app.

### Platform
- **Android app** — separate project; likely a thin client to a desktop/remote MatHud server, or a WebView wrapper once the desktop shell (A1) exists.
- PWA, touch/stylus input, shareable links, notebook-style cell mode.

### Advanced image analysis
- Server-side Meta SAM segmentation (point/box/text prompts), segment-to-canvas projection.
- Shape detection (OpenCV), math OCR (pix2tex-class models), curve fitting from images.
- Physics from images: trajectories from video frames, structural geometry, symmetry/pattern detection, perspective geometry.

### Canvas and workflow
- Construction history timeline, reusable macros and analysis templates.
- Session snapshots with diff/restore, "what changed" summaries, command palette.
- CLI replay of action traces.
- Tabular data workflow and CSV import/export; plotting from table columns; box plots; residual diagnostics.
- Numerical-methods visualizations (Newton iterations, bisection steps).
- Graph workflows: random/complete generators, edit tools, algorithm playback.
- Heatmaps.

### 3D
- 3D graphing, shapes, constraints, camera navigation — after 2D is mature.

### Collaboration and education
- Real-time collaborative sessions, teacher/student mode.
- Guided lessons, quiz/challenge mode, step-by-step animation mode.
- AI-initiated suggestions, multi-turn planning UI.

### Wearable HUD (long-term vision)
- AR canvas overlay: world-anchored objects, ghosted solutions, live measurement lines, construction guides.
- Continuous scene understanding: live camera analysis, real-time segmentation, material recognition.
- Voice-first interaction: wake word, spatial audio, gaze-anchored context, dictation.
- Live analysis modes: structural, trajectory, surveying, symmetry, optics.
- Educational, practical/professional (navigation, cooking, fitness, DIY, astronomy) and social/multi-user AR.
- Architecture: edge compute, spatial anchors, gaze/gesture input, low-latency vision, battery-aware modes, privacy controls.

### Other parked topics
- Truth tables and boolean algebra.
- Financial mathematics (compound interest, annuities, amortization).
- Q-Q / P-P plots, cumulative frequency tables.

### Dropped
- WebGL renderer parity (renderer removed).
- Semantic snapping, tolerance-policy management, conflict handling, resolution traces (depended on a snapping system that was never built; revisit only if direct-manipulation UX becomes a goal).
- Coordinate readout HUD overlay (tool results and canvas state already give the AI coordinates).

---

## Milestones

| Milestone | Focus |
|---|---|
| **1 — Stable** ✓ | A0 complete: known math bugs fixed, workspace round-trip, tool-result plumbing, canvas-state format, renderer fixes and speed |
| **2 — Unified app** | Desktop shell, vendored libs, client-side snapshots, per-response metrics, chat persistence |
| **3 — Workbench** | Benchmark suite, CAS audit, side-by-side comparison, local-model tuning |
| **4 — Explore I** | Sliders, roots/extrema/intersections, adaptive plotting, polar, calculus visuals, adaptive quadrature |
| **5 — UI** | Symbol palette, tabs, export, highlight tool |
| **6 — Explore II** | Implicit curves and inequalities, conics, inversion/tangents, ODEs, distributions, sequences |
| **7 — Dynamic geometry** | Dependent constructions, loci, relations "in general" |
| **8 — Photo to math** | Image underlay, calibration, AI-assisted extraction |
