# MatHud — Roadmap

## Direction

MatHud is a personal workbench for exploring mathematics with an AI assistant: the user describes intent in chat, the model executes tool workflows, and the canvas shows the result.

The long-term goal is the name: a mathematical **heads-up display** that looks at a real situation, models it mathematically, and explains it, by voice and with drawings over what the user sees, on smart glasses or in VR, in 3D as well as 2D. This app is the prototype and foundation for it: a desktop and web app now; Android is the next platform, then glasses and VR. 3D comes to this app first (after 2D dynamic geometry, independent of Android), then carries over to the HUD. It may be rewritten in another language or stack along the way, so what must carry over is the design (the AI-first operating model, the tool contract, math-only drawable models, the scene state shared by user and model, checked math, the saved-state schema and the scenario catalogue), not this code; `documentation/Project Architecture.txt` lists where each part lives. Everything below is a step toward the HUD; [The path to the HUD](#the-path-to-the-hud) shows which phase builds which part.

Near-term goals:

1. **Correct math** — tools must not return silently wrong answers.
2. **One stable, unified app** — start one thing, get a window; works offline with a local model.
3. **Model workbench** — run the same explorations against several models (default: local) and compare quality, speed, and tool-use reliability.
4. **Exploration power** — parameters, curve analysis, implicit curves, dynamic constructions.
5. **Explanation** — the AI shows a result step by step on the canvas: hides, dashes and highlights objects, reveals steps in order, reads them aloud, replays what it built.

UI gestures remain secondary unless they directly support AI workflows.

**Scope: practical math, from the scientific graphing calculator tradition.** This is one of the things that set MatHud apart from other math apps. MatHud grows out of the scientific and graphing calculator, not toward every advanced field of mathematics. The yardstick is the graphing calculators people use (TI-84 Plus CE, Casio fx-CG50, NumWorks, Desmos); features found only on CAS handhelds (TI-Nspire CAS, HP Prime) need a practical reason of their own. A feature belongs when a graphing-calculator user would expect it, or when a real-world problem the HUD should explain needs it (measuring, fitting data, rates, probabilities, geometry of real objects). Advanced math for its own sake (research-level topics, constructions that only matter inside geometry theory) stays parked in Part B however interesting it is. Exploration items below marked *(parked: scope)* fail this test: they keep their numbers but are not scheduled (parked by V, 2026-10-16).

This file has two parts. **Part A** is the committed roadmap for this project, in priority order. **Part B** collects ideas that are interesting but optional — future work, or better suited to a separate app. The previous, longer roadmap (with step-by-step breakdowns of every Part B item) is kept unchanged in [`archive/Roadmap_2026-09_full.md`](archive/Roadmap_2026-09_full.md), with an index of possible spinoff apps.

Items marked *(GeoGebra-inspired)* come from a concept review of the GeoGebra source. GeoGebra is GPL-licensed: implement these from the named public algorithms, never by porting its code.

## The path to the HUD

A HUD that explains real problems needs the abilities below. MatHud builds them on the desktop first, where each can be tested and measured, and keeps one core throughout: deterministic tools, a scene state that the user and the model both see, and checked results. That core, together with the scope rule above (practical math from the graphing calculator tradition, aimed at real problems), is what sets MatHud apart: general assistants (ChatGPT, Claude, Gemini) now draw interactive math visuals, but those are not checked and not fed back to the model as state it can build on, while GeoGebra's suite and research tools reach well beyond practical math. A HUD that people act on needs persistent, inspectable, correct state and math chosen for real problems. MatHud also runs on local models and measures itself with its scenario suite. The README's Vision section lists these differences.

| What the HUD needs | Why | Built by | State |
|---|---|---|---|
| Understand intent from speech | On glasses, conversation is the main input | AI-first chat and tool layer, `search_tools`, providers (done); speech input (A3.6) | chat done, speech open |
| Correct, checked math | A wrong answer laid over the real world misleads | A0 audits, CAS reliability (A1), closing the math set (A2), step-equivalence checker (A1) | mostly done, math list in progress |
| A scene the AI can reason over | The model must know exactly what is there and how it relates | canvas text state and change reports (done), named datasets (A2.20), dynamic geometry (A2.11, A2.12, A2.16) | partly done |
| Explain visually and aloud | The HUD's job is understanding, not only answers | explanation layer (A3), Kokoro read-aloud (done) | next direction |
| Model real situations | Problems come from the world, not typed equations | photo to math (A4): underlay, calibration, AI-assisted extraction | planned |
| Run on small, local models | Glasses need low latency, privacy and offline use | LocalAgent and the model workbench (A1) | in progress |
| Follow live, changing input | The world moves, and the drawings must follow it | parameters and sliders (A2.1), dynamic geometry (A2.11), then a live camera (Part B) | planned |
| 3D as well as 2D | The world is three-dimensional; a HUD or VR scene needs 3D objects and views | Part B: 3D models, graphing and camera, as another renderer over math-only models | after 2D dynamic geometry |
| Other platforms | The HUD runs on a phone first, then glasses and VR | desktop shell (done), Android (Part B), then the wearable HUD; possibly in another stack | later |
| A wearable display | The last step | Part B: AR overlay, world anchoring, gaze, voice-first; VR | long term |

Stages, each useful on its own:

1. **Workbench** (now): correct math on a desktop canvas, with local models, measured by the scenario suite; data arrives as named datasets.
2. **Explainer**: the AI walks through a result step by step on the canvas and reads it aloud.
3. **Live scenes**: constructions follow changing inputs (sliders, dynamic geometry).
4. **Real world**: a photo becomes a calibrated model (A4), then a live camera on a phone or tablet (Android).
5. **HUD**: the same tools and explanations, in 3D as well as 2D (3D arrives in this app first), drawn over the world on glasses or placed in a VR scene.

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
- ~~**Comprehension benchmark for the canvas format**~~ Done: `scripts/benchmark_canvas_formats.py`. On OpenRouter (DeepSeek V4.1 Flash, MiMo V2.6 Pro) both formats answer every static-scene question, and `text` needs about 45% fewer prompt tokens and is faster and cheaper; after a tool batch `json` sends no canvas, so 5 of 8 change questions are not answerable there, while `text` answers all of them. On the local GPU (Qwen3.8 27B, Ternary Bonsai 2 27B, medium reasoning effort) `text` and `min_json` each scored 37 or 38 of 38 static questions and 8 of 8 change questions, with the one miss per model split across the two formats (a reasoning loop that never answered, and one wrong nearest point); the old LocalAgent count line left 40 of 46 questions unanswerable. `min_json` stays as the fallback format. Still open: tune the local budget to the model's context size.
- ~~One user-visible undo step per AI action (nested manager archives currently create several).~~ Done: each model reply (all its tool batches, one undo group per chat turn) and each workspace restore is one undo step.
- ~~Serialize drawable colors/styles (points, segments, vectors, circles, polygons, graphs, function curves) so they survive reload and reach the model; colours not stored in drawable state are missing from every canvas format, and `text` also omits default colours.~~ Done: those drawables (and ellipses) store a non-default colour in their state, so it survives save/load and undo and shows in every canvas format; the formats omit only each type's default colour (blue for angles, black otherwise). Angle names and grid visibility survive a load too.
- ~~OpenAI Responses API keeps earlier turns (including their `<canvas>` blocks) in server-side history via previous_response_id.~~ Done: responses are no longer chained, so each request sends MatHud's own stripped history (older canvas blocks, images and view notes removed), with `store=False`.
- ~~`localStorage` mirror of the canvas so an accidental page reload doesn't lose work.~~ Done: `static/client/canvas_backup_manager.py` mirrors the canvas and the view, and `main.py` restores it on page load (not the chat).
- ~~Custom names for circles (currently always `<center>(<radius>)`; needs a custom-name flag honoured by `regenerate_name()` and `__deepcopy__`).~~ Done: `create_circle` keeps a requested name such as `c1` through transforms, undo and reload, as `create_ellipse` does for ellipses; a name in use gets a `_<n>` suffix.
- ~~Drawable names are unique only within some types: each manager checks its own type (ellipses check every drawable, but only when an ellipse is named), so a function or label created later can take an ellipse's custom name, and a segment and a vector, or a point and a graph, can share a name (`AB`, `G`; scenario invariant I1 warns about these). `rotate_object` and the other transform tools then reach whichever comes first, and an ellipse whose custom name a drawable restored before it also holds comes back from a reload with a `_<n>` suffix, losing the areas that refer to it. Needs one name-uniqueness check shared by every manager (`DrawablesContainer.names_in_use`).~~ Done: `DrawablesContainer.unique_name` gives a name another drawable has the first free `_<n>` suffix, the point, function, parametric curve and label name generators avoid every name in use, and graphs, angles, arcs, bars, parametric curves, colored areas, graph vertices and point renames use the same check. Segments, vectors and polygons still take their names from their points.
- ~~Region boolean operations ignore holes and use only outer boundaries when results are combined further.~~ Done: a combined region is a signed sum of simple pieces (holes subtract), so nested `&`, `|`, `-` and `^` keep every hole, and its drawn boundary fills exactly that region. Of each pair of pieces that meet, one must still be convex.
- ~~Undirected graph analysis collapses parallel edges (a doubled edge is still reported as a bridge).~~ Done: a doubled edge is not a bridge, Euler status counts each parallel edge (a loop adds 2 to its vertex's degree), and highlights pick the lightest of parallel edges.
- ~~Server-side mypy errors in `cli/server.py`, `static/canvas_state_formatter.py`, `static/providers/__init__.py` and `cli/scenarios/live_config.py`.~~ Done: fixed, and those modules are in `mypy.ini`'s file list. (Route tests no longer reach a live provider: `server_tests/conftest.py` blanks the keys unless `MATHUD_LIVE_TESTS=1`.)
- ~~Correctness audit of the non-symbolic math tools (October 2026).~~ Done: bar charts with zero values, QR least squares, shifted polynomial expressions, negative-data logistic fits and negative R², exact descriptive sums, interior singularities in `numeric_integrate`, exact and bounded number helpers; arcs that follow their circle through transforms, convex-only regular polygons, zero-length construction guards, direction-vector parametric tangents, scale-free relation checks, positive radii; graphs that no longer merge on shared points, exact diameter, tree and DAG validation, deterministic traversals, grid layouts without stacked vertices, graph translation; shading against unbounded functions, circle cuts by a segment's left half-plane, exact curved region areas, roots at domain edges and cuts, parametric paths broken at poles, and nerdamer's decimal mode restored after a failed numeric evaluation. The CAS side is the CAS reliability item under A1.

## A1. Unified app and model workbench

### Single desktop app
Goal: one command (later one executable) opens a MatHud window; no separate server and browser.
1. ~~**Desktop shell via pywebview**~~ Done: `mathud_desktop.py` / `python -m cli.main desktop` starts Flask in a background thread and opens a native window (Edge WebView2 on Windows); closing the window stops the server.
2. ~~**Vendor browser libraries** (Brython, math.js, nerdamer, MathJax) into `static/vendor/` with pinned versions, so the app works fully offline with LocalAgent.~~ Done: `scripts/vendor_js_libs.py`.
3. ~~**Vision snapshots without Selenium/Firefox:** capture the canvas client-side and send it with the request; drop the headless-Firefox startup dependency.~~ Done: `static/client/canvas_snapshot.py` composites the SVG and Canvas2D layers; `WebDriverManager` and `/init_webdriver` are removed.
4. ~~**Lazy-load optional heavy pieces** (Kokoro/torch) so startup is fast and memory is low when TTS isn't used.~~ Done: startup ~1 s / 72 MB instead of ~7 s / 1.4 GB.
5. **Packaging** (later): PyInstaller one-folder build + Start-menu shortcut.

### Model workbench
1. ~~**Per-response metrics** shown in chat and logged: provider, model, latency, time-to-first-token, tokens/s, prompt/completion tokens, number of tool calls, tool errors.~~ Done: footer under each answer, JSON log lines, `window.getMatHudLastTurnMetrics()`.
2. **Benchmark suite (CLI):** *(mostly done)* a curated set of math prompts with machine-checkable expectations (canvas state or tool results), run against a list of models; outputs a comparison table (accuracy, tool-call validity, speed). Builds on the existing tool-discovery benchmark and action traces. Design: `documentation/development/agentic_scenario_testing.md`.
   - ~~**Replay mode**~~ Done: `python -m cli.main test scenarios --mode replay` runs the scenario catalogue in `scenarios/` (94 scenarios) through the app's real tool path in headless Chrome with no model, checks every step by geometry and invariants, and reports open known bugs as expected failures; fixed ones stay in the catalogue as regression guards.
   - ~~**Live mode**~~ Done: `--mode live` sends the prompts to LocalAgent (default) or OpenRouter through `sendMatHudMessage`, with per-turn caps, spend guards, `--repeats`, retrace (`--mode retrace`) and failure classification (app, model, nondeterministic, known, infra) in a per-model summary. Live runs are manual, on the workstation's GPU.
   - ~~**First local-model run**~~ Done: two smoke-subset runs on a local Qwen at medium reasoning effort (the first: 18 turns in 232 s, 6 of 11 scenarios passing, every failure classed `model`); the app and checker problems they found are fixed (vl3c/MatHud#98 and #105). Still open: a live run of the whole catalogue, and the same run on a second model for a side-by-side table.
   - ~~**CI job**~~ Done: the `scenario-replay` job in `tests.yml` replays the catalogue on every push and pull request. It never calls a model: CI has no keys and no GPU, so it runs only the recorded reference tool calls. Still open: make it blocking once it is stable.
3. **Side-by-side mode:** send the same prompt to two models (pairs naturally with tabs, A3).
4. **Local-model tuning:** tool descriptions and search-first prompts tuned for small local models; measure with the suite. LocalAgent now sends a reasoning effort (`chat_template_kwargs.reasoning_effort`, default `medium`, `MATHUD_LOCAL_REASONING_EFFORT`): without it Qwen3.8 27B thought to the 16,000-token limit on a trivial question. The live smoke runs also tuned local tool-search ranking (graph tools for graph requests, vl3c/MatHud#105).
5. **Small-model reliability** (from the October 2026 landscape survey):
   - *Variant scenarios*: several versions of a scenario with random coordinates or coefficients, reporting each model's worst-case accuracy (DynaMath found worst case far below average). Replay needs templated reference calls; otherwise variants run live only.
   - *"Does the canvas help?"*: Visual Sketchpad-style geometry and graph tasks run with and without the canvas on the local model, building on `scripts/benchmark_canvas_formats.py`. A measured answer to the project's central claim.
   - *Construction benchmarks*: a subset of GeoBuildBench (natural language to construction, graded by constraints) as live scenarios.
   - *State, then verify*: the model states the relations it intends ("AB ⟂ CD") and `inspect_relation` confirms them after building; a scenario rule first, a prompt habit later.
   - *Error hints and tool companions*: a known-pitfall hint on common tool errors, and `search_tools` returning the tools usually called together, to save round trips.

### CAS reliability
1. ~~Known-answer audit of `derive`, `integrate`, `limit`, `solve`, `simplify`, `factor`, systems against nerdamer (unmaintained).~~ Done: 237 cases. nerdamer alone answered 173 correctly and was confidently wrong on many of the rest (ln read as a variable, lim 1/x at 0 = Infinity, lost cubic roots, garbage antiderivatives); raw SymPy answered 228.
2. ~~Numerically verify symbolic results where possible (substitute roots back, compare derivative/integral numerically).~~ Done: every CAS tool checks nerdamer's result with math.js (`static/client/utils/symbolic_checks.py`, `cas_numeric.py`, `cas_text.py`) and refuses or replaces a wrong one; 226/237, and the 11 left are honest errors rather than wrong answers.
3. ~~Decide on a server-side SymPy fallback~~ Done: SymPy on the server is the main engine for `limit`, `integrate`, `solve` and `simplify` (`static/sympy_cas.py`, `/api/cas`, a safe `ast` parser and a 6 s worker time limit), with nerdamer kept for `derive`/`expand`/`factor` and as the offline fallback. 237/237 on the audit.
4. ~~Calculator functions the evaluator lacked (October 2026).~~ Done: `mod` and `%` as the remainder, `nthroot`, `cbrt`, `atan2`, `sign`, `gamma`, `erf`, and `re`, `im`, `arg`, `conj` for complex numbers are registry rows (Python, math.js, plots and the SymPy table); `sin^2(x)`, `sin^-1(x)`, `ln|x|` and spelled-out superscripts such as `sine²(x)` are read as calculators read them; `simplify` is the tool for exact fractions and radicals, and tool search sends "exact" and "remainder" requests to the right tool. Scenarios MC-05 and MC-06 cover them.
5. ~~Statistics fixes (October 2026).~~ Done: `compute_descriptive_statistics` names its spread figures `sample_*` (n - 1, null for one value) and `population_*` (n) instead of the ambiguous `standard_deviation` and `variance`, and with `y_data` returns two-variable statistics (Pearson's r, r squared, both covariances, and each list's statistics); a linear `fit_regression` reports the same r. Expressions gain `pstdev` and `pvariance`, and refuse math.js's misleading normalization argument (`std(L, "biased")` divides by n + 1). Scenarios ST-04, ST-06 and ST-07 cover them. Still open: Spearman's r and the sums (Σx, Σx², Σxy), frequency lists, the aliases `stdevp`/`varp`, and the math.js `stdev([5])` = 0 divergence.
6. **Step-equivalence checker** — given a list of expressions or equations, say whether each step is equivalent to the previous one and point at the first wrong step; checks the model's own multi-step derivations and a user's worked solution (including from a photo). ~~Expressions (October 2026).~~ Done: `check_steps` takes 2 to 12 expressions and gives each step a verdict (equivalent, not_equivalent, undecided) with the first wrong step and a summary (`static/client/utils/step_checker.py`). A stricter numeric check (`utils/equivalence.py`) varies every variable over 24 Kronecker points in each of four ranges, needs 8 agreeing points, and calls a step wrong only on two mismatches or one clear one; SymPy's `equivalent` operation (`static/sympy_cas.py`) settles the pairs it cannot, within a 12 s budget per derivation. A wrong step's reason gives SymPy's difference and a readable counterexample (`step 2 - step 1 = -2*x (not 0); at x = 1, step 1 = 3 and step 2 = 1`); a step that holds only where both sides are defined (`log(x^2)` to `2log(x)`) is equivalent with a domain note. Without SymPy the numeric check answers alone and says so. Tools accept standard syntax only; unclear spellings are refused with a hint (V, 2026-10-17). Other names the engines would misread (`sinxy`, `xy` against x and y, `x2` beside x, `gamma` as a variable) make the step undecided with a hint too. A name only one step has (`sqrt2`, `sinxe`, `mx`) never makes a pair not_equivalent: the pair is undecided with a hint; `solve` refuses names built from function names (`sinxe`, `ln2`, `sinxcosx`) and a variable missing from the equation (`y = mx + b` for x). Scenario MC-08 covers it. Next: equations and inequalities (solution sets, steps that gain or lose roots), refused today with a hint. Known minor gaps (left for later, October 2026):
   - A removed hole is not noted: `(x^2-1)/(x-1)` to `x+1` is equivalent with no note about x = 1; differences only at integers are not seen (`floor(x)+1` and `ceil(x)` count as equivalent).
   - Constants that differ only slightly (`pi/4` against `0.785398`) are undecided without SymPy.
   - `xy` against `x` and `y` is undecided (with a hint to write `x*y`) rather than read as a product.
   - Real cube roots of negatives follow math.js's principal root: `x^(1/3)` has no value for negative x (a step to `cbrt(x)` gets a domain note), and `(-8)^(1/3)` is not `-2`.
   - Not read yet: `arcsin`/`arccos`, `x|x|`, `∛`, `½` and `log_2(8)`.
   - A glued-looking name that holds no other variable stays a variable: `solve` answers `profit = revenue - cost` for revenue as main does, and check_steps calls `revenue - cost` and `-(cost - revenue)` equivalent with a note. A name that starts with a function name is a misspelling in check_steps when the steps differ, so `seconds*60` against `61*seconds` is undecided rather than not_equivalent; `costheta` gets the general hint, not `cos(theta)`.
   - `misspellings` (`utils/cas_text.py`) takes quadratic time on 3000 bare calls.
   - `sign(x) = 1` gives [0] and `floor(x) = 2` gives [2], the same as main.
   - The tool search ranks check_steps first for "test whether the data is normally distributed".
   - A space between two names inside or after a call (`θ ln(θ)`, `x sin(x)`, `sin(pi x)`) is read by math.js but not by SymPy, so only the numeric check decides; `pi r^2` likewise; `d/dx x^2` is read as the variables d and dx.
   - `x/2y` is x/(2y) to math.js and (x/2)*y to SymPy.
   - `E` is read as e.
   - `derive` (nerdamer) reads `xe^x` as one name without saying so; only `solve` and `check_steps` refuse it.
   - `solve` gives [] for `k*e^x = 2` and `A*e^(k*t) = 2*A`, as on main.
   - The sampling range is not widened when too few points are defined: without SymPy, `sqrt(x-300)^2` to `x - 300` is undecided.
   - A step that overflows at the sample points is reported as "never both defined".
   - The worst-case wait is about 12 + 6 s (the budget, then one last SymPy request).
   - `pi/4` against `0.78539816` counts as equivalent (relative tolerance 1e-8).
   - Values in reasons are printed to 6 significant digits (more only where needed to tell them apart).
   - Tool search: "compare two expressions" and "check if x = 2 is a root of x^2 - 4" rank `check_steps` first, "did I simplify A to B correctly?" ranks `simplify` first, and `check_steps` takes rank 2 or 3 on some other math queries. (Units and fractions, "the equivalent of 5 miles in km" and "equivalent fractions of 1/2", keep their own tools.)
7. ~~Inequalities in `solve` (October 2026).~~ Done: `solve` answers inequalities in one variable (`<`, `<=`, `>`, `>=`, `≤`, `≥`, chains such as `1 < 2x+1 <= 5`) with the solution set as inequalities and in interval notation. SymPy solves them (`solve_inequality` in `static/sympy_cas.py`; a periodic one on one period, shifted across the interval), the client checks the set with math.js on a dense grid and beside its ends, and a refuted or missing answer gives way to a numeric sign chart that says it was found numerically (`static/client/utils/inequality_solver.py`, `real_sets.py`). An optional `interval` restricts the answer, brackets honoured, and also lists the roots of an equation in an interval (SymPy `solveset`; nerdamer's roots are filtered). Tool search sends inequalities to solve without taking piecewise rules or shading from their tools; scenario MC-07 covers it. Still open: `!=` and `and`/`or` between inequalities (refused with a hint), and inequalities in two variables, which are shading (A2.7). Known minor gaps in the numeric sign chart (left for later, October 2026):
   - Isolated points of step functions are missed: `ceil(x) > floor(x)` on [0.5, 3.5] keeps 1, 2 and 3.
   - Some ends near 0 come out as tiny numbers: `(exp(x)-1)/x > 1` gives an end of 9.99e-16 instead of 0.
   - Two touching roots inside one grid gap give only one root: `abs(x-50)*abs(x-50.3) <= 0` gives {50}.
   - `nice_number` snaps roots above about 10^6 to wrong fractions: for `x^2 > 1e15` it gives 158113883/5.
   - Ends at touching roots show more digits than are accurate.
   - "No real x" is stated beyond the checked reach of ±10^6.
   - `cas_numeric.polynomial_roots` merges close roots and drops a tiny leading coefficient.
   - Nested absolute bars `||x|-1|` are misread.
   - math.js's domain makes `x^(2/3)` undefined for x < 0.
   - The test harness re-implements `InequalitySolver._seeds` instead of calling it.
   - An implicit product after a space is refused, in equations too: `x sin(x) > 0`, `sin(pi x) > 0`.
   - `solve` gives wrong answers on main too for some equations with parameters: `exp(k*t) = 2` for t gives log(2) instead of log(2)/k, and `N = N0*e^(-k*t)` for t gives []. `5 = 5` now says the variable does not appear, where main said identity.
   - The decimal endpoints of `x^2 - 1e6 x + 1 > 0` lose precision through cancellation; the exact form is right.
   - A periodic inequality on a long interval is answered on one period: `sin(x) > 0` on [0, 1000].
   - High-degree polynomials and `exp(x) - x^100 > 0` take 5 to 8 s, and the call blocks the UI.
   - Tool search still misses `solve` for "for what values of x is x^2 - 4 >= 0", and ranks it first for "find out if 2^10 > 10^3".
8. ~~Taylor polynomials and partial fractions (October 2026).~~ Done: the `taylor_polynomial` tool gives a function's Taylor polynomial at a point up to order 10 (a Maclaurin polynomial at 0) in powers of (x - a), ready for `draw_function`, with its first omitted term and a note on the error (`static/client/utils/taylor.py`; SymPy's `taylor` operation in `static/sympy_cas.py`). SymPy takes the coefficients from the derivatives at the point when its series would be slow (away from 0, as for `e^(sin(x))` at pi/3, or an exponential of a non-polynomial such as `e^(e^x)` at 0) and the derivatives are all finite real numbers and stay small, and otherwise from the series on both sides of the point, so a corner or a jump that a one-sided series hides (`abs(x)` at 0 is x from the right) is refused, as are poles, fractional powers and logarithms; a function that is the polynomial says "Exact", a polynomial of higher degree names its true next term (`x^14` to order 10), and a flat one (`exp(-1/x^2)` at 0) says so instead of claiming exactness. The client checks the polynomial against the function's math.js values beside the point (`utils/series_checks.py`: the error must shrink like h^(n+1) as the step halves). `expand` takes a nullable `variable`: a ratio of polynomials in it is split into partial fractions over the rationals (SymPy's `apart`; irreducible quadratics stay whole, a single fraction is noted as not splitting further, denominators up to degree 10), checked with the step checker's numeric check; anything else (a polynomial, a second variable, a variable the expression lacks) is expanded as before. Both refuse an ambiguous division such as `1/(x+1)(x+2)` with a hint to write `1/((x+1)(x+2))`. Neither has a nerdamer fallback: without SymPy they say so. Scenarios MC-09 and MC-10 cover them. Known minor gaps (left for later, October 2026):
   - The numeric check cannot see a tiny relative error in a high coefficient (order 10 of e^x with 1/3628801), and leaves flat functions unchecked; SymPy's result is trusted there. For quotients whose math.js values lose precision near the point (`(1-cos(x))/x^2`, `(2^x-1)/x`, `(tan(x)-x)/x^3`) it reads only the steps before that rounding and never refutes after it, so a wrong polynomial there is often left unchecked rather than refuted. A coefficient off by a few percent or less whose error passes through 0 near the point (`sqrt(1+x)` to order 3 with 0.063125 for 1/16) can be passed, since the error just past that zero looks like rounding.
   - The error note covers TAYLOR_LOOKAHEAD (3) orders beyond the polynomial: `cos(x^10)` to order 5 says there are no further terms up to x^8.
   - Order 10 of a product such as `exp(sin(x))*ln(1+x)` at 0 or `x^x` at 2 takes about 2 s, and of `e^(x*sin(x))` or `e^(x*ln(1+x))` at 0 about 3 s (the derivatives are given up after 2 s for the series); heavier inputs can run out of the 6 s worker time and ask for a lower order.
   - A fractional power smooth to the order is refused when it breaks within the lookahead (`x^(7/3)` at 0 to order 2), and `max(x, 0)` away from its corner gets SymPy's generic error (Max has no series); a point outside the domain (`ln(x)` at -1) is called "not smooth", and the refusal names "logarithm" for `ln` too.
   - Spelling hints for names such as `arctan`, `arcsin` and glued products (`x e^x`) are the app-wide input pipeline's, not specific to these tools.
   - `x^(1/3)` at a negative point is refused as not smooth (the principal root is complex there); the refusal could suggest `cbrt(x)`. `(1+x)^(1/x)` at 0 past order 6 and at 1 past order 3, `e^(x*cos(x))` at 0 past order 7, `x^sin(x)` at 1, `atan((1+x)/(1-x))` at 0 past order 7, `e^(sqrt(1+x))` at 0 past order 6 and `ln(x+sqrt(1+x^2))` at 1 at order 10 run out of time. At points without a simple closed form the coefficients can be long unsimplified expressions (radicals of the point), and a polynomial longer than the 1000-character CAS input limit cannot be passed on to other CAS tools.
   - `expand` with a variable on a ratio with a parameter (`1/(x^2-a^2)`) expands as usual without saying that partial fractions need one variable; "apart 1/(x^2-1)" finds no tool; `(x^2-1)/(x+1)` with a variable is expanded as usual rather than reduced to `x - 1`.
   - A function with a parameter (`(1+x)^k`, `e^(a*x)`) is refused for naming another variable; the order must be a number, so the string "3" is refused rather than read as 3.
   - `e^2x` is read as `e^2*x` (as by the solve tool), not refused with a hint to write `e^(2x)`.
   - Fractions are ordered by their denominators (x, x - 1, x + 1, x - 2), so `1/(t^2-t)` reads `-1/t + 1/(t-1)`; coefficients such as pi in `1/((x-pi)(x+1))` are kept, not refused.
   - Without SymPy, `expand` with a variable decides "ratio of polynomials" by math.js's `rationalize`.

### Conversation persistence
~~Save and restore the chat transcript with the workspace (truncate older turns on restore).~~ Done: workspaces (schema version 2) carry the user and assistant messages as markdown, with image counts and compact tool-call entries, capped at 200 messages / 400k characters. `/load` replaces the chat and rebuilds a plain-text AI history from it (newest 40 turns, about 10k tokens) through `/restore_conversation`; the AI's `load_workspace` tool restores the canvas only, so it never replaces the chat mid-turn. Older workspaces load with an empty chat. See `static/workspace_chat.py` and `static/client/chat_persistence_manager.py`. Still open: summarize dropped turns instead of only noting them; a provider created after the restore (switching to another provider) starts without the restored history, as it does for any conversation today.

## A2. Exploration features

Numbered by topic; the build order is in the next paragraph.

**Closing the math feature set (October 2026)** builds these, in this order: statistics fixes (sample and population SD, r and covariance; done), the SymPy gap-fill (inequalities, A1 CAS reliability 7, and Taylor polynomials and partial fractions, 8, done; closed-form sums) with the step-equivalence checker (A1, CAS reliability 6; expressions done), polar plots (4), arc length (5), distributions and then hypothesis tests (13), histogram, box and scatter plots (18), sequences (14), parameters and sliders (1), custom-model regression (19), implicit curves (7), and a plot-correctness pass (3, scoped down). Sliders come before implicit curves so implicit curves take parameters from the start. Dynamic geometry (11, 12, 16) follows as its own direction. Optional, as wanted: symbolic ODE solving (a later option), adaptive plotting beyond the correctness pass, calculus visuals, slope fields, conic analysis, tangent constructions, a table of values, gamma/beta/log-normal distributions, self-intersections of one curve. Parked by the scope rule (V, 2026-10-16): 5-point conics, pole/polar, circle inversion, phase portraits, vector fields, curvature and point sets. The gap-fill keeps three CAS-only features (Taylor series, partial fractions, closed-form sums) and leaves symbolic ODE solving for later (V, 2026-10-16). After the list come the step checker for equations and inequalities, then a refactor pass for readability and structure.

1. **Parameters and sliders** *(GeoGebra-inspired)* — named scalars with range/step (`a ∈ [-5, 5]`) usable in functions, parametric curves, point coordinates, radii; a small slider panel; AI can set or animate them ("sweep a from 0 to 2π"). Animation runs on `requestAnimationFrame` without undo entries per frame.
2. ~~**Roots, extrema, inflection points, intersections of plotted functions**~~ Done: the `find_function_features` tool finds roots, local extrema, inflection points and function-function intersections of plotted functions and piecewise functions (`utils/function_features.py`), and the `find_intersections` tool intersects any two segments, vectors, circles, circle arcs, ellipses (rotated too), functions, piecewise functions, parametric curves (two of them too) or polygons (`utils/object_intersections.py`: closed forms for lines, circles and line/ellipse, a numeric search along a conic or curve otherwise, polyline crossings refined by 2D Newton for two parametric curves in `utils/curve_pair_intersections.py`, and a polygon as its merged edges, a vertex crossing reported once with its edges and vertex), with each point's parameter on both objects, tangency flags, overlaps for coinciding objects and an `extend_lines` option; both can place points as one undo step. The function sampling refines itself where the grid does not resolve the curve, so features within a step or two of each other ((x^2 - a^2)^3 with a = 0.001, two roots 1e-4 apart, tanh(1e4 x)) are found, and inflection points are found far from 0 (up to about |x| = 1e9 in an interval of width 10). Still open: self-intersections of one curve, and implicit curves once they exist (item 7).
3. **Adaptive plotting with numeric discontinuity detection** *(GeoGebra-inspired)* — Gillam-style bisection on distance + turning angle; continuity test at max depth to split jumps/asymptotes; same sampler for parametric and polar curves. Replaces the string-based asymptote guessing (keep it only as seeds).
4. **Polar plots** `r(θ)` — thin wrapper over the parametric plotter.
5. **Adaptive numeric integration and arc length** *(GeoGebra-inspired; numeric integration already has an error estimate, so only arc length remains, in the closing list)* — Gauss–Kronrod G7/K15 or adaptive Simpson with error estimate; tanh-sinh for endpoint singularities; arc length for functions and parametric curves.
6. **Calculus visualizations** *(GeoGebra-inspired)* — Riemann sums (left/right/mid/upper/lower/trapezoid) as a grouped composite, Taylor polynomial overlay at a point (the polynomial comes from `taylor_polynomial`, A1 CAS reliability 8, and plots with `draw_function`), slope triangle; these explain textbook calculus. Curvature and the osculating circle are *(parked: scope)*.
7. **Implicit curves `F(x,y)=0` and inequality shading** *(GeoGebra-inspired)* — marching squares on an adaptive quadtree, segment linking; regions like `y < x^2`, `x^2+y^2 < 4` with AND/OR. Also gives contour lines and level sets.
8. **General conics** *(GeoGebra-inspired)* — conic through 5 points, classification (ellipse/hyperbola/parabola/degenerate), foci, directrix, eccentricity, asymptotes, pole/polar, tangents from a point. Renders via parametric or implicit plotting. Analysis of a typed conic (classification, foci, vertices, directrix, eccentricity, asymptotes, tangents from a point) is on graphing calculators (fx-CG50 Conic Graphs, the TI-84's conic app) and is optional, as wanted; the conic through 5 points and pole/polar are *(parked: scope)*.
9. **Circle inversion and tangent constructions** *(GeoGebra-inspired)* — reflect in a circle (lines/circles map to lines/circles), tangents from an external point, common tangents of two circles. Tangents are practical (belt and pulley lengths) and optional, as wanted; circle inversion is *(parked: scope)*.
10. **Slope fields and ODE solutions** *(GeoGebra-inspired)* — direction field for `y' = f(x,y)`; solution curves via Dormand–Prince RK5(4) with adaptive step; phase portraits for 2D systems; vector fields on the same base. Slope fields and solution curves are on the TI-Nspire and in GeoGebra (not the TI-84 or Desmos) and are optional, as wanted, for rates-of-change problems; phase portraits and vector fields are *(parked: scope)*.
11. **Dependent (dynamic) constructions** *(GeoGebra-inspired)* — derived objects store `{op, inputs, params}` and recompute in topological order when an input changes; undefined results hide instead of being deleted; definitions saved in workspaces. Builds on `drawable_dependency_manager.resolve_dependency_order()`. Unlocks loci and "is this true in general?" checks. It matters more in chat than in a manual tool: after "move A to (2,3)" the user expects the midpoint and bisector to follow, and on the HUD the inputs come from a moving world. Cinderella/CindyJS's continuity of intersections is the reference for keeping branches stable while inputs move. Once it exists, the canvas text should list dependency edges.
12. **Loci and points on paths** *(GeoGebra-inspired)* — a point constrained to a curve by parameter; the trace of a dependent point as a driver moves (or a slider varies), with adaptive step control.
13. **Probability distributions** *(GeoGebra-inspired)* — t, χ², F, exponential, uniform, binomial, Poisson, geometric (gamma, beta and log-normal are optional, after the list, since mainly GeoGebra has them); pdf/cdf/quantile and shaded `P(a < X < b)` (incomplete gamma/beta, Lanczos log-gamma, Brent for quantiles). Then hypothesis tests and confidence intervals. Each distribution is one `DISTRIBUTIONS` entry (`utils/statistics/distributions.py`); `StatisticsManager` already dispatches on it. Still normal-only: the `plot_distribution` schema (enum, `distribution_params`, the "mean ± 4σ" `plot_bounds` descriptions), `Canvas.plot_distribution`'s `distribution_type="normal"` default (an omitted type plots a normal, an explicit null errors), `perf_benchmark.py`'s mean and sigma, and the normal/gaussian tool-search keywords.
14. **Sequences, iteration, cobweb diagrams** *(GeoGebra-inspired)* — `sequence(expr(k), k, a, b)` creates many objects in one call (one undo step); `iterate(g, x0, n)` for orbits and cobweb plots.
15. ~~**Linear algebra hardening**~~ Done: tests for `eigs`, `lup`, `qr` against the real math.js; `eigs`/`lup`/`qr` object results now convert to plain lists and dicts (they used to come back as `unknown`); added `rref(A[, tol])` and `rank(A[, tol])`.
16. **Relation checks "in general"** *(GeoGebra-inspired)* — after dependent constructions exist: randomly perturb free objects, recompute, re-check the relation (randomized identity testing).
17. **Point-set geometry** *(GeoGebra-inspired; parked: scope)* — Delaunay (Bowyer–Watson), Voronoi as its dual, TSP tour (nearest neighbour + 2-opt).
18. **Scatter plot and histogram with binning** — small additions on existing plot infrastructure. A new plot kind is one `PLOT_KINDS` row (`static/client/plot_kinds.py`) plus its formatter spec; a histogram may instead be a `BarsPlot` with binning.
19. **Custom-model regression** — fit any model the user gives with named constants (`a·e^(-b·x)·cos(c·x) + d`), with residuals, as Desmos `y1 ~ f(x1)` and GeoGebra `Fit` do; `fit_regression` has 7 fixed models. The fitted constants become parameters (item 1), so it follows sliders. Starting values are the hard part: seed with a grid or several starts, as the logistic fit already does.
20. **Named datasets** — attach or paste a CSV once, show it as a table, and let the statistics, regression and plot tools take the dataset's name instead of every number in every call (which costs tokens and invites copying errors). Needs a non-image attachment path, a table view, a saved state bucket shown in the canvas text, and schema changes across those tools, so it comes after the math list and the refactor pass, done once over the finished tool set. Absorbs Part B's tabular data workflow.

## A3. Explanation, UI and workflow

**Explanation layer** — the next direction after the math list, the refactor pass and named datasets (stage 2 of [the path to the HUD](#the-path-to-the-hud)). MatHud can build and compute but has few ways to *show* a result step by step; every AI-first tool that explains well (GeoGebra's construction protocol, OpenMAIC's spotlight and laser actions timed to speech, narrated explanation agents) has this layer. Order: style and visibility (7), highlight (4), walkthrough (8), turn replay (9), visible checks (10), export (3), click to reference (11). Each is one tool call where possible, so small local models can use it.

1. ~~**Math symbol input for the chat box**~~ Done (vl3c/MatHud#70): the Σ button (or `Ctrl+↑`) next to the chat input opens a grouped palette (Operators, Calculus, Sets & logic, Geometry, Greek, plus a Recent row) that inserts at the caret; `\alpha`-style completion and Alt shortcuts (`Alt+P` → π); and Unicode-aware parsing, so the same text works when passed into tools (`static/client/math_symbols.py`, `math_symbol_palette.py`, `math_symbol_input.py`).
2. **Tabs for parallel problems** — each tab holds its own canvas state, chat history, and selected model. Implement by swapping state through the workspace serializer (depends on the A0 round-trip fixes) rather than running multiple `Canvas` instances; AI tools `create_tab`, `switch_tab`, `list_tabs`. Enables side-by-side model comparison (A1).
3. **Export** — canvas to PNG/SVG; session (chat + results) to Markdown/LaTeX.
4. **One `highlight`/`annotate` tool** — lets the AI point at objects and label them temporarily (replaces the earlier list of 11 temporary-drawable types).
5. **Dark mode.**
6. **Speech input** (TTS output already exists). With highlights and read-aloud it gives "talk and point", locally; on glasses it is the main input.
7. **Object style and visibility** — hide/show, dashed or dotted lines, width and opacity for any drawable ("construction lines dashed, result solid"). There is no per-object hide/show, line style or width today (tools take `color`; areas also take opacity, labels font size and rotation). Hide/show needs a new `visible` flag: `is_renderable` cannot serve, because triangles, polygons, graphs and most plots are drawn through their parts (segments and points, or bars, curves and areas), so hiding or dashing a polygon acts on parts another object may share. Dashes and widths are new in Canvas2D (the frozen SVG renderer may ignore them), and each style must reach saved state, workspaces, the canvas text and undo.
8. **Walkthrough / step reveal** — one tool call with an ordered list of steps, each with text and the drawables it introduces or highlights; steps reveal one at a time, optionally read aloud with Kokoro.
9. **Turn replay** — step through what a turn built (GeoGebra's construction protocol). Undo groups are one per turn, so replay uses the per-batch action traces and `replay_trace` (`managers/action_trace_collector.py`), re-executing older traces from the state archived in the turn's undo group. Traces live in memory only (newest 100 batches), so the first version covers the current session; replay after a reload needs trace persistence. Replaces Part B's construction history timeline.
10. **Visible checks** — mark each CAS result "checked numerically" or "unchecked". A result the check cannot decide passes today with no mark, so each CAS method reports three states (passed, failed, undecided), using the stricter check built with the step-equivalence checker.
11. **Click to reference** — clicking an object puts its name into the chat input ("selected: A, c1"), so the user can say "make this one dashed". Extends the double-click that already inserts coordinates; needs hit-testing, which the client does not have yet.

## A4. Photo to math (real-world problems)

Goal: take a photo and let the AI model a real problem in it mathematically — measurements, trajectories, areas, angles, perspective — not just describe it.

1. **Photo as a canvas underlay** — place an image behind the math canvas with an adjustable transform (scale, rotation, offset); it pans/zooms with the canvas.
2. **Calibration** — the user or AI marks a known length (reference object) to set real-world scale; optional 4-point perspective rectification (homography) for photos of planar surfaces.
3. **AI-assisted extraction** — a vision-capable model proposes points/segments/curves on the calibrated image as normal drawables; the user confirms or adjusts; then regular tools compute (areas, angles, curve fits for trajectories, etc.).
4. **Worked-problem flow** — "how tall is this tree", "what's the area of this room", "fit the ball's path" as benchmark prompts in the model workbench.

Heavier segmentation (SAM) and OCR pipelines are in Part B; modern vision models cover most of this without them.

This is stage 4 of [the path to the HUD](#the-path-to-the-hud), and it rests on the stages before it: extraction produces ordinary drawables and datasets, dynamic geometry keeps the model consistent when a calibration point moves, and the explanation layer presents the answer. Solving a photographed homework problem is not the goal (Photomath, Mathway and Symbolab do that); turning a real scene into a calibrated, checkable model is what no product found does.

---

# Part B — Optional / future / separate projects

Not scheduled. Worth revisiting once Part A is solid, or better built as a separate app. Detailed step lists for these items, grouped into spinoff app ideas, are in [`archive/Roadmap_2026-09_full.md`](archive/Roadmap_2026-09_full.md).

### Platform
- **Android app** — the next platform on the way to glasses (live camera, stage 4 of [the path to the HUD](#the-path-to-the-hud)); likely a thin client to a desktop/remote MatHud server, or a WebView wrapper once the desktop shell (A1) exists, or a native rewrite.
- PWA, touch/stylus input, shareable links, notebook-style cell mode.
- **MatHud as an MCP server**, so any agent (Claude Code, Codex) can use the canvas as a visual sketchpad; community GeoGebra MCP servers show the demand.
- Accessibility baseline: ARIA live regions for answers, and the canvas text summary as a screen-reader alternative.

### Advanced image analysis
- Server-side Meta SAM segmentation (point/box/text prompts), segment-to-canvas projection.
- Shape detection (OpenCV), math OCR (pix2tex-class models), curve fitting from images.
- Live camera mode on a phone or tablet: the A4 pipeline on a video stream (stage 4 of [the path to the HUD](#the-path-to-the-hud)).
- Physics from images: trajectories from video frames, structural geometry, symmetry/pattern detection, perspective geometry.

### Canvas and workflow
- Reusable macros and analysis templates (the construction history timeline is now A3.9).
- Session snapshots with diff/restore, "what changed" summaries, command palette.
- ~~CLI replay of action traces.~~ Done: `python -m cli.main test scenarios --mode retrace` replays a live run's executed tool calls on a fresh canvas.
- CSV export, plotting from table columns (CSV import is now A2.20, box plots are in the math list, residuals come with A2.19).
- Numerical-methods visualizations (Newton iterations, bisection steps).
- Graph workflows: random/complete generators, edit tools, algorithm playback.
- Heatmaps.

### 3D
- 3D graphing, shapes, constraints, camera navigation — in this app (desktop first), after 2D dynamic geometry, as another renderer over math-only models; then it carries over to glasses and VR. Part of the long-term vision, not an optional extra; it is in Part B only because it is not scheduled yet.

### Collaboration and education
- Real-time collaborative sessions, teacher/student mode.
- Guided lessons, quiz/challenge mode, step-by-step animation mode.
- AI-initiated suggestions, multi-turn planning UI.
- Optional "guide me" mode (hints first, the answer on request) for learning; studies found unrestricted AI help raised practice scores but lowered exam scores.

### Wearable HUD (long-term vision)
Stage 5 of [the path to the HUD](#the-path-to-the-hud): the tools, scene state and explanation layer built in Part A, drawn over the world on glasses, or placed in a VR scene.
- AR canvas overlay: world-anchored objects, ghosted solutions, live measurement lines, construction guides.
- Continuous scene understanding: live camera analysis, real-time segmentation, material recognition.
- Voice-first interaction: wake word, spatial audio, gaze-anchored context, dictation.
- Live analysis modes: structural, trajectory, surveying, symmetry, optics.
- Educational, practical/professional (navigation, cooking, fitness, DIY, astronomy) and social/multi-user AR.
- Architecture: edge compute, spatial anchors, gaze/gesture input, low-latency vision, battery-aware modes, privacy controls.

### Other parked topics
- Advanced math parked by the scope rule (see Direction; parked by V, 2026-10-16): conics through 5 points and pole/polar (A2.8), circle inversion (A2.9), phase portraits and vector fields (A2.10), curvature and osculating circles (A2.6), point-set geometry (A2.17).
- Truth tables and boolean algebra.
- Financial mathematics (compound interest, annuities, amortization).
- Q-Q / P-P plots, cumulative frequency tables.

### Dropped
- WebGL renderer parity (renderer removed).
- Semantic snapping, tolerance-policy management, conflict handling, resolution traces (depended on a snapping system that was never built; revisit only if direct-manipulation UX becomes a goal).
- Coordinate readout HUD overlay (tool results and canvas state already give the AI coordinates).

---

## Milestones

| Milestone | Focus | HUD stage |
|---|---|---|
| **1 — Stable** ✓ | A0 complete: known math bugs fixed, workspace round-trip, tool-result plumbing, canvas-state format, renderer fixes and speed | 1 |
| **2 — Unified app** ✓ | Desktop shell ✓, vendored libs ✓, client-side snapshots ✓, per-response metrics ✓, chat persistence ✓ | 1 |
| **3 — Workbench** | Benchmark suite (built; full-catalogue runs open), CAS audit ✓, side-by-side comparison, local-model tuning, small-model reliability (A1) | 1 (small local models are what the glasses will run) |
| **4 — Math complete** | The closing list in A2 (statistics fixes ✓, SymPy gap-fill (inequalities ✓, Taylor and partial fractions ✓) and step checker (expressions ✓), polar, arc length, distributions and tests, histogram/box/scatter, sequences, sliders, custom regression, implicit curves, plot correctness); roots/extrema/intersections ✓; then the step checker for equations and inequalities and a refactor pass | 1, 3 |
| **5 — Data** | Named datasets (A2.20) | 1 |
| **6 — Explain** | Explanation layer: style and visibility, highlight, walkthrough, turn replay, visible checks, export, click to reference (A3); speech input (A3.6, already on the Roadmap) fits here | 2 |
| **7 — Dynamic geometry** | Dependent constructions, loci, relations "in general" | 3 |
| **8 — Photo to math** | Image underlay, calibration, AI-assisted extraction | 4 |
| **9 — 3D, live camera and HUD** | Part B: 3D in this app (can start any time after dynamic geometry, independent of Android), Android with a live camera, then the wearable HUD and VR | 4, 5 |

Alongside, as wanted: tabs, dark mode, packaging, and the optional exploration items listed under A2's build order.
