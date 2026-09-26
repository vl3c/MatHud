# MatHud Project Overview

## Purpose
MatHud pairs a canvas with an AI assistant so users can sketch geometric scenes, analyze mathematically, and iterate quickly inside a single workspace.

## AI-First Product Direction
1. Primary UX is conversational: users tell the AI what they want; AI executes tool workflows.
2. HUD canvas is the output and inspection surface for AI actions, not the primary control channel.
3. Direct gesture interactions are optional accelerators and should not be required to complete core tasks.
4. New features should improve intent resolution, deterministic execution, and explainable AI actions tied to canvas state.

## Architecture at a Glance
1. Frontend: HTML plus Brython (`static/client/`) render the canvas, manage UI flows, and execute client tests inside the browser.
2. Backend: Flask (`app.py`, `static/`) exposes HTTP routes, workspace persistence, and AI provider calls (OpenAI, Anthropic, OpenRouter, local).
3. AI and vision: `static/functions_definitions.py` specifies callable tools; with vision on, the browser captures the canvas (`static/client/canvas_snapshot.py`) and sends it with the prompt as `canvas_snapshot`.
   Providers live in `static/providers/`; `static/providers/local/` holds LocalAgent, which serves whatever model a local llama-server reports from `/v1/models`.
4. Math tooling: nerdamer.js provides symbolic algebra, math.js handles numeric evaluation, and MathJax renders LaTeX.

## Core Capabilities
1. Canvas control: reset, clear, undo, redo, zoom, grid toggles, and math-bounds fitting.
2. Geometry primitives: create, edit, and relate points, segments, vectors, triangles, rectangles, circles, ellipses, and angles.
3. Mathematics: evaluate expressions, solve equations, differentiate, integrate, work with complex numbers, and run statistical routines.
4. Graph theory: create graphs, trees, DAGs; run analyses (shortest path, MST, topological sort, BFS/DFS).
5. Statistics: plot probability distributions (normal, discrete) and bar charts; compute descriptive statistics (mean, median, mode, quartiles, etc.).
6. Workspace operations: save, load, list, delete, import, and export named workspaces.

## Key References
1. `static/functions_definitions.py` lists every callable tool and its parameters (90+ AI function definitions).
2. `documentation/Project Architecture.txt` - deep dive into system design.
3. `documentation/Reference Manual.txt` - comprehensive API and module reference.
4. `documentation/Example Prompts.txt` - curated prompts for common workflows.

---

# Environment Setup

## Prerequisites
1. Python 3.11+.
2. An API key for at least one provider: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY` (local providers need none).

## Setup Steps
1. Clone the repository and create a virtual environment: `python -m venv venv`
2. Activate the environment:
   - Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
   - macOS/Linux: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Provide credentials by setting `OPENAI_API_KEY` in your shell or by creating `.env` in the project root.

## Configuration (.env)
```env
OPENAI_API_KEY=sk-...
AUTH_PIN=123456            # Optional: access code when auth is enabled
REQUIRE_AUTH=true          # Force authentication in local development
PORT=5000                  # Set by hosting platforms to indicate deployed mode
SECRET_KEY=override-me     # Optional: otherwise random key generated per launch
TOOL_SEARCH_MODE=hybrid    # Tool discovery: local | api | hybrid (default: hybrid)
LOCAL_AGENT_BASE_URL=http://127.0.0.1:8080  # LocalAgent llama-server (default shown)
MATHUD_LOCAL_REASONING_EFFORT=medium  # LocalAgent chat_template_kwargs.reasoning_effort: low|medium|high|xhigh|max; unset = medium; default = omit the field
MATHUD_TOOL_EXPOSURE=search  # search: search_tools + essentials up front (default) | full: every tool
MATHUD_CANVAS_FORMAT=text    # Canvas in prompts: text (default) | min_json | json (legacy prompt JSON)
MATHUD_CANVAS_BUDGET_TOKENS= # Canvas block cap; default 4000 (cloud) / 1500 (local), 0 = unlimited
```

## Running the App
1. Launch: `python app.py`
2. Open `http://127.0.0.1:5000/` in a browser (Chrome, Firefox, or Edge).
3. Stop with `Ctrl+C`.

### Desktop Window
`python mathud_desktop.py` (or `python -m cli.main desktop`) serves the app on localhost in a background thread and opens it in a native pywebview window titled "MatHud" (Edge WebView2 on Windows); closing the window stops the server.
1. pywebview is optional: `pip install -r requirements-desktop.txt`. Without it the launcher prints that command and offers the browser instead.
2. `--port N` picks the port (default 5100, falling back to a free port), `--browser` opens the default browser instead of a window, `--devtools` enables the WebView inspector.
3. Debug mode is off and no WebDriver starts. Window geometry and WebView storage live in the per-user data dir (`%LOCALAPPDATA%\MatHud` on Windows).
4. Kokoro TTS is imported on the first `/api/tts` request, not at startup (startup only checks the packages are installed).
5. Server tests for the launcher (`server_tests/test_mathud_desktop.py`) fake pywebview; they never open a window.

### Running on a Specific Port
If port 5000 is stale or occupied, start on a different port (e.g., 5004):
```bash
python app.py --port 5004
```
Then navigate to `http://127.0.0.1:5004/` in the browser.

## Cloud Sessions (Claude Code on the web)
1. The cloud environment's setup script installs `requirements-ci.txt` into `/opt/mathud-venv`, outside the repo, so the environment cache keeps it across the fresh clone each session gets.
2. A SessionStart hook in `.claude/settings.json` runs `scripts/cloud_session_start.sh` when `CLAUDE_CODE_REMOTE` is `true`; locally it does nothing. The script links `venv` to `/opt/mathud-venv`, because the CLI runs the app with `./venv/bin/python`, and adds each sandbox proxy CA in `/root/.ccr/agent-proxy-ca.crt` to Chrome's certificate store (`~/.pki/nssdb`) so headless Chrome trusts HTTPS through the sandbox proxy (the page itself loads its libraries from `static/vendor/`).
3. If `venv` is missing in a cloud session, check that `/opt/mathud-venv/bin/python` exists; the setup script may have failed.
4. If client tests fail with `net::ERR_CERT_AUTHORITY_INVALID` in the browser console, Chrome does not trust the proxy: `certutil -d sql:$HOME/.pki/nssdb -L` should list the proxy CAs with `C,,` trust, under any name (the platform may add them as `ccr-agent-proxy*` before the hook's `mathud-agent-proxy-N`). If none are listed, check that `certutil` is installed (`libnss3-tools`, from the setup script).

---

# MatHud Project Structure

## Repository Layout
1. `app.py`: Flask entry point and WSGI hookup.
2. `static/`: Server modules plus the Brython client bundle.
3. `templates/`: HTML shells that load Brython and bootstrap the UI.
4. `workspaces/`: Saved user state as JSON.
5. `server_tests/`: Backend pytest suites.
6. `documentation/`: Manuals such as `Reference Manual.txt` and `Example Prompts.txt`.
7. `logs/`: Application log output (rotated by `log_manager.py`, which keeps the newest 50 session logs).

## Backend Highlights (`static/`)
1. `app_manager.py`, `routes.py`, and `route_helpers.py` wire Flask endpoints to the provider layer: `openai_api_base.py` (shared history and system prompt), `openai_completions_api.py` / `openai_responses_api.py`, and `providers/` (Anthropic, OpenRouter, local).
2. `tool_call_processor.py`, `ai_model.py`, and `functions_definitions.py` define the function-call surface exposed to every provider.
3. Vision snapshots are captured in the browser (`static/client/canvas_snapshot.py`) and travel in the prompt JSON; `routes.py` only enforces their size limit.
4. `workspace_manager.py` and `log_manager.py` handle persistence and auditing.
5. `config.py` centralizes server-side constants (workspace dirs, schema version, request and image size caps).
6. `env_config.py` provides shared environment variable loading, replacing duplicated `load_dotenv` patterns.
7. `route_helpers.py` contains extracted route helper functions for provider management and tool lifecycle.
8. `canvas_state_formatter.py` renders the canvas for the model: a compact text block in each user message, `[canvas changes]` after tool batches, and `get_current_canvas_state` results (see `documentation/development/canvas_prompt_summary_rollout.md`).
9. `style.css` and other assets shared with the frontend live here for Flask to serve.
10. `response_metrics.py` measures every model request (time to first token, latency, tokens/s, prompt/completion/cached tokens, llama-server `timings`); providers attach the record to the final stream event as `metrics` and `log_manager.py` logs it as a `response_metrics {...}` JSON line.

## Client Highlights (`static/client/`)
1. `main.py` bootstraps Brython and registers managers.
2. `canvas.py`, `canvas_event_handler.py`, and `managers/` orchestrate SVG drawing, selection, undo, and edit policies.
3. `drawables/` contains shape classes (point, segment, vector, triangle, rectangle, circle, ellipse, angle, etc.).
4. `ai_interface.py` orchestrates AI communication and request building, delegating to five extracted managers.
5. `chat_ui_manager.py` handles message rendering, streaming display, and markdown parsing.
6. `message_menu_manager.py` manages context menus, clipboard, and raw text storage.
7. `tool_call_log_manager.py` provides tool call visualization and state tracking.
8. `image_attachment_manager.py` manages the file picker, preview, and image modal.
9. `tts_ui_manager.py` controls text-to-speech UI (read aloud, voice settings).
10. `process_function_calls.py` and `result_processor.py` coordinate tool execution and result handling.
11. `expression_evaluator.py`, `expression_validator.py`, and `result_validator.py` provide math parsing, validation, and error messaging.
12. `managers/base_drawable_manager.py` provides a shared base for all drawable managers.
13. `managers/visibility_manager.py` handles viewport culling extracted from Canvas.
14. `rendering/base_telemetry.py` provides a shared telemetry base for renderers.
15. `client_tests/` plus `test_runner.py` implement the Brython test harness (register new tests in `client_tests/tests.py`).
16. `turn_metrics.py` aggregates per-request metrics into per-turn summaries (chat footer under the final answer; `window.getMatHudLastTurnMetrics()` / `window.getMatHudTurnMetricsHistory()` return them as JSON strings for benchmarks).

---

# Cross-Cutting Guidelines

1. Keep backend-only Python in `static/` and browser-only Brython modules in `static/client/`.
2. Favor small, single-purpose classes (managers encapsulate behavior, drawables encapsulate geometry).
3. Favor small, easily readable public methods which use clearly named private methods to keep code structured in logical units.
4. Register new Brython tests and use the browser-based runner for anything importing `from browser import ...`.
5. Follow snake_case for modules and functions, PascalCase for classes, and UPPER_SNAKE_CASE for constants.

## Static Typing Guidelines
- Type checking uses `mypy` with configuration in `mypy.ini`.
- New modules should enable postponed evaluation: `from __future__ import annotations` beneath the module docstring.
- Prefer precise return types and explicit `-> None` annotations.
- Replace dynamic dicts with `TypedDict`, `Protocol`, or dataclasses where practical.
- When Brython-only modules require stubs, place them in `static/client/typing/` and register via `mypy_path`.

## Code Quality
- Run linting at the end of every editing cycle before committing.
- Use mypy on modified server-side files: `python -m mypy <files>`
- Note: Brython files (`static/client/`) cannot be checked with mypy due to browser imports.

---

# Brython Environment Context

## What is Brython?
**Brython** (Python in the browser) - Python runs in the browser, NOT server-side Python. This is critical for understanding the development environment and testing approach.

## Key Brython Concepts
1. `from browser import document, window, html, svg` pulls Brython DOM helpers.
2. Client-side Python lives in `static/client/` and executes in the browser.
3. Brython-only modules are unavailable to plain CPython tests.
4. Brython transpiles Python to JavaScript at runtime. It is served from `static/vendor/` (with math.js, nerdamer, MathJax and the Inter font), not from a CDN, so the app works offline; versions and SHA-256 hashes are pinned in `scripts/vendor_js_libs.py` (`--check` verifies the files, `server_tests/test_vendored_libs.py` enforces no external URLs in `templates/`).

---

# Rendering Architecture

## Renderer Selection
`static/client/rendering/factory.create_renderer` builds a preference chain (`canvas2d` → `svg`) and instantiates the first backend that succeeds. Each renderer module is imported only when it is attempted.

## Renderers
1. **Canvas2DRenderer** (`canvas2d_renderer.py`): Targets a Canvas 2D context and draws all shapes and labels. Sizes its bitmap by `devicePixelRatio` for sharp output on HiDPI screens while drawing in CSS pixels. Long polylines are traced by the JavaScript helpers in `static/canvas2d_paths.js` (loaded by `templates/index.html`), with a pure-Python fallback. Supports optional offscreen compositing via `_use_layer_compositing`.
2. **SvgRenderer** (`svg_renderer.py`): Fallback renderer, frozen (keep it working, no new features). Maintains plan caches for grid and drawables. Prunes unused DOM groups between frames. The `#math-svg` element stays in the DOM as the pointer-event surface for every renderer.

## Shared Components
- `shared_drawable_renderers.py`: Drawing helpers that coordinate metadata emission for labels.
- `style_manager.get_renderer_style()`: Returns cloned dictionaries so callers cannot mutate global defaults.
- `cached_render_plan.py`: Owns `OptimizedPrimitivePlan` type used by every renderer.
- Primitive adapters translate plan commands into backend-specific APIs.

## Feature Flags (Diagnostics)
- `window.MatHudCanvas2DOffscreen` or `localStorage["mathud.canvas2d.offscreen"]` - enables Canvas2D layer compositing.
- `window.MatHudSvgOffscreen` or `localStorage["mathud.svg.offscreen"]` - toggles SVG offscreen staging.

## DOM Layering
Surfaces are layered inside `#math-container`: Canvas2D (z-index 10) above the SVG base layer, which receives pointer events.

---

# Testing Environment

## Testing Split
1. **Pure Python tests** (regular `pytest`): `test_markdown_parser.py`, `test_expression_validator.py`, `test_math_functions.py`, and server-side Flask route suites.
2. **Brython-dependent tests** (browser runner): `test_canvas.py`, geometry suites such as `test_angle.py` or `test_point.py`, any `test_drawable_*.py`, and every module that imports `from browser import ...`.

> **MANDATORY — Fixture registration:** whenever you add a new client test file under `static/client/client_tests/`, you **must** perform both steps below or the tests will be silently skipped:
> 1. **Import** the `Test*` class at the top of `static/client/client_tests/tests.py` (alongside the other imports).
> 2. **Append** it to the list returned by `_get_test_cases()` in the same file.
>
> Without both steps the Brython runner will never execute the new tests.

## Running Tests
- **Server tests**: `python -m cli.main test server` (or `python run_server_tests.py`)
- **Client tests (CLI)**: `python -m cli.main test client --port PORT` (preferred for iteration)
- **Client tests (UI)**: Click **Run Tests** in the UI or ask the assistant to "run tests"
- **Single pytest file**: `python -m pytest path/to/test.py`
- **Provider keys**: server tests blank provider API keys (including ones from `.env`) unless `MATHUD_LIVE_TESTS=1` is set in the shell; live tests make paid API calls. The guard lives in `server_tests/conftest.py` and applies only under pytest: running a test file directly (`python file.py`) bypasses it.

## Running Client Tests via CLI (Recommended)
The CLI provides the fastest iteration loop for development:

```bash
# Start server once
python -m cli.main server start --port 5007

# Run client tests (automatically captures screenshot of results)
python -m cli.main test client --port 5007

# Disable screenshot for faster runs
python -m cli.main test client --port 5007 --no-screenshot
```

The CLI returns structured test results (pass/fail counts, failing test names, error messages) directly in the terminal output, enabling fast edit-test-fix cycles without browser interaction.

## Running Client Tests via Claude Code Browser Control
Alternatively, Claude Code can run Brython tests using the Chrome browser extension:

1. **Prerequisites**: Install the Claude Chrome extension and ensure Chrome is open with the extension active.
2. **Workflow**: Claude Code can start the Flask app, navigate to `http://127.0.0.1:5000/`, and run tests programmatically.
3. **Duration**: The full client test suite (~2,950 tests) takes approximately 1 minute to complete.
4. **Usage**: Ask Claude Code to "run the client tests" or "run the brython tests".

### Programmatic Test API
The app exposes JavaScript functions for programmatic test execution:

```javascript
// Start tests (returns immediately, tests run async)
window.startMatHudTests()  // Returns: {"status":"started"}

// Poll for results (call after ~1 minute)
window.getMatHudTestResults()
// Returns while running: {"status":"running"}
// Returns when complete: {"tests_run":2951,"failures":0,"errors":0,"failing_tests":[],"error_tests":[]}
```

This allows Claude Code to get test results as structured JSON without parsing screenshots.

## Running Client Tests Headlessly (CI / Sandbox / Unattended)

Client tests CANNOT be run with pytest — they depend on the Brython runtime which only
exists inside a browser. Running `pytest` on files under `static/client/` will always fail
with `ImportError: cannot import name 'aio' from 'browser'`.

To run client tests headlessly (e.g. inside a Docker container or CI):

```bash
# 1. Start the Flask server in the background
python app.py &
sleep 5  # wait for server to be ready

# 2. Run client tests via the CLI (uses Selenium + headless Chromium)
python -m cli.main test client --start-server --timeout 600 --json

# Or if server is already running on a specific port:
python -m cli.main test client --port 5000 --timeout 600 --json
```

The `--json` flag returns structured results. The `--start-server` flag auto-starts Flask. The CLI runs Python through `./venv`'s interpreter, or the interpreter running the CLI when `./venv` is missing (e.g. in a git worktree).

For a lightweight test environment without the text-to-speech stack (several GB), install `requirements-ci.txt` (the package list CI uses) instead of `requirements.txt`, either into `./venv` or into a separate venv whose interpreter runs the CLI. TTS then reports itself unavailable and its tests are skipped.

**Important**: When running "all tests", run server and client test suites separately:
```bash
# Server tests (pytest-based, pure Python)
python -m pytest server_tests/ -v

# Client tests (Selenium-based, runs in browser)
python -m cli.main test client --start-server --timeout 600 --json
```

Never pass `static/client/` paths to pytest — it will fail on every Brython import.

## Common Issues & Solutions
1. If a test raises `ModuleNotFoundError: No module named 'browser'`, move it to the Brython runner.
2. Remember everything under `static/client/` requires the browser runtime; CPython cannot import `browser`.
3. Use the Brython DOM helpers (`document`, `window`, `html`, `svg`) for any canvas interaction.

## When in doubt:
1. If a module imports `from browser import ...`, treat it as Brython-only.
2. If logic stays pure Python, exercise it with pytest.
3. If pytest reports "No module named 'browser'", rerun the test suite inside the browser harness.

---

# Adding New Features

This section covers two scenarios: adding a new drawable type (full checklist) and adding capabilities to an existing manager (simpler path).

## Commit Ordering for PRs

When implementing features, structure commits to isolate concerns:

1. **Core model/algorithm** - Standalone, testable implementation
2. **Manager integration** - CRUD operations with undo/redo archiving
3. **Rendering support** - Renderable class and renderer registration (if visual)
4. **Canvas/API wiring** - Canvas methods, FunctionRegistry, WorkspaceManager
5. **AI tool definitions** - JSON schema in `functions_definitions.py`
6. **Tests** - Validation, manager, and rendering tests
7. **Documentation** - Example Prompts, Reference Manual, update `documentation/Roadmap.md`

## Adding a New Drawable Type

Follow this checklist when introducing a new drawable type:

### 1. Define Core Constants
- Create styling defaults or limits in `static/client/constants.py`.
- Update `rendering/style_manager.py` if the renderer needs access to new constants.

### 2. Implement the Drawable Model
- Add the drawable class under `static/client/drawables/`.
- Inherit from `drawables.drawable.Drawable` and provide:
  - Comprehensive docstring describing the drawable's purpose.
  - Required attributes with validation.
  - `get_class_name`, serialization via `get_state`, and deep copy logic.
  - Optional helpers for transformations (`translate`, `rotate`).
- Keep the class free of rendering-specific code.

### 3. Extend Managers and Containers
- Update `DrawablesContainer` with a convenience property for the new class.
- Implement a dedicated manager in `static/client/managers/`.
- Register the manager within `DrawableManager` and expose delegating methods.
- Add undo/redo archiving in manager methods before mutating state.
- Update `workspace_manager.py` to handle serialization/deserialization.

### 4. Wire the Canvas API
- Register the drawable in `Canvas._register_renderer_handlers`.
- Add convenience methods on `Canvas` (create/get/delete) that delegate to the manager.
- Expose new methods in `FunctionRegistry`.

### 5. Update AI Function Definitions
- Extend `static/functions_definitions.py` with JSON schema entries for create/delete commands.
- If server-side tools need client constants, mirror values through `static/mirror_client_modules.py`.

### 6. Implement Rendering Support
- Create a helper in `rendering/shared_drawable_renderers.py`.
- Register the helper in `_HELPERS` inside `rendering/cached_render_plan.py`.
- Register the new drawable in each renderer's `register_default_drawables()`:
  - `canvas2d_renderer.py`
  - `svg_renderer.py`

### 7. Add Tests
- Introduce unit tests in `static/client/client_tests/` covering:
  - Drawable validation and state serialization.
  - Manager operations (create/delete with undo archiving).
  - Rendering helper behavior via `SimpleMock` primitives.

### 8. Update Documentation
- Describe new capabilities in relevant documents.

## Adding Capabilities to Existing Managers

For features that extend existing managers (e.g., `fit_regression` in `StatisticsManager`), the path is simpler:

1. **Core algorithm** - Implement in a new module or within the manager
2. **Manager method** - Add method with undo/redo archiving if it mutates state
3. **Canvas/API wiring** - Add delegation through `DrawableManager` → `Canvas` → `FunctionRegistry`
4. **AI tool definition** - JSON schema in `functions_definitions.py`
5. **Tests** - Pure Python tests for algorithm, Brython tests for Canvas integration
6. **Documentation** - Example Prompts, Reference Manual

**Note:** If the feature requires custom expression parsing (e.g., parametric `t` variable), update `expression_validator.py` with a new parsing method.

---

# Sample Prompts

## Basic Drawing
- `create point at (10, 20) named A`
- `draw a segment from (0,0) to (3,4) called s1`
- `construct a circle with center A and radius 50`
- `create a triangle with vertices (0,0), (4,0), (2,3.464)`
- `plot the function y = x^2 from x = -5 to 5`

## Statistics
- `plot a continuous normal distribution with mean 0 and sigma 1, shade from -1 to 1`
- `plot a bar chart with values [10, 20, 5] and labels ["A", "B", "C"]`
- `compute descriptive statistics for [10, 20, 30, 40, 50]`

## Parametric Curves
- `draw a parametric circle with x(t) = cos(t) and y(t) = sin(t)`
- `plot a spiral using x(t) = t*cos(t) and y(t) = t*sin(t) with t from 0 to 6*pi`
- `create a Lissajous curve with x(t) = sin(3*t) and y(t) = sin(2*t)`

## Graph Theory
- `create an undirected weighted graph named G1 with vertices A,B,C,D and edges A-B (1), B-C (2), A-C (4), C-D (1)`
- `on graph G1, find the shortest path from A to D and highlight the edges`
- `create a DAG named D1 with vertices A,B,C,D and edges A->B, A->C, B->D, C->D; topologically sort it`

## Calculus & Algebra
- `evaluate expression 3*A.x + 10`
- `what is the derivative of x^3 + 2x - 1 with respect to x?`
- `calculate the integral of 2x from x=0 to x=3`
- `solve the equation x^2 - 5x + 6 = 0 for x`

## Linear Algebra
- `evaluate linear algebra expression with A=[[1,2],[3,4]] and B=[[5,6],[7,8]]: inv(A) * B`

## Canvas Control
- `zoom in on the canvas`
- `fit the view to show math bounds left -20, right 30, top 15, bottom -10`
- `undo the last action`
- `clear the entire canvas`

## Workspace
- `save the current workspace as 'MyProject'`
- `load workspace 'MyProject'`
- `list all saved workspaces`

---

# Git Commit Guidelines

- ALWAYS review/reread all diffs (`git diff`) before staging or committing changes
- Do NOT add `Co-Authored-By` lines to commit messages (this is an open source MIT project)
- Keep commit messages clean and focused on describing the changes
- Use imperative mood (e.g., "Add feature" not "Added feature")

---

# Worktree Configuration

- IDE configs (`.vscode/`, `.cursor/`) are in `.git/info/exclude` (local ignore, not in public `.gitignore`)
- `.worktreeinclude` lists gitignored files to auto-copy to new worktrees (`.env`, IDE configs)
- When creating worktrees, Claude Code reads `.worktreeinclude` and copies matching gitignored files automatically
