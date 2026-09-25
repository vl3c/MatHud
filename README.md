# MatHud - Mathematics Heads-Up Display

MatHud pairs an interactive drawing canvas with an AI assistant to help visualize, analyze, and solve real-world arithmetic, geometry, algebra, calculus and statistics problems in real-time.

![MatHud - Interactive Mathematics Visualization Tool](MatHud%20-%20Screenshot%202025-12-21.png)

## 1. AI-First Operating Model

1. Primary interaction is conversational: users express intent in chat and the AI executes tool workflows.
2. The HUD canvas is the visual output surface for AI actions, not the primary control surface.
3. Direct UI gestures are optional support tools (inspection, quick anchoring) and should not be required for core workflows.
4. Features should optimize for intent resolution, deterministic execution, and explainable AI responses tied to canvas state.

## 2. Key Capabilities

1. Draw and manipulate geometric objects (points, segments, vectors, polygons, circles, ellipses, angles) directly on the canvas.
2. Ask the assistant to solve algebra, calculus, trigonometry, statistics, and linear algebra problems with LaTeX-formatted explanations.
3. Plot functions, compare intersections, shade bounded regions, and translate/rotate objects to explore relationships visually.
4. Plot statistics visualizations (probability distributions and bar charts).
5. Fit regression models to data (linear, polynomial, exponential, logarithmic, power, logistic, sinusoidal) and visualize fitted curves with R² statistics.
6. Compute descriptive statistics (mean, median, mode, standard deviation, variance, min, max, quartiles, IQR) for any dataset.
7. Create and analyze graph theory graphs (graphs, trees, DAGs).
8. Save, list, load, and delete named workspaces so projects can be resumed or shared later.
9. Share the current canvas with the assistant using Vision mode to get feedback grounded in your drawing.
10. Attach images directly to chat messages for the AI to analyze alongside your prompts.
11. Use slash commands (`/help`, `/vision`, `/model`, `/image`, etc.) for quick local operations without waiting for an AI response.
12. Type math symbols with the Σ palette, backslash completion (`\alpha` → α) or Alt shortcuts (`Alt+P` → π).
13. Choose from multiple AI providers — a local `llama-server` (LocalAgent, the default), OpenAI, Anthropic (Claude), and OpenRouter — with the model dropdown automatically filtered by which API keys you have configured and which local server is reachable.
14. Trigger client-side tests from the UI or chat to verify canvas behavior without leaving the app.

## 3. Architecture Overview

1. **Frontend (Brython)** – `static/client/` hosts the Brython application (`main.py`) that wires a `Canvas`, `AIInterface`, `CanvasEventHandler`, and numerous managers. Canvas objects stay math-only; renderers translate them to screen primitives via shared plan builders.
2. **Backend (Flask)** – `app.py` boots a Flask app assembled by `static/app_manager.py`, registers routes (`static/routes.py`), and injects OpenAI, workspace, and logging services.
3. **AI integration** – `static/providers/` implements a multi-provider architecture supporting LocalAgent (`static/providers/local/`, a local `llama-server`), OpenAI, Anthropic (Claude), and OpenRouter. `static/ai_model.py` stores model configs with per-model vision and reasoning flags. The model dropdown is populated dynamically from `GET /api/available_models`, which filters by which API keys are present in the environment and whether the local server answers. Each user message carries the canvas as a compact text block rendered by `static/canvas_state_formatter.py` (see 5.1).
4. **Rendering** – `static/client/rendering/factory.py` prefers Canvas2D and falls back to SVG if Canvas2D fails. Canvas and SVG renderers include opt-in offscreen staging toggled by `window.MatHudCanvas2DOffscreen` / `window.MatHudSvgOffscreen` or matching `localStorage` flags.
5. **Vision pipeline** – When vision is on, the browser composites the visible canvas layers (`static/client/canvas_snapshot.py`) into a PNG and sends it as `canvas_snapshot` in the prompt JSON; the provider adds it as an image part next to the `<canvas>` text block. No server-side browser is involved.

## 4. Getting Started

### 4.1 Prerequisites

1. Python 3.11+.
2. At least one AI provider API key (see Configuration below).

### 4.2 Environment Setup

1. Clone the repository and create a virtual environment:
   ```sh
   python -m venv venv
   ```
2. Activate the environment:
   - macOS/Linux: `source venv/bin/activate`
   - Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Provide at least one AI provider API key by setting environment variables or creating `.env` in the project root:
   ```env
   OPENAI_API_KEY=sk-...          # OpenAI models (GPT-6 Sol/Astra/Luna, GPT-5.6 Sol)
   ANTHROPIC_API_KEY=sk-ant-...   # Anthropic models (Claude Fable 5.1, Opus 5.5, Sonnet 5, Haiku 4.5)
   OPENROUTER_API_KEY=sk-or-...   # OpenRouter models (Gemini, Grok, Qwen, DeepSeek, free models, etc.)
   ```
   Only models for configured providers will appear in the model dropdown. A local
   `llama-server` needs no key: start one and MatHud picks it up automatically (see
   [6.7 AI Provider Configuration](#67-ai-provider-configuration)).
5. (Optional, for contributors) Install the pre-commit hook, which runs ruff on staged Python files:
   ```sh
   python -m cli.main test lint --install-hook
   ```
   Re-run it after `hooks/pre-commit` changes. See [cli/README.md](cli/README.md#linting) for details.

### 4.3 Run MatHud

**Desktop window (one command).** The desktop launcher starts the server and opens MatHud in its own window; closing the window stops the server.

1. Install the optional window dependency once ([pywebview](https://pywebview.flowrl.com/); on Windows it uses the Edge WebView2 runtime that ships with Windows 11):
   ```sh
   pip install -r requirements-desktop.txt
   ```
2. Launch from the project root (either command):
   ```sh
   python mathud_desktop.py
   python -m cli.main desktop
   ```
   Options: `--port N` serves on a specific port (default 5100, or a free port if 5100 is taken), `--browser` opens your default browser instead of a window (no pywebview needed; stop with `Ctrl+C`), and `--devtools` enables the WebView developer tools. Without pywebview the launcher prints the install command and offers the browser instead.
3. The window's size and position, and the page's local storage, are kept in your user profile (`%LOCALAPPDATA%\MatHud` on Windows, `~/Library/Application Support/MatHud` on macOS, `~/.config/mathud` on Linux).
4. Two desktop instances can run at once: the second one gets a different port, so its page has its own local storage (settings are not shared), while both use the same window-geometry file and whichever closes last decides the next launch's size and position.

**Server plus browser.**

1. Launch the Flask server from the project root:
   ```sh
   python app.py
   ```
   Use `python app.py --port 5004` if port 5000 is taken.
2. Open `http://127.0.0.1:5000/` in a desktop browser (Chrome, Firefox, or Edge confirmed). The Brython client loads automatically.
3. Stop the server with `Ctrl+C`.

Text-to-speech (Kokoro) is not loaded at startup; its model loads on the first read-aloud request, which takes a few seconds once.

### 4.4 Offline / vendored libraries

The page loads no scripts, stylesheets or fonts from the internet. Brython 3.12.5, math.js 14.5.2, nerdamer 1.1.13, MathJax 3.2.2 and the Inter font are committed under `static/vendor/`, so MatHud runs fully offline with LocalAgent. Versions, source URLs and SHA-256 hashes are pinned in `scripts/vendor_js_libs.py`:

```sh
python scripts/vendor_js_libs.py --check   # verify the committed files (no network)
python scripts/vendor_js_libs.py           # re-download missing or changed files
```

Licenses are listed in `static/vendor/LICENSES.md`.

## 5. Configuration and Authentication

1. The server reads configuration from environment variables or `.env` (loaded via `python-dotenv`). It looks for `.env` in the project root, then in the directory above it; a git worktree under `<repo>/.claude/worktrees/<name>` also reads the `.env` above the main checkout. A nearer file wins, and variables that are already set are never overridden. Common options:
   ```env
   OPENAI_API_KEY=sk-...          # OpenAI provider
   ANTHROPIC_API_KEY=sk-ant-...   # Anthropic provider
   OPENROUTER_API_KEY=sk-or-...   # OpenRouter provider
   AUTH_PIN=123456                 # Optional: access code required when auth is enabled
   REQUIRE_AUTH=true               # Force authentication in local development
   PORT=5000                       # Set by hosting platforms to indicate deployed mode
   SECRET_KEY=override-me          # Optional: otherwise a random key is generated per launch
   TOOL_SEARCH_MODE=hybrid         # Tool discovery: local | api | hybrid (default: hybrid)
   MATHUD_TOOL_EXPOSURE=search     # search: model starts with search_tools + essentials (default); full: all tools up front
   MATHUD_CANVAS_FORMAT=text       # How the canvas reaches the model: text (default) | min_json | json (original prompt JSON)
   MATHUD_CANVAS_BUDGET_TOKENS=    # Canvas token budget; default 4000 (cloud) / 1500 (local), 0 = unlimited
   LOCAL_AGENT_BASE_URL=http://127.0.0.1:8080  # LocalAgent server (default shown)
   ```
2. Authentication rules (`static/app_manager.py`):
   1. When `PORT` is set (typical in hosted deployments), authentication is enforced automatically.
   2. Locally, you can opt-in by setting `REQUIRE_AUTH=true`. The login page accepts the `AUTH_PIN` value.
   3. Sessions use `flask-session` with a CacheLib-backed store; cookies are upgraded to secure/HTTP-only in deployed mode.
3. Vision snapshots are captured in the browser; no Firefox or WebDriver is needed on the server.

### 5.1 Canvas Prompt Controls

Every user message carries the current canvas. `MATHUD_CANVAS_FORMAT` chooses how the model sees it (all providers, LocalAgent included):

1. `text` (default): a `<canvas>` block in front of the user's text, one object per line in math notation, with lengths, areas, angle sizes and similar facts computed from the coordinates (`AB = Segment(A, B)  len 5`). Numbers keep at most 6 significant digits.
2. `min_json`: the same block holding the state as compact JSON (render-only fields, defaults and float noise removed).
3. `json`: the original canvas payload, sending the whole prompt JSON; the `AI_CANVAS_SUMMARY_MODE` options below apply only here. Not a byte-for-byte replay of older requests: the hybrid `metrics` block is no longer in the prompt, search tool mode adds its tool-loading paragraph to the system prompt, and each tool call gets its own result message.

With `text` and `min_json`, the last tool result of each tool batch ends with `[canvas changes]` (what the batch added, changed or removed), and `get_current_canvas_state` results use the same format. `MATHUD_CANVAS_BUDGET_TOKENS` caps the canvas block (default 4000 estimated tokens for cloud models, 1500 for local ones, `0` for no limit): larger scenes pack points several per line, then list the least important objects as omitted with a pointer to `get_current_canvas_state` (`min_json` keeps the same fraction of every object list). `get_current_canvas_state` results get twice that budget.

```env
MATHUD_CANVAS_FORMAT=text              # text | min_json | json
MATHUD_CANVAS_BUDGET_TOKENS=4000       # 0 = unlimited; unset = provider default
AI_CANVAS_SUMMARY_MODE=hybrid          # json format only: off | hybrid | summary_only
AI_CANVAS_HYBRID_FULL_MAX_BYTES=6000   # json format only: hybrid threshold for sending full canvas_state
AI_CANVAS_SUMMARY_TELEMETRY=0          # 1/true/on to emit canvas_prompt_telemetry logs
```

Summary modes (json format only):
1. `off`: send original payload unchanged.
2. `hybrid` (default): keep full `canvas_state` for small scenes, attach `canvas_state_summary` and remove full state for large scenes.
3. `summary_only`: always remove full `canvas_state` and send summary envelope.

Developer utilities:
1. Browser console helper: `window.compareCanvasState()` (development mode) prints full vs summary structures with byte/token metrics.
2. Log report script: `python scripts/canvas_prompt_telemetry_report.py --mode hybrid --json-out /tmp/canvas_summary_report.json`
3. Deep-dive rollout notes: `documentation/development/canvas_prompt_summary_rollout.md`

## 6. Working with MatHud

### 6.1 Canvas Interaction

1. Use chat as the default control channel: describe what you want and let the AI perform the steps.
2. Gesture support remains available for quick inspection:
   - Double-click the canvas to log precise math coordinates into the chat box.
   - Pan by click-dragging; zoom with the mouse wheel (anchored around the cursor).
3. The canvas tracks undo/redo, dependencies, and name generation automatically through managers in `static/client/managers/`.

### 6.2 Conversing with the Assistant

1. Type a request in the chat input and press Enter or click **Send**. The assistant inspects the current canvas state and can call functions on your behalf.
2. Responses support Markdown and LaTeX; MathJax renders inline (`\( ... \)`) and block (`$$ ... $$`) math.
3. Sample prompts that map directly to available tools:
   Note: In this section, "plot" refers to function plots. "graph" refers to graph theory vertices/edges (not dependency graphs).
   1. `create point at (2, 3) named A`
   2. `draw a segment from (0,0) to (3,4) called s1`
   3. `plot y = sin(x) from -pi to pi`
   4. `evaluate expression 2*sin(pi/4)`
   5. `derive x^3 + 2x - 1`
   6. `solve system of equations: x + y = 5, x - y = 1`
   7. `evaluate linear algebra expression with matrices A=[[1,2],[3,4]]; compute inv(A)`
   8. `plot a normal distribution with mean 0 and sigma 1, continuous, shade from -1 to 1`
   9. `plot a bar chart with values [10,20,5] and labels ["A","B","C"]`
   10. `fit a linear regression to x_data=[1,2,3,4,5] and y_data=[2,4,6,8,10], show points and report R²`
   11. `compute descriptive statistics for [10, 20, 30, 40, 50]`
   12. `create an undirected weighted graph named G1 with vertices A,B,C,D and edges A-B (1), B-C (2), A-C (4), C-D (1)`
   13. `on graph G1, find the shortest path from A to D and highlight the edges`
   14. `create a DAG named D1 with vertices A,B,C,D and edges A->B, A->C, B->D, C->D; then topologically sort it`
   15. `save workspace as "demo"` / `load workspace "demo"`
   16. `run tests`
4. Each finished answer ends with a muted metrics footer, e.g. `qwen3.8-27b · 4.2 s · first token 0.8 s · 38 tok/s · 2 requests · 3 tool calls`; hover it for token counts (prompt, cached, completion), the per-request breakdown and tool errors. Speeds come from llama-server `timings` or the provider's usage report, and are prefixed with `~` when estimated from the streamed text. Every model request is also logged as a `response_metrics {...}` JSON line, and `window.getMatHudLastTurnMetrics()` / `window.getMatHudTurnMetricsHistory()` return the per-turn summaries as JSON strings for benchmarking.

### 6.3 Slash Commands

Type `/` in the chat input to access local commands that execute instantly without contacting the AI:

| Command | Description |
|---------|-------------|
| `/help [command]` | Show available commands or detailed help for a specific command |
| `/undo` / `/redo` | Undo or redo the last canvas action |
| `/clear` / `/reset` | Clear all objects or reset view to default |
| `/save [name]` / `/load [name]` | Save or load a named workspace |
| `/workspaces` | List all saved workspaces |
| `/fit` | Fit the view to show all objects |
| `/zoom <in\|out\|factor>` | Zoom the canvas |
| `/grid` / `/axes` | Toggle grid or axes visibility |
| `/polar` / `/cartesian` | Switch coordinate system |
| `/status` | Show canvas info (object count, bounds) |
| `/vision` | Toggle vision mode (vision-capable models only) |
| `/image` | Attach an image to your next message (vision-capable models only) |
| `/model [name]` | Show or switch the current AI model |
| `/test` | Run the client test suite |
| `/export` / `/import <json>` | Export or import canvas state as JSON |
| `/list` | List all objects on the canvas |
| `/new` | Start fresh (clear canvas + new conversation) |

Autocomplete suggestions appear as you type. Unknown commands trigger fuzzy-match suggestions.

### 6.4 Math Symbols

Type math symbols straight into the chat input (they are sent as plain Unicode text):

1. **Palette**: click the **Σ** button next to the chat input (or press `Ctrl+↑` in the input; on macOS `Ctrl+↑` usually opens Mission Control, so use the Σ button or `\name` there) to open the symbol palette with Operators, Calculus, Sets & logic, Geometry and Greek groups (it opens on Operators the first time, then on the group you used last), plus a **Recent** row of the last 10 symbols you used. Clicking a symbol inserts it at the caret (replacing any selection) and keeps focus in the input. With the palette open, arrow keys move the highlight, `Enter` inserts, `Tab` / `Shift+Tab` switch group and `Esc` or a click elsewhere closes it; `Enter` sends the message as usual if you have not moved the highlight. Hover a symbol for its name, Alt shortcut and `\` names.
2. **Backslash completion**: type `\` followed by a name, e.g. `\alpha`, `\le`, `\pi`, `\int`, `\in`, `\R`; a suggestion list shows matching symbols (prefix matches first). `Tab` accepts the highlighted suggestion; `↑`/`↓` pick another, after which `Enter` accepts it too. Otherwise `Enter` closes the list and sends the message as typed, so a message ending in `\n` is not turned into `≠`. `Esc` dismisses, and typing a space after an exact name converts it too (`\theta ` becomes `θ `). With the caret inside a name (`\al|pha`), completion replaces the whole name. Plain words such as `pi` are never converted.
3. **Alt shortcuts** (Windows and Linux, left Alt). Letters follow the key labels of your layout (on AZERTY, `Alt+A` is the key labelled A); on layouts that type digits with Shift, such as AZERTY, add Shift for the superscripts. `Ctrl+Alt`/AltGr and right-Alt combinations are left alone. On macOS, Option keeps typing its own characters, which already include π (`Option+P`), ≤ ≥ ≠ (`Option+,` `.` `=`), ∞ (`Option+5`) and √ (`Option+V`) on the US layout.

| Keys | Symbol | Keys | Symbol |
|------|--------|------|--------|
| `Alt+A` | α | `Alt+P` / `Alt+Shift+P` | π / Π |
| `Alt+B` | β | `Alt+R` | √ |
| `Alt+D` / `Alt+Shift+D` | δ / Δ | `Alt+S` / `Alt+Shift+S` | σ / Σ |
| `Alt+F` / `Alt+Shift+F` | φ / Φ | `Alt+T` / `Alt+Shift+T` | θ / Θ |
| `Alt+G` / `Alt+Shift+G` | γ / Γ | `Alt+U` | ∞ |
| `Alt+L` | λ | `Alt+W` / `Alt+Shift+W` | ω / Ω |
| `Alt+M` | μ | `Alt+0` … `Alt+9` | ⁰ … ⁹ |
| `Alt+O` | ° | `Alt+-` | ⁻ |
| `Alt+,` / `Alt+.` | ≤ / ≥ | `Alt+=` | ≠ |

The symbol table lives in `static/client/math_symbols.py`.

### 6.5 Image Attachment

1. Click the paperclip button next to the chat input (or use `/image`) to attach images to your message.
2. Multiple images can be attached per message (up to the configured limit).
3. Image previews appear below the chat input; click the X on a preview to remove it.
4. Images are sent alongside your text message for the AI to analyze.
5. The attach button and `/image` command are only available when the selected model supports vision. Non-vision models show "(text only)" in the dropdown.

### 6.6 Vision Mode

1. Use the **Enable Vision** checkbox in the chat header to include screenshots of the current canvas.
2. The vision toggle and attach button are hidden for models without vision support. Models marked "(text only)" in the dropdown do not support image input.
3. The snapshot is taken in the browser when the message is sent: the Canvas2D and SVG layers composited at CSS-pixel size (longest side capped at 1280 px) on a white background. It is not saved on the server.

### 6.7 AI Provider Configuration

MatHud supports four AI providers. The model dropdown dynamically shows only models for providers that are configured (an API key) or reachable (a running local server):

| Provider | Environment Variable | Models |
|----------|---------------------|--------|
| **LocalAgent** | `LOCAL_AGENT_BASE_URL` (optional) | Whichever model the local `llama-server` currently hosts |
| **OpenAI** | `OPENAI_API_KEY` | GPT-6 Sol (the default when no local model is running), GPT-6 Astra, GPT-6 Luna, GPT-5.6 Sol |
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude Fable 5.1, Claude Opus 5.5, Claude Sonnet 5, Claude Haiku 4.5 |
| **OpenRouter** | `OPENROUTER_API_KEY` | Paid: Claude Opus 5.5, Claude Sonnet 5, Gemini 3.8 Flash, Grok 4.7, MiMo V2.6 Pro, GLM 5.3 Flash, DeepSeek V4.1 Flash, Qwen3.8 Max. Free: Qwen3.8 27B, Nemotron 3 Ultra (text only), Gemma 4 31B, Gemma 4 26B A4B, Inkling |

Models without vision support are labeled "(text only)" in the dropdown. When nothing is configured or reachable, the dropdown shows "No API keys configured".

Every OpenAI model is a reasoning model served through the Responses API with an explicit reasoning effort (GPT-6 Luna low, the others medium). Claude Fable 5.1, Opus 5.5 and Sonnet 5 get their effort (Fable low, the others medium) as `output_config.effort`. OpenRouter rate-limits its free models to 20 requests per minute, and to 50 requests per day on accounts with less than $10 of lifetime credits (1000 per day otherwise); a MatHud turn with tool calls makes several requests.

#### LocalAgent (llama.cpp)

LocalAgent is the default provider. It talks to a llama.cpp `llama-server` (or any other
backend serving the same OpenAI-compatible API) over `/v1`, and needs no API key:

1. Start `llama-server` on `http://127.0.0.1:8080`, or point `LOCAL_AGENT_BASE_URL` elsewhere.
2. Reload MatHud. The provider reads `/v1/models` and offers whichever model the server
   reports, whether that is an `--alias` value or a `.gguf` file path.
3. The server hosts one model at a time, so restarting it with a different model is enough
   to switch: MatHud drops the stale entry and picks up the new one on the next page load.

Local models are used in search-first tool mode and currently receive text only; attached
images are not forwarded.

### 6.8 Workspace Management

1. Workspaces are persisted as JSON under `workspaces/`.
2. The chat tools `save_workspace`, `load_workspace`, `list_workspaces`, and `delete_workspace` are exposed to the assistant and UI.
3. Client-side restores rebuild the Brython objects through `static/client/workspace_manager.py`.
4. Saves are atomic (written to a temporary file, then swapped in); overwriting a workspace keeps its previous version as `<name>.json.bak`.
5. Deleting a workspace moves it to `workspaces/.trash/` under a timestamped name instead of removing it. The `/delete_workspace` route only accepts a POST with a JSON body (`{"name": ...}`).

### 6.9 Testing

1. Server tests: run `python run_server_tests.py` (add `--with-auth` to exercise authenticated flows). Provider API keys, including those in `.env`, are blanked for the run unless `MATHUD_LIVE_TESTS=1` is set in the shell; live tests make paid API calls. The guard lives in `server_tests/conftest.py` and applies only under pytest, so running a test file directly with `python file.py` bypasses it.
2. Client tests: click **Run Tests** in the UI or ask the assistant to "run tests". Results stream back into the chat after execution (`static/client/test_runner.py`).
3. Linting: run `python -m cli.main test lint` (ruff + mypy). The optional pre-commit hook runs ruff on staged files (see 4.2).
4. CI: every pull request and every push to `main` runs lint, the server tests and the client tests (headless Chrome) via `.github/workflows/tests.yml`, with the packages in `requirements-ci.txt` (no text-to-speech stack). Client-test failures are reported as a warning, not a failed check, for now.

## 7. Rendering Notes

1. `static/client/rendering/factory.py` instantiates renderers in preference order `canvas2d → svg`, importing each renderer module only when it is attempted. If a constructor raises, the factory continues down the chain.
2. Canvas2D rendering (`canvas2d_renderer.py`) draws every shape and label. Its bitmap is sized by `devicePixelRatio`, so output stays sharp on HiDPI screens, and long polylines are traced by the JavaScript helpers in `static/canvas2d_paths.js` (with a pure-Python fallback). It supports optional offscreen compositing: toggle it with `window.MatHudCanvas2DOffscreen = true` or `localStorage["mathud.canvas2d.offscreen"] = "1"`.
3. SVG rendering (`svg_renderer.py`) is the frozen fallback (kept working, no new features). It mirrors the same offscreen staging controls through `window.MatHudSvgOffscreen` or `localStorage["mathud.svg.offscreen"]`.

## 8. Diagram Generation

1. Generate the full suite of diagrams from the project root:
   ```sh
   python generate_diagrams_launcher.py
   ```
2. Output directories:
   1. `diagrams/generated_png/` – raster versions for quick sharing.
   2. `diagrams/generated_svg/` – scalable diagrams for documentation.
3. Additional guidance lives in `diagrams/README.md` and `diagrams/WORKFLOW_SUMMARY.md`.

## 9. Repository Guide

1. `app.py` – entry point with graceful shutdown and threaded dev server.
2. `static/`
   a. `app_manager.py`, `routes.py`, `openai_api_base.py` / `openai_completions_api.py` / `openai_responses_api.py`, `ai_model.py`, `tool_call_processor.py`, `tool_search_service.py`, `canvas_state_formatter.py`, `workspace_manager.py`, `log_manager.py`.
   b. `providers/` – Multi-provider AI backend (LocalAgent, Anthropic, OpenRouter; OpenAI lives in the `openai_*_api.py` modules) with `ProviderRegistry` for provider detection.
   c. `client/` – Brython modules (canvas, managers, rendering, slash commands, tests, utilities, workspace manager).
3. `templates/index.html` – main HTML shell that loads Brython, MathJax, styles, and UI controls.
4. `workspaces/` – saved canvas states.
5. `server_tests/` – pytest suites, including renderer plan tests under `server_tests/client_renderer/`.
6. `documentation/` – extended reference material.
7. `logs/` – session-specific server logs (the newest 50 are kept).

## 10. Additional Documentation

1. `documentation/Project Architecture.txt` – deep dive into system design.
2. `documentation/Reference Manual.txt` – comprehensive API and module reference.
3. `documentation/Example Prompts.txt` – curated prompts for common workflows.
