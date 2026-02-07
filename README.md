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
6. Create and analyze graph theory graphs (graphs, trees, DAGs).
7. Save, list, load, and delete named workspaces so projects can be resumed or shared later.
8. Share the current canvas with the assistant using Vision mode to get feedback grounded in your drawing.
9. Attach images directly to chat messages for the AI to analyze alongside your prompts.
10. Use slash commands (`/help`, `/vision`, `/model`, `/image`, etc.) for quick local operations without waiting for an AI response.
11. Choose from multiple AI providers — OpenAI, Anthropic (Claude), and OpenRouter — with the model dropdown automatically filtered by which API keys you have configured.
12. Trigger client-side tests from the UI or chat to verify canvas behavior without leaving the app.

## 3. Architecture Overview

1. **Frontend (Brython)** – `static/client/` hosts the Brython application (`main.py`) that wires a `Canvas`, `AIInterface`, `CanvasEventHandler`, and numerous managers. Canvas objects stay math-only; renderers translate them to screen primitives via shared plan builders.
2. **Backend (Flask)** – `app.py` boots a Flask app assembled by `static/app_manager.py`, registers routes (`static/routes.py`), and injects OpenAI, workspace, webdriver, and logging services.
3. **AI integration** – `static/providers/` implements a multi-provider architecture supporting OpenAI, Anthropic (Claude), and OpenRouter. `static/ai_model.py` stores model configs with per-model vision and reasoning flags. The model dropdown is populated dynamically from `GET /api/available_models`, which filters by which API keys are present in the environment.
4. **Rendering** – `static/client/rendering/factory.py` prefers Canvas2D, then SVG, and finally the still-incomplete WebGL path if earlier options fail. Canvas and SVG renderers include opt-in offscreen staging toggled by `window.MatHudCanvas2DOffscreen` / `window.MatHudSvgOffscreen` or matching `localStorage` flags.
5. **Vision pipeline** – When the chat payload signals vision, the server either stores a data URL snapshot or drives Selenium (`static/webdriver_manager.py`) to replay SVG state in headless Firefox and capture `canvas_snapshots/canvas.png` for the model.

## 4. Getting Started

### 4.1 Prerequisites

1. Python 3.10+ (tested with Python 3.11).
2. Firefox installed locally for the vision workflow (the `geckodriver-autoinstaller` package handles the driver).
3. At least one AI provider API key (see Configuration below).

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
   OPENAI_API_KEY=sk-...          # OpenAI models (GPT-4o, GPT-5, o3, etc.)
   ANTHROPIC_API_KEY=sk-ant-...   # Anthropic models (Claude Opus/Sonnet/Haiku 4.5)
   OPENROUTER_API_KEY=sk-or-...   # OpenRouter models (Gemini, DeepSeek, Llama, etc.)
   ```
   Only models for configured providers will appear in the model dropdown.

### 4.3 Run MatHud

1. Launch the Flask server from the project root:
   ```sh
   python app.py
   ```
2. Open `http://127.0.0.1:5000/` in a desktop browser (Chrome, Firefox, or Edge confirmed). The Brython client loads automatically.
3. Stop the server with `Ctrl+C`. The shutdown handler closes any active Selenium session before exiting.

## 5. Configuration and Authentication

1. The server reads configuration from environment variables or `.env` (loaded via `python-dotenv`). Common options:
   ```env
   OPENAI_API_KEY=sk-...          # OpenAI provider
   ANTHROPIC_API_KEY=sk-ant-...   # Anthropic provider
   OPENROUTER_API_KEY=sk-or-...   # OpenRouter provider
   AUTH_PIN=123456                 # Optional: access code required when auth is enabled
   REQUIRE_AUTH=true               # Force authentication in local development
   PORT=5000                       # Set by hosting platforms to indicate deployed mode
   SECRET_KEY=override-me          # Optional: otherwise a random key is generated per launch
   ```
2. Authentication rules (`static/app_manager.py`):
   1. When `PORT` is set (typical in hosted deployments), authentication is enforced automatically.
   2. Locally, you can opt-in by setting `REQUIRE_AUTH=true`. The login page accepts the `AUTH_PIN` value.
   3. Sessions use `flask-session` with a CacheLib-backed store; cookies are upgraded to secure/HTTP-only in deployed mode.
3. Vision capture requires Firefox. The first request that needs Selenium will call `/init_webdriver`, which in turn relies on `geckodriver-autoinstaller` to download the driver if necessary.

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
   11. `create an undirected weighted graph named G1 with vertices A,B,C,D and edges A-B (1), B-C (2), A-C (4), C-D (1)`
   12. `on graph G1, find the shortest path from A to D and highlight the edges`
   13. `create a DAG named D1 with vertices A,B,C,D and edges A->B, A->C, B->D, C->D; then topologically sort it`
   14. `save workspace as "demo"` / `load workspace "demo"`
   15. `run tests`

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

### 6.4 Image Attachment

1. Click the paperclip button next to the chat input (or use `/image`) to attach images to your message.
2. Multiple images can be attached per message (up to the configured limit).
3. Image previews appear below the chat input; click the X on a preview to remove it.
4. Images are sent alongside your text message for the AI to analyze.
5. The attach button and `/image` command are only available when the selected model supports vision. Non-vision models show "(text only)" in the dropdown.

### 6.5 Vision Mode

1. Use the **Enable Vision** checkbox in the chat header to include screenshots of the current canvas.
2. The vision toggle and attach button are hidden for models without vision support. Models marked "(text only)" in the dropdown do not support image input.
3. The server stores the latest snapshot under `canvas_snapshots/canvas.png` for troubleshooting.

### 6.6 AI Provider Configuration

MatHud supports three AI providers. The model dropdown dynamically shows only models for providers with configured API keys:

| Provider | Environment Variable | Models |
|----------|---------------------|--------|
| **OpenAI** | `OPENAI_API_KEY` | GPT-5.2, GPT-5, GPT-4.1, GPT-4o, o3, o4-mini, GPT-3.5 Turbo, etc. |
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude Opus 4.5, Claude Sonnet 4.5, Claude Haiku 4.5 |
| **OpenRouter** | `OPENROUTER_API_KEY` | Gemini 3 Pro/Flash, Gemini 2.5 Pro, DeepSeek V3.2, Grok, Llama, Gemma, and more (paid and free tiers) |

Models without vision support are labeled "(text only)" in the dropdown. When no API keys are configured, the dropdown shows "No API keys configured".

### 6.7 Workspace Management

1. Workspaces are persisted as JSON under `workspaces/`.
2. The chat tools `save_workspace`, `load_workspace`, `list_workspaces`, and `delete_workspace` are exposed to the assistant and UI.
3. Client-side restores rebuild the Brython objects through `static/client/workspace_manager.py`.

### 6.8 Testing

1. Server tests: run `python run_server_tests.py` (add `--with-auth` to exercise authenticated flows).
2. Client tests: click **Run Tests** in the UI or ask the assistant to "run tests". Results stream back into the chat after execution (`static/client/test_runner.py`).

## 7. Rendering Notes

1. `static/client/rendering/factory.py` instantiates renderers in preference order `canvas2d → svg → webgl`. If a constructor raises (for example, WebGL unavailable), the factory continues down the chain.
2. Canvas2D rendering (`canvas2d_renderer.py`) supports optional offscreen compositing. Toggle it with `window.MatHudCanvas2DOffscreen = true` or `localStorage["mathud.canvas2d.offscreen"] = "1"`.
3. SVG rendering (`svg_renderer.py`) mirrors the same offscreen staging controls through `window.MatHudSvgOffscreen` or `localStorage["mathud.svg.offscreen"]`.
4. The WebGL renderer (`webgl_renderer.py`) is experimental, not feature complete, and only instantiates when the browser exposes a WebGL context.

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
   a. `app_manager.py`, `routes.py`, `openai_api.py`, `ai_model.py`, `tool_call_processor.py`, `workspace_manager.py`, `log_manager.py`, `webdriver_manager.py`.
   b. `providers/` – Multi-provider AI backend (OpenAI, Anthropic, OpenRouter) with `ProviderRegistry` for API key detection.
   c. `client/` – Brython modules (canvas, managers, rendering, slash commands, tests, utilities, workspace manager).
3. `templates/index.html` – main HTML shell that loads Brython, MathJax, styles, and UI controls.
4. `workspaces/` – saved canvas states.
5. `canvas_snapshots/` – latest Selenium captures used for vision.
6. `server_tests/` – pytest suites, including renderer plan tests under `server_tests/client_renderer/`.
7. `documentation/` – extended reference material.
8. `logs/` – session-specific server logs.

## 10. Additional Documentation

1. `documentation/Project Architecture.txt` – deep dive into system design.
2. `documentation/Reference Manual.txt` – comprehensive API and module reference.
3. `documentation/Example Prompts.txt` – curated prompts for common workflows.
