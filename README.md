# MatHud - Mathematics Heads-Up Display

MatHud is an interactive mathematical visualization tool that combines a drawing canvas with an AI assistant to help understand and solve real-world mathematical problems. It serves as a heads-up display system for mathematical analysis, allowing users to visualize, analyze, and solve problems in real-time.

![MatHud - Interactive Mathematics Visualization Tool](MatHud%20-%20Screenshot%202025-06-28.png)

## 1. Key Capabilities

1. Draw and manipulate geometric objects (points, segments, vectors, polygons, circles, ellipses, angles) directly on the canvas.
2. Ask the assistant to solve algebra, calculus, trigonometry, statistics, and linear algebra problems with LaTeX-formatted explanations.
3. Plot functions, compare intersections, shade bounded regions, and translate/rotate objects to explore relationships visually.
4. Save, list, load, and delete named workspaces so projects can be resumed or shared later.
5. Share the current canvas with the assistant using Vision mode to get feedback grounded in your drawing.
6. Trigger client-side tests from the UI or chat to verify canvas behavior without leaving the app.

## 2. Architecture Overview

1. **Frontend (Brython)** – `static/client/` hosts the Brython application (`main.py`) that wires a `Canvas`, `AIInterface`, `CanvasEventHandler`, and numerous managers. Canvas objects stay math-only; renderers translate them to screen primitives via shared plan builders.
2. **Backend (Flask)** – `app.py` boots a Flask app assembled by `static/app_manager.py`, registers routes (`static/routes.py`), and injects OpenAI, workspace, webdriver, and logging services.
3. **AI integration** – `static/openai_api.py` wraps the OpenAI SDK, loads tool schemas from `static/functions_definitions.py`, and maintains model state (`static/ai_model.py`).
4. **Rendering** – `static/client/rendering/factory.py` prefers Canvas2D, then SVG, and finally the still-incomplete WebGL path if earlier options fail. Canvas and SVG renderers include opt-in offscreen staging toggled by `window.MatHudCanvas2DOffscreen` / `window.MatHudSvgOffscreen` or matching `localStorage` flags.
5. **Vision pipeline** – When the chat payload signals vision, the server either stores a data URL snapshot or drives Selenium (`static/webdriver_manager.py`) to replay SVG state in headless Firefox and capture `canvas_snapshots/canvas.png` for the model.

## 3. Getting Started

### 3.1 Prerequisites

1. Python 3.10+ (tested with Python 3.11).
2. Firefox installed locally for the vision workflow (the `geckodriver-autoinstaller` package handles the driver).
3. An OpenAI API key with access to the desired models.

### 3.2 Environment Setup

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
4. Provide credentials by setting `OPENAI_API_KEY` in your shell or by creating `.env` in the project root:
   ```env
   OPENAI_API_KEY=sk-...
   ```

### 3.3 Run MatHud

1. Launch the Flask server from the project root:
   ```sh
   python app.py
   ```
2. Open `http://127.0.0.1:5000/` in a desktop browser (Chrome, Firefox, or Edge confirmed). The Brython client loads automatically.
3. Stop the server with `Ctrl+C`. The shutdown handler closes any active Selenium session before exiting.

## 4. Configuration and Authentication

1. The server reads configuration from environment variables or `.env` (loaded via `python-dotenv`). Common options:
   ```env
   OPENAI_API_KEY=sk-...
   AUTH_PIN=123456            # Optional: access code required when auth is enabled
   REQUIRE_AUTH=true          # Force authentication in local development
   PORT=5000                  # Set by hosting platforms to indicate deployed mode
   SECRET_KEY=override-me     # Optional: otherwise a random key is generated per launch
   ```
2. Authentication rules (`static/app_manager.py`):
   1. When `PORT` is set (typical in hosted deployments), authentication is enforced automatically.
   2. Locally, you can opt-in by setting `REQUIRE_AUTH=true`. The login page accepts the `AUTH_PIN` value.
   3. Sessions use `flask-session` with a CacheLib-backed store; cookies are upgraded to secure/HTTP-only in deployed mode.
3. Vision capture requires Firefox. The first request that needs Selenium will call `/init_webdriver`, which in turn relies on `geckodriver-autoinstaller` to download the driver if necessary.

## 5. Working with MatHud

### 5.1 Canvas Interaction

1. Double-click within the canvas to log the precise math coordinates into the chat box.
2. Pan by click-dragging; zoom with the mouse wheel (anchored around the cursor).
3. The canvas tracks undo/redo, dependencies, and name generation automatically through managers in `static/client/managers/`.

### 5.2 Conversing with the Assistant

1. Type a request in the chat input and press Enter or click **Send**. The assistant inspects the current canvas state and can call functions on your behalf.
2. Responses support Markdown and LaTeX; MathJax renders inline (`\( ... \)`) and block (`$$ ... $$`) math.
3. Sample prompts that map directly to available tools:
   1. `create point at (2, 3) named A`
   2. `draw a segment from (0,0) to (3,4) called s1`
   3. `plot y = sin(x) from -pi to pi`
   4. `evaluate expression 2*sin(pi/4)`
   5. `derive x^3 + 2x - 1`
   6. `solve system of equations: x + y = 5, x - y = 1`
   7. `evaluate linear algebra expression with matrices A=[[1,2],[3,4]]; compute inv(A)`
   8. `save workspace as "demo"` / `load workspace "demo"`
   9. `run tests`

### 5.3 Vision Mode

1. Use the **Enable Vision** checkbox in the chat header to include screenshots of the current canvas.
2. The toggle is enabled only for models marked `has_vision` in `static/ai_model.py`; non-vision models automatically disable it to avoid unsupported requests.
3. The server stores the latest snapshot under `canvas_snapshots/canvas.png` for troubleshooting.

### 5.4 Workspace Management

1. Workspaces are persisted as JSON under `workspaces/`.
2. The chat tools `save_workspace`, `load_workspace`, `list_workspaces`, and `delete_workspace` are exposed to the assistant and UI.
3. Client-side restores rebuild the Brython objects through `static/client/workspace_manager.py`.

### 5.5 Testing

1. Server tests: run `python run_server_tests.py` (add `--with-auth` to exercise authenticated flows).
2. Client tests: click **Run Tests** in the UI or ask the assistant to “run tests”. Results stream back into the chat after execution (`static/client/test_runner.py`).

## 6. Rendering Notes

1. `static/client/rendering/factory.py` instantiates renderers in preference order `canvas2d → svg → webgl`. If a constructor raises (for example, WebGL unavailable), the factory continues down the chain.
2. Canvas2D rendering (`canvas2d_renderer.py`) supports optional offscreen compositing. Toggle it with `window.MatHudCanvas2DOffscreen = true` or `localStorage["mathud.canvas2d.offscreen"] = "1"`.
3. SVG rendering (`svg_renderer.py`) mirrors the same offscreen staging controls through `window.MatHudSvgOffscreen` or `localStorage["mathud.svg.offscreen"]`.
4. The WebGL renderer (`webgl_renderer.py`) is experimental, not feature complete, and only instantiates when the browser exposes a WebGL context.

## 7. Diagram Generation

1. Generate the full suite of diagrams from the project root:
   ```sh
   python generate_diagrams_launcher.py
   ```
2. Output directories:
   1. `diagrams/generated_png/` – raster versions for quick sharing.
   2. `diagrams/generated_svg/` – scalable diagrams for documentation.
3. Additional guidance lives in `diagrams/README.md` and `diagrams/WORKFLOW_SUMMARY.md`.

## 8. Repository Guide

1. `app.py` – entry point with graceful shutdown and threaded dev server.
2. `static/`
   a. `app_manager.py`, `routes.py`, `openai_api.py`, `ai_model.py`, `tool_call_processor.py`, `workspace_manager.py`, `log_manager.py`, `webdriver_manager.py`.
   b. `client/` – Brython modules (canvas, managers, rendering, tests, utilities, workspace manager).
3. `templates/index.html` – main HTML shell that loads Brython, MathJax, styles, and UI controls.
4. `workspaces/` – saved canvas states.
5. `canvas_snapshots/` – latest Selenium captures used for vision.
6. `server_tests/` – pytest suites, including renderer plan tests under `server_tests/client_renderer/`.
7. `documentation/` – extended reference material.
8. `logs/` – session-specific server logs.

## 9. Additional Documentation

1. `documentation/Project Architecture.txt` – deep dive into system design.
2. `documentation/Reference Manual.txt` – comprehensive API and module reference.
3. `documentation/Example Prompts.txt` – curated prompts for common workflows.
