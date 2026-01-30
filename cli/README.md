# MatHud CLI Module

Command-line interface for automating MatHud web application operations.

## Overview

The CLI module provides terminal-based control over the MatHud application, enabling:
- Server lifecycle management (start/stop/status)
- Automated test execution (server + client tests)
- Canvas operations via browser automation
- Workspace management
- AI chat interface
- Screenshot capture

## Installation

The CLI dependencies are included in the main `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required packages:
- `click>=8.0.0` - CLI framework
- `webdriver-manager>=4.0.0` - Automatic ChromeDriver management
- `psutil>=5.9.0` - Cross-platform process management

## Usage

Run commands via Python module:

```bash
python -m cli.main [COMMAND] [OPTIONS]
```

Or if installed via pip (pyproject.toml entry point):

```bash
mathud [COMMAND] [OPTIONS]
```

## Commands

### Server Management

```bash
# Start server (background process)
python -m cli.main server start [--port PORT] [--no-wait]

# Check server status
python -m cli.main server status [--port PORT] [--json]

# Stop server
python -m cli.main server stop [--port PORT]
```

**Options:**
- `--port, -p`: Server port (default: 5000)
- `--no-wait`: Don't wait for server to be ready
- `--json`: Output status as JSON

### Test Execution

```bash
# Run server-side pytest tests
python -m cli.main test server [TEST_PATH] [-k KEYWORD] [--with-auth]

# Run client-side Brython tests via headless Chrome
python -m cli.main test client [--port PORT] [--timeout SECONDS] [--start-server]

# Run all tests
python -m cli.main test all [--port PORT]
```

**Options:**
- `--port, -p`: Server port (default: 5000)
- `--timeout, -t`: Test timeout in seconds (default: 180)
- `--start-server`: Automatically start server if not running
- `--with-auth`: Enable authentication during tests
- `-k KEYWORD`: Run only tests matching keyword
- `--screenshot-output, -o`: Screenshot output path (default: `cli/output/test_results_<timestamp>.png`)
- `--no-screenshot`: Disable automatic screenshot capture

**Note:** Client tests automatically capture a screenshot showing test results before the browser closes. Screenshots are saved to `cli/output/` by default.

### Canvas Operations

```bash
# Clear canvas
python -m cli.main canvas clear [--port PORT]

# Reset view (zoom/offset)
python -m cli.main canvas reset [--port PORT]

# Zoom in/out
python -m cli.main canvas zoom [--in|--out|--factor FLOAT] [--port PORT]

# Undo/Redo
python -m cli.main canvas undo [--port PORT]
python -m cli.main canvas redo [--port PORT]

# Get canvas state as JSON
python -m cli.main canvas state [--port PORT] [--pretty]

# Execute FunctionRegistry function
python -m cli.main canvas exec FUNCTION_NAME [--args JSON] [--port PORT]
```

**Examples:**
```bash
python -m cli.main canvas exec create_point --args '{"x": 5, "y": 3, "name": "A"}'
python -m cli.main canvas exec draw_function --args '{"expression": "x**2", "x_min": -5, "x_max": 5}'
```

### Workspace Management

```bash
# List workspaces
python -m cli.main workspace list [--port PORT] [--json]

# Save workspace
python -m cli.main workspace save NAME [--port PORT]

# Load workspace
python -m cli.main workspace load NAME [--port PORT] [--json]

# Delete workspace
python -m cli.main workspace delete NAME [--port PORT] [-y]
```

### AI Chat

```bash
# Send message (streaming)
python -m cli.main chat send "MESSAGE" [--port PORT] [--model MODEL] [--vision]

# Send message (non-streaming)
python -m cli.main chat send "MESSAGE" --no-stream [--json]

# Start new conversation
python -m cli.main chat new [--port PORT]
```

**Options:**
- `--model, -m`: AI model (e.g., gpt-4o, claude-3-5-sonnet)
- `--vision, -v`: Include canvas snapshot
- `--no-stream`: Wait for complete response
- `--json`: Output as JSON

### Screenshot Capture

```bash
# Capture full page (default output: cli/output/screenshot_<timestamp>.png)
python -m cli.main screenshot capture [--port PORT]

# Capture with custom output path
python -m cli.main screenshot capture -o path/to/file.png

# Capture canvas only
python -m cli.main screenshot capture --canvas-only

# Custom viewport size
python -m cli.main screenshot capture --width 2560 --height 1440
```

**Options:**
- `--output, -o`: Output file path (default: `cli/output/screenshot_<timestamp>.png`)
- `--canvas-only`: Capture only the canvas area
- `--width, -w`: Viewport width (default: 1920)
- `--height, -h`: Viewport height (default: 1080)
- `--wait`: Seconds to wait after page load (default: 0.5)

## Module Structure

```
cli/
├── __init__.py      # Package initialization
├── config.py        # Configuration constants
├── main.py          # CLI entry point (Click groups)
├── server.py        # ServerManager class + commands
├── browser.py       # BrowserAutomation class (Selenium/Chrome)
├── tests.py         # Test execution commands
├── canvas.py        # Canvas operation commands
├── workspace.py     # Workspace management commands
├── chat.py          # AI chat interface commands
├── screenshot.py    # Screenshot capture commands
├── output/          # Generated files directory
│   └── .gitkeep
└── README.md        # This file
```

## Architecture

### Server Management (`server.py`)

- `ServerManager` class handles Flask server lifecycle
- Background process spawning with PID tracking (`.mathud_server.pid`)
- Health checks via HTTP requests to `/auth_status`
- Graceful shutdown with SIGINT (Unix) or terminate (Windows)

### Browser Automation (`browser.py`)

- `BrowserAutomation` class wraps Selenium WebDriver
- Headless Chrome with automatic ChromeDriver management
- JavaScript execution for canvas interaction
- Configurable viewport size and timeouts
- Methods: `setup()`, `navigate_to_app()`, `wait_for_app_ready()`, `execute_js()`, `capture_screenshot()`

### Test Execution (`tests.py`)

- Server tests: Wraps pytest with proper environment (PYTHONPATH, REQUIRE_AUTH)
- Client tests:
  1. Navigates to app via headless Chrome
  2. Waits for `window.startMatHudTests` to be available
  3. Calls `window.startMatHudTests()` to begin
  4. Polls `window.getMatHudTestResults()` until complete or timeout

## Configuration

Edit `cli/config.py` to customize:

```python
DEFAULT_PORT = 5000                    # Default server port
HEALTH_CHECK_TIMEOUT = 5               # HTTP request timeout (seconds)
HEALTH_CHECK_RETRIES = 60              # Max retries when starting server
BROWSER_WAIT_TIMEOUT = 60              # Browser element wait timeout
APP_READY_TIMEOUT = 30                 # Wait for app initialization
CLIENT_TEST_TIMEOUT = 180              # Client test max duration
DEFAULT_VIEWPORT_WIDTH = 1920          # Screenshot viewport width
DEFAULT_VIEWPORT_HEIGHT = 1080         # Screenshot viewport height
```

## Output Directory

Generated files (screenshots, etc.) are saved to `cli/output/` by default.

This directory is:
- Git-ignored (except `.gitkeep`)
- Auto-created on first use
- Used by screenshot commands when no output path specified

## Notes

### Stateless Browser Sessions

Each canvas/screenshot command creates a new browser session. Operations are not persisted between commands. For complex workflows, use:
- Workspace save/load to persist state
- The web UI directly for interactive work

### Headless Browser Limitations

Some tests may fail in headless mode:
- Vision toggle tests
- File picker tests
- Tests requiring user interaction

### Windows Compatibility

The CLI is tested on Windows with:
- PowerShell and Git Bash
- Python 3.11+
- Chrome browser installed

## Examples

### Quick Test Workflow

```bash
# Start server, run tests, stop server
python -m cli.main server start --port 5007
python -m cli.main test client --port 5007
python -m cli.main server stop --port 5007
```

### Canvas Automation

```bash
python -m cli.main server start --port 5007
python -m cli.main canvas clear --port 5007
python -m cli.main canvas exec create_point --args '{"x": 0, "y": 0, "name": "Origin"}'
python -m cli.main canvas exec create_point --args '{"x": 5, "y": 5, "name": "A"}'
python -m cli.main screenshot capture --port 5007
python -m cli.main server stop --port 5007
```

### Workspace Management

```bash
python -m cli.main workspace list --port 5007
python -m cli.main workspace load MyProject --port 5007
python -m cli.main screenshot capture -o my_project.png --port 5007
```

## Unit Tests

The CLI module has comprehensive unit tests in `server_tests/test_cli/`:

```
server_tests/test_cli/
├── __init__.py
├── test_config.py      # Configuration constants tests
├── test_server.py      # ServerManager class tests
├── test_browser.py     # BrowserAutomation class tests
├── test_commands.py    # Click command tests
└── test_screenshot.py  # Screenshot utility tests
```

### Running CLI Tests

```bash
# Run all CLI tests
python -m pytest server_tests/test_cli/ -v

# Run specific test file
python -m pytest server_tests/test_cli/test_server.py -v

# Run with coverage
python -m pytest server_tests/test_cli/ --cov=cli
```

### Test Coverage

- **88 tests** covering:
  - Configuration validation
  - Server lifecycle (start/stop/status)
  - Browser automation (setup, cleanup, navigation, JS execution)
  - Screenshot capture
  - Click command behavior
  - Error handling

## Known Limitations

1. **Stateless Sessions**: Each command opens a fresh browser - state is not preserved
2. **Headless Restrictions**: Vision toggle and file picker tests fail in headless mode
3. **Server Startup Time**: May need 60+ seconds on slower systems (configurable via `HEALTH_CHECK_RETRIES`)
