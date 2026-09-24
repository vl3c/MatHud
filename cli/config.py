"""Configuration constants for the MatHud CLI."""

from __future__ import annotations

import os
import platform as _platform
import sys
from pathlib import Path

# Project root directory (parent of cli/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Default server settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000

# PID file for tracking server process
PID_FILE = PROJECT_ROOT / ".mathud_server.pid"

# Health check settings
HEALTH_CHECK_TIMEOUT = 5  # seconds
HEALTH_CHECK_RETRIES = 60  # number of retries when waiting for server (60 seconds max)

# Browser automation settings
BROWSER_WAIT_TIMEOUT = 60  # seconds to wait for page elements
APP_READY_TIMEOUT = 30  # seconds to wait for app to be ready

# Test settings
# Client tests in Brython are ~3-5x slower on ARM64 than x86_64
if _platform.machine() in ("aarch64", "arm64"):
    CLIENT_TEST_TIMEOUT = 600  # 10 minutes for ARM64
else:
    CLIENT_TEST_TIMEOUT = 180  # 3 minutes for x86_64
CLIENT_TEST_POLL_INTERVAL = 2  # seconds between polling for results

# Screenshot settings
DEFAULT_VIEWPORT_WIDTH = 1920
DEFAULT_VIEWPORT_HEIGHT = 1080
DEFAULT_SCREENSHOT_FORMAT = "png"

# Workspace directory
WORKSPACES_DIR = PROJECT_ROOT / "workspaces"

# CLI output directory (for screenshots, etc.)
CLI_OUTPUT_DIR = Path(__file__).parent / "output"


# Python interpreter path
def get_python_path() -> Path:
    """Get the project venv's Python interpreter, or the running one if there is no venv.

    Git worktrees (and other checkouts without ./venv) fall back to sys.executable.
    """
    if os.name == "nt":  # Windows
        venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    else:  # Unix-like
        venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


# Flask app entry point
APP_MODULE = PROJECT_ROOT / "app.py"
