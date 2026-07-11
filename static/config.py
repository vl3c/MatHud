"""
MatHud Server-Side Configuration Constants

Centralized configuration values for server-side modules.
Defines directory paths, schema versions, and other shared settings.

Categories:
    - Workspace Management: Storage directories and schema versioning
    - Canvas Snapshots: Screenshot storage paths

Dependencies:
    - os: Path construction for snapshot paths
"""

from __future__ import annotations

import os

# ===== WORKSPACE MANAGEMENT CONSTANTS =====
# Directory and versioning for workspace persistence
WORKSPACES_DIR: str = "workspaces"
CURRENT_WORKSPACE_SCHEMA_VERSION: int = 1

# ===== CANVAS SNAPSHOT CONSTANTS =====
# Paths for Selenium-captured canvas screenshots
CANVAS_SNAPSHOT_DIR: str = "canvas_snapshots"
CANVAS_SNAPSHOT_PATH: str = os.path.join(CANVAS_SNAPSHOT_DIR, "canvas.png")
