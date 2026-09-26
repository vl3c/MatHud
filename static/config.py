"""
MatHud Server-Side Configuration Constants

Centralized configuration values for server-side modules.
Defines directory paths, schema versions, and other shared settings.

Categories:
    - Workspace Management: Storage directories and schema versioning
    - Request Size & Image Validation: Upload and image size caps
"""

from __future__ import annotations

import os

# ===== WORKSPACE MANAGEMENT CONSTANTS =====
# Directory and versioning for workspace persistence
WORKSPACES_DIR: str = "workspaces"
CURRENT_WORKSPACE_SCHEMA_VERSION: int = 1
# Environment variable that moves workspace storage elsewhere. Test harnesses
# (the scenario runner) point it at a temporary directory so their saves never
# touch the user's workspaces.
WORKSPACES_DIR_ENV: str = "MATHUD_WORKSPACES_DIR"


def get_workspaces_dir() -> str:
    """Return the workspace directory: ``MATHUD_WORKSPACES_DIR`` when set, else ``WORKSPACES_DIR``."""
    override = os.environ.get(WORKSPACES_DIR_ENV, "").strip()
    return override or WORKSPACES_DIR


# ===== REQUEST SIZE & IMAGE VALIDATION CONSTANTS =====
# Server-side hard caps enforced independently of the (advisory) client-side
# limits in static/client/constants.py. These defend the backend against
# oversized or abusive uploads regardless of what the client claims to enforce.

# Maximum size of an entire request body (bytes). Sized to comfortably hold the
# legitimate worst case: MAX_ATTACHED_IMAGES base64 data URLs (~13.4 MB each for
# a 10 MB binary image) plus message text and JSON overhead. 80 MB leaves ample
# headroom while still bounding memory use per request.
MAX_CONTENT_LENGTH_BYTES: int = 80 * 1024 * 1024

# Maximum number of attached images accepted per request.
MAX_ATTACHED_IMAGES: int = 5

# Per-image base64 payload cap (bytes). ~20 MB base64 ≈ 15 MB of binary image
# data, comfortably above the client's 10 MB warning threshold so legitimate
# uploads pass, while still bounding a single image's footprint.
MAX_IMAGE_BASE64_BYTES: int = 20 * 1024 * 1024
