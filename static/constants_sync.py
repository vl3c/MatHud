"""
Utilities for synchronizing client-side constants to the server runtime.

Some server modules need access to values that are defined for the Brython
frontend (for example maximum label length). Importing the Brython module
directly is undesirable, so we mirror the file into the server namespace
before it is imported.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# Source constants live in the client package.
_CLIENT_CONSTANTS_PATH = Path(__file__).resolve().parent / "client" / "constants.py"

# Mirrored copy that server-side modules can import.
_SERVER_MIRROR_NAME = "_client_constants_runtime"
_SERVER_CONSTANTS_PATH = Path(__file__).resolve().parent / f"{_SERVER_MIRROR_NAME}.py"

# Fully-qualified module path for imports.
SERVER_CONSTANTS_MODULE = f"{__package__}.{_SERVER_MIRROR_NAME}"


def ensure_client_constants_available() -> None:
    """Copy the client constants module into the server package if needed."""
    try:
        src_mtime = _CLIENT_CONSTANTS_PATH.stat().st_mtime
        dest_mtime = _SERVER_CONSTANTS_PATH.stat().st_mtime if _SERVER_CONSTANTS_PATH.exists() else -1
        if dest_mtime < src_mtime:
            _SERVER_CONSTANTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_CLIENT_CONSTANTS_PATH, _SERVER_CONSTANTS_PATH)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Client constants file not found: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to synchronize client constants: {exc}") from exc

