"""
Utilities for mirroring client-side modules to the server runtime.

Some server modules need access to values that are defined for the Brython
frontend (for example maximum label length). Importing the Brython module
directly is undesirable, so we mirror the file into the server namespace
before it is imported.
"""

from __future__ import annotations

import shutil
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent

_CLIENT_CONSTANTS_PATH = _BASE_DIR / "client" / "constants.py"
_SERVER_MIRROR_NAME = "_client_constants_runtime"
_SERVER_CONSTANTS_PATH = _BASE_DIR / f"{_SERVER_MIRROR_NAME}.py"
SERVER_CONSTANTS_MODULE = f"{__package__}.{_SERVER_MIRROR_NAME}"

_POLYGON_SUBTYPES_CLIENT_PATH = _BASE_DIR / "client" / "utils" / "polygon_subtypes.py"
_POLYGON_SUBTYPES_MIRROR_NAME = "_polygon_subtypes_runtime"
_POLYGON_SUBTYPES_SERVER_PATH = _BASE_DIR / f"{_POLYGON_SUBTYPES_MIRROR_NAME}.py"
POLYGON_SUBTYPES_MODULE = f"{__package__}.{_POLYGON_SUBTYPES_MIRROR_NAME}"


def _mirror_if_stale(source: Path, destination: Path) -> None:
    try:
        src_mtime = source.stat().st_mtime
        dest_mtime = destination.stat().st_mtime if destination.exists() else -1
        if dest_mtime < src_mtime:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Client module not found: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to synchronize client module {source.name}: {exc}") from exc


def ensure_client_constants_available() -> None:
    """Mirror the client constants module into the server package if needed."""
    _mirror_if_stale(_CLIENT_CONSTANTS_PATH, _SERVER_CONSTANTS_PATH)


def ensure_polygon_subtypes_available() -> None:
    """Mirror the polygon subtype enums into the server package if needed."""
    _mirror_if_stale(_POLYGON_SUBTYPES_CLIENT_PATH, _POLYGON_SUBTYPES_SERVER_PATH)

