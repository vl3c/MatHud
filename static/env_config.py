"""Centralized environment variable loading for the MatHud backend.

Provides helpers that consolidate the duplicated .env discovery logic
(project root, then its parent directory, then the main checkout's parent
when running from a git worktree) used by multiple API modules.
"""

from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv

# Resolve relative to this file, not the working directory the app was launched from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env_files() -> None:
    """Load .env files from the project root and the directories above it.

    The project-root .env is loaded first so its values take precedence.
    Each .env from :func:`_parent_env_candidates` is loaded afterwards, in
    order, only when the file exists, which is useful when API keys are
    stored one level above the repository.

    Calling this function multiple times is safe; ``python-dotenv`` will
    not overwrite variables that are already present in ``os.environ``.
    """
    load_dotenv()
    for env_path in _parent_env_candidates(_PROJECT_ROOT):
        if os.path.exists(env_path):
            load_dotenv(env_path)


def _parent_env_candidates(project_root: str) -> List[str]:
    """Return the .env paths above the project, nearest first.

    The first is in the project's parent directory. When the project is a git
    worktree under ``<repo>/.claude/worktrees/<name>``, that parent is the
    worktrees folder, so the main checkout's parent directory follows.
    """
    candidates = [os.path.join(os.path.dirname(project_root), ".env")]
    main_checkout = _main_checkout_root(project_root)
    if main_checkout is not None:
        candidates.append(os.path.join(os.path.dirname(main_checkout), ".env"))
    return candidates


def _main_checkout_root(project_root: str) -> Optional[str]:
    """Return ``<repo>`` when *project_root* is ``<repo>/.claude/worktrees/<name>``, else None."""
    worktrees_dir = os.path.dirname(project_root)
    claude_dir = os.path.dirname(worktrees_dir)
    if os.path.basename(worktrees_dir) == "worktrees" and os.path.basename(claude_dir) == ".claude":
        return os.path.dirname(claude_dir)
    return None


def get_api_key(
    name: str,
    *,
    required: bool = True,
    fallback: Optional[str] = None,
) -> str:
    """Return an API key from the environment, loading .env files first.

    The function checks ``os.environ`` before touching disk so that
    explicitly-set variables are returned immediately.

    Callers choose *required* based on provider semantics:
    - Opt-in providers (Anthropic, OpenRouter) use ``required=True`` because a
      missing key means the user misconfigured an explicit provider choice.
    - The default provider (OpenAI) uses ``required=False`` so the app can
      start without an OpenAI key when only third-party providers are used.

    Args:
        name: Environment variable name (e.g. ``"OPENAI_API_KEY"``).
        required: When *True* and the key is missing, raise ``ValueError``.
            When *False*, return *fallback* instead.
        fallback: Value to return when the key is absent and *required* is
            ``False``.  Defaults to ``None``, but the return type is always
            ``str`` — callers that pass ``required=False`` should supply a
            non-``None`` fallback or handle the empty-string case.

    Returns:
        The API key string.

    Raises:
        ValueError: If *required* is True and the key cannot be found.
    """
    # Fast path: already in environment (avoids disk I/O)
    api_key = os.getenv(name)
    if api_key:
        return api_key

    # Load .env files and retry
    load_env_files()
    api_key = os.getenv(name)

    if api_key:
        return api_key

    if required:
        raise ValueError(f"{name} not found in environment or .env file")

    return fallback if fallback is not None else ""
