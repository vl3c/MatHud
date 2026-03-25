"""Centralized environment variable loading for the MatHud backend.

Provides helpers that consolidate the duplicated .env discovery logic
(project root then parent directory) used by multiple API modules.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv


def load_env_files() -> None:
    """Load .env files from the project root and its parent directory.

    The project-root .env is loaded first so its values take precedence.
    A parent-directory .env is loaded afterwards only when the file exists,
    which is useful when API keys are stored one level above the repository.

    Calling this function multiple times is safe; ``python-dotenv`` will
    not overwrite variables that are already present in ``os.environ``.
    """
    load_dotenv()
    parent_env = os.path.join(os.path.dirname(os.getcwd()), ".env")
    if os.path.exists(parent_env):
        load_dotenv(parent_env)


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
