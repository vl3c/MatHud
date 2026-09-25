"""Shared pytest setup for the server test suite.

Live provider tests (``test_provider_connections.py`` and the integration
classes in the OpenAI API tests) run whenever an API key is present, and some
route tests reach the real provider when one is. The app loads ``.env`` files
(including the one above the main checkout), so a plain local test run could
otherwise pick up real keys and make paid API calls.

Unless ``MATHUD_LIVE_TESTS=1`` is set, provider keys are kept out of the test
process:

- they are blanked when the suite starts, and
- every ``.env`` load goes through a wrapper that blanks them again, because
  tests that save and restore a key may delete the variable, after which a
  later load would bring the real key back.

Tests that need a key set a fake one themselves (for example ``test-api-key``).
"""

from __future__ import annotations

import os
from typing import Any

import dotenv
import dotenv.main

PROVIDER_KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")
LIVE_TESTS_ENV = "MATHUD_LIVE_TESTS"


def live_tests_enabled() -> bool:
    return os.environ.get(LIVE_TESTS_ENV, "").lower() in ("1", "true", "yes")


def _blank_real_provider_keys() -> None:
    """Blank provider keys that hold anything other than a test placeholder."""
    for key in PROVIDER_KEY_VARS:
        value = os.environ.get(key)
        if value is None or value.startswith("sk-") or len(value) > 32:
            os.environ[key] = ""


if not live_tests_enabled():
    _original_load_dotenv = dotenv.main.load_dotenv

    def _guarded_load_dotenv(*args: Any, **kwargs: Any) -> bool:
        loaded = bool(_original_load_dotenv(*args, **kwargs))
        _blank_real_provider_keys()
        return loaded

    # Patch both names before any app module runs `from dotenv import load_dotenv`.
    dotenv.load_dotenv = _guarded_load_dotenv
    dotenv.main.load_dotenv = _guarded_load_dotenv
    _blank_real_provider_keys()
