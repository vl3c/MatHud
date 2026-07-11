"""
Tests for OpenRouter client timeout and retry configuration.

Verifies the OpenAI SDK client used for OpenRouter is constructed with an
explicit timeout and retry policy so a stalled upstream provider surfaces as
a server-side error event instead of hanging until the UI times out blind.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from static.ai_model import AIModel
from static.providers.openrouter_api import OpenRouterAPI


class TestOpenRouterClientConfig(unittest.TestCase):
    """OpenRouterAPI must configure explicit timeouts and retries."""

    def _make_api(self) -> OpenRouterAPI:
        model = AIModel.from_identifier("google/gemini-2.5-pro")
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            return OpenRouterAPI(model=model, tools=[])

    def test_client_uses_explicit_timeout(self) -> None:
        api = self._make_api()
        timeout = api.client.timeout
        self.assertIsInstance(timeout, httpx.Timeout)
        assert isinstance(timeout, httpx.Timeout)
        self.assertEqual(timeout.connect, 10.0)
        self.assertEqual(timeout.read, 60.0)

    def test_read_timeout_below_client_reasoning_timeout(self) -> None:
        # The browser client gives post-tool-call rounds REASONING_TIMEOUT_MS
        # (300s, static/client/constants.py); the server must fail first so
        # the UI receives a real error event rather than timing out blind.
        self.assertLess(OpenRouterAPI.REQUEST_TIMEOUT.read, 300.0)

    def test_client_retries_enabled(self) -> None:
        api = self._make_api()
        self.assertEqual(api.client.max_retries, 2)


if __name__ == "__main__":
    unittest.main()
