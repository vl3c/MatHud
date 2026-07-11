"""
Tests for OpenRouter client timeout configuration and timeout error surfacing.

Verifies the OpenAI SDK client used for OpenRouter is constructed with an
explicit timeout and retry policy so a stalled upstream provider surfaces as
a server-side error event instead of hanging until the UI times out blind,
and that both pre-stream and mid-stream timeouts yield the user-facing
timeout message through the streaming error path.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List
from unittest.mock import MagicMock, patch

import httpx
from openai import APITimeoutError

from static.ai_model import AIModel
from static.openai_api_base import PROVIDER_TIMEOUT_MESSAGE, stream_error_user_message
from static.providers.openrouter_api import OpenRouterAPI


def _timeout_error() -> APITimeoutError:
    request = httpx.Request("POST", OpenRouterAPI.OPENROUTER_BASE_URL + "/chat/completions")
    return APITimeoutError(request=request)


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

    def test_worst_case_below_client_reasoning_timeout(self) -> None:
        # The browser client gives post-tool-call rounds REASONING_TIMEOUT_MS
        # (300s, static/client/constants.py); the server must fail first so
        # the UI receives a real error event rather than timing out blind.
        # Worst case is (1 + MAX_RETRIES) read timeouts plus retry backoff.
        backoff_allowance = 10.0
        worst_case = (1 + OpenRouterAPI.MAX_RETRIES) * OpenRouterAPI.REQUEST_TIMEOUT.read + backoff_allowance
        self.assertLess(worst_case, 300.0)

    def test_client_retries_limited_to_one(self) -> None:
        # Fail fast on a silent upstream: one retry, not the SDK default of 2.
        api = self._make_api()
        self.assertEqual(api.client.max_retries, 1)


class TestStreamTimeoutMessage(unittest.TestCase):
    """Timeouts must surface the user-facing timeout message via the stream."""

    def _make_api_with_mock_client(self) -> tuple[OpenRouterAPI, MagicMock]:
        model = AIModel.from_identifier("google/gemini-2.5-pro")
        with patch("static.providers.openrouter_api.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
                api = OpenRouterAPI(model=model, tools=[])
        return api, mock_client

    def _collect_final_events(self, api: OpenRouterAPI) -> List[Dict[str, Any]]:
        prompt = json.dumps({"user_message": "Hi", "use_vision": False})
        events = list(api.create_chat_completion_stream(prompt))
        return [e for e in events if e.get("type") == "final"]

    def test_pre_stream_timeout_yields_timeout_message(self) -> None:
        # No response headers at all: the SDK raises APITimeoutError from the
        # initial request. This is the stall observed in production.
        api, mock_client = self._make_api_with_mock_client()
        mock_client.chat.completions.create.side_effect = _timeout_error()

        final_events = self._collect_final_events(api)

        self.assertEqual(len(final_events), 1)
        self.assertEqual(final_events[0]["ai_message"], PROVIDER_TIMEOUT_MESSAGE)
        self.assertEqual(final_events[0]["finish_reason"], "error")

    def test_mid_stream_timeout_yields_timeout_message(self) -> None:
        # After headers arrive the SDK no longer wraps transport timeouts:
        # a silent gap between chunks raises raw httpx.ReadTimeout.
        def stalled_stream() -> Iterator[Any]:
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Partial", tool_calls=None), finish_reason=None)]
            )
            raise httpx.ReadTimeout("read timed out")

        api, mock_client = self._make_api_with_mock_client()
        mock_client.chat.completions.create.return_value = stalled_stream()

        final_events = self._collect_final_events(api)

        self.assertEqual(len(final_events), 1)
        self.assertEqual(final_events[0]["ai_message"], PROVIDER_TIMEOUT_MESSAGE)
        self.assertEqual(final_events[0]["finish_reason"], "error")

    def test_non_timeout_error_keeps_generic_message(self) -> None:
        api, mock_client = self._make_api_with_mock_client()
        mock_client.chat.completions.create.side_effect = ValueError("boom")

        final_events = self._collect_final_events(api)

        self.assertEqual(len(final_events), 1)
        self.assertNotEqual(final_events[0]["ai_message"], PROVIDER_TIMEOUT_MESSAGE)
        self.assertEqual(final_events[0]["finish_reason"], "error")


class TestStreamErrorUserMessage(unittest.TestCase):
    """Unit tests for the shared timeout-to-message mapping."""

    def test_api_timeout_error_maps_to_timeout_message(self) -> None:
        self.assertEqual(stream_error_user_message(_timeout_error(), "default"), PROVIDER_TIMEOUT_MESSAGE)

    def test_httpx_read_timeout_maps_to_timeout_message(self) -> None:
        self.assertEqual(stream_error_user_message(httpx.ReadTimeout("slow"), "default"), PROVIDER_TIMEOUT_MESSAGE)

    def test_other_exceptions_map_to_default(self) -> None:
        self.assertEqual(stream_error_user_message(RuntimeError("boom"), "default"), "default")

    def test_responses_api_stream_error_uses_timeout_message(self) -> None:
        from static.openai_responses_api import OpenAIResponsesAPI

        with patch("static.openai_api_base.OpenAI"):
            api = OpenAIResponsesAPI()
        events = list(api._handle_stream_error(_timeout_error()))
        final_events = [e for e in events if e.get("type") == "final"]
        self.assertEqual(len(final_events), 1)
        self.assertEqual(final_events[0]["ai_message"], PROVIDER_TIMEOUT_MESSAGE)


if __name__ == "__main__":
    unittest.main()
