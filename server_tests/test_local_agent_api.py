"""
Tests for the LocalAgent local LLM provider.

Covers base URL resolution, server availability, model discovery from
``/v1/models``, registry self-registration in
static/providers/local/local_agent_api.py, and the reasoning effort sent with
every chat completion (asserted on the HTTP request body). Every HTTP call is
faked; no network, no .env files and no API keys are involved.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List, Optional

import httpx2
import pytest
from openai import OpenAI

from static.ai_model import AIModel
from static.providers.local import REASONING_EFFORT_ENV, LocalProviderRegistry
from static.providers.local.local_agent_api import (
    PROVIDER_KEY,
    LocalAgentAPI,
    _display_name_for,
)


class FakeResponse:
    """Minimal stand-in for a ``requests.Response``."""

    def __init__(self, status_code: int = 200, payload: Any = None, raise_for_status: bool = False) -> None:
        self.status_code = status_code
        self._payload = payload
        self._raise_for_status = raise_for_status

    def json(self) -> Any:
        """Return the canned payload."""
        return self._payload

    def raise_for_status(self) -> None:
        """Raise when the fake was configured to fail."""
        if self._raise_for_status:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeRequests:
    """Fake ``requests`` module recording every GET it serves."""

    def __init__(self, response: Optional[FakeResponse] = None, error: Optional[Exception] = None) -> None:
        self.response = response
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    def get(self, url: str, timeout: float | None = None) -> FakeResponse:
        """Record the call and return the canned response, or raise."""
        self.calls.append({"url": url, "timeout": timeout})
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without a configured base URL override."""
    monkeypatch.delenv(LocalAgentAPI.ENV_VAR, raising=False)
    monkeypatch.delenv(REASONING_EFFORT_ENV, raising=False)


@pytest.fixture
def fake_requests(monkeypatch: pytest.MonkeyPatch) -> FakeRequests:
    """Install a fake ``requests`` module for the provider's lazy imports."""
    fake = FakeRequests()
    monkeypatch.setitem(sys.modules, "requests", fake)
    return fake


def _models_payload(*ids: str) -> Dict[str, Any]:
    """Build an OpenAI-style ``/v1/models`` payload for the given ids."""
    return {"object": "list", "data": [{"id": model_id, "object": "model"} for model_id in ids]}


class TestDisplayNameFor:
    """Label derivation for llama-server model identifiers."""

    def test_plain_alias_unchanged(self) -> None:
        """A bare alias is returned as-is so the caller can format it."""
        assert _display_name_for("local") == "local"

    def test_windows_path_reduced_to_file_name(self) -> None:
        """A Windows .gguf path keeps only the file name without extension."""
        model_id = r"C:\models\qwen\Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"
        assert _display_name_for(model_id) == "Qwen3.8-27B-GSQ-RCO-IQ3_XXS"

    def test_posix_path_reduced_to_file_name(self) -> None:
        """A POSIX .gguf path keeps only the file name without extension."""
        assert _display_name_for("/srv/models/Ternary-Bonsai-2-27B-PQ2_0.gguf") == "Ternary-Bonsai-2-27B-PQ2_0"

    def test_extension_match_is_case_insensitive(self) -> None:
        """An upper-case .GGUF extension is stripped as well."""
        assert _display_name_for("/m/Model.GGUF") == "Model"

    def test_bare_file_name_without_directory(self) -> None:
        """A file name with no directory part still loses its extension."""
        assert _display_name_for("model.gguf") == "model"

    def test_empty_result_falls_back_to_identifier(self) -> None:
        """An identifier that reduces to nothing falls back to itself."""
        assert _display_name_for("/") == "/"


class TestResolveBaseUrl:
    """Base URL resolution from the environment."""

    def test_default_when_unset(self) -> None:
        """The documented default is used when the env var is unset."""
        assert LocalAgentAPI.resolve_base_url() == "http://127.0.0.1:8080"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The env var overrides the default."""
        monkeypatch.setenv(LocalAgentAPI.ENV_VAR, "http://192.168.1.5:9000")
        assert LocalAgentAPI.resolve_base_url() == "http://192.168.1.5:9000"

    def test_trailing_slash_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A trailing slash is removed so URL joins stay well formed."""
        monkeypatch.setenv(LocalAgentAPI.ENV_VAR, "http://localhost:8080/")
        assert LocalAgentAPI.resolve_base_url() == "http://localhost:8080"

    def test_instance_uses_class_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_get_base_url`` on an uninitialized instance matches the class method."""
        monkeypatch.setenv(LocalAgentAPI.ENV_VAR, "http://host:1234")
        instance = object.__new__(LocalAgentAPI)
        assert instance._get_base_url() == "http://host:1234"


class TestIsServerRunning:
    """Availability probing against ``/v1/models``."""

    def test_true_on_200(self, fake_requests: FakeRequests) -> None:
        """A 200 response means the server is up."""
        fake_requests.response = FakeResponse(status_code=200, payload=_models_payload("local"))
        assert LocalAgentAPI.is_server_running() is True
        assert fake_requests.calls[0]["url"] == "http://127.0.0.1:8080/v1/models"
        assert fake_requests.calls[0]["timeout"] == LocalAgentAPI.AVAILABILITY_TIMEOUT

    def test_false_on_non_200(self, fake_requests: FakeRequests) -> None:
        """A non-200 response means the server is not usable."""
        fake_requests.response = FakeResponse(status_code=503, payload=None)
        assert LocalAgentAPI.is_server_running() is False

    def test_false_on_connection_error(self, fake_requests: FakeRequests) -> None:
        """A transport failure is swallowed and reported as unavailable."""
        fake_requests.error = OSError("connection refused")
        assert LocalAgentAPI.is_server_running() is False

    def test_uses_configured_base_url(self, monkeypatch: pytest.MonkeyPatch, fake_requests: FakeRequests) -> None:
        """The probe targets the configured base URL."""
        monkeypatch.setenv(LocalAgentAPI.ENV_VAR, "http://elsewhere:9999")
        fake_requests.response = FakeResponse(status_code=200, payload=_models_payload("local"))
        LocalAgentAPI.is_server_running()
        assert fake_requests.calls[0]["url"] == "http://elsewhere:9999/v1/models"


class TestIsAvailable:
    """The abstract availability hook used by LocalProviderRegistry."""

    def test_delegates_to_is_server_running(self, fake_requests: FakeRequests) -> None:
        """``_is_available`` works on an instance built without ``__init__``."""
        fake_requests.response = FakeResponse(status_code=200, payload=_models_payload("local"))
        instance = object.__new__(LocalAgentAPI)
        assert instance._is_available() is True

    def test_registry_availability_without_init(self, fake_requests: FakeRequests) -> None:
        """The registry's object.__new__ path reaches the same probe."""
        fake_requests.response = FakeResponse(status_code=200, payload=_models_payload("local"))
        assert LocalProviderRegistry.is_provider_available(PROVIDER_KEY) is True

    def test_registry_availability_when_down(self, fake_requests: FakeRequests) -> None:
        """An unreachable server makes the provider unavailable."""
        fake_requests.error = OSError("connection refused")
        assert LocalProviderRegistry.is_provider_available(PROVIDER_KEY) is False


class TestFetchModels:
    """Model discovery from the ``/v1/models`` payload."""

    def test_alias_identifier(self, fake_requests: FakeRequests) -> None:
        """An aliased id is reported verbatim with no display_name override."""
        fake_requests.response = FakeResponse(payload=_models_payload("local"))
        assert LocalAgentAPI.fetch_models() == [{"name": "local"}]

    def test_path_identifier_gets_display_name(self, fake_requests: FakeRequests) -> None:
        """A .gguf path id is reported verbatim alongside a readable label."""
        model_id = r"C:\models\Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"
        fake_requests.response = FakeResponse(payload=_models_payload(model_id))
        assert LocalAgentAPI.fetch_models() == [{"name": model_id, "display_name": "Qwen3.8-27B-GSQ-RCO-IQ3_XXS"}]

    def test_multiple_models(self, fake_requests: FakeRequests) -> None:
        """Every advertised model is returned in payload order."""
        fake_requests.response = FakeResponse(payload=_models_payload("a", "b"))
        assert LocalAgentAPI.fetch_models() == [{"name": "a"}, {"name": "b"}]

    def test_discovery_timeout_used(self, fake_requests: FakeRequests) -> None:
        """Discovery uses the longer discovery timeout."""
        fake_requests.response = FakeResponse(payload=_models_payload("local"))
        LocalAgentAPI.fetch_models()
        assert fake_requests.calls[0]["timeout"] == LocalAgentAPI.DISCOVERY_TIMEOUT

    def test_empty_on_transport_error(self, fake_requests: FakeRequests) -> None:
        """A transport failure yields no models rather than raising."""
        fake_requests.error = OSError("connection refused")
        assert LocalAgentAPI.fetch_models() == []

    def test_empty_on_http_error(self, fake_requests: FakeRequests) -> None:
        """A failing status code yields no models rather than raising."""
        fake_requests.response = FakeResponse(status_code=500, payload=None, raise_for_status=True)
        assert LocalAgentAPI.fetch_models() == []


class TestParseModelsPayload:
    """Defensive parsing of malformed ``/v1/models`` bodies."""

    def test_non_dict_payload(self) -> None:
        """A non-dict body yields no models."""
        assert LocalAgentAPI._parse_models_payload(["local"]) == []

    def test_missing_data_key(self) -> None:
        """A body without 'data' yields no models."""
        assert LocalAgentAPI._parse_models_payload({"object": "list"}) == []

    def test_data_not_a_list(self) -> None:
        """A non-list 'data' value yields no models."""
        assert LocalAgentAPI._parse_models_payload({"data": {"id": "local"}}) == []

    def test_skips_non_dict_entries(self) -> None:
        """Entries that are not objects are skipped."""
        payload = {"data": ["local", {"id": "kept"}]}
        assert LocalAgentAPI._parse_models_payload(payload) == [{"name": "kept"}]

    def test_skips_missing_and_blank_ids(self) -> None:
        """Entries without a usable string id are skipped."""
        payload = {"data": [{"object": "model"}, {"id": ""}, {"id": 7}, {"id": "kept"}]}
        assert LocalAgentAPI._parse_models_payload(payload) == [{"name": "kept"}]


class TestToolCapableModels:
    """Tool-capable discovery bypasses the model-family allowlist."""

    def test_classmethod_keeps_unknown_alias(self, fake_requests: FakeRequests) -> None:
        """An id the allowlist has never heard of is still returned."""
        fake_requests.response = FakeResponse(payload=_models_payload("local"))
        assert LocalAgentAPI.get_tool_capable_models() == [{"name": "local"}]

    def test_classmethod_keeps_path_identifier(self, fake_requests: FakeRequests) -> None:
        """A .gguf path id survives discovery instead of being filtered out."""
        model_id = "/srv/models/Ternary-Bonsai-2-27B-PQ2_0.gguf"
        fake_requests.response = FakeResponse(payload=_models_payload(model_id))
        models = LocalAgentAPI.get_tool_capable_models()
        assert [model["name"] for model in models] == [model_id]

    def test_instance_discovery_bypasses_allowlist(self, fake_requests: FakeRequests) -> None:
        """The overridden instance path agrees with the classmethod path."""
        fake_requests.response = FakeResponse(payload=_models_payload("local"))
        instance = object.__new__(LocalAgentAPI)
        assert instance.discover_models_with_tool_support() == [{"name": "local"}]

    def test_discover_models_returns_everything(self, fake_requests: FakeRequests) -> None:
        """``_discover_models`` reports the raw server list."""
        fake_requests.response = FakeResponse(payload=_models_payload("local"))
        instance = object.__new__(LocalAgentAPI)
        assert instance._discover_models() == [{"name": "local"}]


class TestRegistration:
    """Self-registration and provider identity."""

    def test_registered_under_provider_key(self) -> None:
        """Importing the module registers the class with the local registry."""
        assert LocalProviderRegistry.get_provider_class(PROVIDER_KEY) is LocalAgentAPI

    def test_provider_key_value(self) -> None:
        """The registry key is the stable 'local_agent' identifier."""
        assert PROVIDER_KEY == "local_agent"

    def test_provider_name(self) -> None:
        """The human-facing provider name is used in log and error messages."""
        instance = object.__new__(LocalAgentAPI)
        assert instance._get_provider_name() == "LocalAgent"

    def test_declares_env_var_and_default(self) -> None:
        """The base URL contract is declared as class attributes."""
        assert LocalAgentAPI.ENV_VAR == "LOCAL_AGENT_BASE_URL"
        assert LocalAgentAPI.DEFAULT_URL == "http://127.0.0.1:8080"


class RecordingServer:
    """Mocked llama-server transport recording each chat completion request body."""

    def __init__(self) -> None:
        self.bodies: List[Dict[str, Any]] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if body.get("stream"):
            chunk = {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
            }
            sse = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=sse)
        completion = {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"},
            ],
        }
        return httpx2.Response(200, json=completion)


def _api_for(server: RecordingServer) -> LocalAgentAPI:
    api = LocalAgentAPI(model=AIModel("qwen3.8-27b", has_vision=False, provider="local_agent"), tools=[])
    http_client = httpx2.Client(transport=httpx2.MockTransport(server.handle))
    api.client = OpenAI(api_key="k", base_url="http://mock/v1", http_client=http_client, max_retries=0)
    return api


def _send_both(api: LocalAgentAPI) -> None:
    """One non-streamed and one streamed request."""
    api.create_chat_completion(json.dumps({"user_message": "hi"}))
    list(api.create_chat_completion_stream(json.dumps({"user_message": "again"})))


class TestReasoningEffort:
    """chat_template_kwargs.reasoning_effort on every request (MATHUD_LOCAL_REASONING_EFFORT)."""

    def test_default_effort_is_medium_in_both_request_paths(self) -> None:
        server = RecordingServer()
        _send_both(_api_for(server))
        assert len(server.bodies) == 2
        assert not server.bodies[0].get("stream")
        assert server.bodies[1]["stream"] is True
        for body in server.bodies:
            assert body["chat_template_kwargs"] == {"reasoning_effort": "medium"}
            assert body["temperature"] == 0.2
            assert body["max_tokens"] == 16000

    @pytest.mark.parametrize("raw, expected", [("low", "low"), (" HIGH ", "high"), ("xhigh", "xhigh"), ("max", "max")])
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: str) -> None:
        monkeypatch.setenv(REASONING_EFFORT_ENV, raw)
        server = RecordingServer()
        _send_both(_api_for(server))
        assert [body["chat_template_kwargs"] for body in server.bodies] == [{"reasoning_effort": expected}] * 2

    @pytest.mark.parametrize("raw", ["default", "Default"])
    def test_default_omits_the_field(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(REASONING_EFFORT_ENV, raw)
        server = RecordingServer()
        _send_both(_api_for(server))
        assert len(server.bodies) == 2
        assert all("chat_template_kwargs" not in body for body in server.bodies)

    # "none" is not a way to turn reasoning off (omitting the field leaves the template default).
    @pytest.mark.parametrize("raw", ["turbo", "none"])
    def test_invalid_value_warns_and_uses_the_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
    ) -> None:
        monkeypatch.setenv(REASONING_EFFORT_ENV, raw)
        with caplog.at_level(logging.WARNING, logger="mathud"):
            api = _api_for(RecordingServer())
        assert api.reasoning_effort == "medium"
        assert any(REASONING_EFFORT_ENV in record.getMessage() for record in caplog.records)

    def test_empty_value_uses_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(REASONING_EFFORT_ENV, "  ")
        assert _api_for(RecordingServer()).reasoning_effort == "medium"
