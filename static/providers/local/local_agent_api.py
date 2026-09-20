"""
MatHud LocalAgent API Provider

Local LLM provider backed by a llama.cpp ``llama-server`` instance (or any other
backend exposing the same OpenAI-compatible surface) running on the machine.

The server hosts exactly one model at a time and advertises it through
``/v1/models``. The identifier reported there is whatever the server was started
with: an ``--alias`` value such as ``local``, or the ``.gguf`` file path when no
alias was given. The provider always uses the advertised identifier verbatim so
that swapping the running model needs no code change.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any, Dict, List, Optional

from static.ai_model import AIModel
from static.functions_definitions import FunctionDefinition
from static.providers.local import LocalLLMBase, LocalProviderRegistry

_logger = logging.getLogger("mathud")

PROVIDER_KEY = "local_agent"

_GGUF_SUFFIX = ".gguf"


def _display_name_for(model_id: str) -> str:
    """Build a readable label for a llama-server model identifier.

    File-path identifiers are reduced to their bare file name without the
    ``.gguf`` extension. Identifiers that are already plain names are returned
    unchanged so the caller can fall back to its own formatting.

    Args:
        model_id: The identifier advertised by the server

    Returns:
        A display-friendly label, or the identifier itself when nothing to strip
    """
    name = model_id.replace("\\", "/").rstrip("/")
    name = name.rsplit("/", 1)[-1]
    if name.lower().endswith(_GGUF_SUFFIX):
        name = name[: -len(_GGUF_SUFFIX)]
    return name or model_id


class LocalAgentAPI(LocalLLMBase):
    """Local LLM provider for an OpenAI-compatible ``llama-server``.

    Discovers whichever single model the server currently hosts and exposes it
    without consulting the model-family allowlist used by other local providers:
    llama.cpp implements tool calling through the model's own chat template, so
    the identifier carries no reliable capability information.
    """

    ENV_VAR = "LOCAL_AGENT_BASE_URL"
    DEFAULT_URL = "http://127.0.0.1:8080"

    # Short enough to keep the model dropdown responsive when no server runs.
    AVAILABILITY_TIMEOUT = 2.0
    DISCOVERY_TIMEOUT = 5.0

    def __init__(
        self,
        model: Optional[AIModel] = None,
        temperature: float = 0.2,
        tools: Optional[Sequence[FunctionDefinition]] = None,
        max_tokens: int = 16000,
    ) -> None:
        """Initialize the LocalAgent client.

        Args:
            model: AI model to use. If None, the base class placeholder is used.
            temperature: Sampling temperature.
            tools: Custom tool definitions.
            max_tokens: Maximum tokens in response.
        """
        super().__init__(
            model=model,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
        )

    @classmethod
    def resolve_base_url(cls) -> str:
        """Resolve the server base URL from the environment.

        Returns:
            The configured base URL without a trailing slash
        """
        return os.getenv(cls.ENV_VAR, cls.DEFAULT_URL).rstrip("/")

    def _get_base_url(self) -> str:
        """Get the server base URL.

        Returns:
            The server URL from environment or default
        """
        return self.resolve_base_url()

    def _get_provider_name(self) -> str:
        """Get the provider name.

        Returns:
            'LocalAgent'
        """
        return "LocalAgent"

    def _is_available(self) -> bool:
        """Check whether the local server is running.

        Called on instances built with ``object.__new__`` by
        ``LocalProviderRegistry``, so it must not touch instance state.

        Returns:
            True if the server answers ``/v1/models``
        """
        return self.is_server_running()

    def _discover_models(self) -> List[Dict[str, Any]]:
        """Query the server for the model it currently hosts.

        Returns:
            List of model info dicts with 'name' and optional 'display_name'
        """
        return self.fetch_models()

    def discover_models_with_tool_support(self) -> List[Dict[str, Any]]:
        """Return the hosted models without the model-family allowlist filter.

        Returns:
            List of model info dicts for every model the server advertises
        """
        return self.fetch_models()

    @classmethod
    def is_server_running(cls) -> bool:
        """Check whether the local server answers requests.

        Returns:
            True if ``/v1/models`` responds with 200
        """
        import requests

        try:
            response = requests.get(
                f"{cls.resolve_base_url()}/v1/models",
                timeout=cls.AVAILABILITY_TIMEOUT,
            )
            return response.status_code == 200
        except Exception as e:
            _logger.debug(f"LocalAgent server not available: {e}")
            return False

    @classmethod
    def fetch_models(cls) -> List[Dict[str, Any]]:
        """Read the model list advertised by the server.

        Returns:
            List of model info dicts with 'name' and optional 'display_name'
        """
        import requests

        try:
            response = requests.get(
                f"{cls.resolve_base_url()}/v1/models",
                timeout=cls.DISCOVERY_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            _logger.warning(f"Failed to discover LocalAgent models: {e}")
            return []

        return cls._parse_models_payload(payload)

    @classmethod
    def get_tool_capable_models(cls) -> List[Dict[str, Any]]:
        """Get the models usable for tool calling.

        Every hosted model qualifies: llama.cpp drives tool calls from the
        model's chat template, so the shared ``supports_tools`` allowlist would
        only discard identifiers it has never heard of, such as ``local`` or a
        ``.gguf`` path.

        Returns:
            List of model info dicts for every model the server advertises
        """
        return cls.fetch_models()

    @classmethod
    def _parse_models_payload(cls, payload: Any) -> List[Dict[str, Any]]:
        """Convert an OpenAI ``/v1/models`` payload into model info dicts.

        Args:
            payload: The decoded JSON response body

        Returns:
            List of model info dicts with 'name' and optional 'display_name'
        """
        if not isinstance(payload, dict):
            return []

        entries = payload.get("data")
        if not isinstance(entries, list):
            return []

        models: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue

            info: Dict[str, Any] = {"name": model_id}
            display_name = _display_name_for(model_id)
            if display_name != model_id:
                info["display_name"] = display_name
            models.append(info)

        return models


# Self-register with local provider registry
LocalProviderRegistry.register(PROVIDER_KEY, LocalAgentAPI)
