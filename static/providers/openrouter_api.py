"""
MatHud OpenRouter API Provider

OpenRouter API implementation as a self-contained provider module.
Uses OpenAI SDK with custom base_url for OpenRouter's OpenAI-compatible API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import httpx2
from openai import OpenAI

from static.ai_model import AIModel
from static.env_config import get_api_key
from static.functions_definitions import FunctionDefinition
from static.openai_api_base import ToolMode
from static.openai_completions_api import OpenAIChatCompletionsAPI
from static.providers import PROVIDER_OPENROUTER, ProviderRegistry


def _get_openrouter_api_key() -> str:
    """Get the OpenRouter API key from environment."""
    # required=True (default): OpenRouter is an explicitly opted-in provider,
    # so a missing key is a configuration error rather than a graceful fallback.
    return get_api_key("OPENROUTER_API_KEY")


class OpenRouterAPI(OpenAIChatCompletionsAPI):
    """OpenRouter API provider.

    Uses OpenAI SDK with OpenRouter's OpenAI-compatible endpoint.
    Inherits all streaming and completion logic from OpenAIChatCompletionsAPI.
    """

    # OpenRouter base URL
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    # OpenRouter returns reasoning as reasoning_details, which must be sent back
    # unchanged after a tool call (Gemini 3 thought signatures).
    PRESERVE_REASONING_DETAILS = True

    # OpenRouter emits keepalive bytes while the upstream model is still
    # processing, so the read timeout only trips when the connection goes
    # truly silent (a stalled provider), not while a model is thinking.
    # With one retry the worst case is ~2x60s + backoff (~125s), which must
    # stay below the client's REASONING_TIMEOUT_MS (300s) so a stall surfaces
    # as a proper error event instead of a blind UI timeout.
    REQUEST_TIMEOUT = httpx2.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
    MAX_RETRIES = 1

    def __init__(
        self,
        model: Optional[AIModel] = None,
        temperature: float = 0.2,
        tools: Optional[Sequence[FunctionDefinition]] = None,
        max_tokens: int = 16000,
        tool_mode: ToolMode = "full",
    ) -> None:
        """Initialize OpenRouter API client.

        Args:
            model: AI model to use. Defaults to Gemini 3.8 Flash.
            temperature: Sampling temperature.
            tools: Custom tool definitions.
            max_tokens: Maximum tokens in response.
            tool_mode: Tool mode - "full" or "search".
        """
        # Initialize the OpenAI client with OpenRouter's base URL
        self.client = OpenAI(
            api_key=_get_openrouter_api_key(),
            base_url=self.OPENROUTER_BASE_URL,
            timeout=self.REQUEST_TIMEOUT,
            max_retries=self.MAX_RETRIES,
            default_headers={
                "HTTP-Referer": "https://mathud.app",
                "X-Title": "MatHud",
            },
        )

        # Set model (default to Gemini 3.8 Flash if not specified)
        self.model: AIModel = model if model is not None else AIModel.from_identifier("google/gemini-3.8-flash")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._tool_mode: ToolMode = tool_mode
        self._custom_tools: Optional[Sequence[FunctionDefinition]] = tools
        self._injected_tools: bool = False
        self.tools: Sequence[FunctionDefinition] = self._resolve_tools()

        # Initialize message history with the system prompt
        self.messages = [{"role": self._system_role(), "content": self._build_system_prompt()}]

    def _system_role(self) -> str:
        """Role of the system prompt: "developer" for OpenAI models, "system" for the rest,
        since not every upstream provider understands the developer role."""
        return "developer" if self.model.id.startswith("openai/") else "system"

    def _sync_system_role(self) -> None:
        """Give the leading system prompt the role the current model expects."""
        if self.messages and self.messages[0].get("role") in ("developer", "system"):
            self.messages[0]["role"] = self._system_role()

    def set_model(self, identifier: str) -> None:
        """Set the model and match the system prompt role to it."""
        super().set_model(identifier)
        self._sync_system_role()

    def reset_conversation(self) -> None:
        """Reset the conversation, keeping the system prompt role of the current model."""
        super().reset_conversation()
        self._sync_system_role()


# Self-register with provider registry
ProviderRegistry.register(PROVIDER_OPENROUTER, OpenRouterAPI)
