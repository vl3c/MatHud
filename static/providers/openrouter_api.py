"""
MatHud OpenRouter API Provider

OpenRouter API implementation as a self-contained provider module.
Uses OpenAI SDK with custom base_url for OpenRouter's OpenAI-compatible API.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from static.ai_model import AIModel
from static.functions_definitions import FunctionDefinition
from static.openai_api_base import ToolMode
from static.openai_completions_api import OpenAIChatCompletionsAPI
from static.providers import PROVIDER_OPENROUTER, ProviderRegistry


def _get_openrouter_api_key() -> str:
    """Get the OpenRouter API key from environment."""
    load_dotenv()
    parent_env = os.path.join(os.path.dirname(os.getcwd()), ".env")
    if os.path.exists(parent_env):
        load_dotenv(parent_env)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment or .env file")
    return api_key


class OpenRouterAPI(OpenAIChatCompletionsAPI):
    """OpenRouter API provider.

    Uses OpenAI SDK with OpenRouter's OpenAI-compatible endpoint.
    Inherits all streaming and completion logic from OpenAIChatCompletionsAPI.
    """

    # OpenRouter base URL
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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
            model: AI model to use. Defaults to Gemini 2.5 Pro.
            temperature: Sampling temperature.
            tools: Custom tool definitions.
            max_tokens: Maximum tokens in response.
            tool_mode: Tool mode - "full" or "search".
        """
        # Initialize the OpenAI client with OpenRouter's base URL
        self.client = OpenAI(
            api_key=_get_openrouter_api_key(),
            base_url=self.OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://mathud.app",
                "X-Title": "MatHud",
            },
        )

        # Set model (default to Gemini 2.5 Pro if not specified)
        self.model: AIModel = model if model is not None else AIModel.from_identifier("google/gemini-2.5-pro")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._tool_mode: ToolMode = tool_mode
        self._custom_tools: Optional[Sequence[FunctionDefinition]] = tools
        self._injected_tools: bool = False
        self.tools: Sequence[FunctionDefinition] = self._resolve_tools()

        # Initialize message history with developer message
        from static.openai_api_base import OpenAIAPIBase

        self.messages = [{"role": "developer", "content": OpenAIAPIBase.DEV_MSG}]


# Self-register with provider registry
ProviderRegistry.register(PROVIDER_OPENROUTER, OpenRouterAPI)
