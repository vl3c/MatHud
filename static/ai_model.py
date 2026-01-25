"""
MatHud AI Model Configuration

AI model definitions and capability management for multiple AI providers.
Handles model-specific features like vision support and provides factory methods.

Dependencies:
    - None (pure configuration module)
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, TypedDict

# Provider constants
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENROUTER = "openrouter"


class ModelConfig(TypedDict, total=False):
    """Configuration for an AI model."""

    has_vision: bool
    is_reasoning_model: bool
    reasoning_effort: str
    provider: str


ModelConfigDict = Dict[str, ModelConfig]
ModelIdentifier = Literal[
    # OpenAI models
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5-chat-latest",
    "gpt-5.2-chat-latest",
    "gpt-5.2",
    "gpt-5-nano",
    "gpt-3.5-turbo",
    "o3",
    "o4-mini",
    # Anthropic models
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    # OpenRouter models (paid)
    "google/gemini-2.5-pro",
    "google/gemini-3-pro-preview",
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v3.2",
    "x-ai/grok-code-fast-1",
    "z-ai/glm-4.7",
    "minimax/minimax-m2.1",
    # OpenRouter models (free)
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "openai/gpt-oss-20b:free",
    "openai/gpt-oss-120b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "z-ai/glm-4.5-air:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
]


class AIModel:
    """AI model configuration and capability management.

    Manages model-specific capabilities (like vision support) and provides
    factory methods for creating model instances from identifiers.
    """

    # Dictionary of model configurations
    MODEL_CONFIGS: ModelConfigDict = {
        # ===================
        # OpenAI Models
        # ===================
        # Reasoning models (use Responses API)
        "gpt-5-chat-latest": {
            "has_vision": True,
            "is_reasoning_model": True,
            "provider": PROVIDER_OPENAI,
        },
        "gpt-5.2-chat-latest": {
            "has_vision": True,
            "is_reasoning_model": True,
            "provider": PROVIDER_OPENAI,
        },
        "gpt-5.2": {
            "has_vision": True,
            "is_reasoning_model": True,
            "reasoning_effort": "medium",
            "provider": PROVIDER_OPENAI,
        },
        "o3": {
            "has_vision": False,
            "is_reasoning_model": True,
            "provider": PROVIDER_OPENAI,
        },
        "o4-mini": {
            "has_vision": True,
            "is_reasoning_model": True,
            "provider": PROVIDER_OPENAI,
        },
        # Standard models (use Chat Completions API)
        "gpt-4.1": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENAI,
        },
        "gpt-4.1-mini": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENAI,
        },
        "gpt-4.1-nano": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENAI,
        },
        "gpt-4o": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENAI,
        },
        "gpt-4o-mini": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENAI,
        },
        "gpt-5-nano": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENAI,
        },
        "gpt-3.5-turbo": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENAI,
        },
        # ===================
        # Anthropic Models
        # ===================
        "claude-sonnet-4-5-20250929": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_ANTHROPIC,
        },
        "claude-haiku-4-5-20251001": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_ANTHROPIC,
        },
        # ===================
        # OpenRouter Models (Paid)
        # ===================
        "google/gemini-2.5-pro": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
        },
        "google/gemini-3-pro-preview": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
        },
        "google/gemini-3-flash-preview": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
        },
        "deepseek/deepseek-v3.2": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
        },
        "x-ai/grok-code-fast-1": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
        },
        "z-ai/glm-4.7": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
        },
        "minimax/minimax-m2.1": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
        },
        # ===================
        # OpenRouter Models (Free)
        # ===================
        "meta-llama/llama-3.3-70b-instruct:free": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
        },
        "google/gemma-3-27b-it:free": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
        },
        "openai/gpt-oss-20b:free": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
        },
        "openai/gpt-oss-120b:free": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
        },
        "qwen/qwen3-next-80b-a3b-instruct:free": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
        },
        "z-ai/glm-4.5-air:free": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
        },
        "nvidia/nemotron-3-nano-30b-a3b:free": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
        },
    }

    DEFAULT_MODEL = "gpt-5.2"

    def __init__(
        self,
        identifier: str,
        has_vision: bool,
        is_reasoning_model: bool = False,
        reasoning_effort: Optional[str] = None,
        provider: str = PROVIDER_OPENAI,
    ) -> None:
        """Initialize AIModel instance.

        Args:
            identifier: Model identifier string (e.g., 'gpt-4.1')
            has_vision: Boolean indicating vision capability support
            is_reasoning_model: Boolean indicating if model uses Responses API with reasoning
            reasoning_effort: Optional reasoning effort setting for reasoning models (e.g., 'medium')
            provider: API provider name ('openai', 'anthropic', 'openrouter')
        """
        self.id: str = identifier
        self.has_vision: bool = has_vision
        self.is_reasoning_model: bool = is_reasoning_model
        self.reasoning_effort: Optional[str] = reasoning_effort
        self.provider: str = provider

    @staticmethod
    def from_identifier(identifier: str) -> AIModel:
        """Create AIModel instance from identifier string.

        Args:
            identifier: Model identifier string

        Returns:
            AIModel: Configured model instance
        """
        config = AIModel.MODEL_CONFIGS.get(
            identifier,
            {"has_vision": False, "is_reasoning_model": False, "provider": PROVIDER_OPENAI},
        )
        return AIModel(
            identifier=identifier,
            has_vision=config.get("has_vision", False),
            is_reasoning_model=config.get("is_reasoning_model", False),
            reasoning_effort=config.get("reasoning_effort"),
            provider=config.get("provider", PROVIDER_OPENAI),
        )

    @staticmethod
    def get_default_model() -> AIModel:
        """Get the default AI model instance.
        
        Returns:
            AIModel: Default model configuration
        """
        return AIModel.from_identifier(AIModel.DEFAULT_MODEL)

    def __str__(self) -> str:
        """String representation of the model.
        
        Returns:
            str: Model identifier
        """
        return self.id