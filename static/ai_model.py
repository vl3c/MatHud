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
PROVIDER_OLLAMA = "ollama"


class ModelConfig(TypedDict, total=False):
    """Configuration for an AI model."""

    has_vision: bool
    is_reasoning_model: bool
    reasoning_effort: str
    provider: str
    display_name: str


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
    "claude-opus-4-5-20251101",
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
            "display_name": "GPT-5 Chat Latest",
        },
        "gpt-5.2-chat-latest": {
            "has_vision": True,
            "is_reasoning_model": True,
            "provider": PROVIDER_OPENAI,
            "display_name": "GPT-5.2 Chat Latest",
        },
        "gpt-5.2": {
            "has_vision": True,
            "is_reasoning_model": True,
            "reasoning_effort": "medium",
            "provider": PROVIDER_OPENAI,
            "display_name": "GPT-5.2 (Medium Reasoning)",
        },
        "o3": {
            "has_vision": True,
            "is_reasoning_model": True,
            "provider": PROVIDER_OPENAI,
            "display_name": "o3 (Reasoning)",
        },
        "o4-mini": {
            "has_vision": True,
            "is_reasoning_model": True,
            "provider": PROVIDER_OPENAI,
            "display_name": "o4-mini (Reasoning)",
        },
        # Standard models (use Chat Completions API)
        "gpt-4.1": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENAI,
            "display_name": "GPT-4.1",
        },
        "gpt-4.1-mini": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENAI,
            "display_name": "GPT-4.1 Mini",
        },
        "gpt-4.1-nano": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENAI,
            "display_name": "GPT-4.1 Nano",
        },
        "gpt-4o": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENAI,
            "display_name": "GPT-4o",
        },
        "gpt-4o-mini": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENAI,
            "display_name": "GPT-4o Mini",
        },
        "gpt-5-nano": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENAI,
            "display_name": "GPT-5 Nano",
        },
        "gpt-3.5-turbo": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENAI,
            "display_name": "GPT-3.5 Turbo",
        },
        # ===================
        # Anthropic Models
        # ===================
        "claude-opus-4-5-20251101": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_ANTHROPIC,
            "display_name": "Claude Opus 4.5",
        },
        "claude-sonnet-4-5-20250929": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_ANTHROPIC,
            "display_name": "Claude Sonnet 4.5",
        },
        "claude-haiku-4-5-20251001": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_ANTHROPIC,
            "display_name": "Claude Haiku 4.5",
        },
        # ===================
        # OpenRouter Models (Paid)
        # ===================
        "google/gemini-2.5-pro": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Gemini 2.5 Pro",
        },
        "google/gemini-3-pro-preview": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Gemini 3 Pro Preview",
        },
        "google/gemini-3-flash-preview": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Gemini 3 Flash Preview",
        },
        "deepseek/deepseek-v3.2": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "DeepSeek V3.2",
        },
        "x-ai/grok-code-fast-1": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Grok Code Fast",
        },
        "z-ai/glm-4.7": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "GLM 4.7",
        },
        "minimax/minimax-m2.1": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "MiniMax M2.1",
        },
        # ===================
        # OpenRouter Models (Free)
        # ===================
        "meta-llama/llama-3.3-70b-instruct:free": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Llama 3.3 70B",
        },
        "google/gemma-3-27b-it:free": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Gemma 3 27B",
        },
        "openai/gpt-oss-20b:free": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "GPT-OSS 20B",
        },
        "openai/gpt-oss-120b:free": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "GPT-OSS 120B",
        },
        "qwen/qwen3-next-80b-a3b-instruct:free": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Qwen3 80B",
        },
        "z-ai/glm-4.5-air:free": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "GLM 4.5 Air",
        },
        "nvidia/nemotron-3-nano-30b-a3b:free": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Nemotron 3 30B",
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

    @classmethod
    def register_local_models(
        cls,
        provider: str,
        models_info: list[dict[str, object]],
    ) -> list[str]:
        """Register dynamically discovered local models.

        Args:
            provider: The provider name (e.g., 'ollama')
            models_info: List of model info dicts with 'name' key

        Returns:
            List of registered model identifiers
        """
        registered = []
        for model_info in models_info:
            model_name = str(model_info.get("name", ""))
            if not model_name:
                continue

            # Create display name from model name
            display_name = _format_display_name(model_name)

            # Register the model config
            cls.MODEL_CONFIGS[model_name] = {
                "has_vision": False,  # Local models generally don't support vision
                "is_reasoning_model": False,
                "provider": provider,
                "display_name": display_name,
            }
            registered.append(model_name)

        return registered

    @classmethod
    def refresh_local_models(cls, provider: str) -> list[str]:
        """Refresh and register models from a local provider.

        Args:
            provider: The provider name (e.g., 'ollama')

        Returns:
            List of registered model identifiers
        """
        # Import here to avoid circular imports
        from static.providers.local import LocalProviderRegistry

        provider_class = LocalProviderRegistry.get_provider_class(provider)
        if provider_class is None:
            return []

        if not LocalProviderRegistry.is_provider_available(provider):
            return []

        # Get tool-capable models from the provider
        try:
            # Use class method if available
            if hasattr(provider_class, "get_tool_capable_models"):
                models_info = provider_class.get_tool_capable_models()
            else:
                # Create instance and discover
                instance = object.__new__(provider_class)
                models_info = instance.discover_models_with_tool_support()
        except Exception:
            return []

        return cls.register_local_models(provider, models_info)


def _format_display_name(model_name: str) -> str:
    """Format a model name into a display-friendly name.

    Examples:
        'llama3.1:8b' -> 'Llama 3.1 8B'
        'qwen2.5-coder:7b' -> 'Qwen 2.5 Coder 7B'
        'mistral:latest' -> 'Mistral Latest'

    Args:
        model_name: The raw model name

    Returns:
        A human-friendly display name
    """
    # Split on colon to separate base name and tag
    parts = model_name.split(":")
    base = parts[0]
    tag = parts[1] if len(parts) > 1 else ""

    # Convert base name
    # Replace hyphens and underscores with spaces
    base = base.replace("-", " ").replace("_", " ")

    # Add spaces between numbers and letters
    formatted = ""
    for i, char in enumerate(base):
        if i > 0:
            prev_char = base[i - 1]
            # Add space between letter and digit
            if (prev_char.isalpha() and char.isdigit()) or (
                prev_char.isdigit() and char.isalpha()
            ):
                # But not for decimal points in version numbers
                if not (prev_char.isdigit() and char == "."):
                    formatted += " "
        formatted += char

    # Title case
    words = formatted.split()
    formatted_words = []
    for word in words:
        # Handle version numbers (keep as-is)
        if word[0].isdigit():
            formatted_words.append(word.upper() if word.isalpha() else word)
        else:
            formatted_words.append(word.capitalize())

    display = " ".join(formatted_words)

    # Append tag if present and not 'latest'
    if tag and tag.lower() != "latest":
        display += f" {tag.upper()}"

    return display