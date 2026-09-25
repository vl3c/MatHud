"""
MatHud AI Model Configuration

AI model definitions and capability management for multiple AI providers.
Handles model-specific features like vision support and provides factory methods.

Dependencies:
    - None (pure configuration module)
"""

from __future__ import annotations

import re

from typing import Dict, Iterable, Literal, Optional, TypedDict

# Provider constants
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_LOCAL_AGENT = "local_agent"


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
    "gpt-6-sol",
    "gpt-6-astra",
    "gpt-6-luna",
    "gpt-5.6-sol",
    # Anthropic models
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5",
    "claude-haiku-4-5",
    # OpenRouter models (paid)
    "anthropic/claude-opus-5.5",
    "anthropic/claude-sonnet-5",
    "google/gemini-3.8-flash",
    "x-ai/grok-4.7",
    "xiaomi/mimo-v2.6-pro",
    "z-ai/glm-5.3-flash",
    "deepseek/deepseek-v4.1-flash",
    "qwen/qwen3.8-max-0902",
    # OpenRouter models (free)
    "qwen/qwen3.8-27b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "thinkingmachines/inkling:free",
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
        # All OpenAI models are reasoning models served through the Responses API.
        # GPT-6 Astra rejects reasoning effort "none"; keep it at "low" or above.
        "gpt-6-sol": {
            "has_vision": True,
            "is_reasoning_model": True,
            "reasoning_effort": "medium",
            "provider": PROVIDER_OPENAI,
            "display_name": "GPT-6 Sol",
        },
        "gpt-6-astra": {
            "has_vision": True,
            "is_reasoning_model": True,
            "reasoning_effort": "medium",
            "provider": PROVIDER_OPENAI,
            "display_name": "GPT-6 Astra",
        },
        "gpt-6-luna": {
            "has_vision": True,
            "is_reasoning_model": True,
            "reasoning_effort": "low",
            "provider": PROVIDER_OPENAI,
            "display_name": "GPT-6 Luna",
        },
        "gpt-5.6-sol": {
            "has_vision": True,
            "is_reasoning_model": True,
            "reasoning_effort": "medium",
            "provider": PROVIDER_OPENAI,
            "display_name": "GPT-5.6 Sol",
        },
        # ===================
        # Anthropic Models
        # ===================
        # Adaptive-thinking Claude models (Fable 5.1, Opus 5.5, Sonnet 5) reject the
        # temperature sampling parameter, so they are flagged is_reasoning_model=True and
        # AnthropicAPI omits temperature for them. Their reasoning_effort is sent as
        # output_config.effort. Haiku 4.5 takes temperature and has no effort setting.
        "claude-fable-5-1": {
            "has_vision": True,
            "is_reasoning_model": True,
            "reasoning_effort": "low",
            "provider": PROVIDER_ANTHROPIC,
            "display_name": "Claude Fable 5.1",
        },
        "claude-opus-5-5": {
            "has_vision": True,
            "is_reasoning_model": True,
            "reasoning_effort": "medium",
            "provider": PROVIDER_ANTHROPIC,
            "display_name": "Claude Opus 5.5",
        },
        "claude-sonnet-5": {
            "has_vision": True,
            "is_reasoning_model": True,
            "reasoning_effort": "medium",
            "provider": PROVIDER_ANTHROPIC,
            "display_name": "Claude Sonnet 5",
        },
        "claude-haiku-4-5": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_ANTHROPIC,
            "display_name": "Claude Haiku 4.5",
        },
        # ===================
        # OpenRouter Models (Paid)
        # ===================
        "anthropic/claude-opus-5.5": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Claude Opus 5.5 (OpenRouter)",
        },
        "anthropic/claude-sonnet-5": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Claude Sonnet 5 (OpenRouter)",
        },
        "google/gemini-3.8-flash": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Gemini 3.8 Flash",
        },
        "x-ai/grok-4.7": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Grok 4.7",
        },
        "xiaomi/mimo-v2.6-pro": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "MiMo V2.6 Pro",
        },
        "z-ai/glm-5.3-flash": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "GLM 5.3 Flash",
        },
        "deepseek/deepseek-v4.1-flash": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "DeepSeek V4.1 Flash",
        },
        "qwen/qwen3.8-max-0902": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Qwen3.8 Max",
        },
        # ===================
        # OpenRouter Models (Free)
        # ===================
        # Free-tier limits: 20 requests/minute; 50 requests/day with under $10 of
        # lifetime credits, 1000 requests/day otherwise.
        "qwen/qwen3.8-27b:free": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Qwen3.8 27B",
        },
        "nvidia/nemotron-3-ultra-550b-a55b:free": {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Nemotron 3 Ultra",
        },
        "google/gemma-4-31b-it:free": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Gemma 4 31B",
        },
        "google/gemma-4-26b-a4b-it:free": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Gemma 4 26B A4B",
        },
        "thinkingmachines/inkling:free": {
            "has_vision": True,
            "is_reasoning_model": False,
            "provider": PROVIDER_OPENROUTER,
            "display_name": "Inkling",
        },
    }

    DEFAULT_MODEL = "gpt-6-sol"

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
            identifier: Model identifier string (e.g., 'gpt-6-sol')
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
            provider: The provider name (e.g., 'local_agent')
            models_info: List of model info dicts with a 'name' key and an
                optional 'display_name' override

        Returns:
            List of registered model identifiers
        """
        registered = []
        for model_info in models_info:
            name = model_info.get("name")
            if name is None:
                continue
            model_name = str(name)
            if not model_name.strip():
                continue

            # Providers may supply their own label when the identifier is
            # not a conventional model name (a file path, for instance).
            display_name = str(model_info.get("display_name") or "") or _format_display_name(model_name)

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
    def unregister_local_models(cls, provider: str, keep: Optional[Iterable[str]] = None) -> list[str]:
        """Drop models previously registered for a local provider.

        A local server hosts one model at a time, so entries discovered earlier
        go stale as soon as the server is restarted with a different model.

        Args:
            provider: The provider name whose models should be removed
            keep: Identifiers to leave registered, typically the ones a refresh
                just returned. When None, every model of the provider is removed.

        Returns:
            List of removed model identifiers
        """
        kept = set(keep or ())
        stale = [
            model_id
            for model_id, config in cls.MODEL_CONFIGS.items()
            if config.get("provider") == provider and model_id not in kept
        ]
        for model_id in stale:
            del cls.MODEL_CONFIGS[model_id]
        return stale

    @classmethod
    def refresh_local_models(cls, provider: str) -> list[str]:
        """Refresh and register models from a local provider.

        Args:
            provider: The provider name (e.g., 'local_agent')

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


_DISPLAY_ACRONYMS = {"gpt", "oss", "vl"}


def _format_display_name(model_name: str) -> str:
    """Format a model name into a display-friendly name.

    Examples:
        'llama3.1:8b' -> 'Llama 3.1 8B'
        'qwen2.5-coder:7b' -> 'Qwen 2.5 Coder 7B'
        'mistral:latest' -> 'Mistral'
        'gpt-oss:20b' -> 'GPT OSS 20B'
        'deepseek-r1:14b' -> 'Deepseek R1 14B'
        '1.5b' -> '1.5B'

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

    formatted = []
    for word in base.split():
        # A version-like word holds a digit and every maximal run of letters is
        # exactly one letter; it is kept intact and upper-cased.
        letter_runs = re.findall(r"[a-zA-Z]+", word)
        if any(c.isdigit() for c in word) and all(len(run) == 1 for run in letter_runs):
            formatted.append(word.upper())
            continue

        # Otherwise split at letter/digit boundaries (a dot stays with digits)
        # and format each resulting piece.
        pieces = []
        piece = ""
        for i, char in enumerate(word):
            if i > 0:
                prev_char = word[i - 1]
                if (prev_char.isalpha() and char.isdigit()) or (prev_char.isdigit() and char.isalpha()):
                    pieces.append(piece)
                    piece = ""
            piece += char
        pieces.append(piece)

        for piece in pieces:
            if piece[0].isdigit():
                formatted.append(piece)
            elif piece.lower() in _DISPLAY_ACRONYMS:
                formatted.append(piece.upper())
            else:
                formatted.append(piece.capitalize())

    display = " ".join(formatted)

    # Append tag if present and not 'latest'
    if tag and tag.lower() != "latest":
        display += f" {tag.upper()}"

    return display
