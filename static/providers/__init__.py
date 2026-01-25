"""
MatHud Provider Registry

Discovers and registers AI providers dynamically.
Each provider is a self-contained module that can be added/removed independently.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Type

from dotenv import load_dotenv

if TYPE_CHECKING:
    from static.openai_api_base import OpenAIAPIBase

_logger = logging.getLogger("mathud")

# Provider name constants
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENROUTER = "openrouter"


class ProviderRegistry:
    """Registry for AI API providers.

    Discovers available providers based on API keys in environment.
    Providers self-register when their modules are imported.
    """

    _providers: Dict[str, Type["OpenAIAPIBase"]] = {}
    _api_key_names: Dict[str, str] = {
        PROVIDER_OPENAI: "OPENAI_API_KEY",
        PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
        PROVIDER_OPENROUTER: "OPENROUTER_API_KEY",
    }

    @classmethod
    def register(cls, provider_name: str, provider_class: Type["OpenAIAPIBase"]) -> None:
        """Register a provider class.

        Args:
            provider_name: Unique provider identifier (e.g., 'anthropic')
            provider_class: The API class for this provider
        """
        cls._providers[provider_name] = provider_class
        _logger.info(f"Registered provider: {provider_name}")

    @classmethod
    def get_provider_class(cls, provider_name: str) -> Optional[Type["OpenAIAPIBase"]]:
        """Get the provider class by name.

        Args:
            provider_name: The provider identifier

        Returns:
            The provider class, or None if not registered
        """
        return cls._providers.get(provider_name)

    @classmethod
    def is_provider_available(cls, provider_name: str) -> bool:
        """Check if a provider is available (has API key configured).

        Args:
            provider_name: The provider identifier

        Returns:
            True if the provider has an API key set
        """
        load_dotenv()
        key_name = cls._api_key_names.get(provider_name)
        if not key_name:
            return False
        return bool(os.getenv(key_name))

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of providers with API keys configured.

        Returns:
            List of available provider names
        """
        load_dotenv()
        available = []
        for provider_name, key_name in cls._api_key_names.items():
            if os.getenv(key_name):
                available.append(provider_name)
        return available

    @classmethod
    def get_registered_providers(cls) -> List[str]:
        """Get list of registered provider names.

        Returns:
            List of registered provider names
        """
        return list(cls._providers.keys())


def get_provider_for_model(model_id: str) -> Optional[str]:
    """Get the provider name for a given model ID.

    Args:
        model_id: The model identifier

    Returns:
        Provider name, or None if unknown
    """
    from static.ai_model import AIModel
    model = AIModel.from_identifier(model_id)
    return getattr(model, 'provider', PROVIDER_OPENAI)


def create_provider_instance(
    provider_name: str,
    **kwargs: object,
) -> Optional["OpenAIAPIBase"]:
    """Create an instance of the specified provider.

    Args:
        provider_name: The provider identifier
        **kwargs: Arguments to pass to the provider constructor

    Returns:
        Provider instance, or None if provider not available
    """
    provider_class = ProviderRegistry.get_provider_class(provider_name)
    if provider_class is None:
        _logger.warning(f"Provider not registered: {provider_name}")
        return None

    if not ProviderRegistry.is_provider_available(provider_name):
        _logger.warning(f"Provider not available (no API key): {provider_name}")
        return None

    try:
        return provider_class(**kwargs)
    except Exception as e:
        _logger.error(f"Failed to create provider {provider_name}: {e}")
        return None


def discover_providers() -> None:
    """Import provider modules to trigger registration.

    Called during application startup to make providers available.
    """
    # Import provider modules - they self-register on import
    try:
        from static.providers import anthropic_api  # noqa: F401
        _logger.debug("Loaded anthropic_api provider module")
    except ImportError as e:
        _logger.debug(f"Could not load anthropic_api: {e}")

    try:
        from static.providers import openrouter_api  # noqa: F401
        _logger.debug("Loaded openrouter_api provider module")
    except ImportError as e:
        _logger.debug(f"Could not load openrouter_api: {e}")
