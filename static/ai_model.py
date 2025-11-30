"""
MatHud AI Model Configuration

AI model definitions and capability management for OpenAI models.
Handles model-specific features like vision support and provides factory methods.

Dependencies:
    - None (pure configuration module)
"""

from __future__ import annotations

from typing import Dict, Literal, TypedDict


class ModelConfig(TypedDict, total=False):
    """Configuration for an AI model."""

    has_vision: bool
    is_reasoning_model: bool


ModelConfigDict = Dict[str, ModelConfig]
ModelIdentifier = Literal[
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5-chat-latest",
    "gpt-5-nano",
    "gpt-3.5-turbo",
    "o3",
    "o4-mini",
]


class AIModel:
    """AI model configuration and capability management.
    
    Manages model-specific capabilities (like vision support) and provides
    factory methods for creating model instances from identifiers.
    """
    
    # Dictionary of model configurations
    MODEL_CONFIGS = {
        # Reasoning models (use Responses API)
        "gpt-5-chat-latest": {
            "has_vision": True,
            "is_reasoning_model": True,
        },
        "o3": {
            "has_vision": False,
            "is_reasoning_model": True,
        },
        "o4-mini": {
            "has_vision": True,
            "is_reasoning_model": True,
        },
        # Standard models (use Chat Completions API)
        "gpt-4.1": {
            "has_vision": True,
            "is_reasoning_model": False,
        },
        "gpt-4.1-mini": {
            "has_vision": True,
            "is_reasoning_model": False,
        },
        "gpt-4.1-nano": {
            "has_vision": True,
            "is_reasoning_model": False,
        },
        "gpt-4o": {
            "has_vision": True,
            "is_reasoning_model": False,
        },
        "gpt-4o-mini": {
            "has_vision": True,
            "is_reasoning_model": False,
        },
        "gpt-5-nano": {
            "has_vision": True,
            "is_reasoning_model": False,
        },
        "gpt-3.5-turbo": {
            "has_vision": False,
            "is_reasoning_model": False,
        },
    }

    DEFAULT_MODEL = "gpt-5-chat-latest"

    def __init__(self, identifier: str, has_vision: bool, is_reasoning_model: bool = False) -> None:
        """Initialize AIModel instance.
        
        Args:
            identifier: Model identifier string (e.g., 'gpt-4.1')
            has_vision: Boolean indicating vision capability support
            is_reasoning_model: Boolean indicating if model uses Responses API with reasoning
        """
        self.id: str = identifier
        self.has_vision: bool = has_vision
        self.is_reasoning_model: bool = is_reasoning_model
    
    @staticmethod
    def from_identifier(identifier: str) -> AIModel:
        """Create AIModel instance from identifier string.
        
        Args:
            identifier: Model identifier string
            
        Returns:
            AIModel: Configured model instance
        """
        config = AIModel.MODEL_CONFIGS.get(identifier, {"has_vision": False, "is_reasoning_model": False})
        return AIModel(
            identifier=identifier,
            has_vision=config.get("has_vision", False),
            is_reasoning_model=config.get("is_reasoning_model", False),
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