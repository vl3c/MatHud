"""
MatHud AI Model Configuration

AI model definitions and capability management for OpenAI models.
Handles model-specific features like vision support and provides factory methods.

Dependencies:
    - None (pure configuration module)
"""


class AIModel:
    """AI model configuration and capability management.
    
    Manages model-specific capabilities (like vision support) and provides
    factory methods for creating model instances from identifiers.
    """
    
    # Dictionary of model configurations
    MODEL_CONFIGS = {
        "gpt-4.1": {
            "has_vision": True,
            # We can add more capabilities here later, like:
            # "max_tokens": 32000,
            # "supports_functions": True,
            # etc.
        },
        "gpt-4.1-mini": {
            "has_vision": True,
        },
        "gpt-4.1-nano": {
            "has_vision": True,
        },
        "gpt-4o": {
            "has_vision": True,
        },
        "gpt-4o-mini": {
            "has_vision": True,
        },
        "gpt-5-chat-latest": {
            "has_vision": True,
        },
        "gpt-5-nano": {
            "has_vision": True,
        },
        "gpt-3.5-turbo": {
            "has_vision": False,
        }
    }

    DEFAULT_MODEL = "gpt-5-chat-latest"

    def __init__(self, identifier, has_vision):
        """Initialize AIModel instance.
        
        Args:
            identifier: Model identifier string (e.g., 'gpt-4.1')
            has_vision: Boolean indicating vision capability support
        """
        self.id = identifier
        self.has_vision = has_vision
    
    @staticmethod
    def from_identifier(identifier):
        """Create AIModel instance from identifier string.
        
        Args:
            identifier: Model identifier string
            
        Returns:
            AIModel: Configured model instance
        """
        config = AIModel.MODEL_CONFIGS.get(identifier, {"has_vision": False})
        return AIModel(identifier=identifier, has_vision=config["has_vision"])

    @staticmethod
    def get_default_model():
        """Get the default AI model instance.
        
        Returns:
            AIModel: Default model configuration
        """
        return AIModel.from_identifier(AIModel.DEFAULT_MODEL)

    def __str__(self):
        """String representation of the model.
        
        Returns:
            str: Model identifier
        """
        return self.id 