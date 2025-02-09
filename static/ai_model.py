class AIModel:
    # Dictionary of model configurations
    MODEL_CONFIGS = {
        "gpt-4o": {
            "has_vision": True,
            # We can add more capabilities here later, like:
            # "max_tokens": 32000,
            # "supports_functions": True,
            # etc.
        },
        "gpt-3.5-turbo": {
            "has_vision": False,
        }
    }

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, identifier, has_vision):
        self.id = identifier
        self.has_vision = has_vision
    
    @staticmethod
    def from_identifier(identifier):
        config = AIModel.MODEL_CONFIGS.get(identifier, {"has_vision": False})
        return AIModel(identifier=identifier, has_vision=config["has_vision"])

    @staticmethod
    def get_default_model():
        return AIModel.from_identifier(AIModel.DEFAULT_MODEL)

    def __str__(self):
        return self.id 