"""Route helper functions for MatHud Flask routes.

Extracts duplicated logic from routes.py into reusable helpers for
provider model synchronization and provider lookup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, cast

if TYPE_CHECKING:
    from static.app_manager import MatHudFlask
    from static.openai_api_base import OpenAIAPIBase


def update_all_provider_models(app: MatHudFlask, model_id: str) -> None:
    """Synchronize the model selection across the built-in OpenAI providers.

    Both ``app.ai_api`` and ``app.responses_api`` are updated so that
    whichever one handles the next request uses the correct model.

    Args:
        app: The Flask application instance.
        model_id: The model identifier string (e.g. ``"gpt-6-sol"``).
    """
    app.ai_api.set_model(model_id)
    app.responses_api.set_model(model_id)


def get_active_provider(app: MatHudFlask, model_id: Optional[str]) -> OpenAIAPIBase:
    """Return the correct provider for a given model, updating built-in APIs.

    When *model_id* is provided the built-in OpenAI providers are
    synchronized via :func:`update_all_provider_models` and the
    appropriate provider instance is resolved (creating one lazily if
    needed).  When *model_id* is ``None`` the default ``app.ai_api``
    provider is returned.

    Args:
        app: The Flask application instance.
        model_id: The model identifier, or ``None`` for the default.

    Returns:
        The provider instance that should handle the current request.
    """
    from static.routes import get_provider_for_model

    if model_id:
        update_all_provider_models(app, model_id)
        return cast("OpenAIAPIBase", get_provider_for_model(app, model_id))

    return cast("OpenAIAPIBase", app.ai_api)


def conversation_apis(app: MatHudFlask, model_id: Optional[str]) -> List[OpenAIAPIBase]:
    """Return every provider instance that keeps conversation history.

    That is the two built-in OpenAI APIs, the providers created so far and, when
    *model_id* is given, the provider for that model (created if needed), so a
    restored conversation reaches the model the user continues with. A model
    whose provider cannot be created (no API key, for instance) is skipped.
    """
    apis: List[OpenAIAPIBase] = [cast("OpenAIAPIBase", app.ai_api), cast("OpenAIAPIBase", app.responses_api)]
    apis.extend(cast("OpenAIAPIBase", provider) for provider in app.providers.values())
    if model_id:
        try:
            apis.append(get_active_provider(app, model_id))
        except Exception:
            pass
    unique: List[OpenAIAPIBase] = []
    for api in apis:
        if all(api is not seen for seen in unique):
            unique.append(api)
    return unique
