"""Route helper functions for MatHud Flask routes.

Extracts duplicated logic from routes.py into reusable helpers for
provider model synchronization and tool lifecycle management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

if TYPE_CHECKING:
    from static.app_manager import MatHudFlask
    from static.openai_api_base import OpenAIAPIBase


def reset_tools_for_all_providers(
    app: MatHudFlask,
    finish_reason: Optional[str],
    *,
    active_provider: Optional[OpenAIAPIBase] = None,
) -> None:
    """Reset injected tools on all provider instances when a conversation turn ends.

    Called when the AI finish reason is anything other than ``"tool_calls"``
    (i.e. the model is done calling tools) or on error.  Resets the built-in
    OpenAI APIs (``app.ai_api`` and ``app.responses_api``) plus the
    *active_provider* if it differs from the built-in ones.

    Args:
        app: The Flask application instance carrying provider references.
        finish_reason: The finish reason string from the AI response.
            When equal to ``"tool_calls"`` this function is a no-op.
        active_provider: The provider that handled the current request.
            If ``None`` only the two built-in OpenAI providers are checked.
    """
    if finish_reason == "tool_calls":
        return

    if app.ai_api.has_injected_tools():
        app.ai_api.reset_tools()
    if app.responses_api.has_injected_tools():
        app.responses_api.reset_tools()

    if (
        active_provider is not None
        and active_provider not in (app.ai_api, app.responses_api)
        and active_provider.has_injected_tools()
    ):
        active_provider.reset_tools()


def update_all_provider_models(app: MatHudFlask, model_id: str) -> None:
    """Synchronize the model selection across the built-in OpenAI providers.

    Both ``app.ai_api`` and ``app.responses_api`` are updated so that
    whichever one handles the next request uses the correct model.

    Args:
        app: The Flask application instance.
        model_id: The model identifier string (e.g. ``"gpt-4.1"``).
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
