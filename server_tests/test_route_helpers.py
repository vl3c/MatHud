"""Tests for static.route_helpers module.

Covers reset_tools_for_all_providers, update_all_provider_models,
and get_active_provider using MagicMock-based app/provider stubs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from static.route_helpers import (
    get_active_provider,
    reset_tools_for_all_providers,
    update_all_provider_models,
)


def _make_app(
    ai_injected: bool = False,
    responses_injected: bool = False,
) -> MagicMock:
    """Build a MagicMock that mimics MatHudFlask with two built-in providers."""
    app = MagicMock()

    app.ai_api.has_injected_tools.return_value = ai_injected
    app.responses_api.has_injected_tools.return_value = responses_injected

    return app


# ---------------------------------------------------------------------------
# reset_tools_for_all_providers
# ---------------------------------------------------------------------------

class TestResetToolsForAllProviders:
    """Tests for reset_tools_for_all_providers."""

    def test_noop_when_finish_reason_is_tool_calls(self) -> None:
        """No reset happens when the model still wants to call tools."""
        app = _make_app(ai_injected=True, responses_injected=True)

        reset_tools_for_all_providers(app, "tool_calls")

        app.ai_api.has_injected_tools.assert_not_called()
        app.responses_api.has_injected_tools.assert_not_called()
        app.ai_api.reset_tools.assert_not_called()
        app.responses_api.reset_tools.assert_not_called()

    def test_resets_ai_api_when_injected_and_stop(self) -> None:
        """ai_api is reset when it has injected tools and finish_reason is 'stop'."""
        app = _make_app(ai_injected=True, responses_injected=False)

        reset_tools_for_all_providers(app, "stop")

        app.ai_api.reset_tools.assert_called_once()
        app.responses_api.reset_tools.assert_not_called()

    def test_resets_responses_api_when_injected(self) -> None:
        """responses_api is reset when it has injected tools."""
        app = _make_app(ai_injected=False, responses_injected=True)

        reset_tools_for_all_providers(app, "stop")

        app.ai_api.reset_tools.assert_not_called()
        app.responses_api.reset_tools.assert_called_once()

    def test_no_reset_when_no_injected_tools(self) -> None:
        """Neither provider is reset when neither has injected tools."""
        app = _make_app(ai_injected=False, responses_injected=False)

        reset_tools_for_all_providers(app, "stop")

        app.ai_api.reset_tools.assert_not_called()
        app.responses_api.reset_tools.assert_not_called()

    def test_resets_active_provider_when_different_and_injected(self) -> None:
        """An active_provider that differs from the built-ins is reset."""
        app = _make_app(ai_injected=False, responses_injected=False)
        extra_provider = MagicMock()
        extra_provider.has_injected_tools.return_value = True

        reset_tools_for_all_providers(app, "stop", active_provider=extra_provider)

        extra_provider.reset_tools.assert_called_once()

    def test_does_not_reset_active_provider_when_same_as_ai_api(self) -> None:
        """active_provider is NOT reset when it is the same object as ai_api."""
        app = _make_app(ai_injected=True, responses_injected=False)
        # Pass the same object that lives on app.ai_api.
        same_provider = app.ai_api

        reset_tools_for_all_providers(app, "stop", active_provider=same_provider)

        # ai_api.reset_tools should be called once (from the built-in path),
        # but NOT a second time from the active_provider path.
        app.ai_api.reset_tools.assert_called_once()

    def test_handles_none_active_provider(self) -> None:
        """None active_provider is handled gracefully (no AttributeError)."""
        app = _make_app(ai_injected=False, responses_injected=False)

        # Should not raise
        reset_tools_for_all_providers(app, "stop", active_provider=None)

        app.ai_api.reset_tools.assert_not_called()
        app.responses_api.reset_tools.assert_not_called()

    def test_resets_all_three_when_all_injected(self) -> None:
        """All three providers reset when each has injected tools."""
        app = _make_app(ai_injected=True, responses_injected=True)
        extra_provider = MagicMock()
        extra_provider.has_injected_tools.return_value = True

        reset_tools_for_all_providers(app, "stop", active_provider=extra_provider)

        app.ai_api.reset_tools.assert_called_once()
        app.responses_api.reset_tools.assert_called_once()
        extra_provider.reset_tools.assert_called_once()

    def test_does_not_reset_active_provider_when_same_as_responses_api(self) -> None:
        """active_provider is NOT reset when it is the same object as responses_api."""
        app = _make_app(ai_injected=False, responses_injected=True)
        same_provider = app.responses_api

        reset_tools_for_all_providers(app, "stop", active_provider=same_provider)

        # Only the built-in path calls reset_tools, not the active_provider path.
        app.responses_api.reset_tools.assert_called_once()

    def test_does_not_reset_active_provider_without_injected_tools(self) -> None:
        """active_provider is NOT reset when it has no injected tools."""
        app = _make_app(ai_injected=False, responses_injected=False)
        extra_provider = MagicMock()
        extra_provider.has_injected_tools.return_value = False

        reset_tools_for_all_providers(app, "stop", active_provider=extra_provider)

        extra_provider.reset_tools.assert_not_called()


# ---------------------------------------------------------------------------
# update_all_provider_models
# ---------------------------------------------------------------------------

class TestUpdateAllProviderModels:
    """Tests for update_all_provider_models."""

    def test_sets_model_on_both_providers(self) -> None:
        """Both ai_api and responses_api receive set_model with the correct id."""
        app = _make_app()
        model_id = "gpt-4.1"

        update_all_provider_models(app, model_id)

        app.ai_api.set_model.assert_called_once_with(model_id)
        app.responses_api.set_model.assert_called_once_with(model_id)


# ---------------------------------------------------------------------------
# get_active_provider
# ---------------------------------------------------------------------------

class TestGetActiveProvider:
    """Tests for get_active_provider."""

    def test_returns_ai_api_when_model_id_is_none(self) -> None:
        """Default provider is app.ai_api when model_id is None."""
        app = _make_app()

        result = get_active_provider(app, None)

        assert result is app.ai_api
        app.ai_api.set_model.assert_not_called()
        app.responses_api.set_model.assert_not_called()

    @patch("static.route_helpers.update_all_provider_models")
    @patch("static.routes.get_provider_for_model")
    def test_calls_update_and_get_provider_when_model_id_given(
        self,
        mock_get_provider: MagicMock,
        mock_update: MagicMock,
    ) -> None:
        """update_all_provider_models and get_provider_for_model are called."""
        app = _make_app()
        sentinel_provider = MagicMock(name="resolved_provider")
        mock_get_provider.return_value = sentinel_provider

        result = get_active_provider(app, "gpt-4.1")

        mock_update.assert_called_once_with(app, "gpt-4.1")
        mock_get_provider.assert_called_once_with(app, "gpt-4.1")
        assert result is sentinel_provider

    @patch("static.route_helpers.update_all_provider_models")
    @patch("static.routes.get_provider_for_model")
    def test_returns_resolved_provider(
        self,
        mock_get_provider: MagicMock,
        mock_update: MagicMock,
    ) -> None:
        """The provider object returned by get_provider_for_model is passed through."""
        app = _make_app()
        expected = MagicMock(name="expected_provider")
        mock_get_provider.return_value = expected

        result = get_active_provider(app, "o4-mini")

        assert result is expected
