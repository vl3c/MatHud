"""Tests for static.route_helpers module.

Covers update_all_provider_models and get_active_provider using MagicMock-based app/provider stubs.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from static.route_helpers import (
    get_active_provider,
    update_all_provider_models,
)


def _make_app() -> MagicMock:
    """Build a MagicMock that mimics MatHudFlask with two built-in providers."""
    return MagicMock()


# ---------------------------------------------------------------------------
# update_all_provider_models
# ---------------------------------------------------------------------------

class TestUpdateAllProviderModels(unittest.TestCase):
    """Tests for update_all_provider_models."""

    def test_sets_model_on_both_providers(self) -> None:
        """Both ai_api and responses_api receive set_model with the correct id."""
        app = _make_app()
        model_id = "gpt-6-sol"

        update_all_provider_models(app, model_id)

        app.ai_api.set_model.assert_called_once_with(model_id)
        app.responses_api.set_model.assert_called_once_with(model_id)


# ---------------------------------------------------------------------------
# get_active_provider
# ---------------------------------------------------------------------------

class TestGetActiveProvider(unittest.TestCase):
    """Tests for get_active_provider."""

    def test_returns_ai_api_when_model_id_is_none(self) -> None:
        """Default provider is app.ai_api when model_id is None."""
        app = _make_app()

        result = get_active_provider(app, None)

        self.assertIs(result, app.ai_api)
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

        result = get_active_provider(app, "gpt-6-sol")

        mock_update.assert_called_once_with(app, "gpt-6-sol")
        mock_get_provider.assert_called_once_with(app, "gpt-6-sol")
        self.assertIs(result, sentinel_provider)

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

        self.assertIs(result, expected)
