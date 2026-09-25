"""
Tests for the provider registry.

Covers registration, availability checks, provider instantiation and
model-to-provider resolution in static/providers/__init__.py using
fake providers only (no .env files, no real API keys, no network).
"""

import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from static.providers import (
    ProviderRegistry,
    create_provider_instance,
    get_provider_for_model,
    is_local_provider,
)
from static.providers.local import LocalProviderRegistry


class FakeProvider:
    """Fake provider that records every constructed instance."""

    _instances: list["FakeProvider"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        FakeProvider._instances.append(self)


class RaisingProvider:
    """Fake provider whose constructor always fails."""

    def __init__(self, **kwargs: Any) -> None:
        raise RuntimeError("provider constructor failed")


@pytest.fixture(autouse=True)
def mock_load_env_files(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch load_env_files where the registry looks it up so no .env file is read."""
    load_env_files = MagicMock(name="load_env_files")
    monkeypatch.setattr("static.providers.load_env_files", load_env_files)
    return load_env_files


@pytest.fixture(autouse=True)
def clean_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with a clean env and fresh shared registry dicts."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(ProviderRegistry, "_providers", {})
    monkeypatch.setattr(LocalProviderRegistry, "_providers", {})
    monkeypatch.setattr(FakeProvider, "_instances", [])


class TestRegister:
    """Registering providers and looking them up."""

    def test_register_then_get_provider_class(self) -> None:
        """Registered provider class is returned by its name."""
        ProviderRegistry.register("x", FakeProvider)
        assert ProviderRegistry.get_provider_class("x") is FakeProvider

    def test_get_provider_class_missing(self) -> None:
        """Unregistered provider name returns None."""
        assert ProviderRegistry.get_provider_class("missing") is None

    def test_get_registered_providers(self) -> None:
        """Registered names are returned in registration order."""
        ProviderRegistry.register("a", FakeProvider)
        ProviderRegistry.register("b", RaisingProvider)
        assert ProviderRegistry.get_registered_providers() == ["a", "b"]


class TestIsProviderAvailable:
    """Availability checks for API and local providers."""

    def test_anthropic_available_when_key_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_load_env_files: MagicMock,
    ) -> None:
        """Configured ANTHROPIC_API_KEY makes anthropic available."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        assert ProviderRegistry.is_provider_available("anthropic") is True
        mock_load_env_files.assert_called_once()

    def test_anthropic_unavailable_without_key(self, mock_load_env_files: MagicMock) -> None:
        """Unset ANTHROPIC_API_KEY makes anthropic unavailable."""
        assert ProviderRegistry.is_provider_available("anthropic") is False
        mock_load_env_files.assert_called_once()

    def test_anthropic_unavailable_with_empty_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty ANTHROPIC_API_KEY is treated as unset."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        assert ProviderRegistry.is_provider_available("anthropic") is False

    def test_unknown_provider_not_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown provider name is never available, regardless of keys."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        assert ProviderRegistry.is_provider_available("unknown") is False

    def test_local_agent_available_when_local_says_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_load_env_files: MagicMock,
    ) -> None:
        """LocalAgent availability delegates to the local registry."""
        local_available = MagicMock(name="local_is_available", return_value=True)
        monkeypatch.setattr(LocalProviderRegistry, "is_provider_available", local_available)
        assert ProviderRegistry.is_provider_available("local_agent") is True
        local_available.assert_called_once_with("local_agent")
        mock_load_env_files.assert_not_called()

    def test_local_agent_available_when_local_says_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_load_env_files: MagicMock,
    ) -> None:
        """LocalAgent returns the local registry's False as-is."""
        local_available = MagicMock(name="local_is_available", return_value=False)
        monkeypatch.setattr(LocalProviderRegistry, "is_provider_available", local_available)
        assert ProviderRegistry.is_provider_available("local_agent") is False
        local_available.assert_called_once_with("local_agent")
        mock_load_env_files.assert_not_called()


class TestGetAvailableProviders:
    """Provider lists reported by get_available_providers."""

    def test_api_providers_with_local_down(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_load_env_files: MagicMock,
    ) -> None:
        """Only keyed API providers are returned when local_agent is down."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        monkeypatch.setattr(
            LocalProviderRegistry,
            "is_provider_available",
            MagicMock(return_value=False),
        )
        assert ProviderRegistry.get_available_providers() == ["openai", "openrouter"]
        mock_load_env_files.assert_called_once()

    def test_local_provider_with_no_keys(self, monkeypatch: pytest.MonkeyPatch, mock_load_env_files: MagicMock) -> None:
        """Only local_agent is returned when no API keys are set."""
        monkeypatch.setattr(
            LocalProviderRegistry,
            "is_provider_available",
            MagicMock(return_value=True),
        )
        assert ProviderRegistry.get_available_providers() == ["local_agent"]
        mock_load_env_files.assert_called_once()

    def test_all_providers(self, monkeypatch: pytest.MonkeyPatch, mock_load_env_files: MagicMock) -> None:
        """All providers appear in openai, anthropic, openrouter, local_agent order."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        monkeypatch.setattr(
            LocalProviderRegistry,
            "is_provider_available",
            MagicMock(return_value=True),
        )
        assert ProviderRegistry.get_available_providers() == [
            "openai",
            "anthropic",
            "openrouter",
            "local_agent",
        ]
        mock_load_env_files.assert_called_once()


class TestIsLocalProvider:
    """Membership in the local provider set."""

    def test_local_agent_is_local(self) -> None:
        """LocalAgent is a local provider."""
        assert is_local_provider("local_agent") is True

    def test_openai_is_not_local(self) -> None:
        """API providers are not local."""
        assert is_local_provider("openai") is False


class TestCreateProviderInstanceApi:
    """Instance creation for API-based providers."""

    def test_anthropic_instance_with_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Registered and keyed provider is instantiated with the kwargs."""
        ProviderRegistry.register("anthropic", FakeProvider)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        provider = create_provider_instance("anthropic", a=1)
        assert isinstance(provider, FakeProvider)
        assert provider.kwargs == {"a": 1}

    def test_not_registered(self) -> None:
        """An unregistered provider returns None."""
        assert create_provider_instance("anthropic") is None
        assert FakeProvider._instances == []

    def test_unkeyed_not_constructed(self) -> None:
        """Registered provider without a key is never constructed."""
        ProviderRegistry.register("anthropic", FakeProvider)
        assert create_provider_instance("anthropic") is None
        assert FakeProvider._instances == []

    def test_raising_class_suppressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Failing constructor returns None without raising."""
        ProviderRegistry.register("anthropic", RaisingProvider)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        assert create_provider_instance("anthropic") is None


class TestCreateProviderInstanceLocal:
    """Instance creation for local providers."""

    def test_local_agent_instance_with_kwargs(self, monkeypatch: pytest.MonkeyPatch, mock_load_env_files: MagicMock) -> None:
        """Available local provider is instantiated with the kwargs."""
        monkeypatch.setattr(
            LocalProviderRegistry,
            "get_provider_class",
            MagicMock(return_value=FakeProvider),
        )
        monkeypatch.setattr(
            LocalProviderRegistry,
            "is_provider_available",
            MagicMock(return_value=True),
        )
        provider = create_provider_instance("local_agent", b=2)
        assert isinstance(provider, FakeProvider)
        assert provider.kwargs == {"b": 2}
        mock_load_env_files.assert_not_called()

    def test_local_agent_class_missing(self, monkeypatch: pytest.MonkeyPatch, mock_load_env_files: MagicMock) -> None:
        """Missing local provider class returns None without an availability check."""
        monkeypatch.setattr(
            LocalProviderRegistry,
            "get_provider_class",
            MagicMock(return_value=None),
        )
        local_available = MagicMock(name="local_is_available")
        monkeypatch.setattr(LocalProviderRegistry, "is_provider_available", local_available)
        assert create_provider_instance("local_agent") is None
        local_available.assert_not_called()
        mock_load_env_files.assert_not_called()

    def test_local_agent_unavailable_not_constructed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unavailable local provider is never constructed."""
        monkeypatch.setattr(
            LocalProviderRegistry,
            "get_provider_class",
            MagicMock(return_value=FakeProvider),
        )
        monkeypatch.setattr(
            LocalProviderRegistry,
            "is_provider_available",
            MagicMock(return_value=False),
        )
        assert create_provider_instance("local_agent") is None
        assert FakeProvider._instances == []

    def test_local_agent_raising_class_suppressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Failing local constructor returns None without raising."""
        monkeypatch.setattr(
            LocalProviderRegistry,
            "get_provider_class",
            MagicMock(return_value=RaisingProvider),
        )
        monkeypatch.setattr(
            LocalProviderRegistry,
            "is_provider_available",
            MagicMock(return_value=True),
        )
        assert create_provider_instance("local_agent") is None


class TestGetProviderForModel:
    """Model-to-provider resolution."""

    def test_model_with_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A model with a provider attribute maps to it."""
        monkeypatch.setattr(
            "static.ai_model.AIModel.from_identifier",
            MagicMock(return_value=types.SimpleNamespace(provider="anthropic")),
        )
        assert get_provider_for_model("claude-3-5") == "anthropic"

    def test_model_without_provider_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A model without a provider attribute falls back to openai."""
        monkeypatch.setattr(
            "static.ai_model.AIModel.from_identifier",
            MagicMock(return_value=object()),
        )
        assert get_provider_for_model("gpt-6-sol") == "openai"
