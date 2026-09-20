"""
Tests for AIModel.refresh_local_models.

Covers only refresh_local_models in static/ai_model.py using fake local provider
classes and patched LocalProviderRegistry lookups (no network, no .env, no API keys).
"""

from __future__ import annotations

import copy
from typing import Optional
from unittest.mock import MagicMock

import pytest

from static.ai_model import AIModel
from static.providers.local import LocalProviderRegistry


class ClassMethodProvider:
    """Fake local provider that exposes a classmethod discovery path."""

    calls: list[str] = []

    @classmethod
    def get_tool_capable_models(cls) -> list[dict[str, object]]:
        """Record the classmethod call and return two models."""
        ClassMethodProvider.calls.append("get_tool_capable_models")
        return [
            {"name": "refresh-class-model-one"},
            {"name": "refresh-class-model-two"},
        ]


class MixedNameProvider:
    """Fake local provider that returns names needing filtering."""

    @classmethod
    def get_tool_capable_models(cls) -> list[dict[str, object]]:
        """Return a good name alongside a None name and a blank name."""
        return [
            {"name": None},
            {"name": "   "},
            {"name": "refresh-good-model"},
        ]


class EmptyDiscoveryProvider:
    """Fake local provider whose discovery returns no models."""

    calls: list[str] = []

    @classmethod
    def get_tool_capable_models(cls) -> list[dict[str, object]]:
        """Record the classmethod call and return no models."""
        EmptyDiscoveryProvider.calls.append("get_tool_capable_models")
        return []


class RaisingClassMethodProvider:
    """Fake local provider whose classmethod discovery fails."""

    calls: list[str] = []

    @classmethod
    def get_tool_capable_models(cls) -> list[dict[str, object]]:
        """Record the classmethod call, then raise."""
        RaisingClassMethodProvider.calls.append("get_tool_capable_models")
        raise RuntimeError("classmethod discovery failed")


class BothPathsProvider:
    """Fake provider exposing both discovery paths so neither can run unnoticed."""

    classmethod_calls: list[str] = []
    discover_calls: list[str] = []

    @classmethod
    def get_tool_capable_models(cls) -> list[dict[str, object]]:
        """Record a classmethod discovery call."""
        BothPathsProvider.classmethod_calls.append("get_tool_capable_models")
        return [{"name": "should-not-run"}]

    def discover_models_with_tool_support(self) -> list[dict[str, object]]:
        """Record an instance discovery call."""
        BothPathsProvider.discover_calls.append("discover_models_with_tool_support")
        return [{"name": "should-not-run"}]


class InitProbeProvider:
    """Fake local provider whose instance discovery proves __init__ is bypassed."""

    init_calls: list[str] = []
    discover_calls: list[str] = []

    def __init__(self) -> None:
        """Record that __init__ ran."""
        InitProbeProvider.init_calls.append("init")

    def discover_models_with_tool_support(self) -> list[dict[str, object]]:
        """Record that discovery ran and return one model."""
        InitProbeProvider.discover_calls.append("discover_models_with_tool_support")
        return [{"name": "refresh-instance-model"}]


class RaisingInstanceProvider:
    """Fake local provider whose instance discovery fails."""

    discover_calls: list[str] = []

    def discover_models_with_tool_support(self) -> list[dict[str, object]]:
        """Record the instance discovery call, then raise."""
        RaisingInstanceProvider.discover_calls.append("discover_models_with_tool_support")
        raise RuntimeError("instance discovery failed")


@pytest.fixture(autouse=True)
def isolated_model_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test a private deep copy of MODEL_CONFIGS and restore the original."""
    monkeypatch.setattr(AIModel, "MODEL_CONFIGS", copy.deepcopy(AIModel.MODEL_CONFIGS))


@pytest.fixture(autouse=True)
def reset_fake_provider_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset all fake provider call counters before each test."""
    monkeypatch.setattr(ClassMethodProvider, "calls", [])
    monkeypatch.setattr(EmptyDiscoveryProvider, "calls", [])
    monkeypatch.setattr(RaisingClassMethodProvider, "calls", [])
    monkeypatch.setattr(BothPathsProvider, "classmethod_calls", [])
    monkeypatch.setattr(BothPathsProvider, "discover_calls", [])
    monkeypatch.setattr(InitProbeProvider, "init_calls", [])
    monkeypatch.setattr(InitProbeProvider, "discover_calls", [])
    monkeypatch.setattr(RaisingInstanceProvider, "discover_calls", [])


def _patch_registry(
    monkeypatch: pytest.MonkeyPatch,
    provider_class: Optional[object],
    available: bool,
) -> tuple[MagicMock, MagicMock]:
    """Patch LocalProviderRegistry lookups and return the mocks for assertions."""
    get_provider_class = MagicMock(name="get_provider_class", return_value=provider_class)
    is_provider_available = MagicMock(name="is_provider_available", return_value=available)
    monkeypatch.setattr(LocalProviderRegistry, "get_provider_class", get_provider_class)
    monkeypatch.setattr(LocalProviderRegistry, "is_provider_available", is_provider_available)
    return get_provider_class, is_provider_available


class TestRefreshEarlyExits:
    """Paths where refresh_local_models stops before registering anything."""

    def test_unregistered_provider_returns_empty_without_availability(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unregistered provider returns [] and never calls is_provider_available."""
        get_provider_class, is_provider_available = _patch_registry(monkeypatch, None, False)
        assert AIModel.refresh_local_models("local_agent") == []
        get_provider_class.assert_called_once_with("local_agent")
        is_provider_available.assert_not_called()

    def test_registered_unavailable_provider_returns_empty_without_discovery(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Registered unavailable provider returns [] without running either discovery path."""
        get_provider_class, is_provider_available = _patch_registry(monkeypatch, BothPathsProvider, False)
        assert AIModel.refresh_local_models("local_agent") == []
        get_provider_class.assert_called_once_with("local_agent")
        is_provider_available.assert_called_once_with("local_agent")
        assert BothPathsProvider.classmethod_calls == []
        assert BothPathsProvider.discover_calls == []

    def test_empty_discovery_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A discovery result with no models returns [] and registers nothing."""
        before = copy.deepcopy(AIModel.MODEL_CONFIGS)
        _patch_registry(monkeypatch, EmptyDiscoveryProvider, True)
        assert AIModel.refresh_local_models("local_agent") == []
        assert EmptyDiscoveryProvider.calls == ["get_tool_capable_models"]
        assert AIModel.MODEL_CONFIGS == before


class TestRefreshClassMethodDiscovery:
    """Discovery via the provider classmethod."""

    def test_classmethod_models_are_registered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The classmethod is used and each returned model is registered."""
        before = copy.deepcopy(AIModel.MODEL_CONFIGS)
        _patch_registry(monkeypatch, ClassMethodProvider, True)
        result = AIModel.refresh_local_models("local_agent")
        assert result == ["refresh-class-model-one", "refresh-class-model-two"]
        assert ClassMethodProvider.calls == ["get_tool_capable_models"]

        expected = dict(before)
        expected["refresh-class-model-one"] = {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": "local_agent",
            "display_name": "Refresh Class Model One",
        }
        expected["refresh-class-model-two"] = {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": "local_agent",
            "display_name": "Refresh Class Model Two",
        }
        assert AIModel.MODEL_CONFIGS == expected

    def test_classmethod_exception_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A classmethod discovery exception returns [] and registers nothing."""
        before = copy.deepcopy(AIModel.MODEL_CONFIGS)
        _patch_registry(monkeypatch, RaisingClassMethodProvider, True)
        assert AIModel.refresh_local_models("local_agent") == []
        assert RaisingClassMethodProvider.calls == ["get_tool_capable_models"]
        assert AIModel.MODEL_CONFIGS == before


class TestRefreshInstanceDiscovery:
    """Discovery via an instance built without __init__."""

    def test_instance_discovery_bypasses_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The instance path runs, discovers one model, and __init__ never ran."""
        before = copy.deepcopy(AIModel.MODEL_CONFIGS)
        _patch_registry(monkeypatch, InitProbeProvider, True)
        result = AIModel.refresh_local_models("local_agent")
        assert result == ["refresh-instance-model"]
        assert InitProbeProvider.discover_calls == ["discover_models_with_tool_support"]
        assert InitProbeProvider.init_calls == []

        expected = dict(before)
        expected["refresh-instance-model"] = {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": "local_agent",
            "display_name": "Refresh Instance Model",
        }
        assert AIModel.MODEL_CONFIGS == expected

    def test_instance_discovery_exception_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An instance discovery exception returns [] and registers nothing."""
        before = copy.deepcopy(AIModel.MODEL_CONFIGS)
        _patch_registry(monkeypatch, RaisingInstanceProvider, True)
        assert AIModel.refresh_local_models("local_agent") == []
        assert RaisingInstanceProvider.discover_calls == ["discover_models_with_tool_support"]
        assert AIModel.MODEL_CONFIGS == before


class TestRefreshRegistrationFiltering:
    """Filtering of discovered model names during registration."""

    def test_none_and_blank_names_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only a non-blank name is returned and registered."""
        before = copy.deepcopy(AIModel.MODEL_CONFIGS)
        _patch_registry(monkeypatch, MixedNameProvider, True)
        result = AIModel.refresh_local_models("local_agent")
        assert result == ["refresh-good-model"]
        assert "None" not in AIModel.MODEL_CONFIGS
        assert "   " not in AIModel.MODEL_CONFIGS

        expected = dict(before)
        expected["refresh-good-model"] = {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": "local_agent",
            "display_name": "Refresh Good Model",
        }
        assert AIModel.MODEL_CONFIGS == expected
