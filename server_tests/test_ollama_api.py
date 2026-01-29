"""
Tests for Ollama API provider.

Tests Ollama-specific functionality including server availability checking,
model discovery, and tool-capable model filtering.
"""

import shutil
from unittest.mock import MagicMock, patch

import pytest
import requests as requests_lib

from static.providers.local.ollama_api import OllamaAPI


class TestOllamaAPIServerChecks:
    """Tests for Ollama server availability checking."""

    def test_is_server_running_success(self) -> None:
        """Returns True when server responds with 200."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = OllamaAPI.is_server_running()

            assert result is True
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "/api/tags" in call_args[0][0]
            assert call_args[1]["timeout"] == 2

    def test_is_server_running_connection_error(self) -> None:
        """Returns False when connection fails."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")

            result = OllamaAPI.is_server_running()

            assert result is False

    def test_is_server_running_server_error(self) -> None:
        """Returns False when server returns error status."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response

            result = OllamaAPI.is_server_running()

            assert result is False

    def test_is_server_running_timeout(self) -> None:
        """Returns False on timeout."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests_lib.exceptions.Timeout()

            result = OllamaAPI.is_server_running()

            assert result is False


class TestOllamaAPIModelDiscovery:
    """Tests for Ollama model discovery."""

    def test_get_tool_capable_models_success(self) -> None:
        """Returns tool-capable models from server response."""
        mock_models_response = {
            "models": [
                {"name": "llama3.1:8b", "size": 4700000000},
                {"name": "llama2:7b", "size": 3800000000},  # Not tool-capable
                {"name": "qwen2.5:7b", "size": 4400000000},
                {"name": "phi:3b", "size": 2000000000},  # Not tool-capable
                {"name": "mistral:7b", "size": 4100000000},
            ]
        }

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_models_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = OllamaAPI.get_tool_capable_models()

            # Should only include tool-capable models
            assert len(result) == 3
            names = [m["name"] for m in result]
            assert "llama3.1:8b" in names
            assert "qwen2.5:7b" in names
            assert "mistral:7b" in names
            # Non-tool-capable models should be excluded
            assert "llama2:7b" not in names
            assert "phi:3b" not in names

    def test_get_tool_capable_models_empty_server(self) -> None:
        """Returns empty list when no models installed."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": []}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = OllamaAPI.get_tool_capable_models()

            assert result == []

    def test_get_tool_capable_models_no_tool_capable(self) -> None:
        """Returns empty list when no tool-capable models installed."""
        mock_models_response = {
            "models": [
                {"name": "llama2:7b", "size": 3800000000},
                {"name": "phi:3b", "size": 2000000000},
            ]
        }

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_models_response
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = OllamaAPI.get_tool_capable_models()

            assert result == []

    def test_get_tool_capable_models_server_error(self) -> None:
        """Returns empty list on server error."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")

            result = OllamaAPI.get_tool_capable_models()

            assert result == []


class TestOllamaAPIConfiguration:
    """Tests for Ollama API configuration."""

    def test_default_base_url(self) -> None:
        """Uses default URL when env var not set."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove the env var if it exists
            import os
            os.environ.pop("OLLAMA_BASE_URL", None)

            instance = object.__new__(OllamaAPI)
            url = instance._get_base_url()

            assert url == "http://localhost:11434"

    def test_custom_base_url_from_env(self) -> None:
        """Uses URL from environment variable."""
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://custom-host:8080"}):
            instance = object.__new__(OllamaAPI)
            url = instance._get_base_url()

            assert url == "http://custom-host:8080"

    def test_provider_name(self) -> None:
        """Returns correct provider name."""
        instance = object.__new__(OllamaAPI)
        assert instance._get_provider_name() == "Ollama"


class TestOllamaAPIRegistration:
    """Tests for provider registration."""

    def test_registered_with_local_registry(self) -> None:
        """Ollama API is registered with LocalProviderRegistry."""
        from static.providers.local import LocalProviderRegistry

        provider_class = LocalProviderRegistry.get_provider_class("ollama")
        assert provider_class is OllamaAPI

    def test_in_registered_providers(self) -> None:
        """Ollama appears in registered providers list."""
        from static.providers.local import LocalProviderRegistry

        providers = LocalProviderRegistry.get_registered_providers()
        assert "ollama" in providers


class TestOllamaAPIServerLifecycle:
    """Tests for Ollama server lifecycle management."""

    def test_get_ollama_executable_in_path(self) -> None:
        """Finds ollama in PATH."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/ollama"
            assert OllamaAPI.get_ollama_executable() == "/usr/bin/ollama"
            mock_which.assert_called_once_with("ollama")

    def test_get_ollama_executable_not_found(self) -> None:
        """Returns None when ollama not found anywhere."""
        with patch("shutil.which", return_value=None):
            with patch("os.path.isfile", return_value=False):
                assert OllamaAPI.get_ollama_executable() is None

    def test_is_ollama_installed_true(self) -> None:
        """Returns True when executable found."""
        with patch.object(OllamaAPI, "get_ollama_executable", return_value="/usr/bin/ollama"):
            assert OllamaAPI.is_ollama_installed() is True

    def test_is_ollama_installed_false(self) -> None:
        """Returns False when executable not found."""
        with patch.object(OllamaAPI, "get_ollama_executable", return_value=None):
            assert OllamaAPI.is_ollama_installed() is False

    def test_start_server_already_running(self) -> None:
        """Returns success if server already running."""
        with patch.object(OllamaAPI, "is_server_running", return_value=True):
            success, message = OllamaAPI.start_server()
            assert success is True
            assert "already running" in message.lower()

    def test_start_server_not_installed(self) -> None:
        """Returns error if Ollama not installed."""
        with patch.object(OllamaAPI, "is_server_running", return_value=False):
            with patch.object(OllamaAPI, "get_ollama_executable", return_value=None):
                success, message = OllamaAPI.start_server()
                assert success is False
                assert "not installed" in message.lower()


class TestOllamaAPIModelLoading:
    """Tests for model loading functionality."""

    def test_get_loaded_models_success(self) -> None:
        """Returns list of loaded models."""
        mock_response_data = {
            "models": [
                {"name": "llama3.1:8b", "size": 4700000000},
                {"name": "mistral:7b", "size": 4100000000},
            ]
        }

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = OllamaAPI.get_loaded_models()

            assert len(result) == 2
            assert "llama3.1:8b" in result
            assert "mistral:7b" in result

    def test_get_loaded_models_empty(self) -> None:
        """Returns empty list when no models loaded."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": []}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = OllamaAPI.get_loaded_models()

            assert result == []

    def test_get_loaded_models_server_error(self) -> None:
        """Returns empty list on server error."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")

            result = OllamaAPI.get_loaded_models()

            assert result == []

    def test_is_model_loaded_exact_match(self) -> None:
        """Detects exact model name match."""
        with patch.object(OllamaAPI, "get_loaded_models", return_value=["llama3.1:8b"]):
            assert OllamaAPI.is_model_loaded("llama3.1:8b") is True

    def test_is_model_loaded_base_name_match(self) -> None:
        """Detects base name match (ignoring tags)."""
        with patch.object(OllamaAPI, "get_loaded_models", return_value=["llama3.1:8b"]):
            assert OllamaAPI.is_model_loaded("llama3.1:70b") is True

    def test_is_model_loaded_not_loaded(self) -> None:
        """Returns False when model not loaded."""
        with patch.object(OllamaAPI, "get_loaded_models", return_value=["mistral:7b"]):
            assert OllamaAPI.is_model_loaded("llama3.1:8b") is False

    def test_preload_model_already_loaded(self) -> None:
        """Returns success if model already loaded."""
        with patch.object(OllamaAPI, "is_model_loaded", return_value=True):
            success, message = OllamaAPI.preload_model("llama3.1:8b")
            assert success is True
            assert "already loaded" in message.lower()

    def test_preload_model_success(self) -> None:
        """Successfully preloads a model."""
        with patch.object(OllamaAPI, "is_model_loaded", return_value=False):
            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_post.return_value = mock_response

                success, message = OllamaAPI.preload_model("llama3.1:8b")

                assert success is True
                assert "loaded" in message.lower()
                # Verify the API was called correctly
                call_args = mock_post.call_args
                assert "/api/generate" in call_args[0][0]
                assert call_args[1]["json"]["model"] == "llama3.1:8b"

    def test_preload_model_failure(self) -> None:
        """Returns error on preload failure."""
        with patch.object(OllamaAPI, "is_model_loaded", return_value=False):
            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 500
                mock_response.text = "Model not found"
                mock_post.return_value = mock_response

                success, message = OllamaAPI.preload_model("nonexistent:model")

                assert success is False
                assert "failed" in message.lower()

    def test_preload_model_timeout(self) -> None:
        """Returns error on timeout."""
        with patch.object(OllamaAPI, "is_model_loaded", return_value=False):
            with patch("requests.post") as mock_post:
                mock_post.side_effect = requests_lib.exceptions.Timeout()

                success, message = OllamaAPI.preload_model("llama3.1:8b", timeout=5)

                assert success is False
                assert "timed out" in message.lower()
