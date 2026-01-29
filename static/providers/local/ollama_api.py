"""
MatHud Ollama API Provider

Ollama local LLM implementation using the OpenAI-compatible API.
Provides model discovery, tool-capable model filtering, and server lifecycle management.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

from static.ai_model import AIModel
from static.functions_definitions import FunctionDefinition
from static.providers.local import LocalLLMBase, LocalProviderRegistry

_logger = logging.getLogger("mathud")

# Module-level process reference for the Ollama server we started
_ollama_process: Optional[subprocess.Popen[bytes]] = None


class OllamaAPI(LocalLLMBase):
    """Ollama local LLM provider.

    Connects to the Ollama server's OpenAI-compatible API endpoint.
    Automatically discovers available models and filters for tool support.
    """

    ENV_VAR = "OLLAMA_BASE_URL"
    DEFAULT_URL = "http://localhost:11434"

    def __init__(
        self,
        model: Optional[AIModel] = None,
        temperature: float = 0.2,
        tools: Optional[Sequence[FunctionDefinition]] = None,
        max_tokens: int = 16000,
    ) -> None:
        """Initialize Ollama API client.

        Args:
            model: AI model to use. If None, uses the first available tool-capable model.
            temperature: Sampling temperature.
            tools: Custom tool definitions.
            max_tokens: Maximum tokens in response.
        """
        super().__init__(
            model=model,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
        )

    def _get_base_url(self) -> str:
        """Get the Ollama server base URL.

        Returns:
            The server URL from environment or default
        """
        return os.getenv(self.ENV_VAR, self.DEFAULT_URL)

    def _get_provider_name(self) -> str:
        """Get the provider name.

        Returns:
            'Ollama'
        """
        return "Ollama"

    def _is_available(self) -> bool:
        """Check if the Ollama server is running.

        Returns:
            True if the server responds to /api/tags
        """
        import requests

        try:
            response = requests.get(
                f"{self._get_base_url()}/api/tags",
                timeout=2,
            )
            return response.status_code == 200
        except Exception as e:
            _logger.debug(f"Ollama server not available: {e}")
            return False

    def _discover_models(self) -> List[Dict[str, Any]]:
        """Query Ollama for available models.

        Returns:
            List of model info dicts with 'name' and 'size' keys
        """
        import requests

        try:
            response = requests.get(
                f"{self._get_base_url()}/api/tags",
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()

            models = []
            for model in data.get("models", []):
                models.append({
                    "name": model.get("name", ""),
                    "size": model.get("size", 0),
                    "modified_at": model.get("modified_at", ""),
                })

            return models
        except Exception as e:
            _logger.warning(f"Failed to discover Ollama models: {e}")
            return []

    @classmethod
    def is_server_running(cls) -> bool:
        """Check if Ollama server is running (class method for convenience).

        Returns:
            True if server is accessible
        """
        import requests

        base_url = os.getenv(cls.ENV_VAR, cls.DEFAULT_URL)
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    @classmethod
    def get_tool_capable_models(cls) -> List[Dict[str, Any]]:
        """Get list of installed models that support tool calling (class method).

        Returns:
            List of model info dicts for tool-capable models
        """
        import requests
        from static.providers.local import supports_tools

        base_url = os.getenv(cls.ENV_VAR, cls.DEFAULT_URL)
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()

            tool_capable = []
            for model in data.get("models", []):
                name = model.get("name", "")
                if supports_tools(name):
                    tool_capable.append({
                        "name": name,
                        "size": model.get("size", 0),
                        "modified_at": model.get("modified_at", ""),
                    })

            return tool_capable
        except Exception as e:
            _logger.debug(f"Failed to get Ollama models: {e}")
            return []

    @classmethod
    def get_ollama_executable(cls) -> Optional[str]:
        """Find the Ollama executable path.

        Checks PATH first, then common installation locations on Windows.

        Returns:
            Path to ollama executable, or None if not found
        """
        # Check PATH first
        path_result = shutil.which("ollama")
        if path_result:
            return path_result

        # On Windows, check common installation locations
        if os.name == "nt":
            common_paths = [
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
                os.path.expandvars(r"%PROGRAMFILES%\Ollama\ollama.exe"),
                os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe"),
            ]
            for path in common_paths:
                if os.path.isfile(path):
                    return path

        return None

    @classmethod
    def is_ollama_installed(cls) -> bool:
        """Check if Ollama CLI is installed on the system.

        Returns:
            True if 'ollama' command is available
        """
        return cls.get_ollama_executable() is not None

    @classmethod
    def start_server(cls, timeout: float = 30.0) -> Tuple[bool, str]:
        """Start the Ollama server if not already running.

        Args:
            timeout: Maximum seconds to wait for server to become ready

        Returns:
            Tuple of (success, message)
        """
        global _ollama_process

        # Already running?
        if cls.is_server_running():
            return True, "Ollama server already running"

        # Find Ollama executable
        ollama_exe = cls.get_ollama_executable()
        if not ollama_exe:
            return False, "Ollama is not installed"

        # Start the server
        try:
            _logger.info(f"Starting Ollama server from {ollama_exe}...")
            # Use CREATE_NO_WINDOW on Windows to avoid console popup
            kwargs: Dict[str, Any] = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            _ollama_process = subprocess.Popen([ollama_exe, "serve"], **kwargs)

            # Wait for server to become ready
            start_time = time.time()
            while time.time() - start_time < timeout:
                if cls.is_server_running():
                    _logger.info("Ollama server started successfully")
                    return True, "Ollama server started"
                time.sleep(0.5)

            return False, f"Ollama server did not start within {timeout}s"

        except Exception as e:
            _logger.error(f"Failed to start Ollama server: {e}")
            return False, f"Failed to start Ollama: {e}"

    @classmethod
    def unload_all_models(cls) -> None:
        """Unload all models from memory to release GPU resources."""
        if not cls.is_server_running():
            return

        loaded = cls.get_loaded_models()
        for model_name in loaded:
            try:
                cls.unload_model(model_name)
                _logger.info(f"Unloaded model: {model_name}")
            except Exception as e:
                _logger.warning(f"Failed to unload {model_name}: {e}")

    @classmethod
    def stop_server(cls) -> None:
        """Stop the Ollama server if we started it.

        First unloads all models to release GPU memory, then terminates the process.
        """
        global _ollama_process

        # First, unload all models to release GPU memory
        try:
            cls.unload_all_models()
        except Exception as e:
            _logger.warning(f"Error unloading models: {e}")

        if _ollama_process is not None:
            _logger.info("Stopping Ollama server...")
            try:
                _ollama_process.terminate()
                _ollama_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _logger.warning("Ollama server did not terminate gracefully, killing...")
                _ollama_process.kill()
                _ollama_process.wait(timeout=5)
            except Exception as e:
                _logger.warning(f"Error stopping Ollama server: {e}")
            finally:
                _ollama_process = None
            _logger.info("Ollama server stopped")

    @classmethod
    def get_loaded_models(cls) -> List[str]:
        """Get list of models currently loaded in memory.

        Returns:
            List of model names that are loaded
        """
        import requests

        base_url = os.getenv(cls.ENV_VAR, cls.DEFAULT_URL)
        try:
            response = requests.get(f"{base_url}/api/ps", timeout=5)
            response.raise_for_status()
            data = response.json()

            loaded = []
            for model in data.get("models", []):
                name = model.get("name", "")
                if name:
                    loaded.append(name)

            return loaded
        except Exception as e:
            _logger.debug(f"Failed to get loaded models: {e}")
            return []

    @classmethod
    def is_model_loaded(cls, model_name: str) -> bool:
        """Check if a specific model is loaded in memory.

        Args:
            model_name: The model name to check

        Returns:
            True if the model is currently loaded
        """
        loaded = cls.get_loaded_models()
        # Normalize for comparison (handle with/without tags)
        model_base = model_name.split(":")[0].lower()
        for loaded_model in loaded:
            loaded_base = loaded_model.split(":")[0].lower()
            if model_base == loaded_base or model_name.lower() == loaded_model.lower():
                return True
        return False

    @classmethod
    def preload_model(cls, model_name: str, timeout: float = 120.0) -> Tuple[bool, str]:
        """Preload a model into memory by sending a minimal request.

        Args:
            model_name: The model to preload
            timeout: Maximum seconds to wait for model to load

        Returns:
            Tuple of (success, message)
        """
        import requests

        # Already loaded?
        if cls.is_model_loaded(model_name):
            return True, f"Model {model_name} is already loaded"

        base_url = os.getenv(cls.ENV_VAR, cls.DEFAULT_URL)

        try:
            _logger.info(f"Preloading model {model_name}...")

            # Use the generate endpoint with keep_alive to load the model
            # This is more reliable than chat for just loading
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",  # Empty prompt just loads the model
                    "keep_alive": "10m",  # Keep loaded for 10 minutes
                },
                timeout=timeout,
            )

            if response.status_code == 200:
                _logger.info(f"Model {model_name} loaded successfully")
                return True, f"Model {model_name} loaded"
            else:
                error_msg = response.text[:200] if response.text else f"Status {response.status_code}"
                return False, f"Failed to load model: {error_msg}"

        except requests.exceptions.Timeout:
            return False, f"Model loading timed out after {timeout}s"
        except Exception as e:
            _logger.error(f"Failed to preload model {model_name}: {e}")
            return False, f"Failed to load model: {e}"

    @classmethod
    def unload_model(cls, model_name: str) -> Tuple[bool, str]:
        """Unload a model from memory.

        Args:
            model_name: The model to unload

        Returns:
            Tuple of (success, message)
        """
        import requests

        base_url = os.getenv(cls.ENV_VAR, cls.DEFAULT_URL)

        try:
            # Setting keep_alive to 0 unloads the model
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",
                    "keep_alive": 0,
                },
                timeout=10,
            )

            if response.status_code == 200:
                _logger.info(f"Model {model_name} unloaded")
                return True, f"Model {model_name} unloaded"
            else:
                return False, f"Failed to unload model: {response.status_code}"

        except Exception as e:
            _logger.error(f"Failed to unload model {model_name}: {e}")
            return False, f"Failed to unload model: {e}"


# Self-register with local provider registry
LocalProviderRegistry.register("ollama", OllamaAPI)
