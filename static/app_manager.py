"""
MatHud Flask Application Manager

Core Flask application configuration and initialization.
Manages dependency injection for AI API, workspace operations, and logging.

Dependencies:
    - Flask: Web framework core
    - static.openai_completions_api: OpenAI Chat Completions API integration
    - static.workspace_manager: Workspace file operations
    - static.log_manager: Application logging
    - static.routes: Route definitions and registration
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import TYPE_CHECKING, Dict, Optional, Tuple, TypedDict, Union

from cachelib.file import FileSystemCache
from flask import Flask, Response, jsonify
from flask_session import Session as FlaskSession

from static.config import MAX_CONTENT_LENGTH_BYTES
from static.env_config import load_env_files
from static.log_manager import LogManager
from static.openai_api_base import get_configured_tool_mode
from static.openai_completions_api import OpenAIChatCompletionsAPI
from static.openai_responses_api import OpenAIResponsesAPI
from static.providers import discover_providers
from static.workspace_manager import WorkspaceManager


if TYPE_CHECKING:
    from static.openai_api_base import OpenAIAPIBase


_logger = logging.getLogger(__name__)


JsonValue = Union[str, int, float, bool, None, Dict[str, "JsonValue"], list["JsonValue"]]


class ApiResponseDict(TypedDict, total=False):
    """Standard API response structure."""

    status: str
    message: Optional[str]
    data: JsonValue


# Set by the desktop launcher: the app is local-only even if a .env file sets PORT.
LOCAL_MODE_ENV = "MATHUD_LOCAL_MODE"


def is_local_mode_forced() -> bool:
    """True when MATHUD_LOCAL_MODE marks this process as a local, single-user app."""
    return os.environ.get(LOCAL_MODE_ENV, "").lower() in ("1", "true", "yes")


class MatHudFlask(Flask):
    """Flask subclass with MatHud service attributes."""

    log_manager: LogManager
    ai_api: OpenAIChatCompletionsAPI
    responses_api: OpenAIResponsesAPI
    workspace_manager: WorkspaceManager
    current_attached_images: Optional[list[str]]  # User-attached images for current request
    providers: Dict[str, "OpenAIAPIBase"]  # Lazily-loaded provider instances by name

    # static/vendor/ paths include the library version, so their contents never
    # change under a given URL and browsers may cache them for a year.
    VENDOR_STATIC_PREFIX = "vendor/"
    VENDOR_CACHE_MAX_AGE_S = 365 * 24 * 60 * 60

    def get_send_file_max_age(self, filename: Optional[str]) -> Optional[int]:
        """Cache vendored libraries for long; other static files keep Flask's default."""
        if filename is not None and filename.replace("\\", "/").startswith(self.VENDOR_STATIC_PREFIX):
            return self.VENDOR_CACHE_MAX_AGE_S
        default_max_age: Optional[int] = super().get_send_file_max_age(filename)
        return default_max_age


class AppManager:
    """Manages core Flask application setup and utilities for the MatHud mathematical visualization system.

    Coordinates Flask application initialization with comprehensive dependency injection for all core services.
    Provides standardized API response formatting and error handling across the entire application.

    Core Responsibilities:
        - Flask Application Factory: Creates and configures Flask app instances
        - Dependency Injection: Initializes and coordinates OpenAI API, workspace, and logging managers
        - Response Standardization: Consistent JSON API response formatting
        - Service Integration: Bridges Flask web framework with specialized application managers
        - Authentication: Session management and pseudo-login for deployed environments

    Managed Dependencies:
        - OpenAIChatCompletionsAPI: Chat Completions API for standard models
        - OpenAIResponsesAPI: Responses API for reasoning models (GPT-6, GPT-5.6)
        - WorkspaceManager: File system operations and workspace organization
        - LogManager: Application-wide logging and debugging support
        - Route Registration: RESTful API endpoint configuration
    """

    @staticmethod
    def is_deployed() -> bool:
        """Check if the application is running in a deployed environment.

        Returns:
            bool: True if deployed (PORT environment variable is set), False for local development
            or when MATHUD_LOCAL_MODE forces local mode (the desktop launcher sets it)
        """
        if is_local_mode_forced():
            return False
        return os.environ.get("PORT") is not None

    @staticmethod
    def _load_env() -> None:
        """Load environment from project .env and parent .env (API keys)."""
        load_env_files()

    @staticmethod
    def requires_auth() -> bool:
        """Check if authentication is required.

        Returns:
            bool: True if authentication should be required
        """
        AppManager._load_env()
        # Require auth if deployed OR if explicitly enabled via REQUIRE_AUTH
        return AppManager.is_deployed() or os.getenv("REQUIRE_AUTH", "").lower() in ("true", "1", "yes")

    @staticmethod
    def get_auth_pin() -> Optional[str]:
        """Get the authentication PIN from environment variables.

        Returns:
            str: The authentication PIN, or None if not set
        """
        AppManager._load_env()
        return os.getenv("AUTH_PIN")

    @staticmethod
    def make_response(
        data: JsonValue | None = None,
        message: Optional[str] = None,
        status: str = "success",
        code: int = 200,
    ) -> Tuple[Response, int]:
        """Create a consistent JSON response format.

        Args:
            data: Response payload data
            message: Human-readable status message
            status: Response status ('success', 'error', etc.)
            code: HTTP status code

        Returns:
            tuple: (Flask JSON response, HTTP status code)
        """
        response: ApiResponseDict = {
            "status": status,
            "message": message,
            "data": data,
        }
        return jsonify(response), code

    @staticmethod
    def create_app() -> MatHudFlask:
        """Create and configure the Flask application.

        Initializes all core managers (logging, AI API, workspace management)
        and registers application routes. Configures session management
        for authentication in deployed environments using modern CacheLib backend.

        Returns:
            Flask: Configured Flask application instance
        """
        app = MatHudFlask(__name__, template_folder="../templates", static_folder="../static")

        # Load environment variables from project .env and parent .env (API keys)
        AppManager._load_env()

        # Configure session management for authentication using modern CacheLib backend.
        # A persistent SECRET_KEY is required for sessions to survive process restarts.
        # In deployed mode an ephemeral fallback silently logs everyone out on every
        # restart, so warn prominently (but do not hard-fail).
        secret_key_env = os.getenv("SECRET_KEY")
        if secret_key_env:
            app.secret_key = secret_key_env
        else:
            app.secret_key = secrets.token_hex(32)
            if AppManager.is_deployed():
                _logger.warning(
                    "SECRET_KEY is not set in a deployed environment. A random key was "
                    "generated for this process, so ALL user sessions will be invalidated "
                    "on every restart. Set SECRET_KEY to a stable secret to persist sessions."
                )

        # Cap total request body size and return clean JSON on overflow so clients
        # do not receive Flask's default HTML 413 page.
        app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_BYTES

        @app.errorhandler(413)
        def _handle_request_entity_too_large(_error: Exception) -> Tuple[Response, int]:
            return jsonify({"error": "Request payload too large"}), 413

        # Create session directory if it doesn't exist
        session_dir = os.path.join(os.getcwd(), "flask_session")
        os.makedirs(session_dir, exist_ok=True)

        # Modern Flask-Session configuration using CacheLib
        app.config["SESSION_TYPE"] = "cachelib"
        app.config["SESSION_CACHELIB"] = FileSystemCache(cache_dir=session_dir)
        app.config["SESSION_PERMANENT"] = False
        app.config["SESSION_KEY_PREFIX"] = "mathud:"

        # Security settings for deployed environments
        if AppManager.is_deployed():
            app.config["SESSION_COOKIE_SECURE"] = True
            app.config["SESSION_COOKIE_HTTPONLY"] = True
            app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

        # Initialize Flask-Session
        FlaskSession(app)

        # Discover and register providers
        discover_providers()

        # Initialize managers
        app.log_manager = LogManager()
        # Default to minimal search-first tool exposure; routes inject matching
        # tools dynamically after search_tools returns. MATHUD_TOOL_EXPOSURE=full
        # exposes every tool up front instead.
        tool_mode = get_configured_tool_mode()
        app.ai_api = OpenAIChatCompletionsAPI()
        app.ai_api.set_tool_mode(tool_mode)
        app.responses_api = OpenAIResponsesAPI()
        app.responses_api.set_tool_mode(tool_mode)
        app.current_attached_images = None  # User-attached images for current request
        app.providers = {}  # Lazily-loaded provider instances

        # Initialize workspace manager
        app.workspace_manager = WorkspaceManager()

        # Report TTS availability; the Kokoro model itself loads on first use
        AppManager._initialize_tts()

        # Import and register routes
        from static.routes import register_routes

        register_routes(app)

        return app

    @staticmethod
    def _initialize_tts() -> None:
        """Create the TTS manager and log whether Kokoro is installed.

        Only checks that the packages are importable; Kokoro and torch are
        imported by the first speech request so startup stays fast and light.
        """
        try:
            from static.tts_manager import get_tts_manager

            manager = get_tts_manager()
            if manager.is_available():
                print("TTS: Kokoro available (model loads on first use)")
            else:
                print("TTS: Kokoro not available (install with: pip install kokoro)")
        except SystemExit as e:
            # Some third-party imports can call sys.exit() in unsupported
            # environments; only suppress the known externally-managed error.
            message = str(e)
            if "externally-managed-environment" in message.lower():
                print(f"TTS: Failed to initialize ({e})")
            else:
                raise
        except Exception as e:
            print(f"TTS: Failed to initialize ({e})")
