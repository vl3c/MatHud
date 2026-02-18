"""
MatHud Flask Application Manager

Core Flask application configuration and initialization.
Manages dependency injection for AI API, WebDriver, workspace operations, and logging.

Dependencies:
    - Flask: Web framework core
    - static.openai_completions_api: OpenAI Chat Completions API integration
    - static.workspace_manager: Workspace file operations
    - static.log_manager: Application logging
    - static.routes: Route definitions and registration
"""

from __future__ import annotations

import os
import secrets
from typing import TYPE_CHECKING, Dict, Optional, Tuple, TypedDict, Union

from cachelib.file import FileSystemCache
from dotenv import load_dotenv
from flask import Flask, Response, jsonify
from flask_session import Session as FlaskSession

from static.log_manager import LogManager
from static.openai_completions_api import OpenAIChatCompletionsAPI
from static.openai_responses_api import OpenAIResponsesAPI
from static.providers import discover_providers
from static.workspace_manager import WorkspaceManager


if TYPE_CHECKING:
    from static.openai_api_base import OpenAIAPIBase
    from static.webdriver_manager import WebDriverManager


JsonValue = Union[str, int, float, bool, None, Dict[str, "JsonValue"], list["JsonValue"]]


class ApiResponseDict(TypedDict, total=False):
    """Standard API response structure."""

    status: str
    message: Optional[str]
    data: JsonValue


class MatHudFlask(Flask):
    """Flask subclass with MatHud service attributes."""

    log_manager: LogManager
    ai_api: OpenAIChatCompletionsAPI
    responses_api: OpenAIResponsesAPI
    webdriver_manager: Optional["WebDriverManager"]
    workspace_manager: WorkspaceManager
    current_attached_images: Optional[list[str]]  # User-attached images for current request
    providers: Dict[str, "OpenAIAPIBase"]  # Lazily-loaded provider instances by name


class AppManager:
    """Manages core Flask application setup and utilities for the MatHud mathematical visualization system.

    Coordinates Flask application initialization with comprehensive dependency injection for all core services.
    Provides standardized API response formatting and error handling across the entire application.

    Core Responsibilities:
        - Flask Application Factory: Creates and configures Flask app instances
        - Dependency Injection: Initializes and coordinates OpenAI API, WebDriver, workspace, and logging managers
        - Response Standardization: Consistent JSON API response formatting
        - Service Integration: Bridges Flask web framework with specialized application managers
        - Authentication: Session management and pseudo-login for deployed environments

    Managed Dependencies:
        - OpenAIChatCompletionsAPI: Chat Completions API for standard models
        - OpenAIResponsesAPI: Responses API for reasoning models (GPT-5, o3, o4-mini)
        - WorkspaceManager: File system operations and workspace organization
        - LogManager: Application-wide logging and debugging support
        - Route Registration: RESTful API endpoint configuration
    """

    @staticmethod
    def is_deployed() -> bool:
        """Check if the application is running in a deployed environment.

        Returns:
            bool: True if deployed (PORT environment variable is set), False for local development
        """
        return os.environ.get("PORT") is not None

    @staticmethod
    def _load_env() -> None:
        """Load environment from project .env and parent .env (API keys)."""
        load_dotenv()
        parent_env = os.path.join(os.path.dirname(os.getcwd()), ".env")
        if os.path.exists(parent_env):
            load_dotenv(parent_env)

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
        and registers application routes. WebDriver is initialized separately
        after Flask startup to avoid blocking. Configures session management
        for authentication in deployed environments using modern CacheLib backend.

        Returns:
            Flask: Configured Flask application instance
        """
        app = MatHudFlask(__name__, template_folder="../templates", static_folder="../static")

        # Load environment variables from project .env and parent .env (API keys)
        AppManager._load_env()

        # Configure session management for authentication using modern CacheLib backend
        app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))

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
        # tools dynamically after search_tools returns.
        app.ai_api = OpenAIChatCompletionsAPI()
        app.ai_api.set_tool_mode("search")
        app.responses_api = OpenAIResponsesAPI()
        app.responses_api.set_tool_mode("search")
        app.webdriver_manager = None  # Will be set after Flask starts
        app.current_attached_images = None  # User-attached images for current request
        app.providers = {}  # Lazily-loaded provider instances

        # Initialize workspace manager
        app.workspace_manager = WorkspaceManager()

        # Initialize TTS manager (eager load to check availability at startup)
        AppManager._initialize_tts()

        # Import and register routes
        from static.routes import register_routes

        register_routes(app)

        return app

    @staticmethod
    def _initialize_tts() -> None:
        """Initialize TTS manager and log availability status."""
        try:
            from static.tts_manager import get_tts_manager

            manager = get_tts_manager()
            if manager.is_available():
                print("TTS: Kokoro initialized successfully")
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
