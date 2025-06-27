"""
MatHud Flask Application Manager

Core Flask application configuration and initialization.
Manages dependency injection for AI API, WebDriver, workspace operations, and logging.

Dependencies:
    - Flask: Web framework core
    - static.openai_api: OpenAI API integration
    - static.workspace_manager: Workspace file operations
    - static.log_manager: Application logging
    - static.routes: Route definitions and registration
"""

import os
import secrets
from flask import Flask, jsonify
from flask_session import Session
from dotenv import load_dotenv
from static.openai_api import OpenAIChatCompletionsAPI
from static.workspace_manager import WorkspaceManager
from static.log_manager import LogManager


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
        - OpenAIChatCompletionsAPI: AI-powered mathematical problem solving
        - WorkspaceManager: File system operations and workspace organization
        - LogManager: Application-wide logging and debugging support
        - Route Registration: RESTful API endpoint configuration
    """
    
    @staticmethod
    def is_deployed():
        """Check if the application is running in a deployed environment.
        
        Returns:
            bool: True if deployed (PORT environment variable is set), False for local development
        """
        return os.environ.get('PORT') is not None
    
    @staticmethod
    def requires_auth():
        """Check if authentication is required.
        
        Returns:
            bool: True if authentication should be required
        """
        # Load .env file if it exists
        load_dotenv()
        # Require auth if deployed OR if explicitly enabled via REQUIRE_AUTH
        return AppManager.is_deployed() or os.getenv('REQUIRE_AUTH', '').lower() in ('true', '1', 'yes')
    
    @staticmethod
    def get_auth_pin():
        """Get the authentication PIN from environment variables.
        
        Returns:
            str: The authentication PIN, or None if not set
        """
        # Load .env file if it exists
        load_dotenv()
        return os.getenv("AUTH_PIN")
    
    @staticmethod
    def make_response(data=None, message=None, status='success', code=200):
        """Create a consistent JSON response format.
        
        Args:
            data: Response payload data
            message: Human-readable status message
            status: Response status ('success', 'error', etc.)
            code: HTTP status code
            
        Returns:
            tuple: (Flask JSON response, HTTP status code)
        """
        response = {
            'status': status,
            'message': message,
            'data': data
        }
        return jsonify(response), code
    
    @staticmethod
    def create_app():
        """Create and configure the Flask application.
        
        Initializes all core managers (logging, AI API, workspace management)
        and registers application routes. WebDriver is initialized separately
        after Flask startup to avoid blocking. Configures session management
        for authentication in deployed environments.
        
        Returns:
            Flask: Configured Flask application instance
        """
        app = Flask(__name__, template_folder='../templates', static_folder='../static')
        
        # Load environment variables
        load_dotenv()
        
        # Configure session management for authentication
        app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
        app.config['SESSION_TYPE'] = 'filesystem'
        app.config['SESSION_PERMANENT'] = False
        app.config['SESSION_USE_SIGNER'] = True
        app.config['SESSION_KEY_PREFIX'] = 'mathud:'
        
        # Security settings for deployed environments
        if AppManager.is_deployed():
            app.config['SESSION_COOKIE_SECURE'] = True
            app.config['SESSION_COOKIE_HTTPONLY'] = True
            app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
        
        # Initialize Flask-Session
        Session(app)
        
        # Initialize managers
        app.log_manager = LogManager()
        app.ai_api = OpenAIChatCompletionsAPI()
        app.webdriver_manager = None  # Will be set after Flask starts
        
        # Initialize workspace manager
        app.workspace_manager = WorkspaceManager()
        
        # Import and register routes
        from static.routes import register_routes
        register_routes(app)
        
        return app 