"""
MatHud Flask Application Manager

Core Flask application configuration and initialization.
Manages dependency injection for AI API, WebDriver, workspace operations, and logging.

Dependencies:
    - Flask: Web framework core
    - static.openai_api: OpenAI API integration
    - static.webdriver_manager: Selenium WebDriver for vision system
    - static.workspace_manager: Workspace file operations
    - static.log_manager: Application logging
    - static.routes: Route definitions and registration
"""

from flask import Flask, jsonify
from static.openai_api import OpenAIChatCompletionsAPI
from static.webdriver_manager import WebDriverManager
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
        
    Managed Dependencies:
        - OpenAIChatCompletionsAPI: AI-powered mathematical problem solving
        - WebDriverManager: Selenium-based vision system for canvas image capture
        - WorkspaceManager: File system operations and workspace organization
        - LogManager: Application-wide logging and debugging support
        - Route Registration: RESTful API endpoint configuration
    """
    
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
        after Flask startup to avoid blocking.
        
        Returns:
            Flask: Configured Flask application instance
        """
        app = Flask(__name__, template_folder='../templates', static_folder='../static')
        
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