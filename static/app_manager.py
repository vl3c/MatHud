from flask import Flask, jsonify
from static.openai_api import OpenAIChatCompletionsAPI
from static.webdriver_manager import WebDriverManager
from static.workspace_manager import WorkspaceManager
from static.log_manager import LogManager


class AppManager:
    """Manages core Flask application setup and utilities."""
    
    @staticmethod
    def make_response(data=None, message=None, status='success', code=200):
        """Create a consistent JSON response format."""
        response = {
            'status': status,
            'message': message,
            'data': data
        }
        return jsonify(response), code
    
    @staticmethod
    def create_app():
        """Create and configure the Flask application."""
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