from flask import Flask, json, request, render_template, jsonify
import time
import signal
import sys
from static.openai_api import OpenAIChatCompletionsAPI
from static.webdriver_manager import WebDriverManager
from static.tool_call_processor import ToolCallProcessor
from static.log_manager import LogManager


def signal_handler(sig, frame):
    """Handle graceful shutdown on interrupt signal."""
    print('\nShutting down gracefully...')
    # Clean up WebDriverManager
    if hasattr(app, 'webdriver_manager') and app.webdriver_manager:
        try:
            app.webdriver_manager.cleanup()
        except Exception as e:
            print(f"Error closing WebDriver: {e}")
    print("Goodbye!")
    sys.exit(0)


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Initialize managers
    app.log_manager = LogManager()
    app.ai_api = OpenAIChatCompletionsAPI()
    app.webdriver_manager = None  # Will be set after Flask starts
    
    # Initialize workspace manager
    from static.workspace_manager import WorkspaceManager
    workspace_manager = WorkspaceManager()

    @app.route('/')
    def get_index():
        return render_template('index.html')

    @app.route('/init_webdriver')
    def init_webdriver_route():
        """Route to initialize WebDriver after Flask has started"""
        if not app.webdriver_manager:
            try:
                app.webdriver_manager = WebDriverManager()
            except Exception as e:
                print(f"Failed to initialize WebDriverManager: {str(e)}")
                return f"WebDriver initialization failed: {str(e)}", 500
        return "WebDriver initialization successful"

    @app.route('/save_workspace', methods=['POST'])
    def save_workspace_route():
        """Save the current workspace state."""
        try:
            data = request.get_json()
            state = data.get('state')
            name = data.get('name')
            
            success = workspace_manager.save_workspace(state, name)
            if success:
                return jsonify({'status': 'success', 'message': 'Workspace saved successfully'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to save workspace'}), 500
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/load_workspace', methods=['GET'])
    def load_workspace_route():
        """Load a workspace state."""
        try:
            name = request.args.get('name')
            state = workspace_manager.load_workspace(name)
            
            return jsonify({
                'status': 'success',
                'state': state
            })
        except FileNotFoundError as e:
            return jsonify({'status': 'error', 'message': str(e)}), 404
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/list_workspaces', methods=['GET'])
    def list_workspaces_route():
        """List all saved workspaces."""
        try:
            workspaces = workspace_manager.list_workspaces()
            return jsonify(workspaces)
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/delete_workspace', methods=['GET'])
    def delete_workspace_route():
        """Delete a workspace."""
        try:
            name = request.args.get('name')
            if not name:
                return jsonify({'status': 'error', 'message': 'Workspace name is required'}), 400
                
            success = workspace_manager.delete_workspace(name)
            if success:
                return jsonify({'status': 'success', 'message': 'Workspace deleted successfully'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to delete workspace'}), 404
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/send_message', methods=['POST'])
    def send_message():
        message = request.json.get('message')
        svg_state = request.json.get('svg_state')  # Get SVG state from request
        message_json = json.loads(message)
        use_vision = message_json.get('use_vision', False)  # Get vision state from message
        ai_model = message_json.get('ai_model')  # Get AI model from message
        
        if ai_model:
            app.ai_api.set_model(ai_model)

        app.log_manager.log_user_message(message)

        # Check if WebDriver needs to be initialized
        if use_vision and (not hasattr(app, 'webdriver_manager') or app.webdriver_manager is None):
            print("WebDriver not found, attempting to initialize...")
            init_webdriver_route()

        # Capture canvas image before sending to AI
        if use_vision and hasattr(app, 'webdriver_manager') and app.webdriver_manager:
            app.webdriver_manager.capture_svg_state(svg_state)

        # Proceed with creating chat completion
        response = app.ai_api.create_chat_completion(message)

        ai_message = response.content if response.content is not None else ""
        app.log_manager.log_ai_response(ai_message)

        ai_tool_calls = response.tool_calls if response.tool_calls is not None else []
        ai_tool_calls = ToolCallProcessor.jsonify_tool_calls(ai_tool_calls)
        app.log_manager.log_ai_tool_calls(ai_tool_calls)

        response = json.dumps({"ai_message": ai_message, "ai_tool_calls": ai_tool_calls})
        return response

    return app


# Create the app at module level for VS Code debugger
app = create_app()

# Register signal handler at module level for both run modes
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    try:
        # Start Flask in a thread
        from threading import Thread
        server = Thread(target=app.run, kwargs={
            'host': '127.0.0.1',
            'port': 5000,
            'debug': False,
            'use_reloader': False
        })
        server.daemon = True  # Make the server thread a daemon so it exits when main thread exits
        server.start()
        
        # Wait for Flask to start
        time.sleep(3)
        
        # Initialize WebDriver
        if not app.webdriver_manager:
            import requests
            try:
                response = requests.get('http://127.0.0.1:5000/init_webdriver')
            except Exception as e:
                print(f"Failed to initialize WebDriver: {str(e)}")
        
        # Keep the main thread alive but responsive to keyboard interrupts
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)