from flask import request, render_template, json
from static.tool_call_processor import ToolCallProcessor
from static.app_manager import AppManager


def register_routes(app):
    """Register all routes with the Flask application."""
    
    @app.route('/')
    def get_index():
        return render_template('index.html')

    @app.route('/init_webdriver')
    def init_webdriver_route():
        """Route to initialize WebDriver after Flask has started"""
        if not app.webdriver_manager:
            try:
                from static.webdriver_manager import WebDriverManager
                app.webdriver_manager = WebDriverManager()
            except Exception as e:
                print(f"Failed to initialize WebDriverManager: {str(e)}")
                return AppManager.make_response(
                    message=f"WebDriver initialization failed: {str(e)}", 
                    status='error',
                    code=500
                )
        return AppManager.make_response(message="WebDriver initialization successful")

    @app.route('/save_workspace', methods=['POST'])
    def save_workspace_route():
        """Save the current workspace state."""
        try:
            data = request.get_json()
            state = data.get('state')
            name = data.get('name')
            
            success = app.workspace_manager.save_workspace(state, name)
            if success:
                return AppManager.make_response(message='Workspace saved successfully')
            else:
                return AppManager.make_response(
                    message='Failed to save workspace',
                    status='error',
                    code=500
                )
        except Exception as e:
            return AppManager.make_response(
                message=str(e),
                status='error',
                code=500
            )

    @app.route('/load_workspace', methods=['GET'])
    def load_workspace_route():
        """Load a workspace state."""
        try:
            name = request.args.get('name')
            state = app.workspace_manager.load_workspace(name)
            
            return AppManager.make_response(data={'state': state})
        except FileNotFoundError as e:
            return AppManager.make_response(
                message=str(e),
                status='error',
                code=404
            )
        except Exception as e:
            return AppManager.make_response(
                message=str(e),
                status='error',
                code=500
            )

    @app.route('/list_workspaces', methods=['GET'])
    def list_workspaces_route():
        """List all saved workspaces."""
        try:
            workspaces = app.workspace_manager.list_workspaces()
            return AppManager.make_response(data=workspaces)
        except Exception as e:
            return AppManager.make_response(
                message=str(e),
                status='error',
                code=500
            )

    @app.route('/delete_workspace', methods=['GET'])
    def delete_workspace_route():
        """Delete a workspace."""
        try:
            name = request.args.get('name')
            if not name:
                return AppManager.make_response(
                    message='Workspace name is required',
                    status='error',
                    code=400
                )
                
            success = app.workspace_manager.delete_workspace(name)
            if success:
                return AppManager.make_response(message='Workspace deleted successfully')
            else:
                return AppManager.make_response(
                    message='Failed to delete workspace',
                    status='error',
                    code=404
                )
        except Exception as e:
            return AppManager.make_response(
                message=str(e),
                status='error',
                code=500
            )

    @app.route('/send_message', methods=['POST'])
    def send_message():
        message = request.json.get('message')
        if not message:
            return AppManager.make_response(
                message='Message is required',
                status='error',
                code=400
            )

        try:
            message_json = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return AppManager.make_response(
                message='Invalid message format',
                status='error',
                code=400
            )

        svg_state = request.json.get('svg_state')  # Get SVG state from request
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

        try:
            # Proceed with creating chat completion
            response = app.ai_api.create_chat_completion(message)

            ai_message = response.content if response.content is not None else ""
            app.log_manager.log_ai_response(ai_message)

            ai_tool_calls = response.tool_calls if response.tool_calls is not None else []
            ai_tool_calls = ToolCallProcessor.jsonify_tool_calls(ai_tool_calls)
            app.log_manager.log_ai_tool_calls(ai_tool_calls)

            return AppManager.make_response(data={
                "ai_message": ai_message,
                "ai_tool_calls": ai_tool_calls
            })
        except Exception as e:
            return AppManager.make_response(
                message=str(e),
                status='error',
                code=500
            ) 