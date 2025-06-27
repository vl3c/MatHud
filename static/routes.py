"""
MatHud Flask Route Definitions

Defines all Flask application routes for AI communication, workspace management,
WebDriver initialization, and authentication. Handles JSON requests and provides consistent API responses.

Dependencies:
    - flask: Request handling, templating, JSON processing, and session management
    - static.tool_call_processor: OpenAI tool call format conversion
    - static.app_manager: Consistent API response formatting and deployment detection
"""

import functools
import hmac
import time
import math
from flask import request, render_template, json, session, redirect, url_for, flash
from static.tool_call_processor import ToolCallProcessor
from static.app_manager import AppManager

# Global dictionary to track login attempts by IP address
# Format: {ip_address: last_attempt_timestamp}
login_attempts = {}


def require_auth(f):
    """Decorator to require authentication for routes when deployed.
    
    Only enforces authentication when running in deployed environments.
    In development mode, allows unrestricted access.
    
    Args:
        f: The route function to protect
        
    Returns:
        Wrapped function that checks authentication before proceeding
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        # Require authentication when deployed or explicitly enabled
        if AppManager.requires_auth():
            if not session.get('authenticated'):
                return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def register_routes(app):
    """Register all routes with the Flask application.
    
    Configures all application endpoints including main page, AI communication,
    workspace operations, WebDriver management, and authentication routes.
    
    Args:
        app: Flask application instance
    """
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """Handle user authentication with access code."""
        if not AppManager.requires_auth():
            return redirect(url_for('get_index'))
        
        # If user is already authenticated, redirect to main page
        if session.get('authenticated'):
            return redirect(url_for('get_index'))
        
        if request.method == 'POST':
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            current_time = time.time()
            
            pin_submitted = request.form.get('pin', '')
            auth_pin = AppManager.get_auth_pin()
            
            if not auth_pin:
                flash('Authentication not configured')
                return render_template('login.html')

            is_pin_correct = hmac.compare_digest(pin_submitted, auth_pin)

            # --- NEW, CORRECT LOGIC ---

            if is_pin_correct:
                # 1. PIN is correct. Login succeeds immediately.
                session['authenticated'] = True
                login_attempts.pop(client_ip, None) # Clear any old rate limit.
                return redirect(url_for('get_index'))
            else:
                # 2. PIN is incorrect. Now we handle rate limiting.
                if client_ip in login_attempts:
                    time_since_last_failed = current_time - login_attempts[client_ip]
                    
                    if time_since_last_failed < 5.0:
                        # 2a. Cooldown is ACTIVE. Block and show countdown.
                        remaining_cooldown = 5.0 - time_since_last_failed
                        display_time = math.ceil(remaining_cooldown)
                        flash(f'Too many attempts. Please wait {display_time} seconds.')
                        return render_template('login.html')

                # 2b. PIN was wrong, but no active cooldown. Start a new one.
                login_attempts[client_ip] = current_time
                flash('Invalid access code')
                
                # Cleanup logic (runs only after a failed attempt)
                if len(login_attempts) > 1000:
                    cutoff = current_time - 3600 # 1 hour
                    for ip, timestamp in list(login_attempts.items()):
                        if timestamp < cutoff:
                            del login_attempts[ip]
                
                return render_template('login.html')

        # For GET request
        return render_template('login.html')
    
    @app.route('/logout')
    def logout():
        """Handle user logout and session cleanup."""
        session.pop('authenticated', None)
        if AppManager.requires_auth():
            return redirect(url_for('login'))
        return redirect(url_for('get_index'))
    
    @app.route('/auth_status')
    def auth_status():
        """Return authentication status information."""
        return AppManager.make_response(data={
            'auth_required': AppManager.requires_auth(),
            'authenticated': session.get('authenticated', False)
        })
    
    @app.route('/')
    @require_auth
    def get_index():
        return render_template('index.html')

    @app.route('/init_webdriver')
    @require_auth
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
    @require_auth
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
    @require_auth
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
    @require_auth
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
    @require_auth
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

    @app.route('/new_conversation', methods=['POST'])
    @require_auth
    def new_conversation_route():
        """Reset the AI conversation history for a new session."""
        try:
            app.ai_api.reset_conversation()
            app.log_manager.log_new_session()
            return AppManager.make_response(message='New conversation started.')
        except Exception as e:
            return AppManager.make_response(
                message=str(e),
                status='error',
                code=500
            )

    def _process_ai_response(app, choice):
        """Process the AI response choice and log the results.
        
        Args:
            app: The Flask application instance
            choice: The AI response choice object
            
        Returns:
            tuple: (ai_message, ai_tool_calls) - The processed message and tool calls
        """
        ai_message = choice.message.content if choice.message.content is not None else ""
        app.log_manager.log_ai_response(ai_message)

        ai_tool_calls = choice.message.tool_calls if choice.message.tool_calls is not None else []
        ai_tool_calls = ToolCallProcessor.jsonify_tool_calls(ai_tool_calls)
        app.log_manager.log_ai_tool_calls(ai_tool_calls)
        
        return ai_message, ai_tool_calls

    @app.route('/send_message', methods=['POST'])
    @require_auth
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
            choice = app.ai_api.create_chat_completion(message)
            
            # Process and log the AI response
            ai_message, ai_tool_calls = _process_ai_response(app, choice)

            return AppManager.make_response(data={
                "ai_message": ai_message,
                "ai_tool_calls": ai_tool_calls,
                "finish_reason": choice.finish_reason
            })
        except Exception as e:
            return AppManager.make_response(
                message=str(e),
                status='error',
                code=500
            ) 