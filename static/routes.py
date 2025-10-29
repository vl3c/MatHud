"""
MatHud Flask Route Definitions

Defines all Flask application routes for AI communication, workspace management,
WebDriver initialization, and authentication. Handles JSON requests and provides consistent API responses.

Dependencies:
    - flask: Request handling, templating, JSON processing, and session management
    - static.tool_call_processor: OpenAI tool call format conversion
    - static.app_manager: Consistent API response formatting and deployment detection
"""

from __future__ import annotations

import functools
import hmac
import json
import math
import time
from collections.abc import Callable, Iterator
from typing import Any, Dict, List, Optional, TypeVar, cast

from flask import Response, flash, redirect, render_template, request, session, stream_with_context, url_for
from flask.typing import ResponseReturnValue

from static.app_manager import AppManager, MatHudFlask
from static.tool_call_processor import ToolCallProcessor

F = TypeVar("F", bound=Callable[..., ResponseReturnValue])

# Global dictionary to track login attempts by IP address
# Format: {ip_address: last_attempt_timestamp}
login_attempts: Dict[str, float] = {}
ToolCallList = List[Dict[str, Any]]


def require_auth(f: F) -> F:
    """Decorator to require authentication for routes when deployed.
    
    Only enforces authentication when running in deployed environments.
    In development mode, allows unrestricted access.
    
    Args:
        f: The route function to protect
        
    Returns:
        Wrapped function that checks authentication before proceeding
    """
    @functools.wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> ResponseReturnValue:
        # Require authentication when deployed or explicitly enabled
        if AppManager.requires_auth():
            if not session.get('authenticated'):
                return redirect(url_for('login'))
        return f(*args, **kwargs)

    return cast(F, decorated_function)


def register_routes(app: MatHudFlask) -> None:
    """Register all routes with the Flask application.
    
    Configures all application endpoints including main page, AI communication,
    workspace operations, WebDriver management, and authentication routes.
    
    Args:
        app: Flask application instance
    """
    
    @app.route('/login', methods=['GET', 'POST'])
    def login() -> ResponseReturnValue:
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
    def logout() -> ResponseReturnValue:
        """Handle user logout and session cleanup."""
        session.pop('authenticated', None)
        if AppManager.requires_auth():
            return redirect(url_for('login'))
        return redirect(url_for('get_index'))
    
    @app.route('/auth_status')
    def auth_status() -> ResponseReturnValue:
        """Return authentication status information."""
        return AppManager.make_response(data={
            'auth_required': AppManager.requires_auth(),
            'authenticated': session.get('authenticated', False)
        })
    
    @app.route('/')
    @require_auth
    def get_index() -> ResponseReturnValue:
        return render_template('index.html')

    @app.route('/init_webdriver')
    @require_auth
    def init_webdriver_route() -> ResponseReturnValue:
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
    def save_workspace_route() -> ResponseReturnValue:
        """Save the current workspace state."""
        try:
            data = request.get_json(silent=True)
            if not isinstance(data, dict):
                return AppManager.make_response(
                    message='Invalid request body',
                    status='error',
                    code=400
                )

            state = data.get('state')
            name = data.get('name')

            if name is not None and not isinstance(name, str):
                return AppManager.make_response(
                    message='Workspace name must be a string',
                    status='error',
                    code=400
                )
            
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

    @app.route('/send_message_stream', methods=['POST'])
    @require_auth
    def send_message_stream() -> ResponseReturnValue:
        """Stream AI response tokens for the provided message payload.
        
        Returns a newline-delimited JSON stream with events of shape:
        {"type":"token","text":"..."}\n for incremental tokens and
        {"type":"final","ai_message":str,"ai_tool_calls":list,"finish_reason":str}\n at the end.
        """
        request_payload = request.get_json(silent=True)
        if not isinstance(request_payload, dict):
            return AppManager.make_response(
                message='Invalid request body',
                status='error',
                code=400,
            )

        message = request_payload.get('message')
        if not isinstance(message, str) or not message:
            return AppManager.make_response(
                message='Message is required',
                status='error',
                code=400,
            )

        try:
            message_json_raw = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return AppManager.make_response(
                message='Invalid message format',
                status='error',
                code=400,
            )

        if not isinstance(message_json_raw, dict):
            return AppManager.make_response(
                message='Invalid message format',
                status='error',
                code=400,
            )

        svg_state = request_payload.get('svg_state')
        use_vision = bool(message_json_raw.get('use_vision', False))
        ai_model_raw = message_json_raw.get('ai_model')
        ai_model = ai_model_raw if isinstance(ai_model_raw, str) else None

        if ai_model:
            app.ai_api.set_model(ai_model)

        app.log_manager.log_user_message(message)

        if use_vision and app.webdriver_manager is None:
            try:
                init_webdriver_route()
            except Exception:
                pass

        if use_vision and app.webdriver_manager is not None:
            try:
                app.webdriver_manager.capture_svg_state(svg_state)
            except Exception:
                pass

        @stream_with_context
        def generate() -> Iterator[str]:
            try:
                for event in app.ai_api.create_chat_completion_stream(message):
                    if isinstance(event, dict) and event.get('type') == 'final':
                        try:
                            app.log_manager.log_ai_response(str(event.get('ai_message', '')))
                            tool_calls = event.get('ai_tool_calls')
                            if isinstance(tool_calls, list):
                                app.log_manager.log_ai_tool_calls(tool_calls)
                        except Exception:
                            pass
                    yield json.dumps(event) + "\n"
            except Exception as exc:
                error_payload = {
                    "type": "final",
                    "ai_message": f"I encountered an error processing your request: {exc}",
                    "ai_tool_calls": [],
                    "finish_reason": "error",
                }
                try:
                    yield json.dumps(error_payload) + "\n"
                except Exception:
                    fallback_payload = {
                        "type": "final",
                        "ai_message": "I encountered an error processing your request.",
                        "ai_tool_calls": [],
                        "finish_reason": "error",
                    }
                    yield json.dumps(fallback_payload) + "\n"

        response = Response(generate(), mimetype='application/x-ndjson')
        # Headers to reduce buffering in some proxies
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Accel-Buffering'] = 'no'
        return response

    @app.route('/load_workspace', methods=['GET'])
    @require_auth
    def load_workspace_route() -> ResponseReturnValue:
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
    def list_workspaces_route() -> ResponseReturnValue:
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
    def delete_workspace_route() -> ResponseReturnValue:
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
    def new_conversation_route() -> ResponseReturnValue:
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

    def _process_ai_response(app: MatHudFlask, choice: Any) -> tuple[str, ToolCallList]:
        """Process the AI response choice and log the results.
        
        Args:
            app: The Flask application instance
            choice: The AI response choice object
            
        Returns:
            tuple: (ai_message, ai_tool_calls) - The processed message and tool calls
        """
        message_obj = getattr(choice, "message", None)
        message_content = getattr(message_obj, "content", "")
        ai_message = message_content if isinstance(message_content, str) else ""
        app.log_manager.log_ai_response(ai_message)

        raw_tool_calls = getattr(message_obj, "tool_calls", None)
        tool_calls: ToolCallList = []
        if raw_tool_calls:
            tool_calls = cast(ToolCallList, ToolCallProcessor.jsonify_tool_calls(raw_tool_calls))
            app.log_manager.log_ai_tool_calls(tool_calls)
        else:
            app.log_manager.log_ai_tool_calls([])

        return ai_message, tool_calls

    @app.route('/send_message', methods=['POST'])
    @require_auth
    def send_message() -> ResponseReturnValue:
        request_payload = request.get_json(silent=True)
        if not isinstance(request_payload, dict):
            return AppManager.make_response(
                message='Invalid request body',
                status='error',
                code=400,
            )

        message = request_payload.get('message')
        if not isinstance(message, str) or not message:
            return AppManager.make_response(
                message='Message is required',
                status='error',
                code=400,
            )

        try:
            message_json_raw = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return AppManager.make_response(
                message='Invalid message format',
                status='error',
                code=400,
            )

        if not isinstance(message_json_raw, dict):
            return AppManager.make_response(
                message='Invalid message format',
                status='error',
                code=400,
            )

        svg_state = request_payload.get('svg_state')
        use_vision = bool(message_json_raw.get('use_vision', False))
        ai_model_raw = message_json_raw.get('ai_model')
        ai_model = ai_model_raw if isinstance(ai_model_raw, str) else None

        if ai_model:
            app.ai_api.set_model(ai_model)

        app.log_manager.log_user_message(message)

        if use_vision and app.webdriver_manager is None:
            print("WebDriver not found, attempting to initialize...")
            init_webdriver_route()

        if use_vision and app.webdriver_manager is not None:
            app.webdriver_manager.capture_svg_state(svg_state)

        try:
            choice = app.ai_api.create_chat_completion(message)
            ai_message, ai_tool_calls = _process_ai_response(app, choice)
            finish_reason = getattr(choice, 'finish_reason', None)

            return AppManager.make_response(data={
                "ai_message": ai_message,
                "ai_tool_calls": ai_tool_calls,
                "finish_reason": finish_reason,
            })
        except Exception as exc:
            return AppManager.make_response(
                message=str(exc),
                status='error',
                code=500,
            )