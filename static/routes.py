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

import base64
import functools
import hmac
import json
import math
import os
import time
from collections.abc import Callable, Iterator
from typing import Any, Dict, List, Optional, TypeVar, Union, cast

from flask import Response, flash, redirect, render_template, request, session, stream_with_context, url_for
from flask.typing import ResponseReturnValue

from static.app_manager import AppManager, MatHudFlask
from static.openai_api_base import OpenAIAPIBase
from static.tool_call_processor import ProcessedToolCall, ToolCallProcessor
from static.webdriver_manager import SvgState

F = TypeVar("F", bound=Callable[..., ResponseReturnValue])

# Global dictionary to track login attempts by IP address
# Format: {ip_address: last_attempt_timestamp}
JsonValue = Union[str, int, float, bool, None, Dict[str, "JsonValue"], List["JsonValue"]]
JsonObject = Dict[str, JsonValue]
StreamEventDict = Dict[str, JsonValue]

login_attempts: Dict[str, float] = {}
ToolCallList = List[ProcessedToolCall]
CANVAS_SNAPSHOT_DIR = "canvas_snapshots"
CANVAS_SNAPSHOT_PATH = os.path.join(CANVAS_SNAPSHOT_DIR, "canvas.png")


def save_canvas_snapshot_from_data_url(data_url: str) -> bool:
    if not isinstance(data_url, str):
        return False
    parts = data_url.split(",", 1)
    if len(parts) != 2:
        return False
    metadata, encoded = parts
    metadata = metadata.strip().lower()
    if not metadata.startswith("data:image"):
        return False
    try:
        image_bytes = base64.b64decode(encoded)
    except Exception as exc:
        print(f"Failed to decode canvas snapshot: {exc}")
        return False
    try:
        os.makedirs(CANVAS_SNAPSHOT_DIR, exist_ok=True)
        with open(CANVAS_SNAPSHOT_PATH, "wb") as snapshot_file:
            snapshot_file.write(image_bytes)
        return True
    except Exception as exc:
        print(f"Failed to write canvas snapshot: {exc}")
        return False


def extract_vision_payload(request_payload: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    svg_state: Optional[Dict[str, Any]] = None
    canvas_image: Optional[str] = None
    renderer_mode: Optional[str] = None

    raw_svg_state = request_payload.get('svg_state')
    if isinstance(raw_svg_state, dict):
        svg_state = raw_svg_state
    raw_renderer = request_payload.get('renderer_mode')
    if isinstance(raw_renderer, str):
        renderer_mode = raw_renderer

    vision_snapshot = request_payload.get('vision_snapshot')
    if isinstance(vision_snapshot, dict):
        snapshot_svg = vision_snapshot.get('svg_state')
        if isinstance(snapshot_svg, dict):
            svg_state = snapshot_svg
        snapshot_renderer = vision_snapshot.get('renderer_mode')
        if isinstance(snapshot_renderer, str):
            renderer_mode = snapshot_renderer
        snapshot_canvas = vision_snapshot.get('canvas_image')
        if isinstance(snapshot_canvas, str):
            canvas_image = snapshot_canvas

    return svg_state, canvas_image, renderer_mode


def handle_vision_capture(
    app: MatHudFlask,
    use_vision: bool,
    svg_state: Optional[Dict[str, Any]],
    canvas_image: Optional[str],
    init_webdriver: Callable[[], ResponseReturnValue],
) -> None:
    if not use_vision:
        return
    if canvas_image and save_canvas_snapshot_from_data_url(canvas_image):
        return

    if svg_state is None:
        return

    if app.webdriver_manager is None:
        try:
            init_webdriver()
        except Exception as exc:
            print(f"Failed to initialize WebDriver for vision capture: {exc}")

    if app.webdriver_manager is not None:
        try:
            app.webdriver_manager.capture_svg_state(cast(SvgState, svg_state))
        except Exception as exc:
            print(f"WebDriver capture failed: {exc}")


def _maybe_inject_search_tools(api: OpenAIAPIBase, tool_call_results: str) -> None:
    """Inject tools if search_tools was called in the previous turn.

    Parses the tool_call_results JSON and looks for a search_tools result.
    If found, extracts the tools array and injects them into the API client.

    Args:
        api: The OpenAI API instance to inject tools into.
        tool_call_results: JSON string containing tool call results.
    """
    try:
        results = json.loads(tool_call_results)
        if not isinstance(results, dict):
            return

        # Look for search_tools result - it has both "tools" and "query" keys
        for tool_id, result in results.items():
            if isinstance(result, dict) and "tools" in result and "query" in result:
                # This is a search_tools result
                tools = result.get("tools", [])
                if tools and isinstance(tools, list):
                    api.inject_tools(tools, include_essentials=True)
                    return
    except (json.JSONDecodeError, TypeError):
        pass


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
        
        Routes to appropriate API based on model type:
        - Reasoning models (GPT-5, o3, o4-mini): Uses Responses API with reasoning streaming
        - Standard models (GPT-4.1, GPT-4o): Uses Chat Completions API
        
        Returns a newline-delimited JSON stream with events of shape:
        {"type":"reasoning","text":"..."}\n for reasoning tokens (reasoning models only)
        {"type":"token","text":"..."}\n for incremental response tokens
        {"type":"final","ai_message":str,"ai_tool_calls":list,"finish_reason":str}\n at the end.
        """
        request_payload_raw: JsonValue = request.get_json(silent=True)
        if not isinstance(request_payload_raw, dict):
            return AppManager.make_response(
                message='Invalid request body',
                status='error',
                code=400,
            )
        request_payload: JsonObject = request_payload_raw

        message = request_payload.get('message')
        if not isinstance(message, str) or not message:
            return AppManager.make_response(
                message='Message is required',
                status='error',
                code=400,
            )

        try:
            message_json_value: JsonValue = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return AppManager.make_response(
                message='Invalid message format',
                status='error',
                code=400,
            )

        if not isinstance(message_json_value, dict):
            return AppManager.make_response(
                message='Invalid message format',
                status='error',
                code=400,
            )
        message_json: JsonObject = message_json_value

        svg_state, canvas_image_data, _ = extract_vision_payload(request_payload)
        use_vision = bool(message_json.get('use_vision', False))
        ai_model_raw = message_json.get('ai_model')
        ai_model = ai_model_raw if isinstance(ai_model_raw, str) else None

        if ai_model:
            app.ai_api.set_model(ai_model)
            app.responses_api.set_model(ai_model)

        app.log_manager.log_user_message(message)

        handle_vision_capture(
            app,
            use_vision,
            svg_state if isinstance(svg_state, dict) else None,
            canvas_image_data,
            init_webdriver_route,
        )

        # Check for search_tools results and inject tools if found
        tool_call_results_raw = message_json.get('tool_call_results')
        if isinstance(tool_call_results_raw, str) and tool_call_results_raw:
            _maybe_inject_search_tools(app.ai_api, tool_call_results_raw)
            _maybe_inject_search_tools(app.responses_api, tool_call_results_raw)

        @stream_with_context
        def generate() -> Iterator[str]:
            try:
                # Route to appropriate API based on model type
                model = app.ai_api.get_model()
                if model.is_reasoning_model:
                    stream = app.responses_api.create_response_stream(message)
                else:
                    stream = app.ai_api.create_chat_completion_stream(message)

                for event in stream:
                    if isinstance(event, dict):
                        event_dict = cast(StreamEventDict, event)
                        if event_dict.get('type') == 'final':
                            try:
                                app.log_manager.log_ai_response(str(event_dict.get('ai_message', '')))
                                tool_calls = event_dict.get('ai_tool_calls')
                                if isinstance(tool_calls, list):
                                    dict_tool_calls: List[Dict[str, Any]] = [
                                        cast(Dict[str, Any], call)
                                        for call in tool_calls
                                        if isinstance(call, dict)
                                    ]
                                    if dict_tool_calls:
                                        app.log_manager.log_ai_tool_calls(dict_tool_calls)
                            except Exception:
                                pass
                            # Reset tools if AI finished (not requesting more tool calls)
                            finish_reason = event_dict.get('finish_reason')
                            if finish_reason != 'tool_calls':
                                if app.ai_api.has_injected_tools():
                                    app.ai_api.reset_tools()
                                if app.responses_api.has_injected_tools():
                                    app.responses_api.reset_tools()
                        yield json.dumps(event_dict) + "\n"
                    else:
                        yield json.dumps(event) + "\n"
            except Exception as exc:
                print(f"[Routes /send_message] Streaming exception: {exc}")
                # Reset tools on error
                if app.ai_api.has_injected_tools():
                    app.ai_api.reset_tools()
                if app.responses_api.has_injected_tools():
                    app.responses_api.reset_tools()
                error_payload: StreamEventDict = {
                    "type": "final",
                    "ai_message": "I encountered an error processing your request. Please try again.",
                    "ai_tool_calls": [],
                    "finish_reason": "error",
                }
                try:
                    yield json.dumps(error_payload) + "\n"
                except Exception:
                    print("[Routes /send_message] Failed to send detailed error payload; falling back.")
                    fallback_payload: StreamEventDict = {
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
            return AppManager.make_response(data=cast(JsonValue, workspaces))
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
            app.responses_api.reset_conversation()
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
            tool_calls = ToolCallProcessor.jsonify_tool_calls(raw_tool_calls)
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

        svg_state, canvas_image_data, _ = extract_vision_payload(request_payload)
        use_vision = bool(message_json_raw.get('use_vision', False))
        ai_model_raw = message_json_raw.get('ai_model')
        ai_model = ai_model_raw if isinstance(ai_model_raw, str) else None

        if ai_model:
            app.ai_api.set_model(ai_model)
            app.responses_api.set_model(ai_model)

        app.log_manager.log_user_message(message)

        handle_vision_capture(
            app,
            use_vision,
            svg_state if isinstance(svg_state, dict) else None,
            canvas_image_data,
            init_webdriver_route,
        )

        # Check for search_tools results and inject tools if found
        tool_call_results_raw = message_json_raw.get('tool_call_results')
        if isinstance(tool_call_results_raw, str) and tool_call_results_raw:
            _maybe_inject_search_tools(app.ai_api, tool_call_results_raw)
            _maybe_inject_search_tools(app.responses_api, tool_call_results_raw)

        def _reset_tools_if_needed(finish_reason: Any) -> None:
            """Reset tools if AI finished (not requesting more tool calls)."""
            if finish_reason != 'tool_calls':
                if app.ai_api.has_injected_tools():
                    app.ai_api.reset_tools()
                if app.responses_api.has_injected_tools():
                    app.responses_api.reset_tools()

        try:
            # Route to appropriate API based on model type. This matters for
            # streaming fallback (client retries via /send_message).
            model = app.ai_api.get_model()
            if model.is_reasoning_model:
                stream = app.responses_api.create_response_stream(message)
                final_event: Optional[StreamEventDict] = None
                for event in stream:
                    if isinstance(event, dict) and event.get("type") == "final":
                        final_event = cast(StreamEventDict, event)
                        break

                if final_event is None:
                    _reset_tools_if_needed('error')
                    return AppManager.make_response(
                        message="No final response event produced",
                        status="error",
                        code=500,
                    )

                ai_message = str(final_event.get("ai_message", ""))
                ai_tool_calls_raw = final_event.get("ai_tool_calls")
                ai_tool_calls: List[Dict[str, Any]] = (
                    [cast(Dict[str, Any], call) for call in ai_tool_calls_raw if isinstance(call, dict)]
                    if isinstance(ai_tool_calls_raw, list)
                    else []
                )
                finish_reason = final_event.get("finish_reason")

                app.log_manager.log_ai_response(ai_message)
                app.log_manager.log_ai_tool_calls(ai_tool_calls)

                _reset_tools_if_needed(finish_reason)
                return AppManager.make_response(data=cast(JsonObject, {
                    "ai_message": ai_message,
                    "ai_tool_calls": cast(JsonValue, ai_tool_calls),
                    "finish_reason": finish_reason,
                }))

            choice = app.ai_api.create_chat_completion(message)
            ai_message, ai_tool_calls = _process_ai_response(app, choice)
            finish_reason = getattr(choice, 'finish_reason', None)

            _reset_tools_if_needed(finish_reason)
            return AppManager.make_response(data=cast(JsonObject, {
                "ai_message": ai_message,
                "ai_tool_calls": cast(JsonValue, ai_tool_calls),
                "finish_reason": finish_reason,
            }))
        except Exception as exc:
            _reset_tools_if_needed('error')
            return AppManager.make_response(
                message=str(exc),
                status='error',
                code=500,
            )

    @app.route('/search_tools', methods=['POST'])
    @require_auth
    def search_tools_route() -> ResponseReturnValue:
        """Search for tools matching a query description.
        
        Uses AI-powered semantic matching to find the most relevant tools
        for a given task description.
        
        Request body:
            query (str): Description of what the user wants to accomplish
            max_results (int, optional): Maximum number of tools to return (default: 10)
            
        Returns:
            JSON response with matching tool definitions
        """
        from static.tool_search_service import ToolSearchService

        request_payload = request.get_json(silent=True)
        if not isinstance(request_payload, dict):
            return AppManager.make_response(
                message='Invalid request body',
                status='error',
                code=400,
            )

        query = request_payload.get('query')
        if not isinstance(query, str) or not query.strip():
            return AppManager.make_response(
                message='Query is required',
                status='error',
                code=400,
            )

        max_results_raw = request_payload.get('max_results')
        max_results = 10  # default
        if isinstance(max_results_raw, int):
            max_results = max_results_raw
        elif isinstance(max_results_raw, float):
            max_results = int(max_results_raw)

        try:
            service = ToolSearchService(client=app.ai_api.client)
            result = service.search_tools_formatted(
                query=query,
                max_results=max_results,
            )
            return AppManager.make_response(data=cast(JsonValue, result))
        except Exception as exc:
            return AppManager.make_response(
                message=str(exc),
                status='error',
                code=500,
            )