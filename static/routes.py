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

from flask import Response, flash, jsonify, redirect, render_template, request, session, stream_with_context, url_for
from flask.typing import ResponseReturnValue

from static.ai_model import AIModel, PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_OPENROUTER, PROVIDER_OLLAMA
from static.app_manager import AppManager, MatHudFlask
from static.canvas_state_summarizer import compare_canvas_states
from static.openai_api_base import OpenAIAPIBase
from static.providers import ProviderRegistry, create_provider_instance
from static.tool_call_processor import ProcessedToolCall, ToolCallProcessor
from static.tts_manager import get_tts_manager
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


def get_provider_for_model(app: MatHudFlask, model_id: str) -> OpenAIAPIBase:
    """Get or create the appropriate provider instance for a model.

    Args:
        app: The Flask application instance
        model_id: The model identifier

    Returns:
        The API provider instance for the model
    """
    model = AIModel.from_identifier(model_id)
    provider_name = model.provider

    # For OpenAI, use the existing api instances
    if provider_name == PROVIDER_OPENAI:
        if model.is_reasoning_model:
            return app.responses_api
        return app.ai_api

    # For other providers, use lazy-loaded instances
    if provider_name not in app.providers:
        create_kwargs: Dict[str, Any] = {"model": model}
        # Keep providers in search-first mode by default to reduce initial tool payload.
        if provider_name != PROVIDER_OLLAMA:
            create_kwargs["tool_mode"] = "search"

        provider_instance = create_provider_instance(provider_name, **create_kwargs)
        if provider_instance is None:
            raise ValueError(
                f"Provider '{provider_name}' is not available. "
                f"Check that the API key is configured in .env."
            )
        app.providers[provider_name] = provider_instance

    # Update the model on the cached provider instance
    provider = app.providers[provider_name]
    provider.set_model(model_id)
    return provider


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


def extract_vision_payload(
    request_payload: Dict[str, Any]
) -> tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[List[str]]]:
    """Extract vision-related data from the request payload.

    Args:
        request_payload: The full request payload dictionary

    Returns:
        Tuple of (svg_state, canvas_image, renderer_mode, attached_images)
    """
    svg_state: Optional[Dict[str, Any]] = None
    canvas_image: Optional[str] = None
    renderer_mode: Optional[str] = None
    attached_images: Optional[List[str]] = None

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

    # Extract attached images from the request payload
    raw_attached_images = request_payload.get('attached_images')
    if isinstance(raw_attached_images, list):
        attached_images = [img for img in raw_attached_images if isinstance(img, str)]

    return svg_state, canvas_image, renderer_mode, attached_images


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


def _intercept_search_tools(
    app: MatHudFlask,
    tool_calls: List[Dict[str, Any]],
    provider: Optional[OpenAIAPIBase] = None,
) -> List[Dict[str, Any]]:
    """Intercept search_tools calls and filter other tool calls.

    When search_tools is called alongside other tools, this function:
    1. Executes search_tools server-side
    2. Injects the returned tools
    3. Filters out tool calls not in the allowed set

    Args:
        app: The Flask application instance.
        tool_calls: List of tool calls from the AI response.
        provider: The current API provider (uses its client/model for search).

    Returns:
        Filtered list of tool calls (only allowed tools).
    """
    from static.tool_search_service import ToolSearchService
    from static.openai_api_base import ESSENTIAL_TOOLS

    search_tools_call = _find_search_tools_call(tool_calls)

    if search_tools_call is None:
        return tool_calls  # No search_tools, return as-is

    query, max_results = _extract_search_query_and_limit(search_tools_call)

    if not query:
        return tool_calls  # No query, return as-is

    # Execute search_tools server-side using the current provider's client/model
    try:
        active_provider = provider or app.ai_api
        service = ToolSearchService(
            client=active_provider.client,
            default_model=active_provider.get_model(),
        )
        result = service.search_tools(query=query, max_results=max_results or 10)

        allowed_names = _collect_allowed_tool_names(result, ESSENTIAL_TOOLS)

        # Inject tools into shared APIs and the active provider (if distinct).
        if result:
            app.ai_api.inject_tools(result, include_essentials=True)
            app.responses_api.inject_tools(result, include_essentials=True)
            if provider is not None and provider not in (app.ai_api, app.responses_api):
                provider.inject_tools(result, include_essentials=True)

        return _filter_tool_calls_by_allowed_names(tool_calls, allowed_names)

    except Exception:
        return tool_calls  # On error, return original calls


def _tool_call_name(call: Dict[str, Any]) -> Optional[str]:
    """Extract a tool call function name from normalized or nested shapes."""
    function_name = call.get('function_name')
    if isinstance(function_name, str):
        return function_name
    function = call.get('function')
    if isinstance(function, dict):
        nested_name = function.get('name')
        if isinstance(nested_name, str):
            return nested_name
    return None


def _find_search_tools_call(tool_calls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the first search_tools call, if any."""
    for call in tool_calls:
        if _tool_call_name(call) == 'search_tools':
            return call
    return None


def _normalize_search_tools_args(call: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize search_tools arguments to a dictionary."""
    args = call.get('arguments', {})
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            loaded = json.loads(args)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass
    return {}


def _extract_search_query_and_limit(call: Dict[str, Any]) -> tuple[str, int]:
    """Extract (query, max_results) from a search_tools call."""
    args = _normalize_search_tools_args(call)
    query = args.get('query', '')
    if not isinstance(query, str):
        query = ''
    raw_max_results = args.get('max_results', 10)
    if isinstance(raw_max_results, int):
        max_results = raw_max_results
    else:
        max_results = 10
    return query, max_results


def _collect_allowed_tool_names(
    search_result: List[Dict[str, Any]],
    essential_tools: set[str],
) -> set[str]:
    """Collect allowed tool names from search result plus essentials."""
    allowed_names = {
        t.get('function', {}).get('name')
        for t in search_result
        if isinstance(t, dict)
    }
    allowed_names.update(essential_tools)
    return allowed_names


def _filter_tool_calls_by_allowed_names(
    tool_calls: List[Dict[str, Any]],
    allowed_names: set[str],
) -> List[Dict[str, Any]]:
    """Filter tool calls by allowed function names."""
    filtered_calls: List[Dict[str, Any]] = []
    for call in tool_calls:
        if _tool_call_name(call) in allowed_names:
            filtered_calls.append(call)
    return filtered_calls


def _maybe_inject_search_tools(api: OpenAIAPIBase, tool_call_results: str) -> None:
    """Inject tools if search_tools was called in the previous turn.

    Parses the tool_call_results JSON and looks for a search_tools result.
    If found, extracts the tools array and injects them into the API client.

    Args:
        api: The OpenAI API instance to inject tools into.
        tool_call_results: JSON string containing tool call results.
    """
    tools = _extract_injectable_tools(tool_call_results)
    if tools:
        api.inject_tools(tools, include_essentials=True)


def _extract_injectable_tools(tool_call_results: str) -> Optional[List[Dict[str, Any]]]:
    """Extract tools payload from a search_tools tool_call_results JSON string."""
    try:
        results = json.loads(tool_call_results)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(results, dict):
        return None

    for result in results.values():
        if not (isinstance(result, dict) and "tools" in result and "query" in result):
            continue
        tools = result.get("tools")
        if isinstance(tools, list) and tools:
            return [cast(Dict[str, Any], tool) for tool in tools if isinstance(tool, dict)]
    return None


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

    @app.route('/api/available_models', methods=['GET'])
    @require_auth
    def get_available_models() -> ResponseReturnValue:
        """Return models grouped by provider, only for available providers."""
        available = ProviderRegistry.get_available_providers()

        # Refresh local provider models if available
        if PROVIDER_OLLAMA in available:
            AIModel.refresh_local_models(PROVIDER_OLLAMA)

        models_by_provider: Dict[str, List[Dict[str, Any]]] = {
            "openai": [],
            "anthropic": [],
            "openrouter_paid": [],
            "openrouter_free": [],
            "ollama": [],
        }

        for model_id, config in AIModel.MODEL_CONFIGS.items():
            provider = config.get("provider", PROVIDER_OPENAI)
            has_vision = config.get("has_vision", False)
            display_name = config.get("display_name", model_id)

            if provider not in available:
                continue

            entry: Dict[str, Any] = {
                "id": model_id,
                "has_vision": has_vision,
                "display_name": display_name,
            }

            if provider == PROVIDER_OPENAI:
                models_by_provider["openai"].append(entry)
            elif provider == PROVIDER_ANTHROPIC:
                models_by_provider["anthropic"].append(entry)
            elif provider == PROVIDER_OPENROUTER:
                if model_id.endswith(":free"):
                    models_by_provider["openrouter_free"].append(entry)
                else:
                    models_by_provider["openrouter_paid"].append(entry)
            elif provider == PROVIDER_OLLAMA:
                models_by_provider["ollama"].append(entry)

        return jsonify(models_by_provider)

    @app.route('/api/preload_model', methods=['POST'])
    @require_auth
    def preload_model() -> ResponseReturnValue:
        """Preload an Ollama model into memory.

        Request body:
            model_id (str): The model identifier to preload

        Returns:
            JSON response with success status and message
        """
        from static.providers.local.ollama_api import OllamaAPI

        request_payload = request.get_json(silent=True)
        if not isinstance(request_payload, dict):
            return AppManager.make_response(
                message='Invalid request body',
                status='error',
                code=400,
            )

        model_id = request_payload.get('model_id')
        if not isinstance(model_id, str) or not model_id:
            return AppManager.make_response(
                message='model_id is required',
                status='error',
                code=400,
            )

        # Only preload Ollama models
        model = AIModel.from_identifier(model_id)
        if model.provider != PROVIDER_OLLAMA:
            return AppManager.make_response(
                message='Model preloading only supported for Ollama models',
                status='error',
                code=400,
            )

        # Check if server is running
        if not OllamaAPI.is_server_running():
            return AppManager.make_response(
                message='Ollama server is not running',
                status='error',
                code=503,
            )

        # Check if already loaded
        if OllamaAPI.is_model_loaded(model_id):
            return AppManager.make_response(
                data={'already_loaded': True},
                message=f'Model {model_id} is already loaded',
            )

        # Preload the model (this may take a while)
        success, message = OllamaAPI.preload_model(model_id)

        if success:
            return AppManager.make_response(
                data={'already_loaded': False},
                message=message,
            )
        else:
            return AppManager.make_response(
                message=message,
                status='error',
                code=500,
            )

    @app.route('/api/model_status', methods=['GET'])
    @require_auth
    def get_model_status() -> ResponseReturnValue:
        """Get the loading status of Ollama models.

        Query params:
            model_id (str, optional): Specific model to check

        Returns:
            JSON response with loaded models or specific model status
        """
        from static.providers.local.ollama_api import OllamaAPI

        model_id = request.args.get('model_id')

        if not OllamaAPI.is_server_running():
            return AppManager.make_response(
                data={'server_running': False, 'loaded_models': []},
            )

        loaded_models = OllamaAPI.get_loaded_models()

        if model_id:
            is_loaded = OllamaAPI.is_model_loaded(model_id)
            return AppManager.make_response(
                data={
                    'server_running': True,
                    'model_id': model_id,
                    'is_loaded': is_loaded,
                    'loaded_models': loaded_models,
                },
            )

        return AppManager.make_response(
            data={
                'server_running': True,
                'loaded_models': loaded_models,
            },
        )

    @app.route('/api/debug/conversation', methods=['GET'])
    @require_auth
    def debug_conversation() -> ResponseReturnValue:
        """Debug endpoint to view conversation history for all providers.

        Returns the message history for debugging purposes.
        """
        provider_name = request.args.get('provider')

        def summarize_message(msg: Dict[str, Any]) -> Dict[str, Any]:
            """Summarize a message for display, truncating long content."""
            summary: Dict[str, Any] = {"role": msg.get("role", "unknown")}

            content = msg.get("content", "")
            if isinstance(content, str):
                summary["content"] = content[:500] + "..." if len(content) > 500 else content
            elif isinstance(content, list):
                summary["content"] = f"[{len(content)} content blocks]"
            else:
                summary["content"] = str(content)[:200]

            if "tool_calls" in msg:
                tool_calls = msg["tool_calls"]
                summary["tool_calls"] = [
                    {
                        "id": tc.get("id", "?"),
                        "name": tc.get("function", {}).get("name", "?"),
                        "args_preview": tc.get("function", {}).get("arguments", "")[:100],
                    }
                    for tc in tool_calls
                ]

            if "tool_call_id" in msg:
                summary["tool_call_id"] = msg["tool_call_id"]

            return summary

        result: Dict[str, Any] = {}

        # OpenAI APIs
        result["ai_api"] = {
            "model": str(app.ai_api.model),
            "message_count": len(app.ai_api.messages),
            "messages": [summarize_message(m) for m in app.ai_api.messages[-20:]],
        }

        result["responses_api"] = {
            "model": str(app.responses_api.model),
            "message_count": len(app.responses_api.messages),
            "messages": [summarize_message(m) for m in app.responses_api.messages[-20:]],
        }

        # Other providers
        for name, provider in app.providers.items():
            if provider_name and name != provider_name:
                continue
            result[name] = {
                "model": str(provider.model),
                "message_count": len(provider.messages),
                "messages": [summarize_message(m) for m in provider.messages[-20:]],
            }

        return AppManager.make_response(data=cast(JsonValue, result))

    @app.route('/api/debug/canvas-state-comparison', methods=['POST'])
    @require_auth
    def debug_canvas_state_comparison() -> ResponseReturnValue:
        """Development-only endpoint for full vs summary canvas-state comparison."""
        if AppManager.is_deployed():
            return AppManager.make_response(
                message='Not found',
                status='error',
                code=404,
            )

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return AppManager.make_response(
                message='Invalid request body',
                status='error',
                code=400,
            )

        canvas_state = payload.get('canvas_state')
        if not isinstance(canvas_state, dict):
            return AppManager.make_response(
                message='canvas_state must be an object',
                status='error',
                code=400,
            )

        comparison = compare_canvas_states(canvas_state)
        return AppManager.make_response(data=cast(JsonValue, comparison))

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
                port = app.config.get('SERVER_PORT', 5000)
                base_url = f"http://127.0.0.1:{port}/"
                app.webdriver_manager = WebDriverManager(base_url=base_url)
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

        svg_state, canvas_image_data, _, _ = extract_vision_payload(request_payload)
        use_vision = bool(message_json.get('use_vision', False))
        ai_model_raw = message_json.get('ai_model')
        ai_model = ai_model_raw if isinstance(ai_model_raw, str) else None

        # Extract attached images from the message JSON (not request_payload)
        attached_images_raw = message_json.get('attached_images')
        attached_images: Optional[List[str]] = None
        if isinstance(attached_images_raw, list):
            attached_images = [img for img in attached_images_raw if isinstance(img, str)]

        # Get the provider for this model and update all relevant APIs
        if ai_model:
            app.ai_api.set_model(ai_model)
            app.responses_api.set_model(ai_model)
            # Get or create provider instance for this model
            provider = get_provider_for_model(app, ai_model)
        else:
            # Use default OpenAI provider
            provider = app.ai_api

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
            # Also inject into the active provider if different
            if provider not in (app.ai_api, app.responses_api):
                _maybe_inject_search_tools(provider, tool_call_results_raw)

        # Store attached images in app context for API access
        app.current_attached_images = attached_images

        @stream_with_context
        def generate() -> Iterator[str]:
            def _yield_pending_logs() -> Iterator[str]:
                """Yield any pending log entries as stream events."""
                for log_entry in app.log_manager.get_pending_logs():
                    log_event: StreamEventDict = {"type": "log", **log_entry}
                    yield json.dumps(log_event) + "\n"

            try:
                # TEMPORARY TEST TRIGGER - REMOVE AFTER TESTING
                if "TEST_ERROR_TRIGGER_12345" in message:
                    raise ValueError("Test error triggered for message recovery testing")
                # END TEMPORARY TEST TRIGGER

                # Route to appropriate API based on model and provider
                model = provider.get_model()
                if model.provider == PROVIDER_OPENAI and model.is_reasoning_model:
                    stream = app.responses_api.create_response_stream(message)
                else:
                    stream = provider.create_chat_completion_stream(message)

                for event in stream:
                    # Yield any pending log events before each stream event
                    yield from _yield_pending_logs()

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
                                    # Intercept search_tools and filter other tool calls
                                    if dict_tool_calls:
                                        filtered_calls = _intercept_search_tools(app, dict_tool_calls, provider)
                                        event_dict['ai_tool_calls'] = cast(JsonValue, filtered_calls)
                                        app.log_manager.log_ai_tool_calls(filtered_calls)
                            except Exception:
                                pass
                            # Reset tools if AI finished (not requesting more tool calls)
                            finish_reason = event_dict.get('finish_reason')
                            if finish_reason != 'tool_calls':
                                if app.ai_api.has_injected_tools():
                                    app.ai_api.reset_tools()
                                if app.responses_api.has_injected_tools():
                                    app.responses_api.reset_tools()
                                # Reset tools on active provider if different
                                if provider not in (app.ai_api, app.responses_api) and provider.has_injected_tools():
                                    provider.reset_tools()
                        yield json.dumps(event_dict) + "\n"
                    else:
                        yield json.dumps(event) + "\n"

                # Yield any remaining logs after stream completes
                yield from _yield_pending_logs()
            except Exception as exc:
                error_msg = f"Streaming exception: {exc}"
                print(f"[Routes /send_message] {error_msg}")
                app.log_manager.log_error(error_msg, source="routes")
                # Reset tools on error
                if app.ai_api.has_injected_tools():
                    app.ai_api.reset_tools()
                if app.responses_api.has_injected_tools():
                    app.responses_api.reset_tools()
                # Reset tools on active provider if different
                if provider not in (app.ai_api, app.responses_api) and provider.has_injected_tools():
                    provider.reset_tools()
                # Yield pending logs so client sees them before error
                yield from _yield_pending_logs()
                # Include error details in the payload for transparency
                error_payload: StreamEventDict = {
                    "type": "final",
                    "ai_message": f"Error: {exc}",
                    "ai_tool_calls": [],
                    "finish_reason": "error",
                    "error_details": str(exc),
                }
                try:
                    yield json.dumps(error_payload) + "\n"
                except Exception:
                    fallback_error_msg = "Failed to send detailed error payload; falling back."
                    print(f"[Routes /send_message] {fallback_error_msg}")
                    app.log_manager.log_error(fallback_error_msg, source="routes")
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
            # Reset all cached providers
            for provider in app.providers.values():
                provider.reset_conversation()
            app.log_manager.log_new_session()
            return AppManager.make_response(message='New conversation started.')
        except Exception as e:
            return AppManager.make_response(
                message=str(e),
                status='error',
                code=500
            )

    @app.route('/save_partial_response', methods=['POST'])
    @require_auth
    def save_partial_response() -> ResponseReturnValue:
        """Save a partial AI response that was interrupted by the user."""
        try:
            request_payload = request.get_json(silent=True)
            if not isinstance(request_payload, dict):
                return AppManager.make_response(
                    message='Invalid request body',
                    status='error',
                    code=400,
                )

            partial_message = request_payload.get('partial_message', '')
            if not isinstance(partial_message, str) or not partial_message.strip():
                return AppManager.make_response(
                    message='No partial message to save',
                    status='error',
                    code=400,
                )

            # Add the partial response to all API conversation histories
            app.ai_api.add_partial_assistant_message(partial_message)
            app.responses_api.add_partial_assistant_message(partial_message)
            for provider in app.providers.values():
                provider.add_partial_assistant_message(partial_message)

            return AppManager.make_response(message='Partial response saved.')
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

        svg_state, canvas_image_data, _, _ = extract_vision_payload(request_payload)
        use_vision = bool(message_json_raw.get('use_vision', False))
        ai_model_raw = message_json_raw.get('ai_model')
        ai_model = ai_model_raw if isinstance(ai_model_raw, str) else None

        # Extract attached images from the message JSON
        attached_images_raw = message_json_raw.get('attached_images')
        attached_images: Optional[List[str]] = None
        if isinstance(attached_images_raw, list):
            attached_images = [img for img in attached_images_raw if isinstance(img, str)]

        # Get the provider for this model and update all relevant APIs
        if ai_model:
            app.ai_api.set_model(ai_model)
            app.responses_api.set_model(ai_model)
            # Get or create provider instance for this model
            provider = get_provider_for_model(app, ai_model)
        else:
            # Use default OpenAI provider
            provider = app.ai_api

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
            # Also inject into the active provider if different
            if provider not in (app.ai_api, app.responses_api):
                _maybe_inject_search_tools(provider, tool_call_results_raw)

        # Store attached images in app context for API access
        app.current_attached_images = attached_images

        def _reset_tools_if_needed(finish_reason: Any) -> None:
            """Reset tools if AI finished (not requesting more tool calls)."""
            if finish_reason != 'tool_calls':
                if app.ai_api.has_injected_tools():
                    app.ai_api.reset_tools()
                if app.responses_api.has_injected_tools():
                    app.responses_api.reset_tools()
                # Reset tools on active provider if different
                if provider not in (app.ai_api, app.responses_api) and provider.has_injected_tools():
                    provider.reset_tools()

        try:
            # Route to appropriate API based on model and provider
            model = provider.get_model()
            if model.provider == PROVIDER_OPENAI and model.is_reasoning_model:
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
                # Intercept search_tools and filter other tool calls
                if ai_tool_calls:
                    ai_tool_calls = _intercept_search_tools(app, ai_tool_calls, provider)
                finish_reason = final_event.get("finish_reason")

                app.log_manager.log_ai_response(ai_message)
                app.log_manager.log_ai_tool_calls(ai_tool_calls)

                _reset_tools_if_needed(finish_reason)
                return AppManager.make_response(data=cast(JsonObject, {
                    "ai_message": ai_message,
                    "ai_tool_calls": cast(JsonValue, ai_tool_calls),
                    "finish_reason": finish_reason,
                }))

            choice = provider.create_chat_completion(message)
            ai_message, ai_tool_calls_processed = _process_ai_response(app, choice)
            ai_tool_calls = cast(List[Dict[str, Any]], ai_tool_calls_processed)
            # Intercept search_tools and filter other tool calls
            if ai_tool_calls:
                ai_tool_calls = _intercept_search_tools(app, ai_tool_calls, provider)
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

        # Get current model from request to use same provider for search
        ai_model_raw = request_payload.get('ai_model')
        ai_model = ai_model_raw if isinstance(ai_model_raw, str) else None

        try:
            # Use current provider's client/model if a local model is specified
            if ai_model:
                provider = get_provider_for_model(app, ai_model)
                service = ToolSearchService(
                    client=provider.client,
                    default_model=provider.get_model(),
                )
            else:
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

    @app.route('/api/tts', methods=['POST'])
    @require_auth
    def generate_tts() -> ResponseReturnValue:
        """Generate text-to-speech audio from text.

        Uses Kokoro-82M for local TTS generation.

        Request body:
            text (str): Text to convert to speech
            voice (str, optional): Voice ID (default: am_michael)

        Returns:
            audio/wav: WAV audio bytes on success
            JSON error response on failure
        """
        request_payload = request.get_json(silent=True)
        if not isinstance(request_payload, dict):
            return AppManager.make_response(
                message='Invalid request body',
                status='error',
                code=400,
            )

        text = request_payload.get('text')
        if not isinstance(text, str) or not text.strip():
            return AppManager.make_response(
                message='Text is required',
                status='error',
                code=400,
            )

        voice_raw = request_payload.get('voice')
        voice = voice_raw if isinstance(voice_raw, str) else None

        tts_manager = get_tts_manager()

        if not tts_manager.is_available():
            return AppManager.make_response(
                message='TTS service is not available. Kokoro may not be installed.',
                status='error',
                code=503,
            )

        # Use threaded version to allow Ctrl+C signal handling
        success, result = tts_manager.generate_speech_threaded(
            text=text,
            voice=voice,
        )

        if not success:
            return AppManager.make_response(
                message=str(result),
                status='error',
                code=500,
            )

        # Return WAV audio bytes
        return Response(
            result,
            mimetype='audio/wav',
            headers={
                'Content-Type': 'audio/wav',
                'Content-Disposition': 'inline; filename="tts_output.wav"',
            }
        )

    @app.route('/api/tts/voices', methods=['GET'])
    @require_auth
    def get_tts_voices() -> ResponseReturnValue:
        """Get available TTS voices.

        Returns:
            JSON response with list of voice IDs
        """
        tts_manager = get_tts_manager()
        return AppManager.make_response(data=cast(JsonValue, {
            'voices': tts_manager.get_voices(),
            'default_voice': tts_manager.DEFAULT_VOICE,
            'available': tts_manager.is_available(),
        }))
