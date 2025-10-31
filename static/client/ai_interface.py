"""
MatHud AI Communication Interface

Handles bidirectional communication between the client-side canvas and backend AI services.
Manages user interactions, function call processing, and visual presentation of AI responses.

Key Features:
    - AJAX-based communication with backend AI API
    - Function call execution and result aggregation
    - Markdown parsing and MathJax rendering for AI responses
    - Chat interface management with styled messages
    - SVG state transmission for AI vision capabilities
    - Computation history integration with canvas state
    - Testing framework integration

Communication Flow:
    1. User input → JSON payload creation with canvas state
    2. Backend AI processing with function calls
    3. Function execution and result collection
    4. Response rendering with markdown and math support
    5. Canvas state updates with computation results

Dependencies:
    - browser: DOM manipulation and AJAX requests
    - function_registry: Available AI function mappings
    - process_function_calls: Function execution coordination
    - workspace_manager: File persistence operations
    - markdown_parser: Rich text formatting support
"""

from __future__ import annotations

import json
import traceback
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from browser import document, html, ajax, window, console
from function_registry import FunctionRegistry
from process_function_calls import ProcessFunctionCalls
from workspace_manager import WorkspaceManager
from markdown_parser import MarkdownParser

if TYPE_CHECKING:
    from canvas import Canvas


class AIInterface:
    """Communication bridge between the mathematical canvas and backend AI services.
    
    Orchestrates the complete interaction cycle from user input to AI response processing,
    including function call execution, state management, and visual presentation.
    
    Attributes:
        canvas (Canvas): Mathematical canvas for visualization and state management
        workspace_manager (WorkspaceManager): Handles workspace persistence operations
        is_processing (bool): Tracks whether an AI request is currently being processed
        available_functions (dict): Registry of all functions available to the AI
        undoable_functions (tuple): Functions that support undo/redo operations
        markdown_parser (MarkdownParser): Converts markdown text to HTML for rich formatting
    """
    def __init__(self, canvas: "Canvas") -> None:
        """Initialize the AI interface with canvas integration and function registry.
        
        Sets up all necessary components for AI communication including function mappings,
        workspace management, and markdown processing capabilities.
        
        Args:
            canvas (Canvas): The mathematical canvas instance to interact with
        """
        self.canvas: "Canvas" = canvas
        self.workspace_manager: WorkspaceManager = WorkspaceManager(canvas)
        self.is_processing: bool = False  # Track whether we're processing an AI response
        self.available_functions: Dict[str, Any] = FunctionRegistry.get_available_functions(canvas, self.workspace_manager, self)
        self.undoable_functions: tuple[str, ...] = FunctionRegistry.get_undoable_functions()
        self.markdown_parser: MarkdownParser = MarkdownParser()
        # Streaming state
        self._stream_buffer: str = ""
        self._stream_content_element: Optional[Any] = None  # DOMNode
        self._stream_message_container: Optional[Any] = None  # DOMNode

    def run_tests(self) -> Dict[str, Any]:
        """Run unit tests for the AIInterface class and return results to the AI as the function result."""
        try:
            from test_runner import TestRunner
            test_runner = TestRunner(self.canvas, self.available_functions, self.undoable_functions)
            
            # Run tests and get formatted results in one step
            results = test_runner.run_tests()
            return cast(Dict[str, Any], test_runner.format_results_for_ai(results))
        except ImportError as e:
            print(f"Test runner not available: {e}")
            return cast(Dict[str, Any], {
                "tests_run": 0,
                "failures": 0,
                "errors": 1,
                "failing_tests": [],
                "error_tests": [{"test": "Test Runner Import", "error": f"Could not import test runner: {e}"}]
            })

    def _store_results_in_canvas_state(self, call_results: Dict[str, Any]) -> None:
        """Store valid function call results in the canvas state, skipping special cases and formatting values."""
        if not ProcessFunctionCalls.validate_results(call_results):
            return
            
        for key, value in call_results.items():
            # Skip storing workspace management functions and test results in computations
            if key.startswith("list_workspaces") or \
               key.startswith("save_workspace") or \
               key.startswith("load_workspace") or \
               key.startswith("run_tests"):
                continue

            if not ProcessFunctionCalls.is_successful_result(value):
                continue

            # Format numeric results consistently
            if isinstance(value, (int, float)):
                formatted_value = float(value)  # Always convert numeric values to float
            else:
                formatted_value = value
                
            # DISABLED: Saving basic calculations to canvas state (takes up too many tokens, not useful info to store)
            # self.canvas.add_computation(
            #     expression=key,  # The key is already the expression
            #     result=formatted_value
            # )

    def _parse_markdown_to_html(self, text: str) -> str:
        """Parse markdown text to HTML using the dedicated markdown parser."""
        return cast(str, self.markdown_parser.parse(text))
    
    def _render_math(self) -> None:
        """Trigger MathJax rendering for newly added content."""
        try:
            # Check if MathJax is available
            if hasattr(window, 'MathJax') and hasattr(window.MathJax, 'typesetPromise'):
                # Re-render math in the chat history
                window.MathJax.typesetPromise([document["chat-history"]])
        except Exception as e:
            # MathJax not available or error occurred, continue silently
            pass

    
    def _create_message_element(self, sender: str, message: str, message_type: str = "normal") -> Any:  # DOMNode
        """Create a styled message element with markdown support."""
        try:
            # Create message container
            message_container = html.DIV(Class=f"chat-message {message_type}")
            
            # Create sender label
            sender_label = html.SPAN(f"{sender}: ", Class=f"chat-sender {sender.lower()}")
            
            # Parse markdown and create content element
            if sender == "AI":
                parsed_content = self._parse_markdown_to_html(message)
                content_element = html.DIV(Class="chat-content markdown")
                content_element.innerHTML = parsed_content
            else:
                # For user messages, keep them as plain text for now
                content_element = html.SPAN(message, Class="chat-content")
            
            # Assemble the message
            message_container <= sender_label
            message_container <= content_element
            
            return message_container
            
        except Exception as e:
            print(f"Error creating message element: {e}")
            # Fall back to simple paragraph
            if sender == "AI":
                content = message.replace('\n', '<br>')
                return html.P(f'<strong>{sender}:</strong> {content}', innerHTML=True)
            else:
                return html.P(f'<strong>{sender}:</strong> {message}')

    def _print_ai_message_in_chat(self, ai_message: str) -> None:
        """Print an AI message to the chat history with markdown support and scroll to bottom."""
        if ai_message:
            message_element = self._create_message_element("AI", ai_message)
            document["chat-history"] <= message_element
            # Trigger MathJax rendering for new content
            self._render_math()
            # Scroll the chat history to the bottom
            document["chat-history"].scrollTop = document["chat-history"].scrollHeight

    def _ensure_stream_message_element(self) -> None:
        """Create the streaming AI message element if it does not exist yet."""
        if self._stream_content_element is None:
            try:
                container = html.DIV(Class="chat-message normal")
                label = html.SPAN("AI: ", Class="chat-sender ai")
                content = html.DIV(Class="chat-content")
                content.text = ""
                container <= label
                container <= content
                document["chat-history"] <= container
                self._stream_message_container = container
                self._stream_content_element = content
            except Exception as e:
                print(f"Error creating streaming element: {e}")

    def _on_stream_token(self, text: str) -> None:
        """Handle a streamed token: append to buffer and update the UI element."""
        try:
            self._stream_buffer += text
            self._ensure_stream_message_element()
            if self._stream_content_element is not None:
                self._stream_content_element.text = self._stream_buffer
            document["chat-history"].scrollTop = document["chat-history"].scrollHeight
        except Exception as e:
            print(f"Error handling stream token: {e}")

    def _finalize_stream_message(self, final_message: Optional[str] = None) -> None:
        """Convert the streamed plain text to parsed markdown and render math."""
        try:
            text_to_render = final_message if final_message is not None else self._stream_buffer
            if not text_to_render:
                return

            final_element = self._create_message_element("AI", text_to_render)

            history = document["chat-history"]
            if self._stream_message_container is not None:
                try:
                    history.replaceChild(final_element, self._stream_message_container)
                except Exception:
                    history <= final_element
            else:
                history <= final_element

            self._render_math()
            history.scrollTop = history.scrollHeight
        except Exception as e:
            print(f"Error finalizing stream message: {e}")
        finally:
            self._stream_buffer = ""
            self._stream_content_element = None
            self._stream_message_container = None

    def _on_stream_final(self, event_obj: Any) -> None:
        """Handle the final event from the streaming response."""
        try:
            event = self._normalize_stream_event(event_obj)

            finish_reason = event.get('finish_reason', 'stop')
            ai_tool_calls = event.get('ai_tool_calls', [])
            ai_message = event.get('ai_message', '')

            if finish_reason in ("stop", "error"):
                if not self._stream_buffer and ai_message:
                    self._stream_buffer = ai_message
                self._finalize_stream_message(ai_message or None)
                self._enable_send_controls()
                return

            try:
                call_results = ProcessFunctionCalls.get_results(ai_tool_calls, self.available_functions, self.undoable_functions, self.canvas)
                self._store_results_in_canvas_state(call_results)
                self._send_prompt_to_ai(None, json.dumps(call_results))
            except Exception as e:
                print(f"Error processing streamed tool calls: {e}")
                self._enable_send_controls()
        except Exception as e:
            print(f"Error handling stream final: {e}")
            self._enable_send_controls()

    def _on_stream_error(self, err: Any) -> None:
        """Handle streaming errors and re-enable controls."""
        error_message = self._format_stream_error(err)
        print(f"Streaming error: {error_message}")
        try:
            console.error("Streaming error", err)
        except Exception:
            pass
        self._enable_send_controls()

    def _format_stream_error(self, err: Any) -> str:
        """Convert a streaming error object into a readable string."""
        try:
            if err is None:
                return "Unknown error"
            if isinstance(err, str):
                return err
            # Brython-wrapped JS Error objects expose message and stack
            message = getattr(err, "message", None)
            stack = getattr(err, "stack", None)
            if message:
                if stack and message not in stack:
                    return cast(str, f"{message} | stack: {stack}")
                return cast(str, message)
            # Fall back to toString() if available
            to_string = getattr(err, "toString", None)
            if callable(to_string):
                return cast(str, to_string())
            return str(err)
        except Exception as format_exc:
            return f"Error while formatting streaming error: {format_exc}"

    def _normalize_stream_event(self, event_obj: Any) -> Dict[str, Any]:
        """Convert JS objects or dicts into plain Python dicts."""
        try:
            if event_obj is None:
                return {}
            if isinstance(event_obj, dict):
                return event_obj
            try:
                return cast(Dict[str, Any], json.loads(window.JSON.stringify(event_obj)))
            except Exception:
                pass
            result = {}
            for key in ["type", "text", "ai_message", "ai_tool_calls", "finish_reason"]:
                try:
                    result[key] = getattr(event_obj, key)
                except Exception:
                    pass
            return cast(Dict[str, Any], result)
        except Exception:
            return {}

    def _print_user_message_in_chat(self, user_message: str) -> None:
        """Print a user message to the chat history, clear input field, and scroll to bottom."""
        # Add the user's message to the chat history with markdown support
        message_element = self._create_message_element("User", user_message)
        document["chat-history"] <= message_element
        # Trigger MathJax rendering for new content
        self._render_math()
        # Clear the input field
        document["chat-input"].value = ''
        # Scroll the chat history to the bottom
        document["chat-history"].scrollTop = document["chat-history"].scrollHeight

    def _debug_log_ai_response(self, ai_message: str, ai_function_calls: Any, finish_reason: str) -> None:
        """Log debug information about the AI response."""
        print(f"### AI message: {ai_message}")
        print(f"### AI function calls: {ai_function_calls}")
        print(f"### AI finish reason: {finish_reason}")

    def _disable_send_controls(self) -> None:
        """Disable only the send functionality while processing."""
        try:
            self.is_processing = True
            if "send-button" in document:
                document["send-button"].disabled = True
        except Exception as e:
            print(f"Error disabling send controls: {e}")

    def _enable_send_controls(self) -> None:
        """Enable send functionality after processing."""
        try:
            self.is_processing = False
            if "send-button" in document:
                document["send-button"].disabled = False
        except Exception as e:
            print(f"Error enabling send controls: {e}")

    def _process_ai_response(self, ai_message: str, tool_calls: Any, finish_reason: str) -> None:     
        self._debug_log_ai_response(ai_message, tool_calls, finish_reason)

        if finish_reason == "stop" or finish_reason == "error":
            self._print_ai_message_in_chat(ai_message)
            self._enable_send_controls()
        else: # finish_reason == "tool_calls" or "function_call"
            try:
                call_results = ProcessFunctionCalls.get_results(tool_calls, self.available_functions, self.undoable_functions, self.canvas)
                self._store_results_in_canvas_state(call_results)
                self._send_prompt_to_ai(None, json.dumps(call_results))
            except Exception as e:
                print(f"Error processing tool calls: {e}")
                traceback.print_exc()
                self._enable_send_controls()  # Enable controls if there's an error

    def _on_error(self, request: Any) -> None:
        """Handle request errors and ensure send controls are re-enabled."""
        print(f"Error: {request.status}, {request.text}")
        self._enable_send_controls()

    def _on_complete(self, request: Any) -> None:
        """Handle request completion and process AI response."""
        try:
            if request.status == 200 or request.status == 0:                
                # Extract data from the proper response structure
                response_data = request.json.get('data')
                if not response_data:
                    error_msg = request.json.get('message', 'Invalid response format')
                    print(f"Error: {error_msg}")
                    document["ai-response"].text = error_msg
                    self._enable_send_controls()
                    return

                ai_message = response_data.get('ai_message')
                ai_function_calls = response_data.get('ai_tool_calls')
                finish_reason = response_data.get('finish_reason')

                # Parse the AI's response and create / delete drawables as needed
                self._process_ai_response(ai_message, ai_function_calls, finish_reason)
            else:
                self._on_error(request)
        except Exception as e:
            print(f"Error processing AI response: {e}")
            traceback.print_exc()
            self._enable_send_controls()

    def _create_request_payload(self, prompt: Optional[str], include_svg: bool = True) -> Dict[str, Any]:
        """Create the JSON payload for the request, optionally including SVG state."""
        if not include_svg:
            return {'message': prompt}
            
        # Get the SVG element and its content
        svg_element = document["math-svg"]
        svg_content = svg_element.outerHTML
        
        # Get container dimensions
        container = document["math-container"]
        rect = container.getBoundingClientRect()
        
        payload = {
            'message': prompt,
            'svg_state': {
                'content': svg_content,
                'dimensions': {
                    'width': rect.width,
                    'height': rect.height
                },
                'viewBox': svg_element.getAttribute("viewBox"),
                'transform': svg_element.getAttribute("transform")
            }
        }
        return payload

    def _make_request(self, payload: Dict[str, Any]) -> None:
        """Send an AJAX request with the given payload."""
        req = ajax.ajax()
        req.bind('complete', self._on_complete)
        req.bind('error', self._on_error)
        req.open('POST', '/send_message', True)
        req.set_header('content-type', 'application/json')
        req.send(json.dumps(payload))

    def _start_streaming_request(self, payload: Dict[str, Any]) -> None:
        """Start a streaming request using a JS helper for Fetch streaming."""
        try:
            payload_json = json.dumps(payload)
            payload_js = window.JSON.parse(payload_json)
            self._stream_buffer = ""
            self._stream_content_element = None
            self._stream_message_container = None
            window.sendMessageStream(
                payload_js,
                self._on_stream_token,
                self._on_stream_final,
                self._on_stream_error
            )
        except Exception as e:
            print(f"Falling back to non-streaming request due to error: {e}")
            self._make_request(payload)

    def _send_request(self, prompt: Optional[str]) -> None:
        try:
            # Try to send request with SVG state
            payload = self._create_request_payload(prompt, include_svg=True)
            self._start_streaming_request(payload)
        except Exception as e:
            print(f"Error preparing request with SVG: {str(e)}")
            # Fall back to sending request without SVG state
            payload = self._create_request_payload(prompt, include_svg=False)
            self._start_streaming_request(payload)

    def _send_prompt_to_ai_stream(self, user_message: Optional[str] = None, tool_call_results: Optional[str] = None) -> None:
        canvas_state = self.canvas.get_canvas_state()
        use_vision = document["vision-toggle"].checked and user_message is not None and tool_call_results is None
        prompt_json = {
            "canvas_state": canvas_state,
            "user_message": user_message,
            "tool_call_results": tool_call_results,
            "use_vision": use_vision,
            "ai_model": document["ai-model-selector"].value
        }
        prompt = json.dumps(prompt_json)
        print(f'Prompt for AI (stream): {prompt}')
        try:
            payload = self._create_request_payload(prompt, include_svg=True)
            self._start_streaming_request(payload)
        except Exception as e:
            print(f"Error preparing streaming request: {str(e)}")
            payload = self._create_request_payload(prompt, include_svg=False)
            self._start_streaming_request(payload)

    def _send_prompt_to_ai(self, user_message: Optional[str] = None, tool_call_results: Optional[str] = None) -> None:
        canvas_state = self.canvas.get_canvas_state()
        
        # Only use vision when we have a user message and no tool call results
        use_vision = document["vision-toggle"].checked and user_message is not None and tool_call_results is None
        
        prompt_json = {
            "canvas_state": canvas_state,
            "user_message": user_message,
            "tool_call_results": tool_call_results,
            "use_vision": use_vision,
            "ai_model": document["ai-model-selector"].value
        }
        
        # Convert to JSON string
        prompt = json.dumps(prompt_json)
        print(f'Prompt for AI: {prompt}')
        self._send_request(prompt)

    def interact_with_ai(self, event: Any) -> None:
        # Don't process if we're already handling a request
        if self.is_processing:
            return
            
        # Get the user's message from the input field
        user_message = document["chat-input"].value
        # Print the user message in chat
        self._print_user_message_in_chat(user_message)
        # Disable send controls while processing
        self._disable_send_controls()
        # Send the user's message to the AI
        self._send_prompt_to_ai(user_message)

    def start_new_conversation(self, event: Any) -> None:
        """Saves the current workspace, resets the canvas and chat, and starts a new backend session."""
        # 1. Save the current workspace automatically
        self.workspace_manager.save_workspace()

        # 2. Reset the client-side canvas
        self.canvas.clear()

        # 3. Clear the chat history UI
        document["chat-history"].clear()

        # 4. Call the backend to reset the AI conversation state
        req = ajax.ajax()
        req.open('POST', '/new_conversation', True)
        req.set_header('content-type', 'application/json')
        req.send()