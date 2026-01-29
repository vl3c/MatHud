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
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, cast

from browser import document, html, ajax, window, console, aio
from function_registry import FunctionRegistry
from process_function_calls import ProcessFunctionCalls
from result_processor import ResultProcessor
from workspace_manager import WorkspaceManager
from markdown_parser import MarkdownParser
from slash_command_handler import SlashCommandHandler
from command_autocomplete import CommandAutocomplete

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
    
    # Timeout in milliseconds for AI responses (60 seconds for local LLMs)
    AI_RESPONSE_TIMEOUT_MS: int = 60000
    # Extended timeout for reasoning models and local LLMs (5 minutes)
    REASONING_TIMEOUT_MS: int = 300000
    
    # Maximum number of images per message
    MAX_ATTACHED_IMAGES: int = 5
    # Warning threshold for image size (10MB)
    IMAGE_SIZE_WARNING_BYTES: int = 10 * 1024 * 1024

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
        self._stop_requested = False
        self._tests_running = False
        self._stop_tests_requested = False
        self.available_functions: Dict[str, Any] = FunctionRegistry.get_available_functions(canvas, self.workspace_manager, self)
        self.undoable_functions: tuple[str, ...] = FunctionRegistry.get_undoable_functions()
        self.markdown_parser: MarkdownParser = MarkdownParser()
        # Slash command handler for local commands
        self.slash_command_handler: SlashCommandHandler = SlashCommandHandler(canvas, self.workspace_manager, self)
        # Command autocomplete popup (initialized lazily when DOM is ready)
        self.command_autocomplete: Optional[CommandAutocomplete] = None
        # Streaming state
        self._stream_buffer: str = ""
        self._stream_content_element: Optional[Any] = None  # DOMNode
        self._stream_message_container: Optional[Any] = None  # DOMNode
        # Reasoning streaming state
        self._reasoning_buffer: str = ""
        self._reasoning_element: Optional[Any] = None  # DOMNode
        self._reasoning_details: Optional[Any] = None  # DOMNode (details element)
        self._reasoning_summary: Optional[Any] = None  # DOMNode (summary element)
        self._is_reasoning: bool = False
        self._request_start_time: Optional[float] = None  # Timestamp when user request started
        self._needs_continuation_separator: bool = False  # Add newline before next text after tool calls
        # Tool call log state
        self._tool_call_log_entries: list[dict[str, Any]] = []
        self._tool_call_log_element: Optional[Any] = None   # <details> element
        self._tool_call_log_summary: Optional[Any] = None   # <summary> element
        self._tool_call_log_content: Optional[Any] = None   # content container div
        # Timeout state
        self._response_timeout_id: Optional[int] = None
        # Chat message menu state
        self._open_message_menu: Optional[Any] = None  # DOMNode
        self._message_menu_global_bound: bool = False
        # Image attachment state
        self._attached_images: list[str] = []  # Data URLs of attached images
        # Message recovery state
        self._last_user_message: str = ""  # Buffered message for recovery on error

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

    async def run_tests_async(
        self,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Run unit tests asynchronously, yielding to browser between test classes.

        Args:
            should_stop: Optional callback that returns True if tests should be stopped.
        """
        try:
            from test_runner import TestRunner
            test_runner = TestRunner(self.canvas, self.available_functions, self.undoable_functions)

            # Run tests asynchronously and get formatted results
            results = await test_runner.run_tests_async(should_stop=should_stop)
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

    def initialize_autocomplete(self) -> None:
        """Initialize the command autocomplete popup.

        Should be called after the DOM is ready (chat-input element exists).
        This is typically called from main.py after event bindings are set up.
        """
        try:
            if "chat-input" in document:
                input_element = document["chat-input"]
                self.command_autocomplete = CommandAutocomplete(
                    input_element,
                    self.slash_command_handler,
                )
        except Exception as e:
            print(f"Error initializing command autocomplete: {e}")

    def initialize_image_attachment(self) -> None:
        """Initialize image attachment functionality.

        Binds event handlers for the attach button and file input.
        Should be called after the DOM is ready.
        """
        try:
            # Bind attach button click
            if "attach-button" in document:
                document["attach-button"].bind("click", self._on_attach_button_click)

            # Bind file input change
            if "image-attach-input" in document:
                document["image-attach-input"].bind("change", self._on_files_selected)

            # Bind modal close handlers
            if "image-modal" in document:
                modal = document["image-modal"]
                modal.bind("click", self._on_modal_backdrop_click)

            close_btn = document.select_one(".image-modal-close")
            if close_btn:
                close_btn.bind("click", self._close_image_modal)
        except Exception as e:
            print(f"Error initializing image attachment: {e}")

    def _on_attach_button_click(self, event: Any) -> None:
        """Handle attach button click - trigger file picker."""
        try:
            if "image-attach-input" in document:
                document["image-attach-input"].click()
        except Exception as e:
            print(f"Error triggering file picker: {e}")

    def trigger_file_picker(self) -> None:
        """Programmatically trigger the file picker for image attachment.

        This is called by the /attach slash command.
        """
        self._on_attach_button_click(None)

    def _on_files_selected(self, event: Any) -> None:
        """Handle file input change - read selected files as data URLs."""
        try:
            file_input = event.target
            files = file_input.files

            if not files or files.length == 0:
                return

            # Check if we've hit the limit
            current_count = len(self._attached_images)
            remaining = self.MAX_ATTACHED_IMAGES - current_count

            if remaining <= 0:
                self._print_system_message_in_chat(
                    f"Maximum of {self.MAX_ATTACHED_IMAGES} images per message. Remove some to add more."
                )
                file_input.value = ""
                return

            files_to_process = min(files.length, remaining)
            if files.length > remaining:
                self._print_system_message_in_chat(
                    f"Only attaching {remaining} of {files.length} images (limit: {self.MAX_ATTACHED_IMAGES})."
                )

            for i in range(files_to_process):
                file = files[i]
                self._read_and_attach_image(file)

            # Clear the input so the same file can be selected again
            file_input.value = ""
        except Exception as e:
            print(f"Error handling file selection: {e}")

    def _read_and_attach_image(self, file: Any) -> None:
        """Read an image file and add it to the attached images list."""
        try:
            # Check file size
            if hasattr(file, "size") and file.size > self.IMAGE_SIZE_WARNING_BYTES:
                size_mb = file.size / (1024 * 1024)
                self._print_system_message_in_chat(
                    f"Warning: Image '{file.name}' is {size_mb:.1f}MB. Large images may slow down processing."
                )

            # Create FileReader to convert to data URL
            reader = window.FileReader.new()

            def on_load(event: Any) -> None:
                try:
                    data_url = reader.result
                    if isinstance(data_url, str) and data_url.startswith("data:image"):
                        self._attached_images.append(data_url)
                        self._update_preview_area()
                except Exception as e:
                    print(f"Error processing image: {e}")

            reader.onload = on_load
            reader.readAsDataURL(file)
        except Exception as e:
            print(f"Error reading image file: {e}")

    def _update_preview_area(self) -> None:
        """Update the image preview area to reflect current attached images."""
        try:
            preview_area = document["image-preview-area"]

            # Clear existing previews
            preview_area.clear()

            if not self._attached_images:
                preview_area.style.display = "none"
                return

            preview_area.style.display = "flex"

            for idx, data_url in enumerate(self._attached_images):
                # Create preview item container
                item = html.DIV(Class="image-preview-item")

                # Create thumbnail image
                img = html.IMG(src=data_url)
                item <= img

                # Create remove button
                remove_btn = html.BUTTON("\u00d7", Class="remove-btn")
                remove_btn.attrs["title"] = "Remove image"

                # Bind remove handler with closure for index
                def make_remove_handler(index: int) -> Any:
                    def handler(event: Any) -> None:
                        event.stopPropagation()
                        self._remove_attached_image(index)
                    return handler

                remove_btn.bind("click", make_remove_handler(idx))
                item <= remove_btn

                preview_area <= item
        except Exception as e:
            print(f"Error updating preview area: {e}")

    def _remove_attached_image(self, index: int) -> None:
        """Remove an attached image by index."""
        try:
            if 0 <= index < len(self._attached_images):
                self._attached_images.pop(index)
                self._update_preview_area()
        except Exception as e:
            print(f"Error removing attached image: {e}")

    def _clear_attached_images(self) -> None:
        """Clear all attached images."""
        self._attached_images = []
        self._update_preview_area()

    def _show_image_modal(self, data_url: str) -> None:
        """Display an image in full-size modal."""
        try:
            modal = document["image-modal"]
            modal_img = document["image-modal-img"]
            modal_img.src = data_url
            modal.style.display = "flex"
        except Exception as e:
            print(f"Error showing image modal: {e}")

    def _close_image_modal(self, event: Any = None) -> None:
        """Close the image modal."""
        try:
            modal = document["image-modal"]
            modal.style.display = "none"
        except Exception as e:
            print(f"Error closing image modal: {e}")

    def _on_modal_backdrop_click(self, event: Any) -> None:
        """Close modal when clicking outside the image."""
        try:
            if event.target.id == "image-modal":
                self._close_image_modal()
        except Exception as e:
            print(f"Error handling modal click: {e}")

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

    def _set_raw_message_text(self, message_container: Any, raw_text: str) -> None:
        """Attach raw message source text to a message container for later actions (copy, etc.)."""
        try:
            # Store on the element object itself to avoid parsing rendered HTML.
            setattr(message_container, "_raw_message_text", raw_text)
        except Exception:
            pass

    def _get_raw_message_text(self, message_container: Any) -> str:
        """Return the stored raw message text from the container, or empty string if missing."""
        try:
            value = getattr(message_container, "_raw_message_text", "")
            if isinstance(value, str):
                return value
            return str(value)
        except Exception:
            return ""

    def _copy_text_to_clipboard(self, text: str) -> bool:
        """Copy text to clipboard using the modern API with a fallback for older contexts."""
        if text is None:
            text = ""
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                text = ""

        # Prefer navigator.clipboard when available (may require secure context).
        try:
            navigator = getattr(window, "navigator", None)
            clipboard = getattr(navigator, "clipboard", None) if navigator is not None else None
            write_text = getattr(clipboard, "writeText", None) if clipboard is not None else None
            if callable(write_text):
                write_text(text)
                return True
        except Exception:
            pass

        # Fallback: temporary textarea + execCommand('copy')
        try:
            textarea = html.TEXTAREA()
            textarea.value = text
            textarea.attrs["readonly"] = "readonly"
            textarea.style.position = "fixed"
            textarea.style.left = "0"
            textarea.style.top = "0"
            textarea.style.opacity = "0"

            # Append to DOM, select content, copy, then remove.
            document <= textarea
            try:
                textarea.focus()
            except Exception:
                pass
            try:
                textarea.select()
            except Exception:
                pass
            try:
                textarea.setSelectionRange(0, len(text))
            except Exception:
                pass

            copied = False
            try:
                copied = bool(window.document.execCommand("copy"))
            except Exception:
                try:
                    copied = bool(document.execCommand("copy"))
                except Exception:
                    copied = False

            try:
                textarea.remove()
            except Exception:
                pass

            return copied
        except Exception:
            return False

    def _bind_message_menu_global_handlers(self) -> None:
        """Bind global document handlers needed for message menus (close on outside click)."""
        if self._message_menu_global_bound:
            return
        try:
            document.bind("click", self._on_document_click_close_message_menu)
            self._message_menu_global_bound = True
        except Exception:
            self._message_menu_global_bound = False

    def _on_document_click_close_message_menu(self, _event: Any) -> None:
        """Close any open message menu when clicking outside of it."""
        try:
            if self._open_message_menu is not None:
                self._hide_message_menu(self._open_message_menu)
        except Exception:
            self._open_message_menu = None

    def _hide_message_menu(self, menu: Any) -> None:
        try:
            menu.style.display = "none"
        except Exception:
            pass
        if self._open_message_menu is menu:
            self._open_message_menu = None

    def _show_message_menu(self, menu: Any) -> None:
        try:
            if self._open_message_menu is not None and self._open_message_menu is not menu:
                self._hide_message_menu(self._open_message_menu)
        except Exception:
            self._open_message_menu = None

        try:
            menu.style.display = "block"
        except Exception:
            pass
        self._open_message_menu = menu

    def _toggle_message_menu(self, menu: Any) -> None:
        try:
            current_display = getattr(menu.style, "display", "")
        except Exception:
            current_display = ""

        if current_display == "none" or not current_display:
            self._show_message_menu(menu)
        else:
            self._hide_message_menu(menu)

    def _attach_message_menu(self, message_container: Any) -> None:
        """Attach the per-message '...' menu to the message container (idempotent)."""
        try:
            if bool(getattr(message_container, "_has_message_menu", False)):
                return
            setattr(message_container, "_has_message_menu", True)
        except Exception:
            # If we cannot track state on the element, continue anyway.
            pass

        self._bind_message_menu_global_handlers()

        menu_button = html.BUTTON("...", Class="chat-message-menu-button")
        try:
            menu_button.attrs["type"] = "button"
            menu_button.attrs["title"] = "Message options"
            menu_button.attrs["aria-label"] = "Message options"
        except Exception:
            pass

        menu = html.DIV(Class="chat-message-menu")
        try:
            menu.style.display = "none"
        except Exception:
            pass

        copy_item = html.BUTTON("Copy message text", Class="chat-message-menu-item")
        try:
            copy_item.attrs["type"] = "button"
        except Exception:
            pass

        def _stop_propagation(ev: Any) -> None:
            try:
                ev.stopPropagation()
            except Exception:
                pass

        def _on_menu_button_click(ev: Any) -> None:
            _stop_propagation(ev)
            self._toggle_message_menu(menu)

        def _on_menu_click(ev: Any) -> None:
            _stop_propagation(ev)

        def _on_copy_click(ev: Any) -> None:
            _stop_propagation(ev)
            raw_text = self._get_raw_message_text(message_container)
            self._copy_text_to_clipboard(raw_text)
            self._hide_message_menu(menu)

        try:
            menu_button.bind("click", _on_menu_button_click)
            menu.bind("click", _on_menu_click)
            copy_item.bind("click", _on_copy_click)
        except Exception:
            pass

        menu <= copy_item

        # Add button + menu to the message container (positioned by CSS).
        try:
            message_container <= menu_button
            message_container <= menu
        except Exception:
            pass

    
    def _create_message_element(
        self,
        sender: str,
        message: str,
        message_type: str = "normal",
        images: Optional[list[str]] = None,
    ) -> Any:  # DOMNode
        """Create a styled message element with markdown support and optional images.

        Args:
            sender: The message sender ("User" or "AI")
            message: The message text content
            message_type: CSS class for message styling ("normal", "system")
            images: Optional list of image data URLs to display with the message

        Returns:
            DOM element for the message
        """
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

            # Add images if provided
            if images:
                images_container = html.DIV(Class="chat-message-images")
                for data_url in images:
                    img = html.IMG(src=data_url, Class="chat-message-image")
                    img.attrs["alt"] = "Attached image"

                    # Bind click to show modal
                    def make_image_click_handler(url: str) -> Any:
                        def handler(event: Any) -> None:
                            self._show_image_modal(url)
                        return handler

                    img.bind("click", make_image_click_handler(data_url))
                    images_container <= img
                message_container <= images_container

            # Store the raw source text for copy actions (do not rely on rendered HTML)
            self._set_raw_message_text(message_container, message)
            self._attach_message_menu(message_container)

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
                # Initialize raw text storage for streaming content
                self._set_raw_message_text(container, "")
                self._attach_message_menu(container)
            except Exception as e:
                print(f"Error creating streaming element: {e}")

    def _ensure_reasoning_element(self) -> None:
        """Create the reasoning dropdown element inside the AI message box."""
        if self._reasoning_element is None:
            try:
                container = html.DIV(Class="chat-message normal")
                label = html.SPAN("AI: ", Class="chat-sender ai")
                
                # Collapsible dropdown for reasoning
                details = html.DETAILS(Class="reasoning-dropdown")
                # Start collapsed by default (user can expand if curious)
                summary = html.SUMMARY("Thinking...", Class="reasoning-summary")
                reasoning_content = html.DIV(Class="reasoning-content")
                reasoning_content.text = ""
                details <= summary
                details <= reasoning_content
                
                # Content area for the actual response (hidden initially)
                response_content = html.DIV(Class="chat-content")
                response_content.text = ""
                
                container <= label
                container <= details
                container <= response_content
                document["chat-history"] <= container
                
                self._reasoning_element = reasoning_content
                self._reasoning_details = details
                self._reasoning_summary = summary
                self._stream_message_container = container
                self._stream_content_element = response_content
                # Initialize raw text storage for reasoning responses
                self._set_raw_message_text(container, "")
                self._attach_message_menu(container)
            except Exception as e:
                print(f"Error creating reasoning element: {e}")

    def _reset_tool_call_log_state(self) -> None:
        """Reset all tool call log state for a new turn."""
        self._tool_call_log_entries = []
        self._tool_call_log_element = None
        self._tool_call_log_summary = None
        self._tool_call_log_content = None

    def _format_tool_call_args_display(self, args: dict[str, Any]) -> str:
        """Format a tool call's arguments dict for compact display.

        Filters out the ``canvas`` key, truncates individual values to 30
        characters and the total string to 80 characters.
        """
        parts: list[str] = []
        for k, v in args.items():
            if k == "canvas":
                continue
            v_str = str(v)
            if len(v_str) > 30:
                v_str = v_str[:27] + "..."
            parts.append(f"{k}: {v_str}")
        result = ", ".join(parts)
        if len(result) > 80:
            result = result[:77] + "..."
        return result

    def _create_tool_call_entry_element(self, entry: dict[str, Any]) -> Any:
        """Build the DOM element for a single tool call log entry."""
        div = html.DIV(Class="tool-call-entry")

        is_error = entry.get("is_error", False)
        status_class = "tool-call-status error" if is_error else "tool-call-status success"
        status_char = "\u2717" if is_error else "\u2713"
        status_span = html.SPAN(status_char, Class=status_class)
        div <= status_span

        name_span = html.SPAN(entry.get("name", ""), Class="tool-call-name")
        div <= name_span

        short_args = entry.get("args_display", "")
        full_args = entry.get("args_full", short_args)
        args_span = html.SPAN(f"({short_args})", Class="tool-call-args")
        div <= args_span

        # Show error message or result
        result_display = entry.get("result_display", "")
        result_full = entry.get("result_full", result_display)
        result_span: Any = None

        if is_error:
            error_msg = entry.get("error_message", "")
            if error_msg:
                err_span = html.SPAN(f" \u2192 {error_msg}", Class="tool-call-error-msg")
                div <= err_span
        elif result_display:
            result_span = html.SPAN(f" \u2192 {result_display}", Class="tool-call-result")
            div <= result_span

        # Click to toggle between truncated and full view
        def _toggle_expand(event: Any) -> None:
            try:
                if div.classList.contains("expanded"):
                    div.classList.remove("expanded")
                    args_span.text = f"({short_args})"
                    if result_span is not None and result_display:
                        result_span.text = f" \u2192 {result_display}"
                else:
                    div.classList.add("expanded")
                    args_span.text = f"({full_args})"
                    if result_span is not None and result_full:
                        result_span.text = f" \u2192 {result_full}"
            except Exception:
                pass

        div.bind("click", _toggle_expand)

        return div

    def _ensure_tool_call_log_element(self) -> None:
        """Create the tool-call-log ``<details>`` element if it doesn't exist yet."""
        if self._tool_call_log_element is not None:
            return

        # We need a message container to attach to
        if self._stream_message_container is None:
            self._ensure_stream_message_element()

        details = html.DETAILS(Class="tool-call-log-dropdown")
        summary = html.SUMMARY("Using tools...", Class="tool-call-log-summary")
        content_div = html.DIV(Class="tool-call-log-content")
        details <= summary
        details <= content_div

        # Insert before the content element so it appears after reasoning but before text
        if self._stream_message_container is not None and self._stream_content_element is not None:
            try:
                self._stream_message_container.insertBefore(details, self._stream_content_element)
            except Exception:
                self._stream_message_container <= details
        elif self._stream_message_container is not None:
            self._stream_message_container <= details

        self._tool_call_log_element = details
        self._tool_call_log_summary = summary
        self._tool_call_log_content = content_div

    def _add_tool_call_entries(self, tool_calls: list[dict[str, Any]], call_results: dict[str, Any]) -> None:
        """Record tool call entries and update the dropdown UI.

        Args:
            tool_calls: Raw tool call dicts from the AI response.
            call_results: Dict mapping result keys to their outcomes.
        """
        self._ensure_tool_call_log_element()

        for call in tool_calls:
            function_name: str = call.get("function_name", "")
            args: dict[str, Any] = call.get("arguments", {})
            args_display = self._format_tool_call_args_display(args)

            result_key = ResultProcessor._generate_result_key(function_name, args)

            # Special handling for evaluate_expression which uses expression as key
            if function_name == "evaluate_expression" and "expression" in args:
                expr = str(args.get("expression", "")).replace(" ", "")
                variables = args.get("variables")
                if variables and isinstance(variables, dict):
                    vars_str = ", ".join(f"{k}:{v}" for k, v in variables.items())
                    expr_key = f"{expr} for {vars_str}"
                else:
                    expr_key = expr
                result_value = call_results.get(expr_key, call_results.get(result_key, ""))
            else:
                result_value = call_results.get(result_key, call_results.get(function_name, ""))
            is_error = isinstance(result_value, str) and result_value.startswith("Error:")
            error_message = result_value if is_error else ""

            # Full untruncated args for the expanded view
            args_full = ", ".join(
                f"{k}: {v}" for k, v in args.items() if k != "canvas"
            )

            # Format result for display (truncate if too long)
            result_display = ""
            if not is_error and result_value:
                result_str = str(result_value)
                if len(result_str) > 100:
                    result_display = result_str[:97] + "..."
                else:
                    result_display = result_str

            entry: dict[str, Any] = {
                "name": function_name,
                "args_display": args_display,
                "args_full": args_full,
                "is_error": is_error,
                "error_message": error_message,
                "result_display": result_display,
                "result_full": str(result_value) if result_value else "",
            }
            self._tool_call_log_entries.append(entry)

            entry_el = self._create_tool_call_entry_element(entry)
            if self._tool_call_log_content is not None:
                self._tool_call_log_content <= entry_el

        # Update summary with running count
        count = len(self._tool_call_log_entries)
        if self._tool_call_log_summary is not None:
            self._tool_call_log_summary.text = f"Using tools... ({count} so far)"

    def _finalize_tool_call_log(self) -> None:
        """Update the tool call log summary to its final state."""
        if not self._tool_call_log_entries:
            return

        count = len(self._tool_call_log_entries)
        error_count = sum(1 for e in self._tool_call_log_entries if e.get("is_error"))

        label = f"Used {count} tool" if count == 1 else f"Used {count} tools"
        if error_count:
            label += f" ({error_count} failed)"

        if self._tool_call_log_summary is not None:
            self._tool_call_log_summary.text = label

        # Ensure collapsed — removeAttribute is reliable for boolean HTML attributes
        if self._tool_call_log_element is not None:
            try:
                self._tool_call_log_element.removeAttribute("open")
            except Exception:
                pass

    def _on_stream_log(self, event_obj: Any) -> None:
        """Handle a server log event: output to browser console with appropriate level."""
        try:
            event = self._normalize_stream_event(event_obj)
            level = event.get("level", "info")
            message = event.get("message", "")
            source = event.get("source", "")

            prefix = f"[Server{':' + source if source else ''}]"
            full_message = f"{prefix} {message}"

            if level == "error":
                console.error(full_message)
            elif level == "warning":
                console.warn(full_message)
            else:
                console.log(full_message)
        except Exception as e:
            print(f"Error handling server log event: {e}")

    def _on_stream_reasoning(self, text: str) -> None:
        """Handle a reasoning token: append to reasoning buffer and update UI."""
        try:
            # Use extended timeout for reasoning phase
            self._start_response_timeout(use_reasoning_timeout=True)
            self._is_reasoning = True
            
            # Don't repeat the placeholder if we already have it
            if "(Reasoning in progress...)" in text and "(Reasoning in progress...)" in self._reasoning_buffer:
                return
            
            self._reasoning_buffer += text
            self._ensure_reasoning_element()
            if self._reasoning_element is not None:
                self._reasoning_element.text = self._reasoning_buffer
            document["chat-history"].scrollTop = document["chat-history"].scrollHeight
        except Exception as e:
            print(f"Error handling reasoning token: {e}")

    def _on_stream_token(self, text: str) -> None:
        """Handle a streamed token: append to buffer and update the UI element."""
        try:
            # Reset timeout since we're receiving data (use normal timeout for response)
            self._start_response_timeout(use_reasoning_timeout=False)
            
            # If we were in reasoning phase, collapse the reasoning dropdown
            if self._is_reasoning and self._reasoning_details is not None:
                try:
                    del self._reasoning_details.attrs["open"]
                except Exception:
                    try:
                        self._reasoning_details.attrs["open"] = False
                    except Exception:
                        pass
                self._is_reasoning = False
            
            # When continuing after tool calls, clear the buffer and start fresh
            # The AI will re-state any necessary context in its new response
            # This prevents duplication when AI restates previous confirmations
            if self._needs_continuation_separator:
                self._stream_buffer = ""
                self._needs_continuation_separator = False
            
            self._stream_buffer += text
            # Use reasoning element's response area if it exists, otherwise create normal element
            if self._stream_content_element is None and self._reasoning_element is None:
                self._ensure_stream_message_element()
            if self._stream_content_element is not None:
                self._stream_content_element.text = self._stream_buffer
            if self._stream_message_container is not None:
                self._set_raw_message_text(self._stream_message_container, self._stream_buffer)
            document["chat-history"].scrollTop = document["chat-history"].scrollHeight
        except Exception as e:
            print(f"Error handling stream token: {e}")

    def _finalize_stream_message(self, final_message: Optional[str] = None) -> None:
        """Convert the streamed plain text to parsed markdown and render math."""
        try:
            self._finalize_tool_call_log()

            # Prefer the accumulated buffer (contains all text across tool calls)
            # Only use final_message as fallback if buffer is empty
            text_to_render = self._stream_buffer if self._stream_buffer.strip() else (final_message or "")
            
            # If we have reasoning content and actual text, create a combined element
            if self._reasoning_buffer and self._stream_message_container is not None:
                # Preserve raw source for copy actions
                self._set_raw_message_text(self._stream_message_container, text_to_render)
                if text_to_render and self._stream_content_element is not None:
                    # Update the response content with parsed markdown
                    parsed_content = self._parse_markdown_to_html(text_to_render)
                    self._stream_content_element.innerHTML = parsed_content
                    self._stream_content_element.classList.add("markdown")
                    
                    # Update summary to show elapsed time and ensure dropdown stays closed
                    if self._reasoning_summary is not None and self._request_start_time is not None:
                        try:
                            from browser import window
                            elapsed_ms = window.Date.now() - self._request_start_time
                            elapsed_seconds = int(elapsed_ms / 1000)
                            self._reasoning_summary.text = f"Thought for {elapsed_seconds} seconds"
                        except Exception:
                            pass
                    
                    # Ensure dropdown is closed
                    if self._reasoning_details is not None:
                        try:
                            del self._reasoning_details.attrs["open"]
                        except Exception:
                            try:
                                self._reasoning_details.attrs["open"] = False
                            except Exception:
                                pass
                    
                    self._render_math()
                    document["chat-history"].scrollTop = document["chat-history"].scrollHeight
                else:
                    # Reasoning but no text content - remove the empty container
                    self._remove_empty_response_container()
            elif text_to_render:
                if self._tool_call_log_element is not None and self._stream_message_container is not None:
                    # Tool call log exists — update the container in place to preserve the dropdown
                    self._set_raw_message_text(self._stream_message_container, text_to_render)
                    if self._stream_content_element is not None:
                        parsed_content = self._parse_markdown_to_html(text_to_render)
                        self._stream_content_element.innerHTML = parsed_content
                        self._stream_content_element.classList.add("markdown")
                    self._render_math()
                    document["chat-history"].scrollTop = document["chat-history"].scrollHeight
                else:
                    # No reasoning or tool log, use standard finalization
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
            else:
                # No text content at all - remove any empty container
                self._remove_empty_response_container()
        except Exception as e:
            print(f"Error finalizing stream message: {e}")
        finally:
            self._stream_buffer = ""
            self._stream_content_element = None
            self._stream_message_container = None
            self._reasoning_buffer = ""
            self._reasoning_element = None
            self._reasoning_details = None
            self._reasoning_summary = None
            self._is_reasoning = False
            self._request_start_time = None
            self._reset_tool_call_log_state()

    def _remove_empty_response_container(self) -> None:
        """Remove the current response container if it has no actual text content.
        
        This cleans up "Thinking..." boxes when the AI only performs tool calls
        without providing a text response. Never removes a container with actual text.
        """
        try:
            # Check if there's actual text content in buffer or visible in the element
            has_buffer_text = bool(self._stream_buffer.strip())
            has_element_text = False
            if self._stream_content_element is not None:
                try:
                    element_text = self._stream_content_element.text or self._stream_content_element.innerHTML or ""
                    has_element_text = bool(element_text.strip())
                except Exception:
                    pass
            has_tool_call_log = bool(self._tool_call_log_entries)

            # Only remove if there's NO actual text content anywhere and no tool call log
            if self._stream_message_container is not None and not has_buffer_text and not has_element_text and not has_tool_call_log:
                history = document["chat-history"]
                try:
                    history.removeChild(self._stream_message_container)
                except Exception:
                    pass
                # Reset state
                self._stream_message_container = None
                self._stream_content_element = None
                self._reasoning_element = None
                self._reasoning_details = None
                self._reasoning_summary = None
                self._reasoning_buffer = ""
                self._is_reasoning = False
                self._reset_tool_call_log_state()
                # Don't reset _request_start_time here - we want to keep timing across tool calls
        except Exception as e:
            print(f"Error removing empty container: {e}")

    def _on_stream_final(self, event_obj: Any) -> None:
        """Handle the final event from the streaming response."""
        try:
            event = self._normalize_stream_event(event_obj)

            finish_reason = event.get('finish_reason', 'stop')
            ai_tool_calls = event.get('ai_tool_calls', [])
            ai_message = event.get('ai_message', '')
            error_details = event.get('error_details', '')

            # Log error details to console for debugging
            if finish_reason == "error":
                console.error(f"[AI Error] {error_details or ai_message}")

            # If no tool calls OR finish reason indicates completion, finalize the message
            if finish_reason in ("stop", "error", "completed") or not ai_tool_calls:
                if not self._stream_buffer and ai_message:
                    self._stream_buffer = ai_message
                self._finalize_stream_message(ai_message or None)
                # Restore user message on error so they can retry
                if finish_reason == "error":
                    self._restore_user_message_on_error()
                else:
                    # Clear recovery buffer on successful completion
                    self._last_user_message = ""
                self._enable_send_controls()
                return

            # Processing tool calls - keep the "Thinking..." container visible
            # It will be removed/updated when the final response arrives
            try:
                call_results = ProcessFunctionCalls.get_results(ai_tool_calls, self.available_functions, self.undoable_functions, self.canvas)
                self._store_results_in_canvas_state(call_results)
                self._add_tool_call_entries(ai_tool_calls, call_results)

                if self._stop_requested:
                    self._finalize_stream_message()
                    self._print_system_message_in_chat("Generation stopped.")
                    self._enable_send_controls()
                    return

                # Reset timeout with extended duration - AI needs time to process tool results
                self._start_response_timeout(use_reasoning_timeout=True)
                # Mark that we need a newline separator before the next text
                if self._stream_buffer.strip():
                    self._needs_continuation_separator = True
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
        self._restore_user_message_on_error()
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

    def _restore_user_message_on_error(self) -> None:
        """Restore the last user message to the input field after an error.

        This allows the user to retry sending the same message without
        having to retype it. Also applies a brief visual flash to indicate
        the message was restored due to an error.
        """
        if not self._last_user_message:
            return
        try:
            chat_input = document["chat-input"]
            chat_input.value = self._last_user_message
            # Apply visual error feedback
            chat_input.classList.add("error-flash")
            window.setTimeout(
                lambda: chat_input.classList.remove("error-flash"),
                2000
            )
        except Exception as e:
            print(f"Error restoring user message: {e}")

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
            for key in ["type", "text", "ai_message", "ai_tool_calls", "finish_reason", "error_details", "level", "message", "source"]:
                try:
                    result[key] = getattr(event_obj, key)
                except Exception:
                    pass
            return cast(Dict[str, Any], result)
        except Exception:
            return {}

    def _print_user_message_in_chat(self, user_message: str, images: Optional[list[str]] = None) -> None:
        """Print a user message to the chat history and scroll to bottom.

        Args:
            user_message: The text message from the user
            images: Optional list of image data URLs to display with the message
        """
        # Add the user's message to the chat history with markdown support
        message_element = self._create_message_element("User", user_message, images=images)
        document["chat-history"] <= message_element
        # Trigger MathJax rendering for new content
        self._render_math()
        # Scroll the chat history to the bottom
        document["chat-history"].scrollTop = document["chat-history"].scrollHeight

    def _print_system_message_in_chat(self, message: str) -> None:
        """Print a system/command response to the chat history.

        Used for slash command responses that don't come from AI.

        Args:
            message: The message to display (supports markdown)
        """
        try:
            # Create message container with system styling
            message_container = html.DIV(Class="chat-message system")

            # Create sender label
            sender_label = html.SPAN("System: ", Class="chat-sender system")

            # Check if message is long and needs expandable display
            line_count = message.count('\n')
            is_long_message = len(message) > 800 or line_count > 20

            if is_long_message:
                # Create expandable content with details/summary
                content_element = self._create_expandable_content(message)
            else:
                # Parse markdown and create content element
                parsed_content = self._parse_markdown_to_html(message)
                content_element = html.DIV(Class="chat-content markdown")
                content_element.innerHTML = parsed_content

            # Assemble the message
            message_container <= sender_label
            message_container <= content_element

            # Store raw text for copy actions
            self._set_raw_message_text(message_container, message)
            self._attach_message_menu(message_container)

            # Add to chat history
            document["chat-history"] <= message_container

            # Trigger MathJax rendering for new content
            self._render_math()

            # Scroll to bottom
            document["chat-history"].scrollTop = document["chat-history"].scrollHeight
        except Exception as e:
            print(f"Error printing system message: {e}")
            # Fallback to simple paragraph
            fallback = html.P(f"System: {message}")
            document["chat-history"] <= fallback

    def _create_expandable_content(self, message: str) -> Any:
        """Create an expandable content element for long messages.

        Args:
            message: The full message content

        Returns:
            A DOM element with expandable content
        """
        # Create preview (first ~500 chars or 10 lines)
        lines = message.split('\n')
        if len(lines) > 10:
            preview_text = '\n'.join(lines[:10]) + '\n...'
        elif len(message) > 500:
            preview_text = message[:500] + '...'
        else:
            preview_text = message

        # Create container
        container = html.DIV(Class="chat-content expandable-content")

        # Create preview section
        preview = html.DIV(Class="content-preview")
        preview.innerHTML = f"<pre>{self._escape_html(preview_text)}</pre>"

        # Create full content section (hidden initially)
        full_content = html.DIV(Class="content-full", style={"display": "none"})
        full_content.innerHTML = f"<pre>{self._escape_html(message)}</pre>"

        # Create toggle button
        toggle_btn = html.BUTTON("Show more", Class="expand-toggle-btn")

        def toggle_content(event: Any) -> None:
            try:
                if full_content.style.display == "none":
                    preview.style.display = "none"
                    full_content.style.display = "block"
                    toggle_btn.text = "Show less"
                else:
                    preview.style.display = "block"
                    full_content.style.display = "none"
                    toggle_btn.text = "Show more"
            except Exception:
                pass

        toggle_btn.bind("click", toggle_content)

        container <= preview
        container <= full_content
        container <= toggle_btn

        return container

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters.

        Args:
            text: Text to escape

        Returns:
            Escaped text safe for HTML
        """
        return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;"))

    def _debug_log_ai_response(self, ai_message: str, ai_function_calls: Any, finish_reason: str) -> None:
        """Log debug information about the AI response."""
        print(f"### AI message: {ai_message}")
        print(f"### AI function calls: {ai_function_calls}")
        print(f"### AI finish reason: {finish_reason}")

    def _disable_send_controls(self) -> None:
        """Switch send button to stop mode while processing and start a timeout."""
        try:
            self.is_processing = True
            self._stop_requested = False
            if "send-button" in document:
                btn = document["send-button"]
                if "disabled" in btn.attrs:
                    del btn.attrs["disabled"]
                btn.text = "Stop"
                btn.classList.add("stop-mode")
            self._start_response_timeout()
        except Exception as e:
            print(f"Error disabling send controls: {e}")

    def _enable_send_controls(self) -> None:
        """Restore send button to normal mode after processing and cancel the timeout."""
        try:
            self._cancel_response_timeout()
            self.is_processing = False
            self._stop_requested = False
            if "send-button" in document:
                btn = document["send-button"]
                btn.disabled = False
                btn.text = "Send"
                btn.classList.remove("stop-mode")
        except Exception as e:
            print(f"Error enabling send controls: {e}")

    def _start_response_timeout(self, use_reasoning_timeout: bool = False) -> None:
        """Start a timeout that will re-enable controls if no response is received.
        
        Args:
            use_reasoning_timeout: If True, use extended timeout for reasoning models
        """
        try:
            # Cancel any existing timeout first
            self._cancel_response_timeout()
            timeout_ms = self.REASONING_TIMEOUT_MS if use_reasoning_timeout else self.AI_RESPONSE_TIMEOUT_MS
            self._response_timeout_id = window.setTimeout(
                self._on_response_timeout,
                timeout_ms
            )
        except Exception as e:
            print(f"Error starting response timeout: {e}")

    def _cancel_response_timeout(self) -> None:
        """Cancel the response timeout if one is pending."""
        try:
            if self._response_timeout_id is not None:
                window.clearTimeout(self._response_timeout_id)
                self._response_timeout_id = None
        except Exception as e:
            print(f"Error cancelling response timeout: {e}")

    def _on_response_timeout(self) -> None:
        """Handle timeout - abort the stream, re-enable controls and show error message."""
        try:
            self._response_timeout_id = None
            if self.is_processing:
                print("AI response timeout - aborting stream and re-enabling send controls")
                self._abort_current_stream()
                self._print_ai_message_in_chat(
                    "⚠️ Request timed out. The AI is taking too long to respond. Please try again."
                )
                self._enable_send_controls()
        except Exception as e:
            print(f"Error handling response timeout: {e}")
    
    def _abort_current_stream(self) -> None:
        """Abort the current streaming connection if one is active."""
        try:
            if hasattr(window, 'abortCurrentStream'):
                window.abortCurrentStream()
        except Exception as e:
            print(f"Error aborting stream: {e}")

    def stop_ai_processing(self):
        """Stop the current AI processing, abort the stream, and restore UI controls."""
        self._stop_requested = True
        self._abort_current_stream()
        self._cancel_response_timeout()
        # Save partial response to conversation history before finalizing
        if self._stream_buffer and self._stream_buffer.strip():
            self._save_partial_response(self._stream_buffer)
        self._finalize_stream_message()
        self._print_system_message_in_chat("Generation stopped.")
        self._enable_send_controls()

    def _save_partial_response(self, partial_message):
        """Save interrupted partial response to the backend conversation history."""
        try:
            payload = json.dumps({"partial_message": partial_message})
            ajax.post(
                "/save_partial_response",
                data=payload,
                headers={"Content-Type": "application/json"},
                oncomplete=lambda req: None,
                onerror=lambda req: print(f"Failed to save partial response: {req.status}"),
            )
        except Exception as e:
            print(f"Error saving partial response: {e}")

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
        """Create the JSON payload for the request, optionally including SVG and Canvas2D state."""
        payload: Dict[str, Any] = {'message': prompt}
        vision_enabled = self._is_vision_enabled(prompt)
        renderer_mode = getattr(self.canvas, "renderer_mode", None)
        if isinstance(renderer_mode, str):
            payload['renderer_mode'] = renderer_mode

        svg_state_payload: Optional[Dict[str, Any]] = None
        if include_svg:
            try:
                svg_element = document["math-svg"]
                svg_content = svg_element.outerHTML
                container = document["math-container"]
                rect = container.getBoundingClientRect()
                svg_state_payload = {
                    'content': svg_content,
                    'dimensions': {
                        'width': rect.width,
                        'height': rect.height
                    },
                    'viewBox': svg_element.getAttribute("viewBox"),
                    'transform': svg_element.getAttribute("transform")
                }
                payload['svg_state'] = svg_state_payload
            except Exception as exc:
                print(f"Failed to collect SVG state: {exc}")

        if not vision_enabled:
            return payload

        snapshot: Dict[str, Any] = {}
        if isinstance(renderer_mode, str):
            snapshot['renderer_mode'] = renderer_mode
        if svg_state_payload:
            snapshot['svg_state'] = svg_state_payload

        if renderer_mode == "canvas2d":
            canvas_image = self._capture_canvas2d_snapshot()
            if canvas_image:
                snapshot['canvas_image'] = canvas_image

        if snapshot:
            payload['vision_snapshot'] = snapshot

        return payload

    def _is_vision_enabled(self, prompt: Optional[str]) -> bool:
        try:
            if not prompt:
                return False
            parsed = json.loads(prompt)
            if isinstance(parsed, dict):
                return bool(parsed.get("use_vision"))
        except Exception:
            pass
        return False

    def _capture_canvas2d_snapshot(self) -> Optional[str]:
        try:
            canvas_el = document.getElementById("math-canvas-2d")
            if canvas_el is None:
                return None
            to_data_url = getattr(canvas_el, "toDataURL", None)
            if not callable(to_data_url):
                return None
            data_url = to_data_url("image/png")
            if isinstance(data_url, str) and data_url:
                return data_url
        except Exception as exc:
            print(f"Failed to capture Canvas2D snapshot: {exc}")
        return None

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
            # Don't reset any state here - all state management is done in _send_prompt_to_ai_stream
            # This preserves intermediary text and reasoning content across tool call continuations
            # Call JS streaming helper with reasoning and log callbacks
            window.sendMessageStream(
                payload_js,
                self._on_stream_token,
                self._on_stream_final,
                self._on_stream_error,
                self._on_stream_reasoning,
                self._on_stream_log
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

    def _send_prompt_to_ai_stream(
        self,
        user_message: Optional[str] = None,
        tool_call_results: Optional[str] = None,
        attached_images: Optional[list[str]] = None,
    ) -> None:
        canvas_state = self.canvas.get_canvas_state()
        use_vision = document["vision-toggle"].checked and user_message is not None and tool_call_results is None
        prompt_json: Dict[str, Any] = {
            "canvas_state": canvas_state,
            "user_message": user_message,
            "tool_call_results": tool_call_results,
            "use_vision": use_vision,
            "ai_model": document["ai-model-selector"].value
        }
        # Include attached images if provided (works independently of vision toggle)
        if attached_images:
            prompt_json["attached_images"] = attached_images
        prompt = json.dumps(prompt_json)
        print(f'Prompt for AI (stream): {prompt[:500]}...' if len(prompt) > 500 else f'Prompt for AI (stream): {prompt}')
        
        # For new user messages, reset all state including containers and buffers
        # For tool call results, preserve everything to keep intermediary text visible
        if user_message is not None and tool_call_results is None:
            self._request_start_time = window.Date.now()
            # Reset all streaming state for new conversation turn
            self._stream_buffer = ""
            self._stream_content_element = None
            self._stream_message_container = None
            self._reasoning_buffer = ""
            self._reasoning_element = None
            self._reasoning_details = None
            self._reasoning_summary = None
            self._is_reasoning = False
            self._needs_continuation_separator = False
            self._reset_tool_call_log_state()

        try:
            payload = self._create_request_payload(prompt, include_svg=True)
            self._start_streaming_request(payload)
        except Exception as e:
            print(f"Error preparing streaming request: {str(e)}")
            payload = self._create_request_payload(prompt, include_svg=False)
            self._start_streaming_request(payload)

    def _send_prompt_to_ai(
        self,
        user_message: Optional[str] = None,
        tool_call_results: Optional[str] = None,
        attached_images: Optional[list[str]] = None,
    ) -> None:
        canvas_state = self.canvas.get_canvas_state()

        # Only use vision when we have a user message and no tool call results
        use_vision = document["vision-toggle"].checked and user_message is not None and tool_call_results is None

        prompt_json: Dict[str, Any] = {
            "canvas_state": canvas_state,
            "user_message": user_message,
            "tool_call_results": tool_call_results,
            "use_vision": use_vision,
            "ai_model": document["ai-model-selector"].value
        }

        # Include attached images if provided (works independently of vision toggle)
        if attached_images:
            prompt_json["attached_images"] = attached_images

        # Convert to JSON string
        prompt = json.dumps(prompt_json)
        
        # For new user messages, reset all state including containers and buffers
        # For tool call results, preserve everything to keep intermediary text visible
        if user_message is not None and tool_call_results is None:
            self._request_start_time = window.Date.now()
            # Reset all streaming state for new conversation turn
            self._stream_buffer = ""
            self._stream_content_element = None
            self._stream_message_container = None
            self._reasoning_buffer = ""
            self._reasoning_element = None
            self._reasoning_details = None
            self._reasoning_summary = None
            self._is_reasoning = False
            self._needs_continuation_separator = False
            self._reset_tool_call_log_state()

        self._send_request(prompt)

    def send_user_message(self, message: str) -> None:
        """Sends a message as if the user typed it.

        If the message is a slash command (starts with "/"), it is executed
        locally without sending to the AI backend.

        Allows sending with just attached images (empty message).
        """
        has_text = bool(message.strip())
        has_images = len(self._attached_images) > 0

        # Need either text or images to send
        if self.is_processing or (not has_text and not has_images):
            return

        # Check for slash command (only if there's text)
        if has_text and self.slash_command_handler.is_slash_command(message):
            self._print_user_message_in_chat(message)
            result = self.slash_command_handler.execute(message)
            self._print_system_message_in_chat(result.message)
            return

        # Capture attached images before clearing
        images_to_send = list(self._attached_images) if self._attached_images else None

        # Use a default message for image-only sends
        display_message = message if has_text else "[Image attached]"
        ai_message = message if has_text else "What do you see in this image?"

        # Display the user message with images in chat
        self._print_user_message_in_chat(display_message, images=images_to_send)

        # Clear attached images after displaying (not after successful send)
        self._clear_attached_images()

        # Regular AI flow
        self._disable_send_controls()
        self._send_prompt_to_ai(ai_message, attached_images=images_to_send)

    def run_tests_action(self, event: Any) -> None:
        """Trigger the test suite directly on the client side (TEMPORARY).

        See documentation/development/removing_run_tests_button.md for removal instructions.
        """
        # If tests are running, stop them
        if self._tests_running:
            self._stop_tests_requested = True
            return

        # Start tests
        self._stop_tests_requested = False
        aio.run(self._execute_tests_async())

    async def _execute_tests_async(self) -> None:
        """Execute tests asynchronously using browser.aio."""
        run_tests_btn = document["run-tests-button"]

        try:
            self._tests_running = True

            # Switch to "Stop Tests" mode
            run_tests_btn.text = "Stop Tests"
            run_tests_btn.classList.add("stop-mode")

            # Disable Send button while tests run
            if "send-button" in document:
                document["send-button"].disabled = True

            self._print_user_message_in_chat("Run tests (direct execution)")

            # Run tests asynchronously with stop callback
            results = await self.run_tests_async(
                should_stop=lambda: self._stop_tests_requested
            )

            # Check if tests were stopped
            was_stopped = results.get('stopped', False)

            if was_stopped:
                summary = (
                    f"### Test Results (Stopped)\n\n"
                    f"- **Tests Run:** {results.get('tests_run', 0)}\n"
                    f"- **Failures:** {results.get('failures', 0)}\n"
                    f"- **Errors:** {results.get('errors', 0)}\n"
                    f"\n*Tests were stopped by user.*"
                )
            else:
                summary = (
                    f"### Test Results\n\n"
                    f"- **Tests Run:** {results.get('tests_run', 0)}\n"
                    f"- **Failures:** {results.get('failures', 0)}\n"
                    f"- **Errors:** {results.get('errors', 0)}\n"
                )

            if results.get('failing_tests'):
                summary += "\n#### Failures:\n"
                for fail in results['failing_tests']:
                    summary += f"- **{fail['test']}**: {fail['error']}\n"

            if results.get('error_tests'):
                summary += "\n#### Errors:\n"
                for err in results['error_tests']:
                    summary += f"- **{err['test']}**: {err['error']}\n"

            self._print_ai_message_in_chat(summary)

        except Exception as e:
            error_msg = f"Error running tests: {str(e)}"
            print(error_msg)
            self._print_ai_message_in_chat(error_msg)

        finally:
            # Restore buttons and reset stop flag
            self._tests_running = False
            self._stop_tests_requested = False
            run_tests_btn.text = "Run Tests"
            run_tests_btn.classList.remove("stop-mode")
            if "send-button" in document:
                document["send-button"].disabled = False

    def interact_with_ai(self, event: Any) -> None:
        if self.is_processing:
            self.stop_ai_processing()
            return

        # Get the user's message from the input field
        user_message = document["chat-input"].value.strip()
        has_images = len(self._attached_images) > 0

        # Allow sending if there's text OR attached images
        if user_message or has_images:
            # Buffer message for recovery on error before clearing
            self._last_user_message = user_message
            document["chat-input"].value = ''
            self.send_user_message(user_message)

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