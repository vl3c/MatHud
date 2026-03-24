"""Chat UI manager for the AI interface.

Manages message rendering, streaming token display, markdown parsing,
and MathJax rendering for the chat interface. Owns all DOM manipulation
for chat messages (user, AI, system) and the streaming response lifecycle.

Extracted from ``AIInterface`` to reduce god-class complexity while
preserving the identical public behaviour.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

from browser import document, html, window

from markdown_parser import MarkdownParser
from message_menu_manager import MessageMenuManager
from tool_call_log_manager import ToolCallLogManager


class ChatUIManager:
    """Manages chat message rendering, streaming display, and markdown formatting.

    Attributes:
        markdown_parser: Converts markdown text to HTML for rich formatting.
    """

    def __init__(
        self,
        message_menu: MessageMenuManager,
        tool_call_log: ToolCallLogManager,
        on_image_click: Optional[Callable[[str], None]] = None,
        on_start_timeout: Optional[Callable[[bool], None]] = None,
        on_cancel_timeout: Optional[Callable[[], None]] = None,
    ) -> None:
        self._message_menu = message_menu
        self._tool_call_log = tool_call_log
        self._on_image_click = on_image_click
        self._on_start_timeout = on_start_timeout
        self._on_cancel_timeout = on_cancel_timeout

        # Markdown parser
        self.markdown_parser: MarkdownParser = MarkdownParser()

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

    # ── Read-only properties for AIInterface access ──────────────

    @property
    def stream_buffer(self) -> str:
        """Return the current streaming text buffer."""
        return self._stream_buffer

    @stream_buffer.setter
    def stream_buffer(self, value: str) -> None:
        """Set the streaming text buffer."""
        self._stream_buffer = value

    @property
    def stream_container(self) -> Optional[Any]:
        """Return the current streaming message container DOM element."""
        return self._stream_message_container

    @property
    def stream_content(self) -> Optional[Any]:
        """Return the current streaming content DOM element."""
        return self._stream_content_element

    @property
    def is_reasoning(self) -> bool:
        """Return whether the AI is currently in reasoning phase."""
        return self._is_reasoning

    @property
    def needs_continuation_separator(self) -> bool:
        """Return whether a continuation separator is needed before next text."""
        return self._needs_continuation_separator

    @needs_continuation_separator.setter
    def needs_continuation_separator(self, value: bool) -> None:
        """Set the continuation separator flag."""
        self._needs_continuation_separator = value

    @property
    def request_start_time(self) -> Optional[float]:
        """Return the timestamp when the current request started."""
        return self._request_start_time

    @request_start_time.setter
    def request_start_time(self, value: Optional[float]) -> None:
        """Set the timestamp when the current request started."""
        self._request_start_time = value

    # ── Markdown / rendering ─────────────────────────────────────

    def parse_markdown(self, text: str) -> str:
        """Parse markdown text to HTML using the dedicated markdown parser."""
        return cast(str, self.markdown_parser.parse(text))

    def render_math(self) -> None:
        """Trigger MathJax rendering for newly added content."""
        try:
            # Check if MathJax is available
            if hasattr(window, "MathJax") and hasattr(window.MathJax, "typesetPromise"):
                # Re-render math in the chat history
                window.MathJax.typesetPromise([document["chat-history"]])
        except Exception:
            # MathJax not available or error occurred, continue silently
            pass

    # ── Message element creation ─────────────────────────────────

    def create_message_element(
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
                parsed_content = self.parse_markdown(message)
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
                            if self._on_image_click is not None:
                                self._on_image_click(url)

                        return handler

                    img.bind("click", make_image_click_handler(data_url))
                    images_container <= img
                message_container <= images_container

            # Store the raw source text for copy actions (do not rely on rendered HTML)
            self._message_menu.set_raw_text(message_container, message)
            self._message_menu.attach(message_container, is_ai_message=(sender == "AI"))

            return message_container

        except Exception as e:
            print(f"Error creating message element: {e}")
            # Fall back to simple paragraph
            if sender == "AI":
                content = message.replace("\n", "<br>")
                return html.P(f"<strong>{sender}:</strong> {content}", innerHTML=True)
            else:
                return html.P(f"<strong>{sender}:</strong> {message}")

    def print_ai_message(self, ai_message: str) -> None:
        """Print an AI message to the chat history with markdown support and scroll to bottom."""
        if ai_message:
            message_element = self.create_message_element("AI", ai_message)
            document["chat-history"] <= message_element
            # Trigger MathJax rendering for new content
            self.render_math()
            # Scroll the chat history to the bottom
            document["chat-history"].scrollTop = document["chat-history"].scrollHeight

    def print_user_message(self, user_message: str, images: Optional[list[str]] = None) -> None:
        """Print a user message to the chat history and scroll to bottom.

        Args:
            user_message: The text message from the user
            images: Optional list of image data URLs to display with the message
        """
        # Add the user's message to the chat history with markdown support
        message_element = self.create_message_element("User", user_message, images=images)
        document["chat-history"] <= message_element
        # Trigger MathJax rendering for new content
        self.render_math()
        # Scroll the chat history to the bottom
        document["chat-history"].scrollTop = document["chat-history"].scrollHeight

    def print_system_message(self, message: str) -> None:
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
            line_count = message.count("\n")
            is_long_message = len(message) > 800 or line_count > 20

            if is_long_message:
                # Create expandable content with details/summary
                content_element = self._create_expandable_content(message)
            else:
                # Parse markdown and create content element
                parsed_content = self.parse_markdown(message)
                content_element = html.DIV(Class="chat-content markdown")
                content_element.innerHTML = parsed_content

            # Assemble the message
            message_container <= sender_label
            message_container <= content_element

            # Store raw text for copy actions
            self._message_menu.set_raw_text(message_container, message)
            self._message_menu.attach(message_container)

            # Add to chat history
            document["chat-history"] <= message_container

            # Trigger MathJax rendering for new content
            self.render_math()

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
        lines = message.split("\n")
        if len(lines) > 10:
            preview_text = "\n".join(lines[:10]) + "\n..."
        elif len(message) > 500:
            preview_text = message[:500] + "..."
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
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    # ── Streaming container management ───────────────────────────

    def ensure_stream_element(self) -> None:
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
                self._message_menu.set_raw_text(container, "")
                self._message_menu.attach(container, is_ai_message=True)
            except Exception as e:
                print(f"Error creating streaming element: {e}")

    def ensure_reasoning_element(self) -> None:
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
                self._message_menu.set_raw_text(container, "")
                self._message_menu.attach(container, is_ai_message=True)
            except Exception as e:
                print(f"Error creating reasoning element: {e}")

    # ── Streaming token handlers ─────────────────────────────────

    def on_stream_token(self, text: str) -> None:
        """Handle a streamed token: append to buffer and update the UI element."""
        try:
            # Reset timeout since we're receiving data (use normal timeout for response)
            if self._on_start_timeout is not None:
                self._on_start_timeout(False)

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
                self.ensure_stream_element()
            if self._stream_content_element is not None:
                self._stream_content_element.text = self._stream_buffer
            if self._stream_message_container is not None:
                self._message_menu.set_raw_text(self._stream_message_container, self._stream_buffer)
            document["chat-history"].scrollTop = document["chat-history"].scrollHeight
        except Exception as e:
            print(f"Error handling stream token: {e}")

    def on_stream_reasoning(self, text: str) -> None:
        """Handle a reasoning token: append to reasoning buffer and update UI."""
        try:
            # Use extended timeout for reasoning phase
            if self._on_start_timeout is not None:
                self._on_start_timeout(True)
            self._is_reasoning = True

            # Don't repeat the placeholder if we already have it
            if "(Reasoning in progress...)" in text and "(Reasoning in progress...)" in self._reasoning_buffer:
                return

            self._reasoning_buffer += text
            self.ensure_reasoning_element()
            if self._reasoning_element is not None:
                self._reasoning_element.text = self._reasoning_buffer
            document["chat-history"].scrollTop = document["chat-history"].scrollHeight
        except Exception as e:
            print(f"Error handling reasoning token: {e}")

    # ── Stream finalization ──────────────────────────────────────

    def finalize_stream(self, final_message: Optional[str] = None) -> None:
        """Convert the streamed plain text to parsed markdown and render math."""
        try:
            self._tool_call_log.finalize()

            # Prefer the accumulated buffer (contains all text across tool calls)
            # Only use final_message as fallback if buffer is empty
            text_to_render = self._stream_buffer if self._stream_buffer.strip() else (final_message or "")

            # If we have reasoning content and actual text, create a combined element
            if self._reasoning_buffer and self._stream_message_container is not None:
                # Preserve raw source for copy actions
                self._message_menu.set_raw_text(self._stream_message_container, text_to_render)
                if text_to_render and self._stream_content_element is not None:
                    # Update the response content with parsed markdown
                    parsed_content = self.parse_markdown(text_to_render)
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

                    self.render_math()
                    document["chat-history"].scrollTop = document["chat-history"].scrollHeight
                else:
                    # Reasoning but no text content - remove the empty container
                    self.remove_empty_container()
            elif text_to_render:
                if self._tool_call_log.element is not None and self._stream_message_container is not None:
                    # Tool call log exists — update the container in place to preserve the dropdown
                    self._message_menu.set_raw_text(self._stream_message_container, text_to_render)
                    if self._stream_content_element is not None:
                        parsed_content = self.parse_markdown(text_to_render)
                        self._stream_content_element.innerHTML = parsed_content
                        self._stream_content_element.classList.add("markdown")
                    self.render_math()
                    document["chat-history"].scrollTop = document["chat-history"].scrollHeight
                else:
                    # No reasoning or tool log, use standard finalization
                    final_element = self.create_message_element("AI", text_to_render)

                    history = document["chat-history"]
                    if self._stream_message_container is not None:
                        try:
                            history.replaceChild(final_element, self._stream_message_container)
                        except Exception:
                            history <= final_element
                    else:
                        history <= final_element

                    self.render_math()
                    history.scrollTop = history.scrollHeight
            else:
                # No text content at all - remove any empty container
                self.remove_empty_container()
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
            self._tool_call_log.reset()

    def remove_empty_container(self) -> None:
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
            has_tool_call_log = bool(self._tool_call_log.entries)

            # Only remove if there's NO actual text content anywhere and no tool call log
            if (
                self._stream_message_container is not None
                and not has_buffer_text
                and not has_element_text
                and not has_tool_call_log
            ):
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
                self._tool_call_log.reset()
                # Don't reset _request_start_time here - we want to keep timing across tool calls
        except Exception as e:
            print(f"Error removing empty container: {e}")

    # ── State reset ──────────────────────────────────────────────

    def reset_streaming_state(self) -> None:
        """Reset all streaming state for a new conversation turn."""
        self._stream_buffer = ""
        self._stream_content_element = None
        self._stream_message_container = None
        self._reasoning_buffer = ""
        self._reasoning_element = None
        self._reasoning_details = None
        self._reasoning_summary = None
        self._is_reasoning = False
        self._needs_continuation_separator = False
        self._tool_call_log.reset()
