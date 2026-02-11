"""
MatHud OpenAI API Base Module

Base class for OpenAI API implementations providing shared functionality
for both Chat Completions and Responses APIs.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from collections.abc import Iterator, Sequence
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

from static.ai_model import AIModel
from static.canvas_state_summarizer import compare_canvas_states
from static.functions_definitions import FUNCTIONS, FunctionDefinition

# Use the shared MatHud logger for file logging
_logger = logging.getLogger("mathud")

MessageContent = Union[str, List[Dict[str, Any]]]
MessageDict = Dict[str, Any]
StreamEvent = Dict[str, Any]

# Tool mode type
ToolMode = Literal["full", "search"]

# Essential tool names that should always be available after injection
ESSENTIAL_TOOLS = frozenset({
    "search_tools",
    "undo",
    "redo",
    "get_current_canvas_state",
})


def _build_search_mode_tools() -> List[FunctionDefinition]:
    """Build the minimal tool set for search mode.

    Returns:
        List containing search_tools and essential tools only.
    """
    search_tools: List[FunctionDefinition] = []
    for tool in FUNCTIONS:
        func = tool.get("function", {})
        name = func.get("name", "")
        if name == "search_tools" or name in ESSENTIAL_TOOLS:
            search_tools.append(tool)
    return search_tools


# Precomputed search mode tools
SEARCH_MODE_TOOLS: List[FunctionDefinition] = _build_search_mode_tools()


class OpenAIAPIBase:
    """Base class for OpenAI API implementations."""

    DEV_MSG = """You are an educational graphing calculator AI interface that can draw shapes, perform calculations and help users explore mathematics. DO NOT try to perform calculations by yourself, use the tools provided instead. Always analyze the canvas state before proceeding. Canvas state is included with user messages; for large scenes it may be summarized to reduce noise. When you need complete details, call get_current_canvas_state. Canvas state may be stale after tool calls, so re-check live state between actions when needed. Never use emoticons or emoji in your responses. When performing multiple steps, include a succinct summary of all actions taken in your final response. INFO: Point labels and coordinates are hardcoded to be shown next to all points on the canvas."""

    CANVAS_SUMMARY_MODE_ENV = "AI_CANVAS_SUMMARY_MODE"
    CANVAS_HYBRID_MAX_BYTES_ENV = "AI_CANVAS_HYBRID_FULL_MAX_BYTES"
    CANVAS_SUMMARY_TELEMETRY_ENV = "AI_CANVAS_SUMMARY_TELEMETRY"
    DEFAULT_CANVAS_SUMMARY_MODE = "hybrid"
    DEFAULT_CANVAS_HYBRID_FULL_MAX_BYTES = 6000

    @staticmethod
    def _initialize_api_key() -> str:
        """Initialize the OpenAI API key from environment or .env file.

        Returns a placeholder if the key is not found, allowing the application
        to start with other providers configured. Actual OpenAI API calls will
        fail with an authentication error in that case.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key

        dotenv_path = ".env"
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            logging.getLogger("mathud").warning(
                "OPENAI_API_KEY not found. OpenAI models will be unavailable."
            )
            return "not-configured"

        return api_key

    def __init__(
        self,
        model: Optional[AIModel] = None,
        temperature: float = 0.2,
        tools: Optional[Sequence[FunctionDefinition]] = None,
        max_tokens: int = 16000,
        tool_mode: ToolMode = "full",
    ) -> None:
        """Initialize OpenAI API client and conversation state.

        Args:
            model: AI model to use. Defaults to the default model.
            temperature: Sampling temperature.
            tools: Custom tool definitions. Defaults to all FUNCTIONS.
            max_tokens: Maximum tokens in response.
            tool_mode: Tool mode - "full" for all tools, "search" for search_tools + essentials.
        """
        self.client = OpenAI(api_key=self._initialize_api_key())
        self.model: AIModel = model if model is not None else AIModel.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._tool_mode: ToolMode = tool_mode
        self._custom_tools: Optional[Sequence[FunctionDefinition]] = tools
        self._injected_tools: bool = False  # Track if tools were dynamically injected
        self.tools: Sequence[FunctionDefinition] = self._resolve_tools()
        self.messages: List[MessageDict] = [
            {"role": "developer", "content": OpenAIAPIBase.DEV_MSG}
        ]

    def _resolve_tools(self) -> Sequence[FunctionDefinition]:
        """Resolve the active tool set based on mode and custom tools.

        Returns:
            The appropriate tool set for the current configuration.
        """
        if self._custom_tools is not None:
            return list(self._custom_tools)
        if self._tool_mode == "search":
            return SEARCH_MODE_TOOLS
        return list(FUNCTIONS)

    def get_tool_mode(self) -> ToolMode:
        """Get the current tool mode.

        Returns:
            The current tool mode ("full" or "search").
        """
        return self._tool_mode

    def set_tool_mode(self, mode: ToolMode) -> None:
        """Set the tool mode and update available tools.

        Args:
            mode: The tool mode to set ("full" or "search").
        """
        if mode not in ("full", "search"):
            raise ValueError(f"Invalid tool mode: {mode}. Must be 'full' or 'search'.")

        if self._tool_mode != mode:
            self._tool_mode = mode
            # Only update tools if not using custom tools
            if self._custom_tools is None:
                self.tools = self._resolve_tools()
                msg = f"Tool mode changed to: {mode} ({len(self.tools)} tools available)"
                print(msg)
                _logger.info(msg)

    def inject_tools(self, tools: Sequence[FunctionDefinition], include_essentials: bool = True) -> None:
        """Dynamically inject specific tools for the next API call.

        This allows search results to directly influence which tools are available.
        The injected tools replace the current tool set until reset.

        Args:
            tools: List of tool definitions to make available.
            include_essentials: If True, also include essential tools (search_tools, undo, redo, get_current_canvas_state).
        """
        if not tools:
            _logger.debug("inject_tools called with empty tools list, ignoring")
            return

        injected: List[FunctionDefinition] = list(tools)

        if include_essentials:
            # Add essential tools if not already present
            injected_names = {t.get("function", {}).get("name") for t in injected}
            for tool in FUNCTIONS:
                func = tool.get("function", {})
                name = func.get("name", "")
                if name in ESSENTIAL_TOOLS and name not in injected_names:
                    injected.append(tool)

        self._injected_tools = True
        self.tools = injected
        msg = f"Injected {len(injected)} tools (essentials={'included' if include_essentials else 'excluded'})"
        _logger.info(msg)

    def reset_tools(self) -> None:
        """Reset tools to the default set based on current tool mode.

        Call this after using inject_tools to restore normal tool availability.
        """
        if self._injected_tools:
            self._injected_tools = False
            self.tools = self._resolve_tools()
            msg = f"Tools reset to {self._tool_mode} mode ({len(self.tools)} tools)"
            _logger.info(msg)

    def has_injected_tools(self) -> bool:
        """Check if tools were dynamically injected.

        Returns:
            True if tools have been injected via inject_tools(), False otherwise.
        """
        return self._injected_tools

    def get_model(self) -> AIModel:
        """Get the current AI model instance."""
        return self.model

    def create_chat_completion(self, full_prompt: str) -> Any:
        """Create a chat completion. Implemented by subclasses."""
        raise NotImplementedError

    def create_chat_completion_stream(self, full_prompt: str) -> Iterator[StreamEvent]:
        """Stream a chat completion. Implemented by subclasses."""
        raise NotImplementedError

    def reset_conversation(self) -> None:
        """Reset the conversation history to start a new session."""
        self.messages = [{"role": "developer", "content": OpenAIAPIBase.DEV_MSG}]

    def add_partial_assistant_message(self, content: str) -> None:
        """Add a partial assistant message that was interrupted by the user."""
        if content and content.strip():
            self.messages.append({"role": "assistant", "content": content})

    def set_model(self, identifier: str) -> None:
        """Set the AI model by identifier string."""
        if str(self.model) != identifier:
            self.model = AIModel.from_identifier(identifier)
            msg = f"API model updated to: {identifier}"
            print(msg)  # Console output
            _logger.info(msg)  # File logging

    def _remove_canvas_state_from_user_messages(self) -> None:
        """Remove canvas state from all user messages in the conversation history."""
        for message in reversed(self.messages):
            if message.get("role") == "user" and "content" in message:
                content = message["content"]

                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_content = part.get("text", "")
                            if "canvas_state" not in text_content:
                                continue
                            try:
                                text_json = json.loads(text_content)
                                if isinstance(text_json, dict):
                                    text_json.pop("canvas_state", None)
                                    text_json.pop("canvas_state_summary", None)
                                    part["text"] = json.dumps(text_json)
                            except json.JSONDecodeError:
                                pass
                    continue

                if isinstance(content, str) and "canvas_state" in content:
                    try:
                        message_content_json = json.loads(content)
                        if isinstance(message_content_json, dict):
                            message_content_json.pop("canvas_state", None)
                            message_content_json.pop("canvas_state_summary", None)
                            message["content"] = json.dumps(message_content_json)
                    except json.JSONDecodeError:
                        pass

    def _remove_images_from_user_messages(self) -> None:
        """Remove image content from all user messages in the conversation history.

        This reduces token usage for the fallback case when not using previous_response_id.
        When using previous_response_id, OpenAI maintains the full context server-side.
        """
        for message in reversed(self.messages):
            if message.get("role") == "user" and "content" in message:
                content = message["content"]
                if not isinstance(content, list):
                    continue
                text_parts = [part for part in content if isinstance(part, dict) and part.get("type") == "text"]
                if text_parts:
                    text_part = text_parts[0]
                    message["content"] = text_part.get("text", "")

    def _clean_conversation_history(self) -> None:
        """Clean up the conversation history by removing canvas states and images."""
        _logger.debug(f"All messages BEFORE removing canvas_state: \n{self.messages}\n\n")
        self._remove_canvas_state_from_user_messages()
        self._remove_images_from_user_messages()

    def _create_enhanced_prompt_with_image(
        self,
        user_message: str,
        attached_images: Optional[List[str]] = None,
        include_canvas_snapshot: bool = True,
    ) -> Optional[List[Dict[str, Any]]]:
        """Create an enhanced prompt that includes text and optional images.

        Args:
            user_message: The text message from the user
            attached_images: Optional list of data URL images attached by the user
            include_canvas_snapshot: Whether to include the canvas snapshot (vision toggle)

        Returns:
            List of content parts for the message, or None if no images available
        """
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_message}]
        has_images = False

        # Add canvas snapshot if vision is enabled
        if include_canvas_snapshot:
            try:
                with open("canvas_snapshots/canvas.png", "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    })
                    has_images = True
            except Exception as e:
                error_msg = f"Failed to load canvas image: {e}"
                print(error_msg)  # Console output
                _logger.error(error_msg)  # File logging

        # Add user-attached images (these are already data URLs)
        if attached_images:
            for img_url in attached_images:
                if isinstance(img_url, str) and img_url.startswith("data:image"):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img_url}
                    })
                    has_images = True

        return content if has_images else None

    def _prepare_message_content(self, full_prompt: str) -> MessageContent:
        """Prepare message content with optional canvas image for vision-enabled messages.

        Handles both vision toggle (canvas snapshot) and user-attached images.
        Images work independently: attached images are sent regardless of vision toggle.
        """
        telemetry_enabled = self._is_canvas_summary_telemetry_enabled()
        start_time = time.perf_counter() if telemetry_enabled else 0.0
        normalized_prompt = self._normalize_prompt_canvas_state(full_prompt)
        prompt_kind = "text"
        try:
            prompt_json = json.loads(normalized_prompt)
        except json.JSONDecodeError:
            if telemetry_enabled:
                self._log_canvas_summary_telemetry(
                    full_prompt=full_prompt,
                    normalized_prompt=normalized_prompt,
                    normalized_prompt_json=None,
                    prompt_kind=prompt_kind,
                    elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
                )
            return normalized_prompt

        if not isinstance(prompt_json, dict):
            if telemetry_enabled:
                self._log_canvas_summary_telemetry(
                    full_prompt=full_prompt,
                    normalized_prompt=normalized_prompt,
                    normalized_prompt_json=None,
                    prompt_kind=prompt_kind,
                    elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
                )
            return normalized_prompt

        user_message = str(prompt_json.get("user_message", ""))
        use_vision = bool(prompt_json.get("use_vision", False))

        # Extract attached images from the prompt JSON
        attached_images_raw = prompt_json.get("attached_images")
        attached_images: Optional[List[str]] = None
        if isinstance(attached_images_raw, list):
            attached_images = [img for img in attached_images_raw if isinstance(img, str)]

        # If no vision and no attached images, return plain text prompt
        if not use_vision and not attached_images:
            if telemetry_enabled:
                self._log_canvas_summary_telemetry(
                    full_prompt=full_prompt,
                    normalized_prompt=normalized_prompt,
                    normalized_prompt_json=prompt_json,
                    prompt_kind=prompt_kind,
                    elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
                )
            return normalized_prompt

        # Create enhanced prompt with images
        prompt_kind = "multimodal"
        enhanced_prompt = self._create_enhanced_prompt_with_image(
            user_message=user_message,
            attached_images=attached_images,
            include_canvas_snapshot=use_vision,
        )
        result: MessageContent = enhanced_prompt if enhanced_prompt else normalized_prompt
        if telemetry_enabled:
            self._log_canvas_summary_telemetry(
                full_prompt=full_prompt,
                normalized_prompt=normalized_prompt,
                normalized_prompt_json=prompt_json,
                prompt_kind=prompt_kind,
                elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
            )
        return result

    def _normalize_prompt_canvas_state(self, full_prompt: str) -> str:
        """Normalize prompt canvas payload according to summary mode."""
        mode = self._get_canvas_summary_mode()
        if mode == "off":
            return full_prompt

        prompt_json = self._parse_prompt_json(full_prompt)
        if not isinstance(prompt_json, dict):
            return full_prompt

        canvas_state = prompt_json.get("canvas_state")
        if not isinstance(canvas_state, dict):
            return full_prompt

        comparison = compare_canvas_states(canvas_state)
        metrics = comparison.get("metrics", {})
        summary_state = comparison.get("summary", {})
        full_bytes = int(metrics.get("full_bytes", 0) or 0)
        include_full_state = mode == "hybrid" and self._should_include_full_state_in_hybrid(full_bytes)

        summary_payload: Dict[str, Any] = {
            "mode": mode,
            "includes_full_state": include_full_state,
            "metrics": metrics,
        }
        if not include_full_state:
            summary_payload["state"] = summary_state
        prompt_json["canvas_state_summary"] = summary_payload

        if not include_full_state:
            del prompt_json["canvas_state"]

        return json.dumps(prompt_json)

    def _get_canvas_summary_mode(self) -> str:
        raw_mode = os.getenv(self.CANVAS_SUMMARY_MODE_ENV, self.DEFAULT_CANVAS_SUMMARY_MODE).strip().lower()
        if raw_mode in ("off", "hybrid", "summary_only"):
            return raw_mode
        return self.DEFAULT_CANVAS_SUMMARY_MODE

    def _get_canvas_hybrid_full_max_bytes(self) -> int:
        raw_limit = os.getenv(self.CANVAS_HYBRID_MAX_BYTES_ENV, str(self.DEFAULT_CANVAS_HYBRID_FULL_MAX_BYTES))
        try:
            value = int(raw_limit)
            if value <= 0:
                raise ValueError
            return value
        except (TypeError, ValueError):
            return self.DEFAULT_CANVAS_HYBRID_FULL_MAX_BYTES

    def _should_include_full_state_in_hybrid(self, full_bytes: int) -> bool:
        if full_bytes <= 0:
            return False
        return full_bytes <= self._get_canvas_hybrid_full_max_bytes()

    def _is_canvas_summary_telemetry_enabled(self) -> bool:
        raw = os.getenv(self.CANVAS_SUMMARY_TELEMETRY_ENV, "").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _estimate_tokens_from_bytes(self, payload_bytes: int) -> int:
        if payload_bytes <= 0:
            return 0
        return max(1, payload_bytes // 4)

    def _log_canvas_summary_telemetry(
        self,
        *,
        full_prompt: str,
        normalized_prompt: str,
        normalized_prompt_json: Optional[Dict[str, Any]],
        prompt_kind: str,
        elapsed_ms: float,
    ) -> None:
        full_bytes = len(full_prompt.encode("utf-8"))
        normalized_bytes = len(normalized_prompt.encode("utf-8"))
        reduction_pct = 0.0
        if full_bytes > 0:
            reduction_pct = round((1.0 - (normalized_bytes / float(full_bytes))) * 100.0, 2)

        mode = self._get_canvas_summary_mode()
        includes_full_state: Optional[bool] = None
        summary_metrics: Optional[Dict[str, Any]] = None
        if isinstance(normalized_prompt_json, dict):
            summary_payload = normalized_prompt_json.get("canvas_state_summary")
            if isinstance(summary_payload, dict):
                includes_full_state_raw = summary_payload.get("includes_full_state")
                if isinstance(includes_full_state_raw, bool):
                    includes_full_state = includes_full_state_raw
                metrics_raw = summary_payload.get("metrics")
                if isinstance(metrics_raw, dict):
                    summary_metrics = metrics_raw

        telemetry_payload: Dict[str, Any] = {
            "mode": mode,
            "prompt_kind": prompt_kind,
            "normalize_elapsed_ms": round(elapsed_ms, 2),
            "input_bytes": full_bytes,
            "normalized_bytes": normalized_bytes,
            "input_estimated_tokens": self._estimate_tokens_from_bytes(full_bytes),
            "normalized_estimated_tokens": self._estimate_tokens_from_bytes(normalized_bytes),
            "reduction_pct": reduction_pct,
            "includes_full_state": includes_full_state,
        }
        if summary_metrics is not None:
            telemetry_payload["summary_metrics"] = summary_metrics

        _logger.info("canvas_prompt_telemetry %s", json.dumps(telemetry_payload, sort_keys=True))

    def _create_error_response(
        self,
        error_message: str = "I encountered an error processing your request. Please try again.",
    ) -> SimpleNamespace:
        """Create an error response that matches OpenAI's response structure."""
        return SimpleNamespace(
            message=SimpleNamespace(content=error_message, tool_calls=[]),
            finish_reason="error"
        )

    def _create_tool_message(self, tool_call_id: Optional[str], content: str) -> MessageDict:
        """Create a tool message in response to a tool call."""
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    def _append_tool_messages(self, tool_calls: Sequence[Any] | None) -> None:
        """Create and append placeholder tool messages for each tool call."""
        if tool_calls:
            for tool_call in tool_calls:
                tool_message = self._create_tool_message(
                    getattr(tool_call, "id", None),
                    "Awaiting result..."
                )
                self.messages.append(tool_message)

    def _update_tool_messages_with_results(self, tool_call_results: str) -> None:
        """Update placeholder tool messages with actual results from the client."""
        try:
            results = json.loads(tool_call_results)
            if not isinstance(results, dict):
                return
        except (json.JSONDecodeError, TypeError):
            return

        results_str = json.dumps(results)
        for message in reversed(self.messages):
            if message.get("role") == "tool":
                message["content"] = results_str
                return

    def _parse_prompt_json(self, full_prompt: str) -> Optional[Dict[str, Any]]:
        """Parse the prompt JSON and return the parsed dict, or None on failure."""
        try:
            prompt_json = json.loads(full_prompt)
            return prompt_json if isinstance(prompt_json, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
