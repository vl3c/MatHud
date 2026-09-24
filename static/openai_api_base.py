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
import re
import time
from collections.abc import Iterator, Sequence
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import httpx2
from openai import APITimeoutError, OpenAI

from static.ai_model import AIModel
from static.config import CANVAS_SNAPSHOT_PATH
from static.env_config import get_api_key
from static.canvas_state_formatter import CanvasFormat, parse_canvas_format, render_state, render_update
from static.canvas_state_summarizer import compare_canvas_states
from static.functions_definitions import FUNCTIONS, FunctionDefinition
from static.token_estimation import estimate_tokens_from_bytes

# Use the shared MatHud logger for file logging
_logger = logging.getLogger("mathud")

MessageContent = Union[str, List[Dict[str, Any]]]
MessageDict = Dict[str, Any]
StreamEvent = Dict[str, Any]

# Tool mode type
ToolMode = Literal["full", "search"]

TOOL_RESULT_PLACEHOLDER = "Awaiting result..."

PROVIDER_TIMEOUT_MESSAGE = (
    "The AI provider timed out before responding. Please try again or switch to a different model."
)


def stream_error_user_message(exc: BaseException, default: str) -> str:
    """Return the user-facing message for a streaming failure.

    The OpenAI SDK reports timeouts as APITimeoutError. Older SDK versions let a
    stall after streaming had begun escape as a raw transport TimeoutException,
    so httpx2.TimeoutException is still matched as a fallback.
    """
    if isinstance(exc, (APITimeoutError, httpx2.TimeoutException)):
        return PROVIDER_TIMEOUT_MESSAGE
    return default


# Environment variable selecting how many tools the model sees up front:
# "search" (default) exposes search_tools plus essentials, "full" exposes every tool.
TOOL_EXPOSURE_ENV = "MATHUD_TOOL_EXPOSURE"


def get_configured_tool_mode() -> ToolMode:
    """Return the tool mode configured by MATHUD_TOOL_EXPOSURE (default: "search")."""
    raw = os.getenv(TOOL_EXPOSURE_ENV, "search").strip().lower()
    if raw == "full":
        return "full"
    if raw != "search":
        _logger.warning("Unknown %s value %r; using 'search'", TOOL_EXPOSURE_ENV, raw)
    return "search"


# Environment variables selecting how canvas state reaches the model (see
# static/canvas_state_formatter.py): "text" (one object per line with computed
# facts), "min_json" (noise-stripped JSON) or "json" (the raw prompt JSON, the
# original behaviour), plus an optional token budget for the canvas block.
CANVAS_FORMAT_ENV = "MATHUD_CANVAS_FORMAT"
CANVAS_BUDGET_ENV = "MATHUD_CANVAS_BUDGET_TOKENS"

# Non-json formats put the canvas in front of the user's text inside these markers.
CANVAS_BLOCK_START = "<canvas>"
CANVAS_BLOCK_END = "</canvas>"
_LEADING_CANVAS_BLOCK = re.compile(r"\A<canvas>\n.*?\n</canvas>(?:\n\n)?", re.DOTALL)

_DEV_MSG_INTRO = "You are an educational graphing calculator AI interface that can draw shapes, perform calculations and help users explore mathematics. Use the provided tools for calculations rather than computing results yourself, so every result shown comes from the math engine."
_DEV_MSG_OUTRO = "Never use emoticons or emoji in your responses. When performing multiple steps, include a succinct summary of all actions taken in your final response. INFO: Point labels and coordinates are hardcoded to be shown next to all points on the canvas."
_CANVAS_PROMPT_SENTENCES: Dict[CanvasFormat, str] = {
    "json": "Canvas state is included with user messages; base your actions on it. For large scenes it may be summarized to reduce noise; when you need complete details, call get_current_canvas_state. Canvas state may be stale after tool calls, so re-check live state between actions when needed.",
    "min_json": "Each user message starts with the current canvas as compact JSON in a <canvas> block, and after tool calls the last tool result ends with the [canvas changes] (one changed object per line).",
    "text": "Each user message starts with the current canvas in a <canvas> block (one object per line as name = definition, followed after tool calls by [canvas changes] at the end of the last tool result); the lengths, areas and angles it lists come from the math engine and can be quoted directly.",
}


def _is_canvas_state_result(value: Any) -> bool:
    """True for a get_current_canvas_state result value: ``{"type": "canvas_state", "value": {...}}``."""
    return isinstance(value, dict) and value.get("type") == "canvas_state" and isinstance(value.get("value"), dict)


def build_developer_message(canvas_format: CanvasFormat) -> str:
    """Return the system prompt describing how canvas state is presented in ``canvas_format``."""
    return f"{_DEV_MSG_INTRO} {_CANVAS_PROMPT_SENTENCES[canvas_format]} {_DEV_MSG_OUTRO}"


# Essential tool names that should always be available after injection
ESSENTIAL_TOOLS = frozenset(
    {
        "search_tools",
        "undo",
        "redo",
        "get_current_canvas_state",
    }
)


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

    # Canvas format and canvas-block token budget (None = unlimited) unless
    # MATHUD_CANVAS_FORMAT / MATHUD_CANVAS_BUDGET_TOKENS override them.
    DEFAULT_CANVAS_FORMAT: CanvasFormat = "text"
    DEFAULT_CANVAS_BUDGET_TOKENS: Optional[int] = 4000

    # System prompt for the default canvas format; build_developer_message covers the others.
    DEV_MSG = build_developer_message(DEFAULT_CANVAS_FORMAT)

    # get_current_canvas_state results get this multiple of the canvas budget: the model
    # asked for the state, but a huge scene must still not flood the context.
    TOOL_RESULT_BUDGET_MULTIPLIER = 2

    # Last canvas state shown to the model, so tool results can report what changed.
    _last_canvas_state: Optional[Dict[str, Any]] = None

    SEARCH_MODE_MSG = """Tool loading: at the start only search_tools and a few essential tools (undo, redo, get_current_canvas_state) are available. Before using any other tool, call search_tools with a short description of what you want to do (e.g. "plot a function", "evaluate an expression at a point"); the matching tools are then loaded for your following calls until you give your final answer. Calls to tools that were not loaded fail."""

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
        # required=False: OpenAI is the default provider but the app can start
        # without an OpenAI key when the user configures a third-party provider
        # (Anthropic, OpenRouter).  A missing key degrades gracefully — actual
        # OpenAI API calls will fail with an auth error at call time.
        api_key: str = get_api_key("OPENAI_API_KEY", required=False, fallback="")
        if not api_key:
            logging.getLogger("mathud").warning("OPENAI_API_KEY not found. OpenAI models will be unavailable.")
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
        self.messages: List[MessageDict] = [{"role": "developer", "content": self._build_system_prompt()}]

    def _build_system_prompt(self) -> str:
        """Return the system prompt, explaining search-first tool loading when it is active."""
        developer_message = build_developer_message(self._get_canvas_format())
        if self._tool_mode == "search" and self._custom_tools is None:
            return f"{developer_message} {OpenAIAPIBase.SEARCH_MODE_MSG}"
        return developer_message

    def _refresh_system_prompt(self) -> None:
        """Rewrite the leading system/developer message after the tool mode changes."""
        if self.messages and self.messages[0].get("role") in ("developer", "system"):
            self.messages[0]["content"] = self._build_system_prompt()

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
                self._refresh_system_prompt()
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
        self.messages = [{"role": "developer", "content": self._build_system_prompt()}]
        self._last_canvas_state = None

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
        """Remove canvas state from user messages in the conversation history.

        JSON prompts lose their state in every user message. A <canvas> block is
        kept on the latest user message, where tool results report changes
        against it, and is removed from older ones.
        """
        self._strip_canvas_blocks(keep_latest=True)
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

    def _strip_canvas_blocks(self, keep_latest: bool) -> None:
        """Remove the leading <canvas> block from user messages (optionally not the latest one)."""
        user_messages = [m for m in self.messages if m.get("role") == "user"]
        if keep_latest:
            user_messages = user_messages[:-1]
        for message in user_messages:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = _LEADING_CANVAS_BLOCK.sub("", content, count=1)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                        part["text"] = _LEADING_CANVAS_BLOCK.sub("", part["text"], count=1)

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
                with open(CANVAS_SNAPSHOT_PATH, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}})
                    has_images = True
            except Exception as e:
                error_msg = f"Failed to load canvas image: {e}"
                print(error_msg)  # Console output
                _logger.error(error_msg)  # File logging

        # Add user-attached images (these are already data URLs)
        if attached_images:
            for img_url in attached_images:
                if isinstance(img_url, str) and img_url.startswith("data:image"):
                    content.append({"type": "image_url", "image_url": {"url": img_url}})
                    has_images = True

        return content if has_images else None

    def _prepare_message_content(self, full_prompt: str) -> MessageContent:
        """Prepare message content with optional canvas image for vision-enabled messages.

        Handles both vision toggle (canvas snapshot) and user-attached images.
        Images work independently: attached images are sent regardless of vision toggle.
        """
        canvas_format = self._get_canvas_format()
        if canvas_format == "json":
            return self._prepare_json_message_content(full_prompt)
        return self._prepare_canvas_block_message_content(full_prompt, canvas_format)

    def _prepare_canvas_block_message_content(self, full_prompt: str, canvas_format: CanvasFormat) -> MessageContent:
        """Build the <canvas> block plus the user's text, with images when requested."""
        telemetry_enabled = self._is_canvas_summary_telemetry_enabled()
        start_time = time.perf_counter() if telemetry_enabled else 0.0
        prompt_json = self._parse_prompt_json(full_prompt)
        if prompt_json is None:
            return full_prompt

        text = self._build_user_text(prompt_json, canvas_format)
        message_content: MessageContent = text
        prompt_kind = "text"
        attached_images = self._extract_attached_images(prompt_json)
        use_vision = bool(prompt_json.get("use_vision", False))
        if use_vision or attached_images:
            prompt_kind = "multimodal"
            enhanced_prompt = self._create_enhanced_prompt_with_image(
                user_message=text,
                attached_images=attached_images,
                include_canvas_snapshot=use_vision,
            )
            if enhanced_prompt:
                message_content = enhanced_prompt

        if telemetry_enabled:
            self._log_canvas_summary_telemetry(
                full_prompt=full_prompt,
                normalized_prompt=text,
                normalized_prompt_json=None,
                output_content=message_content,
                prompt_kind=prompt_kind,
                elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
            )
        return message_content

    def _build_user_text(self, prompt_json: Dict[str, Any], canvas_format: CanvasFormat) -> str:
        """Return the user's text preceded by the rendered canvas block, if the prompt has a state.

        Older canvas blocks are removed from the history first, so only the newest
        state stays in the conversation.
        """
        user_message = prompt_json.get("user_message")
        text = str(user_message) if user_message is not None else ""
        canvas_state = prompt_json.get("canvas_state")
        if not isinstance(canvas_state, dict):
            return text
        self._strip_canvas_blocks(keep_latest=False)
        self._last_canvas_state = canvas_state
        block = self._render_canvas_block(canvas_state, canvas_format)
        return f"{block}\n\n{text}" if text else block

    def _render_canvas_block(self, canvas_state: Dict[str, Any], canvas_format: CanvasFormat) -> str:
        rendered = render_state(canvas_state, canvas_format, self._get_canvas_budget_tokens())
        return f"{CANVAS_BLOCK_START}\n{rendered}\n{CANVAS_BLOCK_END}"

    @staticmethod
    def _extract_attached_images(prompt_json: Dict[str, Any]) -> Optional[List[str]]:
        attached_images_raw = prompt_json.get("attached_images")
        if not isinstance(attached_images_raw, list):
            return None
        return [img for img in attached_images_raw if isinstance(img, str)]

    def _get_canvas_format(self) -> CanvasFormat:
        """Return the canvas format from MATHUD_CANVAS_FORMAT, else this provider's default."""
        raw = os.getenv(CANVAS_FORMAT_ENV, "").strip()
        if not raw:
            return self.DEFAULT_CANVAS_FORMAT
        canvas_format = parse_canvas_format(raw)
        if canvas_format is None:
            _logger.warning("Unknown %s value %r; using %r", CANVAS_FORMAT_ENV, raw, self.DEFAULT_CANVAS_FORMAT)
            return self.DEFAULT_CANVAS_FORMAT
        return canvas_format

    def _get_canvas_budget_tokens(self) -> Optional[int]:
        """Return the canvas token budget from MATHUD_CANVAS_BUDGET_TOKENS (0 = unlimited), else the default."""
        raw = os.getenv(CANVAS_BUDGET_ENV, "").strip()
        if not raw:
            return self.DEFAULT_CANVAS_BUDGET_TOKENS
        try:
            budget = int(raw)
        except ValueError:
            _logger.warning("Invalid %s value %r; using %r", CANVAS_BUDGET_ENV, raw, self.DEFAULT_CANVAS_BUDGET_TOKENS)
            return self.DEFAULT_CANVAS_BUDGET_TOKENS
        return budget if budget > 0 else None

    def _prepare_json_message_content(self, full_prompt: str) -> MessageContent:
        """Legacy path: send the prompt JSON (with canvas_state or its summary) as the user message."""
        telemetry_enabled = self._is_canvas_summary_telemetry_enabled()
        start_time = time.perf_counter() if telemetry_enabled else 0.0
        normalized_prompt, summary_metrics = self._normalize_prompt_canvas_state_with_metrics(full_prompt)
        prompt_kind = "text"
        message_content: MessageContent = normalized_prompt
        prompt_json: Optional[Dict[str, Any]] = None

        try:
            parsed = json.loads(normalized_prompt)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            prompt_json = parsed
            user_message = str(prompt_json.get("user_message", ""))
            use_vision = bool(prompt_json.get("use_vision", False))

            attached_images = self._extract_attached_images(prompt_json)

            # If vision/images are present, use multimodal payload.
            if use_vision or attached_images:
                prompt_kind = "multimodal"
                enhanced_prompt = self._create_enhanced_prompt_with_image(
                    user_message=user_message,
                    attached_images=attached_images,
                    include_canvas_snapshot=use_vision,
                )
                if enhanced_prompt:
                    message_content = enhanced_prompt

        if telemetry_enabled:
            self._log_canvas_summary_telemetry(
                full_prompt=full_prompt,
                normalized_prompt=normalized_prompt,
                normalized_prompt_json=prompt_json,
                output_content=message_content,
                prompt_kind=prompt_kind,
                elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
                summary_metrics=summary_metrics,
            )
        return message_content

    def _normalize_prompt_canvas_state(self, full_prompt: str) -> str:
        """Normalize prompt canvas payload according to summary mode."""
        return self._normalize_prompt_canvas_state_with_metrics(full_prompt)[0]

    def _normalize_prompt_canvas_state_with_metrics(self, full_prompt: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Normalize the prompt; also return the summary size metrics (telemetry only, never sent)."""
        mode = self._get_canvas_summary_mode()
        if mode == "off":
            return full_prompt, None

        prompt_json = self._parse_prompt_json(full_prompt)
        if not isinstance(prompt_json, dict):
            return full_prompt, None

        canvas_state = prompt_json.get("canvas_state")
        if not isinstance(canvas_state, dict):
            return full_prompt, None

        # Fast path for hybrid mode: keep small full states untouched and avoid
        # running summarization/comparison machinery.
        if mode == "hybrid":
            full_state_bytes = self._measure_canvas_state_bytes(canvas_state)
            if self._should_include_full_state_in_hybrid(full_state_bytes):
                return full_prompt, None

        comparison = compare_canvas_states(canvas_state)
        metrics = comparison.get("metrics", {})
        # If we reach this point, hybrid-under-threshold has already returned
        # via the fast path above, so the full state is always replaced.
        prompt_json["canvas_state_summary"] = {
            "mode": mode,
            "includes_full_state": False,
            "state": comparison.get("summary", {}),
        }
        del prompt_json["canvas_state"]

        return json.dumps(prompt_json), metrics if isinstance(metrics, dict) else None

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

    def _measure_canvas_state_bytes(self, canvas_state: Dict[str, Any]) -> int:
        try:
            return len(json.dumps(canvas_state, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        except (TypeError, ValueError):
            return 0

    def _is_canvas_summary_telemetry_enabled(self) -> bool:
        raw = os.getenv(self.CANVAS_SUMMARY_TELEMETRY_ENV, "").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _log_canvas_summary_telemetry(
        self,
        *,
        full_prompt: str,
        normalized_prompt: str,
        normalized_prompt_json: Optional[Dict[str, Any]],
        output_content: MessageContent,
        prompt_kind: str,
        elapsed_ms: float,
        summary_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        full_prompt_bytes = len(full_prompt.encode("utf-8"))
        normalized_prompt_bytes = len(normalized_prompt.encode("utf-8"))
        output_payload_bytes = normalized_prompt_bytes
        if isinstance(output_content, str):
            output_payload_bytes = len(output_content.encode("utf-8"))
        elif isinstance(output_content, list):
            try:
                output_payload_bytes = len(
                    json.dumps(output_content, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                )
            except (TypeError, ValueError):
                output_payload_bytes = normalized_prompt_bytes

        reduction_pct = 0.0
        if full_prompt_bytes > 0:
            reduction_pct = round((1.0 - (output_payload_bytes / float(full_prompt_bytes))) * 100.0, 2)

        mode = self._get_canvas_summary_mode()
        includes_full_state: Optional[bool] = None
        if isinstance(normalized_prompt_json, dict):
            summary_payload = normalized_prompt_json.get("canvas_state_summary")
            if isinstance(summary_payload, dict):
                includes_full_state_raw = summary_payload.get("includes_full_state")
                if isinstance(includes_full_state_raw, bool):
                    includes_full_state = includes_full_state_raw
            elif mode == "hybrid" and isinstance(normalized_prompt_json.get("canvas_state"), dict):
                includes_full_state = True

        telemetry_payload: Dict[str, Any] = {
            "canvas_format": self._get_canvas_format(),
            "mode": mode,
            "prompt_kind": prompt_kind,
            "normalize_elapsed_ms": round(elapsed_ms, 2),
            "input_bytes": full_prompt_bytes,
            "normalized_prompt_bytes": normalized_prompt_bytes,
            "output_payload_bytes": output_payload_bytes,
            "input_estimated_tokens": estimate_tokens_from_bytes(full_prompt_bytes),
            "normalized_prompt_estimated_tokens": estimate_tokens_from_bytes(normalized_prompt_bytes),
            "output_payload_estimated_tokens": estimate_tokens_from_bytes(output_payload_bytes),
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
        return SimpleNamespace(message=SimpleNamespace(content=error_message, tool_calls=[]), finish_reason="error")

    def _create_tool_message(self, tool_call_id: Optional[str], content: str) -> MessageDict:
        """Create a tool message in response to a tool call."""
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    def _append_tool_messages(self, tool_calls: Sequence[Any] | None) -> None:
        """Create and append placeholder tool messages for each tool call."""
        if tool_calls:
            for tool_call in tool_calls:
                tool_message = self._create_tool_message(getattr(tool_call, "id", None), TOOL_RESULT_PLACEHOLDER)
                self.messages.append(tool_message)

    def _apply_tool_call_results(self, prompt_json: Dict[str, Any]) -> None:
        """Answer the pending tool calls from a tool-results prompt, then report canvas changes."""
        self._update_tool_messages_with_results(prompt_json["tool_call_results"])
        self._append_canvas_changes(prompt_json.get("canvas_state"))

    def _append_canvas_changes(self, canvas_state: Any) -> None:
        """Append what the tool batch changed on the canvas to the batch's last tool message.

        Runs after every result of the batch has been written, so matching results
        to tool-call ids is unaffected. Nothing is added when the canvas did not
        change or the json canvas format is active.
        """
        canvas_format = self._get_canvas_format()
        if canvas_format == "json" or not isinstance(canvas_state, dict):
            return
        pending = self._get_pending_tool_messages()
        if not pending or pending[-1].get("content") == TOOL_RESULT_PLACEHOLDER:
            return
        update = render_update(self._last_canvas_state, canvas_state, canvas_format, self._get_canvas_budget_tokens())
        self._last_canvas_state = canvas_state
        if update:
            pending[-1]["content"] = f"{pending[-1]['content']}\n{update}"

    def _update_tool_messages_with_results(self, tool_call_results: str) -> None:
        """Update placeholder tool messages with actual results from the client.

        Accepts either a list of per-call entries
        (``[{"tool_call_id": ..., "result": {...}}, ...]`` in call order) or the
        legacy single dict of all results.
        """
        try:
            results = json.loads(tool_call_results)
        except (json.JSONDecodeError, TypeError):
            return

        pending = self._get_pending_tool_messages()
        if isinstance(results, list):
            self._apply_per_call_results(pending, results)
        elif isinstance(results, dict):
            self._apply_legacy_results(pending, results)

    def record_tool_call_result(self, tool_call_id: Optional[str], content: str) -> bool:
        """Fill the pending placeholder for one tool call id. Returns True if one was updated."""
        if not tool_call_id:
            return False
        for message in self._get_pending_tool_messages():
            if message.get("tool_call_id") == tool_call_id and message.get("content") == TOOL_RESULT_PLACEHOLDER:
                message["content"] = content
                return True
        return False

    def record_tool_call_result_at(self, position: int, call_count: int, content: str) -> bool:
        """Fill the placeholder of the call at ``position`` of the latest batch of ``call_count`` calls.

        For calls without an id: only applies when the pending tool messages line up
        one-to-one with the batch. Returns True if a placeholder was updated.
        """
        pending = self._get_pending_tool_messages()
        if len(pending) != call_count or not 0 <= position < call_count:
            return False
        message = pending[position]
        if message.get("content") != TOOL_RESULT_PLACEHOLDER:
            return False
        message["content"] = content
        return True

    def _get_pending_tool_messages(self) -> List[MessageDict]:
        """Return the trailing run of tool messages answering the latest tool calls."""
        pending: List[MessageDict] = []
        for message in reversed(self.messages):
            if message.get("role") != "tool":
                break
            pending.append(message)
        pending.reverse()
        return pending

    def _apply_per_call_results(self, pending: List[MessageDict], entries: List[Any]) -> None:
        """Write each per-call result into its own tool message, matched by id then by order.

        Only entries without an id fall back to call order; an entry whose id matches
        no awaiting call (e.g. one already answered) is ignored rather than guessed.
        """
        awaiting = [m for m in pending if m.get("content") == TOOL_RESULT_PLACEHOLDER]
        unmatched: List[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            content = self._format_tool_result(entry.get("result"))
            tool_call_id = entry.get("tool_call_id")
            if not tool_call_id:
                unmatched.append(content)
                continue
            target = next((m for m in awaiting if m.get("tool_call_id") == tool_call_id), None)
            if target is None:
                _logger.warning("Ignoring a result for unknown tool call id %r", tool_call_id)
                continue
            target["content"] = content
            awaiting.remove(target)

        # Entries without a usable id fall back to call order.
        for message, content in zip(list(awaiting), unmatched):
            message["content"] = content
            awaiting.remove(message)

        for message in awaiting:
            message["content"] = "Error: no result was returned for this tool call."

    def _apply_legacy_results(self, pending: List[MessageDict], results: Dict[str, Any]) -> None:
        """Write a legacy combined results dict into the last tool message still awaiting a result.

        Messages already answered (e.g. with a dropped-call error) keep their content.
        """
        awaiting = [m for m in pending if m.get("content") == TOOL_RESULT_PLACEHOLDER]
        if not awaiting:
            return
        awaiting[-1]["content"] = self._format_tool_result(results)
        for message in awaiting[:-1]:
            message["content"] = "See the combined results in the last tool message of this turn."

    def _format_tool_result(self, result: Any) -> str:
        """Return the tool message content for one result (a ``{result_key: value}`` dict).

        get_current_canvas_state values (``{"type": "canvas_state", "value": state}``)
        are rendered in the configured canvas format; a result holding only such a
        state becomes that text. The client has already applied the call's filters.
        The budget is TOOL_RESULT_BUDGET_MULTIPLIER times the canvas budget, since the
        model asked for the state; objects beyond it are listed as omitted with a note
        to request them by name.
        """
        canvas_format = self._get_canvas_format()
        if canvas_format == "json" or not isinstance(result, dict):
            return json.dumps(result)
        budget = self._get_canvas_budget_tokens()
        if budget is not None:
            budget *= self.TOOL_RESULT_BUDGET_MULTIPLIER
        rendered = {
            key: render_state(value["value"], canvas_format, budget) if _is_canvas_state_result(value) else value
            for key, value in result.items()
        }
        if len(result) == 1 and _is_canvas_state_result(next(iter(result.values()))):
            return str(next(iter(rendered.values())))
        return json.dumps(rendered)

    def _parse_prompt_json(self, full_prompt: str) -> Optional[Dict[str, Any]]:
        """Parse the prompt JSON and return the parsed dict, or None on failure."""
        try:
            prompt_json = json.loads(full_prompt)
            return prompt_json if isinstance(prompt_json, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
