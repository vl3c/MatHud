"""
MatHud Anthropic API Provider

Anthropic Claude API implementation as a self-contained provider module.
Inherits shared functionality from OpenAIAPIBase for message history management.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from static.ai_model import AIModel
from static.env_config import get_api_key
from static.functions_definitions import FunctionDefinition
from static.openai_api_base import MessageDict, OpenAIAPIBase, StreamEvent, ToolMode
from static.providers import PROVIDER_ANTHROPIC, ProviderRegistry
from static.response_metrics import ResponseMetricsTracker, tool_call_argument_text, usage_from_anthropic

_logger = logging.getLogger("mathud")

# Values the Messages API accepts for output_config.effort.
_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})

# Streaming requests have no HTTP timeout concern, so they get more room than the
# non-streaming default: thinking tokens count toward max_tokens on adaptive-thinking models.
_STREAM_MAX_TOKENS = 32000

# Output-token ceilings for models that allow fewer than the 128K output tokens of
# the other current Claude models; request max_tokens is capped to them.
_MODEL_MAX_OUTPUT_TOKENS: Dict[str, int] = {"claude-haiku-4-5": 64000}

# Stop reasons for a reply that ran out of room before it was finished.
_TRUNCATION_STOP_REASONS = frozenset({"max_tokens", "model_context_window_exceeded"})

_PAUSE_NOTE = "The model paused before finishing its reply. Send a follow-up message to continue."


@dataclass
class _StopOutcome:
    """How a Claude reply ended and which of its parts MatHud may use."""

    finish_reason: str
    tool_calls: List[Dict[str, Any]]  # the calls that are safe to run
    note: str  # shown to the user after the reply; empty for a normal stop
    keep_text: bool  # whether the reply text enters the conversation history


def _resolve_stop(
    stop_reason: Optional[str],
    stop_details: Any,
    tool_calls: List[Dict[str, Any]],
    cut_off_tool_id: Optional[str],
    max_tokens: int,
) -> _StopOutcome:
    """Decide what to do with a reply given the API's ``stop_reason``.

    Args:
        stop_reason: The reply's stop reason, None when the stream did not report one.
        stop_details: The reply's ``stop_details`` (set on refusals).
        tool_calls: The reply's tool calls, OpenAI-style.
        cut_off_tool_id: ID of the tool call in the reply's last content block, the one
            being written when a truncated reply stopped.
        max_tokens: The request's max_tokens, for the cut-off note.
    """
    if stop_reason == "refusal":
        # A refusal can stop mid-reply, even mid tool call: run nothing and keep nothing.
        _logger.warning("[Anthropic API] Reply refused; stop_details=%s", stop_details)
        return _StopOutcome("refusal", [], _refusal_note(stop_details), keep_text=False)
    if stop_reason == "pause_turn":
        # Only server tools pause a turn and MatHud uses none, so stop instead of resuming.
        _logger.warning("[Anthropic API] Reply paused (pause_turn); dropping %d tool call(s)", len(tool_calls))
        return _StopOutcome("stop", [], _PAUSE_NOTE, keep_text=True)
    if stop_reason in _TRUNCATION_STOP_REASONS:
        return _resolve_truncation(stop_reason, tool_calls, cut_off_tool_id, max_tokens)
    return _StopOutcome("tool_calls" if tool_calls else "stop", tool_calls, "", keep_text=True)


def _resolve_truncation(
    stop_reason: str,
    tool_calls: List[Dict[str, Any]],
    cut_off_tool_id: Optional[str],
    max_tokens: int,
) -> _StopOutcome:
    """Keep only the tool calls the model finished before the reply was cut off.

    The call in the last content block was being written when the limit hit, so its
    arguments are incomplete (or empty) and it must not run. Finished calls still run
    and the tool loop continues; with none left, the turn ends as truncated ("length").
    """
    runnable = [call for call in tool_calls if call.get("id") != cut_off_tool_id and _has_complete_arguments(call)]
    dropped = len(tool_calls) - len(runnable)
    _logger.warning(
        "[Anthropic API] Reply cut off (%s); dropping %d unfinished tool call(s)",
        stop_reason,
        dropped,
    )
    if stop_reason == "model_context_window_exceeded":
        note = "The reply was cut off because the conversation filled the model's context window."
    else:
        note = f"The reply was cut off at the {max_tokens}-token output limit."
    if dropped == 1:
        note += " 1 unfinished tool call was not run."
    elif dropped > 1:
        note += f" {dropped} unfinished tool calls were not run."
    return _StopOutcome("tool_calls" if runnable else "length", runnable, note, keep_text=True)


def _has_complete_arguments(tool_call: Dict[str, Any]) -> bool:
    """Whether a tool call's arguments are a whole JSON object (empty means no arguments)."""
    function = tool_call.get("function")
    arguments = function.get("arguments") if isinstance(function, dict) else None
    if isinstance(arguments, dict) or not arguments:
        return True
    try:
        return isinstance(json.loads(arguments), dict)
    except (TypeError, ValueError):
        return False


def _refusal_note(stop_details: Any) -> str:
    """User-facing explanation of a refusal, with its category and explanation when given."""
    category = _stop_detail(stop_details, "category")
    explanation = _stop_detail(stop_details, "explanation")
    note = "Claude declined to continue this request"
    note += f" (category: {category})." if category else "."
    return f"{note} {explanation}" if explanation else note


def _stop_detail(stop_details: Any, field: str) -> str:
    value = stop_details.get(field) if isinstance(stop_details, dict) else getattr(stop_details, field, None)
    return str(value) if value else ""


def _note_suffix(text: str, note: str) -> str:
    """The text to append to a reply for its note, separated from any reply text."""
    if not note:
        return ""
    return f"\n\n{note}" if text else note


def _get_anthropic_api_key() -> str:
    """Get the Anthropic API key from environment."""
    # required=True (default): Anthropic is an explicitly opted-in provider,
    # so a missing key is a configuration error rather than a graceful fallback.
    return get_api_key("ANTHROPIC_API_KEY")


class AnthropicAPI(OpenAIAPIBase):
    """Anthropic Claude API provider.

    Converts OpenAI-style messages and tools to Anthropic format.
    Implements streaming with the same event interface as OpenAI providers.
    """

    def __init__(
        self,
        model: Optional[AIModel] = None,
        temperature: float = 0.2,
        tools: Optional[Sequence[FunctionDefinition]] = None,
        max_tokens: int = 16000,
        tool_mode: ToolMode = "full",
    ) -> None:
        """Initialize Anthropic API client.

        Args:
            model: AI model to use. Defaults to Claude Sonnet 5.
            temperature: Sampling temperature, sent only to models that accept it
                (Claude Haiku 4.5); see _apply_temperature.
            tools: Custom tool definitions.
            max_tokens: Maximum tokens in a non-streaming response. Thinking tokens count
                toward this limit on adaptive-thinking models. Streaming requests use at
                least _STREAM_MAX_TOKENS.
            tool_mode: Tool mode - "full" or "search".
        """
        # Import anthropic here to avoid import errors if not installed
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package is required for Anthropic provider. Install with: pip install anthropic"
            ) from e

        self._anthropic_client = anthropic.Anthropic(api_key=_get_anthropic_api_key())

        # Initialize base class (uses OpenAI client for compatibility, but we won't use it)
        # We override the key methods to use Anthropic instead
        self.model: AIModel = model if model is not None else AIModel.from_identifier("claude-sonnet-5")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._tool_mode: ToolMode = tool_mode
        self._custom_tools: Optional[Sequence[FunctionDefinition]] = tools
        self._injected_tools: bool = False
        self.tools: Sequence[FunctionDefinition] = self._resolve_tools()

        # Anthropic takes the system prompt as a request parameter (see _build_system_prompt)
        self.messages: List[MessageDict] = []

        # Dummy OpenAI client - not used but needed for base class compatibility
        self.client = None

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.messages = []
        self._last_canvas_state = None

    def _convert_tools_to_anthropic(self) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tools to Anthropic format.

        OpenAI: {"type": "function", "function": {"name": "x", "parameters": {...}}}
        Anthropic: {"name": "x", "input_schema": {...}}
        """
        anthropic_tools = []
        for tool in self.tools:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function", {})
            if not isinstance(func, dict):
                continue

            anthropic_tool = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            }
            anthropic_tools.append(anthropic_tool)

        return anthropic_tools

    def _convert_messages_to_anthropic(self) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Anthropic format.

        Handles:
        - user messages → user messages
        - assistant messages with tool_calls → assistant with tool_use content blocks
        - tool messages → user messages with tool_result content blocks

        Thinking blocks are never stored, so they are never sent back. Claude Fable 5.1
        and Opus 5.5 bind them to the model and the conversation, and MatHud edits
        history between requests (canvas blocks and images are stripped, search mode
        swaps tools), so replaying them after such edits would be rejected.

        Assistant messages with no text and no tool calls (a thinking-only reply, for
        instance) are skipped: the API rejects empty assistant content, and it merges
        the consecutive user messages this leaves.
        """
        anthropic_messages: List[Dict[str, Any]] = []

        for msg in self.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "developer" or role == "system":
                # System messages are handled via system parameter
                continue

            elif role == "user":
                # Convert user message
                if isinstance(content, str):
                    anthropic_messages.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    # Handle multi-modal content (images)
                    anthropic_content = self._convert_content_blocks(content)
                    anthropic_messages.append({"role": "user", "content": anthropic_content})

            elif role == "assistant":
                # Convert assistant message
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    # Convert to tool_use content blocks
                    content_blocks = []
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        func = tc.get("function", {})
                        args_str = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except json.JSONDecodeError:
                            args = {}
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": func.get("name", ""),
                                "input": args,
                            }
                        )
                    if content_blocks:
                        anthropic_messages.append({"role": "assistant", "content": content_blocks})
                elif content:
                    anthropic_messages.append({"role": "assistant", "content": content})

            elif role == "tool":
                # Convert tool result - Anthropic expects this as a user message with tool_result
                tool_call_id = msg.get("tool_call_id", "")
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content if isinstance(content, str) else json.dumps(content),
                }
                # Check if last message is a user message with tool_results, merge if so
                if anthropic_messages and anthropic_messages[-1].get("role") == "user":
                    last_content = anthropic_messages[-1].get("content", [])
                    if isinstance(last_content, list):
                        last_content.append(tool_result)
                    else:
                        anthropic_messages[-1]["content"] = [{"type": "text", "text": last_content}, tool_result]
                else:
                    anthropic_messages.append({"role": "user", "content": [tool_result]})

        return anthropic_messages

    def _convert_content_blocks(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI content blocks to Anthropic format."""
        anthropic_blocks = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")

            if block_type == "text":
                anthropic_blocks.append({"type": "text", "text": block.get("text", "")})
            elif block_type == "image_url":
                # Convert image URL to Anthropic format
                image_url = block.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                if url.startswith("data:"):
                    # Parse data URL
                    parts = url.split(",", 1)
                    if len(parts) == 2:
                        media_type_part = parts[0].replace("data:", "").replace(";base64", "")
                        anthropic_blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type_part,
                                    "data": parts[1],
                                },
                            }
                        )

        return anthropic_blocks

    def _parse_and_prepare_message(self, full_prompt: str) -> Optional[Dict[str, Any]]:
        """Parse the prompt and prepare the user message.

        Returns the prepared message dict or None if this is a tool result.
        """
        prompt_json = self._parse_prompt_json(full_prompt)
        if prompt_json and prompt_json.get("tool_call_results"):
            self._apply_tool_call_results(prompt_json)
            return None

        message_content = self._prepare_message_content(full_prompt)
        return {"role": "user", "content": message_content}

    def _apply_temperature(self, request_kwargs: Dict[str, Any]) -> None:
        """Add the temperature parameter only for models that accept it.

        Adaptive-thinking Claude models (Claude Fable 5.1, Opus 5.5, Sonnet 5) reject
        the ``temperature`` sampling parameter with a 400 error. Those models are
        flagged ``is_reasoning_model`` in the registry, so only send ``temperature`` for
        non-reasoning models (e.g. Claude Haiku 4.5) that still support it.

        The anthropic SDK (1.0+) no longer takes sampling parameters as keyword
        arguments, so ``temperature`` goes into the request body via ``extra_body``.
        """
        if not self.model.is_reasoning_model:
            request_kwargs.setdefault("extra_body", {})["temperature"] = self.temperature

    def _apply_effort(self, request_kwargs: Dict[str, Any]) -> None:
        """Send the model's configured effort level as ``output_config.effort``.

        Thinking runs adaptive by default on Claude Fable 5.1, Opus 5.5 and Sonnet 5,
        so effort is the control for how much the model thinks. Models without a
        ``reasoning_effort`` in the registry (Claude Haiku 4.5, which rejects the
        parameter) get no ``output_config`` and keep the API default.
        """
        effort = self.model.reasoning_effort
        if not effort:
            return
        if effort not in _EFFORT_LEVELS:
            _logger.warning("[Anthropic API] Ignoring unsupported effort %r for model %s", effort, self.model.id)
            return
        request_kwargs["output_config"] = {"effort": effort}

    def _request_max_tokens(self, streaming: bool) -> int:
        """max_tokens for a request, capped to the model's output limit where known.

        Streaming requests get at least _STREAM_MAX_TOKENS; non-streaming ones keep
        ``self.max_tokens`` so they stay well within the SDK's HTTP timeout.
        """
        max_tokens = max(self.max_tokens, _STREAM_MAX_TOKENS) if streaming else self.max_tokens
        return min(max_tokens, _MODEL_MAX_OUTPUT_TOKENS.get(self.model.id, max_tokens))

    def create_chat_completion(self, full_prompt: str) -> Any:
        """Create chat completion with Anthropic API."""
        user_message = self._parse_and_prepare_message(full_prompt)
        if user_message is not None:
            self.messages.append(user_message)
        metrics = self._start_response_metrics("anthropic_messages", streamed=False)

        try:
            anthropic_messages = self._convert_messages_to_anthropic()
            anthropic_tools = self._convert_tools_to_anthropic()

            # Anthropic API doesn't accept empty tools list - must be None or non-empty
            create_kwargs: Dict[str, Any] = {
                "model": self.model.id,
                "max_tokens": self._request_max_tokens(streaming=False),
                "system": self._build_system_prompt(),
                "messages": anthropic_messages,
            }
            self._apply_temperature(create_kwargs)
            self._apply_effort(create_kwargs)
            if anthropic_tools:
                create_kwargs["tools"] = anthropic_tools

            response = self._anthropic_client.messages.create(**create_kwargs)
        except Exception as e:
            error_msg = f"[Anthropic API] Error during API call: {e}"
            print(error_msg)
            _logger.error(error_msg)
            self._finish_response_metrics(metrics, "error", 0, error=str(e))
            return self._create_error_response()

        # Convert Anthropic response to OpenAI-like format
        processed = self._process_anthropic_response(response)
        usage = getattr(response, "usage", None)
        if usage is not None:
            metrics.record_usage(usage_from_anthropic(usage))
        tool_calls = getattr(processed.message, "tool_calls", None) or []
        self._finish_response_metrics(metrics, processed.finish_reason, len(tool_calls))
        return processed

    def _process_anthropic_response(self, response: Any) -> Any:
        """Process Anthropic response and update conversation history."""
        text_content = ""
        all_tool_calls: List[Dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                all_tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )

        last_block = response.content[-1] if response.content else None
        cut_off_tool_id = getattr(last_block, "id", None) if getattr(last_block, "type", "") == "tool_use" else None
        outcome = _resolve_stop(
            getattr(response, "stop_reason", None),
            getattr(response, "stop_details", None),
            all_tool_calls,
            cut_off_tool_id,
            self._request_max_tokens(streaming=False),
        )
        tool_calls = outcome.tool_calls
        self._finalize_anthropic_stream(text_content if outcome.keep_text else "", tool_calls)

        # Return OpenAI-like response object
        return SimpleNamespace(
            message=SimpleNamespace(
                content=text_content + _note_suffix(text_content, outcome.note),
                tool_calls=[
                    SimpleNamespace(
                        id=tc["id"],
                        function=SimpleNamespace(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                    for tc in tool_calls
                ]
                if tool_calls
                else None,
            ),
            finish_reason=outcome.finish_reason,
        )

    def create_chat_completion_stream(self, full_prompt: str) -> Iterator[StreamEvent]:
        """Stream chat completion tokens with Anthropic API.

        The final event carries the request's ``metrics`` (see static/response_metrics.py).
        """
        user_message = self._parse_and_prepare_message(full_prompt)
        if user_message is not None:
            self.messages.append(user_message)

        accumulated_text = ""
        tool_calls: List[Dict[str, Any]] = []
        current_tool: Optional[Dict[str, Any]] = None
        cut_off_tool_id: Optional[str] = None  # the tool call in the latest content block
        stop_reason: Optional[str] = None
        stop_details: Any = None
        max_tokens = self._request_max_tokens(streaming=True)
        metrics = self._start_response_metrics("anthropic_messages")

        try:
            anthropic_messages = self._convert_messages_to_anthropic()
            anthropic_tools = self._convert_tools_to_anthropic()

            # Anthropic API doesn't accept empty tools list - must be None or non-empty
            stream_kwargs: Dict[str, Any] = {
                "model": self.model.id,
                "max_tokens": max_tokens,
                "system": self._build_system_prompt(),
                "messages": anthropic_messages,
            }
            self._apply_temperature(stream_kwargs)
            self._apply_effort(stream_kwargs)
            if anthropic_tools:
                stream_kwargs["tools"] = anthropic_tools

            with self._anthropic_client.messages.stream(**stream_kwargs) as stream:
                for event in stream:
                    event_type = getattr(event, "type", "")
                    self._record_stream_metrics(event, event_type, metrics)

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        cut_off_tool_id = None
                        if block and getattr(block, "type", "") == "tool_use":
                            current_tool = {
                                "id": getattr(block, "id", ""),
                                "function": {
                                    "name": getattr(block, "name", ""),
                                    "arguments": "",
                                },
                            }
                            cut_off_tool_id = current_tool["id"]

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            delta_type = getattr(delta, "type", "")
                            if delta_type == "text_delta":
                                text = getattr(delta, "text", "")
                                if text:
                                    accumulated_text += text
                                    metrics.add_output_text(text)
                                    yield {"type": "token", "text": text}
                            elif delta_type == "input_json_delta" and current_tool:
                                partial_json = getattr(delta, "partial_json", "")
                                if partial_json:
                                    current_tool["function"]["arguments"] += partial_json

                    elif event_type == "content_block_stop":
                        if current_tool:
                            tool_calls.append(current_tool)
                            current_tool = None

                    elif event_type == "message_delta":
                        delta = getattr(event, "delta", None)
                        stop_reason = getattr(delta, "stop_reason", None) or stop_reason
                        stop_details = getattr(delta, "stop_details", None) or stop_details

        except Exception as exc:
            error_msg = f"[Anthropic API] Streaming exception: {exc}"
            print(error_msg)
            _logger.error(error_msg)
            yield {"type": "token", "text": "\n"}
            yield {
                "type": "final",
                "ai_message": "I encountered an error processing your request. Please try again.",
                "ai_tool_calls": [],
                "finish_reason": "error",
                "metrics": dict(self._finish_response_metrics(metrics, "error", 0, error=str(exc))),
            }
            return

        if current_tool is not None and stop_reason in _TRUNCATION_STOP_REASONS:
            tool_calls.append(current_tool)  # never closed; dropped below and counted in the note
        metrics.add_output_text(tool_call_argument_text(tool_calls))
        outcome = _resolve_stop(stop_reason, stop_details, tool_calls, cut_off_tool_id, max_tokens)

        # Update conversation history
        self._finalize_anthropic_stream(accumulated_text if outcome.keep_text else "", outcome.tool_calls)

        # Prepare tool calls for response
        ai_tool_calls = self._prepare_tool_calls_for_response(outcome.tool_calls)
        note = _note_suffix(accumulated_text, outcome.note)
        if note:
            yield {"type": "token", "text": note}

        yield {
            "type": "final",
            "ai_message": accumulated_text + note,
            "ai_tool_calls": ai_tool_calls,
            "finish_reason": outcome.finish_reason,
            "metrics": dict(self._finish_response_metrics(metrics, outcome.finish_reason, len(ai_tool_calls))),
        }

    @staticmethod
    def _record_stream_metrics(event: Any, event_type: str, metrics: ResponseMetricsTracker) -> None:
        """Feed one Anthropic stream event's output timing and usage to the metrics."""
        if event_type == "message_start":
            usage = getattr(getattr(event, "message", None), "usage", None)
            if usage is not None:
                metrics.record_usage(usage_from_anthropic(usage))
        elif event_type == "message_delta":
            usage = getattr(event, "usage", None)
            if usage is not None:
                metrics.record_usage(usage_from_anthropic(usage))
        elif event_type == "content_block_start":
            block_type = getattr(getattr(event, "content_block", None), "type", "")
            if block_type == "tool_use":
                metrics.mark_output("tool_call")
            elif block_type in ("thinking", "redacted_thinking"):
                metrics.mark_output("reasoning")
        elif event_type == "content_block_delta":
            delta_type = getattr(getattr(event, "delta", None), "type", "")
            if delta_type == "text_delta":
                metrics.mark_output("content")
            elif delta_type == "input_json_delta":
                metrics.mark_output("tool_call")
            elif delta_type in ("thinking_delta", "signature_delta"):
                metrics.mark_output("reasoning")

    def _finalize_anthropic_stream(self, accumulated_text: str, tool_calls: List[Dict[str, Any]]) -> None:
        """Record a reply's text and tool calls in the conversation history.

        A reply with neither (a thinking-only reply, a refusal, or a cut-off with no
        finished tool call) adds no assistant message: the API rejects empty ones.
        """
        if not accumulated_text and not tool_calls:
            self._clean_conversation_history()
            return

        # Create assistant message
        assistant_message: MessageDict = {"role": "assistant", "content": accumulated_text}
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                }
                for tc in tool_calls
            ]
        self.messages.append(assistant_message)

        # Append placeholder tool messages
        for tc in tool_calls:
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": "Awaiting result...",
                }
            )

        self._clean_conversation_history()

    def _prepare_tool_calls_for_response(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare tool calls for the final response."""
        result = []
        for tc in tool_calls:
            func = tc.get("function", {})
            func_name = func.get("name", "")
            func_args_raw = func.get("arguments", "{}")
            try:
                func_args = json.loads(func_args_raw) if func_args_raw else {}
            except json.JSONDecodeError:
                func_args = {}
            result.append(
                {
                    "id": tc.get("id"),
                    "function_name": func_name,
                    "arguments": func_args,
                }
            )
        return result


# Self-register with provider registry
ProviderRegistry.register(PROVIDER_ANTHROPIC, AnthropicAPI)
