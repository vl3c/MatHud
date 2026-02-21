"""
MatHud OpenAI Responses API

Responses API implementation for reasoning models (GPT-5, o3, o4-mini).
Streams reasoning tokens during the thinking phase.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from static.openai_api_base import OpenAIAPIBase, MessageDict, StreamEvent

# Use the shared MatHud logger for file logging
_logger = logging.getLogger("mathud")


class OpenAIResponsesAPI(OpenAIAPIBase):
    """OpenAI Responses API for reasoning models (GPT-5, o3, o4-mini).

    Uses `previous_response_id` for multi-turn conversations, allowing OpenAI
    to manage conversation state server-side. This properly handles images
    across turns without manual stripping or context management.
    """

    def __init__(self) -> None:
        """Initialize the Responses API with log deduplication and response ID tracking."""
        super().__init__()
        self._last_log_message: Optional[str] = None
        self._log_repeat_count: int = 0
        self._previous_response_id: Optional[str] = None

    def _log(self, message: str) -> None:
        """Log a message, collapsing consecutive duplicates with a count."""
        if message == self._last_log_message:
            self._log_repeat_count += 1
        else:
            self._flush_log()
            self._last_log_message = message
            self._log_repeat_count = 1

    def _flush_log(self) -> None:
        """Flush any pending repeated log message to both console and log file."""
        if self._last_log_message is not None:
            if self._log_repeat_count > 1:
                msg = f"{self._last_log_message} (x{self._log_repeat_count})"
            else:
                msg = self._last_log_message
            print(msg)  # Console output
            _logger.info(msg)  # File logging
            self._last_log_message = None
            self._log_repeat_count = 0

    def reset_conversation(self) -> None:
        """Reset the conversation history and clear the previous response ID."""
        super().reset_conversation()
        self._previous_response_id = None
        self._log("[Responses API] Conversation reset, cleared previous_response_id")

    def clear_previous_response_id(self) -> None:
        """Clear the stored response ID (e.g. after user interruption)."""
        if self._previous_response_id is not None:
            self._log("[Responses API] Cleared previous_response_id")
            self._previous_response_id = None

    def add_partial_assistant_message(self, content: str) -> None:
        """Add a partial assistant message and clear stale response ID.

        When the user interrupts, the previous response may have pending
        tool calls that will never be answered.  Clearing the ID prevents
        the next request from referencing that broken state.
        """
        super().add_partial_assistant_message(content)
        self.clear_previous_response_id()

    def _is_regular_message_turn(self) -> bool:
        """Check if this is a regular user message turn (not a tool call continuation).

        Returns True if the last message is a user message without pending tool calls.
        """
        if not self.messages:
            return False
        last_msg = self.messages[-1]
        # Regular message turn if last message is from user and not a tool result
        return last_msg.get("role") == "user" and "tool_call_id" not in last_msg

    def _get_latest_user_message_for_input(self) -> List[Dict[str, Any]]:
        """Get only the latest user message for use with previous_response_id.

        When using previous_response_id, we only need to send the new user message
        as OpenAI maintains the conversation context server-side.
        """
        if not self.messages:
            return []

        # Find the last user message
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Convert content format if needed
                converted_content = self._convert_content_for_responses_api(content)
                return [{"role": "user", "content": converted_content}]

        return []

    def _convert_tools_for_responses_api(self) -> List[Dict[str, Any]]:
        """Convert Chat Completions tool format to Responses API format.

        Chat Completions: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        Responses API: {"type": "function", "name": ..., "description": ..., "parameters": ...}
        """
        converted_tools = []
        for tool in self.tools:
            if isinstance(tool, dict) and tool.get("type") == "function":
                func = tool.get("function", {})
                converted_tools.append(
                    {
                        "type": "function",
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    }
                )
        return converted_tools

    def _convert_messages_to_input(self) -> List[Dict[str, Any]]:
        """Convert chat messages format to Responses API input format.

        The Responses API doesn't support tool_calls in input messages.
        We convert tool call/result pairs into assistant+user message pairs.
        """
        input_messages: List[Dict[str, Any]] = []
        i = 0

        while i < len(self.messages):
            msg = self.messages[i]
            role = msg.get("role", "user")

            if role == "developer":
                i = self._handle_developer_message(msg, input_messages, i)
            elif role == "assistant" and msg.get("tool_calls"):
                i = self._handle_assistant_with_tool_calls(msg, input_messages, i)
            elif role == "tool":
                i = self._handle_orphan_tool_message(i)
            else:
                i = self._handle_regular_message(msg, role, input_messages, i)

        self._flush_log()
        self._log(f"[Responses API] Final input has {len(input_messages)} messages")
        self._flush_log()
        return input_messages

    def _handle_developer_message(self, msg: Dict[str, Any], output: List[Dict[str, Any]], index: int) -> int:
        """Convert developer message to system role for Responses API."""
        content = msg.get("content", "")
        output.append({"role": "system", "content": content})
        self._log(f"[Responses API] Including developer message at index {index}")
        return index + 1

    def _handle_assistant_with_tool_calls(self, msg: Dict[str, Any], output: List[Dict[str, Any]], index: int) -> int:
        """Convert assistant tool calls and results to text message pairs."""
        tool_calls = msg.get("tool_calls", [])
        tool_results, end_index = self._collect_tool_results(index + 1)

        if tool_results:
            assistant_msg = self._create_tool_call_description(tool_calls)
            user_msg = self._create_tool_results_message(tool_results)
            output.append(assistant_msg)
            output.append(user_msg)
            self._log(
                f"[Responses API] Converted tool call+results at index {index}-{end_index - 1} to assistant+user messages"
            )
            return end_index
        else:
            self._log(f"[Responses API] Skipping assistant with pending tool calls at index {index}")
            return index + 1

    def _handle_orphan_tool_message(self, index: int) -> int:
        """Skip standalone tool messages (already handled with their assistant message)."""
        self._log(f"[Responses API] Skipping orphan tool message at index {index}")
        return index + 1

    def _handle_regular_message(self, msg: Dict[str, Any], role: str, output: List[Dict[str, Any]], index: int) -> int:
        """Include regular user/assistant messages, converting content format if needed."""
        content = msg.get("content", "")
        # Convert Chat Completions content format to Responses API format
        converted_content = self._convert_content_for_responses_api(content)
        output.append({"role": role, "content": converted_content})
        self._log(f"[Responses API] Including {role} message at index {index}")
        return index + 1

    def _convert_content_for_responses_api(self, content: Any) -> Any:
        """Convert Chat Completions content format to Responses API format.

        Chat Completions uses: {"type": "text", "text": ...}, {"type": "image_url", "image_url": {...}}
        Responses API uses: {"type": "input_text", "text": ...}, {"type": "input_image", "image_url": ...}
        """
        if not isinstance(content, list):
            return content

        converted = []
        for part in content:
            if not isinstance(part, dict):
                converted.append(part)
                continue

            part_type = part.get("type", "")
            if part_type == "text":
                converted.append({"type": "input_text", "text": part.get("text", "")})
            elif part_type == "image_url":
                image_url = part.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                converted.append({"type": "input_image", "image_url": url})
            else:
                converted.append(part)

        return converted

    def _collect_tool_results(self, start_index: int) -> tuple[List[str], int]:
        """Collect tool result contents starting from the given index.

        Returns:
            Tuple of (list of result contents, index after last tool message)
        """
        tool_results: List[str] = []
        j = start_index

        while j < len(self.messages) and self.messages[j].get("role") == "tool":
            tool_content = self.messages[j].get("content", "")
            if tool_content and tool_content != "Awaiting result...":
                tool_results.append(tool_content)
            j += 1

        return tool_results, j

    def _create_tool_call_description(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an assistant message describing the tool calls made."""
        call_descriptions = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args = func.get("arguments", "{}")
            call_descriptions.append(f"Called {name} with {args}")

        content = "I called the following functions: " + "; ".join(call_descriptions)
        return {"role": "assistant", "content": content}

    def _create_tool_results_message(self, tool_results: List[str]) -> Dict[str, Any]:
        """Create a user message containing the tool results."""
        content = "Function results: " + " | ".join(tool_results)
        return {"role": "user", "content": content}

    def create_response_stream(self, full_prompt: str) -> Iterator[StreamEvent]:
        """Stream response using the Responses API with reasoning support."""
        self._prepare_messages_for_stream(full_prompt)
        state = self._create_stream_state()

        try:
            stream = self._create_api_stream_with_fallback()
            yield from self._process_stream_events(stream, state)
        except Exception as exc:
            yield from self._handle_stream_error(exc)
            return

        yield self._build_final_response(state)

    def _prepare_messages_for_stream(self, full_prompt: str) -> None:
        """Prepare messages for the streaming request."""
        prompt_json = self._parse_prompt_json(full_prompt)
        tool_call_results = prompt_json.get("tool_call_results") if prompt_json else None

        if tool_call_results:
            self._update_tool_messages_with_results(tool_call_results)
        else:
            message_content = self._prepare_message_content(full_prompt)
            user_message: MessageDict = {"role": "user", "content": message_content}
            self.messages.append(user_message)

    def _create_stream_state(self) -> Dict[str, Any]:
        """Create initial state for stream processing."""
        return {
            "accumulated_text": "",
            "tool_calls_accumulator": {},
            "finish_reason": None,
            "reasoning_placeholder_sent": False,
        }

    def _create_api_stream_with_fallback(self) -> Any:
        """Create API stream, falling back if some reasoning params are not supported.

        Notes:
        - Uses `previous_response_id` for multi-turn conversations when available,
          allowing OpenAI to manage context server-side (better for images).
        - Some models may not support `reasoning.summary`.
        - GPT-5.2 defaults to reasoning.effort="none"; MatHud sets effort to "medium" for `gpt-5.2`.
        - If the API rejects a reasoning sub-parameter, retry with a reduced set of reasoning params.
        """
        # Determine if we can use previous_response_id for this turn
        # Only use it for regular user messages, not tool call continuations
        use_previous_response = self._previous_response_id is not None and self._is_regular_message_turn()

        if use_previous_response:
            # With previous_response_id, only send the new user message
            input_messages = self._get_latest_user_message_for_input()
            self._log(f"[Responses API] Using previous_response_id: {self._previous_response_id}")
        else:
            # First turn or tool call - send full history
            input_messages = self._convert_messages_to_input()

        base_kwargs: Dict[str, Any] = {
            "model": self.model.id,
            "input": input_messages,
            "tools": self._convert_tools_for_responses_api(),
            "max_output_tokens": self.max_tokens,
            "stream": True,
        }

        # Add previous_response_id if using stateful conversation
        if use_previous_response:
            base_kwargs["previous_response_id"] = self._previous_response_id

        reasoning: Dict[str, Any] = {"summary": "detailed"}
        if getattr(self.model, "reasoning_effort", None):
            reasoning["effort"] = self.model.reasoning_effort

        reasoning_attempts: List[Optional[Dict[str, Any]]] = []
        if reasoning:
            reasoning_attempts.append(reasoning)
            if "effort" in reasoning and "summary" in reasoning:
                reasoning_attempts.append({"effort": reasoning["effort"]})
                reasoning_attempts.append({"summary": reasoning["summary"]})
            reasoning_attempts.append(None)

        last_exc: Optional[Exception] = None
        for attempt in reasoning_attempts:
            try:
                if attempt is None:
                    if "reasoning" in base_kwargs:
                        del base_kwargs["reasoning"]
                    return self.client.responses.create(**base_kwargs)

                base_kwargs["reasoning"] = attempt
                return self.client.responses.create(**base_kwargs)
            except Exception as exc:
                last_exc = exc
                error_text = str(exc).lower()

                is_reasoning_param_error = (
                    ("reasoning.summary" in error_text)
                    or ("reasoning.effort" in error_text)
                    or ("unsupported_value" in error_text and "reasoning" in error_text)
                    or ("invalid_request_error" in error_text and "reasoning" in error_text)
                )
                if is_reasoning_param_error:
                    self._log(f"[Responses API] Reasoning params rejected ({attempt}), retrying: {exc}")
                    continue
                raise

        raise last_exc if last_exc is not None else RuntimeError("Failed to create Responses API stream")

    def _process_stream_events(self, stream: Any, state: Dict[str, Any]) -> Iterator[StreamEvent]:
        """Process all events from the stream."""
        for event in stream:
            event_type = getattr(event, "type", None)
            self._log(f"[Responses API] Event type: {event_type}")

            if event_type == "response.output_item.added":
                yield from self._handle_output_item_added(event, state)
            elif event_type == "response.reasoning_text.delta":
                yield from self._handle_reasoning_delta(event)
            elif event_type == "response.output_text.delta":
                yield from self._handle_output_text_delta(event, state)
            elif event_type == "response.function_call_arguments.delta":
                self._handle_function_call_delta(event, state["tool_calls_accumulator"])
            elif event_type == "response.completed":
                self._handle_response_completed(event, state)
                break
            elif event_type == "response.done":
                break

    def _handle_output_item_added(self, event: Any, state: Dict[str, Any]) -> Iterator[StreamEvent]:
        """Handle response.output_item.added events."""
        item = getattr(event, "item", None)
        if not item:
            return

        item_type = getattr(item, "type", None)
        output_index = getattr(event, "output_index", 0)
        self._log(f"[Responses API]   Item type: {item_type}, index: {output_index}")

        if item_type == "reasoning":
            yield from self._handle_reasoning_item(item, state)
        elif item_type == "function_call":
            self._handle_function_call_item(item, output_index, state["tool_calls_accumulator"])

    def _handle_reasoning_item(self, item: Any, state: Dict[str, Any]) -> Iterator[StreamEvent]:
        """Handle reasoning items, yielding summaries or placeholder."""
        summary_list = getattr(item, "summary", None)
        if summary_list:
            self._log(f"[Responses API]   Summary: {summary_list}")
            for summary_item in summary_list:
                text = getattr(summary_item, "text", "")
                if text:
                    yield {"type": "reasoning", "text": text + "\n"}
        elif not state["reasoning_placeholder_sent"]:
            yield {"type": "reasoning", "text": "(Reasoning in progress...)\n"}
            state["reasoning_placeholder_sent"] = True

    def _handle_function_call_item(self, item: Any, output_index: int, accumulator: Dict[int, Dict[str, Any]]) -> None:
        """Handle function call items from output_item.added events."""
        call_id = getattr(item, "call_id", None)
        name = getattr(item, "name", None)
        self._log(f"[Responses API]   Function call: id={call_id}, name={name}")

        self._upsert_tool_call_entry(
            accumulator,
            index=output_index,
            call_id=call_id,
            name=name,
            args_delta=None,
        )

    def _handle_reasoning_delta(self, event: Any) -> Iterator[StreamEvent]:
        """Handle response.reasoning_text.delta events."""
        delta = getattr(event, "delta", "")
        if delta:
            yield {"type": "reasoning", "text": delta}

    def _handle_output_text_delta(self, event: Any, state: Dict[str, Any]) -> Iterator[StreamEvent]:
        """Handle response.output_text.delta events."""
        delta = getattr(event, "delta", "")
        if delta:
            state["accumulated_text"] += delta
            yield {"type": "token", "text": delta}

    def _handle_response_completed(self, event: Any, state: Dict[str, Any]) -> None:
        """Handle response.completed event and extract final data."""
        response_obj = getattr(event, "response", None)
        if not response_obj:
            return

        # Store response ID for multi-turn conversations
        response_id = getattr(response_obj, "id", None)
        if response_id:
            self._previous_response_id = response_id
            self._log(f"[Responses API] Stored response ID: {response_id}")

        status = getattr(response_obj, "status", "completed")
        self._log(f"[Responses API] Response status: {status}")

        state["finish_reason"] = self._normalize_finish_reason(status)
        self._log(f"[Responses API] Set finish_reason to: {state['finish_reason']}")

        self._extract_tool_calls(response_obj, state["tool_calls_accumulator"])

    def _normalize_finish_reason(self, status: str) -> str:
        """Normalize API status to client-compatible finish_reason."""
        if status == "completed":
            return "stop"
        elif status == "requires_action":
            return "tool_calls"
        return status

    def _handle_stream_error(self, exc: Exception) -> Iterator[StreamEvent]:
        """Handle streaming errors by yielding error response."""
        self._flush_log()
        error_msg = f"[OpenAI Responses API] Streaming exception: {exc}"
        print(error_msg)  # Console output
        _logger.error(error_msg)  # File logging
        yield {"type": "token", "text": "\n"}
        yield {
            "type": "final",
            "ai_message": "I encountered an error processing your request.",
            "ai_tool_calls": [],
            "finish_reason": "error",
        }

    def _build_final_response(self, state: Dict[str, Any]) -> StreamEvent:
        """Build and return the final response event."""
        normalized = self._normalize_tool_calls(state["tool_calls_accumulator"])
        self._log(f"[Responses API] Normalized tool calls: {normalized}")

        self._finalize_stream(state["accumulated_text"], normalized)
        ai_tool_calls = self._prepare_tool_calls_for_response(normalized)
        self._log(f"[Responses API] Final ai_tool_calls: {ai_tool_calls}")

        final_finish_reason = "tool_calls" if ai_tool_calls else (state["finish_reason"] or "stop")
        self._log(
            f"[Responses API] Yielding final event with {len(ai_tool_calls)} tool calls, finish_reason={final_finish_reason}"
        )
        self._flush_log()

        return {
            "type": "final",
            "ai_message": state["accumulated_text"],
            "ai_tool_calls": ai_tool_calls,
            "finish_reason": final_finish_reason,
        }

    def _handle_function_call_delta(self, event: Any, acc: Dict[int, Dict[str, Any]]) -> None:
        """Handle function call argument delta events."""
        try:
            index = getattr(event, "output_index", 0)
            call_id = getattr(event, "call_id", None)
            delta = getattr(event, "delta", "")
            name = getattr(event, "name", None)
            self._upsert_tool_call_entry(
                acc,
                index=index,
                call_id=call_id,
                name=name,
                args_delta=delta,
            )
        except Exception:
            pass

    def _extract_tool_calls(self, response: Any, acc: Dict[int, Dict[str, Any]]) -> None:
        """Extract tool calls from a completed response object."""
        try:
            output = getattr(response, "output", [])
            for idx, item in enumerate(output):
                if getattr(item, "type", None) == "function_call":
                    self._create_tool_call_entry_if_missing(
                        acc,
                        index=idx,
                        call_id=getattr(item, "call_id", None),
                        name=getattr(item, "name", None),
                        arguments=getattr(item, "arguments", None),
                    )
        except Exception:
            pass

    def _create_tool_call_entry_if_missing(
        self,
        accumulator: Dict[int, Dict[str, Any]],
        *,
        index: int,
        call_id: Optional[str],
        name: Optional[str],
        arguments: Optional[str],
    ) -> None:
        """Create a completed tool-call entry only when none exists yet."""
        if index in accumulator:
            return
        accumulator[index] = {
            "id": call_id,
            "function": {
                "name": name or "",
                "arguments": arguments or "",
            },
        }

    def _get_or_create_tool_call_entry(
        self,
        accumulator: Dict[int, Dict[str, Any]],
        index: int,
    ) -> Dict[str, Any]:
        """Get/create accumulator entry for a tool call index."""
        return accumulator.setdefault(
            index,
            {"id": None, "function": {"name": "", "arguments": ""}},
        )

    def _upsert_tool_call_entry(
        self,
        accumulator: Dict[int, Dict[str, Any]],
        *,
        index: int,
        call_id: Optional[str],
        name: Optional[str],
        args_delta: Optional[str],
    ) -> None:
        """Merge call metadata and argument chunks into accumulator entry."""
        entry = self._get_or_create_tool_call_entry(accumulator, index)
        if call_id:
            entry["id"] = call_id
        if name:
            entry["function"]["name"] = name
        if args_delta:
            entry["function"]["arguments"] += args_delta

    def _normalize_tool_calls(self, acc: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize accumulated tool calls into a list."""
        result = []
        for i in sorted(acc):
            tc = acc[i]
            func = tc.get("function", {})
            result.append(
                {
                    "id": tc.get("id"),
                    "function": {
                        "name": func.get("name"),
                        "arguments": func.get("arguments"),
                    },
                }
            )
        return result

    def _finalize_stream(self, text: str, tool_calls: List[Dict[str, Any]]) -> None:
        """Finalize the streaming response by updating messages."""
        assistant_msg = SimpleNamespace(
            content=text,
            tool_calls=[
                SimpleNamespace(
                    id=tc.get("id"),
                    function=SimpleNamespace(
                        name=tc.get("function", {}).get("name"),
                        arguments=tc.get("function", {}).get("arguments"),
                    ),
                )
                for tc in tool_calls
            ],
        )
        self.messages.append(self._create_assistant_message(assistant_msg))
        self._append_tool_messages(
            [
                SimpleNamespace(
                    id=tc.get("id"),
                    function=SimpleNamespace(
                        name=tc.get("function", {}).get("name"),
                        arguments=tc.get("function", {}).get("arguments"),
                    ),
                )
                for tc in tool_calls
            ]
        )
        self._clean_conversation_history()

    def _create_assistant_message(self, response_message: Any) -> MessageDict:
        """Create an assistant message from the API response message."""
        content = getattr(response_message, "content", "")
        msg: MessageDict = {"role": "assistant", "content": content}
        tool_calls = getattr(response_message, "tool_calls", None)
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": getattr(tc, "id", None),
                    "type": "function",
                    "function": {
                        "name": getattr(getattr(tc, "function", None), "name", None),
                        "arguments": getattr(getattr(tc, "function", None), "arguments", None),
                    },
                }
                for tc in tool_calls
            ]
        return msg

    def _prepare_tool_calls_for_response(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare tool calls for the final response."""
        result = []
        for tc in tool_calls:
            func = tc.get("function", {})
            args_raw = func.get("arguments")
            try:
                args = json.loads(args_raw) if args_raw else {}
            except Exception:
                args = {}
            result.append({"function_name": func.get("name") or "", "arguments": args})
        return result
