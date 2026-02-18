"""
MatHud OpenAI Chat Completions API

Chat Completions API implementation for standard GPT models.
Inherits shared functionality from OpenAIAPIBase.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from static.openai_api_base import OpenAIAPIBase, MessageDict, StreamEvent

# Use the shared MatHud logger for file logging
_logger = logging.getLogger("mathud")


class OpenAIChatCompletionsAPI(OpenAIAPIBase):
    """OpenAI Chat Completions API for standard models (GPT-4, GPT-4o, etc.)."""

    def _create_assistant_message(self, response_message: Any) -> MessageDict:
        """Create an assistant message from the API response message."""
        content = getattr(response_message, "content", "")
        assistant_message: MessageDict = {
            "role": "assistant",
            "content": content,
        }

        tool_calls = getattr(response_message, "tool_calls", None)
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": getattr(tool_call, "id", None),
                    "type": "function",
                    "function": {
                        "name": getattr(getattr(tool_call, "function", None), "name", None),
                        "arguments": getattr(getattr(tool_call, "function", None), "arguments", None),
                    },
                }
                for tool_call in tool_calls
            ]

        return assistant_message

    def create_chat_completion(self, full_prompt: str) -> Any:
        """Create chat completion with OpenAI API."""
        self._prepare_messages_for_request(full_prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model.id,
                messages=self.messages,
                tools=self.tools,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            error_msg = f"Error during API call: {str(e)}"
            print(error_msg)  # Console output
            _logger.error(error_msg)  # File logging
            return self._create_error_response()

        choice = response.choices[0]

        assistant_message = self._create_assistant_message(choice.message)
        self.messages.append(assistant_message)

        self._append_tool_messages(getattr(choice.message, "tool_calls", None))
        self._clean_conversation_history()

        return choice

    def create_chat_completion_stream(self, full_prompt: str) -> Iterator[StreamEvent]:
        """Stream chat completion tokens with OpenAI API."""
        self._prepare_messages_for_request(full_prompt)

        accumulated_text = ""
        tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None

        try:
            stream = self.client.chat.completions.create(
                model=self.model.id,
                messages=self.messages,
                tools=self.tools,
                max_tokens=self.max_tokens,
                stream=True,
            )

            for chunk in stream:
                choice = self._extract_choice_from_chunk(chunk)
                if choice is None:
                    continue

                delta = self._extract_delta_from_choice(choice)
                content_piece = self._extract_content_piece(delta)
                if content_piece:
                    accumulated_text += content_piece
                    yield {"type": "token", "text": content_piece}

                tool_calls_delta = self._extract_tool_calls_delta(delta)
                if tool_calls_delta:
                    self._accumulate_tool_calls(tool_calls_delta, tool_calls_accumulator)

                choice_finish_reason = getattr(choice, "finish_reason", None)
                if choice_finish_reason is not None:
                    finish_reason = choice_finish_reason
                    break

        except Exception as exc:
            error_msg = f"[OpenAI API] Streaming exception: {exc}"
            print(error_msg)  # Console output
            _logger.error(error_msg)  # File logging
            yield {"type": "token", "text": "\n"}
            yield {
                "type": "final",
                "ai_message": "I encountered an error processing your request. Please try again.",
                "ai_tool_calls": [],
                "finish_reason": "error",
            }
            return

        normalized_tool_calls = self._normalize_tool_calls(tool_calls_accumulator)
        self._finalize_stream(accumulated_text, normalized_tool_calls)

        ai_tool_calls_json_ready = self._prepare_tool_calls_for_response(normalized_tool_calls)

        yield {
            "type": "final",
            "ai_message": accumulated_text,
            "ai_tool_calls": ai_tool_calls_json_ready,
            "finish_reason": finish_reason or "stop",
        }

    def _prepare_messages_for_request(self, full_prompt: str) -> None:
        """Prepare conversation messages for a new request turn."""
        prompt_json = self._parse_prompt_json(full_prompt)
        tool_call_results = prompt_json.get("tool_call_results") if prompt_json else None
        if tool_call_results:
            self._update_tool_messages_with_results(tool_call_results)
            return

        message_content = self._prepare_message_content(full_prompt)
        user_message: MessageDict = {"role": "user", "content": message_content}
        self.messages.append(user_message)

    def _extract_choice_from_chunk(self, chunk: Any) -> Optional[Any]:
        """Best-effort extraction of first choice from streaming chunk."""
        try:
            return chunk.choices[0]
        except Exception:
            return None

    def _extract_delta_from_choice(self, choice: Any) -> Optional[Any]:
        """Extract stream delta object, supporting both delta and message shapes."""
        delta = getattr(choice, "delta", None)
        if delta is None:
            delta = getattr(choice, "message", None)
        return delta

    def _extract_content_piece(self, delta: Optional[Any]) -> str:
        """Extract text token content from a stream delta."""
        if delta is None:
            return ""
        content_piece = getattr(delta, "content", None)
        if isinstance(content_piece, str):
            return content_piece
        return ""

    def _extract_tool_calls_delta(self, delta: Optional[Any]) -> Any:
        """Extract tool_calls delta payload from a stream delta."""
        if delta is None:
            return None
        return getattr(delta, "tool_calls", None)

    def _accumulate_tool_calls(self, tool_calls_delta: Any, accumulator: Dict[int, Dict[str, Any]]) -> None:
        """Accumulate streaming tool call deltas."""
        for tc in tool_calls_delta:
            try:
                self._accumulate_single_tool_call_delta(tc, accumulator)
            except Exception:
                continue

    def _accumulate_single_tool_call_delta(self, tc: Any, accumulator: Dict[int, Dict[str, Any]]) -> None:
        """Accumulate a single tool call delta into the accumulator."""
        index = self._extract_tool_call_delta_index(tc)
        entry = self._get_or_create_tool_call_entry(accumulator, index)

        tc_id = self._extract_tool_call_delta_id(tc)
        if tc_id is not None:
            entry["id"] = tc_id

        func = self._extract_tool_call_delta_function(tc)
        if func is None:
            return

        name_part = self._extract_function_name_part(func)
        if isinstance(name_part, str) and name_part:
            entry["function"]["name"] = name_part

        args_part = self._extract_function_arguments_part(func)
        if isinstance(args_part, str) and args_part:
            entry["function"]["arguments"] += args_part

    def _extract_tool_call_delta_index(self, tc: Any) -> int:
        """Extract tool call index, defaulting to 0."""
        index = getattr(tc, "index", None)
        if isinstance(tc, dict):
            index = tc.get("index", index)
        if isinstance(index, int):
            return index
        return 0

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

    def _extract_tool_call_delta_id(self, tc: Any) -> Optional[str]:
        """Extract tool call id from object or dict delta."""
        tc_id = getattr(tc, "id", None)
        if isinstance(tc, dict):
            tc_id = tc.get("id", tc_id)
        if isinstance(tc_id, str):
            return tc_id
        return None

    def _extract_tool_call_delta_function(self, tc: Any) -> Any:
        """Extract nested function payload from object or dict delta."""
        func = getattr(tc, "function", None)
        if isinstance(tc, dict):
            func = tc.get("function", func)
        return func

    def _extract_function_name_part(self, func: Any) -> Any:
        """Extract function name chunk from function payload."""
        name_part = getattr(func, "name", None)
        if isinstance(func, dict):
            name_part = func.get("name", name_part)
        return name_part

    def _extract_function_arguments_part(self, func: Any) -> Any:
        """Extract function arguments chunk from function payload."""
        args_part = getattr(func, "arguments", None)
        if isinstance(func, dict):
            args_part = func.get("arguments", args_part)
        return args_part

    def _normalize_tool_calls(self, accumulator: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize accumulated tool calls into a list."""
        tool_calls_list = [accumulator[i] for i in sorted(accumulator)]
        normalized = []
        for tc in tool_calls_list:
            func = tc.get("function", {}) if isinstance(tc, dict) else {}
            normalized.append(
                {
                    "id": tc.get("id") if isinstance(tc, dict) else None,
                    "function": {
                        "name": func.get("name") if isinstance(func, dict) else None,
                        "arguments": func.get("arguments") if isinstance(func, dict) else None,
                    },
                }
            )
        return normalized

    def _finalize_stream(self, accumulated_text: str, normalized_tool_calls: List[Dict[str, Any]]) -> None:
        """Finalize the streaming response by updating messages."""
        assistant_message_like = SimpleNamespace(
            content=accumulated_text,
            tool_calls=[
                SimpleNamespace(
                    id=tc.get("id"),
                    function=SimpleNamespace(
                        name=tc.get("function", {}).get("name"),
                        arguments=tc.get("function", {}).get("arguments"),
                    ),
                )
                for tc in normalized_tool_calls
            ],
        )
        assistant_message = self._create_assistant_message(assistant_message_like)
        self.messages.append(assistant_message)

        self._append_tool_messages(
            [
                SimpleNamespace(
                    id=tc.get("id"),
                    function=SimpleNamespace(
                        name=tc.get("function", {}).get("name"),
                        arguments=tc.get("function", {}).get("arguments"),
                    ),
                )
                for tc in normalized_tool_calls
            ]
        )

        self._clean_conversation_history()

    def _prepare_tool_calls_for_response(self, normalized_tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare tool calls for the final response."""
        import json as _json

        result = []
        for tc in normalized_tool_calls:
            func = tc.get("function", {}) if isinstance(tc, dict) else {}
            func_name = func.get("name") if isinstance(func, dict) else None
            func_args_raw = func.get("arguments") if isinstance(func, dict) else None
            try:
                func_args = _json.loads(func_args_raw) if func_args_raw else {}
            except Exception:
                func_args = {}
            result.append(
                {
                    "function_name": func_name or "",
                    "arguments": func_args,
                }
            )
        return result
