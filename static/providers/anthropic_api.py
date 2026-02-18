"""
MatHud Anthropic API Provider

Anthropic Claude API implementation as a self-contained provider module.
Inherits shared functionality from OpenAIAPIBase for message history management.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator, Sequence
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from static.ai_model import AIModel
from static.functions_definitions import FunctionDefinition
from static.openai_api_base import MessageDict, OpenAIAPIBase, StreamEvent, ToolMode
from static.providers import PROVIDER_ANTHROPIC, ProviderRegistry

_logger = logging.getLogger("mathud")


def _get_anthropic_api_key() -> str:
    """Get the Anthropic API key from environment."""
    load_dotenv()
    parent_env = os.path.join(os.path.dirname(os.getcwd()), ".env")
    if os.path.exists(parent_env):
        load_dotenv(parent_env)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment or .env file")
    return api_key


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
        max_tokens: int = 4096,
        tool_mode: ToolMode = "full",
    ) -> None:
        """Initialize Anthropic API client.

        Args:
            model: AI model to use. Defaults to Claude Sonnet 4.5.
            temperature: Sampling temperature.
            tools: Custom tool definitions.
            max_tokens: Maximum tokens in response.
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
        self.model: AIModel = model if model is not None else AIModel.from_identifier("claude-sonnet-4-5-20250929")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._tool_mode: ToolMode = tool_mode
        self._custom_tools: Optional[Sequence[FunctionDefinition]] = tools
        self._injected_tools: bool = False
        self.tools: Sequence[FunctionDefinition] = self._resolve_tools()

        # Use developer message as system prompt for Anthropic
        self.messages: List[MessageDict] = []
        self._system_prompt = OpenAIAPIBase.DEV_MSG

        # Dummy OpenAI client - not used but needed for base class compatibility
        self.client = None

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.messages = []

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
                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    anthropic_messages.append({"role": "assistant", "content": content or ""})

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
        tool_call_results = prompt_json.get("tool_call_results") if prompt_json else None

        if tool_call_results:
            self._update_tool_messages_with_results(tool_call_results)
            return None

        message_content = self._prepare_message_content(full_prompt)
        return {"role": "user", "content": message_content}

    def create_chat_completion(self, full_prompt: str) -> Any:
        """Create chat completion with Anthropic API."""
        user_message = self._parse_and_prepare_message(full_prompt)
        if user_message is not None:
            self.messages.append(user_message)

        try:
            anthropic_messages = self._convert_messages_to_anthropic()
            anthropic_tools = self._convert_tools_to_anthropic()

            # Anthropic API doesn't accept empty tools list - must be None or non-empty
            create_kwargs = {
                "model": self.model.id,
                "max_tokens": self.max_tokens,
                "system": self._system_prompt,
                "messages": anthropic_messages,
                "temperature": self.temperature,
            }
            if anthropic_tools:
                create_kwargs["tools"] = anthropic_tools

            response = self._anthropic_client.messages.create(**create_kwargs)
        except Exception as e:
            error_msg = f"[Anthropic API] Error during API call: {e}"
            print(error_msg)
            _logger.error(error_msg)
            return self._create_error_response()

        # Convert Anthropic response to OpenAI-like format
        return self._process_anthropic_response(response)

    def _process_anthropic_response(self, response: Any) -> Any:
        """Process Anthropic response and update conversation history."""
        text_content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )

        # Create assistant message for history
        assistant_message: MessageDict = {"role": "assistant", "content": text_content}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        self.messages.append(assistant_message)

        # Append placeholder tool messages
        if tool_calls:
            for tc in tool_calls:
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "Awaiting result...",
                    }
                )

        self._clean_conversation_history()

        # Return OpenAI-like response object
        finish_reason = "tool_calls" if tool_calls else "stop"
        return SimpleNamespace(
            message=SimpleNamespace(
                content=text_content,
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
            finish_reason=finish_reason,
        )

    def create_chat_completion_stream(self, full_prompt: str) -> Iterator[StreamEvent]:
        """Stream chat completion tokens with Anthropic API."""
        user_message = self._parse_and_prepare_message(full_prompt)
        if user_message is not None:
            self.messages.append(user_message)

        accumulated_text = ""
        tool_calls: List[Dict[str, Any]] = []
        current_tool: Optional[Dict[str, Any]] = None
        finish_reason: Optional[str] = None

        try:
            anthropic_messages = self._convert_messages_to_anthropic()
            anthropic_tools = self._convert_tools_to_anthropic()

            # Anthropic API doesn't accept empty tools list - must be None or non-empty
            stream_kwargs = {
                "model": self.model.id,
                "max_tokens": self.max_tokens,
                "system": self._system_prompt,
                "messages": anthropic_messages,
                "temperature": self.temperature,
            }
            if anthropic_tools:
                stream_kwargs["tools"] = anthropic_tools

            with self._anthropic_client.messages.stream(**stream_kwargs) as stream:
                for event in stream:
                    event_type = getattr(event, "type", "")

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", "") == "tool_use":
                            current_tool = {
                                "id": getattr(block, "id", ""),
                                "function": {
                                    "name": getattr(block, "name", ""),
                                    "arguments": "",
                                },
                            }

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            delta_type = getattr(delta, "type", "")
                            if delta_type == "text_delta":
                                text = getattr(delta, "text", "")
                                if text:
                                    accumulated_text += text
                                    yield {"type": "token", "text": text}
                            elif delta_type == "input_json_delta" and current_tool:
                                partial_json = getattr(delta, "partial_json", "")
                                if partial_json:
                                    current_tool["function"]["arguments"] += partial_json

                    elif event_type == "content_block_stop":
                        if current_tool:
                            tool_calls.append(current_tool)
                            current_tool = None

                    elif event_type == "message_stop":
                        finish_reason = "tool_calls" if tool_calls else "stop"

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
            }
            return

        # Update conversation history
        self._finalize_anthropic_stream(accumulated_text, tool_calls)

        # Prepare tool calls for response
        ai_tool_calls = self._prepare_tool_calls_for_response(tool_calls)

        yield {
            "type": "final",
            "ai_message": accumulated_text,
            "ai_tool_calls": ai_tool_calls,
            "finish_reason": finish_reason or "stop",
        }

    def _finalize_anthropic_stream(self, accumulated_text: str, tool_calls: List[Dict[str, Any]]) -> None:
        """Finalize the streaming response by updating messages."""
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
                    "function_name": func_name,
                    "arguments": func_args,
                }
            )
        return result


# Self-register with provider registry
ProviderRegistry.register(PROVIDER_ANTHROPIC, AnthropicAPI)
