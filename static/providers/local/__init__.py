"""
MatHud Local LLM Provider Base Module

Provides the base infrastructure for local LLM providers (llama.cpp, LM Studio, etc.).
Implements shared functionality for tool capability detection and model discovery.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, Dict, List, Optional, Set, Type

from static.ai_model import AIModel
from static.canvas_state_formatter import CanvasFormat
from static.functions_definitions import FunctionDefinition
from static.openai_api_base import OpenAIAPIBase, StreamEvent, get_configured_tool_mode
from static.response_metrics import (
    create_stream_requesting_usage,
    reasoning_text_from_delta,
    record_chat_completions_usage,
    tool_call_argument_text,
)

_logger = logging.getLogger("mathud")

# Models verified to support native function calling
# These model families have tool calling support in their OpenAI-compatible APIs
TOOL_CAPABLE_MODEL_FAMILIES: Set[str] = {
    # OpenAI's open-weight model
    "gpt-oss",
    # Meta Llama 3.x family
    "llama3.1",
    "llama3.2",
    "llama3.3",
    # Qwen 2.5 family
    "qwen2.5",
    "qwen2.5-coder",
    # Mistral family
    "mistral",
    "mistral-nemo",
    "mixtral",
    # Cohere Command R
    "command-r",
    "command-r-plus",
    # NVIDIA
    "nemotron",
    # IBM Granite 3
    "granite3-dense",
    "granite3-moe",
}


def normalize_model_name(model_name: str) -> str:
    """Normalize a model name by extracting the base family name.

    Examples:
        'llama3.1:8b' -> 'llama3.1'
        'qwen2.5:7b-instruct' -> 'qwen2.5'
        'mistral:latest' -> 'mistral'

    Args:
        model_name: The full model name with optional tag

    Returns:
        The normalized base model name
    """
    # Remove tag suffix (after colon)
    base_name = model_name.split(":")[0].strip().lower()
    return base_name


def supports_tools(model_name: str) -> bool:
    """Check if a model supports native function calling.

    Args:
        model_name: The model name to check (can include tags like ':8b')

    Returns:
        True if the model family supports tool calling
    """
    base_name = normalize_model_name(model_name)

    # Check for exact match first
    if base_name in TOOL_CAPABLE_MODEL_FAMILIES:
        return True

    # Check if any tool-capable family is a prefix of the model name
    for family in TOOL_CAPABLE_MODEL_FAMILIES:
        if base_name.startswith(family):
            return True

    return False


class LocalProviderRegistry:
    """Registry for local LLM providers.

    Manages registration and discovery of local LLM backends like llama-server.
    """

    _providers: Dict[str, Type["LocalLLMBase"]] = {}

    @classmethod
    def register(cls, provider_name: str, provider_class: Type["LocalLLMBase"]) -> None:
        """Register a local LLM provider.

        Args:
            provider_name: Unique provider identifier (e.g., 'local_agent')
            provider_class: The provider class
        """
        cls._providers[provider_name] = provider_class
        _logger.info(f"Registered local provider: {provider_name}")

    @classmethod
    def get_provider_class(cls, provider_name: str) -> Optional[Type["LocalLLMBase"]]:
        """Get the provider class by name.

        Args:
            provider_name: The provider identifier

        Returns:
            The provider class, or None if not registered
        """
        return cls._providers.get(provider_name)

    @classmethod
    def is_provider_available(cls, provider_name: str) -> bool:
        """Check if a local provider is available (server running).

        Args:
            provider_name: The provider identifier

        Returns:
            True if the provider's server is accessible
        """
        provider_class = cls._providers.get(provider_name)
        if provider_class is None:
            return False

        try:
            # Create a temporary instance to check availability
            # We use a minimal init pattern here
            instance = object.__new__(provider_class)
            return instance._is_available()
        except Exception as e:
            _logger.debug(f"Provider {provider_name} availability check failed: {e}")
            return False

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available local providers.

        Returns:
            List of provider names with accessible servers
        """
        available = []
        for name in cls._providers:
            if cls.is_provider_available(name):
                available.append(name)
        return available

    @classmethod
    def get_registered_providers(cls) -> List[str]:
        """Get list of registered provider names.

        Returns:
            List of registered provider names
        """
        return list(cls._providers.keys())


class LocalLLMBase(OpenAIAPIBase, ABC):
    """Abstract base class for local LLM providers.

    Provides shared functionality for local providers using OpenAI-compatible APIs.
    Subclasses must implement server availability checking and model discovery.
    """

    # Local models have small contexts and tokenize digits one by one: always send
    # the compact text canvas, and trim large scenes harder than for cloud models.
    DEFAULT_CANVAS_FORMAT: CanvasFormat = "text"
    DEFAULT_CANVAS_BUDGET_TOKENS: Optional[int] = 1500

    # False once the server rejected stream_options (see create_stream_requesting_usage).
    _stream_usage_supported = True

    def __init__(
        self,
        model: Optional[AIModel] = None,
        temperature: float = 0.2,
        tools: Optional[Sequence[FunctionDefinition]] = None,
        max_tokens: int = 16000,
    ) -> None:
        """Initialize local LLM provider.

        Args:
            model: AI model to use. Defaults to first available tool-capable model.
            temperature: Sampling temperature.
            tools: Custom tool definitions.
            max_tokens: Maximum tokens in response.
        """
        from openai import OpenAI

        # Initialize client pointing to local server
        base_url = self._get_base_url()
        self.client = OpenAI(
            api_key="local",  # Placeholder - local servers don't need keys
            base_url=f"{base_url}/v1",
        )

        # If no model specified, we'll set a placeholder that gets overridden
        self.model: AIModel = model if model is not None else AIModel.from_identifier("local-model")
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Default to search mode for local LLMs to avoid context overflow
        # (MATHUD_TOOL_EXPOSURE=full exposes every tool instead)
        self._tool_mode = get_configured_tool_mode()
        self._custom_tools = tools
        self._injected_tools = False
        self.tools: Sequence[FunctionDefinition] = self._resolve_tools()

        # Use developer message as system prompt
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": self._build_system_prompt()}]

    @abstractmethod
    def _is_available(self) -> bool:
        """Check if the local server is running and accessible.

        Returns:
            True if the server responds to health checks
        """
        pass

    @abstractmethod
    def _discover_models(self) -> List[Dict[str, Any]]:
        """Query the server for available models.

        Returns:
            List of model info dicts with at least 'name' key
        """
        pass

    @abstractmethod
    def _get_base_url(self) -> str:
        """Get the server base URL (from env or default).

        Returns:
            The server base URL
        """
        pass

    @abstractmethod
    def _get_provider_name(self) -> str:
        """Get the provider name for this local LLM.

        Returns:
            Provider name string (e.g., 'LocalAgent')
        """
        pass

    def discover_models_with_tool_support(self) -> List[Dict[str, Any]]:
        """Discover models and filter to only those supporting tools.

        Filtering uses the ``TOOL_CAPABLE_MODEL_FAMILIES`` allowlist, which only
        recognises named model families. Providers whose model identifiers carry
        no family information should override this method.

        Returns:
            List of model info dicts for tool-capable models
        """
        try:
            all_models = self._discover_models()
        except Exception as e:
            _logger.warning(f"Failed to discover models: {e}")
            return []

        tool_capable = []
        for model_info in all_models:
            model_name = model_info.get("name", "")
            if supports_tools(model_name):
                tool_capable.append(model_info)

        _logger.info(f"Discovered {len(tool_capable)} tool-capable models out of {len(all_models)} total")
        return tool_capable

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.messages = [{"role": "system", "content": self._build_system_prompt()}]
        self._last_canvas_state = None

    def create_chat_completion(self, full_prompt: str) -> Any:
        """Create a chat completion using the local LLM.

        Args:
            full_prompt: The prompt JSON string

        Returns:
            Response choice object compatible with OpenAI format
        """
        user_message = self._parse_and_prepare_message(full_prompt)
        if user_message is not None:
            self.messages.append(user_message)
        metrics = self._start_response_metrics("chat_completions", streamed=False)

        try:
            response = self.client.chat.completions.create(
                model=self.model.id,
                messages=self.messages,
                tools=list(self.tools) if self.tools else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            choice = response.choices[0]
        except Exception as e:
            error_msg = f"[{self._get_provider_name()}] Error during API call: {e}"
            print(error_msg)
            _logger.error(error_msg)
            self._finish_response_metrics(metrics, "error", 0, error=str(e))
            return self._create_error_response()

        # Process response and update history
        processed = self._process_response(choice)
        record_chat_completions_usage(response, metrics)
        tool_calls = getattr(processed.message, "tool_calls", None) or []
        self._finish_response_metrics(metrics, processed.finish_reason, len(tool_calls))
        return processed

    def create_chat_completion_stream(self, full_prompt: str) -> Iterator[StreamEvent]:
        """Stream chat completion tokens from the local LLM.

        Args:
            full_prompt: The prompt JSON string

        Yields:
            Stream events with type 'token' or 'final'; the final event carries
            the request's ``metrics`` (see static/response_metrics.py), including
            llama-server ``timings`` when the server reports them.
        """
        user_message = self._parse_and_prepare_message(full_prompt)
        if user_message is not None:
            self.messages.append(user_message)

        accumulated_text = ""
        tool_calls: List[Dict[str, Any]] = []
        tool_call_deltas: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        metrics = self._start_response_metrics("chat_completions")

        try:
            stream, self._stream_usage_supported = create_stream_requesting_usage(
                self.client.chat.completions.create,
                self._stream_usage_supported,
                model=self.model.id,
                messages=self.messages,
                tools=list(self.tools) if self.tools else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            for chunk in stream:
                # llama-server puts usage and timings on the last chunk, whose choices may be empty.
                record_chat_completions_usage(chunk, metrics)
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                chunk_finish = chunk.choices[0].finish_reason

                # Reasoning is not shown, but it is generated output (llama-server reasoning_content).
                reasoning_piece = reasoning_text_from_delta(delta)
                if reasoning_piece:
                    metrics.mark_output("reasoning")
                    metrics.add_output_text(reasoning_piece)

                # Handle content tokens
                if delta.content:
                    metrics.mark_output("content")
                    metrics.add_output_text(delta.content)
                    accumulated_text += delta.content
                    yield {"type": "token", "text": delta.content}

                # Handle tool call deltas
                if delta.tool_calls:
                    metrics.mark_output("tool_call")
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_deltas:
                            tool_call_deltas[idx] = {
                                "id": tc_delta.id or "",
                                "function": {
                                    "name": "",
                                    "arguments": "",
                                },
                            }

                        tc = tool_call_deltas[idx]
                        if tc_delta.id:
                            tc["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tc["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc["function"]["arguments"] += tc_delta.function.arguments

                if chunk_finish:
                    finish_reason = chunk_finish

        except Exception as e:
            error_msg = f"[{self._get_provider_name()}] Streaming error: {e}"
            print(error_msg)
            _logger.error(error_msg)
            yield {"type": "token", "text": "\n"}
            yield {
                "type": "final",
                "ai_message": "I encountered an error processing your request. Please try again.",
                "ai_tool_calls": [],
                "finish_reason": "error",
                "metrics": dict(self._finish_response_metrics(metrics, "error", 0, error=str(e))),
            }
            return

        # Convert tool call deltas to final format. Some local servers stream calls
        # without ids; give those a per-response id so results can be matched to them.
        for index, tool_call in tool_call_deltas.items():
            if not tool_call["id"]:
                tool_call["id"] = f"call_{index}"
        tool_calls = list(tool_call_deltas.values())

        # Update conversation history
        self._finalize_stream(accumulated_text, tool_calls)

        # Prepare tool calls for response
        ai_tool_calls = self._prepare_tool_calls_for_response(tool_calls)
        metrics.add_output_text(tool_call_argument_text(tool_calls))
        resolved_finish_reason = finish_reason or "stop"

        yield {
            "type": "final",
            "ai_message": accumulated_text,
            "ai_tool_calls": ai_tool_calls,
            "finish_reason": resolved_finish_reason,
            "metrics": dict(self._finish_response_metrics(metrics, resolved_finish_reason, len(ai_tool_calls))),
        }

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

    def _prepare_message_content(self, full_prompt: str) -> str:
        """Prepare message content for local LLMs.

        Unlike cloud providers, local LLMs work better with plain text messages:
        the user's text preceded by the rendered <canvas> block, without the
        prompt JSON envelope. Images are not sent to local models.

        Args:
            full_prompt: The full JSON prompt string

        Returns:
            The plain text user message, or the original prompt if parsing fails
        """
        canvas_format = self._get_canvas_format()
        if canvas_format == "json":
            return self._prepare_object_count_content(full_prompt)
        prompt_json = self._parse_prompt_json(full_prompt)
        if prompt_json is None:
            return full_prompt
        return self._build_user_text(prompt_json, canvas_format) or full_prompt

    def _prepare_object_count_content(self, full_prompt: str) -> str:
        """Legacy json-format path: the user message plus a "[Canvas: 3 Points, ...]" count line."""
        try:
            prompt_json = json.loads(full_prompt)
        except json.JSONDecodeError:
            return full_prompt

        if not isinstance(prompt_json, dict):
            return full_prompt

        # Extract just the user message for cleaner conversation history
        user_message = prompt_json.get("user_message", "")
        if user_message:
            summary_parts: List[str] = []

            # Prefer server-normalized summary payload when available.
            canvas_state_summary = prompt_json.get("canvas_state_summary")
            if isinstance(canvas_state_summary, dict):
                summary_state = canvas_state_summary.get("state", {})
                if isinstance(summary_state, dict):
                    for key, items in summary_state.items():
                        if isinstance(items, list) and items:
                            summary_parts.append(f"{len(items)} {key}")

            # Backward compatibility fallback for prompts without summary payload.
            if not summary_parts:
                canvas_state = prompt_json.get("canvas_state", {})
                if isinstance(canvas_state, dict):
                    for key, items in canvas_state.items():
                        if isinstance(items, list) and items:
                            summary_parts.append(f"{len(items)} {key}")

            if summary_parts:
                state_summary = f"\n[Canvas: {', '.join(summary_parts)}]"
                return f"{user_message}{state_summary}"
            return str(user_message)

        return full_prompt

    def _process_response(self, choice: Any) -> Any:
        """Process the API response and update conversation history."""
        from types import SimpleNamespace

        message = choice.message
        text_content = message.content or ""
        raw_tool_calls = message.tool_calls or []

        # Build tool calls list
        tool_calls: List[Dict[str, Any]] = []
        for index, tc in enumerate(raw_tool_calls):
            tool_calls.append(
                {
                    "id": tc.id or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )

        # Add assistant message to history
        assistant_message: Dict[str, Any] = {"role": "assistant", "content": text_content}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        self.messages.append(assistant_message)

        # Add placeholder tool messages
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

    def _finalize_stream(self, accumulated_text: str, tool_calls: List[Dict[str, Any]]) -> None:
        """Finalize the streaming response by updating messages."""
        # Create assistant message
        assistant_message: Dict[str, Any] = {"role": "assistant", "content": accumulated_text}
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

        # Add placeholder tool messages
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
        import json

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
