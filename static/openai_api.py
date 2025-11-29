"""
MatHud OpenAI API Integration

Manages OpenAI API communication with chat completions, vision support, and tool calling.
Handles conversation history, message formatting, and model configuration.

Dependencies:
    - openai: OpenAI API client library
    - dotenv: Environment variable loading for API keys
    - base64: Image encoding for vision requests
    - json: Message parsing and formatting
    - os: Environment variable access
    - static.functions_definitions: AI tool definitions
    - static.ai_model: Model configuration and capabilities
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Iterator, Sequence
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

from static.ai_model import AIModel
from static.functions_definitions import FUNCTIONS, FunctionDefinition

MessageContent = Union[str, List[Dict[str, Any]]]
MessageDict = Dict[str, Any]
StreamEvent = Dict[str, Any]


class OpenAIChatCompletionsAPI:
    """OpenAI API integration for chat completions with vision and tool calling support.
    
    Manages conversation history, handles vision-enabled messages with image data,
    and processes tool calls for mathematical operations and canvas manipulation.
    """
    
    DEV_MSG = """You are an educational graphing calculator AI interface that can draw shapes, perform calculations and help users explore mathematics. DO NOT try to perform calculations by yourself, use the tools provided instead. Always analyze the canvas state before proceeding. Never use emoticons or emoji in your responses. INFO: Point labels and coordinates are hardcoded to be shown next to all points on the canvas."""

    @staticmethod
    def _initialize_api_key() -> str:
        """Initialize the OpenAI API key from environment or .env file.
        
        Checks environment variables first, then loads from .env file if needed.
        
        Returns:
            str: The initialized API key
            
        Raises:
            ValueError: If API key is not found in environment or .env file
        """
        # First check if OPENAI_API_KEY is already set in environment
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key

        dotenv_path = ".env"
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or .env file")

        return api_key

    def __init__(
        self,
        model: Optional[AIModel] = None,
        temperature: float = 0.2,
        tools: Optional[Sequence[FunctionDefinition]] = None,
        max_tokens: int = 32000,
    ) -> None:
        """Initialize OpenAI API client and conversation state.
        
        Args:
            model: AI model instance (defaults to AIModel.get_default_model())
            temperature: Response randomness (default: 0.2)
            tools: Available function definitions for tool calling
            max_tokens: Maximum response length (default: 32000)
        """
        self.client = OpenAI(api_key=self._initialize_api_key())
        self.model: AIModel = model if model is not None else AIModel.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools: Sequence[FunctionDefinition] = (
            list(tools) if tools is not None else list(FUNCTIONS)
        )
        self.messages: List[MessageDict] = [
            {"role": "developer", "content": OpenAIChatCompletionsAPI.DEV_MSG}
        ]

    def get_model(self) -> AIModel:
        """Get the current AI model instance.
        
        Returns:
            AIModel: Current model configuration
        """
        return self.model

    def reset_conversation(self) -> None:
        """Reset the conversation history to start a new session."""
        self.messages = [{"role": "developer", "content": OpenAIChatCompletionsAPI.DEV_MSG}]
    
    def set_model(self, identifier: str) -> None:
        """Set the AI model by identifier string.
        
        Args:
            identifier: Model identifier string (e.g., 'gpt-4.1')
        """
        # Only create a new AIModel if the identifier is different
        if str(self.model) != identifier:
            self.model = AIModel.from_identifier(identifier)
            print(f"API model updated to: {identifier}")

    def _remove_canvas_state_from_user_messages(self) -> None:
        """Remove canvas state from the last user message in the conversation history.
        
        Cleans up canvas state data to reduce token usage in subsequent requests
        while preserving other message content.
        """
        # Find the last user message
        for message in reversed(self.messages):
            if message.get("role") == "user" and "content" in message:
                content = message["content"]
                
                # Handle list content (vision messages)
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_content = part.get("text", "")
                            if "canvas_state" not in text_content:
                                continue
                            try:
                                text_json = json.loads(text_content)
                                if isinstance(text_json, dict) and "canvas_state" in text_json:
                                    del text_json["canvas_state"]
                                    part["text"] = json.dumps(text_json)
                            except json.JSONDecodeError:
                                pass
                    continue

                # Handle string content
                if isinstance(content, str) and "canvas_state" in content:
                    try:
                        message_content_json = json.loads(content)
                        if isinstance(message_content_json, dict) and "canvas_state" in message_content_json:
                            del message_content_json["canvas_state"]
                            message["content"] = json.dumps(message_content_json)
                    except json.JSONDecodeError:
                        pass

    def _remove_images_from_user_messages(self) -> None:
        """Remove image content from the last user message in the conversation history.
        
        Extracts text content from vision messages to reduce token usage
        in follow-up requests while preserving conversation context.
        """
        # Find the last user message
        for message in reversed(self.messages):
            if message.get("role") == "user" and "content" in message:
                content = message["content"]
                if not isinstance(content, list):
                    continue
                # Keep only the text part
                text_parts = [part for part in content if isinstance(part, dict) and part.get("type") == "text"]
                if text_parts:
                    text_part = text_parts[0]
                    message["content"] = text_part.get("text", "")

    def _clean_conversation_history(self) -> None:
        """Clean up the conversation history by removing canvas states and images from the last user message.
        
        Optimizes conversation history for token efficiency by removing
        large data elements while preserving conversation flow.
        """
        print(f"All messages BEFORE removing canvas_state: \n{self.messages}\n\n")
        self._remove_canvas_state_from_user_messages()
        self._remove_images_from_user_messages()

    def _create_enhanced_prompt_with_image(
        self,
        user_message: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Create an enhanced prompt that includes both text and base64 encoded image.
        
        Args:
            user_message (str): The user's text message
            
        Returns:
            list: Enhanced prompt with text and image data, or None if image loading fails
        """
        try:
            with open("canvas_snapshots/canvas.png", "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                ret_val: List[Dict[str, Any]] = [
                    {
                        "type": "text", 
                        "text": user_message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
                return ret_val
        except Exception as e:
            print(f"Failed to load canvas image: {e}")
            return None

    def _prepare_message_content(self, full_prompt: str) -> MessageContent:
        """Prepare message content with optional canvas image for vision-enabled messages.
        
        Checks vision toggle and model capabilities to determine whether to include
        canvas image data in the message for multimodal AI analysis.
        
        Args:
            full_prompt (str): The full JSON prompt string containing user message and vision flag
            
        Returns:
            The prepared message content with or without image (str or list)
        """
        # Parse the prompt which is a JSON string
        try:
            prompt_json = json.loads(full_prompt)
        except json.JSONDecodeError:
            return full_prompt

        if not isinstance(prompt_json, dict):
            return full_prompt

        user_message = str(prompt_json.get("user_message", ""))
        use_vision = bool(prompt_json.get("use_vision", True))

        if not use_vision:
            return full_prompt

        enhanced_prompt = self._create_enhanced_prompt_with_image(user_message)
        return enhanced_prompt if enhanced_prompt else full_prompt

    def _create_assistant_message(self, response_message: Any) -> MessageDict:
        """Create an assistant message from the API response message.
        
        Args:
            response_message: The message from the API response
            
        Returns:
            dict: The formatted assistant message
        """
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
                        "arguments": getattr(getattr(tool_call, "function", None), "arguments", None)
                    }
                }
                for tool_call in tool_calls
            ]
        
        return assistant_message

    def _create_error_response(
        self,
        error_message: str = "I encountered an error processing your request. Please try again.",
    ) -> SimpleNamespace:
        """Create an error response that matches OpenAI's response structure.
        
        Args:
            error_message (str): The error message to include in the response
            
        Returns:
            SimpleNamespace: A response choice object matching OpenAI's structure
        """
        return SimpleNamespace(
            message=SimpleNamespace(
                content=error_message,
                tool_calls=[]
            ),
            finish_reason="error"
        )

    def _create_tool_message(self, tool_call_id: Optional[str], content: str) -> MessageDict:
        """Create a tool message in response to a tool call.
        
        Args:
            tool_call_id (str): The ID of the tool call being responded to
            content (str): The content/result of the tool call
            
        Returns:
            dict: The formatted tool message
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        }

    def _append_tool_messages(self, tool_calls: Sequence[Any] | None) -> None:
        """Create and append placeholder tool messages for each tool call.
        
        Creates placeholder tool messages that will be updated with actual results
        when the client sends back tool_call_results.
        
        Args:
            tool_calls: List of tool calls from the assistant's response
        """
        if tool_calls:
            for tool_call in tool_calls:
                tool_message = self._create_tool_message(
                    getattr(tool_call, "id", None),
                    "Awaiting result..."
                )
                self.messages.append(tool_message)

    def _update_tool_messages_with_results(self, tool_call_results: str) -> None:
        """Update placeholder tool messages with actual results from the client.
        
        Finds tool messages at the end of the conversation and updates their content
        with the actual execution results.
        
        Args:
            tool_call_results: JSON string containing tool execution results
        """
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

    def create_chat_completion(self, full_prompt: str) -> Any:
        """Create chat completion with OpenAI API.
        
        Main entry point for AI communication. Handles message preparation,
        API calls, response processing, and conversation history management.
        
        Args:
            full_prompt: JSON string containing user message, canvas state, and options
            
        Returns:
            OpenAI response choice object with message and finish_reason
        """
        prompt_json = self._parse_prompt_json(full_prompt)
        tool_call_results = prompt_json.get("tool_call_results") if prompt_json else None

        if tool_call_results:
            self._update_tool_messages_with_results(tool_call_results)
        else:
            message_content = self._prepare_message_content(full_prompt)
            message: MessageDict = {"role": "user", "content": message_content}
            self.messages.append(message)

        # Make the API call
        try:
            response = self.client.chat.completions.create(
                model=self.model.id,
                messages=self.messages,
                tools=self.tools,
                # temperature=self.temperature,
                # max_tokens=self.max_tokens
            )
        except Exception as e:
            print(f"Error during API call: {str(e)}")
            return self._create_error_response()

        choice = response.choices[0]
        
        # Create and append the assistant message
        assistant_message = self._create_assistant_message(choice.message)
        self.messages.append(assistant_message)
        
        # Append tool messages if there are tool calls
        self._append_tool_messages(getattr(choice.message, "tool_calls", None))
        
        # Clean up the messages
        self._clean_conversation_history()
        
        return choice

    def create_chat_completion_stream(self, full_prompt: str) -> Iterator[StreamEvent]:
        """Stream chat completion tokens with OpenAI API.

        Yields dictionaries for incremental updates and a final summary dict.
        Each yielded dict has a required key 'type' with values:
          - 'token': incremental text content with key 'text'
          - 'final': completion summary with keys 'ai_message', 'ai_tool_calls', 'finish_reason'
        """
        prompt_json = self._parse_prompt_json(full_prompt)
        tool_call_results = prompt_json.get("tool_call_results") if prompt_json else None

        if tool_call_results:
            self._update_tool_messages_with_results(tool_call_results)
        else:
            message_content = self._prepare_message_content(full_prompt)
            user_message: MessageDict = {"role": "user", "content": message_content}
            self.messages.append(user_message)

        accumulated_text = ""
        tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None

        try:
            stream = self.client.chat.completions.create(
                model=self.model.id,
                messages=self.messages,
                tools=self.tools,
                stream=True,
            )

            for chunk in stream:
                try:
                    choice = chunk.choices[0]
                except Exception:
                    continue

                delta = getattr(choice, "delta", None)
                if delta is None:
                    delta = getattr(choice, "message", None)

                if delta is not None:
                    content_piece = getattr(delta, "content", None)
                    if isinstance(content_piece, str) and content_piece:
                        accumulated_text += content_piece
                        yield {"type": "token", "text": content_piece}

                    tool_calls_delta = getattr(delta, "tool_calls", None)
                    if tool_calls_delta:
                        for tc in tool_calls_delta:
                            try:
                                index = getattr(tc, "index", None)
                                if isinstance(tc, dict):
                                    index = tc.get("index", index)
                                if index is None:
                                    index = 0
                                entry = tool_calls_accumulator.setdefault(
                                    index,
                                    {
                                        "id": None,
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                tc_id = getattr(tc, "id", None)
                                if isinstance(tc, dict):
                                    tc_id = tc.get("id", tc_id)
                                if tc_id is not None:
                                    entry["id"] = tc_id
                                func = getattr(tc, "function", None)
                                if isinstance(tc, dict):
                                    func = tc.get("function", func)
                                if func is not None:
                                    name_part = getattr(func, "name", None)
                                    if isinstance(func, dict):
                                        name_part = func.get("name", name_part)
                                    if isinstance(name_part, str) and name_part:
                                        entry["function"]["name"] = name_part
                                    args_part = getattr(func, "arguments", None)
                                    if isinstance(func, dict):
                                        args_part = func.get("arguments", args_part)
                                    if isinstance(args_part, str) and args_part:
                                        entry["function"]["arguments"] += args_part
                            except Exception:
                                continue

                if getattr(choice, "finish_reason", None) is not None:
                    finish_reason = getattr(choice, "finish_reason", None)
                    break

        except Exception as exc:
            print(f"[OpenAI API] Streaming exception: {exc}")
            yield {"type": "token", "text": "\n"}
            yield {
                "type": "final",
                "ai_message": "I encountered an error processing your request. Please try again.",
                "ai_tool_calls": [],
                "finish_reason": "error",
            }
            return

        tool_calls_list: List[Dict[str, Any]] = [
            tool_calls_accumulator[index]
            for index in sorted(tool_calls_accumulator)
        ]

        normalized_tool_calls: List[Dict[str, Any]] = []
        for tc in tool_calls_list:
            func = tc.get("function", {}) if isinstance(tc, dict) else {}
            normalized_tool_calls.append(
                {
                    "id": tc.get("id") if isinstance(tc, dict) else None,
                    "function": {
                        "name": func.get("name") if isinstance(func, dict) else None,
                        "arguments": func.get("arguments") if isinstance(func, dict) else None,
                    },
                }
            )

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

        ai_tool_calls_json_ready: List[Dict[str, Any]] = []
        try:
            import json as _json

            for tc in normalized_tool_calls:
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
                func_name = func.get("name") if isinstance(func, dict) else None
                func_args_raw = func.get("arguments") if isinstance(func, dict) else None
                try:
                    func_args = _json.loads(func_args_raw) if func_args_raw else {}
                except Exception:
                    func_args = {}
                ai_tool_calls_json_ready.append(
                    {
                        "function_name": func_name or "",
                        "arguments": func_args,
                    }
                )
        except Exception:
            pass

        yield {
            "type": "final",
            "ai_message": accumulated_text,
            "ai_tool_calls": ai_tool_calls_json_ready,
            "finish_reason": finish_reason or "stop",
        }