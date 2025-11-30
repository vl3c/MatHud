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
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

from static.ai_model import AIModel
from static.functions_definitions import FUNCTIONS, FunctionDefinition

# Use the shared MatHud logger for file logging
_logger = logging.getLogger("mathud")

MessageContent = Union[str, List[Dict[str, Any]]]
MessageDict = Dict[str, Any]
StreamEvent = Dict[str, Any]


class OpenAIAPIBase:
    """Base class for OpenAI API implementations."""
    
    DEV_MSG = """You are an educational graphing calculator AI interface that can draw shapes, perform calculations and help users explore mathematics. DO NOT try to perform calculations by yourself, use the tools provided instead. Always analyze the canvas state before proceeding. Never use emoticons or emoji in your responses. When performing multiple steps, include a succinct summary of all actions taken in your final response. INFO: Point labels and coordinates are hardcoded to be shown next to all points on the canvas."""

    @staticmethod
    def _initialize_api_key() -> str:
        """Initialize the OpenAI API key from environment or .env file."""
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
        """Initialize OpenAI API client and conversation state."""
        self.client = OpenAI(api_key=self._initialize_api_key())
        self.model: AIModel = model if model is not None else AIModel.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools: Sequence[FunctionDefinition] = (
            list(tools) if tools is not None else list(FUNCTIONS)
        )
        self.messages: List[MessageDict] = [
            {"role": "developer", "content": OpenAIAPIBase.DEV_MSG}
        ]

    def get_model(self) -> AIModel:
        """Get the current AI model instance."""
        return self.model

    def reset_conversation(self) -> None:
        """Reset the conversation history to start a new session."""
        self.messages = [{"role": "developer", "content": OpenAIAPIBase.DEV_MSG}]
    
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
                                if isinstance(text_json, dict) and "canvas_state" in text_json:
                                    del text_json["canvas_state"]
                                    part["text"] = json.dumps(text_json)
                            except json.JSONDecodeError:
                                pass
                    continue

                if isinstance(content, str) and "canvas_state" in content:
                    try:
                        message_content_json = json.loads(content)
                        if isinstance(message_content_json, dict) and "canvas_state" in message_content_json:
                            del message_content_json["canvas_state"]
                            message["content"] = json.dumps(message_content_json)
                    except json.JSONDecodeError:
                        pass

    def _remove_images_from_user_messages(self) -> None:
        """Remove image content from all user messages in the conversation history."""
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

    def _create_enhanced_prompt_with_image(self, user_message: str) -> Optional[List[Dict[str, Any]]]:
        """Create an enhanced prompt that includes both text and base64 encoded image."""
        try:
            with open("canvas_snapshots/canvas.png", "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                return [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
        except Exception as e:
            error_msg = f"Failed to load canvas image: {e}"
            print(error_msg)  # Console output
            _logger.error(error_msg)  # File logging
            return None

    def _prepare_message_content(self, full_prompt: str) -> MessageContent:
        """Prepare message content with optional canvas image for vision-enabled messages."""
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


