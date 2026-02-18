"""
Integration tests for Ollama API provider.

These tests require a running Ollama server with tool-capable models installed.
Tests are skipped if Ollama is not available.

Run with: pytest server_tests/test_ollama_integration.py -v
"""

import json
import pytest
from typing import Any, Dict, List

from static.providers.local.ollama_api import OllamaAPI


# Skip all tests in this module if Ollama is not running
pytestmark = pytest.mark.skipif(not OllamaAPI.is_server_running(), reason="Ollama server is not running")


class TestOllamaServerIntegration:
    """Integration tests for Ollama server operations."""

    def test_server_is_running(self) -> None:
        """Verify Ollama server is accessible."""
        assert OllamaAPI.is_server_running() is True

    def test_discover_models(self) -> None:
        """Can discover installed models."""
        models = OllamaAPI.get_tool_capable_models()
        # Just verify we get a list back (might be empty if no tool-capable models)
        assert isinstance(models, list)
        print(f"Found {len(models)} tool-capable models: {[m['name'] for m in models]}")

    def test_get_loaded_models(self) -> None:
        """Can query loaded models."""
        loaded = OllamaAPI.get_loaded_models()
        assert isinstance(loaded, list)
        print(f"Currently loaded models: {loaded}")


@pytest.fixture
def tool_capable_model() -> str:
    """Get a tool-capable model for testing, skip if none available."""
    models = OllamaAPI.get_tool_capable_models()
    if not models:
        pytest.skip("No tool-capable models installed in Ollama")
    return str(models[0]["name"])


class TestOllamaModelLoading:
    """Integration tests for model loading."""

    def test_preload_model(self, tool_capable_model: str) -> None:
        """Can preload a model into memory."""
        success, message = OllamaAPI.preload_model(tool_capable_model, timeout=120)
        print(f"Preload result: {success}, {message}")
        assert success is True

    def test_is_model_loaded_after_preload(self, tool_capable_model: str) -> None:
        """Model shows as loaded after preloading."""
        # Ensure model is loaded first
        OllamaAPI.preload_model(tool_capable_model, timeout=120)

        is_loaded = OllamaAPI.is_model_loaded(tool_capable_model)
        assert is_loaded is True

    def test_unload_model(self, tool_capable_model: str) -> None:
        """Can unload a model from memory."""
        # First ensure it's loaded
        OllamaAPI.preload_model(tool_capable_model, timeout=120)

        # Then unload
        success, message = OllamaAPI.unload_model(tool_capable_model)
        print(f"Unload result: {success}, {message}")
        assert success is True


class TestOllamaToolCalling:
    """Integration tests for tool calling with Ollama."""

    @pytest.fixture
    def ollama_api(self, tool_capable_model: str) -> OllamaAPI:
        """Create an OllamaAPI instance with a tool-capable model."""
        from static.ai_model import AIModel, PROVIDER_OLLAMA

        # Register the model
        AIModel.MODEL_CONFIGS[tool_capable_model] = {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OLLAMA,
            "display_name": tool_capable_model,
        }

        model = AIModel.from_identifier(tool_capable_model)

        # Preload the model first
        OllamaAPI.preload_model(tool_capable_model, timeout=120)

        return OllamaAPI(model=model)

    def test_simple_chat_completion(self, ollama_api: OllamaAPI) -> None:
        """Can get a simple chat completion without tools."""
        # Create a simple prompt
        prompt = json.dumps(
            {
                "user_message": "Say 'hello' and nothing else.",
                "canvas_state": {},
            }
        )

        response = ollama_api.create_chat_completion(prompt)

        assert response is not None
        assert hasattr(response, "message")
        assert response.message.content is not None
        print(f"Response: {response.message.content}")

    def test_tool_call_request(self, ollama_api: OllamaAPI) -> None:
        """Model can request tool calls."""
        # Limit tools to just create_point for simpler testing
        from static.functions_definitions import FUNCTIONS

        create_point_tool = None
        for tool in FUNCTIONS:
            if tool.get("function", {}).get("name") == "create_point":
                create_point_tool = tool
                break

        if create_point_tool:
            ollama_api.tools = [create_point_tool]

        # Create a prompt that should trigger a tool call
        prompt = json.dumps(
            {
                "user_message": "Create a point at coordinates (50, 100) named 'TestPoint'",
                "canvas_state": {"points": [], "segments": []},
            }
        )

        response = ollama_api.create_chat_completion(prompt)

        assert response is not None
        print(f"Response content: {response.message.content}")
        print(f"Tool calls: {response.message.tool_calls}")
        print(f"Finish reason: {response.finish_reason}")

        # The model should either respond with text or request a tool call
        # We can't guarantee tool call behavior, but we should get a valid response
        assert response.finish_reason in ["stop", "tool_calls"]

    def test_conversation_history_maintained(self, ollama_api: OllamaAPI) -> None:
        """Conversation history is maintained across calls.

        Note: This test verifies the conversation history structure is correct.
        The model's ability to utilize context varies by model quality.
        """
        # First message
        prompt1 = json.dumps(
            {
                "user_message": "My favorite color is blue. Remember this.",
                "canvas_state": {},
            }
        )
        ollama_api.create_chat_completion(prompt1)

        # Check history structure: system + user + assistant
        assert len(ollama_api.messages) >= 3
        assert ollama_api.messages[0]["role"] == "system"
        assert ollama_api.messages[1]["role"] == "user"
        assert ollama_api.messages[2]["role"] == "assistant"

        # Verify first user message contains the color info
        first_user_content = ollama_api.messages[1]["content"]
        assert "blue" in first_user_content.lower()

        # Second message referencing the first
        prompt2 = json.dumps(
            {
                "user_message": "What is my favorite color?",
                "canvas_state": {},
            }
        )
        response = ollama_api.create_chat_completion(prompt2)

        # Verify history grew correctly: system + user + assistant + user + assistant
        assert len(ollama_api.messages) >= 5
        assert ollama_api.messages[3]["role"] == "user"
        assert ollama_api.messages[4]["role"] == "assistant"

        print(f"Response: {response.message.content}")
        print(f"Messages in history: {len(ollama_api.messages)}")

        # Log message structure for debugging
        for i, msg in enumerate(ollama_api.messages):
            content_preview = str(msg.get("content", ""))[:80]
            print(f"  [{i}] {msg['role']}: {content_preview}...")

    def test_reset_conversation(self, ollama_api: OllamaAPI) -> None:
        """Can reset conversation history."""
        # Add some messages
        prompt = json.dumps(
            {
                "user_message": "Hello",
                "canvas_state": {},
            }
        )
        ollama_api.create_chat_completion(prompt)

        initial_count = len(ollama_api.messages)
        assert initial_count > 1

        # Reset
        ollama_api.reset_conversation()

        # Should only have system message
        assert len(ollama_api.messages) == 1
        assert ollama_api.messages[0]["role"] == "system"


class TestOllamaStreamingIntegration:
    """Integration tests for streaming with Ollama."""

    @pytest.fixture
    def ollama_api(self, tool_capable_model: str) -> OllamaAPI:
        """Create an OllamaAPI instance."""
        from static.ai_model import AIModel, PROVIDER_OLLAMA

        AIModel.MODEL_CONFIGS[tool_capable_model] = {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OLLAMA,
            "display_name": tool_capable_model,
        }

        model = AIModel.from_identifier(tool_capable_model)
        OllamaAPI.preload_model(tool_capable_model, timeout=120)

        return OllamaAPI(model=model)

    def test_streaming_response(self, ollama_api: OllamaAPI) -> None:
        """Can stream responses."""
        prompt = json.dumps(
            {
                "user_message": "Count from 1 to 5.",
                "canvas_state": {},
            }
        )

        tokens: List[str] = []
        final_event: Dict[str, Any] = {}

        for event in ollama_api.create_chat_completion_stream(prompt):
            if event.get("type") == "token":
                tokens.append(event.get("text", ""))
            elif event.get("type") == "final":
                final_event = event

        print(f"Received {len(tokens)} tokens")
        print(f"Final message: {final_event.get('ai_message', '')[:200]}")

        assert len(tokens) > 0
        assert "ai_message" in final_event
        assert final_event.get("finish_reason") in ["stop", "tool_calls"]


class TestOllamaErrorHandling:
    """Tests for error handling with Ollama."""

    def test_invalid_model_handling(self) -> None:
        """Handles invalid model gracefully."""
        from static.ai_model import AIModel, PROVIDER_OLLAMA

        # Register a fake model
        AIModel.MODEL_CONFIGS["nonexistent-model:latest"] = {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OLLAMA,
            "display_name": "Nonexistent",
        }

        model = AIModel.from_identifier("nonexistent-model:latest")
        api = OllamaAPI(model=model)

        prompt = json.dumps(
            {
                "user_message": "Hello",
                "canvas_state": {},
            }
        )

        # Should handle error gracefully
        response = api.create_chat_completion(prompt)
        assert response is not None
        # Should get an error response
        assert response.finish_reason == "error" or "error" in response.message.content.lower()


class TestOllamaConversationDebug:
    """Tests for debugging conversation history issues."""

    @pytest.fixture
    def ollama_api(self, tool_capable_model: str) -> OllamaAPI:
        """Create an OllamaAPI instance."""
        from static.ai_model import AIModel, PROVIDER_OLLAMA

        AIModel.MODEL_CONFIGS[tool_capable_model] = {
            "has_vision": False,
            "is_reasoning_model": False,
            "provider": PROVIDER_OLLAMA,
            "display_name": tool_capable_model,
        }

        model = AIModel.from_identifier(tool_capable_model)
        OllamaAPI.preload_model(tool_capable_model, timeout=120)

        return OllamaAPI(model=model)

    def test_tool_call_and_result_flow(self, ollama_api: OllamaAPI) -> None:
        """Test the full flow of tool call and result."""
        # Limit to a simple tool
        from static.functions_definitions import FUNCTIONS

        # Find get_current_canvas_state - a simple tool
        simple_tool = None
        for tool in FUNCTIONS:
            name = tool.get("function", {}).get("name", "")
            if name == "get_current_canvas_state":
                simple_tool = tool
                break

        if simple_tool:
            ollama_api.tools = [simple_tool]

        # First request - might trigger tool call
        prompt1 = json.dumps(
            {
                "user_message": "What's on the canvas? Use get_current_canvas_state to check.",
                "canvas_state": {"points": [{"name": "A", "x": 0, "y": 0}]},
            }
        )

        response1 = ollama_api.create_chat_completion(prompt1)

        print("\n=== After first request ===")
        print(f"Response: {response1.message.content}")
        print(f"Tool calls: {response1.message.tool_calls}")
        print(f"Finish reason: {response1.finish_reason}")
        print(f"Message count: {len(ollama_api.messages)}")

        for i, msg in enumerate(ollama_api.messages):
            role = msg.get("role")
            content = str(msg.get("content", ""))[:100]
            tool_calls = "has_tool_calls" if msg.get("tool_calls") else ""
            tool_id = msg.get("tool_call_id", "")
            print(f"  [{i}] {role}: {content}... {tool_calls} {tool_id}")

        # If there was a tool call, simulate the result
        if response1.finish_reason == "tool_calls" and response1.message.tool_calls:
            tool_call = response1.message.tool_calls[0]
            tool_id = tool_call.id

            # Send tool result
            prompt2 = json.dumps(
                {
                    "tool_call_results": json.dumps(
                        {tool_id: {"points": [{"name": "A", "x": 0, "y": 0}], "segments": []}}
                    ),
                }
            )

            response2 = ollama_api.create_chat_completion(prompt2)

            print("\n=== After tool result ===")
            print(f"Response: {response2.message.content}")
            print(f"Finish reason: {response2.finish_reason}")
            print(f"Message count: {len(ollama_api.messages)}")

            for i, msg in enumerate(ollama_api.messages):
                role = msg.get("role")
                content = str(msg.get("content", ""))[:100]
                print(f"  [{i}] {role}: {content}...")
