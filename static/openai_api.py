import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from static.functions_definitions import FUNCTIONS
import base64
from static.ai_model import AIModel


class OpenAIChatCompletionsAPI:
    DEV_MSG = """You are an educational graphing calculator AI interface that can draw shapes, perform calculations and help users explore mathematics. 

    IMPORTANT: When performing intermediate calculations or multi-step actions:
    1. First call enable_multi_step_mode to enable result tracking (sets the flag which will allow you to receive intermediate results in the canvas state)
    2. Immediately call evaluate_expression or other functions for your calculations
    3. You will receive the results in the next message's canvas state
    4. Use those results for your next calculations or actions (don't forget to call enable_multi_step_mode again if needed)
    5. Repeat this process for complex multi-step calculations

    Always analyze the canvas state to see previously computed results before proceeding."""

    @staticmethod
    def _initialize_api_key():
        """Initialize the OpenAI API key from environment or .env file.
        
        Returns:
            str: The initialized API key
            
        Raises:
            ValueError: If API key is not found in environment or .env file
        """
        # First check if OPENAI_API_KEY is already set in environment
        api_key = os.getenv("OPENAI_API_KEY")

        # If not found, try to load from .env file
        if not api_key:
            dotenv_path = ".env"
            if os.path.exists(dotenv_path):
                load_dotenv(dotenv_path)
                api_key = os.getenv("OPENAI_API_KEY")
            else:
                raise ValueError("OPENAI_API_KEY not found in environment or .env file")
                
        return api_key

    def __init__(self, model=None, temperature=0.2, tools=FUNCTIONS, max_tokens=32000):
        self.client = OpenAI(api_key=self._initialize_api_key())  # Initialize and use the api_key
        self.model = model if model else AIModel.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools
        self.messages = [{"role": "developer", "content": OpenAIChatCompletionsAPI.DEV_MSG}]

    def get_model(self):
        return self.model
    
    def set_model(self, identifier):
        # Only create a new AIModel if the identifier is different
        if str(self.model) != identifier:
            self.model = AIModel.from_identifier(identifier)
            print(f"API model updated to: {identifier}")

    def _remove_canvas_state_from_last_user_message(self):
        """Remove canvas state from the last user message in the conversation history."""
        # Find the last user message
        for message in reversed(self.messages):
            if message["role"] == "user" and "content" in message:
                if isinstance(message["content"], str):
                    try:
                        message_content_json = json.loads(message["content"])
                        if "canvas_state" in message_content_json:
                            del message_content_json["canvas_state"]
                            message["content"] = json.dumps(message_content_json)
                    except json.JSONDecodeError:
                        pass
                break  # Stop after processing the most recent user message

    def _remove_image_from_last_user_message(self):
        """Remove image content from the last user message in the conversation history."""
        # Find the last user message
        for message in reversed(self.messages):
            if message["role"] == "user" and "content" in message:
                content = message["content"]
                if isinstance(content, list):
                    # Keep only the text part
                    text_parts = [part for part in content if part.get("type") == "text"]
                    if text_parts:
                        message["content"] = text_parts[0]["text"]
                break  # Stop after processing the most recent user message

    def _clean_conversation_history(self):
        """Clean up the conversation history by removing canvas states and images from the last user message."""
        print(f"All messages BEFORE removing canvas_state: \n{self.messages}\n\n")
        self._remove_canvas_state_from_last_user_message()
        self._remove_image_from_last_user_message()

    def _create_enhanced_prompt_with_image(self, user_message):
        """Create an enhanced prompt that includes both text and base64 encoded image.
        
        Args:
            user_message (str): The user's text message
            
        Returns:
            list: Enhanced prompt with text and image data, or None if image loading fails
        """
        try:
            with open("CanvasSnapshots/canvas.png", "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                ret_val = [
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

    def _prepare_message_content(self, full_prompt):
        """Prepare message content with optional canvas image for vision-enabled messages.
        
        Args:
            full_prompt (str): The full JSON prompt string containing user message and vision flag
            
        Returns:
            The prepared message content with or without image
        """
        # Parse the prompt which is a JSON string
        prompt_json = json.loads(full_prompt)
        user_message = prompt_json.get("user_message", "")
        use_vision = prompt_json.get("use_vision", True)  # Get vision toggle state

        if not use_vision:
            return full_prompt

        enhanced_prompt = self._create_enhanced_prompt_with_image(user_message)
        return enhanced_prompt if enhanced_prompt else full_prompt

    def _create_assistant_message(self, response_message):
        """Create an assistant message from the API response message.
        
        Args:
            response_message: The message from the API response
            
        Returns:
            dict: The formatted assistant message
        """
        assistant_message = {
            "role": "assistant", 
            "content": response_message.content,
        }

        if response_message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in response_message.tool_calls
            ]
        
        return assistant_message

    def _create_error_response(self, error_message="I encountered an error processing your request. Please try again."):
        """Create an error response that matches OpenAI's response structure.
        
        Args:
            error_message (str): The error message to include in the response
            
        Returns:
            SimpleNamespace: A response choice object matching OpenAI's structure
        """
        from types import SimpleNamespace
        return SimpleNamespace(
            message=SimpleNamespace(
                content=error_message,
                tool_calls=[]
            ),
            finish_reason="error"
        )

    def _create_tool_message(self, tool_call_id, content):
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

    def _append_tool_messages(self, tool_calls):
        """Create and append tool messages for each tool call.
        
        Args:
            tool_calls: List of tool calls from the assistant's response
        """
        if tool_calls:
            for tool_call in tool_calls:
                # Here you would typically execute the tool and get its result
                # For now, we'll just create a placeholder response
                tool_message = self._create_tool_message(
                    tool_call.id,
                    f"Tool {tool_call.function.name} executed with args: {tool_call.function.arguments}"
                )
                self.messages.append(tool_message)

    def create_chat_completion(self, full_prompt):
        # Prepare message content with optional canvas image
        message_content = self._prepare_message_content(full_prompt)

        # Append the new user message
        message = {"role": "user", "content": message_content}
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
        self._append_tool_messages(choice.message.tool_calls)
        
        # Clean up the messages
        self._clean_conversation_history()
        
        return choice