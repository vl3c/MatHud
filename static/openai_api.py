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
        if len(self.messages) >= 2 and "content" in self.messages[-2]:
            previous_message_content = self.messages[-2]["content"]
            if isinstance(previous_message_content, str):
                try:
                    previous_message_content_json = json.loads(previous_message_content)
                    if "canvas_state" in previous_message_content_json:
                        del previous_message_content_json["canvas_state"]
                        self.messages[-2]["content"] = json.dumps(previous_message_content_json)
                except json.JSONDecodeError:
                    pass

    def _remove_image_from_last_user_message(self):
        if len(self.messages) >= 2 and "content" in self.messages[-2]:
            content = self.messages[-2]["content"]
            if isinstance(content, list):
                # Keep only the text part
                text_parts = [part for part in content if part.get("type") == "text"]
                if text_parts:
                    self.messages[-2]["content"] = text_parts[0]["text"]

    def clean_conversation_history(self):
        """Clean up the conversation history by removing canvas states and images from the last user message."""
        print(f"All messages BEFORE removing canvas_state: \n{self.messages}\n\n")
        self._remove_canvas_state_from_last_user_message()
        self._remove_image_from_last_user_message()

    def _prepare_message_content(self, user_message, use_vision, full_prompt):
        """Prepare message content with optional canvas image for vision-enabled messages.
        
        Args:
            text_content (str): The text content of the message
            use_vision (bool): Whether to include canvas image
            prompt (str): The original prompt to use as fallback
            
        Returns:
            The prepared message content with or without image
        """
        if not use_vision:
            return full_prompt

        try:
            with open("CanvasSnapshots/canvas.png", "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                enhanced_prompt = [
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
                return enhanced_prompt
        except Exception as e:
            print(f"Failed to load canvas image: {e}")
            return full_prompt

    def create_chat_completion(self, full_prompt):
        # Parse the prompt which is a JSON string
        prompt_json = json.loads(full_prompt)
        user_message = prompt_json.get("user_message", "")
        use_vision = prompt_json.get("use_vision", True)  # Get vision toggle state

        # Prepare message content with optional canvas image
        message_content = self._prepare_message_content(user_message, use_vision, full_prompt)

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
            # Create a response object that matches OpenAI's structure
            from types import SimpleNamespace
            error_msg = "I encountered an error processing your request. Please try again."
            error_response = SimpleNamespace(
                content=error_msg,
                tool_calls=[],  # Return empty list instead of None
                choices=[SimpleNamespace(message=SimpleNamespace(content=error_msg, tool_calls=[]))]
            )
            return error_response, "error"  # Return both response and finish_reason

        # Extract the response message
        finish_reason = response.choices[0].finish_reason
        response_message = response.choices[0].message
        content = response_message.content
        # Append the AI's response to messages
        self.messages.append({
            "role": "assistant", 
            "content": content,
            "tool_calls": response_message.tool_calls
        })
        
        # Clean up the messages
        self.clean_conversation_history()
        
        return response_message, finish_reason