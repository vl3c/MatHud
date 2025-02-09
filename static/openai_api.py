import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from static.functions_definitions import FUNCTIONS
import base64
from static.ai_model import AIModel


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

class OpenAIChatCompletionsAPI:
    DEV_MSG = """You are an educational graphing calculator AI interface that can draw shapes, perform calculations and help users explore mathematics. 

    IMPORTANT: When performing intermediate calculations or multi-step actions:
    1. First call enable_multi_step_mode to enable result tracking (sets the flag which will allow you to receive intermediate results in the canvas state)
    2. Immediately call evaluate_expression or other functions for your calculations
    3. You will receive the results in the next message's canvas state
    4. Use those results for your next calculations or actions (don't forget to call enable_multi_step_mode again if needed)
    5. Repeat this process for complex multi-step calculations

    Always analyze the canvas state to see previously computed results before proceeding."""

    def __init__(self, model=None, temperature=0.2, tools=FUNCTIONS, max_tokens=32000):
        self.client = OpenAI(api_key=api_key)  # Use the api_key we found
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

    def create_chat_completion(self, prompt):
        def remove_canvas_state_from_last_user_message():
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
        
        def remove_image_from_last_user_message():
            if len(self.messages) >= 2 and "content" in self.messages[-2]:
                content = self.messages[-2]["content"]
                if isinstance(content, list):
                    # Keep only the text part
                    text_parts = [part for part in content if part.get("type") == "text"]
                    if text_parts:
                        self.messages[-2]["content"] = text_parts[0]["text"]

        # Parse the prompt which is a JSON string
        prompt_json = json.loads(prompt)
        text_content = prompt_json.get("user_message", "")
        use_vision = prompt_json.get("use_vision", True)  # Get vision toggle state

        # Try to add the canvas image if vision is enabled
        if use_vision:
            try:
                with open("CanvasSnapshots/canvas.png", "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    message_content = [
                        {"type": "text", "text": text_content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
            except Exception as e:
                print(f"Failed to load canvas image: {e}")
                message_content = prompt
        else:
            message_content = prompt

        # Append the new user message
        message = {"role": "user", "content": message_content}
        self.messages.append(message)

        # Make the API call
        response = self.client.chat.completions.create(
            model=self.model.id,
            messages=self.messages,
            tools=self.tools,
            # temperature=self.temperature,
            # max_tokens=self.max_tokens
        )

        # Extract the response message
        response_message = response.choices[0].message
        content = response_message.content or ""
        # Append the AI's response to messages
        self.messages.append({"role": "assistant", "content": content})
        
        # Clean up the messages
        print(f"All messages BEFORE removing canvas_state: \n{self.messages}\n\n")
        remove_canvas_state_from_last_user_message()
        remove_image_from_last_user_message()
        
        return response_message