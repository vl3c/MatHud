import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from static.functions_definitions import FUNCTIONS

MODEL = "gpt-3.5-turbo"   
# MODEL = "gpt-4-0125-preview" 

dotenv_path = "../../.env"
load_dotenv(dotenv_path)

class OpenAIChatCompletionsAPI:
    def __init__(self, model=MODEL, temperature=0.2, tools=FUNCTIONS, max_tokens=32000):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools
        system_message = "You are an educational graphing calculator AI interface that can draw shapes, perform calculations and help users explore mathematics. IMPORTANT: Before answering, please analize the canvas state and previous results."
        self.messages = [{"role": "system", "content": system_message}]

    def create_chat_completion(self, prompt):
        def remove_canvas_state_from_last_user_message():
            if "content" in self.messages[-2]:
                previous_message_content = self.messages[-2]["content"]
                try:
                    previous_message_content_json = json.loads(previous_message_content)
                    if "canvas_state" in previous_message_content_json:
                        del previous_message_content_json["canvas_state"]
                        self.messages[-2]["content"] = json.dumps(previous_message_content_json)
                except json.JSONDecodeError:
                    # Handle cases where content is not JSON or cannot be decoded
                    pass
        
        def remove_previous_results_from_last_user_message():
            if "content" in self.messages[-2]:
                previous_message_content = self.messages[-2]["content"]
                try:
                    previous_message_content_json = json.loads(previous_message_content)
                    if "previous_results" in previous_message_content_json:
                        del previous_message_content_json["previous_results"]
                        self.messages[-2]["content"] = json.dumps(previous_message_content_json)
                except json.JSONDecodeError:
                    # Handle cases where content is not JSON or cannot be decoded
                    pass

        # Append the new user message to the messages list
        message = {"role": "user", "content": prompt}
        self.messages.append(message)

        # Make the API call to create chat completions
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools = self.tools,
            # temperature=self.temperature,
            # max_tokens=self.max_tokens
        )
        # Extract the response message
        response_message = response.choices[0].message
        content = response_message.content or ""
        # Append the AI's response to messages
        self.messages.append({"role": "assistant", "content": content})
        # Optionally, clean up the canvas state from the last user message
        remove_canvas_state_from_last_user_message()
        
        print(self.messages)   # DEBUG
        
        return response_message