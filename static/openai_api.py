import os
import openai
import json
from static.functions_definitions import FUNCTIONS

MODEL = "gpt-4-0613"   # "gpt-3.5-turbo-0613" 

class OpenAIChatCompletionsAPI:
    def __init__(self, model=MODEL, temperature=0.2, max_tokens=4000):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = [{"role": "system", "content": "You are an educational graphing calculator AI interface that can draw shapes, perform calculations and help explore math."}]
        self.functions = FUNCTIONS

    def create_chat_completion(self, prompt, function_call="auto"):
        def remove_canvas_state_from_last_user_message():
            previous_message_content = self.messages[-2]["content"]
            previous_message_content_json = json.loads(previous_message_content)
            if "canvas_state" in previous_message_content_json:
                del previous_message_content_json["canvas_state"]
                self.messages[-2]["content"] = json.dumps(previous_message_content_json)
        
        message = {"role": "user", "content": prompt}
        self.messages.append(message)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            functions=self.functions,
            function_call=function_call
        )
        response_message = response["choices"][0]["message"]
        self.messages.append(response_message)
        print(self.messages)   # DEBUG
        remove_canvas_state_from_last_user_message()
        return response_message

    def get_model_list(self):
        response = openai.Model.list()
        return response