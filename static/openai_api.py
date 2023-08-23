import os
import openai
import json
from functions_definitions import FUNCTIONS

MODEL = "gpt-3.5-turbo-0613"   # "gpt-4-0613" 

class OpenAIChatCompletionsAPI:
    def __init__(self, model=MODEL, temperature=0.2, max_tokens=4000):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = [{"role": "system", "content": "You are an educational graphing calculator AI interface that can draw shapes, perform calculations and help explore math."}]
        self.functions = FUNCTIONS

    def create_chat_completion(self, prompt, function_call="auto"):
        def remove_canvas_state_from_last_message():
            prompt_json = json.loads(prompt)
            user_message = prompt_json["user_message"]
            self.messages[-1]["content"] = json.dumps(user_message)
        
        message = {"role": "user", "content": prompt}
        self.messages.append(message)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            functions=self.functions,
            function_call=function_call
        )
        remove_canvas_state_from_last_message()
        response_message = response["choices"][0]["message"]
        self.messages.append(response_message)
        return response_message

    def get_model_list(self):
        response = openai.Model.list()
        return response