import os
import openai

MODEL = "gpt-3.5-turbo-0613"

class OpenAIChatCompletionsAPI:
    def __init__(self, model=MODEL, temperature=0.2, max_tokens=250):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = []
        self.functions = self._get_functions()
    
    def _get_functions(self):
        functions = [
            {
                "name": "create_point",
                "description": "Creates and draws a point at the given coordinates",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "The X coordinate of the point",
                        },
                        "y": {
                            "type": "number",
                            "description": "The Y coordinate of the point",
                        }
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "delete_point",
                "description": "Deletes the point with the given coordinates",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "The X coordinate of the point",
                        },
                        "y": {
                            "type": "number",
                            "description": "The Y coordinate of the point",
                        }
                    },
                    "required": ["x", "y"]
                }
            }
        ]
        return functions

    def create_chat_completion(self, prompt, function_call="auto"):
        self.messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            functions=self.functions,
            function_call=function_call
        )
        response_message = response["choices"][0]["message"]
        self.messages.append(response_message)
        return response_message

    def get_model_list(self):
        response = openai.Model.list()
        return response