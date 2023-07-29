import os
import openai

MODEL = "gpt-3.5-turbo-0613"
FUNCTIONS = [
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
                        },
                        "name": {
                            "type": "string",
                            "description": "The name of the point",
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
            },
            {
                "name": "create_segment",
                "description": "Creates and draws a segment at the given coordinates",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x1": {
                            "type": "number",
                            "description": "The X coordinate of the first point",
                        },
                        "y1": {
                            "type": "number",
                            "description": "The Y coordinate of the first point",
                        },
                        "x2": {
                            "type": "number",
                            "description": "The X coordinate of the second point",
                        },
                        "y2": {
                            "type": "number",
                            "description": "The Y coordinate of the second point",
                        },
                        "name": {
                            "type": "string",
                            "description": "The name of the segment",
                        }
                    },
                    "required": ["x1", "y1", "x2", "y2"]
                }
            },
            {
                "name": "delete_segment",
                "description": "Deletes the segment found at the given coordinates",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x1": {
                            "type": "number",
                            "description": "The X coordinate of the first point",
                        },
                        "y1": {
                            "type": "number",
                            "description": "The Y coordinate of the first point",
                        },
                        "x2": {
                            "type": "number",
                            "description": "The X coordinate of the second point",
                        },
                        "y2": {
                            "type": "number",
                            "description": "The Y coordinate of the second point",
                        }
                    },
                    "required": ["x1", "y1", "x2", "y2"]
                }
            },
            {
                "name": "create_vector",
                "description": "Creates and draws a vector at the given coordinates",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin_x": {
                            "type": "number",
                            "description": "The X coordinate of the origin point",
                        },
                        "origin_y": {
                            "type": "number",
                            "description": "The Y coordinate of the origin point",
                        },
                        "tip_x": {
                            "type": "number",
                            "description": "The X coordinate of the tip point",
                        },
                        "tip_y": {
                            "type": "number",
                            "description": "The Y coordinate of the tip point",
                        },
                        "name": {
                            "type": "string",
                            "description": "The name of the vector",
                        }
                    },
                    "required": ["origin_x", "origin_y", "tip_x", "tip_y"]
                }
            },
            {
                "name": "delete_vector",
                "description": "Deletes the vector found at the given coordinates",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin_x": {
                            "type": "number",
                            "description": "The X coordinate of the origin point",
                        },
                        "origin_y": {
                            "type": "number",
                            "description": "The Y coordinate of the origin point",
                        },
                        "tip_x": {
                            "type": "number",
                            "description": "The X coordinate of the tip point",
                        },
                        "tip_y": {
                            "type": "number",
                            "description": "The Y coordinate of the tip point",
                        }
                    },
                    "required": ["origin_x", "origin_y", "tip_x", "tip_y"]
                }
            },
        ]

class OpenAIChatCompletionsAPI:
    def __init__(self, model=MODEL, temperature=0.2, max_tokens=250):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = []
        self.functions = FUNCTIONS

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