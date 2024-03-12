import logging
import os
from datetime import datetime
from flask import Flask, json, request, render_template
from static.openai_api import OpenAIChatCompletionsAPI

def get_log_file_name():
    return datetime.now().strftime('mathud_session_%y_%m_%d_%H_%M_%S.log')

def set_up_logging():
    if not os.path.exists('./logs/'):
        os.makedirs('./logs/')
    logging.basicConfig(filename='./logs/' +  get_log_file_name(), level=logging.INFO, format='%(asctime)s %(message)s')

def log_user_message(user_message):
    user_message_json = json.loads(user_message)
    if "canvas_state" in user_message_json:
        canvas_state = user_message_json["canvas_state"]
        logging.info(f'### Canvas state: {canvas_state}')
    if "previous_results" in user_message_json:
        previous_results = user_message_json["previous_results"]
        logging.info(f'### Previously calculated results: {previous_results}')
    if "user_message" in user_message_json:
        user_message = user_message_json["user_message"]
        logging.info(f'### User message: {user_message}')

def jsonify_tool_calls(tool_calls):
    simple_tool_calls = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        simple_tool_call = {"function_name": function_name, "arguments": arguments}
        simple_tool_calls.append(simple_tool_call)
    return simple_tool_calls

def create_app():
    app = Flask(__name__)
    set_up_logging()
    openai_api = OpenAIChatCompletionsAPI()

    @app.route('/')
    def get_index():
        return render_template('index.html')

    @app.route('/send_message', methods=['POST'])
    def send_message():
        message = request.json.get('message')
        log_user_message(message)
        response = openai_api.create_chat_completion(message)
        ai_message = response.content or ""
        logging.info(f'### AI response: {ai_message}')
        ai_tool_calls = response.tool_calls or []
        ai_tool_calls = jsonify_tool_calls(ai_tool_calls)
        logging.info(f'### AI tool calls: {ai_tool_calls}')
        response = json.dumps({"ai_message": ai_message, "ai_tool_calls": ai_tool_calls})
        return response
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)