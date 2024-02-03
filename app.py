from flask import Flask, json, request, render_template
from static.openai_api import OpenAIChatCompletionsAPI


app = Flask(__name__)
openai_api = OpenAIChatCompletionsAPI()

def jsonify_tool_calls(tool_calls):
    simple_tool_calls = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        simple_tool_call = {"function_name": function_name, "arguments": arguments}
        simple_tool_calls.append(simple_tool_call)
    return simple_tool_calls

@app.route('/')
def get_index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.json.get('message')
    response = openai_api.create_chat_completion(message)
    ai_message = response.content or ""
    ai_tool_calls = response.tool_calls or []
    ai_tool_calls = jsonify_tool_calls(ai_tool_calls)
    response = json.dumps({"ai_message": ai_message, "ai_tool_calls": ai_tool_calls})
    return response

if __name__ == '__main__':
    app.run(debug=True)