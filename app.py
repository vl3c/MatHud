from flask import Flask, request, render_template
from static.openai_api import OpenAIChatCompletionsAPI


app = Flask(__name__)
openai_api = OpenAIChatCompletionsAPI()

@app.route('/')
def get_index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.form['message']
    response_message = openai_api.create_chat_completion(message)
    return response_message

if __name__ == '__main__':
    app.run(debug=True)