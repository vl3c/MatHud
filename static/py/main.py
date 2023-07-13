import json
from browser import document, html
from math_lib import function_mapping

def send_message(event):
    # Get the user's message from the input field
    user_message = document["chat-input"].value
    # Add the user's message to the chat history
    document["chat-history"] <= html.P(f'User: {user_message}')

    # Emulate the AI's reply as a function call in JSON
    try:
        ai_reply = user_message
        ai_reply_text = get_reply_text(ai_reply)

        # Add an empty AI response placeholder to the chat history
        document["chat-history"] <= html.P(f'AI: {ai_reply_text}')    

        # Load the JSON from the AI's reply and call the appropriate function
        ai_reply_json = json.loads(get_reply_json(ai_reply))

        for action in ai_reply_json:
            function_name = action.get('function')
            if not function_name:
                print("No function specified in action")
            elif function_name not in function_mapping:
                print(f"Function {function_name} not found")
            else:
                args = action.get('args', {})
                function_mapping[function_name](**args)
    except json.JSONDecodeError:
        print("Error parsing JSON in AI's reply")
    except Exception as e:
        print(f"Error while processing AI's reply: {e}")
    finally:
        # Clear the input field
        document["chat-input"].value = ''

def get_reply_text(ai_reply):
    # This function should return the text part of the AI's reply
    start = ai_reply.find('[')
    if start == -1:
        return ai_reply
    else:
        return ai_reply[:start].strip()

def get_reply_json(ai_reply):
    # This function should return the JSON part of the AI's reply
    start = ai_reply.find('[')
    end = ai_reply.rfind(']')
    if start != -1 and end != -1:
        return ai_reply[start:end+1]
    else:
        return None

# Bind the send_message function to the send button's click event
document["send-button"].bind("click", send_message)


# Bind the send_message function to the Enter key in the chat-input field
def check_enter(event):
    if event.keyCode == 13:  # 13 is the key code for Enter
        send_message(event)

document["chat-input"].bind("keypress", check_enter)

