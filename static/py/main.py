import json
from browser import window, document, html
from math_lib import function_mapping, Canvas, Point

# Instantiate the canvas
canvas = Canvas()

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
            class_name = action.get('class')
            if not class_name:
                print("No class specified in action")
            elif class_name not in function_mapping:
                print(f"Class {class_name} not found")
            else:
                args = action.get('args', {})
                drawable = function_mapping[class_name](**args)
                canvas.add_drawable(drawable)
        canvas.draw()
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

# Bind the canvas's handle_wheel function to the mouse wheel event
def handle_wheel(event):
    svg_canvas = document['math-svg']
    rect = svg_canvas.getBoundingClientRect()
    
    # Save the current zoom point and update it to the mouse position
    canvas.last_known_zoom_point = canvas.zoom_point
    canvas.zoom_point = Point(event.clientX - rect.left, event.clientY - rect.top)
    
    if event.deltaY < 0:
        # Zoom in
        canvas.scale_factor *= 1.1
    else:
        # Zoom out
        canvas.scale_factor *= 0.9

    # Redraw the canvas with the new scale factor
    canvas.draw()

document["math-svg"].bind("wheel", handle_wheel)