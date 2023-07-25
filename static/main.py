import json
from browser import window, document, html
from canvas import function_mapping, Canvas
from point import Position


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
                try:
                    drawable = function_mapping[class_name](**args, canvas=canvas)
                    canvas.add_drawable(drawable)
                except Exception as e:
                    print(f"Error while creating {class_name}: {e}")
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
    canvas.zoom_point = Position(event.clientX - rect.left, event.clientY - rect.top)
    if event.deltaY < 0:
        # Zoom in
        canvas.scale_factor *= 1.1
        canvas.zoom_direction = -1
    else:
        # Zoom out
        canvas.scale_factor *= 0.9
        canvas.zoom_direction = 1
    canvas.draw(True)

document["math-svg"].bind("wheel", handle_wheel)

# Bind the canvas's handle_mousedown function to the mouse down event
def handle_mousedown(event):
    canvas.dragging = True
    canvas.last_mouse_position = Position(event.clientX, event.clientY)

document["math-svg"].bind("mousedown", handle_mousedown)

# Bind the canvas's handle_mouseup function to the mouse up event
def handle_mouseup(event):
    canvas.dragging = False

document["math-svg"].bind("mouseup", handle_mouseup)

# Bind the canvas's handle_mousemove function to the mouse move event
def handle_mousemove(event):
    if canvas.dragging:
        dx = event.clientX - canvas.last_mouse_position.x
        dy = event.clientY - canvas.last_mouse_position.y
        canvas.offset.x += dx
        canvas.offset.y += dy
        canvas.last_mouse_position = Position(event.clientX, event.clientY)
        canvas.draw(False)

document["math-svg"].bind("mousemove", handle_mousemove)


"""
[
    {"class": "Point", "args": {"x": 500, "y": 500}}, 
    {"class": "Point", "args": {"x": 130, "y": 130}}, 
    {"class": "Point", "args": {"x": 130, "y": 500}},
    {"class": "Point", "args": {"x": 500, "y": 130}},
    {"class": "Segment", "args": {"point1": {"x": 70, "y": 20}, "point2": {"x": 100, "y": 200}}},
    {"class": "Triangle", "args": {"point1": {"x": 100, "y": 100}, "point2": {"x": 100, "y": 200}, "point3": {"x": 300, "y": 300}}},
    {"class": "Rectangle", "args": {"top_left": {"x": 150, "y": 250}, "bottom_right": {"x": 400, "y": 300}}},
    {"class": "Rectangle", "args": {"top_left": {"x": 350, "y": 350}, "bottom_right": {"x": 650, "y": 650}}},
    {"class": "Point", "args": {"x": 700, "y": 700}},
    {"class": "Circle", "args": {"center": {"x": 550, "y": 550}, "radius": 150}},
    {"class": "Ellipse", "args": {"center": {"x": 700, "y": 700}, "rx": 100, "ry": 75}},
    {"class": "Vector", "args": {"origin": {"x": 100, "y": 100}, "tip": {"x": 200, "y": 200}}},
    {"class": "Vector", "args": {"origin": {"x": 450, "y": 200}, "tip": {"x": 320, "y": 110}}},
    {"class": "Label", "args": {"position": {"x": 50, "y": 50}, "text": "Hello World!"}},
    {"class": "Label", "args": {"position": {"x": 350, "y": 350}, "text": "12345"}},
    {"class": "Function", "args": {"function_string": "x**2", "start": -50, "end": 50}}
]
"""