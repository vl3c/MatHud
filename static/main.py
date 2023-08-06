import json
import traceback
from browser import ajax, document, html
from canvas import Canvas
from point import Position


# Instantiate the canvas
canvas = Canvas()

available_functions = {
    "reset_canvas": canvas.reset,
    "clear_canvas": canvas.clear,
    "create_point": canvas.create_point,
    "delete_point": canvas.delete_point,
    "create_segment": canvas.create_segment,
    "delete_segment": canvas.delete_segment,
    "create_vector": canvas.create_vector,
    "delete_vector": canvas.delete_vector,
    "create_triangle": canvas.create_triangle,
    "delete_triangle": canvas.delete_triangle,
    "create_rectangle": canvas.create_rectangle,
    "delete_rectangle": canvas.delete_rectangle,
}

# Send message, receive response from the AI and call functions as needed
def interact_with_ai(event):
    
    def parse_ai_response(response):
        def call_functions(ai_response):
            # This function should call the functions specified in the AI's reply
            if ai_response.get("function_call"):
                function_name = ai_response["function_call"]["name"]
                if function_name not in available_functions:
                    print(f"Function {function_name} not found")
                    return
                fuction_to_call = available_functions[function_name]
                function_args = json.loads(ai_response["function_call"]["arguments"])
                print(f"Calling function {function_name} with arguments {function_args}")
                fuction_to_call(**function_args)
        
        def get_ai_message(ai_response):
            # This function should return the text part of the AI's reply
            print(ai_response)
            ai_response_text = "..."
            if ai_response.get("content"):
                ai_response_text = ai_response["content"]
            return ai_response_text
        
        try:
            # Get the text part of the AI's reply
            ai_response_text = get_ai_message(response).replace('\n', '<br>')
            # Add an empty AI response placeholder to the chat history
            document["chat-history"] <= html.P(f'AI: {ai_response_text}', innerHTML=True)    
            # Load the JSON from the AI's reply and call the appropriate functions
            call_functions(response)
        except Exception as e:
            print(f"Error while processing AI's response: {e}")
            traceback.print_exc()
        finally:
            # Scroll the chat history to the bottom
            document["chat-history"].scrollTop = document["chat-history"].scrollHeight
        
    def build_prompt(canvas_state_json, user_message):
        prompt = json.dumps({"canvas_state": canvas_state_json, "user_message": user_message})
        return prompt

    def on_complete(request):
        if request.status == 200 or request.status == 0:
            ai_response = request.json
            # Parse the AI's response and create / delete drawables as needed
            parse_ai_response(ai_response)
        else:
            on_error(request)
    
    def on_error(request):
        print(f"Error: {request.status}, {request.text}")
    
    def send_request(prompt):
        # Send the prompt to the AI
        req = ajax.ajax()
        req.bind('complete', on_complete)
        req.bind('error', on_error)
        req.open('POST', '/send_message', True)
        req.set_header('content-type', 'application/x-www-form-urlencoded')
        req.send({'message': prompt})
    
    # Get the user's message from the input field
    user_message = document["chat-input"].value
    # Add the user's message to the chat history
    document["chat-history"] <= html.P(f'User: {user_message}')
    # Clear the input field
    document["chat-input"].value = ''
    # Scroll the chat history to the bottom
    document["chat-history"].scrollTop = document["chat-history"].scrollHeight
    # Get the canvas state with on-screen drawables original properties 
    canvas_state_json = canvas.get_drawables_state()
    # Build the prompt for the AI
    prompt = build_prompt(canvas_state_json, user_message)
    print(prompt)
    send_request(prompt)

# Bind the send_message function to the send button's click event
document["send-button"].bind("click", interact_with_ai)


# Bind the send_message function to the Enter key in the chat-input field
def check_enter(event):
    if event.keyCode == 13:  # 13 is the key code for Enter
        interact_with_ai(event)

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