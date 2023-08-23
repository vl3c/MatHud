import json
import time
import traceback
from browser import ajax, document, html
from canvas import Canvas
from point import Position
from math_util import Utilities


# Instantiate the canvas
canvas = Canvas()

def make_multiple_function_calls(function_calls):
    for function_call in function_calls:
        function_name = function_call["name"]
        if function_name not in available_functions:
            print(f"Function {function_name} not found")
            continue
        function_to_call = available_functions[function_name]
        # Exclude the 'name' key to get only the arguments
        function_args = function_call['arguments']
        print(f"Calling function {function_name} with arguments {function_args}")
        function_to_call(**function_args)

def calculate(expression):
    bad_result_msg = "Sorry, that's not a valid mathematical expression."
    try:
        result = Utilities.calculate(expression)
        if result is None:
            return bad_result_msg
        return result
    except Exception as e:
        exception_details = str(e).split(":", 1)[0]
        result = f"{bad_result_msg} Exception details: {exception_details}."
        return result

available_functions = {
    "make_multiple_function_calls": make_multiple_function_calls,
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
    "create_circle": canvas.create_circle,
    "delete_circle": canvas.delete_circle,
    "create_ellipse": canvas.create_ellipse,
    "delete_ellipse": canvas.delete_ellipse,
    "draw_math_function": canvas.draw_math_function,
    "delete_math_function": canvas.delete_math_function,
    "calculate": calculate,
}

# Global variable to store the result of an AI function call
function_call_result = None
def set_global_function_call_result(value):
    global function_call_result
    if not value:
        function_call_result = "Success!"
    function_call_result = value

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
                function_args = json.loads(ai_response["function_call"]["arguments"])
                # If calling multiple functions, pass the whole list
                if function_name == "call_multiple_functions":
                    available_functions[function_name](function_args)
                    return "Called multiple functions."
                else:
                    print(f"Calling function {function_name} with arguments {function_args}")
                    return available_functions[function_name](**function_args)
        
        def create_ai_message(ai_response, function_call_result):
            # This function should return the text part of the AI's reply
            print(ai_response)
            ai_response_text = ""
            if ai_response.get("content"):
                ai_response_text = ai_response["content"]
            if function_call_result and (isinstance(function_call_result, str) or isinstance(function_call_result, int) or isinstance(function_call_result, float) or isinstance(function_call_result, bool)):
                ai_response_text += str(function_call_result)
            if not ai_response_text:
                ai_response_text = "..."
            return ai_response_text
        
        try:
            # Load the JSON from the AI's reply and call the appropriate functions
            function_call_result = call_functions(response)
            set_global_function_call_result(function_call_result)

            # Get the text part of the AI's reply
            ai_response_text = create_ai_message(response, function_call_result).replace('\n', '<br>')               
            # Add an empty AI response placeholder to the chat history
            document["chat-history"] <= html.P(f'<strong>AI:</strong> {ai_response_text}', innerHTML=True)
        except Exception as e:
            print(f"Error while processing AI's response: {e}")
            traceback.print_exc()
        finally:
            # Scroll the chat history to the bottom
            document["chat-history"].scrollTop = document["chat-history"].scrollHeight
        
    def build_prompt(canvas_state_json, function_call_result, user_message):
        prompt = json.dumps({"canvas_state": canvas_state_json, "previous_function_call_result": function_call_result, "user_message": user_message})
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
    document["chat-history"] <= html.P(f'<strong>User:</strong> {user_message}')
    # Clear the input field
    document["chat-input"].value = ''
    # Scroll the chat history to the bottom
    document["chat-history"].scrollTop = document["chat-history"].scrollHeight
    # Get the canvas state with on-screen drawables original properties 
    canvas_state_json = canvas.get_drawables_state()
    # Build the prompt for the AI
    global function_call_result
    prompt = build_prompt(canvas_state_json, function_call_result, user_message)
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

last_click_timestamp = None

# Bind the canvas's handle_mousedown function to the mouse down event
def handle_mousedown(event):
    def get_decimal_places(value):
        # Absolute value to handle negative numbers
        abs_val = abs(value)
        if 0 < abs_val < 1:
            # Number of leading zeros after decimal point + 1
            # Using format to convert the number to string and split by the decimal point
            decimal_part = format(abs_val, ".10f").split(".")[1]
            leading_zeros = len(decimal_part) - len(decimal_part.lstrip('0'))
            decimal_places_needed = leading_zeros + 2
            return decimal_places_needed
        elif 0 < abs_val < 10:
            return 2
        elif abs_val < 100:
            return 1
        else:
            return 0
    global last_click_timestamp
    # Check for double click
    current_timestamp = time.time()
    if last_click_timestamp and (current_timestamp - last_click_timestamp) < 0.5:  # 0.5 seconds threshold
        # It's a double click
        rect = document["math-svg"].getBoundingClientRect()
        canvas_x = event.clientX - rect.left
        canvas_y = event.clientY - rect.top
        scale_factor = canvas.scale_factor
        origin = canvas.cartesian2axis.origin
        # Calculate the coordinates of the clicked point
        x = (canvas_x - origin.x) * 1/scale_factor
        y = (origin.y - canvas_y) * 1/scale_factor
        # Calculate the number of decimal places for x and y
        decimal_places_x = get_decimal_places(x)
        decimal_places_y = get_decimal_places(y)
        # Round x and y to the determined number of decimal places
        x = round(x, decimal_places_x)
        y = round(y, decimal_places_y)
        coordinates = f"({x}, {y}) "
        document["chat-input"].value += coordinates

    last_click_timestamp = current_timestamp

    # Logic for panning the canvas
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