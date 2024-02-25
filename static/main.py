from browser import ajax, document, html
from canvas import Canvas
from math_util import MathUtil
from point import Position
import Tests.tests as tests
import json
import re
import time
import traceback


# Instantiate the canvas
viewport = document['math-svg'].getBoundingClientRect()
canvas = Canvas(viewport.width, viewport.height)


def evaluate_expression(expression, variables=None):
    def evaluate_numeric_expression(expression, variables):
        return MathUtil.evaluate(expression, variables)
    
    def evaluate_function(expression):
        print(f"Evaluating function with expression: {expression}")   # DEBUG
        functions = canvas.get_drawables_by_class_name('Function')
        # Split the expression into function name and argument
        match = re.match(r'(\w+)\((.+)\)', expression)
        if match:
            function_name, argument = match.groups()
            print(f"Function name: {function_name}, argument: {argument}")   # DEBUG
        else:
            raise ValueError(f"Invalid function expression: {expression}")
        for function in functions:
            if function.name.lower() == function_name.lower():
                # If the function name matches, evaluate the function
                print(f"Found function: {function.name} = {function.function_string}")   # DEBUG
                try:
                    argument = float(argument)
                except ValueError:
                    raise ValueError(f"Invalid argument for function: {argument}")
                return function.function(argument)
    
    bad_result_msg = "Sorry, that's not a valid mathematical expression."
    try:
        # First, try to evaluate the expression as a numeric expression
        result = evaluate_numeric_expression(expression, variables)
        if not result:
            return bad_result_msg
        return result
    except Exception as e:
        try:
            # If that fails, try to evaluate the expression as a function from the canvas
            result = evaluate_function(expression)
            if not result:
                return bad_result_msg
            return result
        except Exception as e:
            exception_details = str(e).split(":", 1)[0]
            result = f"{bad_result_msg} Exception details: {exception_details}."
            return result

def process_function_calls(calls):
    if not calls:
        return []
    results = []
    # Archive once at the start and then suspend archiving while calling undoable functions
    contains_undoable_function = any(call.get('function_name', '') in undoable_functions for call in calls)
    if contains_undoable_function:
        canvas.archive()
    # Extract function name and arguments from the tool_call
    for call in calls:
        function_name = call.get('function_name', '')
        if function_name not in available_functions:
            print(f"Error: function {function_name} not found.")
            continue  # Skip this function call and proceed to the next
        try:
            # Assuming arguments are directly accessible as a dictionary
            # If arguments are stored as a JSON string, you might need json.loads(tool_call['function']['arguments'])
            args = call.get('arguments', {})
            # Execute the function with its arguments
            print(f"Calling function {function_name} with arguments {args}")
            result = available_functions[function_name](**args)
            results.append(result)
        except Exception as e:
            error_message = f"Error calling function {function_name}: {e}"
            print(error_message)
            results.append(f"\n{error_message}")
    return results

def run_tests():
    function_calls = [
        {
            "function_name": "create_point",
            "arguments": {"x": -200, "y": 100, "name": "A"}
        },
        {
            "function_name": "create_point",
            "arguments": {"x": 250, "y": -150, "name": "B"}
        },
        {
            "function_name": "create_segment",
            "arguments": {"x1": -200, "y1": 100, "x2": 250, "y2": -150, "name": "AB"}
        },
        {
            "function_name": "create_vector",
            "arguments": {"origin_x": -150, "origin_y": -200, "tip_x": 100, "tip_y": 200, "name": "v1"}
        },
        {
            "function_name": "create_triangle",
            "arguments": {"x1": -100, "y1": -150, "x2": 120, "y2": 130, "x3": 150, "y3": -100, "name": "ABC"}
        },
        {
            "function_name": "create_rectangle",
            "arguments": {"diagonal_p1_x": -250, "diagonal_p1_y": 250, "diagonal_p2_x": 220, "diagonal_p2_y": -220, "name": "Rect1"}
        },
        {
            "function_name": "create_circle",
            "arguments": {"center_x": 0, "center_y": 0, "radius": 150, "name": "Circle1"}
        },
        {
            "function_name": "create_ellipse",
            "arguments": {"center_x": 200, "center_y": -100, "radius_x": 60, "radius_y": 90, "name": "Ellipse1"}
        },
        {
            "function_name": "draw_math_function",
            "arguments": {"function_string": "100 * sin(x / 50) + 50 * tan(x / 100)", "name": "f1", "left_bound": -300, "right_bound": 300}
        },
        {
            "function_name": "draw_math_function",
            "arguments": {"function_string": "100 * sin(x / 30)", "name": "f2", "left_bound": -300, "right_bound": 300}
        },
        {
            "function_name": "clear_canvas",
            "arguments": {}
        },
        {
            "function_name": "undo",
            "arguments": {}
        },
        {
            "function_name": "redo",
            "arguments": {}
        },
        {
            "function_name": "undo",
            "arguments": {}
        }
    ]
    process_function_calls(function_calls)
    tests.run_tests()

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
    "create_circle": canvas.create_circle,
    "delete_circle": canvas.delete_circle,
    "create_ellipse": canvas.create_ellipse,
    "delete_ellipse": canvas.delete_ellipse,
    "draw_math_function": canvas.draw_math_function,
    "delete_math_function": canvas.delete_math_function,
    "evaluate_expression": evaluate_expression,
    "undo": canvas.undo,
    "redo": canvas.redo,
    "run_tests": run_tests,
    "convert": MathUtil.convert,
    "limit": MathUtil.limit,
    "derivative": MathUtil.derivative,
    "integral": MathUtil.integral,
    "simplify": MathUtil.simplify,
    "expand": MathUtil.expand,
    "factor": MathUtil.factor,
    "solve": MathUtil.solve,
    "solve_system_of_equations": MathUtil.solve_system_of_equations,
}

undoable_functions = ["clear_canvas", "reset_canvas", "create_point", "delete_point", "create_segment", "delete_segment", "create_vector", "delete_vector", \
                      "create_triangle", "delete_triangle", "create_rectangle", "delete_rectangle", "create_circle", "delete_circle", "create_ellipse", "delete_ellipse", \
                      "draw_math_function", "delete_math_function"]

def validate_function_call_result(results):
    allowed_types = (str, int, float, bool)
    if not isinstance(results, list):
        return False
    return all(isinstance(result, allowed_types) for result in results)

# Global variable to store the result of an AI function call
function_call_results = []
def set_global_function_call_result(value):
    global function_call_results
    if not validate_function_call_result(value):
        function_call_results = []
    else:
        function_call_results = value
    return function_call_results

# Send message, receive response from the AI and call functions as needed
def interact_with_ai(event):
    def parse_ai_response(ai_message, ai_function_calls):
        def create_ai_message(ai_response, function_call_results):
            # This function should return the text part of the AI's reply
            ai_response_text = ""
            if ai_response:
                ai_response_text = ai_response
            if function_call_results and validate_function_call_result(function_call_results):
                ai_response_text += ', '.join(map(str, function_call_results))
            if not ai_response_text:
                ai_response_text = "..."
            return ai_response_text
        
        try:
            # Load the JSON from the AI's reply and call the appropriate functions
            print(f"AI message: {ai_message}")   # DEBUG
            print(f"AI function calls: {ai_function_calls}")   # DEBUG

            call_results = process_function_calls(ai_function_calls)
            call_results = set_global_function_call_result(call_results)

            # Get the text part of the AI's reply
            ai_response_text = create_ai_message(ai_message, call_results).replace('\n', '<br>')               
            # Add an empty AI response placeholder to the chat history
            document["chat-history"] <= html.P(f'<strong>AI:</strong> {ai_response_text}', innerHTML=True)
        except Exception as e:
            print(f"Error while processing AI's response: {e}")
            traceback.print_exc()
        finally:
            # Scroll the chat history to the bottom
            document["chat-history"].scrollTop = document["chat-history"].scrollHeight
        
    def build_prompt(canvas_state, function_call_results, user_message):
        prompt = json.dumps({"canvas_state": canvas_state, "previous_results": function_call_results, "user_message": user_message})
        return prompt

    def on_complete(request):
        if request.status == 200 or request.status == 0:
            ai_response = request.json
            ai_message, ai_function_calls = ai_response.get('ai_message'), ai_response.get('ai_tool_calls')
            # Parse the AI's response and create / delete drawables as needed
            parse_ai_response(ai_message, ai_function_calls)
        else:
            on_error(request)
    
    def on_error(request):
        print(f"Error: {request.status}, {request.text}")
    
    def send_request(prompt):
        # Updated to handle new API response format and tool calls
        req = ajax.ajax()
        req.bind('complete', on_complete)
        req.bind('error', on_error)
        req.open('POST', '/send_message', True)
        req.set_header('content-type', 'application/json')  # Ensure this matches expected content type
        # Update to send JSON including any necessary data for tools
        req.send(json.dumps({'message': prompt}))

    # Get the user's message from the input field
    user_message = document["chat-input"].value
    # Add the user's message to the chat history
    document["chat-history"] <= html.P(f'<strong>User:</strong> {user_message}')
    # Clear the input field
    document["chat-input"].value = ''
    # Scroll the chat history to the bottom
    document["chat-history"].scrollTop = document["chat-history"].scrollHeight
    # Get the canvas state with on-screen drawables original properties 
    canvas_state = canvas.get_canvas_state()
    # Build the prompt for the AI
    global function_call_results
    prompt = build_prompt(canvas_state, function_call_results, user_message)
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