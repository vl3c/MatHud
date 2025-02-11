import logging
import os
from datetime import datetime
from flask import Flask, json, request, render_template, jsonify
from static.openai_api import OpenAIChatCompletionsAPI
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import signal
import sys

# Keep the utility functions
def get_log_file_name():
    return datetime.now().strftime('mathud_session_%y_%m_%d.log')

def set_up_logging():
    if not os.path.exists('./logs/'):
        os.makedirs('./logs/')
    logging.basicConfig(
        filename=os.path.join('./logs/', get_log_file_name()),
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )
    log_new_session_start()

def log_new_session_start():
    session_delimiter = f"\n\n###### SESSION {datetime.now().strftime('%H:%M:%S')} ######\n"
    logging.info(session_delimiter)

def log_user_message(user_message):
    try:
        user_message_json = json.loads(user_message)
    except json.JSONDecodeError:
        logging.error("Failed to decode user message JSON.")
        return
    
    if "svg_state" in user_message_json:
        svg_state = user_message_json["svg_state"]
        logging.info(f'### SVG state dimensions: {svg_state["dimensions"]}')
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
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            arguments = {}
            logging.error(f"Failed to decode arguments for function {function_name}.")
        simple_tool_call = {"function_name": function_name, "arguments": arguments}
        simple_tool_calls.append(simple_tool_call)
    return simple_tool_calls

def capture_canvas(driver):
    print("\nStarting capture_canvas...")
    try:
        snapshots_dir = "CanvasSnapshots"
        if not os.path.exists(snapshots_dir):
            os.makedirs(snapshots_dir)
        
        # Wait for SVG element and ensure it's visible
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "math-svg"))
        )
        
        # Wait for all SVG child elements to be present
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((
                By.CSS_SELECTOR, 
                "#math-svg > *"
            ))
        )

        # Verify SVG content is present
        svg = driver.find_element(By.ID, "math-svg")
        if not svg.get_attribute("innerHTML").strip():
            raise Exception("SVG content is empty")

        # Get the actual rendered dimensions using JavaScript
        dimensions = driver.execute_script("""
            var container = document.querySelector('.math-container');
            var rect = container.getBoundingClientRect();
            return {
                width: rect.width,
                height: rect.height
            };
        """)
        
        width = dimensions['width']
        height = dimensions['height']
        
        # Set explicit size for SVG and ensure container is properly sized
        driver.execute_script("""
            var container = document.querySelector('.math-container');
            var svg = document.getElementById('math-svg');
            
            // Set container style
            container.style.width = arguments[0] + 'px';
            container.style.height = arguments[1] + 'px';
            
            // Set SVG attributes
            svg.setAttribute('width', arguments[0]);
            svg.setAttribute('height', arguments[1]);
            svg.setAttribute('viewBox', '0 0 ' + arguments[0] + ' ' + arguments[1]);
            svg.style.width = '100%';
            svg.style.height = '100%';
            
            // Force all SVG elements to be visible
            var elements = svg.getElementsByTagName('*');
            for(var i=0; i < elements.length; i++) {
                elements[i].style.visibility = 'visible';
                elements[i].style.opacity = '1';
            }
        """, width, height)
        
        time.sleep(1)  # Give time for the changes to take effect
        
        # Take screenshot of the container
        canvas_path = os.path.join(snapshots_dir, "canvas.png")
        container = driver.find_element(By.CLASS_NAME, "math-container")
        container.screenshot(canvas_path)
        print(f"Canvas capture completed successfully (dimensions: {width}x{height})")
    except Exception as e:
        print(f"Error in capture_canvas: {str(e)}")
        logging.error(f"Error in capture_canvas: {str(e)}")

def init_webdriver(app):
    """Initialize WebDriver after Flask has started"""
    print("Initializing WebDriver...")
    firefox_options = Options()
    firefox_options.add_argument('--headless')  # Make Firefox run in headless mode
    
    # Add retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if not hasattr(app, 'driver') or app.driver is None:
                app.driver = webdriver.Firefox(options=firefox_options)
                print("WebDriver started successfully.")
            
            # Wait for Flask to start
            time.sleep(3)  # Increased delay
            
            print(f"Attempting to navigate (attempt {attempt + 1}/{max_retries})...")
            app.driver.get("http://127.0.0.1:5000/")
            
            # Hide the chat container
            app.driver.execute_script("""
                const chatContainer = document.querySelector('.chat-container');
                if (chatContainer) {
                    chatContainer.style.display = 'none';
                }
                const mathContainer = document.querySelector('.math-container');
                if (mathContainer) {
                    mathContainer.style.width = '100%';
                }
            """)
            
            print("WebDriver navigation successful.")
            return  # Success, exit the function
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if hasattr(app, 'driver') and app.driver:
                app.driver.quit()
                app.driver = None
            
            if attempt < max_retries - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("All attempts failed.")

def create_app():
    app = Flask(__name__)
    set_up_logging()
    app.ai_api = OpenAIChatCompletionsAPI()
    
    # Initialize driver as None; will be set after Flask starts
    app.driver = None
    
    # Initialize workspace manager
    from static.workspace_manager import WorkspaceManager
    workspace_manager = WorkspaceManager()

    @app.route('/')
    def get_index():
        return render_template('index.html')

    @app.route('/init_webdriver')
    def init_webdriver_route():
        """Route to initialize WebDriver after Flask has started"""
        if not app.driver:
            init_webdriver(app)
        return "WebDriver initialization attempted"

    @app.route('/save_workspace', methods=['POST'])
    def save_workspace_route():
        """Save the current workspace state."""
        try:
            data = request.get_json()
            state = data.get('state')
            name = data.get('name')
            
            success = workspace_manager.save_workspace(state, name)
            if success:
                return jsonify({'status': 'success', 'message': 'Workspace saved successfully'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to save workspace'}), 500
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/load_workspace', methods=['GET'])
    def load_workspace_route():
        """Load a workspace state."""
        try:
            name = request.args.get('name')
            state = workspace_manager.load_workspace(name)
            
            return jsonify({
                'status': 'success',
                'state': state
            })
        except FileNotFoundError as e:
            return jsonify({'status': 'error', 'message': str(e)}), 404
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/list_workspaces', methods=['GET'])
    def list_workspaces_route():
        """List all saved workspaces."""
        try:
            workspaces = workspace_manager.list_workspaces()
            return jsonify(workspaces)
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/delete_workspace', methods=['GET'])
    def delete_workspace_route():
        """Delete a workspace."""
        try:
            name = request.args.get('name')
            if not name:
                return jsonify({'status': 'error', 'message': 'Workspace name is required'}), 400
                
            success = workspace_manager.delete_workspace(name)
            if success:
                return jsonify({'status': 'success', 'message': 'Workspace deleted successfully'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to delete workspace'}), 404
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/send_message', methods=['POST'])
    def send_message():
        message = request.json.get('message')
        svg_state = request.json.get('svg_state')  # Get SVG state from request
        message_json = json.loads(message)
        use_vision = message_json.get('use_vision', False)  # Get vision state from message
        ai_model = message_json.get('ai_model')  # Get AI model from message
        
        if ai_model:
            app.ai_api.set_model(ai_model)

        log_user_message(message)

        # Check if WebDriver needs to be initialized
        if use_vision and (not hasattr(app, 'driver') or app.driver is None):
            print("WebDriver not found, attempting to initialize...")
            init_webdriver(app)

        # Capture canvas image before sending to AI
        if use_vision and hasattr(app, 'driver') and app.driver:
            try:
                print("Loading SVG state...")
                # Set the SVG content and attributes directly
                app.driver.execute_script("""
                    const svg = document.getElementById('math-svg');
                    const container = document.querySelector('.math-container');
                    
                    // Set the SVG content
                    svg.outerHTML = arguments[0].content;
                    
                    // Set container dimensions
                    container.style.width = arguments[0].dimensions.width + 'px';
                    container.style.height = arguments[0].dimensions.height + 'px';
                    
                    // Set SVG attributes
                    const newSvg = document.getElementById('math-svg');  // Get reference to new SVG after outerHTML
                    if (arguments[0].viewBox) {
                        newSvg.setAttribute('viewBox', arguments[0].viewBox);
                    }
                    if (arguments[0].transform) {
                        newSvg.setAttribute('transform', arguments[0].transform);
                    }
                    
                    return true;  // Confirm execution
                """, svg_state)
                
                # Add a small delay to ensure the SVG is redrawn
                time.sleep(1)
                
                # Now capture the canvas
                capture_canvas(app.driver)
                
            except Exception as e:
                print(f"Failed to capture canvas: {str(e)}")
                logging.error(f"Failed to capture canvas: {str(e)}")

        # Proceed with creating chat completion
        response = app.ai_api.create_chat_completion(message)
        ai_message = response.content if response.content is not None else ""
        logging.info(f'### AI response: {ai_message}')
        ai_tool_calls = response.tool_calls if response.tool_calls is not None else []
        ai_tool_calls = jsonify_tool_calls(ai_tool_calls)
        logging.info(f'### AI tool calls: {ai_tool_calls}')
        response = json.dumps({"ai_message": ai_message, "ai_tool_calls": ai_tool_calls})
        return response

    return app

# Create the app at module level for VS Code debugger
app = create_app()

def signal_handler(sig, frame):
    print('\nShutting down gracefully...')
    # Only quit the specific WebDriver instance
    if hasattr(app, 'driver') and app.driver:
        try:
            app.driver.quit()
        except Exception as e:
            print(f"Error closing WebDriver: {e}")
    print("Goodbye!")
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    try:
        # Start Flask in a thread
        from threading import Thread
        server = Thread(target=app.run, kwargs={
            'host': '127.0.0.1',
            'port': 5000,
            'debug': False,
            'use_reloader': False
        })
        server.daemon = True  # Make the server thread a daemon so it exits when main thread exits
        server.start()
        
        # Wait for Flask to start
        time.sleep(3)
        
        # Initialize WebDriver
        if not app.driver:
            import requests
            try:
                response = requests.get('http://127.0.0.1:5000/init_webdriver')
            except Exception as e:
                print(f"Failed to initialize WebDriver: {str(e)}")
        
        # Keep the main thread alive but responsive to keyboard interrupts
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)