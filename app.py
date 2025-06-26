"""
MatHud Flask Application Entry Point

Main application launcher that initializes the Flask server, manages graceful shutdown,
and coordinates the WebDriver for vision system functionality.

Dependencies:
    - static.app_manager: Application configuration and initialization
    - Flask: Web framework (initialized via AppManager)
    - Selenium WebDriver: Vision system image capture (lazy-loaded)
    - Threading: Concurrent server execution
"""

import time
import signal
import sys
from static.app_manager import AppManager


def signal_handler(sig, frame):
    """Handle graceful shutdown on interrupt signal.
    
    Cleans up WebDriver resources and exits the application properly.
    """
    print('\nShutting down gracefully...')
    # Clean up WebDriverManager
    if hasattr(app, 'webdriver_manager') and app.webdriver_manager:
        try:
            app.webdriver_manager.cleanup()
        except Exception as e:
            print(f"Error closing WebDriver: {e}")
    print("Goodbye!")
    sys.exit(0)


# Create the app at module level for VS Code debugger
app = AppManager.create_app()

# Register signal handler at module level for both run modes
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    """Main execution block.
    
    Starts Flask server in a daemon thread, initializes WebDriver for vision system,
    and maintains the main thread for graceful interrupt handling.
    """
    try:
        # Get port from environment variable (for deployment platforms like Render)
        import os
        port = int(os.environ.get('PORT', 5000))
        
        # Check if we're running in a deployment environment
        is_deployed = os.environ.get('PORT') is not None
        
        # Enable debug mode for local development
        debug_mode = not is_deployed  # Debug only when running locally
        
        if is_deployed:
            # For deployment: run Flask directly without threading
            host = '0.0.0.0'  # Bind to all interfaces for deployment
            print(f"Starting Flask app on {host}:{port} (deployment mode)")
            app.run(host=host, port=port, debug=False)
        else:
            # For local development: use threading approach with debug capability
            host = '127.0.0.1'  # Localhost for development
            print(f"Starting Flask app on {host}:{port} (development mode, debug={debug_mode})")
            
            from threading import Thread
            server = Thread(target=app.run, kwargs={
                'host': host,
                'port': port,
                'debug': debug_mode,
                'use_reloader': False  # Disable reloader in thread mode to avoid issues
            })
            server.daemon = True  # Make the server thread a daemon so it exits when main thread exits
            server.start()
            
            # Wait for Flask to start
            time.sleep(3)
            
            # Initialize WebDriver (only in local development)
            if not app.webdriver_manager:
                import requests
                try:
                    response = requests.get(f'http://{host}:{port}/init_webdriver')
                    print("WebDriver initialized successfully")
                except Exception as e:
                    print(f"Failed to initialize WebDriver: {str(e)}")
            
            print(f"MatHud is running at http://{host}:{port}")
            print("Press Ctrl+C to stop the server")
            
            # Keep the main thread alive but responsive to keyboard interrupts
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)