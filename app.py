import time
import signal
import sys
from static.app_manager import AppManager


def signal_handler(sig, frame):
    """Handle graceful shutdown on interrupt signal."""
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
        if not app.webdriver_manager:
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