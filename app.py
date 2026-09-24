"""MatHud Flask Application entry point."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from types import FrameType

from static.app_manager import AppManager, MatHudFlask


def signal_handler(sig: int, frame: FrameType | None) -> None:
    """Handle graceful shutdown on interrupt signal and exit the application."""
    print("\nShutting down gracefully...")
    print("Goodbye!")
    sys.exit(0)


# Create the app at module level for VS Code debugger
app: MatHudFlask = AppManager.create_app()

# Register signal handler at module level for both run modes
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    """Main execution block.

    Starts Flask server in a daemon thread and maintains the main thread for
    graceful interrupt handling.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="MatHud Flask Application")
    parser.add_argument(
        "-p", "--port", type=int, default=None, help="Port to run the server on (default: 5000, or PORT env var)"
    )
    args = parser.parse_args()

    try:
        # Priority: CLI argument > environment variable > default (5000)
        env_port = os.environ.get("PORT")
        port = args.port if args.port is not None else int(env_port or 5000)

        # Check if we're running in a deployment environment
        is_deployed = args.port is None and env_port is not None
        force_non_debug = os.environ.get("MATHUD_NON_DEBUG", "").lower() in ("1", "true", "yes")

        # Enable debug mode for local development
        debug_mode = not (is_deployed or force_non_debug)

        if is_deployed:
            # For deployment: run Flask directly without threading
            host = "0.0.0.0"  # Bind to all interfaces for deployment
            print(f"Starting Flask app on {host}:{port} (deployment mode)")
            app.run(host=host, port=port, debug=False)
        else:
            # For local development: use threading approach with debug capability
            host = "127.0.0.1"  # Localhost for development
            print(f"Starting Flask app on {host}:{port} (development mode, debug={debug_mode})")

            from threading import Thread

            server = Thread(
                target=app.run,
                kwargs={
                    "host": host,
                    "port": port,
                    "debug": debug_mode,
                    "use_reloader": False,  # Disable reloader in thread mode to avoid issues
                },
            )
            server.daemon = True  # Make the server thread a daemon so it exits when main thread exits
            server.start()

            from mathud_desktop import wait_for_server

            if not wait_for_server(f"http://{host}:{port}/"):
                print(f"Warning: the server did not respond on {host}:{port} yet")

            print(f"MatHud is running at http://{host}:{port}")
            print("Press Ctrl+C to stop the server")

            # Keep the main thread alive but responsive to keyboard interrupts
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
