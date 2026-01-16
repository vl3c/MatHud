"""
MatHud Client-Side Application Entry Point

Initializes the browser-based Python environment for mathematical canvas operations.
Sets up the SVG canvas, AI communication interface, and event handling system.

Components Initialized:
    - Canvas: SVG-based mathematical visualization system
    - AIInterface: Communication bridge to backend AI services
    - CanvasEventHandler: User interaction and input processing

Dependencies:
    - Brython: Python-in-browser runtime environment
    - browser.document: DOM access for SVG manipulation
    - Custom modules: canvas, ai_interface, canvas_event_handler
"""

from __future__ import annotations

from typing import Optional

from ai_interface import AIInterface
from browser import document, window
from canvas import Canvas
from canvas_event_handler import CanvasEventHandler

# Module-level reference for programmatic test access
_ai_interface: Optional[AIInterface] = None
_test_results: Optional[str] = None
_tests_running: bool = False


def start_tests() -> str:
    """Start running tests asynchronously with UI feedback.

    Call this to begin test execution. Poll getMatHudTestResults() to check completion.
    Also updates the UI like clicking the Run Tests button would.
    Exposed on window as: window.startMatHudTests()

    Returns:
        JSON string with status: "started", "already_running", or "error"
    """
    global _test_results, _tests_running

    if _tests_running:
        return window.JSON.stringify({"status": "already_running"})

    if _ai_interface is None:
        return window.JSON.stringify({"status": "error", "error": "AIInterface not initialized"})

    _tests_running = True
    _test_results = None

    # Disable button and show user message (like the button does)
    try:
        if "run-tests-button" in document:
            document["run-tests-button"].disabled = True
        _ai_interface._print_user_message_in_chat("Run tests (programmatic)")
    except Exception:
        pass  # UI update is optional

    def execute_tests() -> None:
        global _test_results, _tests_running
        try:
            results = _ai_interface.run_tests()
            _test_results = window.JSON.stringify(results)

            # Format and display results in chat (like the button does)
            summary = (
                f"### Test Results\n\n"
                f"- **Tests Run:** {results.get('tests_run', 0)}\n"
                f"- **Failures:** {results.get('failures', 0)}\n"
                f"- **Errors:** {results.get('errors', 0)}\n"
            )

            if results.get('failing_tests'):
                summary += "\n#### Failures:\n"
                for fail in results['failing_tests']:
                    summary += f"- **{fail['test']}**: {fail['error']}\n"

            if results.get('error_tests'):
                summary += "\n#### Errors:\n"
                for err in results['error_tests']:
                    summary += f"- **{err['test']}**: {err['error']}\n"

            _ai_interface._print_ai_message_in_chat(summary)

        except Exception as e:
            _test_results = window.JSON.stringify({
                "tests_run": 0,
                "failures": 0,
                "errors": 1,
                "failing_tests": [],
                "error_tests": [{"test": "Test Runner", "error": str(e)}]
            })
            _ai_interface._print_ai_message_in_chat(f"Error running tests: {str(e)}")
        finally:
            _tests_running = False
            # Re-enable button
            try:
                if "run-tests-button" in document:
                    document["run-tests-button"].disabled = False
            except Exception:
                pass

    # Schedule test execution to run asynchronously
    window.setTimeout(execute_tests, 10)
    return window.JSON.stringify({"status": "started"})


def get_test_results() -> str:
    """Get test results if available.

    Exposed on window as: window.getMatHudTestResults()

    Returns:
        JSON string with test results, or status if not ready:
        - {"status": "running"} if tests are still running
        - {"status": "no_results"} if no tests have been run
        - Full test results JSON if complete
    """
    if _tests_running:
        return window.JSON.stringify({"status": "running"})

    if _test_results is None:
        return window.JSON.stringify({"status": "no_results"})

    return _test_results


def main() -> None:
    """Initialize the MatHud client-side application.

    Creates the mathematical canvas, AI interface, and event handling system.
    Automatically called when the Brython runtime loads this module.
    """
    global _ai_interface

    # Instantiate the canvas with current SVG viewport dimensions
    viewport = document['math-svg'].getBoundingClientRect()
    canvas = Canvas(viewport.width, viewport.height)

    # Instantiate the AIInterface class to handle interactions with the AI backend
    _ai_interface = AIInterface(canvas)

    # Create event handler for user interactions (clicks, keyboard, etc.)
    CanvasEventHandler(canvas, _ai_interface)

    # Expose test runner globally for programmatic access
    # Usage: window.startMatHudTests() to begin, window.getMatHudTestResults() to poll
    window.startMatHudTests = start_tests
    window.getMatHudTestResults = get_test_results


# Run the main function when the script loads
main()