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

from ai_interface import AIInterface
from browser import document
from canvas import Canvas
from canvas_event_handler import CanvasEventHandler


def main():
    """Initialize the MatHud client-side application.
    
    Creates the mathematical canvas, AI interface, and event handling system.
    Automatically called when the Brython runtime loads this module.
    """
    # Instantiate the canvas with current SVG viewport dimensions
    viewport = document['math-svg'].getBoundingClientRect()
    canvas = Canvas(viewport.width, viewport.height)

    # Instantiate the AIInterface class to handle interactions with the AI backend
    ai_interface = AIInterface(canvas)

    # Create event handler for user interactions (clicks, keyboard, etc.)
    CanvasEventHandler(canvas, ai_interface)


# Run the main function when the script loads
main()