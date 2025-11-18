# Guide to removing the temporary "Run Tests" Button

The "Run Tests" button was added to the UI to allow developers to quickly trigger client-side tests without consuming AI tokens. It is intended as a temporary development tool.

## Steps to Remove

1.  **Remove the HTML element**
    *   File: `templates/index.html`
    *   Search for `<button id="run-tests-button" ...>` inside the `.chat-input-container` div.
    *   Delete the button element.

2.  **Remove the CSS styling**
    *   File: `static/style.css`
    *   Search for `#run-tests-button` selectors.
    *   Delete the associated style blocks.
    *   Also check for combined selectors like `#send-button, #run-tests-button` and remove the `#run-tests-button` part.

3.  **Remove the Event Binding**
    *   File: `static/client/canvas_event_handler.py`
    *   Search for `document["run-tests-button"].bind`.
    *   Delete the line that binds the click event.

4.  **Remove the Handler Logic (Optional)**
    *   File: `static/client/ai_interface.py`
    *   The method `run_tests_action` can be removed if it's no longer needed, though keeping it won't harm anything if it's not called.
    *   The `run_tests` method is used by the AI tool call system and **MUST NOT** be removed.

## Purpose

This button directly calls the `run_tests` method in `AIInterface`, bypassing the AI model to save tokens and reduce latency during development iterations.

