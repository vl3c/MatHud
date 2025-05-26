# MatHud Brython 3.11.3 Runtime

Python-in-browser runtime for MatHud's client-side mathematical operations.
Brython enables Python execution in the browser for canvas manipulation, 
mathematical calculations, and AI function processing.

## Key Components:
- brython.js: Core Python interpreter for browser
- brython_stdlib.js: Standard Python library implementations
- Lib/math.js: Enhanced mathematical operations library  
- Lib/nerdamer.js: Symbolic mathematics computation engine
- Lib/site-packages/: MatHud-specific Python modules

## Usage in MatHud:
The main entry point is Lib/site-packages/main.py which initializes
the canvas system, AI interface, and mathematical computing environment.

## Original Brython Demo Instructions:
To run the demo, you can open the file demo.html from the browser "File/Open..." menu.

Another option is to start the built-in Python HTTP server by

    python -m http.server

The default port is 8000. To specify another port:

    python -m http.server 8080

Then load http://localhost:<port>/demo.html in the browser address bar.

For more information please visit http://brython.info.