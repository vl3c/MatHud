# MatHud - Mathematics Heads-Up Display

MathHud is an interactive mathematical visualization tool that combines a drawing canvas with an AI assistant to help understand and solve real-world mathematical problems. It serves as a heads-up display system for mathematical analysis, allowing users to visualize, analyze, and solve problems in real-time.

![MatHud - Interactive Mathematics Visualization Tool](MatHud%20-%20Screenshot%202025-06-28.png)

## Features

- Interactive SVG-based drawing canvas
- AI-powered mathematical assistant
- Real-time geometric visualization and analysis
- Support for mathematical expressions and calculations
- Visual problem-solving with multimodal AI
- Real-world problem analysis and modeling
- Workspace management for saving and loading states
- Comprehensive chat interface with rich markdown support and LaTeX mathematical notation rendering

## Authentication (Deployed Environments)

When MatHud is deployed to production environments, it includes a simple access code authentication system to protect the AI chat functionality.

### Configuration

1. Create a `.env` file in the project root with the following variables:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Authentication Configuration (for deployed environments)
# Can be alphanumeric (letters, numbers, special characters)
AUTH_PIN=your_chosen_access_code_here

# Optional: Custom secret key for session management
SECRET_KEY=your_secret_key_here_for_production
```

2. The authentication system automatically activates when the `PORT` environment variable is set (indicating deployment mode).

### Behavior

- **Development Mode** (no `PORT` env var): No authentication required, direct access to the application
- **Deployed Mode** (`PORT` env var set): Access code authentication required before accessing the mathematical canvas

### Security Features

- Session-based authentication with secure cookie settings
- Access code protection for all application routes and API endpoints
- **Rate limiting**: Only 1 login attempt per IP address every 5 seconds
- **Timing attack protection**: Constant-time string comparison for access codes
- Automatic logout functionality in the deployed interface
- Environment-based configuration management

## Architecture Overview

MatHud is a client-server web application with the following key architectural components:

-   **Frontend**: The user interface is built primarily with HTML and utilizes Brython (a Python implementation for browsers). This allows for Python-based scripting on the client-side to manage the interactive SVG drawing canvas, handle user input, and communicate with the backend server.
-   **Backend**: A Python Flask server forms the backend. It processes requests from the client, manages application logic (including interactions with the OpenAI API), and handles the persistence of workspace data (saving and loading mathematical sessions).
-   **AI Integration**: The application integrates with the OpenAI API to provide AI-powered mathematical assistance. This includes natural language understanding, problem-solving capabilities, and the ability to execute defined functions (tools) based on user queries.
-   **Vision Processing (Server-Side)**: For tasks requiring visual understanding of the canvas, the backend employs Selenium WebDriver. It launches a headless browser instance to render the SVG content sent from the client, captures a PNG image of the canvas, and then forwards this image to a vision-capable AI model via the OpenAI API.
-   **Core Functionality Modules**: The application is structured with modules for:
    -   Mathematical calculations and expression evaluation. Symbolic operations (calculus, equation solving) are primarily handled client-side via Brython interfacing with JavaScript libraries like `nerdamer.js` (for symbolic algebra) and `math.js` (for expression evaluation).
    -   Management of geometric shapes and drawings on the SVG canvas.
    -   Client-side and server-side workspace management for saving and loading user sessions.

## Vision Mechanism

The application includes a visual understanding system that allows the AI to analyze both drawn elements and real-world mathematical scenarios. This enhances problem-solving by providing visual context to the AI.

1.  **How it Works (High-Level)**:
    *   When visual context is needed, the user's current canvas drawing (as SVG data) is sent from the browser to the application server.
    *   The server uses a headless browser (via Selenium WebDriver) to render this SVG data and capture a pixel image (PNG) of the canvas.
    *   This captured image is then sent to a vision-capable AI model along with the textual part of the user's query and relevant mathematical context.
    *   The AI processes both the visual information from the image and the textual data to provide a more comprehensive analysis or solution.

2.  **Key Components Involved**:
    *   **Client-Side (Browser)**: Captures the SVG state of the canvas and sends it to the server when vision is enabled.
    *   **Server-Side (Python/Flask)**: 
        *   Receives the SVG state.
        *   Uses `WebDriverManager` (a custom module employing Selenium) to control a headless Firefox instance, load the SVG, and take a screenshot.
        *   The `OpenAIChatCompletionsAPI` module then includes this image in the request to the AI.

3.  **Benefits**:
    *   Allows the AI to "see" and interpret user-drawn diagrams and mathematical constructions.
    *   Practical problem-solving in real-world contexts.
    *   Dynamic visualization of mathematical concepts.
    *   Enhanced pattern recognition and analysis.
    *   Bridge between theoretical and applied mathematics.

## Installation

To initialize and install Python dependencies for this repository after it's cloned, you can follow these steps:

1. First, make sure you have Python installed on your system. You can check this by running the following command in your terminal:

```sh
python --version
```

2. Next, you need to create a virtual environment. This can be done using the "venv" module that comes with Python. Run the following command in your terminal:

```sh
python -m venv venv
```

3. After the virtual environment is created, you need to activate it. On Unix or MacOS, run:

```sh
source venv/bin/activate
```

On Windows, run:

```sh
.\venv\Scripts\activate
```

4. Once the virtual environment is activated, you can install the required dependencies using pip. The dependencies are listed in the "requirements.txt" file. Run the following command to install them:

```sh
pip install -r requirements.txt
```
5. Now all the dependencies should be installed. Please export your OpenAI API key to the OPENAI_API_KEY system variable or place the key in a file called ".env" one level above the main folder:

```sh
..\\.env should contain the line OPENAI_API_KEY=...
```

6. You can start the application using the VSCode runner or by running the following command in the main folder:
```sh
python app.py
```

## Development

For local development, the application runs without authentication. The access code login only applies to deployed environments.

## Usage

1. **Start the Application**: After installation, start the server:
   ```sh
   python app.py
   ```

2. **Access the Interface**: Open your web browser and navigate to `http://127.0.0.1:5000`

3. **Using the Canvas**:
   - **Drawing**: Double-click on the canvas to capture coordinates
   - **Zooming**: Use mouse wheel to zoom in/out
   - **Panning**: Click and drag to move around the canvas

4. **AI Interaction**:
   - Type mathematical questions or requests in the chat input
   - The AI can create geometric shapes, solve equations, and perform calculations
   - Supports markdown formatting and LaTeX mathematical expressions in responses
   - Enable "Vision" mode for the AI to analyze your drawings (works with GPT-4o, GPT-4o Mini, GPT-4.1, GPT-4.1 Mini, GPT-4.1 Nano)

5. **Available AI Commands**:
   - `create point at (2, 3)` - Create geometric shapes
   - `calculate the distance between points A and B` - Mathematical calculations
   - `solve x^2 + 2x - 3 = 0` - Equation solving
   - `save workspace as "my_project"` - Save your work
   - `load workspace "my_project"` - Load previous work
   - `run tests` - Execute client-side tests

6. **Workspace Management**:
   - Save your work with custom names
   - Load previous sessions
   - List all saved workspaces
   - Delete unwanted workspaces

## Dependencies

- Python 3.x
- Flask
- OpenAI API client
- Python-dotenv (for environment variable management)
- Selenium
- Geckodriver-autoinstaller (automatic Firefox WebDriver setup)
- Firefox WebDriver
- OpenAI API (optionally vision capable)

## Configuration

1. Set up your OpenAI API key using one of these methods:

   a. Environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```

   b. Create a `.env` file in the parent directory:
   ```bash
   # ../.env
   OPENAI_API_KEY=your-api-key
   ```

2. Firefox WebDriver setup:

   The application includes `geckodriver-autoinstaller` in its dependencies, which automatically handles Firefox WebDriver installation. If you prefer manual installation:

   On Ubuntu/Debian:
   ```bash
   sudo apt install firefox-geckodriver
   ```

   On MacOS:
   ```bash
   brew install geckodriver
   ```

   On Windows:
   ```bash
   # Manual download from https://github.com/mozilla/geckodriver/releases
   # Or rely on the automatic installer (recommended)
   ```

## Diagram Generation

MatHud includes a comprehensive diagram generation system for visualizing the codebase architecture and dependencies. This system automatically creates UML class diagrams, package diagrams, dependency graphs, and Flask route documentation.

### Quick Start

```bash
# Generate all diagrams
python generate_diagrams_launcher.py
```

### Generated Diagrams

The system creates multiple visualization types:
- **Class Diagrams** - UML class relationships and inheritance
- **Flask Routes** - Professional API endpoint documentation 
- **Function Analysis** - Function-level dependency tracking
- **Module Diagrams** - Individual component visualizations
- **Dependency Graphs** - Import relationship analysis

### Output Formats

- **PNG** - High-quality raster images for documentation
- **SVG** - Vector graphics for presentations and scaling

All diagrams are automatically generated in both formats and saved to:
- `diagrams/generated_png/` - PNG format diagrams
- `diagrams/generated_svg/` - SVG format diagrams

### Documentation

For detailed setup and usage instructions, see:
- [`diagrams/README.md`](diagrams/README.md) - Complete setup guide
- [`diagrams/WORKFLOW_SUMMARY.md`](diagrams/WORKFLOW_SUMMARY.md) - Quick reference

## Testing

MatHud has both server-side and client-side tests to ensure functionality works as expected.

### Server-Side Tests

We've created a convenient script to run server tests. Simply use:

```sh
python run_server_tests.py
```

This script provides several options:

- Run all server tests:
  ```sh
  python run_server_tests.py
  ```

- Run tests in a specific file:
  ```sh
  python run_server_tests.py test_workspace_management.py
  ```
  or just:
  ```sh
  python run_server_tests.py test_workspace_management
  ```

- Run tests matching a keyword:
  ```sh
  python run_server_tests.py -k list
  ```

- Show help information:
  ```sh
  python run_server_tests.py --help
  ```

The script automatically uses the Python environment from your virtual environment and doesn't require remembering complex pytest commands.

### Client-Side Tests

Client-side tests run in the browser using Brython's unittest implementation. These tests verify the mathematical functions, drawing capabilities, and other client-side features.

You can run client-side tests directly from the AI interface by asking the AI to run tests. The AI has access to a special function that executes all client-side tests and returns the results.

Example AI prompt:
```
Can you run the client-side tests and tell me if there are any failures?
```

This integration allows for convenient testing of the browser-based components without leaving the application.

