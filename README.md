# MathUD - Mathematical Understanding and Drawing

MathUD is an interactive mathematical visualization tool that combines a drawing canvas with an AI assistant to help understand and solve mathematical problems.

## Features

- Interactive SVG-based drawing canvas
- AI-powered mathematical assistant
- Real-time geometric visualization
- Support for mathematical expressions and calculations
- Visual problem-solving with multimodal AI

## Vision Mechanism

The application includes a visual understanding system that allows the AI to "see" and interpret the mathematical drawings:

1. **Visual Context**: The AI assistant can see the actual canvas state through screenshots, enabling it to:
   - Understand geometric relationships visually
   - Interpret mathematical drawings directly
   - Provide more accurate and contextual responses

2. **Implementation**: 
   - Uses a headless Firefox WebDriver to capture canvas state
   - Synchronizes the main browser view with the AI's view
   - Sends canvas screenshots to multimodal AI for visual analysis

3. **Benefits**:
   - More intuitive problem-solving
   - Better understanding of geometric relationships
   - Enhanced ability to spot patterns and symmetries
   - Visual verification of mathematical properties

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
..\.env should contain the line OPENAI_API_KEY=...
```

7. You can start the application using the VSCode runner or by running the following command in the main folder:
```sh
python app.py
```

## Usage

[existing usage instructions...]

## Dependencies

- Python 3.x
- Flask
- Selenium
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

2. Ensure Firefox WebDriver is installed for the vision mechanism:

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
   pip install geckodriver-autoinstaller
   # Or download geckodriver from https://github.com/mozilla/geckodriver/releases
   ```
