"""
MatHud Application Logging System

Session-based logging with timestamped entries for debugging and monitoring.
Creates daily log files and tracks user interactions, AI responses, and tool calls.

Dependencies:
    - logging: Python logging framework
    - os: File system operations
    - datetime: Timestamp generation
    - json: Message parsing and validation
"""

import logging
import os
from datetime import datetime
import json


class LogManager:
    """Manages application logging operations.
    
    Provides session-based logging with automatic file rotation by date.
    Logs user messages, AI responses, tool calls, and system events.
    """

    def __init__(self, logs_dir='./logs/'):
        """Initialize LogManager with specified logs directory.
        
        Args:
            logs_dir: Directory path for log files (default: './logs/')
        """
        self.logs_dir = logs_dir
        self._setup_logging()
    
    def _get_log_file_name(self):
        """Get the log file name based on current date.
        
        Returns:
            str: Date-based log filename (e.g., 'mathud_session_24_03_15.log')
        """
        return datetime.now().strftime('mathud_session_%y_%m_%d.log')
    
    def _setup_logging(self):
        """Initialize logging configuration.
        
        Creates logs directory if needed and configures daily log file rotation.
        """
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        logging.basicConfig(
            filename=os.path.join(self.logs_dir, self._get_log_file_name()),
            level=logging.INFO,
            format='%(asctime)s %(message)s'
        )
        self.log_new_session()
    
    def log_new_session(self):
        """Log a new session delimiter.
        
        Creates a visual separator in the log file for new application sessions.
        """
        session_delimiter = f"\n\n###### SESSION {datetime.now().strftime('%H:%M:%S')} ######\n"
        logging.info(session_delimiter)
    
    def log_user_message(self, user_message):
        """Log user message and its components.
        
        Parses and logs SVG state, canvas state, previous results, and user text.
        
        Args:
            user_message: JSON string containing user interaction data
        """
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
    
    def log_ai_response(self, ai_message):
        """Log AI response message.
        
        Args:
            ai_message: AI-generated response text
        """
        logging.info(f'### AI response: {ai_message}')
    
    def log_ai_tool_calls(self, ai_tool_calls):
        """Log AI tool calls.
        
        Args:
            ai_tool_calls: List of AI-requested function calls
        """
        logging.info(f'### AI tool calls: {ai_tool_calls}') 