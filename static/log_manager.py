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

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Sequence


class LogManager:
    """Manages application logging operations.
    
    Provides session-based logging with automatic file rotation by date.
    Logs user messages, AI responses, tool calls, and system events.
    """

    def __init__(self, logs_dir: str = './logs/') -> None:
        """Initialize LogManager with specified logs directory.
        
        Args:
            logs_dir: Directory path for log files (default: './logs/')
        """
        self.logs_dir = logs_dir
        self._setup_logging()
    
    def _get_log_file_name(self) -> str:
        """Get the log file name based on current date.
        
        Returns:
            str: Date-based log filename (e.g., 'mathud_session_24_03_15.log')
        """
        return datetime.now().strftime('mathud_session_%y_%m_%d.log')
    
    def _setup_logging(self) -> None:
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
    
    def log_new_session(self) -> None:
        """Log a new session delimiter.
        
        Creates a visual separator in the log file for new application sessions.
        """
        session_delimiter = f"\n\n###### SESSION {datetime.now().strftime('%H:%M:%S')} ######\n"
        logging.info(session_delimiter)
    
    def log_user_message(self, user_message: str) -> None:
        """Log user message and its components.
        
        Parses and logs SVG state, canvas state, previous results, and user text.
        
        Args:
            user_message: JSON string containing user interaction data
        """
        try:
            user_message_json = json.loads(user_message)
            if not isinstance(user_message_json, dict):
                logging.error("User message JSON is not an object.")
                return
        except json.JSONDecodeError:
            logging.error("Failed to decode user message JSON.")
            return
        
        svg_state = user_message_json.get("svg_state")
        if isinstance(svg_state, dict):
            logging.info(f'### SVG state dimensions: {svg_state.get("dimensions")}')

        canvas_state = user_message_json.get("canvas_state")
        if canvas_state is not None:
            logging.info(f'### Canvas state: {canvas_state}')

        previous_results = user_message_json.get("previous_results")
        if previous_results is not None:
            logging.info(f'### Previously calculated results: {previous_results}')

        user_message_text = user_message_json.get("user_message")
        if user_message_text is not None:
            logging.info(f'### User message: {user_message_text}')
    
    def log_ai_response(self, ai_message: str) -> None:
        """Log AI response message.
        
        Args:
            ai_message: AI-generated response text
        """
        logging.info(f'### AI response: {ai_message}')
    
    def log_ai_tool_calls(self, ai_tool_calls: Sequence[Dict[str, Any]] | None) -> None:
        """Log AI tool calls.
        
        Args:
            ai_tool_calls: List of AI-requested function calls
        """
        if ai_tool_calls is not None:
            logging.info(f'### AI tool calls: {list(ai_tool_calls)}')