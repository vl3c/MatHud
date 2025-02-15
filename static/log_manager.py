import logging
import os
from datetime import datetime
import json


class LogManager:
    """Manages application logging operations."""

    def __init__(self, logs_dir='./logs/'):
        self.logs_dir = logs_dir
        self._setup_logging()
    
    def _get_log_file_name(self):
        """Get the log file name based on current date."""
        return datetime.now().strftime('mathud_session_%y_%m_%d.log')
    
    def _setup_logging(self):
        """Initialize logging configuration."""
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        logging.basicConfig(
            filename=os.path.join(self.logs_dir, self._get_log_file_name()),
            level=logging.INFO,
            format='%(asctime)s %(message)s'
        )
        self.log_new_session()
    
    def log_new_session(self):
        """Log a new session delimiter."""
        session_delimiter = f"\n\n###### SESSION {datetime.now().strftime('%H:%M:%S')} ######\n"
        logging.info(session_delimiter)
    
    def log_user_message(self, user_message):
        """Log user message and its components."""
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
        """Log AI response message."""
        logging.info(f'### AI response: {ai_message}')
    
    def log_ai_tool_calls(self, ai_tool_calls):
        """Log AI tool calls."""
        logging.info(f'### AI tool calls: {ai_tool_calls}') 