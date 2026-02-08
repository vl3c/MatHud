"""
MatHud Application Logging System

Session-based logging with timestamped entries for debugging and monitoring.
Creates daily log files and tracks user interactions, AI responses, and tool calls.
Supports optional forwarding of logs to the browser console via streaming events.

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
from typing import Any, Dict, List, Optional, Sequence, Union

from static.tool_call_processor import ProcessedToolCall


JsonValue = Union[str, int, float, bool, None, Dict[str, "JsonValue"], List["JsonValue"]]
JsonObject = Dict[str, JsonValue]

# Log levels for browser forwarding (in order of severity)
LOG_LEVELS = ("debug", "info", "warning", "error")


def _get_log_level_index(level: str) -> int:
    """Get numeric index for log level comparison."""
    try:
        return LOG_LEVELS.index(level.lower())
    except ValueError:
        return LOG_LEVELS.index("warning")  # Default to warning


class LogManager:
    """Manages application logging operations.

    Provides session-based logging with automatic file rotation by date.
    Logs user messages, AI responses, tool calls, and system events.
    Supports optional forwarding of logs to the browser console.
    """

    def __init__(self, logs_dir: str = './logs/') -> None:
        """Initialize LogManager with specified logs directory.

        Args:
            logs_dir: Directory path for log files (default: './logs/')
        """
        self.logs_dir = logs_dir
        self._logger: logging.Logger = logging.getLogger("mathud")
        self._setup_logging()

        # Browser forwarding configuration
        self._forward_to_browser: bool = self._should_forward_to_browser()
        self._forward_level_index: int = self._get_forward_level_index()
        self._pending_logs: List[Dict[str, str]] = []

    def _should_forward_to_browser(self) -> bool:
        """Determine if logs should be forwarded to browser.

        Defaults to True in development (no PORT env var), False in production.
        Can be overridden with FORWARD_LOGS_TO_BROWSER env var.
        """
        forward_env = os.environ.get("FORWARD_LOGS_TO_BROWSER", "").lower()
        if forward_env in ("true", "1", "yes"):
            return True
        if forward_env in ("false", "0", "no"):
            return False
        # Default: forward in development, not in production
        return os.environ.get("PORT") is None

    def _get_forward_level_index(self) -> int:
        """Get minimum log level index for browser forwarding.

        Controlled by LOG_FORWARD_LEVEL env var (debug, info, warning, error).
        Defaults to 'warning'.
        """
        level = os.environ.get("LOG_FORWARD_LEVEL", "warning").lower()
        return _get_log_level_index(level)

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

        log_file_path = os.path.join(self.logs_dir, self._get_log_file_name())

        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(
                filename=log_file_path,
                level=logging.INFO,
                format='%(asctime)s %(levelname)s %(message)s'
            )

        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            handler = logging.FileHandler(log_file_path, encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            self._logger.addHandler(handler)
        self.log_new_session()

    def log_new_session(self) -> None:
        """Log a new session delimiter.

        Creates a visual separator in the log file for new application sessions.
        """
        session_delimiter = f"\n\n###### SESSION {datetime.now().strftime('%H:%M:%S')} ######\n"
        self._logger.info(session_delimiter)

    def log_user_message(self, user_message: str) -> None:
        """Log user message and its components.

        Parses and logs SVG state, canvas state, previous results, and user text.

        Args:
            user_message: JSON string containing user interaction data
        """
        try:
            user_message_json_raw: JsonValue = json.loads(user_message)
        except json.JSONDecodeError:
            self._logger.error("Failed to decode user message JSON.")
            return
        if not isinstance(user_message_json_raw, dict):
            self._logger.error("User message JSON is not an object.")
            return
        user_message_json: JsonObject = user_message_json_raw

        svg_state = user_message_json.get("svg_state")
        if isinstance(svg_state, dict):
            self._logger.info(f'### SVG state dimensions: {svg_state.get("dimensions")}')

        canvas_state = user_message_json.get("canvas_state")
        if canvas_state is not None:
            self._logger.info(f'### Canvas state: {canvas_state}')

        previous_results = user_message_json.get("previous_results")
        if previous_results is not None:
            self._logger.info(f'### Previously calculated results: {previous_results}')

        user_message_text = user_message_json.get("user_message")
        if user_message_text is not None:
            self._logger.info(f'### User message: {user_message_text}')

    def log_ai_response(self, ai_message: str) -> None:
        """Log AI response message.

        Args:
            ai_message: AI-generated response text
        """
        self._logger.info(f'### AI response: {ai_message}')

    def log_ai_tool_calls(self, ai_tool_calls: Sequence[ProcessedToolCall] | Sequence[Dict[str, Any]] | None) -> None:
        """Log AI tool calls.

        Args:
            ai_tool_calls: List of AI-requested function calls (ProcessedToolCall or dict)
        """
        if ai_tool_calls is not None:
            self._logger.info(f'### AI tool calls: {list(ai_tool_calls)}')

    # ========== Browser Forwarding Methods ==========

    def queue_for_browser(
        self,
        level: str,
        message: str,
        source: str = "",
    ) -> None:
        """Queue a log entry for forwarding to the browser.

        Only queues if browser forwarding is enabled and the log level
        meets the minimum threshold.

        Args:
            level: Log level ('debug', 'info', 'warning', 'error')
            message: The log message
            source: Optional source identifier (e.g., 'routes', 'api')
        """
        if not self._forward_to_browser:
            return

        level_lower = level.lower()
        level_index = _get_log_level_index(level_lower)
        if level_index < self._forward_level_index:
            return

        entry: Dict[str, str] = {
            "level": level_lower,
            "message": message,
        }
        if source:
            entry["source"] = source

        self._pending_logs.append(entry)

    def get_pending_logs(self) -> List[Dict[str, str]]:
        """Retrieve and clear all pending log entries for browser forwarding.

        Returns:
            List of log entry dicts with 'level', 'message', and optional 'source'.
        """
        logs = self._pending_logs
        self._pending_logs = []
        return logs

    def log_error(
        self,
        message: str,
        source: str = "",
        exc: Optional[BaseException] = None,
    ) -> None:
        """Log an error to both file and browser queue.

        Convenience method for consistent error logging across the application.

        Args:
            message: Error message to log
            source: Optional source identifier (e.g., 'routes', 'api')
            exc: Optional exception to include in log
        """
        if exc is not None:
            full_message = f"{message}: {exc}"
        else:
            full_message = message

        self._logger.error(full_message)
        self.queue_for_browser("error", full_message, source)
