"""Runs SymPy computations (static/sympy_cas.py) in a worker process with a time limit.

SymPy has no way to stop a computation, and some (simplify of a huge power, an integral
it keeps rewriting) run for minutes, so each request goes to a long-lived worker process
(``python -m static.sympy_cas``) that is killed and started again when a request takes
longer than its time limit. Requests are served one at a time.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
from typing import IO, Any, Dict, Optional

_logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Seconds a computation may run before the worker is killed.
DEFAULT_TIME_LIMIT_S = 6.0
# Seconds the worker may take to start (it imports SymPy before it reads requests).
STARTUP_TIME_LIMIT_S = 30.0


def sympy_installed() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("sympy") is not None
    except (ImportError, ValueError):
        return False


class SympyEngine:
    """One worker process, started on the first request and again after a timeout or crash."""

    def __init__(self, time_limit_s: float = DEFAULT_TIME_LIMIT_S) -> None:
        self.time_limit_s = time_limit_s
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen[str]] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def compute(self, operation: str, args: Dict[str, Optional[str]]) -> Dict[str, Any]:
        """The worker's response: {"result": ...}, {"error": message} or {"timeout": True}."""
        with self._lock:
            if not self._ensure_started():
                return {"unavailable": True}
            assert self._process is not None and self._process.stdin is not None
            try:
                self._process.stdin.write(json.dumps({"operation": operation, "args": args}) + "\n")
                self._process.stdin.flush()
            except (OSError, ValueError):
                self._stop()
                return {"unavailable": True}
            line = self._next_line(self.time_limit_s)
            if line is None:
                timed_out = self._process.poll() is None
                self._stop()
                return {"timeout": True} if timed_out else {"unavailable": True}
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                self._stop()
                return {"unavailable": True}
            return response if isinstance(response, dict) else {"unavailable": True}

    def close(self) -> None:
        with self._lock:
            self._stop()

    def _ensure_started(self) -> bool:
        if self._process is not None and self._process.poll() is None:
            return True
        self._stop()
        if not sympy_installed():
            return False
        self._lines = queue.Queue()
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-m", "static.sympy_cas"],
                cwd=PROJECT_ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
            )
        except OSError as error:
            _logger.warning("Could not start the SymPy worker: %s", error)
            self._process = None
            return False
        assert self._process.stdout is not None
        threading.Thread(target=self._read_lines, args=(self._process.stdout, self._lines), daemon=True).start()
        ready = self._next_line(STARTUP_TIME_LIMIT_S)
        if ready is None or '"ready"' not in ready:
            _logger.warning("The SymPy worker did not start.")
            self._stop()
            return False
        return True

    @staticmethod
    def _read_lines(stream: IO[str], lines: "queue.Queue[Optional[str]]") -> None:
        for line in stream:
            lines.put(line)
        lines.put(None)  # the worker exited

    def _next_line(self, timeout_s: float) -> Optional[str]:
        try:
            return self._lines.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        for stream in (process.stdin, process.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass


_engine: Optional[SympyEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> SympyEngine:
    """The app's shared engine."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = SympyEngine()
        return _engine
