"""Helpers that format unittest results for AI-friendly consumption."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Dict, List, Tuple, Type, Union

import unittest

ErrorTuple = Union[
    Tuple[Type[BaseException], BaseException, TracebackType],
    Tuple[None, None, None],
]


class AITestResult(unittest.TextTestResult):
    """Collect failures/errors with simplified messages for display."""

    def __init__(self, stream: Any, descriptions: bool, verbosity: int) -> None:
        super().__init__(stream, descriptions, verbosity)
        self.failures_details: List[Dict[str, str]] = []
        self.errors_details: List[Dict[str, str]] = []
        self.output_stream: Any = stream

    def addFailure(self, test: unittest.TestCase, err: ErrorTuple) -> None:
        super().addFailure(test, err)
        self.failures_details.append(
            {
                "test": str(test),
                "error": self._format_error(err),
            }
        )

    def addError(self, test: unittest.TestCase, err: ErrorTuple) -> None:
        super().addError(test, err)
        self.errors_details.append(
            {
                "test": str(test),
                "error": self._format_error(err),
            }
        )

    def _format_error(self, err: ErrorTuple) -> str:
        """Return a concise error string for the captured exception tuple."""
        exctype, value, _ = err
        if exctype is None or value is None:
            return "Unknown error"
        return f"{exctype.__name__}: {value}"

    def get_failures_and_errors(self) -> Dict[str, Any]:
        """Return error details together with summary metadata."""
        output: Any = (
            self.output_stream.get_output()
            if hasattr(self.output_stream, "get_output")
            else None
        )
        return {
            "failures": self.failures_details,
            "errors": self.errors_details,
            "summary": {
                "tests": self.testsRun,
                "failures": len(self.failures),
                "errors": len(self.errors),
            },
            "output": output,
        }
