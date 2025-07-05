"""
Test result formatter module for AI-friendly output.
Provides custom classes for formatting unittest results in a way that's suitable for AI consumption.
"""

import unittest

class AITestResult(unittest.TextTestResult):
    """
    Custom TestResult that captures failures and errors in a format suitable for AI display.
    Provides simplified error messages without full tracebacks.
    """
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.failures_details = []
        self.errors_details = []
        self.output_stream = stream
        
    def addFailure(self, test, err):
        """Add a test failure with formatted error message."""
        super().addFailure(test, err)
        self.failures_details.append({
            'test': str(test),
            'error': self._format_error(err)
        })
        
    def addError(self, test, err):
        """Add a test error with formatted error message."""
        super().addError(test, err)
        self.errors_details.append({
            'test': str(test),
            'error': self._format_error(err)
        })
        
    def _format_error(self, err):
        """Format the error for display."""
        exctype, value, tb = err
        return f"{exctype.__name__}: {value}"
    
    def get_failures_and_errors(self):
        """Return all failures and errors in a format suitable for AI display."""
        return {
            'failures': self.failures_details,
            'errors': self.errors_details,
            'summary': {
                'tests': self.testsRun,
                'failures': len(self.failures),
                'errors': len(self.errors)
            },
            'output': self.output_stream.get_output() if hasattr(self.output_stream, 'get_output') else None
        } 