"""
IO utilities for Brython unittest framework.
Provides classes for capturing and displaying test output.
"""

class BrythonTestStream:
    """
    Custom stream class that adds the writeln method needed by Brython's unittest.
    This version preserves all output for console display.
    """
    
    def __init__(self):
        self.buffer = []
    
    def write(self, text):
        """Write to buffer and print to console."""
        self.buffer.append(text)
        print(text, end='')  # Print without additional newline
        return len(text)
    
    def writeln(self, text=''):
        """Write to buffer with newline and print to console."""
        full_text = text + '\n'
        self.buffer.append(full_text)
        print(full_text, end='')  # Print without additional newline
        return len(full_text)
    
    def flush(self):
        """Required method for compatibility."""
        pass
    
    def get_output(self):
        """Return all captured output as a string."""
        return ''.join(self.buffer) 