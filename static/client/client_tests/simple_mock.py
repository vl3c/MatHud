class SimpleMock:
    """A simple mock class for testing purposes that tracks calls and manages attributes."""
    
    def __init__(self, **kwargs):
        """Initialize the mock with attributes and return value.
        
        Args:
            **kwargs: Keyword arguments where 'return_value' is treated specially,
                     all other kwargs become attributes of the mock.
        """
        self._attributes = {}
        self._return_value = None
        self.calls = []
        for key, value in kwargs.items():
            if key == 'return_value':
                self._return_value = value
            else:
                self._attributes[key] = value

    def __getitem__(self, item):
        """Allow dictionary-style access to attributes."""
        if item in self._attributes:
            return self._attributes[item]
        else:
            raise KeyError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def __setitem__(self, key, value):
        """Allow dictionary-style setting of attributes."""
        self._attributes[key] = value

    def __contains__(self, item):
        """Support 'in' operator for checking attribute existence."""
        return item in self._attributes

    def __getattr__(self, attr):
        """Handle attribute access, including special handling for return_value."""
        if attr in self._attributes:
            return self._attributes[attr]
        elif attr == 'return_value':
            return self._return_value
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __call__(self, *args, **kwargs):
        """Make the mock callable, tracking calls and returning configured return value."""
        self.calls.append((args, kwargs))
        return self._return_value

    def assert_called_once_with(self, *args, **kwargs):
        """Assert that the mock was called exactly once with the specified arguments."""
        if len(self.calls) != 1:
            raise AssertionError(f'Expected one call, got {len(self.calls)}')
        call_args, call_kwargs = self.calls[0]
        if call_args != args or call_kwargs != kwargs:
            raise AssertionError(f'Expected call with ({args}, {kwargs}), got ({call_args}, {call_kwargs})')

    def assert_called_once(self):
        """Assert that the mock was called exactly once (regardless of arguments)."""
        if len(self.calls) != 1:
            raise AssertionError(f'Expected one call, got {len(self.calls)}')

    def assert_not_called(self):
        """Assert that the mock was never called."""
        if len(self.calls) > 0:
            raise AssertionError(f"Expected 0 calls, got {len(self.calls)}")

    def reset_mock(self):
        """Reset the mock's call history."""
        self.calls = [] 