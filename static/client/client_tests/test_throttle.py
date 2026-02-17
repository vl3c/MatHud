from __future__ import annotations

import unittest

from canvas_event_handler import throttle
from .simple_mock import SimpleMock
from browser import window as browser_window


class TestThrottle(unittest.TestCase):
    def setUp(self) -> None:
        # Create a mock for performance.now that we can update
        self.current_time = 1000

        # Create a now function that returns the current time
        def now() -> int:
            return self.current_time

        # Create the performance mock with our updatable now function
        self.mock_performance = SimpleMock(now=now)

        # Create the window mock with all its parts
        self.mock_window = SimpleMock(
            setTimeout=SimpleMock(return_value=123),  # Return a mock timer ID
            clearTimeout=SimpleMock(),
            performance=self.mock_performance,
        )

        # Save original window references
        self.original_performance = browser_window.performance
        self.original_setTimeout = browser_window.setTimeout
        self.original_clearTimeout = browser_window.clearTimeout

        # Replace the browser window objects
        browser_window.performance = self.mock_performance
        browser_window.setTimeout = self.mock_window.setTimeout
        browser_window.clearTimeout = self.mock_window.clearTimeout

    def set_time(self, new_time: int) -> None:
        """Helper to update the mock time."""
        self.current_time = new_time  # This will automatically update the now function

    def tearDown(self) -> None:
        # Restore original window objects
        browser_window.performance = self.original_performance
        browser_window.setTimeout = self.original_setTimeout
        browser_window.clearTimeout = self.original_clearTimeout

    def test_throttle_first_call_executes_immediately(self) -> None:
        """Test that the first call to a throttled function executes immediately."""
        mock_func = SimpleMock()
        throttled_func = throttle(100)(mock_func)

        # First call at t=1000ms should execute immediately
        self.set_time(1000)
        throttled_func(1, b=2)
        self.assertEqual(len(mock_func.calls), 1)
        mock_func.assert_called_once_with(1, b=2)

    def test_throttle_subsequent_calls_are_delayed(self) -> None:
        """Test that subsequent calls within the wait period are delayed."""
        mock_func = SimpleMock()
        throttled_func = throttle(100)(mock_func)

        # First call at t=1000ms
        self.set_time(1000)
        throttled_func(1)

        # Second call at t=1050ms should be scheduled
        self.set_time(1050)
        throttled_func(2)

        # Function should only have been called once directly
        self.assertEqual(len(mock_func.calls), 1)

        # setTimeout should have been called for the second invocation
        self.assertEqual(len(self.mock_window.setTimeout.calls), 1)

    def test_throttle_clears_previous_timeout(self) -> None:
        """Test that new calls clear previous pending timeouts."""
        mock_func = SimpleMock()
        throttled_func = throttle(100)(mock_func)

        # First call executes immediately at t=1000ms
        self.set_time(1000)
        throttled_func(1)
        self.assertEqual(len(self.mock_window.setTimeout.calls), 0)
        self.assertEqual(len(mock_func.calls), 1)

        # Second call at t=1020ms (20ms after first call) should schedule a timeout
        self.set_time(1020)
        throttled_func(2)
        self.assertEqual(len(self.mock_window.setTimeout.calls), 1)

        # Third call at t=1040ms should clear previous timeout and schedule new one
        self.set_time(1040)
        throttled_func(3)

        # Should have called clearTimeout once for the previous timeout
        self.assertEqual(len(self.mock_window.clearTimeout.calls), 1)

        # Should have called setTimeout twice (once for second call, once for third)
        self.assertEqual(len(self.mock_window.setTimeout.calls), 2)

    def test_throttle_respects_wait_time(self) -> None:
        """Test that throttle function respects the specified wait time."""
        mock_func = SimpleMock()
        throttled_func = throttle(100)(mock_func)

        # First call at t=1000ms
        self.set_time(1000)
        throttled_func(1)

        # Second call at t=1050ms (within wait time)
        self.set_time(1050)
        throttled_func(2)

        # Verify setTimeout was called with correct remaining time
        self.assertEqual(len(self.mock_window.setTimeout.calls), 1)
        call_args, call_kwargs = self.mock_window.setTimeout.calls[0]
        remaining_time = call_args[1]  # Second argument is the wait time

        # Should wait remaining 50ms (100ms - 50ms elapsed)
        self.assertEqual(remaining_time, 50)

    def test_throttle_handles_errors(self) -> None:
        """Test that throttle function handles errors gracefully."""

        def failing_func() -> None:
            raise Exception("Test error")

        throttled_func = throttle(100)(failing_func)

        # Call at t=1000ms
        self.set_time(1000)
        # Should not raise error, just print it
        throttled_func()  # Error should be caught and printed
