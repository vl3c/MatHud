import unittest
from canvas_event_handler import throttle
from .simple_mock import SimpleMock
from browser import window as browser_window


class TestWindowMocks(unittest.TestCase):
    def setUp(self):
        # Create a mock for performance.now that we can update
        self.current_time = 1000
        self.now_mock = SimpleMock(return_value=self.current_time)
        
        # Create the performance mock with our updatable now function
        self.mock_performance = SimpleMock(now=self.now_mock)
        
        # Create the window mock with all its parts
        self.mock_window = SimpleMock(
            setTimeout=SimpleMock(return_value=123),  # Return a mock timer ID
            clearTimeout=SimpleMock(),
            performance=self.mock_performance
        )
        
        # Save original window references
        self.original_performance = browser_window.performance
        self.original_setTimeout = browser_window.setTimeout
        self.original_clearTimeout = browser_window.clearTimeout
        
        # Replace the browser window objects
        browser_window.performance = self.mock_performance
        browser_window.setTimeout = self.mock_window.setTimeout
        browser_window.clearTimeout = self.mock_window.clearTimeout

    def tearDown(self):
        # Restore original window objects
        browser_window.performance = self.original_performance
        browser_window.setTimeout = self.original_setTimeout
        browser_window.clearTimeout = self.original_clearTimeout

    def test_performance_now(self):
        """Test that window.performance.now() returns the correct time and updates properly."""
        # Initial time check
        self.assertEqual(browser_window.performance.now(), 1000)
        
        # Update time and check again
        self.current_time = 2000
        self.now_mock.return_value = self.current_time
        self.assertEqual(browser_window.performance.now(), 2000)
        
        # Verify the mock was actually called
        self.assertTrue(len(self.now_mock.calls) > 0)

    def test_set_timeout(self):
        """Test that setTimeout stores the callback and returns the expected timer ID."""
        callback = lambda: None
        wait_time = 100
        
        # Call setTimeout and verify return value
        timer_id = browser_window.setTimeout(callback, wait_time)
        self.assertEqual(timer_id, 123)  # Our mock always returns 123
        
        # Verify the call was recorded with correct arguments
        self.assertEqual(len(self.mock_window.setTimeout.calls), 1)
        call_args = self.mock_window.setTimeout.calls[0][0]
        self.assertEqual(call_args[0], callback)
        self.assertEqual(call_args[1], wait_time)

    def test_clear_timeout(self):
        """Test that clearTimeout is called with the correct timer ID."""
        timer_id = 123
        
        # Call clearTimeout
        browser_window.clearTimeout(timer_id)
        
        # Verify the call was recorded with correct argument
        self.assertEqual(len(self.mock_window.clearTimeout.calls), 1)
        call_args = self.mock_window.clearTimeout.calls[0][0]
        self.assertEqual(call_args[0], timer_id)

    def test_mock_chain(self):
        """Test that the entire mock chain works together."""
        callback = lambda: None
        
        # Set initial time
        self.current_time = 1000
        self.now_mock.return_value = self.current_time
        
        # Verify initial time
        self.assertEqual(browser_window.performance.now(), 1000)
        
        # Set a timeout
        timer_id = browser_window.setTimeout(callback, 100)
        
        # Update time
        self.current_time = 1050
        self.now_mock.return_value = self.current_time
        
        # Verify time updated
        self.assertEqual(browser_window.performance.now(), 1050)
        
        # Clear the timeout
        browser_window.clearTimeout(timer_id)
        
        # Verify all calls were recorded
        self.assertTrue(len(self.now_mock.calls) >= 2)  # At least two calls to now()
        self.assertEqual(len(self.mock_window.setTimeout.calls), 1)
        self.assertEqual(len(self.mock_window.clearTimeout.calls), 1)


class TestThrottle(unittest.TestCase):
    def setUp(self):
        # Create a mock for performance.now that we can update
        self.current_time = 1000
        
        # Create a now function that returns the current time
        def now():
            return self.current_time
            
        # Create the performance mock with our updatable now function
        self.mock_performance = SimpleMock(now=now)
        
        # Create the window mock with all its parts
        self.mock_window = SimpleMock(
            setTimeout=SimpleMock(return_value=123),  # Return a mock timer ID
            clearTimeout=SimpleMock(),
            performance=self.mock_performance
        )
        
        # Save original window references
        self.original_performance = browser_window.performance
        self.original_setTimeout = browser_window.setTimeout
        self.original_clearTimeout = browser_window.clearTimeout
        
        # Replace the browser window objects
        browser_window.performance = self.mock_performance
        browser_window.setTimeout = self.mock_window.setTimeout
        browser_window.clearTimeout = self.mock_window.clearTimeout

    def set_time(self, new_time):
        """Helper to update the mock time."""
        self.current_time = new_time  # This will automatically update the now function

    def tearDown(self):
        # Restore original window objects
        browser_window.performance = self.original_performance
        browser_window.setTimeout = self.original_setTimeout
        browser_window.clearTimeout = self.original_clearTimeout

    def test_throttle_first_call_executes_immediately(self):
        """Test that the first call to a throttled function executes immediately."""
        mock_func = SimpleMock()
        throttled_func = throttle(100)(mock_func)
        
        # First call at t=1000ms should execute immediately
        self.set_time(1000)
        throttled_func(1, b=2)
        self.assertEqual(len(mock_func.calls), 1)
        mock_func.assert_called_once_with(1, b=2)

    def test_throttle_subsequent_calls_are_delayed(self):
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

    def test_throttle_clears_previous_timeout(self):
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
        
        # Verify clearTimeout was called with the previous timer ID
        self.assertEqual(len(self.mock_window.clearTimeout.calls), 1)
        self.assertEqual(len(self.mock_window.setTimeout.calls), 2)
        
        # Verify the timer ID was cleared
        timer_id = 123  # From our mock setup
        self.assertEqual(self.mock_window.clearTimeout.calls[0][0], (timer_id,))

    def test_throttle_respects_wait_time(self):
        """Test that throttle respects the wait time between calls."""
        mock_func = SimpleMock()
        wait_ms = 100
        throttled_func = throttle(wait_ms)(mock_func)
        
        # First call at t=1000ms
        self.set_time(1000)
        throttled_func(1)
        
        # Second call at t=1040ms (40ms after first call)
        self.set_time(1040)
        throttled_func(2)
        
        # Verify setTimeout was called with correct remaining time
        self.assertEqual(len(self.mock_window.setTimeout.calls), 1)
        call_args = self.mock_window.setTimeout.calls[0][0]
        remaining_time = call_args[1]  # Second argument is the wait time
        
        # Should wait remaining 60ms (100ms - 40ms elapsed)
        self.assertEqual(remaining_time, 60)

    def test_throttle_handles_errors(self):
        """Test that throttle properly handles errors in the throttled function."""
        def failing_func():
            raise ValueError("Test error")
        
        throttled_func = throttle(100)(failing_func)
        
        # Call at t=1000ms
        self.set_time(1000)
        # Should not raise error, just print it
        throttled_func()  # Error should be caught and printed 