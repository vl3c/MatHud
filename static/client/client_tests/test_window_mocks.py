import unittest
from .simple_mock import SimpleMock
from browser import window as browser_window


class TestWindowMocks(unittest.TestCase):
    def setUp(self) -> None:
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

    def test_performance_now(self) -> None:
        """Test that window.performance.now() returns the correct time and updates properly."""
        # Initial time check
        self.assertEqual(browser_window.performance.now(), 1000)
        
        # Update time and check again
        self.current_time = 2000
        self.now_mock._return_value = self.current_time
        self.assertEqual(browser_window.performance.now(), 2000)
        
        # Verify the mock was actually called
        self.assertTrue(len(self.now_mock.calls) > 0)

    def test_set_timeout(self) -> None:
        """Test that setTimeout stores the callback and returns the expected timer ID."""
        callback = lambda: None
        wait_time = 100
        
        # Call setTimeout and verify return value
        timer_id = browser_window.setTimeout(callback, wait_time)
        self.assertEqual(timer_id, 123)  # Our mock always returns 123
        
        # Verify the call was recorded with correct arguments
        self.assertEqual(len(self.mock_window.setTimeout.calls), 1)
        call_args, call_kwargs = self.mock_window.setTimeout.calls[0]
        self.assertEqual(call_args[0], callback)
        self.assertEqual(call_args[1], wait_time)

    def test_clear_timeout(self) -> None:
        """Test that clearTimeout is called with the correct timer ID."""
        timer_id = 123
        
        # Call clearTimeout
        browser_window.clearTimeout(timer_id)
        
        # Verify the call was recorded with correct argument
        self.assertEqual(len(self.mock_window.clearTimeout.calls), 1)
        call_args, call_kwargs = self.mock_window.clearTimeout.calls[0]
        self.assertEqual(call_args[0], timer_id)

    def test_mock_chain(self) -> None:
        """Test that the entire mock chain works together."""
        callback = lambda: None
        
        # Set initial time
        self.current_time = 1000
        self.now_mock._return_value = self.current_time
        
        # Verify initial time
        self.assertEqual(browser_window.performance.now(), 1000)
        
        # Set a timeout
        timer_id = browser_window.setTimeout(callback, 100)
        
        # Update time
        self.current_time = 1050
        self.now_mock._return_value = self.current_time
        
        # Verify time updated
        self.assertEqual(browser_window.performance.now(), 1050)
        
        # Clear the timeout
        browser_window.clearTimeout(timer_id)
        
        # Verify all calls were recorded
        self.assertTrue(len(self.now_mock.calls) >= 2)  # At least two calls to now()
        self.assertEqual(len(self.mock_window.setTimeout.calls), 1)
        self.assertEqual(len(self.mock_window.clearTimeout.calls), 1) 