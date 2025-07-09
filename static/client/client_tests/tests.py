import unittest

from .test_canvas import TestCanvas
from .test_cartesian import TestCartesian2Axis
from .test_circle import TestCircle
from .test_coordinate_mapper import TestCoordinateMapper
from .test_custom_drawable_names import TestCustomDrawableNames
from .test_drawable_dependency_manager import TestDrawableDependencyManager
from .test_drawable_name_generator import TestDrawableNameGenerator
from .test_drawables_container import TestDrawablesContainer
from .test_ellipse import TestEllipse
from .test_event_handler import TestCanvasEventHandlerTouch
from .test_throttle import TestThrottle
from .test_window_mocks import TestWindowMocks
from .test_expression_validator import TestExpressionValidator
from .test_function import TestFunction
from .test_functions_bounded_colored_area import TestFunctionsBoundedColoredArea
from .test_function_segment_bounded_colored_area import TestFunctionSegmentBoundedColoredArea
from .test_segments_bounded_colored_area import TestSegmentsBoundedColoredArea
from .test_point import TestPoint
from .test_position import TestPosition
from .test_rectangle import TestRectangle
from .test_segment import TestSegment
from .test_triangle import TestTriangle
from .test_vector import TestVector
from .test_angle import TestAngle
from .test_angle_manager import TestAngleManager
from .test_markdown_parser import TestMarkdownParser

# Import the utility classes
from .brython_io import BrythonTestStream
from .ai_result_formatter import AITestResult


class Tests:
    """Class encapsulating test functionality for client-side tests."""
    
    @classmethod
    def run_tests(cls):
        """Run all unit tests and return results in a format suitable for AI display."""
        test_runner = cls()
        try:
            # Create test suite and execute tests
            suite = test_runner._create_test_suite()
            result = test_runner._run_test_suite(suite)
            
            # Format results for AI display
            return test_runner._format_results_for_ai(result)
        except Exception as e:
            # If anything goes wrong, return a structured error
            return test_runner._create_error_result(str(e))
    
    def _create_test_suite(self):
        """Create a test suite containing all test cases."""
        suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        
        # Add all test cases to the suite
        test_cases = [
            TestMathFunctions,
            TestDrawableNameGenerator,
            TestPosition,
            TestPoint,
            TestSegment,
            TestVector,
            TestTriangle,
            TestRectangle,
            TestCircle,
            TestAngle,
            TestAngleManager,
            TestEllipse,
            TestFunction,
            TestFunctionsBoundedColoredArea,
            TestFunctionSegmentBoundedColoredArea,
            TestSegmentsBoundedColoredArea,
            TestCartesian2Axis,
            TestCanvas,
            TestCoordinateMapper,
            TestExpressionValidator,
            TestProcessFunctionCalls,
            TestCustomDrawableNames,
            TestThrottle,
            TestWindowMocks,
            TestCanvasEventHandlerTouch,
            TestDrawableDependencyManager,
            TestDrawablesContainer,
            TestMarkdownParser
        ]
        
        for test_case in test_cases:
            suite.addTest(loader.loadTestsFromTestCase(test_case))
            
        return suite
    
    def _run_test_suite(self, suite):
        """Run the test suite using our custom test runner and stream."""
        print("\n========================= TEST OUTPUT =========================")
        custom_stream = BrythonTestStream()
        
        # Use TextTestRunner with our custom stream and result class
        runner = unittest.TextTestRunner(
            stream=custom_stream,
            resultclass=AITestResult,
            verbosity=2
        )
        result = runner.run(suite)
        print("===============================================================\n")
        
        return result
    
    def _format_results_for_ai(self, result):
        """Format the test results for AI display, with concise error messages."""
        total_tests = result.testsRun
        failures = result.failures
        errors = result.errors
        
        # Format failures and errors
        failures_details = self._format_failures(failures)
        errors_details = self._format_errors(errors)
        
        # Return in our standard format
        return {
            'failures': failures_details,
            'errors': errors_details,
            'summary': {
                'tests': total_tests,
                'failures': len(failures),
                'errors': len(errors)
            },
            'output': None  # Don't send the full output to the AI
        }
    
    def _format_failures(self, failures):
        """Format test failures with concise error messages."""
        failures_details = []
        
        for test, error_msg in failures:
            error_message = self._extract_assertion_message(error_msg)
            failures_details.append({
                'test': str(test),
                'error': error_message
            })
            
        return failures_details
    
    def _format_errors(self, errors):
        """Format test errors with concise error messages."""
        errors_details = []
        
        for test, error_msg in errors:
            error_message = self._extract_error_message(error_msg)
            errors_details.append({
                'test': str(test),
                'error': error_message
            })
            
        return errors_details
    
    def _extract_assertion_message(self, error_msg):
        """Extract just the assertion message from a test failure."""
        error_str = str(error_msg)
        if "AssertionError:" in error_str:
            # For assertion errors, extract just the assertion message
            return "AssertionError: " + error_str.split("AssertionError:", 1)[1].strip()
        else:
            # For other errors, get the last non-empty line
            return self._extract_error_message(error_msg)
    
    def _extract_error_message(self, error_msg):
        """Extract the final error message from a traceback."""
        error_str = str(error_msg)
        lines = [line for line in error_str.split('\n') if line.strip()]
        return lines[-1] if lines else error_str
    
    def _create_error_result(self, error_message):
        """Create a standardized error result when test execution fails."""
        formatted_message = f"Error running tests: {error_message}"
        print(f"\nERROR: {formatted_message}")
        
        return {
            'failures': [],
            'errors': [{
                'test': 'unittest.run',
                'error': formatted_message
            }],
            'summary': {
                'tests': 0,
                'failures': 0,
                'errors': 1
            }
        }


# For backward compatibility, keep the run_tests function in the global scope
# but delegate to the class method
def run_tests():
    """Run all unit tests and return results in a format suitable for AI display."""
    return Tests.run_tests()