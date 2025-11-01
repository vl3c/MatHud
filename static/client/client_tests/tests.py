from __future__ import annotations

import unittest
from typing import Any, Dict, List, Tuple, Type, cast

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
from .test_function_calling import TestProcessFunctionCalls
from .test_math_functions import TestMathFunctions
from .test_point import TestPoint
from .test_position import TestPosition
from .test_rectangle import TestRectangle
from .test_segment import TestSegment
from .test_triangle import TestTriangle
from .test_vector import TestVector
from .test_angle import TestAngle
from .test_angle_manager import TestAngleManager
from .test_markdown_parser import TestMarkdownParser
from .test_function_bounded_colored_area_integration import TestFunctionBoundedColoredAreaIntegration

from .brython_io import BrythonTestStream
from .ai_result_formatter import AITestResult


class Tests:
    """Class encapsulating test functionality for client-side tests."""

    @classmethod
    def run_tests(cls) -> Dict[str, Any]:
        """Run all unit tests and return results in a format suitable for AI display."""
        test_runner = cls()
        try:
            suite = test_runner._create_test_suite()
            result = test_runner._run_test_suite(suite)
            return test_runner._format_results_for_ai(result)
        except Exception as exc:
            return test_runner._create_error_result(str(exc))

    def _create_test_suite(self) -> unittest.TestSuite:
        """Create a test suite containing all test cases."""
        suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        test_cases: List[Type[unittest.TestCase]] = [
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
            TestMarkdownParser,
            TestFunctionBoundedColoredAreaIntegration,
        ]
        for test_case in test_cases:
            suite.addTest(loader.loadTestsFromTestCase(test_case))
        return suite

    def _run_test_suite(self, suite: unittest.TestSuite) -> AITestResult:
        """Run the test suite using our custom test runner and stream."""
        print("\n========================= TEST OUTPUT =========================")
        custom_stream = BrythonTestStream()
        runner = unittest.TextTestRunner(
            stream=cast(Any, custom_stream),
            resultclass=AITestResult,
            verbosity=2,
        )
        result = cast(AITestResult, runner.run(suite))
        print("===============================================================\n")
        return result

    def _format_results_for_ai(self, result: AITestResult) -> Dict[str, Any]:
        """Format the test results for AI display, with concise error messages."""
        total_tests = result.testsRun
        failures = result.failures
        errors = result.errors
        failures_details = self._format_failures(failures)
        errors_details = self._format_errors(errors)
        return {
            "failures": failures_details,
            "errors": errors_details,
            "summary": {
                "tests": total_tests,
                "failures": len(failures),
                "errors": len(errors),
            },
            "output": None,
        }

    def _format_failures(
        self,
        failures: List[Tuple[unittest.TestCase, str]],
    ) -> List[Dict[str, str]]:
        failures_details: List[Dict[str, str]] = []
        for test, error_msg in failures:
            error_message = self._extract_assertion_message(error_msg)
            failures_details.append({"test": str(test), "error": error_message})
        return failures_details

    def _format_errors(
        self,
        errors: List[Tuple[unittest.TestCase, str]],
    ) -> List[Dict[str, str]]:
        errors_details: List[Dict[str, str]] = []
        for test, error_msg in errors:
            error_message = self._extract_error_message(error_msg)
            errors_details.append({"test": str(test), "error": error_message})
        return errors_details

    def _extract_assertion_message(self, error_msg: str) -> str:
        error_str = str(error_msg)
        if "AssertionError:" in error_str:
            return "AssertionError: " + error_str.split("AssertionError:", 1)[1].strip()
        return self._extract_error_message(error_msg)

    def _extract_error_message(self, error_msg: str) -> str:
        error_str = str(error_msg)
        lines = [line for line in error_str.split("\n") if line.strip()]
        return lines[-1] if lines else error_str

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        formatted_message = f"Error running tests: {error_message}"
        print(f"\nERROR: {formatted_message}")
        return {
            "failures": [],
            "errors": [
                {
                    "test": "unittest.run",
                    "error": formatted_message,
                }
            ],
            "summary": {
                "tests": 0,
                "failures": 0,
                "errors": 1,
            },
        }


def run_tests() -> Dict[str, Any]:
    """Run all unit tests and return results in a format suitable for AI display."""
    return Tests.run_tests()