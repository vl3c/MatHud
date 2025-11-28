from __future__ import annotations

import unittest
import traceback
from typing import Any, Dict, List, Tuple, Type, cast

from .test_canvas import TestCanvas
from .test_cartesian import TestCartesian2Axis
from .test_circle import TestCircle
from .test_circle_arc import TestCircleArc
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
from .test_closed_shape_colored_area import TestClosedShapeColoredArea
from .test_linear_algebra_utils import TestLinearAlgebraUtils
from .test_label import TestLabel
from .test_segments_bounded_colored_area import TestSegmentsBoundedColoredArea
from .test_function_calling import TestProcessFunctionCalls
from .test_math_functions import TestMathFunctions
from .test_geometry_utils import TestGeometryUtils
from .test_point import TestPoint
from .test_rectangle import TestRectangle
from .test_segment import TestSegment
from .test_segment_manager import TestSegmentManager
from .test_polygon_manager import TestPolygonManager
from .test_vector_manager import TestVectorManager
from .test_circle_manager import TestCircleManager
from .test_ellipse_manager import TestEllipseManager
from .test_colored_area_manager import TestColoredAreaManager
from .test_function_manager import TestFunctionManager
from .test_polygon_canonicalizer import TestPolygonCanonicalizer
from .test_triangle import TestTriangle
from .test_quadrilateral import TestQuadrilateral
from .test_pentagon import TestPentagon
from .test_hexagon import TestHexagon
from .test_heptagon import TestHeptagon
from .test_octagon import TestOctagon
from .test_nonagon import TestNonagon
from .test_decagon import TestDecagon
from .test_generic_polygon import TestGenericPolygon
from .test_vector import TestVector
from .test_angle import TestAngle
from .test_angle_manager import TestAngleManager
from .test_arc_manager import TestArcManager
from .test_workspace_manager import TestWorkspaceSegmentPersistence
from .test_function_bounded_colored_area_integration import TestFunctionBoundedColoredAreaIntegration
from .renderer_performance_tests import TestRendererPerformance
from .test_optimized_renderers import TestOptimizedRendererParity
from .test_renderer_primitives import TestRendererPrimitives
from .test_renderer_logic import TestRendererLogic
from .test_drawable_renderers import (
    TestVectorRenderer,
    TestAngleRenderer,
    TestEllipseRenderer,
    TestLabelRenderer,
    TestRendererEdgeCases as TestDrawableRendererEdgeCases,
)
from .test_function_renderables import (
    TestFunctionRenderable,
    TestFunctionsBoundedAreaRenderable,
    TestSegmentsBoundedAreaRenderable,
    TestBoundaryExtension,
    TestRenderableEdgeCases,
)
from .test_renderer_edge_cases import (
    TestPointEdgeCases,
    TestSegmentEdgeCases,
    TestCircleEdgeCases,
    TestVectorEdgeCases,
    TestEllipseEdgeCases,
    TestLabelEdgeCases,
    TestCartesianEdgeCases,
)
from .test_font_helpers import TestCoerceFontSize, TestComputeZoomAdjustedFontSize
from .test_colored_area_helpers import (
    TestPointsClose,
    TestPathsFormSingleLoop,
    TestFilterValidPoints,
    TestRenderColoredAreaHelper,
)
from .test_transformations_manager import TestTransformationsManager
from .brython_io import BrythonTestStream
from .ai_result_formatter import AITestResult


class Tests:
    """Encapsulates execution and formatting of client-side tests."""

    @classmethod
    def run_tests(cls) -> Dict[str, Any]:
        print("[ClientTests] run_tests invoked.")
        test_runner = cls()
        try:
            suite = test_runner._create_test_suite()
            print("[ClientTests] Test suite created.")
            result = test_runner._run_test_suite(suite)
            print("[ClientTests] Test suite execution finished.")
            return test_runner._format_results_for_ai(result)
        except Exception as exc:  # pragma: no cover - defensive path
            print(f"[ClientTests] Exception during run_tests: {repr(exc)}")
            traceback.print_exc()
            return test_runner._create_error_result(str(exc))

    def _create_test_suite(self) -> unittest.TestSuite:
        suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        test_cases: List[Type[unittest.TestCase]] = [
            TestOptimizedRendererParity,
            # TestRendererPerformance,
            # TestRendererPrimitives,
            TestRendererLogic,
            TestVectorRenderer,
            TestAngleRenderer,
            TestEllipseRenderer,
            TestLabelRenderer,
            TestDrawableRendererEdgeCases,
            TestFunctionRenderable,
            TestFunctionsBoundedAreaRenderable,
            TestSegmentsBoundedAreaRenderable,
            TestBoundaryExtension,
            TestRenderableEdgeCases,
            TestPointEdgeCases,
            TestSegmentEdgeCases,
            TestCircleEdgeCases,
            TestVectorEdgeCases,
            TestEllipseEdgeCases,
            TestLabelEdgeCases,
            TestCartesianEdgeCases,
            TestWorkspaceSegmentPersistence,
            TestMathFunctions,
            TestGeometryUtils,
            TestPolygonCanonicalizer,
            TestDrawableNameGenerator,
            TestPoint,
            TestSegment,
            TestSegmentManager,
            TestPolygonManager,
            TestVectorManager,
            TestCircleManager,
            TestEllipseManager,
            TestColoredAreaManager,
            TestTransformationsManager,
            TestFunctionManager,
            TestVector,
            TestTriangle,
            TestQuadrilateral,
            TestPentagon,
            TestHexagon,
            TestHeptagon,
            TestOctagon,
            TestNonagon,
            TestDecagon,
            TestGenericPolygon,
            TestRectangle,
            TestCircle,
            TestCircleArc,
            TestAngle,
            TestAngleManager,
            TestArcManager,
            TestEllipse,
            TestFunction,
            TestFunctionsBoundedColoredArea,
            TestFunctionSegmentBoundedColoredArea,
            TestClosedShapeColoredArea,
            TestSegmentsBoundedColoredArea,
            TestCartesian2Axis,
            TestCanvas,
            TestExpressionValidator,
            TestProcessFunctionCalls,
            TestLabel,
            TestCustomDrawableNames,
            TestThrottle,
            TestWindowMocks,
            TestCanvasEventHandlerTouch,
            TestDrawableDependencyManager,
            TestDrawablesContainer,
            TestFunctionBoundedColoredAreaIntegration,
            TestLinearAlgebraUtils,
            TestCoerceFontSize,
            TestComputeZoomAdjustedFontSize,
            TestPointsClose,
            TestPathsFormSingleLoop,
            TestFilterValidPoints,
            TestRenderColoredAreaHelper,
        ]

        for test_case in test_cases:
            print(f"[ClientTests] Loading tests for {test_case.__name__}.")
            suite.addTest(loader.loadTestsFromTestCase(test_case))

        return suite

    def _run_test_suite(self, suite: unittest.TestSuite) -> AITestResult:
        print("\n========================= TEST OUTPUT =========================")
        custom_stream = BrythonTestStream()
        runner = unittest.TextTestRunner(
            stream=cast(Any, custom_stream),
            resultclass=AITestResult,
            verbosity=2,
        )
        print("[ClientTests] Beginning unittest runner.")
        result = cast(AITestResult, runner.run(suite))
        print("[ClientTests] Unittest runner completed.")
        print("===============================================================\n")
        return result

    def _format_results_for_ai(self, result: AITestResult) -> Dict[str, Any]:
        failures_details = self._format_failures(result.failures)
        errors_details = self._format_errors(result.errors)
        return {
            "failures": failures_details,
            "errors": errors_details,
            "summary": {
                "tests": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
            },
            "output": None,
        }

    def _format_failures(
        self,
        failures: List[Tuple[unittest.TestCase, str]],
    ) -> List[Dict[str, str]]:
        details: List[Dict[str, str]] = []
        for test, error_msg in failures:
            error_message = self._extract_assertion_message(error_msg)
            details.append({"test": str(test), "error": error_message})
        return details

    def _format_errors(
        self,
        errors: List[Tuple[unittest.TestCase, str]],
    ) -> List[Dict[str, str]]:
        details: List[Dict[str, str]] = []
        for test, error_msg in errors:
            error_message = self._extract_error_message(error_msg)
            details.append({"test": str(test), "error": error_message})
        return details

    def _extract_assertion_message(self, error_msg: str) -> str:
        if "AssertionError:" in error_msg:
            return "AssertionError: " + error_msg.split("AssertionError:", 1)[1].strip()
        return self._extract_error_message(error_msg)

    def _extract_error_message(self, error_msg: str) -> str:
        lines = [line for line in error_msg.split("\n") if line.strip()]
        return lines[-1] if lines else error_msg

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
    return Tests.run_tests()