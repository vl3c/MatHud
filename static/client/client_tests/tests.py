from __future__ import annotations

import unittest
import traceback
from typing import Any, Callable, Dict, List, Tuple, Type, cast

from browser import aio

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
from .test_chat_message_menu import TestChatMessageMenu
from .test_throttle import TestThrottle
from .test_window_mocks import TestWindowMocks
from .test_expression_validator import TestExpressionValidator
from .test_function import TestFunction, TestFunctionUndefinedAt
from .test_functions_bounded_colored_area import TestFunctionsBoundedColoredArea
from .test_function_segment_bounded_colored_area import TestFunctionSegmentBoundedColoredArea
from .test_closed_shape_colored_area import TestClosedShapeColoredArea
from .test_linear_algebra_utils import TestLinearAlgebraUtils
from .test_label import TestLabel
from .test_label_overlap_resolver import TestLabelOverlapResolver
from .test_screen_offset_label_layout import TestScreenOffsetLabelLayout
from .test_bar_manager import TestBarManager
from .test_segments_bounded_colored_area import TestSegmentsBoundedColoredArea
from .test_function_calling import TestProcessFunctionCalls, TestProcessFunctionCallsPlotTools
from .test_math_functions import TestMathFunctions, TestNumberTheory
from .test_periodicity_detection import TestPeriodicityDetection, TestPeriodicityEdgeCases
from .test_geometry_utils import TestGeometryUtils, TestConvexHull, TestPointInConvexHull
from .test_graph_layout import TestGraphLayout, TestGraphLayoutVisibility
from .test_graph_manager import TestGraphManager
from .test_graph_analyzer import (
    TestAnalyzeGraphShortestPath,
    TestAnalyzeGraphMST,
    TestAnalyzeGraphTopologicalSort,
    TestAnalyzeGraphBridges,
    TestAnalyzeGraphArticulationPoints,
    TestAnalyzeGraphEulerStatus,
    TestAnalyzeGraphBipartite,
    TestAnalyzeGraphBFSDFS,
    TestAnalyzeGraphTreeOperations,
    TestAnalyzeGraphTreeTransforms,
    TestAnalyzeGraphEdgeCases,
    TestAnalyzeGraphCompleteGraph,
    TestAnalyzeGraphGridGraph,
    TestAnalyzeGraphConvexHull,
    TestAnalyzeGraphPointInHull,
)
from .test_graph_utils import TestGraphUtils
from .test_statistics_distributions import TestStatisticsDistributions
from .test_statistics_manager import TestStatisticsManager
from .test_regression import TestRegressionCanvas
from .test_path_elements import TestPathElements
from .test_intersections import TestIntersections
from .test_region import TestRegion, TestAreaUtilities
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
from .test_parametric_function import (
    TestParametricFunction,
    TestParametricFunctionRenderable,
    TestExpressionValidatorParametric,
)
from .test_parametric_function_manager import TestParametricFunctionManager
from .test_tangent_manager import (
    TestTangentToFunction,
    TestNormalToFunction,
    TestTangentToCircle,
    TestTangentToEllipse,
    TestTangentToParametricFunction,
    TestMathUtilsTangentFunctions,
    TestUndoRedo as TestTangentUndoRedo,
)
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
from .test_workspace_manager import TestWorkspaceLabelRestore, TestWorkspaceSegmentPersistence
from .test_workspace_plots import TestWorkspacePlotsRestore
from .test_zoom import (
    TestZoomXAxisRange,
    TestZoomYAxisRange,
    TestZoomAspectRatios,
    TestZoomEdgeCases,
)
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
    TestPointLabelRenderer,
    TestSegmentLabelRenderer,
    TestCircleArcRenderer,
    TestRendererEdgeCases as TestDrawableRendererEdgeCases,
)
from .test_point_manager import TestPointManagerUpdates
from .test_bar_renderer import TestBarRenderer
from .test_function_renderables import (
    TestFunctionRenderable,
    TestFunctionsBoundedAreaRenderable,
    TestSegmentsBoundedAreaRenderable,
    TestBoundaryExtension,
    TestRenderableEdgeCases,
)
from .test_piecewise_function import (
    TestPiecewiseFunctionInterval,
    TestPiecewiseFunction,
    TestPiecewiseFunctionRendering,
    TestPiecewiseFunctionUndefinedAt,
    TestPiecewiseFunctionEdgeCases,
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
from .test_area_expression_evaluator import (
    TestAreaCalculation,
    TestRegionGeneration,
)
from .test_polar_grid import TestPolarGrid, TestPolarGridConversions
from .test_coordinate_system_manager import (
    TestCoordinateSystemManager,
    TestCoordinateSystemManagerCanvasIntegration,
)
from .test_slash_commands import (
    TestSlashCommandHandler,
    TestEssentialCommands,
    TestCanvasViewCommands,
    TestUtilityCommands,
    TestCommandResult,
    TestEditDistance,
    TestModelCommand,
    TestVisionCommand,
    TestCommandAutocomplete,
    TestWorkspaceSuggestions,
    TestModelSuggestions,
    TestExpandableContent,
    TestExportCommandOutput,
    TestStatusCommandOutput,
)
from .test_image_attachment import (
    TestAttachedImagesState,
    TestImageValidation,
    TestPayloadGeneration,
    TestSlashCommandImage,
    TestImageLimitLogic,
    TestImageRemovalLogic,
    TestPreviewAreaLogic,
    TestModalLogic,
    TestMessageElementWithImages,
    TestDataURLParsing,
    TestImageOnlySending,
)
from .test_tool_call_log import TestToolCallLog
from .test_numeric_solver import (
    TestNumericSolverHelpers,
    TestNumericSolverIntegration,
    TestNumericSolverFallback,
    TestJacobianComputation,
    TestExpressionEvaluation,
)
from .test_error_recovery import TestErrorRecovery
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

    @classmethod
    async def run_tests_async(
        cls,
        progress_callback: Callable[[int, int], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Dict[str, Any]:
        """Run tests asynchronously, yielding to browser between test classes.

        Args:
            progress_callback: Optional callback(completed, total) for progress updates.
            should_stop: Optional callback that returns True if tests should be stopped.

        Returns:
            Test results in the same format as run_tests(), with 'stopped' key if cancelled.
        """
        print("[ClientTests] run_tests_async invoked.")
        test_runner = cls()
        stopped = False
        try:
            test_cases = test_runner._get_test_cases()
            total = len(test_cases)
            print(f"[ClientTests] Running {total} test classes asynchronously.")

            loader = unittest.TestLoader()
            custom_stream = BrythonTestStream()
            combined_result = AITestResult(cast(Any, custom_stream), descriptions=True, verbosity=2)

            print("\n========================= TEST OUTPUT =========================")
            for i, test_case in enumerate(test_cases):
                # Check if stop was requested
                if should_stop and should_stop():
                    print(f"[ClientTests] Stop requested after {i} test classes.")
                    stopped = True
                    break

                # Yield to browser to keep UI responsive
                await aio.sleep(0)

                suite = unittest.TestSuite()
                suite.addTest(loader.loadTestsFromTestCase(test_case))
                suite.run(combined_result)

                if progress_callback:
                    progress_callback(i + 1, total)

            print("===============================================================\n")
            if stopped:
                print("[ClientTests] Async test execution stopped by user.")
            else:
                print("[ClientTests] Async test execution finished.")

            results = test_runner._format_results_for_ai(combined_result)
            results['stopped'] = stopped
            return results
        except Exception as exc:  # pragma: no cover - defensive path
            print(f"[ClientTests] Exception during run_tests_async: {repr(exc)}")
            traceback.print_exc()
            return test_runner._create_error_result(str(exc))

    def _get_test_cases(self) -> List[Type[unittest.TestCase]]:
        """Return the list of test case classes to run."""
        return [
            TestOptimizedRendererParity,
            # TestRendererPerformance,
            # TestRendererPrimitives,
            TestRendererLogic,
            TestChatMessageMenu,
            TestLabelOverlapResolver,
            TestScreenOffsetLabelLayout,
            TestBarManager,
            TestBarRenderer,
            TestVectorRenderer,
            TestAngleRenderer,
            TestEllipseRenderer,
            TestLabelRenderer,
            TestPointLabelRenderer,
            TestSegmentLabelRenderer,
            TestCircleArcRenderer,
            TestDrawableRendererEdgeCases,
            TestPointManagerUpdates,
            TestFunctionRenderable,
            TestFunctionsBoundedAreaRenderable,
            TestSegmentsBoundedAreaRenderable,
            TestBoundaryExtension,
            TestRenderableEdgeCases,
            TestPiecewiseFunctionInterval,
            TestPiecewiseFunction,
            TestPiecewiseFunctionRendering,
            TestPiecewiseFunctionUndefinedAt,
            TestPiecewiseFunctionEdgeCases,
            TestPointEdgeCases,
            TestSegmentEdgeCases,
            TestCircleEdgeCases,
            TestVectorEdgeCases,
            TestEllipseEdgeCases,
            TestLabelEdgeCases,
            TestCartesianEdgeCases,
            TestWorkspaceSegmentPersistence,
            TestWorkspaceLabelRestore,
            TestWorkspacePlotsRestore,
            TestMathFunctions,
            TestNumberTheory,
            TestPeriodicityDetection,
            TestPeriodicityEdgeCases,
            TestGeometryUtils,
            TestConvexHull,
            TestPointInConvexHull,
            TestGraphLayout,
            TestGraphLayoutVisibility,
            TestGraphManager,
            TestGraphUtils,
            TestStatisticsDistributions,
            TestStatisticsManager,
            TestRegressionCanvas,
            TestAnalyzeGraphShortestPath,
            TestAnalyzeGraphMST,
            TestAnalyzeGraphTopologicalSort,
            TestAnalyzeGraphBridges,
            TestAnalyzeGraphArticulationPoints,
            TestAnalyzeGraphEulerStatus,
            TestAnalyzeGraphBipartite,
            TestAnalyzeGraphBFSDFS,
            TestAnalyzeGraphTreeOperations,
            TestAnalyzeGraphTreeTransforms,
            TestAnalyzeGraphEdgeCases,
            TestAnalyzeGraphCompleteGraph,
            TestAnalyzeGraphGridGraph,
            TestAnalyzeGraphConvexHull,
            TestAnalyzeGraphPointInHull,
            TestPathElements,
            TestIntersections,
            TestRegion,
            TestAreaUtilities,
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
            TestParametricFunction,
            TestParametricFunctionRenderable,
            TestExpressionValidatorParametric,
            TestParametricFunctionManager,
            TestTangentToFunction,
            TestNormalToFunction,
            TestTangentToCircle,
            TestTangentToEllipse,
            TestTangentToParametricFunction,
            TestMathUtilsTangentFunctions,
            TestTangentUndoRedo,
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
            TestFunctionUndefinedAt,
            TestFunctionsBoundedColoredArea,
            TestFunctionSegmentBoundedColoredArea,
            TestClosedShapeColoredArea,
            TestSegmentsBoundedColoredArea,
            TestCartesian2Axis,
            TestCanvas,
            TestZoomXAxisRange,
            TestZoomYAxisRange,
            TestZoomAspectRatios,
            TestZoomEdgeCases,
            TestExpressionValidator,
            TestProcessFunctionCalls,
            TestProcessFunctionCallsPlotTools,
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
            TestAreaCalculation,
            TestRegionGeneration,
            TestPolarGrid,
            TestPolarGridConversions,
            TestCoordinateSystemManager,
            TestCoordinateSystemManagerCanvasIntegration,
            TestSlashCommandHandler,
            TestEssentialCommands,
            TestCanvasViewCommands,
            TestUtilityCommands,
            TestCommandResult,
            TestEditDistance,
            TestModelCommand,
            TestVisionCommand,
            TestCommandAutocomplete,
            TestWorkspaceSuggestions,
            TestModelSuggestions,
            TestExpandableContent,
            TestExportCommandOutput,
            TestStatusCommandOutput,
            TestAttachedImagesState,
            TestImageValidation,
            TestPayloadGeneration,
            TestSlashCommandImage,
            TestImageLimitLogic,
            TestImageRemovalLogic,
            TestPreviewAreaLogic,
            TestModalLogic,
            TestMessageElementWithImages,
            TestDataURLParsing,
            TestImageOnlySending,
            TestToolCallLog,
            TestNumericSolverHelpers,
            TestNumericSolverIntegration,
            TestNumericSolverFallback,
            TestJacobianComputation,
            TestExpressionEvaluation,
            TestErrorRecovery,
        ]

    def _create_test_suite(self) -> unittest.TestSuite:
        suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        test_cases = self._get_test_cases()

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


async def run_tests_async(
    progress_callback: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Dict[str, Any]:
    """Run tests asynchronously, yielding to browser between test classes."""
    return await Tests.run_tests_async(progress_callback, should_stop)
