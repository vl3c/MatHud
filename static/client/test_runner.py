"""
MatHud Testing Framework Runner

Executes test suites for mathematical functions and canvas operations.
Provides formatted test results for AI analysis and validation of system functionality.

Testing Categories:
    - Graphics drawing capabilities and geometric object creation
    - Mathematical function evaluation and computation accuracy
    - Canvas state management and object interactions
    - Error handling and edge case validation

Key Features:
    - Automated test execution with result aggregation
    - AI-formatted output for integration testing
    - Function call validation using real canvas operations
    - Error tracking and debugging information
    - State isolation between test runs

Dependencies:
    - process_function_calls: Function execution framework for testing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, cast

import traceback

from process_function_calls import ProcessFunctionCalls

if TYPE_CHECKING:
    from canvas import Canvas


class TestRunner:
    """Executes test suites and formats results for AI analysis and validation.
    
    Provides comprehensive testing capabilities for mathematical functions, canvas operations,
    and system integration. Manages test execution state and result formatting for both
    human and AI consumption.
    
    Attributes:
        canvas (Canvas): Canvas instance for testing geometric operations
        available_functions (dict): Function registry for validation testing
        undoable_functions (tuple): Functions that support undo/redo testing
        test_results: Aggregated results from test execution
        internal_failures (list): Collection of test failures for debugging
        internal_errors (list): Collection of test errors for analysis
        internal_tests_run (int): Counter of executed test cases
    """
    def __init__(self, canvas: "Canvas", available_functions: Dict[str, Any], undoable_functions: Tuple[str, ...]) -> None:
        """Initialize test runner with canvas and function registry access.
        
        Sets up testing environment with access to canvas operations and function validation.
        
        Args:
            canvas (Canvas): Canvas instance for testing geometric and mathematical operations
            available_functions (dict): Registry of all available AI functions for testing
            undoable_functions (tuple): Functions that support undo/redo operations
        """
        self.canvas: "Canvas" = canvas
        self.available_functions: Dict[str, Any] = available_functions
        self.undoable_functions: Tuple[str, ...] = undoable_functions
        self.test_results: Optional[Dict[str, Any]] = None
        # Initialize to track internal test results
        self._reset_internal_results()

    def _reset_internal_results(self) -> None:
        """Reset all internal test result tracking variables."""
        self.internal_failures: List[Dict[str, str]] = []
        self.internal_errors: List[Dict[str, str]] = []
        self.internal_tests_run: int = 0

    def _test_graphics_drawing(self) -> Dict[str, Any]:
        """Run tests for graphics drawing capabilities.

        Processes each function call individually to enable granular error reporting.
        Each call is tracked as a separate test case.

        Returns:
            Dictionary mapping call keys to their results.
        """
        function_calls: List[Dict[str, Any]] = self._get_graphics_test_function_calls()
        all_results: Dict[str, Any] = {}

        for call in function_calls:
            self.internal_tests_run += 1
            call_key = self._build_graphics_call_key(call)
            try:
                result: Dict[str, Any] = ProcessFunctionCalls.get_results(
                    [call], self.available_functions, self.undoable_functions, self.canvas
                )
                all_results[call_key] = result
            except Exception as e:
                error_message: str = str(e)
                self._add_internal_error(f"Graphics: {call_key}", error_message)
                all_results[call_key] = {"error": error_message}

        return all_results

    def _build_graphics_call_key(self, call: Dict[str, Any]) -> str:
        """Build a descriptive key for a graphics function call.

        Args:
            call: Function call dictionary with 'function_name' and 'arguments'.

        Returns:
            String key in format 'function_name(key_args)' for identification.
        """
        function_name: str = call.get("function_name", "unknown")
        args: Dict[str, Any] = call.get("arguments", {})

        # Extract the most identifying argument for the key
        key_parts: List[str] = []
        for arg_name in ("name", "triangle_name", "rectangle_name", "circle_name",
                         "ellipse_name", "arc_name", "angle_name", "expression",
                         "polygon_segment_names", "drawable1_name", "function_string"):
            if arg_name in args:
                value = args[arg_name]
                key_parts.append(f"{arg_name}:{value}")
                break

        if key_parts:
            return f"{function_name}({', '.join(key_parts)})"
        return function_name
            
    def _get_graphics_test_function_calls(self) -> List[Dict[str, Any]]:
        """Return the list of function calls for graphics drawing tests."""
        return [
            {
                "function_name": "create_segment",
                "arguments": {
                    "x1": -279.0,
                    "y1": 374.0,
                    "x2": -213.0,
                    "y2": 278.0,
                    "name": "CH",
                    "color": None,
                    "label_text": "Segment CH",
                    "label_visible": True,
                },
            },
            {
                "function_name": "create_point",
                "arguments": {"x": -290.0, "y": 240.0, "name": "P"}
            },
            {
                "function_name": "create_vector",
                "arguments": {"origin_x": -143.0, "origin_y": 376.0, "tip_x": -82.0, "tip_y": 272.0, "name": "v1"}
            },
            {
                "function_name": "create_polygon",
                "arguments": {
                    "vertices": [
                        {"x": -60, "y": 380},
                        {"x": 60, "y": 380},
                        {"x": 0, "y": 260}
                    ],
                    "polygon_type": "triangle",
                    "name": "A'BD"
                }
            },
            {
                "function_name": "create_polygon",
                "arguments": {
                    "vertices": [
                        {"x": 170.0, "y": 380.0},
                        {"x": 238.0, "y": 380.0},
                        {"x": 238.0, "y": 260.0},
                        {"x": 170.0, "y": 260.0}
                    ],
                    "polygon_type": "rectangle",
                    "color": None,
                    "name": "REFT"
                }
            },
            {
                "function_name": "create_circle",
                "arguments": {"center_x": -360, "center_y": 320, "radius": 60, "name": "G(60)"}
            },
            {
                "function_name": "create_ellipse",
                "arguments": {"center_x": 360, "center_y": 320, "radius_x": 80, "radius_y": 50, "name": "I(80, 50)"}
            },
            # 12-sided polygon below the x-axis
            {
                "function_name": "create_polygon",
                "arguments": {
                    "vertices": [
                        {"x": -100.0, "y": -240.0},
                        {"x": -70.0, "y": -200.0},
                        {"x": -20.0, "y": -180.0},
                        {"x": 40.0, "y": -190.0},
                        {"x": 90.0, "y": -220.0},
                        {"x": 130.0, "y": -260.0},
                        {"x": 140.0, "y": -310.0},
                        {"x": 100.0, "y": -350.0},
                        {"x": 40.0, "y": -360.0},
                        {"x": -20.0, "y": -350.0},
                        {"x": -70.0, "y": -320.0},
                        {"x": -110.0, "y": -280.0}
                    ],
                    "polygon_type": "generic",
                    "name": "JOKLMNQSUWXY",
                    "color": None,
                    "subtype": None
                }
            },
            {
                "function_name": "draw_function",
                "arguments": {"function_string": "50 * sin(x / 50)", "name": "f1", "left_bound": -300, "right_bound": 300}
            },
            {
                "function_name": "draw_function",
                "arguments": {"function_string": "100 * sin(x / 30)", "name": "f2", "left_bound": -300, "right_bound": 300}
            },
            {
                "function_name": "draw_function",
                "arguments": {"function_string": "100 * sin(x / 50) + 50 * tan(x / 100)", "name": "f3", "left_bound": -300, "right_bound": 300}
            },
            {
                "function_name": "draw_function",
                "arguments": {"function_string": "-1/x", "name": "f4", "left_bound": -10, "right_bound": 10, "color": "red"}
            },
            {
                "function_name": "draw_function",
                "arguments": {"function_string": "tan(x)", "name": "f5", "left_bound": -5, "right_bound": 5, "color": "orange"}
            },
            # Parametric function: red spiral
            {
                "function_name": "draw_parametric_function",
                "arguments": {
                    "x_expression": "t*cos(t)",
                    "y_expression": "t*sin(t)",
                    "t_min": 0,
                    "t_max": 50,
                    "color": "red"
                }
            },
            {
                "function_name": "create_circle_arc",
                "arguments": {
                    "point1_x": 319.0,
                    "point1_y": -146.0,
                    "point2_x": 246.0,
                    "point2_y": -220.0,
                    "center_x": 316.0,
                    "center_y": -206.0,
                    "radius": 60.06,
                    "arc_name": "arc_ZB'",
                    "color": "orange",
                    "use_major_arc": False
                }
            },
            {
                "function_name": "create_circle_arc",
                "arguments": {
                    "point1_x": 233.0,
                    "point1_y": -314.0,
                    "point2_x": 370.0,
                    "point2_y": -52.0,
                    "center_x": 324.0,
                    "center_y": -174.0,
                    "radius": 109.0,
                    "arc_name": "arc_C'D'",
                    "color": "purple",
                    "use_major_arc": True
                }
            },
            {
                "function_name": "create_segment",
                "arguments": {
                    "x1": 365.0,
                    "y1": -286.0,
                    "x2": 440.0,
                    "y2": -132.0,
                    "name": "A''E'",
                    "color": None,
                    "label_text": "should_not_be_visible",
                    "label_visible": False,
                },
            },
            {
                "function_name": "create_region_colored_area",
                "arguments": {
                    "polygon_segment_names": [
                        "JO",
                        "OK",
                        "KL",
                        "LM",
                        "MN",
                        "NQ",
                        "QS",
                        "SU",
                        "UW",
                        "WX",
                        "XY",
                        "YJ"
                    ],
                    "color": "plum",
                    "opacity": 0.3
                }
            },
            {
                "function_name": "create_colored_area",
                "arguments": {"drawable1_name": "f1", "drawable2_name": "f2", "color": "orange", "opacity": 0.3}
            },
            {
                "function_name": "create_colored_area",
                "arguments": {"drawable1_name": "f2", "drawable2_name": "x_axis", "color": "lightgreen", "opacity": 0.3}
            },
            {
                "function_name": "create_colored_area",
                "arguments": {"drawable1_name": "f1", "drawable2_name": "f3", "color": "lightblue", "opacity": 0.3}
            },
            {
                "function_name": "create_colored_area",
                "arguments": {
                    "drawable1_name": "f3",
                    "drawable2_name": "CH",
                    "color": "lightgray",
                    "opacity": 0.25
                }
            },
            {
                "function_name": "create_region_colored_area",
                "arguments": {
                    "triangle_name": "A'BD",
                    "color": "orange",
                    "opacity": 0.4
                }
            },
            {
                "function_name": "create_region_colored_area",
                "arguments": {
                    "circle_name": "G(60)",
                    "color": "red",
                    "opacity": 0.35
                }
            },
            {
                "function_name": "create_region_colored_area",
                "arguments": {
                    "rectangle_name": "REFT",
                    "color": "green",
                    "opacity": 0.35
                }
            },
            {
                "function_name": "create_region_colored_area",
                "arguments": {
                    "ellipse_name": "I(80, 50)",
                    "color": "blue",
                    "opacity": 0.35
                }
            },
            {
                "function_name": "create_angle",
                "arguments": {
                    "vx": -408.0,
                    "vy": 150.0,
                    "p1x": -328.0,
                    "p1y": 150.0,
                    "p2x": -408.0,
                    "p2y": 230.0,
                    "color": "red",
                    "angle_name": "Angle1",
                    "is_reflex": True
                }
            },
            {
                "function_name": "create_angle",
                "arguments": {
                    "vx": -118.0,
                    "vy": 150.0,
                    "p1x": -38.0,
                    "p1y": 150.0,
                    "p2x": -158.0,
                    "p2y": 219.3,
                    "color": "blue",
                    "angle_name": "Angle2",
                    "is_reflex": False
                }
            },
            {
                "function_name": "create_angle",
                "arguments": {
                    "vx": 170.0,
                    "vy": 220.0,
                    "p1x": 250.0,
                    "p1y": 220.0,
                    "p2x": 210.0,
                    "p2y": 150.3,
                    "color": "green",
                    "angle_name": "Angle3",
                    "is_reflex": False
                }
            },
            {
                "function_name": "create_label",
                "arguments": {
                    "x": -412.0,
                    "y": 69.5,
                    "text": "Left Label",
                    "name": "LabelLeft",
                    "color": "purple",
                    "font_size": 18,
                    "rotation_degrees": 15.0,
                },
            },
            {
                "function_name": "create_label",
                "arguments": {
                    "x": 300.0,
                    "y": 100.0,
                    "text": "Right Label with\nMultiple Lines",
                    "name": "LabelRight",
                    "color": "teal",
                    "font_size": 14,
                    "rotation_degrees": -10.0,
                },
            },
            {
                "function_name": "create_region_colored_area",
                "arguments": {
                    "expression": "ArcMaj_C'D' & A''E'",
                    "color": "#D2B48C",
                    "opacity": 0.5
                }
            },
            {
                "function_name": "update_segment",
                "arguments": {
                    "name": "F'H'",
                    "new_color": None,
                    "new_label_text": "Segment updated",
                    "new_label_visible": True,
                },
            },
            # 3-layer binary tree in the lower-left quadrant
            {
                "function_name": "generate_graph",
                "arguments": {
                    "name": "BinaryTree",
                    "graph_type": "tree",
                    "directed": True,
                    "root": "R",
                    "layout": "tree",
                    "placement_box": {
                        "x": -500,
                        "y": -350,
                        "width": 300,
                        "height": 325
                    },
                    "vertices": [
                        {"name": "R", "x": None, "y": None, "color": None, "label": None},
                        {"name": "L1", "x": None, "y": None, "color": None, "label": None},
                        {"name": "L2", "x": None, "y": None, "color": None, "label": None},
                        {"name": "L1A", "x": None, "y": None, "color": None, "label": None},
                        {"name": "L1B", "x": None, "y": None, "color": None, "label": None},
                        {"name": "L2A", "x": None, "y": None, "color": None, "label": None},
                        {"name": "L2B", "x": None, "y": None, "color": None, "label": None}
                    ],
                    "edges": [
                        {"source": 0, "target": 1, "weight": None, "name": "R_L1", "color": None, "directed": True},
                        {"source": 0, "target": 2, "weight": None, "name": "R_L2", "color": None, "directed": True},
                        {"source": 1, "target": 3, "weight": None, "name": "L1_L1A", "color": None, "directed": True},
                        {"source": 1, "target": 4, "weight": None, "name": "L1_L1B", "color": None, "directed": True},
                        {"source": 2, "target": 5, "weight": None, "name": "L2_L2A", "color": None, "directed": True},
                        {"source": 2, "target": 6, "weight": None, "name": "L2_L2B", "color": None, "directed": True}
                    ],
                    "adjacency_matrix": None
                }
            },
            {
                "function_name": "clear_canvas",
                "arguments": {}
            },
            {
                "function_name": "undo",
                "arguments": {}
            },
            {
                "function_name": "redo",
                "arguments": {}
            },
            {
                "function_name": "undo",
                "arguments": {}
            }
        ]

    def _test_undoable_functions(self) -> bool:
        """Test that all undoable functions are available."""
        self.internal_tests_run += 1
        try:
            self._validate_undoable_functions()
            print("All undoable functions are available.")   # DEBUG
            return True
        except Exception as e:
            error_message: str = f"Error in undoable functions test: {str(e)}"
            print(error_message)
            if not any(failure['test'] == 'Undoable Functions Test' for failure in self.internal_failures):
                self._add_internal_failure('Undoable Functions Test', error_message)
            return False
            
    def _validate_undoable_functions(self) -> None:
        """Validate that all undoable functions are in the available functions list."""
        for function_name in self.undoable_functions:
            if function_name not in self.available_functions:
                error_message: str = f"Function '{function_name}' is not available."
                self._add_internal_failure('Undoable Functions Test', error_message)
                raise Exception(error_message)

    def _add_internal_failure(self, test_name: str, error_message: str) -> None:
        """Add a failure to the internal failures list."""
        self.internal_failures.append({
            'test': test_name,
            'error': error_message
        })
        
    def _add_internal_error(self, test_name: str, error_message: str) -> None:
        """Add an error to the internal errors list."""
        self.internal_errors.append({
            'test': test_name,
            'error': error_message
        })

    def run_tests(self) -> Dict[str, Any]:
        """Run unit tests for the graphics and function capabilities."""
        # Reset internal test results
        self._reset_internal_results()

        client_results: Optional[Dict[str, Any]] = None
        client_error_message: Optional[str] = None
        client_import_error: bool = False

        try:
            # Run the client-side main tests first
            client_results = self._run_client_tests()
            print("[TestRunner] Client tests completed successfully.")
        except ImportError:
            print("[TestRunner] client_tests module not available - skipping additional tests.")
            client_import_error = True
        except Exception as e:
            print(f"[TestRunner] Error running client tests: {repr(e)}")
            traceback.print_exc()
            client_error_message = str(e)

        print("Running graphics and function tests...")
        self._test_graphics_drawing()
        self._test_undoable_functions()

        if client_results is not None:
            self.test_results = self._merge_test_results(client_results)
        elif client_error_message is not None:
            self.test_results = self._create_results_with_client_error(client_error_message)
        elif client_import_error:
            self.test_results = self._create_results_from_internal_only()
        else:
            self.test_results = self._create_results_from_internal_only()

        return self.test_results

    async def run_tests_async(
        self,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Run unit tests asynchronously, yielding to browser between test classes.

        Args:
            should_stop: Optional callback that returns True if tests should be stopped.
        """
        # Reset internal test results
        self._reset_internal_results()

        client_results: Optional[Dict[str, Any]] = None
        client_error_message: Optional[str] = None
        client_import_error: bool = False

        try:
            # Run the client-side main tests asynchronously
            client_results = await self._run_client_tests_async(should_stop=should_stop)
            print("[TestRunner] Async client tests completed successfully.")
        except ImportError:
            print("[TestRunner] client_tests module not available - skipping additional tests.")
            client_import_error = True
        except Exception as e:
            print(f"[TestRunner] Error running async client tests: {repr(e)}")
            traceback.print_exc()
            client_error_message = str(e)

        # Skip additional tests if stopped
        if not (should_stop and should_stop()):
            print("Running graphics and function tests...")
            self._test_graphics_drawing()
            self._test_undoable_functions()

        if client_results is not None:
            self.test_results = self._merge_test_results(client_results)
        elif client_error_message is not None:
            self.test_results = self._create_results_with_client_error(client_error_message)
        elif client_import_error:
            self.test_results = self._create_results_from_internal_only()
        else:
            self.test_results = self._create_results_from_internal_only()

        return self.test_results
        
    def _run_client_tests(self) -> Dict[str, Any]:
        """Run the client-side tests and return the results."""
        try:
            from client_tests.tests import run_tests
            return cast(Dict[str, Any], run_tests())
        except ImportError as e:
            print(f"client_tests import failed: {e}")
            # Re-raise ImportError to be handled by the calling method
            raise ImportError(f"client_tests module not available: {e}")

    async def _run_client_tests_async(
        self,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Run the client-side tests asynchronously and return the results."""
        try:
            from client_tests.tests import run_tests_async
            return cast(Dict[str, Any], await run_tests_async(should_stop=should_stop))
        except ImportError as e:
            print(f"client_tests import failed: {e}")
            raise ImportError(f"client_tests module not available: {e}")
        
    def _merge_test_results(self, client_results: Dict[str, Any]) -> Dict[str, Any]:
        """Merge client test results with internal test results."""
        if not client_results:
            return self._create_results_from_internal_only()
            
        # Start with the client results
        merged_results: Dict[str, Any] = client_results.copy()
        
        # Add internal failures and errors to client test results
        merged_results['failures'].extend(self.internal_failures)
        merged_results['errors'].extend(self.internal_errors)
        
        # Update summary
        merged_results['summary']['tests'] += self.internal_tests_run
        merged_results['summary']['failures'] += len(self.internal_failures)
        merged_results['summary']['errors'] += len(self.internal_errors)
        
        return merged_results
        
    def _create_results_from_internal_only(self) -> Dict[str, Any]:
        """Create test results containing only internal test results."""
        return {
            'failures': self.internal_failures,
            'errors': self.internal_errors,
            'summary': {
                'tests': self.internal_tests_run,
                'failures': len(self.internal_failures),
                'errors': len(self.internal_errors)
            }
        }
        
    def _create_results_with_client_error(self, error_message: str) -> Dict[str, Any]:
        """Create test results with internal results plus a client test runner error."""
        client_error: Dict[str, str] = {
            'test': 'Client Tests Runner',
            'error': f"Error running client tests: {error_message}"
        }
        
        return {
            'failures': self.internal_failures,
            'errors': self.internal_errors + [client_error],
            'summary': {
                'tests': self.internal_tests_run,
                'failures': len(self.internal_failures),
                'errors': len(self.internal_errors) + 1
            }
        }
            
    def get_test_results(self) -> Optional[Dict[str, Any]]:
        """Return the most recent test results."""
        return self.test_results

    def format_results_for_ai(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format test results for the AI in a clean, concise format."""
        if not results:
            return {"tests_run": 0, "failures": 0, "errors": 0, "failing_tests": [], "error_tests": []}
        
        # Create a summary for the AI
        formatted_results: Dict[str, Any] = self._create_formatted_results_summary(results)
        
        # Add details of failures and errors
        self._add_formatted_failure_details(formatted_results, results)
        self._add_formatted_error_details(formatted_results, results)
        
        # Log detailed test results to console
        self._log_test_results_to_console(formatted_results)
            
        return formatted_results
        
    def _create_formatted_results_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic summary of test results for AI consumption."""
        return {
            "tests_run": results['summary']['tests'],
            "failures": results['summary']['failures'],
            "errors": results['summary']['errors'],
            "failing_tests": [],
            "error_tests": []
        }
        
    def _add_formatted_failure_details(self, formatted_results: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Add failure details to the formatted results."""
        if results['failures']:
            for failure in results['failures']:
                formatted_results["failing_tests"].append({
                    "test": failure['test'],
                    "error": failure['error']
                })
                
    def _add_formatted_error_details(self, formatted_results: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Add error details to the formatted results."""
        if results['errors']:
            for error in results['errors']:
                formatted_results["error_tests"].append({
                    "test": error['test'],
                    "error": error['error']
                })
                
    def _log_test_results_to_console(self, formatted_results: Dict[str, Any]) -> None:
        """Log detailed test results to the console for debugging."""
        print("========================= TEST RESULTS =========================")
        print(f"Tests Run: {formatted_results['tests_run']}")
        print(f"Failures: {formatted_results['failures']}")
        print(f"Errors: {formatted_results['errors']}")
        
        if formatted_results['failing_tests']:
            print("\nFAILURES:")
            for i, failure in enumerate(formatted_results['failing_tests'], 1):
                print(f"{i}. {failure['test']}: {failure['error']}")
            
        if formatted_results['error_tests']:
            print("\nERRORS:")
            for i, error in enumerate(formatted_results['error_tests'], 1):
                print(f"{i}. {error['test']}: {error['error']}")
        
        print("===============================================================") 