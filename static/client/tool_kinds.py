"""Which kind each AI tool is, kept in pure Python so server tests can import it.

Every tool in ``static/functions_definitions.py`` sits on exactly one list
(``server_tests/test_tool_wiring.py`` checks it):
- ``UNDOABLE_TOOLS`` change the canvas; a tool batch containing one is one undo step,
  and their results are reported with a standard success message.
- ``VALUE_TOOLS`` compute a value the model must see. Some also draw (``fit_regression``
  plots its curve and data, ``find_function_features`` and ``find_intersections`` may
  place points); those group their changes into an undo step of their own.
- ``META_TOOLS`` act on the history, workspaces or tests rather than on the canvas
  content: they are not undoable, and their results are not computations.
"""

from __future__ import annotations

from typing import Tuple

UNDOABLE_TOOLS: Tuple[str, ...] = (
    # Canvas operations that modify state
    "clear_canvas",
    "reset_canvas",
    "zoom",
    "set_coordinate_system",
    "set_grid_visible",
    # Point operations
    "create_point",
    "delete_point",
    "update_point",
    # Segment operations
    "create_segment",
    "delete_segment",
    "update_segment",
    # Vector operations
    "create_vector",
    "delete_vector",
    "update_vector",
    # Polygon operations
    "create_polygon",
    "delete_polygon",
    "update_polygon",
    # Circle operations
    "create_circle",
    "delete_circle",
    "update_circle",
    # Circle arc operations
    "create_circle_arc",
    "delete_circle_arc",
    "update_circle_arc",
    # Ellipse operations
    "create_ellipse",
    "delete_ellipse",
    "update_ellipse",
    # Label operations
    "create_label",
    "delete_label",
    "update_label",
    # Function operations
    "draw_function",
    "delete_function",
    "update_function",
    # Piecewise function operations
    "draw_piecewise_function",
    "delete_piecewise_function",
    "update_piecewise_function",
    # Parametric function operations
    "draw_parametric_function",
    "delete_parametric_function",
    "update_parametric_function",
    # Tangent and normal line operations
    "draw_tangent_line",
    "draw_normal_line",
    # Geometric construction operations
    "construct_midpoint",
    "construct_perpendicular_bisector",
    "construct_perpendicular_from_point",
    "construct_angle_bisector",
    "construct_parallel_line",
    "construct_circumcircle",
    "construct_incircle",
    # Object transformations
    "translate_object",
    "rotate_object",
    "reflect_object",
    "scale_object",
    "shear_object",
    # Colored area operations
    "create_colored_area",
    "create_region_colored_area",
    "delete_colored_area",
    "update_colored_area",
    # Graph operations
    "generate_graph",
    "delete_graph",
    # Plot operations
    "plot_distribution",
    "plot_bars",
    "delete_plot",
    # Angle operations
    "create_angle",
    "delete_angle",
    "update_angle",
)

VALUE_TOOLS: Tuple[str, ...] = (
    "get_current_canvas_state",
    "find_function_features",
    "evaluate_expression",
    "evaluate_linear_algebra_expression",
    "convert",
    "limit",
    "derive",
    "integrate",
    "numeric_integrate",
    "simplify",
    "expand",
    "factor",
    "solve",
    "solve_system_of_equations",
    "solve_numeric",
    "analyze_graph",
    "inspect_relation",
    "find_intersections",
    "fit_regression",
    "compute_descriptive_statistics",
    "calculate_area",
    "convert_coordinates",
    "search_tools",
)

META_TOOLS: Tuple[str, ...] = (
    "undo",
    "redo",
    "run_tests",
    "list_workspaces",
    "save_workspace",
    "load_workspace",
    "delete_workspace",
)
