FUNCTIONS = [
            {
                "type": "function",
                "function": {
                    "name": "reset_canvas",
                    "description": "Resets the canvas zoom and offset",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clear_canvas",
                    "description": "Clears the canvas by deleting all drawable objects",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "undo",
                    "description": "Undoes the last action on the canvas",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "redo",
                    "description": "Redoes the last action on the canvas",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_tests",
                    "description": "Runs the test suite for the canvas",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_point",
                    "description": "Creates and draws a point at the given coordinates. If a name is provided, it will try to use the first available letter from that name as the point's name.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "The X coordinate of the point"
                            },
                            "y": {
                                "type": "number",
                                "description": "The Y coordinate of the point"
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name for the point. If provided, the first available letter from this name will be used."
                            }
                        },
                        "required": ["x", "y", "name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_point",
                    "description": "Deletes the point with the given coordinates",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "The X coordinate of the point"
                            },
                            "y": {
                                "type": "number",
                                "description": "The Y coordinate of the point"
                            }
                        },
                        "required": ["x", "y"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_segment",
                    "description": "Creates and draws a segment at the given coordinates for two points. If a name is provided, the first two available letters from that name will be used to name the endpoints.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x1": {
                                "type": "number",
                                "description": "The X coordinate of the first point"
                            },
                            "y1": {
                                "type": "number",
                                "description": "The Y coordinate of the first point"
                            },
                            "x2": {
                                "type": "number",
                                "description": "The X coordinate of the second point"
                            },
                            "y2": {
                                "type": "number",
                                "description": "The Y coordinate of the second point"
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name for the segment. If provided, the first two available letters will be used to name the endpoints."
                            }
                        },
                        "required": ["x1", "y1", "x2", "y2", "name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_segment",
                    "description": "Deletes the segment found at the given coordinates for two points. If only a name is given, search for appropriate point coordinates in the canvas state.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x1": {
                                "type": "number",
                                "description": "The X coordinate of the first point"
                            },
                            "y1": {
                                "type": "number",
                                "description": "The Y coordinate of the first point"
                            },
                            "x2": {
                                "type": "number",
                                "description": "The X coordinate of the second point"
                            },
                            "y2": {
                                "type": "number",
                                "description": "The Y coordinate of the second point"
                            }
                        },
                        "required": ["x1", "y1", "x2", "y2"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_vector",
                    "description": "Creates and draws a vector at the given coordinates for two points called origin and tip. If a name is provided, the first two available letters will be used to name the origin and tip points.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin_x": {
                                "type": "number",
                                "description": "The X coordinate of the origin point"
                            },
                            "origin_y": {
                                "type": "number",
                                "description": "The Y coordinate of the origin point"
                            },
                            "tip_x": {
                                "type": "number",
                                "description": "The X coordinate of the tip point"
                            },
                            "tip_y": {
                                "type": "number",
                                "description": "The Y coordinate of the tip point"
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name for the vector. If provided, the first two available letters will be used to name the origin and tip points."
                            }
                        },
                        "required": ["origin_x", "origin_y", "tip_x", "tip_y", "name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_vector",
                    "description": "Deletes the vector found at the given coordinates for two points called origin and tip. If only a name is given, search for appropriate point coordinates in the canvas state.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin_x": {
                                "type": "number",
                                "description": "The X coordinate of the origin point",
                            },
                            "origin_y": {
                                "type": "number",
                                "description": "The Y coordinate of the origin point",
                            },
                            "tip_x": {
                                "type": "number",
                                "description": "The X coordinate of the tip point",
                            },
                            "tip_y": {
                                "type": "number",
                                "description": "The Y coordinate of the tip point",
                            }
                        },
                        "required": ["origin_x", "origin_y", "tip_x", "tip_y"],
                        "additionalProperties": False
                    }
                }
            },            
            {
                "type": "function",
                "function": {
                    "name": "create_triangle",
                    "description": "Creates and draws a triangle at the given coordinates for three points. If a name is provided, the first three available letters will be used to name the vertices.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x1": {
                                "type": "number",
                                "description": "The X coordinate of the first point"
                            },
                            "y1": {
                                "type": "number",
                                "description": "The Y coordinate of the first point"
                            },
                            "x2": {
                                "type": "number",
                                "description": "The X coordinate of the second point"
                            },
                            "y2": {
                                "type": "number",
                                "description": "The Y coordinate of the second point"
                            },
                            "x3": {
                                "type": "number",
                                "description": "The X coordinate of the third point"
                            },
                            "y3": {
                                "type": "number",
                                "description": "The Y coordinate of the third point"
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name for the triangle. If provided, the first three available letters will be used to name the vertices."
                            }
                        },
                        "required": ["x1", "y1", "x2", "y2", "x3", "y3", "name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_triangle",
                    "description": "Deletes the triangle found at the given coordinates for three points. If only a name is given, search for appropriate point coordinates in the canvas state.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x1": {
                                "type": "number",
                                "description": "The X coordinate of the first point",
                            },
                            "y1": {
                                "type": "number",
                                "description": "The Y coordinate of the first point",
                            },
                            "x2": {
                                "type": "number",
                                "description": "The X coordinate of the second point",
                            },
                            "y2": {
                                "type": "number",
                                "description": "The Y coordinate of the second point",
                            },
                            "x3": {
                                "type": "number",
                                "description": "The X coordinate of the third point",
                            },
                            "y3": {
                                "type": "number",
                                "description": "The Y coordinate of the third point",
                            }
                        },
                        "required": ["x1", "y1", "x2", "y2", "x3", "y3"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_rectangle",
                    "description": "Creates and draws a rectangle at the given coordinates for two diagonal points. If a name is provided, the first four available letters will be used to name the corners.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "px": {
                                "type": "number",
                                "description": "The X coordinate of the origin point"
                            },
                            "py": {
                                "type": "number",
                                "description": "The Y coordinate of the origin point"
                            },
                            "opposite_px": {
                                "type": "number",
                                "description": "The X coordinate of the opposite point"
                            },
                            "opposite_py": {
                                "type": "number",
                                "description": "The Y coordinate of the opposite point"
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name for the rectangle. If provided, the first four available letters will be used to name the corners."
                            }
                        },
                        "required": ["px", "py", "opposite_px", "opposite_py", "name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_rectangle",
                    "description": "Deletes the rectangle with the given name",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the rectangle",
                            }
                        },
                        "required": ["name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_circle",
                    "description": "Creates and draws a circle with the specified center coordinates and radius. If a name is provided, it will be used to reference the circle.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "center_x": {
                                "type": "number",
                                "description": "The X coordinate of the circle's center"
                            },
                            "center_y": {
                                "type": "number",
                                "description": "The Y coordinate of the circle's center"
                            },
                            "radius": {
                                "type": "number",
                                "description": "The radius of the circle"
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name for the circle"
                            }
                        },
                        "required": ["center_x", "center_y", "radius", "name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_circle",
                    "description": "Deletes the circle with the given name",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the circle"
                            }
                        },
                        "required": ["name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_ellipse",
                    "description": "Creates an ellipse with the specified center point, x-radius, y-radius, and optional rotation angle",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "center_x": {
                                "type": "number",
                                "description": "The x-coordinate of the ellipse's center"
                            },
                            "center_y": {
                                "type": "number",
                                "description": "The y-coordinate of the ellipse's center"
                            },
                            "radius_x": {
                                "type": "number",
                                "description": "The radius of the ellipse in the x-direction (half the width)"
                            },
                            "radius_y": {
                                "type": "number",
                                "description": "The radius of the ellipse in the y-direction (half the height)"
                            },
                            "rotation_angle": {
                                "type": ["number", "null"],
                                "description": "Optional angle in degrees to rotate the ellipse around its center (default: 0)"
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name for the ellipse"
                            }
                        },
                        "required": ["center_x", "center_y", "radius_x", "radius_y", "rotation_angle", "name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_ellipse",
                    "description": "Deletes the ellipse with the given name",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the ellipse"
                            }
                        },
                        "required": ["name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "draw_function",
                    "description": "Plots the given mathematical function on the canvas between the specified left and right bounds.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "function_string": {
                                "type": "string",
                                "description": "The mathematical expression represented as a string, e.g., '2*x + 3'."
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "The name or label for the plotted function. Useful for referencing later."
                            },
                            "left_bound": {
                                "type": ["number", "null"],
                                "description": "The left bound of the interval on which to plot the function."
                            },
                            "right_bound": {
                                "type": ["number", "null"],
                                "description": "The right bound of the interval on which to plot the function."
                            }
                        },
                        "required": ["function_string", "name", "left_bound", "right_bound"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_function",
                    "description": "Removes the plotted mathematical function with the given name from the canvas.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name or label of the function to be deleted."
                            }
                        },
                        "required": ["name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "evaluate_expression",
                    "description": "Evaluates a mathematical expression provided as a string and returns the numerical result. The expression can include variables like x, y; constants like e, pi; mathematical operations and functions like sin, cos, tan, sqrt, log, log10, log2, factorial, asin, acos, atan, sinh, cosh, tanh, exp, abs, pi, e, pow, det, bin, round, ceil, floor, trunc, max, min, sum, gcd, lcm, mean, median, mode, stdev, variance, random, randint.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to be evaluated. Example: '5*x - 1' or 'sin(x)'"
                            },
                            "variables": {
                                "type": ["object", "null"],
                                "description": "Dictionary containing key-value pairs of the variables and values to be substituted in the expression. Example: {'x': 2, 'y': 3}"
                            }
                        },
                        "required": ["expression", "variables"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "convert",
                    "description": "Converts a value from one unit to another",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "number",
                                "description": "The value to be converted"
                            },
                            "from_unit": {
                                "type": "string",
                                "description": "The unit to convert from"
                            },
                            "to_unit": {
                                "type": "string",
                                "description": "The unit to convert to"
                            }
                        },
                        "required": ["value", "from_unit", "to_unit"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "limit",
                    "description": "Computes the limit of a function as it approaches a value",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression represented as a string. Example: 'log(x)^2'."
                            },
                            "variable": {
                                "type": "string",
                                "description": "The variable with respect to which the limit is computed."
                            },
                            "value_to_approach": {
                                "type": "string",
                                "description": "The value the variable approaches."
                            }
                        },
                        "required": ["expression", "variable", "value_to_approach"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "derive",
                    "description": "Computes the derivative of a function with respect to a variable",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression represented as a string. Example: '2*x + 3'."
                            },
                            "variable": {
                                "type": "string",
                                "description": "The variable with respect to which the derivative is computed."
                            }
                        },
                        "required": ["expression", "variable"],
                        "additionalProperties": False
                    }
                }
            },            
            {
                "type": "function",
                "function": {
                    "name": "integrate",
                    "description": "Computes the integral of a function with respect to a variable. Specify the lower and upper bounds only for definite integrals.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression represented as a string. Example: '2*x + 3'"
                            },
                            "variable": {
                                "type": "string",
                                "description": "The variable with respect to which the integral is computed. Example: 'x'"
                            },
                            "lower_bound": {
                                "type": ["number", "null"],
                                "description": "The lower bound of the integral."
                            },
                            "upper_bound": {
                                "type": ["number", "null"],
                                "description": "The upper bound of the integral."
                            }
                        },
                        "required": ["expression", "variable", "lower_bound", "upper_bound"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "simplify",
                    "description": "Simplifies a mathematical expression.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression represented as a string. Example: 'x^2 + 2*x + 1'"
                            }
                        },
                        "required": ["expression"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "expand",
                    "description": "Expands a mathematical expression.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression represented as a string. Example: '(x+1)^2'"
                            }
                        },
                        "required": ["expression"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "factor",
                    "description": "Factors a mathematical expression.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression represented as a string. Example: 'x^2 - 1'"
                            }
                        },
                        "required": ["expression"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "solve",
                    "description": "Solves a mathematical equation for a given variable.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "equation": {
                                "type": "string",
                                "description": "The mathematical equation represented as a string. Example: 'x^2 - 1'"
                            },
                            "variable": {
                                "type": "string",
                                "description": "The variable to solve for. Example: 'x'"
                            }
                        },
                        "required": ["equation", "variable"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "solve_system_of_equations",
                    "description": "Solves a system of mathematical equations.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "equations": {
                                "type": "array",
                                "description": "An array of mathematical equations represented as strings. Example: ['2*x/3 = y', 'x-2 = y']",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": ["equations"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "translate_object",
                    "description": "Translates a drawable object or function by the specified x and y offsets",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The exact name of the object to translate taken from the canvas state"
                            },
                            "x_offset": {
                                "type": "number",
                                "description": "The horizontal translation distance (positive moves right, negative moves left)"
                            },
                            "y_offset": {
                                "type": "number",
                                "description": "The vertical translation distance (positive moves up, negative moves down)"
                            }
                        },
                        "required": ["name", "x_offset", "y_offset"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "rotate_object",
                    "description": "Rotates a drawable object around its center by the specified angle",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the object to rotate"
                            },
                            "angle": {
                                "type": "number",
                                "description": "The angle in degrees to rotate the object (positive for counterclockwise)"
                            }
                        },
                        "required": ["name", "angle"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_workspace",
                    "description": "Saves the current workspace state to a file. If no name is provided, saves to the current workspace file with timestamp. The workspace name MUST only contain alphanumeric characters, underscores, or hyphens (no spaces, dots, slashes, or other special characters).",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name for the workspace. Must contain only alphanumeric characters, underscores, or hyphens (e.g., 'my_workspace', 'workspace-1', 'test123'). If not provided, saves to current workspace."
                            }
                        },
                        "required": ["name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "load_workspace",
                    "description": "Loads a workspace from a file. If no name is provided, loads the (most recent) current workspace. The workspace name MUST only contain alphanumeric characters, underscores, or hyphens (no spaces, dots, slashes, or other special characters).",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name of the workspace to load. Must contain only alphanumeric characters, underscores, or hyphens (e.g., 'my_workspace', 'workspace-1', 'test123'). If not provided, loads current workspace."
                            }
                        },
                        "required": ["name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_workspaces",
                    "description": "Lists all saved workspaces. Only shows workspaces with valid names (containing only alphanumeric characters, underscores, or hyphens).",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_workspace",
                    "description": "Delete a workspace by name. The workspace name MUST only contain alphanumeric characters, underscores, or hyphens (no spaces, dots, slashes, or other special characters).",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the workspace to delete. Must contain only alphanumeric characters, underscores, or hyphens (e.g., 'my_workspace', 'workspace-1', 'test123')."
                            }
                        },
                        "required": ["name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_colored_area",
                    "description": "Creates a colored area between two drawables (functions, segments, or a function and a segment). If only one drawable is provided, the area will be between that drawable and the x-axis.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "drawable1_name": {
                                "type": "string",
                                "description": "Name of the first drawable (function or segment). Use 'x_axis' for the x-axis."
                            },
                            "drawable2_name": {
                                "type": ["string", "null"],
                                "description": "Optional name of the second drawable (function or segment). Use 'x_axis' for the x-axis. If not provided, area will be between drawable1 and x-axis."
                            },
                            "left_bound": {
                                "type": ["number", "null"],
                                "description": "Optional left bound for function areas. Only used when at least one drawable is a function."
                            },
                            "right_bound": {
                                "type": ["number", "null"],
                                "description": "Optional right bound for function areas. Only used when at least one drawable is a function."
                            },
                            "color": {
                                "type": ["string", "null"],
                                "description": "Optional color for the area. Default is 'lightblue'."
                            },
                            "opacity": {
                                "type": ["number", "null"],
                                "description": "Optional opacity for the area between 0 and 1. Default is 0.3."
                            }
                        },
                        "required": ["drawable1_name", "drawable2_name", "left_bound", "right_bound", "color", "opacity"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_colored_area",
                    "description": "Deletes a colored area by its name",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the colored area to delete"
                            }
                        },
                        "required": ["name"],
                        "additionalProperties": False
                    }
                }
            },
            # START ANGLE FUNCTIONS
            {
                "type": "function",
                "function": {
                    "name": "create_angle_by_points",
                    "description": "Creates an angle defined by three points: a common vertex and one point on each arm. Segments will be created if they don't exist.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vx": {"type": "number", "description": "X-coordinate of the common vertex."},
                            "vy": {"type": "number", "description": "Y-coordinate of the common vertex."},
                            "p1x": {"type": "number", "description": "X-coordinate of a point on the first arm."},
                            "p1y": {"type": "number", "description": "Y-coordinate of a point on the first arm."},
                            "p2x": {"type": "number", "description": "X-coordinate of a point on the second arm."},
                            "p2y": {"type": "number", "description": "Y-coordinate of a point on the second arm."},
                            "label": {"type": ["string", "null"], "description": "Optional label for the angle."},
                            "color": {"type": ["string", "null"], "description": "Optional color for the angle (e.g., 'red', '#FF0000')."},
                            "angle_name": {"type": ["string", "null"], "description": "Optional specific name for the angle."}
                        },
                        "required": ["vx", "vy", "p1x", "p1y", "p2x", "p2y", "label", "color", "angle_name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "remove_angle",
                    "description": "Removes an angle by its name.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The name of the angle to remove."}
                        },
                        "required": ["name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_angle_properties",
                    "description": "Updates the label and/or color of an existing angle.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The name of the angle to update."},
                            "new_label": {"type": ["string", "null"], "description": "Optional new label for the angle. If null, label is not changed."},
                            "new_color": {"type": ["string", "null"], "description": "Optional new color for the angle. If null, color is not changed."}
                        },
                        "required": ["name", "new_label", "new_color"],
                        "additionalProperties": False
                    }
                }
            }
            # END ANGLE FUNCTIONS
        ]