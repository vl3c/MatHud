FUNCTIONS = [
            {
                "type": "function",
                "function": {
                    "name": "reset_canvas",
                    "description": "Resets the canvas zoom and offset",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clear_canvas",
                    "description": "Clears the canvas by deleting all drawable objects",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "undo",
                    "description": "Undoes the last action on the canvas",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "redo",
                    "description": "Redoes the last action on the canvas",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_tests",
                    "description": "Runs the test suite for the canvas",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_point",
                    "description": "Creates and draws a point at the given coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "The X coordinate of the point",
                            },
                            "y": {
                                "type": "number",
                                "description": "The Y coordinate of the point",
                            },
                            "name": {
                                "type": "string",
                                "description": "The name of the point",
                            }
                        },
                        "required": ["x", "y"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_point",
                    "description": "Deletes the point with the given coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "The X coordinate of the point",
                            },
                            "y": {
                                "type": "number",
                                "description": "The Y coordinate of the point",
                            }
                        },
                        "required": ["x", "y"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_segment",
                    "description": "Creates and draws a segment at the given coordinates for two points. If only a name is given, search for appropriate point coordinates in the canvas state.",
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
                            "name": {
                                "type": "string",
                                "description": "The name of the segment",
                            }
                        },
                        "required": ["x1", "y1", "x2", "y2"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_segment",
                    "description": "Deletes the segment found at the given coordinates for two points. If only a name is given, search for appropriate point coordinates in the canvas state.",
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
                            }
                        },
                        "required": ["x1", "y1", "x2", "y2"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_vector",
                    "description": "Creates and draws a vector at the given coordinates for two points called origin and tip. If only a name is given, search for appropriate point coordinates in the canvas state.",
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
                            },
                            "name": {
                                "type": "string",
                                "description": "The name of the vector",
                            }
                        },
                        "required": ["origin_x", "origin_y", "tip_x", "tip_y"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_vector",
                    "description": "Deletes the vector found at the given coordinates for two points called origin and tip. If only a name is given, search for appropriate point coordinates in the canvas state.",
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
                        "required": ["origin_x", "origin_y", "tip_x", "tip_y"]
                    }
                }
            },            
            {
                "type": "function",
                "function": {
                    "name": "create_triangle",
                    "description": "Creates and draws a triangle at the given coordinates for three points. If only a name is given, search for appropriate point coordinates in the canvas state.",
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
                            },
                            "name": {
                                "type": "string",
                                "description": "The name of the triangle",
                            }
                        },
                        "required": ["x1", "y1", "x2", "y2", "x3", "y3"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_triangle",
                    "description": "Deletes the triangle found at the given coordinates for three points. If only a name is given, search for appropriate point coordinates in the canvas state.",
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
                        "required": ["x1", "y1", "x2", "y2", "x3", "y3"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_rectangle",
                    "description": "Creates and draws a rectangle at the given coordinates for two diagonal points. If only a name is given, search for appropriate point coordinates in the canvas state.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "diagonal_p1_x": {
                                "type": "number",
                                "description": "The X coordinate of the origin point",
                            },
                            "diagonal_p1_y": {
                                "type": "number",
                                "description": "The Y coordinate of the origin point",
                            },
                            "diagonal_p2_x": {
                                "type": "number",
                                "description": "The X coordinate of the tip point",
                            },
                            "diagonal_p2_y": {
                                "type": "number",
                                "description": "The Y coordinate of the tip point",
                            },
                            "name": {
                                "type": "string",
                                "description": "The name of the rectangle",
                            }
                        },
                        "required": ["diagonal_p1_x", "diagonal_p1_y", "diagonal_p2_x", "diagonal_p2_y"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_rectangle",
                    "description": "Deletes the rectangle with the given name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the rectangle",
                            }
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_circle",
                    "description": "Creates and draws a circle with the specified center coordinates and radius. If only a name is given, search for appropriate point coordinates in the canvas state.",
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
                                "type": "string",
                                "description": "The name of the circle"
                            }
                        },
                        "required": ["center_x", "center_y", "radius"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_circle",
                    "description": "Deletes the circle with the given name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the circle"
                            }
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_ellipse",
                    "description": "Creates and draws an ellipse with the specified center coordinates and x and y radii. If only a name is given, search for appropriate point coordinates in the canvas state.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "center_x": {
                                "type": "number",
                                "description": "The X coordinate of the ellipse's center"
                            },
                            "center_y": {
                                "type": "number",
                                "description": "The Y coordinate of the ellipse's center"
                            },
                            "radius_x": {
                                "type": "number",
                                "description": "The X radius of the ellipse"
                            },
                            "radius_y": {
                                "type": "number",
                                "description": "The Y radius of the ellipse"
                            },
                            "name": {
                                "type": "string",
                                "description": "The name of the ellipse"
                            }
                        },
                        "required": ["center_x", "center_y", "radius_x", "radius_y"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_ellipse",
                    "description": "Deletes the ellipse with the given name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the ellipse"
                            }
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "draw_math_function",
                    "description": "Plots the given mathematical function on the canvas between the specified left and right bounds.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "function_string": {
                                "type": "string",
                                "description": "The mathematical expression represented as a string, e.g., '2*x + 3'."
                            },
                            "name": {
                                "type": "string",
                                "description": "The name or label for the plotted function. Useful for referencing later."
                            },
                            "left_bound": {
                                "type": "number",
                                "description": "The left bound of the interval on which to plot the function."
                            },
                            "right_bound": {
                                "type": "number",
                                "description": "The right bound of the interval on which to plot the function."
                            },
                        },
                        "required": ["function_string", "name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_math_function",
                    "description": "Removes the plotted function with the given name from the canvas.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name or label of the function to be deleted."
                            }
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "evaluate_expression",
                    "description": "Evaluates a mathematical expression provided as a string and returns the numerical result",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to be evaluated"
                            },
                            "variables": {
                                "type": "object",
                                "description": "Key-value pairs of the variables and values to be substituted in the expression. Examples: {'x': 2}, {'x': 2, 'y': 3}"
                            },
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "convert",
                    "description": "Converts a value from one unit to another",
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
                        "required": ["value", "from_unit", "to_unit"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "limit",
                    "description": "Computes the limit of a function as it approaches a value",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression represented as a string, e.g., '2*x + 3'."
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
                        "required": ["expression", "variable", "value_to_approach"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "derivative",
                    "description": "Computes the derivative of a function with respect to a variable",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression represented as a string, e.g., '2*x + 3'."
                            },
                            "variable": {
                                "type": "string",
                                "description": "The variable with respect to which the derivative is computed."
                            }
                        },
                        "required": ["expression", "variable"]
                    }
                }
            },            
            {
                "type": "function",
                "function": {
                    "name": "integral",
                    "description": "Computes the integral of a function with respect to a variable. Specify the lower and upper bounds only for definite integrals.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression represented as a string, e.g., '2*x + 3'."
                            },
                            "variable": {
                                "type": "string",
                                "description": "The variable with respect to which the integral is computed."
                            },
                            "lower_bound": {
                                "type": "number",
                                "description": "The lower bound of the integral."
                            },
                            "upper_bound": {
                                "type": "number",
                                "description": "The upper bound of the integral."
                            }
                        },
                        "required": ["expression", "variable"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "simplify",
                    "description": "Simplifies a mathematical expression. Example: x^2 + 2*x + 1 simplified is (x+1)^2",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression represented as a string."
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "expand",
                    "description": "Expands a mathematical expression. Example: (x+1)^2 expanded is x^2 + 2*x + 1",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression represented as a string."
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "factor",
                    "description": "Factors a mathematical expression. Example: x^2 - 1 factored is (x-1)*(x+1)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression represented as a string."
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "solve",
                    "description": "Solves a mathematical equation for a given variable. Example: x^2 - 1 = 0 solved is [-1, 1] and solve('x^3 + 1', 'x') is [-1, (1/2)*i*sqrt(3)+1/2, (-1/2)*i*sqrt(3)+1/2]",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "equation": {
                                "type": "string",
                                "description": "The mathematical equation represented as a string."
                            },
                            "variable": {
                                "type": "string",
                                "description": "The variable to solve for."
                            }
                        },
                        "required": ["equation", "variable"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "solve_system_of_equations",
                    "description": "Solves a system of mathematical equations. Example: solve_system_of_equations(['x+y=4', 'x-y=2']) is [3, 1]",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "equations": {
                                "type": "array",
                                "description": "An array of mathematical equations represented as strings.",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": ["equations"]
                    }
                }
            }        
        ]