FUNCTIONS = [
            {
                "name": "make_multiple_function_calls",
                "description": "Executes a sequence of functions in the order they are provided. Functions are described below. Each function call in the 'function_calls' array must include the name of the function and the corresponding arguments, separated into 'name' and 'arguments' keys.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_calls": {
                            "type": "array",
                            "description": "An array of function call objects, each containing the 'name' of the function and the 'arguments' for a specific function call. Example: [{'name': 'create_point', 'arguments': {'x': 1, 'y': 1, 'name': 'A'}}, {'name': 'create_point', 'arguments': {'x': 2, 'y': 2}}]",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The name of the function to be called. Must match one of the available functions."
                                    },
                                    "arguments": {
                                        "type": "object",
                                        "description": "The arguments to be passed to the function, represented as key-value pairs."
                                    }
                                },
                                "required": ["name", "arguments"]
                            }
                        }
                    },
                    "required": ["function_calls"]
                }
            },
            {
                "name": "reset_canvas",
                "description": "Resets the canvas zoom and offset",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "clear_canvas",
                "description": "Clears the canvas by deleting all drawable objects",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
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
            },
            {
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
            },
            {
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
            },
            {
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
            },
            {
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
            },
            {
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
            },
            {
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
            },
            {
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
            },
            {
                "name": "create_rectangle",
                "description": "Creates and draws a rectangle at the given coordinates for two diagonal points. If only a name is given, search for appropriate point coordinates in the canvas state.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "p_x": {
                            "type": "number",
                            "description": "The X coordinate of the origin point",
                        },
                        "p_y": {
                            "type": "number",
                            "description": "The Y coordinate of the origin point",
                        },
                        "opposite_x": {
                            "type": "number",
                            "description": "The X coordinate of the tip point",
                        },
                        "opposite_y": {
                            "type": "number",
                            "description": "The Y coordinate of the tip point",
                        },
                        "name": {
                            "type": "string",
                            "description": "The name of the rectangle",
                        }
                    },
                    "required": ["p_x", "p_y", "opposite_x", "opposite_y"]
                }
            },
            {
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
            },
            {
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
            },
            {
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
            },
            {
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
            },
            {
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
            },
            {
                "name": "draw_math_function",
                "description": "Plots the given mathematical function on the canvas between the specified left and right bounds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_string": {
                            "type": "string",
                            "description": "The mathematical expression represented as a string, e.g., '2*x + 3'."
                        },
                        "left_bound": {
                            "type": "number",
                            "description": "The initial or starting x-value (left bound) where the function begins plotting."
                        },
                        "right_bound": {
                            "type": "number",
                            "description": "The final x-value (right bound) where the function stops plotting."
                        },
                        "name": {
                            "type": "string",
                            "description": "The name or label for the plotted function. Useful for referencing later."
                        }
                    },
                    "required": ["function_string", "left_bound", "right_bound"]
                }
            },
            {
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
            },
        ]