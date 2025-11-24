"""
MatHud AI Function Definitions

Comprehensive set of 40+ AI function definitions for mathematical operations and canvas manipulation.
Provides OpenAI tool calling schema for geometric shapes, calculations, transformations, and workspace management.

Categories:
    - Canvas Operations: reset, clear, undo, redo, run_tests
    - Geometric Shapes: points, segments, vectors, triangles, rectangles, circles, ellipses, angles
    - Mathematical Functions: plotting, colored areas, bounded regions
    - Calculations: expressions, trigonometry, algebra, calculus
    - Transformations: translate, rotate, scale geometric objects
    - Workspace Management: save, load, list, delete workspaces

Dependencies:
    - OpenAI Function Calling: Structured function definitions with strict schema validation
    - JSON Schema: Parameter validation and type checking for all function arguments
"""

from __future__ import annotations

from typing import Any, Dict, List

from importlib import import_module

from static.mirror_client_modules import (
    POLYGON_SUBTYPES_MODULE,
    SERVER_CONSTANTS_MODULE,
    ensure_client_constants_available,
    ensure_polygon_subtypes_available,
)

ensure_client_constants_available()
_client_constants = import_module(SERVER_CONSTANTS_MODULE)

LABEL_TEXT_MAX_LENGTH: int = getattr(_client_constants, "label_text_max_length", 160)
LABEL_LINE_WRAP_THRESHOLD: int = getattr(_client_constants, "label_line_wrap_threshold", 40)
DEFAULT_CIRCLE_ARC_COLOR: str = getattr(
    _client_constants,
    "DEFAULT_CIRCLE_ARC_COLOR",
    getattr(_client_constants, "default_color", "black"),
)


ensure_polygon_subtypes_available()
_polygon_subtypes_module = import_module(POLYGON_SUBTYPES_MODULE)
_TriangleSubtype = getattr(_polygon_subtypes_module, "TriangleSubtype")
_QuadrilateralSubtype = getattr(_polygon_subtypes_module, "QuadrilateralSubtype")
TRIANGLE_SUBTYPE_VALUES = _TriangleSubtype.values()
QUADRILATERAL_SUBTYPE_VALUES = _QuadrilateralSubtype.values()
POLYGON_SUBTYPE_VALUES = sorted(set(TRIANGLE_SUBTYPE_VALUES + QUADRILATERAL_SUBTYPE_VALUES))


FunctionDefinition = Dict[str, Any]


FUNCTIONS: List[Dict[str, Any]] = [
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
                    "name": "zoom_to_bounds",
                    "description": "Fits the viewport to the specified math-space bounds",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "left_bound": {
                                "type": "number",
                                "description": "Left math bound to include in view"
                            },
                            "right_bound": {
                                "type": "number",
                                "description": "Right math bound to include in view"
                            },
                            "top_bound": {
                                "type": "number",
                                "description": "Top math bound to include in view"
                            },
                            "bottom_bound": {
                                "type": "number",
                                "description": "Bottom math bound to include in view"
                            }
                        },
                        "required": ["left_bound", "right_bound", "top_bound", "bottom_bound"],
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
                            "color": {
                                "type": ["string", "null"],
                                "description": "Optional color for the point"
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name for the point. If provided, the first available letter from this name will be used."
                            }
                        },
                        "required": ["x", "y", "color", "name"],
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
                    "name": "update_point",
                    "description": "Updates the name, color, or position of a solitary point without recreating it. Provide at least one property to change.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "point_name": {
                                "type": "string",
                                "description": "Existing name of the point to edit"
                            },
                            "new_name": {
                                "type": ["string", "null"],
                                "description": "Optional new name for the point"
                            },
                            "new_x": {
                                "type": ["number", "null"],
                                "description": "Optional new x-coordinate (requires new_y)"
                            },
                            "new_y": {
                                "type": ["number", "null"],
                                "description": "Optional new y-coordinate (requires new_x)"
                            },
                            "new_color": {
                                "type": ["string", "null"],
                                "description": "Optional new display color for the point"
                            }
                        },
                        "required": ["point_name", "new_name", "new_x", "new_y", "new_color"],
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
                            "color": {
                                "type": ["string", "null"],
                                "description": "Optional color for the segment"
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name for the segment. If provided, the first two available letters will be used to name the endpoints."
                            }
                        },
                        "required": ["x1", "y1", "x2", "y2", "color", "name"],
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
                    "name": "update_segment",
                    "description": "Updates editable properties of an existing segment (currently just color). Provide null for fields that should remain unchanged.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Existing name of the segment to edit"
                            },
                            "new_color": {
                                "type": ["string", "null"],
                                "description": "Optional new color for the segment"
                            }
                        },
                        "required": ["name", "new_color"],
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
                            "color": {
                                "type": ["string", "null"],
                                "description": "Optional color for the vector"
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name for the vector. If provided, the first two available letters will be used to name the origin and tip points."
                            }
                        },
                        "required": ["origin_x", "origin_y", "tip_x", "tip_y", "color", "name"],
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
                    "name": "update_vector",
                    "description": "Updates editable properties of an existing vector (currently just color). Provide null for fields that should remain unchanged.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Existing name of the vector to edit"
                            },
                            "new_color": {
                                "type": ["string", "null"],
                                "description": "Optional new color for the vector"
                            }
                        },
                        "required": ["name", "new_color"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_polygon",
                    "description": "Creates a polygon from ordered vertex coordinates. For rectangle and square types, coordinates are normalized through the canonicalizer so near-rectangles snap into valid rectangles. Triangle inputs can optionally request canonicalization toward special subtypes such as equilateral or right triangles.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vertices": {
                                "type": "array",
                                "minItems": 3,
                                "description": "Ordered list of polygon vertex coordinates.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "x": {
                                            "type": "number",
                                            "description": "X coordinate of the vertex."
                                        },
                                        "y": {
                                            "type": "number",
                                            "description": "Y coordinate of the vertex."
                                        }
                                    },
                                    "required": ["x", "y"],
                                    "additionalProperties": False
                                }
                            },
                            "polygon_type": {
                                "type": ["string", "null"],
                                "description": "Optional polygon classification (triangle, quadrilateral, pentagon, or hexagon).",
                                "enum": ["triangle", "quadrilateral", "pentagon", "hexagon", None],
                            },
                            "color": {
                                "type": ["string", "null"],
                                "description": "Optional stroke color for the polygon edges."
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name for the polygon. Letters are reused to label vertices."
                            },
                            "subtype": {
                                "type": ["string", "null"],
                                "description": "Optional polygon subtype hint. Triangles support equilateral, isosceles, right, right_isosceles. Quadrilaterals support rectangle, square, parallelogram, rhombus, kite, trapezoid, isosceles_trapezoid, right_trapezoid.",
                                "enum": POLYGON_SUBTYPE_VALUES + [None],
                            }
                        },
                        "required": ["vertices", "polygon_type", "color", "name", "subtype"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_polygon",
                    "description": "Deletes a polygon by name or by matching a set of vertex coordinates. Specify polygon_type to limit the search.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "polygon_type": {
                                "type": ["string", "null"],
                                "description": "Optional polygon classification (triangle, quadrilateral, rectangle, square, pentagon, or hexagon)."
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Existing name of the polygon to delete."
                            },
                            "vertices": {
                                "type": "array",
                                "minItems": 3,
                                "description": "Ordered list of polygon vertex coordinates used for lookup.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "x": {
                                            "type": "number",
                                            "description": "X coordinate of the vertex."
                                        },
                                        "y": {
                                            "type": "number",
                                            "description": "Y coordinate of the vertex."
                                        }
                                    },
                                    "required": ["x", "y"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["polygon_type", "name", "vertices"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_polygon",
                    "description": "Updates editable properties of an existing polygon (currently just color). Provide null for fields that should remain unchanged.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "polygon_name": {
                                "type": "string",
                                "description": "Existing name of the polygon to edit."
                            },
                            "polygon_type": {
                                "type": ["string", "null"],
                                "description": "Optional polygon classification to disambiguate the lookup."
                            },
                            "new_color": {
                                "type": ["string", "null"],
                                "description": "Optional new color for the polygon edges."
                            }
                        },
                        "required": ["polygon_name", "polygon_type", "new_color"],
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
                            "color": {
                                "type": ["string", "null"],
                                "description": "Optional color to assign to the circle"
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name for the circle"
                            }
                        },
                        "required": ["center_x", "center_y", "radius", "color", "name"],
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
                    "name": "update_circle",
                    "description": "Updates editable properties of an existing circle (color or center position). Provide null for fields to keep them unchanged.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Existing name of the circle to edit."
                            },
                            "new_color": {
                                "type": ["string", "null"],
                                "description": "Optional new color for the circle."
                            },
                            "new_center_x": {
                                "type": ["number", "null"],
                                "description": "Optional new x-coordinate for the circle center (requires y value when provided)."
                            },
                            "new_center_y": {
                                "type": ["number", "null"],
                                "description": "Optional new y-coordinate for the circle center (requires x value when provided)."
                            }
                        },
                        "required": ["name", "new_color", "new_center_x", "new_center_y"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_circle_arc",
                    "description": "Creates a circle arc between two points. When explicit center/radius data is supplied or an existing circle is referenced, all provided endpoints are projected onto that circle along their originating rays. Alternatively, provide three points plus 'center_point_choice' to indicate which point represents the center; the remaining two points define the arc and are projected onto the derived circle.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "point1_x": {"type": "number", "description": "Reference X coordinate for the first arc point (snapped to the circle when center/radius are provided)"},
                            "point1_y": {"type": "number", "description": "Reference Y coordinate for the first arc point (snapped to the circle when center/radius are provided)"},
                            "point2_x": {"type": "number", "description": "Reference X coordinate for the second arc point (snapped to the circle when center/radius are provided)"},
                            "point2_y": {"type": "number", "description": "Reference Y coordinate for the second arc point (snapped to the circle when center/radius are provided)"},
                            "point1_name": {"type": ["string", "null"], "description": "Optional name for the first arc point"},
                            "point2_name": {"type": ["string", "null"], "description": "Optional name for the second arc point"},
                            "point3_x": {"type": ["number", "null"], "description": "Optional reference X coordinate for a third point when deriving the circle from three points"},
                            "point3_y": {"type": ["number", "null"], "description": "Optional reference Y coordinate for a third point when deriving the circle from three points"},
                            "point3_name": {"type": ["string", "null"], "description": "Optional name for the third point (used when deriving the circle from three points)"},
                            "center_point_choice": {"type": ["string", "null"], "description": "Optional selector ('point1', 'point2', or 'point3') indicating which provided point should be treated as the circle center"},
                            "circle_name": {"type": ["string", "null"], "description": "Existing circle to attach the arc to"},
                            "center_x": {"type": ["number", "null"], "description": "Circle center x-coordinate when defining a standalone arc"},
                            "center_y": {"type": ["number", "null"], "description": "Circle center y-coordinate when defining a standalone arc"},
                            "radius": {"type": ["number", "null"], "description": "Circle radius when defining a standalone arc"},
                            "use_major_arc": {"type": "boolean", "description": "True to draw the major arc, False for the minor arc"},
                            "arc_name": {"type": ["string", "null"], "description": "Optional custom arc name"},
                            "color": {"type": ["string", "null"], "description": f"Optional CSS color for the arc (defaults to {DEFAULT_CIRCLE_ARC_COLOR})"}
                        },
                        "required": [
                            "point1_x",
                            "point1_y",
                            "point2_x",
                            "point2_y",
                            "point1_name",
                            "point2_name",
                            "point3_x",
                            "point3_y",
                            "point3_name",
                            "center_point_choice",
                            "circle_name",
                            "center_x",
                            "center_y",
                            "radius",
                            "use_major_arc",
                            "arc_name",
                            "color",
                        ],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_circle_arc",
                    "description": "Deletes a circle arc by name.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the circle arc to delete"}
                        },
                        "required": ["name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_circle_arc",
                    "description": "Updates editable properties of an existing circle arc (color or major/minor toggle). Provide null for fields to keep them unchanged.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Existing name of the arc to edit"},
                            "new_color": {"type": ["string", "null"], "description": "Optional new color for the arc"},
                            "use_major_arc": {"type": ["boolean", "null"], "description": "Set to true for the major arc, false for the minor arc"}
                        },
                        "required": [
                            "name",
                            "new_color",
                            "use_major_arc"
                        ],
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
                            "color": {
                                "type": ["string", "null"],
                                "description": "Optional color for the ellipse"
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name for the ellipse"
                            }
                        },
                        "required": ["center_x", "center_y", "radius_x", "radius_y", "rotation_angle", "color", "name"],
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
                    "name": "update_ellipse",
                    "description": "Updates editable properties of an existing ellipse (color, radii, rotation, or center). Provide null for fields that should remain unchanged.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Existing name of the ellipse to edit."
                            },
                            "new_color": {
                                "type": ["string", "null"],
                                "description": "Optional new color for the ellipse."
                            },
                            "new_radius_x": {
                                "type": ["number", "null"],
                                "description": "Optional new horizontal radius (requires ellipse to be solitary)."
                            },
                            "new_radius_y": {
                                "type": ["number", "null"],
                                "description": "Optional new vertical radius (requires ellipse to be solitary)."
                            },
                            "new_rotation_angle": {
                                "type": ["number", "null"],
                                "description": "Optional new rotation angle in degrees."
                            },
                            "new_center_x": {
                                "type": ["number", "null"],
                                "description": "Optional new x-coordinate for the center (requires y value when provided)."
                            },
                            "new_center_y": {
                                "type": ["number", "null"],
                                "description": "Optional new y-coordinate for the center (requires x value when provided)."
                            }
                        },
                        "required": ["name", "new_color", "new_radius_x", "new_radius_y", "new_rotation_angle", "new_center_x", "new_center_y"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_label",
                        "description": (
                            "Creates a text label anchored at a math-space coordinate "
                            f"(max {LABEL_TEXT_MAX_LENGTH} chars, wraps every {LABEL_LINE_WRAP_THRESHOLD} chars)"
                        ),
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "Math-space X coordinate for the label anchor"
                            },
                            "y": {
                                "type": "number",
                                "description": "Math-space Y coordinate for the label anchor"
                            },
                            "text": {
                                "type": "string",
                                "description": "Label text content; lines wrap after 40 characters",
                                "maxLength": 160
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional label name used for later updates"
                            },
                            "color": {
                                "type": ["string", "null"],
                                "description": "Optional CSS color for the label text"
                            },
                            "font_size": {
                                "type": ["number", "null"],
                                "description": "Optional font size in pixels"
                            },
                            "rotation_degrees": {
                                "type": ["number", "null"],
                                "description": "Optional angle in degrees to rotate the label text"
                            }
                        },
                        "required": [
                            "x",
                            "y",
                            "text",
                            "name",
                            "color",
                            "font_size",
                            "rotation_degrees",
                        ],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_label",
                    "description": "Deletes an existing label by name",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the label to delete"
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
                    "name": "update_label",
                    "description": "Updates editable properties of an existing label (text, color, position, font size, rotation). Provide null for fields that should remain unchanged.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Existing name of the label to edit"
                            },
                            "new_text": {
                                "type": ["string", "null"],
                                "description": "Optional replacement text for the label"
                            },
                            "new_x": {
                                "type": ["number", "null"],
                                "description": "Optional new x-coordinate (requires new_y)"
                            },
                            "new_y": {
                                "type": ["number", "null"],
                                "description": "Optional new y-coordinate (requires new_x)"
                            },
                            "new_color": {
                                "type": ["string", "null"],
                                "description": "Optional new text color"
                            },
                            "new_font_size": {
                                "type": ["number", "null"],
                                "description": "Optional new font size in math-space units"
                            },
                            "new_rotation_degrees": {
                                "type": ["number", "null"],
                                "description": "Optional rotation angle in degrees"
                            }
                        },
                        "required": [
                            "name",
                            "new_text",
                            "new_x",
                            "new_y",
                            "new_color",
                            "new_font_size",
                            "new_rotation_degrees"
                        ],
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
                            },
                            "color": {
                                "type": ["string", "null"],
                                "description": "Optional color for the plotted function."
                            }
                        },
                        "required": ["function_string", "name", "left_bound", "right_bound", "color"],
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
                    "name": "update_function",
                    "description": "Updates editable properties of an existing plotted function (color and/or bounds). Provide null for fields to leave them unchanged.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Existing name of the function to edit."
                            },
                            "new_color": {
                                "type": ["string", "null"],
                                "description": "Optional new color for the function plot."
                            },
                            "new_left_bound": {
                                "type": ["number", "null"],
                                "description": "Optional new left plotting bound."
                            },
                            "new_right_bound": {
                                "type": ["number", "null"],
                                "description": "Optional new right plotting bound."
                            }
                        },
                        "required": ["name", "new_color", "new_left_bound", "new_right_bound"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "evaluate_expression",
                    "description": "Evaluates a mathematical expression provided as a string and returns the numerical result. The expression can include variables like x, y; constants like e, pi; mathematical operations and functions like sin, cos, tan, sqrt, log, log10, log2, factorial, arrangements, permutations, combinations, asin, acos, atan, sinh, cosh, tanh, exp, abs, pi, e, pow, det, bin, round, ceil, floor, trunc, max, min, sum, gcd, lcm, mean, median, mode, stdev, variance, random, randint.",
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
                    "name": "evaluate_linear_algebra_expression",
                    "description": "Evaluates a linear algebra expression using named matrices, vectors, or scalars. Supports operations such as addition, subtraction, multiplication, transpose and inverse.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "objects": {
                                "type": "array",
                                "minItems": 1,
                                "description": "List of named linear algebra objects available to the expression evaluator.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Identifier used to reference the object inside the expression. Must start with a letter or underscore."
                                        },
                                        "value": {
                                            "description": "Scalar, vector, or matrix definition for the object.",
                                            "anyOf": [
                                                {
                                                    "type": "number"
                                                },
                                                {
                                                    "type": "array",
                                                    "minItems": 1,
                                                    "items": {
                                                        "type": "number"
                                                    }
                                                },
                                                {
                                                    "type": "array",
                                                    "minItems": 1,
                                                    "items": {
                                                        "type": "array",
                                                        "minItems": 1,
                                                        "items": {
                                                            "type": "number"
                                                        }
                                                    }
                                                }
                                            ]
                                        }
                                    },
                                    "required": ["name", "value"],
                                    "additionalProperties": False
                                }
                            },
                            "expression": {
                                "type": "string",
                                "description": "Math.js compatible expression composed of the provided object names and supported linear algebra functions. Example: 'A + B' or 'inv(A) * b'."
                            }
                        },
                        "required": ["objects", "expression"],
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
                    "name": "create_closed_shape_colored_area",
                    "description": "Fill the interior of a closed shape (triangle, rectangle, polygonal loop, circle, ellipse, or a round shape clipped by a single segment). Provide the relevant identifiers for the shape you want to fill.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "triangle_name": {
                                "type": ["string", "null"],
                                "description": "Name of an existing triangle to fill."
                            },
                            "rectangle_name": {
                                "type": ["string", "null"],
                                "description": "Name of an existing rectangle to fill."
                            },
                            "polygon_segment_names": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                                "description": "List of segment names that form a closed polygon loop (at least three segments)."
                            },
                            "circle_name": {
                                "type": ["string", "null"],
                                "description": "Name of the circle to fill or to use with a chord segment."
                            },
                            "ellipse_name": {
                                "type": ["string", "null"],
                                "description": "Name of the ellipse to fill or to use with a chord segment."
                            },
                            "chord_segment_name": {
                                "type": ["string", "null"],
                                "description": "Segment name that serves as the chord/clip when creating a circle or ellipse segment region."
                            },
                            "arc_clockwise": {
                                "type": ["boolean", "null"],
                                "description": "Set to true to trace the arc clockwise when using a round shape with a chord segment. Default is false (counter-clockwise)."
                            },
                            "resolution": {
                                "type": ["number", "null"],
                                "description": "Number of samples used to approximate curved boundaries. Defaults to 96."
                            },
                            "color": {
                                "type": ["string", "null"],
                                "description": "Optional color for the filled area. Default is 'lightblue'."
                            },
                            "opacity": {
                                "type": ["number", "null"],
                                "description": "Optional opacity between 0 and 1. Default is 0.3."
                            }
                        },
                        "required": [
                            "triangle_name",
                            "rectangle_name",
                            "polygon_segment_names",
                            "circle_name",
                            "ellipse_name",
                            "chord_segment_name",
                            "arc_clockwise",
                            "resolution",
                            "color",
                            "opacity"
                        ],
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
            {
                "type": "function",
                "function": {
                    "name": "update_colored_area",
                    "description": "Updates editable properties of an existing colored area (color, opacity, and for function-bounded areas, optional left/right bounds). Provide null for fields that should remain unchanged.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Existing name of the colored area to edit."
                            },
                            "new_color": {
                                "type": ["string", "null"],
                                "description": "Optional new color for the area."
                            },
                            "new_opacity": {
                                "type": ["number", "null"],
                                "description": "Optional new opacity between 0 and 1."
                            },
                            "new_left_bound": {
                                "type": ["number", "null"],
                                "description": "Optional new left bound (functions-bounded areas only)."
                            },
                            "new_right_bound": {
                                "type": ["number", "null"],
                                "description": "Optional new right bound (functions-bounded areas only)."
                            }
                        },
                        "required": ["name", "new_color", "new_opacity", "new_left_bound", "new_right_bound"],
                        "additionalProperties": False
                    }
                }
            },
            # START ANGLE FUNCTIONS
            {
                "type": "function",
                "function": {
                    "name": "create_angle",
                    "description": "Creates and draws an angle defined by three points. The first point (vx, vy) is the common vertex, and the other two points (p1x, p1y and p2x, p2y) define the angle's arms. For example, in an angle ABC, (vx, vy) would be the coordinates of point B. The angle's visual representation (arc and degree value) will be drawn. Segments forming the angle will be created if they don't exist.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vx": {
                                "type": "number",
                                "description": "The X coordinate of the common vertex point (e.g., point B in an angle ABC)."
                            },
                            "vy": {
                                "type": "number",
                                "description": "The Y coordinate of the common vertex point (e.g., point B in an angle ABC)."
                            },
                            "p1x": {
                                "type": "number",
                                "description": "The X coordinate of the first arm point."
                            },
                            "p1y": {
                                "type": "number",
                                "description": "The Y coordinate of the first arm point."
                            },
                            "p2x": {
                                "type": "number",
                                "description": "The X coordinate of the second arm point."
                            },
                            "p2y": {
                                "type": "number",
                                "description": "The Y coordinate of the second arm point."
                            },
                            "color": {
                                "type": [ "string", "null" ],
                                "description": "Optional color for the angle's arc and text. Defaults to the canvas default color."
                            },
                            "angle_name": {
                                "type": [ "string", "null" ],
                                "description": "Optional name for the angle. If not provided, a name might be generated (e.g., 'angle_ABC')."
                            },
                            "is_reflex": {
                                "type": ["boolean", "null"],
                                "description": "Optional. If true, the reflex angle will be created. Defaults to false (smallest angle)."
                            }
                        },
                        "required": ["vx", "vy", "p1x", "p1y", "p2x", "p2y", "color", "angle_name", "is_reflex"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_angle",
                    "description": "Removes an angle by its name. This will also attempt to remove its constituent segments if they are no longer part of other drawables.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the angle to remove (e.g., 'angle_ABC')."
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
                    "name": "update_angle",
                    "description": "Updates editable properties of an existing angle (currently just its color).",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the angle to update."
                            },
                            "new_color": {
                                "type": [ "string", "null" ],
                                "description": "The new color for the angle. Provide null to leave unchanged."
                            }
                        },
                        "required": ["name", "new_color"],
                        "additionalProperties": False
                    }
                }
            }
            # END ANGLE FUNCTIONS
        ]