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
                    "name": "zoom",
                    "description": "Centers viewport on (center_x, center_y). The range_val specifies half-width (if range_axis='x') or half-height (if range_axis='y'); the other axis scales with canvas aspect ratio. Example: 'zoom x in range +-2, y around 10' uses center_x=0, center_y=10, range_val=2, range_axis='x'.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "center_x": {
                                "type": "number",
                                "description": "X coordinate to center on"
                            },
                            "center_y": {
                                "type": "number",
                                "description": "Y coordinate to center on"
                            },
                            "range_val": {
                                "type": "number",
                                "description": "Half-size for the specified axis (shows center +/- this value)"
                            },
                            "range_axis": {
                                "type": "string",
                                "enum": ["x", "y"],
                                "description": "Which axis range applies to; other axis scales with aspect ratio"
                            }
                        },
                        "required": ["center_x", "center_y", "range_val", "range_axis"],
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
                    "name": "get_current_canvas_state",
                    "description": "Returns the current serialized canvas state (drawables, cartesian state, computations) without modifying the canvas",
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
                            },
                            "label_text": {
                                "type": ["string", "null"],
                                "description": "Optional text for the segment-owned label (default empty)"
                            },
                            "label_visible": {
                                "type": ["boolean", "null"],
                                "description": "Whether to display the segment-owned label (default false)"
                            }
                        },
                        "required": ["x1", "y1", "x2", "y2", "color", "name", "label_text", "label_visible"],
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
                    "description": "Updates editable properties of an existing segment (color, label text, or label visibility). Provide null for fields that should remain unchanged.",
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
                            },
                            "new_label_text": {
                                "type": ["string", "null"],
                                "description": "Optional new text for the segment-owned label"
                            },
                            "new_label_visible": {
                                "type": ["boolean", "null"],
                                "description": "Optional visibility flag for the segment-owned label"
                            }
                        },
                        "required": ["name", "new_color", "new_label_text", "new_label_visible"],
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
                                "description": "Optional polygon classification (triangle, quadrilateral, pentagon, hexagon, heptagon, octagon, nonagon, decagon, or generic).",
                                "enum": ["triangle", "quadrilateral", "pentagon", "hexagon", "heptagon", "octagon", "nonagon", "decagon", "generic", None],
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
                                "description": "Optional polygon classification (triangle, quadrilateral, rectangle, square, pentagon, hexagon, heptagon, octagon, nonagon, decagon, or generic)."
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
                            },
                            "undefined_at": {
                                "type": ["array", "null"],
                                "description": "Optional list of x-values where the function is explicitly undefined (holes). E.g., [0, 2] means the function has holes at x=0 and x=2.",
                                "items": {
                                    "type": "number"
                                }
                            }
                        },
                        "required": ["function_string", "name", "left_bound", "right_bound", "color", "undefined_at"],
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
                    "name": "draw_piecewise_function",
                    "description": "Plots a piecewise-defined function with different expressions for different intervals. Each piece specifies an expression and its valid interval bounds. Use null for unbounded intervals (extending to infinity). Use undefined_at for explicit holes (points where the function is undefined).",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pieces": {
                                "type": "array",
                                "minItems": 1,
                                "description": "List of function pieces, each defining an expression and its interval.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "expression": {
                                            "type": "string",
                                            "description": "Mathematical expression for this piece (e.g., 'x^2', 'sin(x)')."
                                        },
                                        "left": {
                                            "type": ["number", "null"],
                                            "description": "Left interval bound (null for negative infinity)."
                                        },
                                        "right": {
                                            "type": ["number", "null"],
                                            "description": "Right interval bound (null for positive infinity)."
                                        },
                                        "left_inclusive": {
                                            "type": "boolean",
                                            "description": "Whether the left bound is included in the interval."
                                        },
                                        "right_inclusive": {
                                            "type": "boolean",
                                            "description": "Whether the right bound is included in the interval."
                                        },
                                        "undefined_at": {
                                            "type": ["array", "null"],
                                            "description": "Optional list of x-values where this piece is explicitly undefined (holes).",
                                            "items": {
                                                "type": "number"
                                            }
                                        }
                                    },
                                    "required": ["expression", "left", "right", "left_inclusive", "right_inclusive", "undefined_at"],
                                    "additionalProperties": False
                                }
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional name for the piecewise function."
                            },
                            "color": {
                                "type": ["string", "null"],
                                "description": "Optional color for the plotted function."
                            }
                        },
                        "required": ["pieces", "name", "color"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_piecewise_function",
                    "description": "Removes the plotted piecewise function with the given name from the canvas.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the piecewise function to delete."
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
                    "name": "update_piecewise_function",
                    "description": "Updates editable properties of an existing piecewise function (currently just color). Provide null for fields to leave them unchanged.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Existing name of the piecewise function to edit."
                            },
                            "new_color": {
                                "type": ["string", "null"],
                                "description": "Optional new color for the function plot."
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
                    "name": "create_region_colored_area",
                    "description": "Fill a region defined by a boolean expression or a closed shape. Supports expressions with operators (& | - ^), arcs, circles, ellipses, polygons, and segments. Expression takes precedence if provided.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": ["string", "null"],
                                "description": "Boolean region expression using shape names and operators. Examples: 'ArcMaj_AB & CD' (arc intersected with segment), 'circle_A - triangle_ABC' (difference). Takes precedence over other parameters."
                            },
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
                            "expression",
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
            # START GRAPH FUNCTIONS
            {
                "type": "function",
                "function": {
                    "name": "generate_graph",
                    "description": "Generates a graph or tree on the canvas using provided vertices/edges or an adjacency matrix. Returns the created graph state and drawable names for follow-up highlighting.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": ["string", "null"]},
                            "graph_type": {
                                "type": "string",
                                "enum": ["graph", "tree", "dag"],
                                "description": "Type of graph to create."
                            },
                            "directed": {"type": ["boolean", "null"]},
                            "root": {"type": ["string", "null"], "description": "Root id for trees."},
                            "layout": {"type": ["string", "null"], "description": "Layout hint: 'tree' or 'hierarchical' for top-down tree display (default for trees), 'radial' for concentric rings from root, 'circular' for nodes on a circle, 'grid' for rectangular grid, 'force' for force-directed."},
                            "placement_box": {
                                "type": ["object", "null"],
                                "description": "Bounding box for vertex placement. Defined from bottom-left corner in math coordinates (y increases upward). Box spans from (x, y) to (x + width, y + height).",
                                "properties": {
                                    "x": {"type": "number", "description": "Left edge X coordinate (bottom-left corner)"},
                                    "y": {"type": "number", "description": "Bottom edge Y coordinate (bottom-left corner, in math coords where y increases upward)"},
                                    "width": {"type": "number", "description": "Box width extending rightward (positive X direction)"},
                                    "height": {"type": "number", "description": "Box height extending upward (positive Y direction)"}
                                },
                                "required": ["x", "y", "width", "height"],
                                "additionalProperties": False
                            },
                            "vertices": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": ["string", "null"]},
                                        "x": {"type": ["number", "null"]},
                                        "y": {"type": ["number", "null"]},
                                        "color": {"type": ["string", "null"]},
                                        "label": {"type": ["string", "null"]}
                                    },
                                    "required": ["name", "x", "y", "color", "label"],
                                    "additionalProperties": False
                                },
                                "description": "List of vertex descriptors. Vertex id is implied by array index starting at 0."
                            },
                            "edges": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source": {"type": "number", "description": "Source vertex index (0-based, matches vertices array order)"},
                                        "target": {"type": "number", "description": "Target vertex index (0-based, matches vertices array order)"},
                                        "weight": {"type": ["number", "null"]},
                                        "name": {"type": ["string", "null"]},
                                        "color": {"type": ["string", "null"]},
                                        "directed": {"type": ["boolean", "null"]}
                                    },
                                    "required": ["source", "target", "weight", "name", "color", "directed"],
                                    "additionalProperties": False
                                },
                                "description": "List of edge descriptors."
                            },
                            "adjacency_matrix": {
                                "type": ["array", "null"],
                                "items": {
                                    "type": "array",
                                    "items": {"type": "number"}
                                },
                                "description": "Optional adjacency matrix (weights allowed). Rows/columns follow the order of the provided vertices array (0-based)."
                            }
                        },
                        "required": ["name", "graph_type", "directed", "root", "layout", "placement_box", "vertices", "edges", "adjacency_matrix"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_graph",
                    "description": "Deletes a graph or tree and its associated drawables by name.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"}
                        },
                        "required": ["name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_graph",
                    "description": "Analyzes a graph or tree for operations like shortest path, MST, bridges, articulation points, Euler status, bipartite check, BFS/DFS orders, levels, diameter, LCA, convex hull of vertices, or point-in-hull containment. Accepts an existing graph name.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "graph_name": {"type": "string", "description": "Existing graph name to analyze (must exist on canvas)."},
                            "operation": {
                                "type": "string",
                                "enum": ["shortest_path", "mst", "topological_sort", "bridges", "articulation_points", "euler_status", "bipartite", "bfs", "dfs", "levels", "diameter", "lca", "balance_children", "invert_children", "reroot", "convex_hull", "point_in_hull"]
                            },
                            "params": {"type": ["object", "null"], "description": "Operation-specific parameters (start, goal, root, a, b, new_root, x, y for point_in_hull, etc.)."}
                        },
                        "required": ["graph_name", "operation"],
                        "additionalProperties": False
                    }
                }
            },
            # END GRAPH FUNCTIONS
            # START PLOT FUNCTIONS
            {
                "type": "function",
                "function": {
                    "name": "plot_distribution",
                    "description": "Plots a probability distribution on the canvas. Choose representation 'continuous' for a function curve or 'discrete' for bar rectangles. For continuous plots, you can optionally draw the curve over plot_bounds while shading only over shade_bounds (clamped into plot_bounds). Creates a tracked plot composite for reliable deletion.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional plot name. If null, a name will be generated."
                            },
                            "representation": {
                                "type": "string",
                                "enum": ["continuous", "discrete"],
                                "description": "Plot representation. 'continuous' draws a smooth curve. 'discrete' draws bars (rectangles)."
                            },
                            "distribution_type": {
                                "type": "string",
                                "enum": ["normal"],
                                "description": "Distribution to plot. v1 supports only 'normal' (Gaussian)."
                            },
                            "distribution_params": {
                                "type": ["object", "null"],
                                "description": "Parameters for the selected distribution type. For 'normal', provide mean and sigma.",
                                "properties": {
                                    "mean": {
                                        "type": ["number", "null"],
                                        "description": "Mean (mu) for the normal distribution. Defaults to 0 if null."
                                    },
                                    "sigma": {
                                        "type": ["number", "null"],
                                        "description": "Standard deviation (sigma) for the normal distribution. Defaults to 1 if null. Must be > 0."
                                    }
                                },
                                "required": ["mean", "sigma"],
                                "additionalProperties": False
                            },
                            "plot_bounds": {
                                "type": ["object", "null"],
                                "description": "Optional bounds for plotting the curve. If null, or either side is null, defaults to mean +/- 4*sigma.",
                                "properties": {
                                    "left_bound": {
                                        "type": ["number", "null"],
                                        "description": "Optional left bound for plotting the curve. Defaults to mean - 4*sigma when null."
                                    },
                                    "right_bound": {
                                        "type": ["number", "null"],
                                        "description": "Optional right bound for plotting the curve. Defaults to mean + 4*sigma when null."
                                    }
                                },
                                "required": ["left_bound", "right_bound"],
                                "additionalProperties": False
                            },
                            "shade_bounds": {
                                "type": ["object", "null"],
                                "description": "Continuous only. Optional bounds for shading under the curve. If null, defaults to plot_bounds. Bounds are clamped into plot_bounds.",
                                "properties": {
                                    "left_bound": {
                                        "type": ["number", "null"],
                                        "description": "Optional left bound for shading under the curve. If null, defaults to plot_bounds.left_bound."
                                    },
                                    "right_bound": {
                                        "type": ["number", "null"],
                                        "description": "Optional right bound for shading under the curve. If null, defaults to plot_bounds.right_bound."
                                    }
                                },
                                "required": ["left_bound", "right_bound"],
                                "additionalProperties": False
                            },
                            "curve_color": {
                                "type": ["string", "null"],
                                "description": "Optional color for the plotted curve."
                            },
                            "fill_color": {
                                "type": ["string", "null"],
                                "description": "Optional fill color for the area under the curve. Defaults to the standard area fill color."
                            },
                            "fill_opacity": {
                                "type": ["number", "null"],
                                "description": "Optional fill opacity (0 to 1). Defaults to the standard area opacity."
                            },
                            "bar_count": {
                                "type": ["number", "null"],
                                "description": "Discrete only. Number of bars to draw across the bounds. If null, a default is used."
                            }
                        },
                        "required": [
                            "name",
                            "representation",
                            "distribution_type",
                            "distribution_params",
                            "plot_bounds",
                            "shade_bounds",
                            "curve_color",
                            "fill_color",
                            "fill_opacity",
                            "bar_count"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "plot_bars",
                    "description": "Plots a bar chart from tabular data (values with labels). Creates a tracked plot composite for reliable deletion.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional plot name. If null, a name will be generated."
                            },
                            "values": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Bar heights (math-space units). Must have at least one entry."
                            },
                            "labels_below": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Label under each bar. Must have one label per value."
                            },
                            "labels_above": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                                "description": "Optional label above each bar (for example, formatted values). If provided, must have one label per value."
                            },
                            "bar_spacing": {
                                "type": ["number", "null"],
                                "description": "Optional spacing between bars in math-space units. Defaults to 0.2."
                            },
                            "bar_width": {
                                "type": ["number", "null"],
                                "description": "Optional bar width in math-space units. Defaults to 1.0."
                            },
                            "stroke_color": {
                                "type": ["string", "null"],
                                "description": "Optional stroke color for each bar."
                            },
                            "fill_color": {
                                "type": ["string", "null"],
                                "description": "Optional fill color for each bar."
                            },
                            "fill_opacity": {
                                "type": ["number", "null"],
                                "description": "Optional fill opacity (0 to 1)."
                            },
                            "x_start": {
                                "type": ["number", "null"],
                                "description": "Optional left x coordinate for the first bar. Defaults to 0."
                            },
                            "y_base": {
                                "type": ["number", "null"],
                                "description": "Optional baseline y coordinate for bars. Defaults to 0."
                            }
                        },
                        "required": [
                            "name",
                            "values",
                            "labels_below",
                            "labels_above",
                            "bar_spacing",
                            "bar_width",
                            "stroke_color",
                            "fill_color",
                            "fill_opacity",
                            "x_start",
                            "y_base"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_plot",
                    "description": "Deletes a previously created plot composite by name, including any underlying components (curve and filled area, or derived bars).",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"}
                        },
                        "required": ["name"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fit_regression",
                    "description": "Fits a regression model to data points and plots the resulting curve. Supported model types: linear (y = mx + b), polynomial (y = a0 + a1*x + ... + an*x^n), exponential (y = a*e^(bx)), logarithmic (y = a + b*ln(x)), power (y = a*x^b), logistic (y = L/(1+e^(-k(x-x0)))), and sinusoidal (y = a*sin(bx+c)+d). Returns the function_name, fitted expression, coefficients, R-squared, and point_names. Use delete_function to remove the curve; delete points individually.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": ["string", "null"],
                                "description": "Optional base name for the function and data points. If null, a name will be generated based on model type."
                            },
                            "x_data": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Array of x values (independent variable). Must have at least 2 points (more for polynomial)."
                            },
                            "y_data": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Array of y values (dependent variable). Must have same length as x_data."
                            },
                            "model_type": {
                                "type": "string",
                                "enum": ["linear", "polynomial", "exponential", "logarithmic", "power", "logistic", "sinusoidal"],
                                "description": "Type of regression model to fit. Note: exponential and power require positive y values; logarithmic and power require positive x values."
                            },
                            "degree": {
                                "type": ["integer", "null"],
                                "description": "Polynomial degree (required for polynomial model, ignored otherwise). Must be >= 1 and less than the number of data points."
                            },
                            "plot_bounds": {
                                "type": ["object", "null"],
                                "description": "Optional bounds for plotting the fitted curve. Defaults to data range with 10% padding.",
                                "properties": {
                                    "left_bound": {
                                        "type": ["number", "null"],
                                        "description": "Left bound for plotting. Defaults to min(x_data) - 10% range."
                                    },
                                    "right_bound": {
                                        "type": ["number", "null"],
                                        "description": "Right bound for plotting. Defaults to max(x_data) + 10% range."
                                    }
                                },
                                "required": ["left_bound", "right_bound"],
                                "additionalProperties": False
                            },
                            "curve_color": {
                                "type": ["string", "null"],
                                "description": "Optional color for the fitted curve."
                            },
                            "show_points": {
                                "type": ["boolean", "null"],
                                "description": "Whether to plot the data points. Defaults to true."
                            },
                            "point_color": {
                                "type": ["string", "null"],
                                "description": "Optional color for data points (if show_points is true)."
                            }
                        },
                        "required": [
                            "name",
                            "x_data",
                            "y_data",
                            "model_type",
                            "degree",
                            "plot_bounds",
                            "curve_color",
                            "show_points",
                            "point_color"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            # END PLOT FUNCTIONS
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
            },
            # END ANGLE FUNCTIONS
            # START AREA CALCULATION FUNCTIONS
            {
                "type": "function",
                "function": {
                    "name": "calculate_area",
                    "description": "Calculates the area of a region defined by a boolean expression over drawable shapes. Supports circles, ellipses, arcs (circular segments), polygons (triangles, rectangles, etc.), and segments (treated as half-planes - area to the LEFT of segment direction). Use operators: & (intersection), | (union), - (difference), ^ (symmetric difference). Parentheses supported for grouping.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Boolean expression with drawable names. Examples: 'circle_A' (single shape), 'circle_A & triangle_ABC' (intersection), 'C(5) & AB' (circle cut by segment AB), 'ArcMaj_CD & triangle_ABC' (arc segment intersected with triangle), 'circle_A - triangle_ABC' (difference), '(circle_A & quad_ABCD) & EF' (shapes intersected then cut by segment)."
                            }
                        },
                        "required": ["expression"],
                        "additionalProperties": False
                    }
                }
            },
            # END AREA CALCULATION FUNCTIONS
            # START COORDINATE SYSTEM FUNCTIONS
            {
                "type": "function",
                "function": {
                    "name": "set_coordinate_system",
                    "description": "Sets the coordinate system mode for the canvas grid. Choose 'cartesian' for the standard x-y grid or 'polar' for a polar coordinate grid with concentric circles and radial lines.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["cartesian", "polar"],
                                "description": "The coordinate system mode: 'cartesian' for x-y grid, 'polar' for polar grid"
                            }
                        },
                        "required": ["mode"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "convert_coordinates",
                    "description": "Converts coordinates between rectangular (Cartesian) and polar coordinate systems. For rectangular to polar: returns (r, theta) where r is radius and theta is angle in radians. For polar to rectangular: returns (x, y) coordinates.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "coord1": {
                                "type": "number",
                                "description": "First coordinate: x for rectangular-to-polar, r (radius) for polar-to-rectangular"
                            },
                            "coord2": {
                                "type": "number",
                                "description": "Second coordinate: y for rectangular-to-polar, theta (angle in radians) for polar-to-rectangular"
                            },
                            "from_system": {
                                "type": "string",
                                "enum": ["rectangular", "cartesian", "polar"],
                                "description": "The source coordinate system ('rectangular' and 'cartesian' are equivalent)"
                            },
                            "to_system": {
                                "type": "string",
                                "enum": ["rectangular", "cartesian", "polar"],
                                "description": "The target coordinate system ('rectangular' and 'cartesian' are equivalent)"
                            }
                        },
                        "required": ["coord1", "coord2", "from_system", "to_system"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_grid_visible",
                    "description": "Sets the visibility of the active coordinate grid (Cartesian or Polar). Use this to show or hide the grid lines without changing the coordinate system mode.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "visible": {
                                "type": "boolean",
                                "description": "Whether the grid should be visible (true to show, false to hide)"
                            }
                        },
                        "required": ["visible"],
                        "additionalProperties": False
                    }
                }
            },
            # END COORDINATE SYSTEM FUNCTIONS
            # START TOOL SEARCH FUNCTIONS
            {
                "type": "function",
                "function": {
                    "name": "search_tools",
                    "description": "Search for the best tools to accomplish a task. Use this when you're unsure which specific tool to use. Provide a description of what you want to do, and receive the most relevant tool definitions.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Description of what you want to accomplish (e.g., 'draw a triangle with vertices at specific coordinates', 'calculate the derivative of a function')"
                            },
                            "max_results": {
                                "type": ["integer", "null"],
                                "description": "Maximum number of tools to return (default: 10, max: 20)"
                            }
                        },
                        "required": ["query", "max_results"],
                        "additionalProperties": False
                    }
                }
            }
            # END TOOL SEARCH FUNCTIONS
        ]