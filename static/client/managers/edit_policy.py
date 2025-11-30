"""
Edit policy definitions for drawable mutation operations.

Provides structured metadata describing which fields of a drawable
can be modified in-place, the category of the edit (cosmetic vs.
geometry), and any safety requirements (such as needing the target
to be isolated from dependency graphs).
"""

from __future__ import annotations

from typing import Dict, Optional


class EditRule:
    """Describes a single editable property."""

    def __init__(
        self,
        field: str,
        category: str,
        requires_solitary: bool = False,
        description: str = "",
        requires_complete_coordinate_pair: bool = False,
    ) -> None:
        self.field = field
        self.category = category
        self.requires_solitary = requires_solitary
        self.description = description
        self.requires_complete_coordinate_pair = requires_complete_coordinate_pair


class DrawableEditPolicy:
    """Collection of edit rules for a drawable type."""

    def __init__(self, drawable_type: str, rules: Dict[str, EditRule]) -> None:
        self.drawable_type = drawable_type
        self.rules = rules

    def get_rule(self, field: str) -> Optional[EditRule]:
        return self.rules.get(field)


POINT_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="Point",
    rules={
        "name": EditRule(
            field="name",
            category="cosmetic",
            requires_solitary=True,
            description="Rename an isolated point without affecting other drawables.",
        ),
        "color": EditRule(
            field="color",
            category="cosmetic",
            requires_solitary=True,
            description="Adjust display color for a solitary point.",
        ),
        "position": EditRule(
            field="position",
            category="local_geometry",
            requires_solitary=True,
            requires_complete_coordinate_pair=True,
            description="Move an isolated point to a new (x, y) location.",
        ),
    },
)

ANGLE_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="Angle",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Update the arc color for an angle without recreating it.",
        ),
    },
)

LABEL_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="Label",
    rules={
        "text": EditRule(
            field="text",
            category="content",
            description="Update the label text while preserving wrapping constraints.",
        ),
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Adjust the label text color.",
        ),
        "position": EditRule(
            field="position",
            category="local_geometry",
            requires_complete_coordinate_pair=True,
            description="Move the label anchor to a new (x, y) position.",
        ),
        "font_size": EditRule(
            field="font_size",
            category="style",
            description="Change the label font size.",
        ),
        "rotation": EditRule(
            field="rotation",
            category="local_geometry",
            description="Rotate the label text by a specified degree value.",
        ),
    },
)

SEGMENT_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="Segment",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Update the segment color.",
        ),
    },
)

VECTOR_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="Vector",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Update the vector stroke color.",
        ),
    },
)

TRIANGLE_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="Triangle",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Adjust the fill or outline color of the triangle.",
        ),
    },
)

RECTANGLE_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="Rectangle",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Adjust the fill or outline color of the rectangle.",
        ),
    },
)

QUADRILATERAL_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="Quadrilateral",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Adjust the outline color of the quadrilateral.",
        ),
    },
)

PENTAGON_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="Pentagon",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Adjust the outline color of the pentagon.",
        ),
    },
)

HEXAGON_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="Hexagon",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Adjust the outline color of the hexagon.",
        ),
    },
)

CIRCLE_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="Circle",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Update the circle stroke or fill color.",
        ),
        "center": EditRule(
            field="center",
            category="global_geometry",
            requires_solitary=True,
            requires_complete_coordinate_pair=True,
            description="Move the circle center to a new math-space coordinate.",
        ),
    },
)

CIRCLE_ARC_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="CircleArc",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Update the circle arc stroke color.",
        ),
        "use_major_arc": EditRule(
            field="use_major_arc",
            category="style",
            description="Toggle whether the arc displays the major or minor sweep.",
        ),
    },
)

ELLIPSE_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="Ellipse",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Update the ellipse stroke or fill color.",
        ),
        "radius_x": EditRule(
            field="radius_x",
            category="local_geometry",
            requires_solitary=True,
            description="Adjust the horizontal radius for an isolated ellipse.",
        ),
        "radius_y": EditRule(
            field="radius_y",
            category="local_geometry",
            requires_solitary=True,
            description="Adjust the vertical radius for an isolated ellipse.",
        ),
        "rotation_angle": EditRule(
            field="rotation_angle",
            category="local_geometry",
            requires_solitary=True,
            description="Rotate the ellipse around its center.",
        ),
        "center": EditRule(
            field="center",
            category="global_geometry",
            requires_solitary=True,
            requires_complete_coordinate_pair=True,
            description="Move the ellipse center to a new math-space coordinate.",
        ),
    },
)

FUNCTION_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="Function",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Update the plotted function color.",
        ),
        "left_bound": EditRule(
            field="left_bound",
            category="range",
            description="Update the left plotting bound.",
        ),
        "right_bound": EditRule(
            field="right_bound",
            category="range",
            description="Update the right plotting bound.",
        ),
    },
)

FUNCTIONS_BOUNDED_COLORED_AREA_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="FunctionsBoundedColoredArea",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Adjust the fill color for the bounded area.",
        ),
        "opacity": EditRule(
            field="opacity",
            category="style",
            description="Update the fill opacity between 0 and 1.",
        ),
        "left_bound": EditRule(
            field="left_bound",
            category="range",
            description="Update the left bound where the area is evaluated.",
        ),
        "right_bound": EditRule(
            field="right_bound",
            category="range",
            description="Update the right bound where the area is evaluated.",
        ),
    },
)

FUNCTION_SEGMENT_BOUNDED_COLORED_AREA_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="FunctionSegmentBoundedColoredArea",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Adjust the fill color for the area.",
        ),
        "opacity": EditRule(
            field="opacity",
            category="style",
            description="Update the fill opacity between 0 and 1.",
        ),
    },
)

SEGMENTS_BOUNDED_COLORED_AREA_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="SegmentsBoundedColoredArea",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Adjust the fill color for the area between segments.",
        ),
        "opacity": EditRule(
            field="opacity",
            category="style",
            description="Update the fill opacity between 0 and 1.",
        ),
    },
)

PIECEWISE_FUNCTION_EDIT_POLICY = DrawableEditPolicy(
    drawable_type="PiecewiseFunction",
    rules={
        "color": EditRule(
            field="color",
            category="cosmetic",
            description="Update the plotted piecewise function color.",
        ),
    },
)

DRAWABLE_EDIT_POLICIES: Dict[str, DrawableEditPolicy] = {
    POINT_EDIT_POLICY.drawable_type: POINT_EDIT_POLICY,
    ANGLE_EDIT_POLICY.drawable_type: ANGLE_EDIT_POLICY,
    LABEL_EDIT_POLICY.drawable_type: LABEL_EDIT_POLICY,
    SEGMENT_EDIT_POLICY.drawable_type: SEGMENT_EDIT_POLICY,
    VECTOR_EDIT_POLICY.drawable_type: VECTOR_EDIT_POLICY,
    TRIANGLE_EDIT_POLICY.drawable_type: TRIANGLE_EDIT_POLICY,
    RECTANGLE_EDIT_POLICY.drawable_type: RECTANGLE_EDIT_POLICY,
    QUADRILATERAL_EDIT_POLICY.drawable_type: QUADRILATERAL_EDIT_POLICY,
    PENTAGON_EDIT_POLICY.drawable_type: PENTAGON_EDIT_POLICY,
    HEXAGON_EDIT_POLICY.drawable_type: HEXAGON_EDIT_POLICY,
    CIRCLE_EDIT_POLICY.drawable_type: CIRCLE_EDIT_POLICY,
    CIRCLE_ARC_EDIT_POLICY.drawable_type: CIRCLE_ARC_EDIT_POLICY,
    ELLIPSE_EDIT_POLICY.drawable_type: ELLIPSE_EDIT_POLICY,
    FUNCTION_EDIT_POLICY.drawable_type: FUNCTION_EDIT_POLICY,
    FUNCTIONS_BOUNDED_COLORED_AREA_EDIT_POLICY.drawable_type: FUNCTIONS_BOUNDED_COLORED_AREA_EDIT_POLICY,
    FUNCTION_SEGMENT_BOUNDED_COLORED_AREA_EDIT_POLICY.drawable_type: FUNCTION_SEGMENT_BOUNDED_COLORED_AREA_EDIT_POLICY,
    SEGMENTS_BOUNDED_COLORED_AREA_EDIT_POLICY.drawable_type: SEGMENTS_BOUNDED_COLORED_AREA_EDIT_POLICY,
    PIECEWISE_FUNCTION_EDIT_POLICY.drawable_type: PIECEWISE_FUNCTION_EDIT_POLICY,
}


def get_drawable_edit_policy(drawable_type: str) -> Optional[DrawableEditPolicy]:
    """Lookup the policy for a drawable type."""

    return DRAWABLE_EDIT_POLICIES.get(drawable_type)


