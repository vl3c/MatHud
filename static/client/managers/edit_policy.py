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

DRAWABLE_EDIT_POLICIES: Dict[str, DrawableEditPolicy] = {
    POINT_EDIT_POLICY.drawable_type: POINT_EDIT_POLICY,
}


def get_drawable_edit_policy(drawable_type: str) -> Optional[DrawableEditPolicy]:
    """Lookup the policy for a drawable type."""

    return DRAWABLE_EDIT_POLICIES.get(drawable_type)


