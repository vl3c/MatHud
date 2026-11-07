"""Refusals for tools that do not take implicit curves or inequality regions.

Intersections, arc length and tangents work on curves with a parameter (functions,
parametric and polar curves, conics); an implicit curve has none. Given the name of one,
those tools say so and name what to use instead, rather than "no curve named c1".

Pure Python (no browser imports).
"""

from __future__ import annotations

from typing import Any, Optional


def implicit_refusal(drawables: Any, name: str, tool: str) -> Optional[str]:
    """The refusal for ``tool`` when ``name`` is an implicit curve or an inequality region, else None."""
    for curve in getattr(drawables, "ImplicitCurves", None) or []:
        if getattr(curve, "name", None) == name:
            return (
                f"'{name}' is an implicit curve ({curve.equation}); {tool} does not take implicit curves. "
                "For points where it meets another curve, use solve_system_of_equations with the equations."
            )
    for area in getattr(drawables, "InequalityColoredAreas", None) or []:
        if getattr(area, "name", None) == name:
            return f"'{name}' is a shaded inequality region ({area.inequality}); {tool} does not take regions."
    return None
