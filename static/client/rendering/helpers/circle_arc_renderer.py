"""Circle arc rendering helper for drawing arc segments on circles.

This module provides the render_circle_arc_helper function that renders
a circular arc between two points on a circle's circumference.

Key Features:
    - Minor and major arc selection support
    - Automatic sweep direction computation
    - Coordinate transformation from math to screen space
    - Metadata emission for debugging and hit testing
    - Circle synchronization for dependent arcs
"""

from __future__ import annotations

import math

from rendering.helpers.shape_decorator import _manages_shape
from rendering.primitives import StrokeStyle
from utils.math_utils import MathUtils


def _map_circle_arc_to_screen(circle_arc, coordinate_mapper, style):
    """Transform circle arc coordinates from math to screen space.

    Args:
        circle_arc: The CircleArc drawable with center, radius, and endpoints.
        coordinate_mapper: Mapper for coordinate transformation.
        style: Style dictionary with circle_arc_radius_scale.

    Returns:
        Tuple of (center_screen, radius_on_screen, point1_screen, point2_screen)
        or None if transformation fails.
    """
    try:
        center_screen = coordinate_mapper.math_to_screen(circle_arc.center_x, circle_arc.center_y)
        radius_screen = coordinate_mapper.scale_value(circle_arc.radius)
        point1_screen = coordinate_mapper.math_to_screen(circle_arc.point1.x, circle_arc.point1.y)
        point2_screen = coordinate_mapper.math_to_screen(circle_arc.point2.x, circle_arc.point2.y)
    except Exception:
        return None

    if not center_screen or radius_screen is None or not point1_screen or not point2_screen:
        return None

    radius_scale = style.get("circle_arc_radius_scale", 1.0) or 1.0
    try:
        radius_on_screen = float(radius_screen) * float(radius_scale)
    except Exception:
        radius_on_screen = 0.0
    if radius_on_screen <= 0:
        return None

    return center_screen, radius_on_screen, point1_screen, point2_screen


def _compute_circle_arc_sweep(circle_arc, center_screen, point1_screen):
    """Compute sweep angles and direction for a circle arc.

    Determines whether to render the minor or major arc based on the
    use_major_arc attribute and calculates the correct sweep direction.

    Args:
        circle_arc: The CircleArc drawable with point coordinates and use_major_arc.
        center_screen: Screen coordinate tuple (cx, cy) for arc center.
        point1_screen: Screen coordinate tuple (p1x, p1y) for first endpoint.

    Returns:
        Tuple of (start_angle_screen, end_angle_final, sweep_clockwise)
        or None if the arc has negligible angular extent.
    """
    cx, cy = center_screen
    p1x, p1y = point1_screen
    start_angle_math = math.atan2(circle_arc.point1.y - circle_arc.center_y, circle_arc.point1.x - circle_arc.center_x)
    target_angle_math = math.atan2(circle_arc.point2.y - circle_arc.center_y, circle_arc.point2.x - circle_arc.center_x)

    full_turn = 2 * math.pi
    delta_ccw_math = (target_angle_math - start_angle_math) % full_turn
    delta_cw_math = (start_angle_math - target_angle_math) % full_turn
    if min(delta_ccw_math, delta_cw_math) < MathUtils.EPSILON:
        return None

    minor_is_ccw = delta_ccw_math <= delta_cw_math
    minor_delta = delta_ccw_math if minor_is_ccw else delta_cw_math
    major_delta = delta_cw_math if minor_is_ccw else delta_ccw_math

    use_major = bool(getattr(circle_arc, "use_major_arc", False))
    if use_major:
        math_ccw = not minor_is_ccw
        sweep_delta = major_delta
    else:
        math_ccw = minor_is_ccw
        sweep_delta = minor_delta

    start_angle_screen = math.atan2(p1y - cy, p1x - cx)
    if math_ccw:
        sweep_clockwise = False
        end_angle_final = start_angle_screen - sweep_delta
    else:
        sweep_clockwise = True
        end_angle_final = start_angle_screen + sweep_delta

    return start_angle_screen, end_angle_final, sweep_clockwise


@_manages_shape
def _stroke_circle_arc(primitives, circle_arc, center_screen, radius_on_screen, start_angle, end_angle, sweep_clockwise, style):
    """Stroke a circle arc with the computed parameters.

    Args:
        primitives: The renderer primitives interface.
        circle_arc: The CircleArc drawable for color and metadata.
        center_screen: Screen coordinate tuple for arc center.
        radius_on_screen: Arc radius in screen pixels.
        start_angle: Starting angle in radians.
        end_angle: Ending angle in radians.
        sweep_clockwise: True for clockwise sweep direction.
        style: Style dictionary with circle_arc_color and stroke width.
    """
    color = str(getattr(circle_arc, "color", style.get("circle_arc_color", "#000")))
    stroke = StrokeStyle(
        color=color,
        width=float(style.get("circle_arc_stroke_width", 1) or 1),
    )
    metadata = {
        "circle_arc": {
            "center_math": (float(circle_arc.center_x), float(circle_arc.center_y)),
            "radius_math": float(circle_arc.radius),
            "use_major_arc": bool(getattr(circle_arc, "use_major_arc", False)),
            "sweep_clockwise": sweep_clockwise,
            "point1": (float(circle_arc.point1.x), float(circle_arc.point1.y)),
            "point2": (float(circle_arc.point2.x), float(circle_arc.point2.y)),
        }
    }
    primitives.stroke_arc(
        center_screen,
        radius_on_screen,
        float(start_angle),
        float(end_angle),
        sweep_clockwise,
        stroke,
        screen_space=True,
        metadata=metadata,
    )


def render_circle_arc_helper(primitives, circle_arc, coordinate_mapper, style):
    """Render a circle arc drawable.

    Synchronizes with the parent circle if applicable, then computes
    screen coordinates and sweep direction before stroking the arc.

    Args:
        primitives: The renderer primitives interface.
        circle_arc: The CircleArc drawable with center, radius, and endpoints.
        coordinate_mapper: Mapper for math-to-screen coordinate conversion.
        style: Style dictionary with circle_arc_color and circle_arc_stroke_width.
    """
    if hasattr(circle_arc, "sync_with_circle") and callable(circle_arc.sync_with_circle):
        circle_arc.sync_with_circle()

    screen_data = _map_circle_arc_to_screen(circle_arc, coordinate_mapper, style)
    if not screen_data:
        return
    center_screen, radius_on_screen, point1_screen, point2_screen = screen_data

    sweep_data = _compute_circle_arc_sweep(circle_arc, center_screen, point1_screen)
    if not sweep_data:
        return

    start_angle_screen, end_angle_final, sweep_clockwise = sweep_data
    _stroke_circle_arc(
        primitives,
        circle_arc,
        center_screen,
        radius_on_screen,
        start_angle_screen,
        end_angle_final,
        sweep_clockwise,
        style,
    )

