"""
Area Expression Evaluator

Parses and evaluates boolean expressions over drawable regions to compute area.
Supports intersection (&), union (|), difference (-), and symmetric difference (^).

Supported drawables:
    - Circles, Ellipses: Full closed shapes (e.g., C(25), E(50, 30))
    - Polygons (Triangle, Quadrilateral, etc.): Closed polygonal regions
    - Segments: Treated as half-planes (area to the LEFT of segment direction)

Name conventions (supports prime symbols for point names):
    - Points: A, A', A'', B'
    - Segments: AB, A'B, A''E'
    - Arcs: ArcMajor, ArcMinor

Example expressions:
    - "circle_A"                    (area of single drawable)
    - "circle_A & triangle_ABC"     (intersection)
    - "circle_A | triangle_ABC"     (union)
    - "circle_A - triangle_ABC"     (difference)
    - "(circle_A & triangle_ABC) - square_DEFG"  (nested)
    - "C(5) & AB"                   (circle cut by segment - circular segment area)
    - "ArcMajor & A''E'"            (arc intersected with segment using prime names)
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from geometry import Region

if TYPE_CHECKING:
    from canvas import Canvas
    from drawables.drawable import Drawable


class _RegionWithSource:
    """Wrapper to track the source drawable for special handling."""

    def __init__(
        self, region: Region, source_type: str, source_drawable: Optional["Drawable"] = None
    ) -> None:
        self.region = region
        self.source_type = source_type  # "arc", "segment", "polygon", "circle", "ellipse"
        self.source_drawable = source_drawable


class AreaExpressionResult:
    """Result of an area expression evaluation."""

    def __init__(
        self,
        area: float,
        regions: Optional[List[Region]] = None,
        region: Optional[Region] = None,
        error: Optional[str] = None,
    ) -> None:
        self.area = area
        self.regions = regions or []
        self.region = region
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        if self.error:
            return {"type": "error", "value": self.error}
        return {"type": "area", "value": self.area}


class _ASTNode:
    """Base class for AST nodes."""
    pass


class _NameNode(_ASTNode):
    """Leaf node representing a drawable name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Name({self.name})"


class _BinaryOpNode(_ASTNode):
    """Binary operation node."""

    def __init__(self, left: _ASTNode, op: str, right: _ASTNode) -> None:
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self) -> str:
        return f"BinOp({self.left} {self.op} {self.right})"


class AreaExpressionEvaluator:
    """Evaluates boolean expressions over drawable regions to compute area."""

    OPERATORS: Tuple[str, ...] = ("&", "|", "-", "^")

    OPERATOR_PRECEDENCE: Dict[str, int] = {
        "|": 1,
        "^": 1,
        "&": 2,
        "-": 2,
    }

    @staticmethod
    def evaluate(expression: str, canvas: "Canvas") -> AreaExpressionResult:
        """Evaluate an area expression and return the computed area.

        Args:
            expression: Boolean expression with drawable names
            canvas: Canvas instance to resolve drawable names

        Returns:
            AreaExpressionResult with computed area or error
        """
        try:
            AreaExpressionEvaluator._validate_expression(expression)
            tokens = AreaExpressionEvaluator._tokenize(expression)
            ast = AreaExpressionEvaluator._parse(tokens)
            result = AreaExpressionEvaluator._evaluate_ast(ast, canvas)

            # Normalize _RegionWithSource to Region
            if isinstance(result, _RegionWithSource):
                result = AreaExpressionEvaluator._normalize_to_single(result)

            if isinstance(result, list):
                total_area = sum(r.area() for r in result)
                combined = result[0] if len(result) == 1 else None
                return AreaExpressionResult(area=max(0.0, total_area), regions=result, region=combined)
            elif result is None:
                return AreaExpressionResult(area=0.0, regions=[], region=None)
            else:
                area = result.area()
                return AreaExpressionResult(area=max(0.0, area), regions=[result], region=result)

        except ValueError as e:
            return AreaExpressionResult(area=0.0, error=str(e))
        except Exception as e:
            return AreaExpressionResult(area=0.0, error=f"Evaluation error: {e}")

    @staticmethod
    def _validate_expression(expression: str) -> None:
        """Validate the expression is non-empty and has balanced parentheses."""
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("Expression must be a non-empty string")

        depth = 0
        for char in expression:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    raise ValueError("Unbalanced parentheses in expression")

        if depth != 0:
            raise ValueError("Unbalanced parentheses in expression")

    @staticmethod
    def _tokenize(expression: str) -> List[str]:
        """Split expression into tokens (names, operators, parentheses).

        Drawable names can contain letters, digits, underscores, and parentheses
        with numbers inside (e.g., "A(25)" for circles).
        """
        tokens: List[str] = []
        i = 0
        expr = expression.strip()

        while i < len(expr):
            char = expr[i]

            if char.isspace():
                i += 1
                continue

            if char in AreaExpressionEvaluator.OPERATORS:
                tokens.append(char)
                i += 1
                continue

            if char == "(":
                if tokens and tokens[-1] not in AreaExpressionEvaluator.OPERATORS and tokens[-1] != "(":
                    name_part = AreaExpressionEvaluator._read_drawable_name_suffix(expr, i)
                    if name_part:
                        tokens[-1] += name_part
                        i += len(name_part)
                        continue
                tokens.append("(")
                i += 1
                continue

            if char == ")":
                tokens.append(")")
                i += 1
                continue

            name = AreaExpressionEvaluator._read_name(expr, i)
            if name:
                tokens.append(name)
                i += len(name)
            else:
                raise ValueError(f"Unexpected character '{char}' at position {i}")

        return tokens

    @staticmethod
    def _read_name(expr: str, start: int) -> str:
        """Read a drawable name starting at position start.

        Names can be alphanumeric with underscores and prime symbols ('),
        and may include parenthesized suffixes like (25) or (25, 15) for circles/ellipses.

        Supported name patterns:
            - Points: A, A', A'', B'
            - Segments: AB, A'B, A''E'
            - Polygons: triangle_ABC, ABC, ABCD
            - Circles: C(25), C(50)
            - Ellipses: E(50, 30), E(3, 2)
            - Arcs: ArcMajor, ArcMinor
            - Functions: f, g, h
        """
        if start >= len(expr):
            return ""

        if not (expr[start].isalpha() or expr[start] == "_"):
            return ""

        i = start
        while i < len(expr) and (expr[i].isalnum() or expr[i] == "_" or expr[i] == "'"):
            i += 1

        if i < len(expr) and expr[i] == "(":
            suffix = AreaExpressionEvaluator._read_drawable_name_suffix(expr, i)
            if suffix:
                i += len(suffix)

        return expr[start:i]

    @staticmethod
    def _read_drawable_name_suffix(expr: str, start: int) -> str:
        """Read a parenthesized suffix like (25) or (25, 15)."""
        if start >= len(expr) or expr[start] != "(":
            return ""

        depth = 1
        i = start + 1
        while i < len(expr) and depth > 0:
            if expr[i] == "(":
                depth += 1
            elif expr[i] == ")":
                depth -= 1
            i += 1

        if depth == 0:
            suffix = expr[start:i]
            if re.match(r"^\([0-9., ]+\)$", suffix):
                return suffix

        return ""

    @staticmethod
    def _parse(tokens: List[str]) -> _ASTNode:
        """Parse tokens into an AST using precedence climbing."""
        if not tokens:
            raise ValueError("Empty expression")

        pos = [0]
        result = AreaExpressionEvaluator._parse_expression(tokens, pos, 0)

        if pos[0] < len(tokens):
            raise ValueError(f"Unexpected token '{tokens[pos[0]]}' at position {pos[0]}")

        return result

    @staticmethod
    def _parse_expression(tokens: List[str], pos: List[int], min_prec: int) -> _ASTNode:
        """Parse an expression with minimum precedence."""
        left = AreaExpressionEvaluator._parse_primary(tokens, pos)

        while pos[0] < len(tokens):
            op = tokens[pos[0]]
            if op not in AreaExpressionEvaluator.OPERATOR_PRECEDENCE:
                break

            prec = AreaExpressionEvaluator.OPERATOR_PRECEDENCE[op]
            if prec < min_prec:
                break

            pos[0] += 1
            right = AreaExpressionEvaluator._parse_expression(tokens, pos, prec + 1)
            left = _BinaryOpNode(left, op, right)

        return left

    @staticmethod
    def _parse_primary(tokens: List[str], pos: List[int]) -> _ASTNode:
        """Parse a primary expression (name or parenthesized expression)."""
        if pos[0] >= len(tokens):
            raise ValueError("Unexpected end of expression")

        token = tokens[pos[0]]

        if token == "(":
            pos[0] += 1
            node = AreaExpressionEvaluator._parse_expression(tokens, pos, 0)
            if pos[0] >= len(tokens) or tokens[pos[0]] != ")":
                raise ValueError("Missing closing parenthesis")
            pos[0] += 1
            return node

        if token in AreaExpressionEvaluator.OPERATORS:
            raise ValueError(f"Unexpected operator '{token}'")

        if token == ")":
            raise ValueError("Unexpected closing parenthesis")

        pos[0] += 1
        return _NameNode(token)

    @staticmethod
    def _evaluate_ast(
        node: _ASTNode,
        canvas: "Canvas",
    ) -> Union[_RegionWithSource, Region, List[Region], None]:
        """Recursively evaluate an AST node."""
        if isinstance(node, _NameNode):
            drawable = AreaExpressionEvaluator._resolve_drawable(node.name, canvas)
            return AreaExpressionEvaluator._drawable_to_region_with_source(drawable)

        if isinstance(node, _BinaryOpNode):
            left_result = AreaExpressionEvaluator._evaluate_ast(node.left, canvas)
            right_result = AreaExpressionEvaluator._evaluate_ast(node.right, canvas)
            return AreaExpressionEvaluator._apply_operation(
                left_result, right_result, node.op
            )

        raise ValueError(f"Unknown AST node type: {type(node)}")

    @staticmethod
    def _resolve_drawable(name: str, canvas: "Canvas") -> "Drawable":
        """Look up a drawable by name from the canvas."""
        drawable = canvas.drawable_manager.get_region_capable_drawable_by_name(name)
        if drawable is None:
            raise ValueError(f"Drawable '{name}' not found or cannot be converted to a region")
        return drawable

    @staticmethod
    def _drawable_to_region_with_source(drawable: "Drawable") -> _RegionWithSource:
        """Convert a drawable to a Region wrapped with source info."""
        class_name = drawable.get_class_name()

        if class_name == "Circle":
            center = (drawable.center.x, drawable.center.y)
            region = Region.from_circle(center, drawable.radius)
            return _RegionWithSource(region, "circle", drawable)

        if class_name == "Ellipse":
            center = (drawable.center.x, drawable.center.y)
            rotation_rad = math.radians(drawable.rotation_angle)
            region = Region.from_ellipse(
                center,
                drawable.radius_x,
                drawable.radius_y,
                rotation_rad,
            )
            return _RegionWithSource(region, "ellipse", drawable)

        if class_name == "CircleArc":
            region = AreaExpressionEvaluator._arc_to_region(drawable)
            return _RegionWithSource(region, "arc", drawable)

        if class_name == "Segment":
            return _RegionWithSource(None, "segment", drawable)  # type: ignore

        if hasattr(drawable, "get_segments"):
            region = Region.from_polygon(drawable)
            return _RegionWithSource(region, "polygon", drawable)

        if hasattr(drawable, "_segments"):
            region = AreaExpressionEvaluator._region_from_segments(drawable._segments)
            return _RegionWithSource(region, "polygon", drawable)

        raise ValueError(f"Cannot convert drawable of type '{class_name}' to a region")

    @staticmethod
    def _drawable_to_region(drawable: "Drawable") -> Region:
        """Convert a drawable to a Region.

        Segments are treated as half-planes (the area to the LEFT of the segment direction).
        To get the other half, use the segment with reversed point order in the expression.
        """
        class_name = drawable.get_class_name()

        if class_name == "Circle":
            center = (drawable.center.x, drawable.center.y)
            return Region.from_circle(center, drawable.radius)

        if class_name == "Ellipse":
            center = (drawable.center.x, drawable.center.y)
            rotation_rad = math.radians(drawable.rotation_angle)
            return Region.from_ellipse(
                center,
                drawable.radius_x,
                drawable.radius_y,
                rotation_rad,
            )

        if class_name == "CircleArc":
            return AreaExpressionEvaluator._arc_to_region(drawable)

        if class_name == "Segment":
            point1 = (drawable.point1.x, drawable.point1.y)
            point2 = (drawable.point2.x, drawable.point2.y)
            return Region.from_half_plane(point1, point2)

        if hasattr(drawable, "get_segments"):
            return Region.from_polygon(drawable)

        if hasattr(drawable, "_segments"):
            return AreaExpressionEvaluator._region_from_segments(drawable._segments)

        raise ValueError(f"Cannot convert drawable of type '{class_name}' to a region")

    @staticmethod
    def _arc_to_region(arc: "Drawable") -> Region:
        """Convert a CircleArc to a Region (circular segment between chord and arc curve)."""
        center = (arc.center_x, arc.center_y)
        radius = arc.radius
        p1 = (arc.point1.x, arc.point1.y)
        p2 = (arc.point2.x, arc.point2.y)

        # Calculate angles from center to endpoints
        angle1 = math.atan2(p1[1] - center[1], p1[0] - center[0])
        angle2 = math.atan2(p2[1] - center[1], p2[0] - center[0])

        # Calculate CCW sweep from angle1 to angle2
        ccw_sweep = angle2 - angle1
        if ccw_sweep < 0:
            ccw_sweep += 2 * math.pi

        # Determine which direction to sweep based on major/minor
        if arc.use_major_arc:
            # Major arc: take the longer path
            if ccw_sweep < math.pi:
                # CCW is shorter, go CW (negative direction)
                sweep = -(2 * math.pi - ccw_sweep)
            else:
                # CCW is longer, use it
                sweep = ccw_sweep
        else:
            # Minor arc: take the shorter path
            if ccw_sweep > math.pi:
                # CCW is longer, go CW (negative direction)
                sweep = -(2 * math.pi - ccw_sweep)
            else:
                # CCW is shorter, use it
                sweep = ccw_sweep

        # Create region from circular segment (chord + arc curve, no center)
        num_points = max(32, int(abs(sweep) * 16))
        points: List[Tuple[float, float]] = []

        # Trace along arc from p1 to p2
        for i in range(num_points + 1):
            t = i / num_points
            angle = angle1 + t * sweep
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))

        return Region.from_points(points)

    @staticmethod
    def _region_from_segments(segments: list) -> Region:
        """Create a Region from a list of segments."""
        points: List[Tuple[float, float]] = []
        for seg in segments:
            p1 = (seg.point1.x, seg.point1.y)
            if not points or points[-1] != p1:
                points.append(p1)
            p2 = (seg.point2.x, seg.point2.y)
            if points[-1] != p2:
                points.append(p2)
        if len(points) >= 3:
            return Region.from_points(points)
        raise ValueError("Not enough points to form a region")

    @staticmethod
    def _normalize_to_single(
        result: Union[_RegionWithSource, Region, List[Region], None]
    ) -> Optional[Region]:
        """Convert a result to a single Region or None."""
        if result is None:
            return None
        if isinstance(result, _RegionWithSource):
            # For segments without a pre-computed region, create a half-plane
            if result.region is None and result.source_type == "segment":
                seg = result.source_drawable
                p1 = (seg.point1.x, seg.point1.y)
                p2 = (seg.point2.x, seg.point2.y)
                return Region.from_half_plane(p1, p2)
            return result.region
        if isinstance(result, list):
            if len(result) == 0:
                return None
            if len(result) == 1:
                return result[0]
            combined_points: List[Tuple[float, float]] = []
            for r in result:
                combined_points.extend(r._sample_to_points(50))
            if len(combined_points) < 3:
                return None
            hull = Region._convex_hull(combined_points)
            if len(hull) < 3:
                return None
            return Region.from_points(hull)
        return result

    @staticmethod
    def _apply_operation(
        left: Union[_RegionWithSource, Region, List[Region], None],
        right: Union[_RegionWithSource, Region, List[Region], None],
        op: str,
    ) -> Union[Region, List[Region], None]:
        """Apply a boolean operation to two region results."""
        # Extract source info if available
        left_source = left if isinstance(left, _RegionWithSource) else None
        right_source = right if isinstance(right, _RegionWithSource) else None

        # Handle segment intersection with shape specially
        if op == "&":
            result = AreaExpressionEvaluator._handle_segment_intersection(
                left_source, right_source
            )
            if result is not None:
                return result

        # Normalize to regions for standard operations
        left_region = AreaExpressionEvaluator._normalize_to_single(left)
        right_region = AreaExpressionEvaluator._normalize_to_single(right)

        if left_region is None and right_region is None:
            return None

        if op == "&":
            if left_region is None or right_region is None:
                return None
            return left_region.intersection(right_region)

        if op == "|":
            if left_region is None:
                return right_region
            if right_region is None:
                return left_region
            return left_region.union(right_region)

        if op == "-":
            if left_region is None:
                return None
            if right_region is None:
                return left_region
            return left_region.difference(right_region)

        if op == "^":
            if left_region is None:
                return right_region
            if right_region is None:
                return left_region
            return left_region.symmetric_difference(right_region)

        raise ValueError(f"Unknown operator: {op}")

    @staticmethod
    def _handle_segment_intersection(
        left: Optional[_RegionWithSource],
        right: Optional[_RegionWithSource],
    ) -> Optional[Region]:
        """Handle special cases of segment intersection with shapes.

        For arc & segment or circle & segment, creates the enclosed region
        bounded by both the shape's curve and the segment line.
        """
        if left is None or right is None:
            return None

        # Determine which is the segment and which is the shape
        if left.source_type == "segment" and right.source_type != "segment":
            segment_source = left
            shape_source = right
        elif right.source_type == "segment" and left.source_type != "segment":
            segment_source = right
            shape_source = left
        else:
            return None  # Not a segment-shape combination

        segment = segment_source.source_drawable
        shape = shape_source.source_drawable

        if shape_source.source_type == "arc":
            return AreaExpressionEvaluator._arc_segment_enclosed_region(shape, segment)
        elif shape_source.source_type == "circle":
            return AreaExpressionEvaluator._circle_segment_enclosed_region(shape, segment)
        elif shape_source.source_type in ("polygon", "ellipse"):
            # For polygons/ellipses, use half-plane intersection
            p1 = (segment.point1.x, segment.point1.y)
            p2 = (segment.point2.x, segment.point2.y)
            half_plane = Region.from_half_plane(p1, p2)
            return shape_source.region.intersection(half_plane)

        return None

    @staticmethod
    def _line_circle_intersections(
        x1: float, y1: float, x2: float, y2: float,
        cx: float, cy: float, radius: float
    ) -> List[Dict[str, float]]:
        """Find intersections between a line (infinite) and a circle.

        Returns list of dicts with x, y, angle keys.
        """
        dx = x2 - x1
        dy = y2 - y1
        fx = x1 - cx
        fy = y1 - cy

        a = dx * dx + dy * dy
        if abs(a) < 1e-12:
            return []

        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < -1e-9:
            return []

        discriminant = max(discriminant, 0.0)
        sqrt_disc = math.sqrt(discriminant)

        intersections: List[Dict[str, float]] = []
        signs = [-1.0, 1.0] if sqrt_disc > 1e-9 else [0.0]

        for sign in signs:
            t = (-b + sign * sqrt_disc) / (2 * a)
            ix = x1 + t * dx
            iy = y1 + t * dy
            angle = math.atan2(iy - cy, ix - cx)
            intersections.append({"x": ix, "y": iy, "angle": angle})

        return intersections

    @staticmethod
    def _arc_segment_enclosed_region(
        arc: "Drawable", segment: "Drawable"
    ) -> Optional[Region]:
        """Create the enclosed region bounded by an arc curve and a segment.

        Finds intersection points between the segment line and the arc,
        then creates a region bounded by the arc between those points
        and the segment connecting them.
        """
        center = (arc.center_x, arc.center_y)
        radius = arc.radius
        p1 = (arc.point1.x, arc.point1.y)
        p2 = (arc.point2.x, arc.point2.y)

        # Calculate arc angles and sweep
        angle1 = math.atan2(p1[1] - center[1], p1[0] - center[0])
        angle2 = math.atan2(p2[1] - center[1], p2[0] - center[0])

        ccw_sweep = angle2 - angle1
        if ccw_sweep < 0:
            ccw_sweep += 2 * math.pi

        if arc.use_major_arc:
            if ccw_sweep < math.pi:
                sweep = -(2 * math.pi - ccw_sweep)
            else:
                sweep = ccw_sweep
        else:
            if ccw_sweep > math.pi:
                sweep = -(2 * math.pi - ccw_sweep)
            else:
                sweep = ccw_sweep

        # Find intersections between the segment LINE (extended) and the circle
        intersections = AreaExpressionEvaluator._line_circle_intersections(
            segment.point1.x, segment.point1.y,
            segment.point2.x, segment.point2.y,
            center[0], center[1], radius
        )

        if len(intersections) < 2:
            # No intersection or tangent - return full arc region
            return AreaExpressionEvaluator._arc_to_region(arc)

        # Check which intersection points are on the arc
        def angle_on_arc(angle: float) -> bool:
            """Check if an angle is within the arc's sweep."""
            # Normalize angle relative to arc start
            rel = angle - angle1
            if sweep >= 0:
                while rel < 0:
                    rel += 2 * math.pi
                while rel > 2 * math.pi:
                    rel -= 2 * math.pi
                return rel <= sweep + 1e-6
            else:
                while rel > 0:
                    rel -= 2 * math.pi
                while rel < -2 * math.pi:
                    rel += 2 * math.pi
                return rel >= sweep - 1e-6

        arc_intersections = []
        for inter in intersections:
            if angle_on_arc(inter["angle"]):
                arc_intersections.append(inter)

        if len(arc_intersections) < 2:
            # Segment line doesn't cut through the arc - return full arc region
            return AreaExpressionEvaluator._arc_to_region(arc)

        # Create region bounded by arc (between intersections) and segment
        i1_angle = arc_intersections[0]["angle"]
        i2_angle = arc_intersections[1]["angle"]

        # The enclosed region is the arc portion that does NOT contain the arc endpoints
        # Go from i1 to i2 in the direction OPPOSITE to where the arc endpoints are

        # Calculate CCW sweep from i1 to i2
        ccw_sweep_i = i2_angle - i1_angle
        if ccw_sweep_i < 0:
            ccw_sweep_i += 2 * math.pi

        # Check if arc endpoint P1 is in the CCW direction from i1 to i2
        p1_rel = angle1 - i1_angle
        if p1_rel < 0:
            p1_rel += 2 * math.pi

        # If P1 is between i1 and i2 (CCW direction), go CW instead (and vice versa)
        if p1_rel < ccw_sweep_i:
            # P1 is in CCW direction, so go CW (negative sweep)
            arc_sweep = -(2 * math.pi - ccw_sweep_i)
        else:
            # P1 is in CW direction, so go CCW
            arc_sweep = ccw_sweep_i

        # Sample the arc from i1 to i2
        points: List[Tuple[float, float]] = []
        num_points = max(16, int(abs(arc_sweep) * 16))
        for i in range(num_points + 1):
            t = i / num_points
            angle = i1_angle + t * arc_sweep
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))

        if len(points) < 3:
            return AreaExpressionEvaluator._arc_to_region(arc)

        return Region.from_points(points)

    @staticmethod
    def _circle_segment_enclosed_region(
        circle: "Drawable", segment: "Drawable"
    ) -> Optional[Region]:
        """Create the enclosed region (circular segment) cut by a segment from a circle.

        Creates the smaller circular segment (minor segment) on the side
        opposite to the circle center relative to the segment.
        """
        center = (circle.center.x, circle.center.y)
        radius = circle.radius

        intersections = AreaExpressionEvaluator._line_circle_intersections(
            segment.point1.x, segment.point1.y,
            segment.point2.x, segment.point2.y,
            center[0], center[1], radius
        )

        if len(intersections) < 2:
            # No intersection - return full circle
            return Region.from_circle(center, radius)

        i1_angle = intersections[0]["angle"]
        i2_angle = intersections[1]["angle"]

        # Determine which side of the segment the center is on
        seg_p1 = (segment.point1.x, segment.point1.y)
        seg_p2 = (segment.point2.x, segment.point2.y)

        dx = seg_p2[0] - seg_p1[0]
        dy = seg_p2[1] - seg_p1[1]
        cx = center[0] - seg_p1[0]
        cy = center[1] - seg_p1[1]
        cross = dx * cy - dy * cx

        # Calculate sweep to get the arc on the opposite side of the center
        ccw_sweep = i2_angle - i1_angle
        if ccw_sweep < 0:
            ccw_sweep += 2 * math.pi

        # Choose the arc opposite to the center
        if cross >= 0:
            # Center is on left - use right arc (smaller sweep if < pi, larger otherwise)
            if ccw_sweep <= math.pi:
                sweep = ccw_sweep
            else:
                sweep = -(2 * math.pi - ccw_sweep)
        else:
            # Center is on right - use left arc
            if ccw_sweep >= math.pi:
                sweep = ccw_sweep
            else:
                sweep = -(2 * math.pi - ccw_sweep)

        # Sample the arc
        points: List[Tuple[float, float]] = []
        num_points = max(16, int(abs(sweep) * 16))
        for i in range(num_points + 1):
            t = i / num_points
            angle = i1_angle + t * sweep
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))

        if len(points) < 3:
            return None

        return Region.from_points(points)


