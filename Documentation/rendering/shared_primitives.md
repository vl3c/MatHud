# Renderer Primitive Surface

This document captures the minimal drawing primitives required so SVG, Canvas2D, and WebGL renderers can share drawable algorithms while delegating backend-specific details.

## Primitive List

| Primitive | Description | Example usages |
|-----------|-------------|----------------|
| `stroke_line(start, end, stroke)` | Draw a line segment between two screen points with style (color, width). | Segments, triangle/rectangle edges, grid ticks |
| `stroke_polyline(points, stroke)` | Draw an open path through ordered screen points. | Function plots, multi-segment outlines |
| `stroke_circle(center, radius, stroke)` | Stroke a circle with given center and radius. | Circles |
| `fill_circle(center, radius, fill, stroke?)` | Fill a circle with optional stroke outline. | Point glyphs |
| `stroke_ellipse(center, radius_x, radius_y, rotation_rad, stroke)` | Stroke an ellipse with rotation in radians. | Ellipses |
| `fill_polygon(points, fill, stroke?)` | Fill (and optionally outline) a closed polygon. | Vector arrowheads, generic filled shapes |
| `fill_joined_area(forward, reverse, fill)` | Fill an area defined by two boundary polylines. | Colored regions between functions/segments |
| `stroke_arc(center, radius, start_angle_rad, end_angle_rad, sweep_clockwise, stroke, css_class?)` | Stroke an arc section with direction control (angles in radians); `css_class` is consumed by surfaces that expose DOM styling. | Angle arcs |
| `draw_text(text, position, font, color, alignment)` | Render text at a screen position with alignment/size. | Point labels, function names, angle measures, tick labels |
| `clear_surface()` | Clear renderer surface; typically used before redraw. | Scene resets |
| `resize_surface(width, height)` | Resize backing surface to match container. | Canvas resizing |

### Primitive Parameters

All primitives operate in screen coordinates. Core parameter groups:

- `stroke`: `{ color, width, line_join?, line_cap? }`
- `fill`: `{ color, opacity }`
- `font`: `{ family, size, weight? }`
- `alignment`: `{ horiz: "left"|"center"|"right", vert: "baseline"|"middle"|... }`
- `css_class`: Optional string used by DOM-backed surfaces (`SvgPrimitiveAdapter`) to apply styling hooks; other adapters safely ignore it.

### Drawable→Primitive Mapping

- Point: `fill_circle` + `draw_text`
- Segment: `stroke_line`
- Vector: `stroke_line` + `fill_polygon`
- Circle: `stroke_circle`
- Triangle/Rectangle: multiple `stroke_line`
- Angle: `stroke_arc` + `draw_text`
- Function: `stroke_polyline` + `draw_text`
- Colored Area (functions/segments): `fill_joined_area`
- Ellipse: `stroke_ellipse`
- Cartesian Grid: combinations of `stroke_line`, `draw_text`

### Colored Area Metadata

`build_*_colored_area` renderables attach `color` and `opacity` values to the resulting `ClosedArea`. Shared helpers forward these to primitive adapters so each renderer can respect user-selected fill styling while falling back to theme defaults when metadata is missing.

### Cartesian Axis Helper

`render_cartesian_helper` consumes the canvas dimensions and coordinate mapper to emit axis lines, grid lines, tick marks, and labels through the primitive surface. The helper normalises tick spacing and scale, then:

- draws horizontal and vertical axes via `stroke_line`
- emits grid columns/rows with `stroke_line`
- produces tick marks using short `stroke_line` calls
- prints origin and numeric tick labels using `draw_text`

Because the helper only references `RendererPrimitives`, every backend renderer (SVG, Canvas2D, WebGL) now renders the Cartesian system through the shared pipeline with consistent styling.

### Backend Adapter Responsibilities

Each renderer exposes implementations of the primitives mapped to its drawing API:

- **SVG**: `svg.line`, `svg.path`, `svg.circle`, `svg.ellipse`, `svg.polygon`, `svg.text`
- **Canvas2D**: `ctx.beginPath`, `ctx.moveTo/lineTo`, `ctx.arc`, `ctx.fillText`
- **WebGL**: polyline sampling translated to `_draw_line_strip`/`_draw_lines`; filled shapes fall back to outlined line strips and text remains a no-op pending shader support

By funnelling all drawable logic through these primitives, helper functions can generate consistent geometry while renderers focus on execution details and style integration.

