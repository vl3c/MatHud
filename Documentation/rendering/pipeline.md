# MatHud SVG Rendering Lifecycle

## 1. Canvas Construction and Renderer Wiring
Canvas objects are created in `static/client/main.py`. The viewport dimensions are read from `document['math-svg']`, then `Canvas(width, height)` is instantiated. During `Canvas.__init__` (`static/client/canvas.py`):

1. `CoordinateMapper` holds zoom, pan, and math-to-screen transforms.
2. `Cartesian2Axis` draws the grid and axes using the mapper.
3. `DrawableManager` manages all math-space objects via managers and `DrawablesContainer` for layered access.
4. `SvgRenderer` from `static/client/rendering/svg_renderer.py` is created unless a renderer is injected explicitly; handler registration happens in `_register_renderer_handlers`.
5. The renderer immediately draws the Cartesian grid when `draw_enabled` is true.

## 2. Drawable Lifecycle and Storage
`DrawableManager` (`static/client/managers/drawable_manager.py`) builds specialized managers (points, segments, shapes, functions, colored areas, angles). All new drawables are inserted into `DrawablesContainer`, which separates background colored areas from foreground geometry. `Canvas.add_drawable` only attaches the canvas reference and delegates storage to the manager, keeping render logic out of the models.

### Layering
`DrawablesContainer.get_all_with_layering()` returns colored areas first, then other shapes. `Canvas.draw` relies on this ordering to paint filled regions before outlines.

## 3. Draw Triggers
Rendering happens exclusively through `Canvas.draw(apply_zoom: bool = False)`:

1. Clear the SVG surface with `renderer.clear()`.
2. Redraw the Cartesian grid via `renderer.render_cartesian()`.
3. Iterate through layered drawables, optionally invalidating per-drawable zoom caches (`_invalidate_cache_on_zoom`).
4. Dispatch to the renderer; `_is_drawable_visible` performs quick culling for points, segments, and vectors before rendering.

Primary call sites:

- `Canvas.__init__` draws the initial grid.
- `Canvas.reset()` redraws after state resets.
- User interactions in `CanvasEventHandler` (`static/client/canvas_event_handler.py`) call `canvas.draw()` when zooming (`handle_wheel`, pinch) or panning (`_update_canvas_position`). Zoom operations update the mapper first, then trigger a redraw.
- Manager actions (creation/deletion) typically invoke `canvas.draw()` indirectly through higher-level commands routed via AI or UI flows.

## 4. SVG Renderer Dispatch
`SvgRenderer` maintains a registry of math-class types to handlers. `_register_renderer_handlers` dynamically imports models and binds renderer functions:

- Points, segments, triangles, rectangles render via SVG primitives (circle, line, polygon).
- Circle and ellipse radii are mapped with `CoordinateMapper.scale_value` and optional rotation transforms.
- Vectors reuse the underlying segment plus an arrowhead polygon.
- Angles leverage model-provided arc calculations and label placement.
- Functions are sampled through `FunctionRenderable`, which caches screen-space polylines keyed by visible bounds and scale.
- Colored areas convert renderable outputs into closed SVG paths filled with opacity.
- `render_cartesian` reproduces axes, tick marks, and labels using mapper scale and MathUtils formatting.

If no handler exists or an exception occurs, `Canvas.draw` treats the drawable as unrendered but continues processing, leaving the math model intact.

## 5. Caching Hooks and Invalidation

- `FunctionRenderable` stores screen polylines and recalculates when visible bounds or zoom scale change beyond thresholds. `Canvas.draw(apply_zoom=True)` signals zoom-driven invalidation.
- `Cartesian2Axis` exposes `_invalidate_cache_on_zoom`; the canvas calls it before grid redraws when zooming so tick spacing recalculates.
- Individual drawables can implement `_invalidate_cache_on_zoom()` for custom cache clearing; the canvas invokes it during redraw when `apply_zoom` is True.

## 6. Coordinate Mapping Responsibilities
`CoordinateMapper` centralizes math-to-screen conversions, scaling, and visible bounds queries. Renderers rely on it for:

- Converting positions (`math_to_screen` / `screen_to_math`).
- Scaling scalar values (radius, arrowhead size).
- Determining visible math bounds (used by renderer culling and function sampling).
Event handlers mutate the mapper (pan offsets, scale factor) and store zoom-point metadata so redraws happen using the newest transform state.

## 7. Undo/Redo and State Persistence
Undo/redo operations archive drawable states via `UndoRedoManager`. Restoring snapshots updates model data, after which `Canvas.draw()` re-renders the entire scene using the current renderer; no renderer state is serialized, keeping visual output deterministic.

## 8. Observed Constraints for Future Renderers

- Rendering is side-effect free beyond DOM updates; models never receive renderer callbacks.
- Drawables assume rendering order matches existing layering (colored areas behind shapes).
- Zoom and pan logic must continue to use `CoordinateMapper` to maintain consistent coordinate math and text placement.
- Any alternative renderer must supply `clear`, `render`, and `render_cartesian` methods compatible with the current call sequence, or `Canvas` must be abstracted accordingly in later steps.

