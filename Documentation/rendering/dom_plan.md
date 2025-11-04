# DOM & Runtime Selection Plan

## DOM Changes
1. **Layered surfaces**: Keep `<svg id="math-svg">` but add sibling canvases for Canvas 2D (`<canvas id="math-canvas-2d">`) and WebGL (`<canvas id="math-webgl">`). Position them absolutely inside `#math-container`, stacking order:
   - WebGL (background) for high-performance drawing.
   - Canvas 2D (middle) used when selected.
   - SVG (foreground) for legacy mode and text overlays.
2. **Visibility control**: Apply CSS classes (`.renderer-active`) to toggle which surface is visible based on the selected renderer while keeping pointer events disabled for inactive layers.
3. **Text overlay**: Reserve a lightweight SVG or DOM layer for labels when Canvas/WebGL renderers are active, ensuring crisp text without implementing text rendering in each backend.

## Runtime Selection Flow
1. `rendering/factory.create_renderer()` resolves the backend in order of:
   - Explicit argument passed to `Canvas`
   - `window.MatHudRenderer` global (handy for console toggling)
   - `localStorage['mathud.renderer']`
   - Fallback to SVG.
2. `Canvas` now calls `create_renderer()` by default; future work should store the selected mode on the canvas instance for UI display.
3. `CanvasEventHandler` remains unchanged because renderers share the same `CoordinateMapper` contract.

## Activation & Toggling Steps
1. Add a renderer selector to the chat sidebar (drop-down or toggle) that writes the choice to `localStorage` and reloads the canvas (destroy & rebuild) so `create_renderer()` picks up the new mode.
2. When switching renderers at runtime without reload:
   - Instantiate the new renderer via `create_renderer(mode)`
   - Assign it to `canvas.renderer`
   - Call `canvas._register_renderer_handlers()` and `canvas.draw(apply_zoom=True)`
   - Update DOM classnames to show the corresponding surface.
3. Expose a console helper `window.setMatHudRenderer(mode)` that stores the preference and forces a re-render. This allows developers to switch quickly during profiling.

## Event & Layout Considerations
1. Ensure `math-container` maintains `position: relative` so absolute canvases align with existing pointer math.
2. Keep the 2D and WebGL canvases `pointer-events: none` to preserve the current interaction layer (`math-svg` or a dedicated transparent overlay) until event forwarding is introduced.
3. On resize, `Canvas` should notify alternate renderers via `CoordinateMapper`; both prototypes already resize against the container in `clear`/`render_cartesian`.

