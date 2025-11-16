# Adding a New Drawable

This checklist documents the workflow for introducing a new drawable type to MatHud. Follow each section in order to keep the implementation consistent across managers, renderers, and AI integrations.

## 1. Define Core Constants
1. Create any styling defaults or limits in `static/client/constants.py`.
2. Update `rendering/style_manager.py` if the renderer needs access to the new constants.

## 2. Implement the Drawable Model
1. Add the drawable class under `static/client/drawables/`.
2. Inherit from `drawables.drawable.Drawable` and provide:
   - A comprehensive docstring that describes the drawable’s purpose.
   - Required attributes (positions, radii, etc.) with validation.
   - `get_class_name`, serialization via `get_state`, and deep copy logic.
   - Optional helpers for transformations (`translate`, `rotate`) when required so canvas operations behave consistently.
3. Ensure the class remains free of rendering-specific code.

## 3. Extend Managers and Containers
1. Update `DrawablesContainer` with a convenience property returning instances of the new class.
2. Implement a dedicated manager in `static/client/managers/` for create/delete/retrieve operations.
3. Register the manager within `DrawableManager` and expose delegating methods.
4. Add undo/redo archiving in manager methods before mutating state.
5. When a drawable depends optionally on a parent (for example, circle arcs that can reference an existing circle or stand alone), update dependency cleanup paths so removing either the parent shape or a shared point also removes the child drawable.

## 4. Wire the Canvas API
1. Register the drawable when the canvas configures renderers (`Canvas._register_renderer_handlers`).
2. Add convenience methods on `Canvas` (create/get/delete) that delegate to the manager.
3. Expose the new methods in `FunctionRegistry`.

## 5. Update AI Function Definitions
1. Extend `static/functions_definitions.py` with JSON schema entries for create/delete commands.
2. Document required parameters, optional styling, and validation limits (length, ranges, etc.).
3. If server-side tools need access to client constants, mirror any new values through `static/constants_sync.py` and import them from there so the schema stays in sync with the Brython defaults.

## 6. Implement Rendering Support
1. Create a helper in `rendering/shared_drawable_renderers.py` that transforms math-space values to screen-space primitives.
2. Register the helper in `_HELPERS` inside `rendering/optimized_drawable_renderers.py`.
3. Ensure each renderer (`canvas2d_renderer.py`, `svg_renderer.py`, `webgl_renderer.py`) registers the new drawable class and delegates to `_render_drawable`.
4. Add constants to style manager or adapters if special handling is required.

## 7. Add Tests
1. Introduce unit tests in `static/client/client_tests/` covering:
   - Drawable validation and state serialization.
   - Manager operations (create/delete with undo archiving).
   - Rendering helper behavior via `SimpleMock` primitives.
2. Update or create integration tests if the drawable interacts with existing shapes.

## 8. Update Documentation
1. Describe new capabilities in relevant documents (rendering guides, rollout checklist, AI capability docs).
2. Keep documentation focused on the current state of the repository.

## 9. Verify Locally
1. Run targeted client tests with `python -m pytest static/client/client_tests/<test_file>.py`.
2. Launch the application if possible and confirm the drawable renders as expected.
3. Review telemetry or renderer logs if the drawable relies on optimized plans.

