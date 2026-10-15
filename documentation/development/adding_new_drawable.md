# Adding a New Drawable

This checklist documents the workflow for introducing a new drawable type to MatHud. Follow each section in order to keep the implementation consistent across managers, renderers, and AI integrations.

## AI-First Requirement
Before implementation, define how the drawable is created/updated/deleted through AI tool calls and how outcomes are explained back to the user in chat. HUD rendering should reflect AI actions deterministically.

## 1. Define Core Constants
1. Create any styling defaults or limits in `static/client/constants.py`.
2. Update `rendering/style_manager.py` if the renderer needs access to the new constants.

## 2. Implement the Drawable Model
1. Add the drawable class under `static/client/drawables/`.
2. Inherit from `drawables.drawable.Drawable` and provide:
   - A comprehensive docstring that describes the drawable's purpose.
   - Required attributes (positions, radii, etc.) with validation.
   - `get_class_name`, serialization via `get_state`, and deep copy logic.
   - Optional helpers for transformations (`translate`, `rotate`) when required so canvas operations behave consistently.
3. Ensure the class remains free of rendering-specific code.
4. For complex drawables with sub-components, place data classes in separate files under `static/client/drawables/` for clarity and reusability.

## 3. Extend Managers and Containers
1. Update `DrawablesContainer` with a convenience property returning instances of the new class.
2. Implement a dedicated manager in `static/client/managers/` for create/delete/retrieve operations.
3. Register the manager within `DrawableManager` and expose delegating methods.
4. Add undo/redo archiving in manager methods before mutating state.
5. When a drawable depends optionally on a parent (for example, circle arcs that can reference an existing circle or stand alone), update dependency cleanup paths so removing either the parent shape or a shared point also removes the child drawable.
6. Update `workspace_manager.py` to handle serialization/deserialization of the new drawable type, including any nested sub-components that need reconstruction, and add its saved bucket to `WorkspaceManager.RESTORED_BUCKETS`.

## 4. Wire the Canvas API
1. Add convenience methods on `Canvas` (create/get/delete) that delegate to the manager.
2. Expose the new methods in `FunctionRegistry`.

## 5. Update AI Function Definitions
1. Extend `static/functions_definitions.py` with JSON schema entries for create/delete commands.
2. Document required parameters, optional styling, and validation limits (length, ranges, etc.).
3. If server-side tools need access to client constants, mirror any new values through `static/mirror_client_modules.py` and import them from there so the schema stays in sync with the Brython defaults.

## 6. Implement Rendering Support

### New Rendering Logic
1. Create a helper in `rendering/shared_drawable_renderers.py` that transforms math-space values to screen-space primitives.
2. Add it to `_HELPERS` in `rendering/cached_render_plan.py` under the class name `get_class_name` returns. Its `ViewMode` says when a cached plan must be rebuilt instead of reprojected: `FIXED` (never), `SCALE` (on zoom), `SCALE_BUCKETED_OFFSET` (also on pans past half a screen) or `SCALE_OFFSET` (also on every pan). Both renderers draw every class listed in `_HELPERS`; nothing else needs registering.
3. Add constants to style manager or adapters if special handling is required.

### Reusing Existing Rendering Logic
If the new drawable shares rendering behavior with an existing type, give its class name the existing helper in `_HELPERS` and populate the properties that helper reads.

### Drawables Drawn by Their Parts
Polygons, graphs and plots have no helper: their segments, points, functions and bars are drawn instead. A new drawable like that goes on `DRAWN_BY_PARTS` in `server_tests/test_drawable_wiring.py`, which fails for a drawable class that is on neither list.

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

