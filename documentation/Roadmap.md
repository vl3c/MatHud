# MatHud — Unified Project Roadmap

## Product Direction (Source of Truth)

MatHud is an AI-augmented mathematics heads-up display: user intent is expressed in chat, AI executes tool workflows, and the HUD canvas visualizes results. The long-term vision extends from a web-based 2D math playground to a real-world visual analysis tool — and ultimately to a wearable HUD on smart glasses that interprets the physical world mathematically.

Priorities at every tier:
1. AI intent interpretation and deterministic execution
2. Clear, explainable visual feedback in the HUD canvas
3. Real-world image understanding and mathematical extraction
4. Platform reach (mobile, wearable, offline)

UI gestures remain secondary unless they directly support AI workflows.

---

## P0 — Reliability & Foundations (implement first)

### Completed
- ~~Refactor oversized classes/functions into smaller composable units~~
- ~~Make `search_tools` the default tool-discovery path~~
- ~~Audit canvas state structure; redesign AI-facing state summaries for large scenes~~
- ~~Deep review of drawable dependency architecture and relationship rules~~
- ~~Deterministic tool-execution logs and replayability for AI action traces~~
- ~~OpenAI OAuth investigation~~ — blocked (subscription and API billing are separate systems; revisit if OpenAI opens subscription credits to third-party apps)
- ~~Strict argument validation: canonicalization of tool arguments before execution; deterministic ordering for ambiguous target resolution (stable tie-break rules)~~
- ~~Static typing rollout: add type stubs or protocols for Brython `browser` module usage in client code~~

### Remaining

#### Android packaging pipeline
1. Research and select a packaging approach (e.g., PWA wrapper via TWA, Cordova, or Capacitor; or a lightweight WebView shell)
2. Create a minimal Android project scaffold that wraps the Flask-served web app in a WebView
3. Configure build scripts (Gradle) for reproducible APK generation
4. Handle local Flask server lifecycle within the Android app (start on launch, stop on background/close)
5. Add touch-friendly viewport meta tags and CSS adjustments for mobile screen sizes
6. Test on physical device and emulator; fix layout/interaction issues
7. Set up CI pipeline for automated APK builds on commit/tag
8. Document the build and deployment process

#### WebGL parity
1. Audit current WebGL renderer against Canvas2D/SVG feature matrix; document missing drawable types
2. Implement text/label rendering in WebGL (bitmap font atlas or SDF text)
3. Add remaining drawable type support (polygons, ellipses, colored areas, angles, graphs) one type at a time
4. Port offscreen compositing and plan caching patterns from Canvas2D renderer
5. Run renderer performance harness against WebGL and compare with Canvas2D/SVG baselines
6. Add WebGL to the client test suite; verify parity with existing renderer tests

---

## P1 — Core User Value (implement after P0)

### Intent Resolution & Constraints

#### ~~Constraint resolution and semantic snapping~~ [done]

#### Coordinate readout and confirmations
1. Define a `CoordinateReadout` data structure returned by tool calls containing resolved coordinates, object references, and snap type
2. Include resolved coordinates in tool call results so the AI can quote exact values in responses
3. Add a transient HUD overlay element that displays coordinates near the cursor or near the resolved target
4. Auto-dismiss the overlay after a timeout or on next interaction
5. Add tests for readout accuracy across snap types (endpoint, midpoint, intersection, on-curve)

#### Tolerance policy management
1. Define a `TolerancePolicy` config object with global defaults (snap radius in math units, grid quantization step)
2. Store the active policy in `Canvas` or a dedicated `SnapPolicyManager`
3. Allow per-command tolerance overrides via an optional `tolerance` argument on relevant tool schemas
4. Expose a `/snap_settings` slash command or tool call to view/update the active policy
5. Add tests verifying that override tolerances take precedence over globals

#### Conflict handling
1. Define a conflict-detection step in the tool execution pipeline: when multiple valid targets exist within tolerance, flag ambiguity
2. Implement an ask-back strategy: return a clarification request to the AI with candidate options and let the AI ask the user
3. Implement a deterministic fallback strategy: use stable tie-break rules (closest first, then alphabetical by name) when ask-back is disabled
4. Add a config flag (`SNAP_CONFLICT_MODE=ask-back|deterministic`) to choose behavior
5. Add tests with intentionally ambiguous scenes to verify both modes

#### Explainable resolution traces
1. Extend tool call result payloads with a `resolution_trace` field: snap type used, candidates considered, winner, distance
2. Format resolution traces in the AI system prompt so the model can narrate them naturally
3. Optionally render a brief trace summary in the tool call log dropdown
4. Add tests verifying trace content for each snap type

### AI-to-User Visual Communication

#### Temporary drawable infrastructure (shared foundation)
1. Create a `TemporaryDrawable` base class (or mixin) with an auto-expiry timer and fade-out animation
2. Add a `TemporaryDrawableManager` that tracks active temporaries, ticks their timers, and removes expired ones on each frame
3. Register renderer support for temporary drawables (distinct z-layer rendered above persistent objects)
4. Define AI tool schemas: `show_pointer`, `highlight_object`, `show_annotation`, etc., all returning a temporary drawable ID
5. Add a `dismiss_temporaries` tool call and auto-dismiss-all on next user message
6. Add tests for creation, rendering, expiry, and manual dismissal

#### Attention pointer
1. Create a `TemporaryArrow` drawable: an arrow from an offset position pointing at a target coordinate, rendered in red with a pulsing animation
2. Add tool schema: `show_pointer(target, label?, duration_ms?)` — target can be coordinates or an object name
3. Renderer support: draw the arrow with an animated tip (CSS animation for SVG, requestAnimationFrame for Canvas2D)
4. Add tests for pointer creation, target resolution, and auto-expiry

#### Highlight pulse
1. Add a `highlight_object(object_name, color?, duration_ms?)` tool call
2. Implement a pulse animation: temporarily change the object's stroke/fill to the highlight color, scale up slightly, then animate back over 1–2 seconds
3. Ensure the original style is restored after the pulse completes
4. Add tests for highlight on each drawable type (point, segment, circle, function, etc.)

#### Annotation bubbles
1. Create a `TemporaryAnnotation` drawable: a rounded-rect speech bubble anchored to a canvas coordinate or object, containing short text
2. Add tool schema: `show_annotation(target, text, duration_ms?)`
3. Renderer support: position the bubble with a tail pointing at the anchor; handle viewport clipping (flip direction if near edge)
4. Add tests for annotation placement, text content, and auto-expiry

#### Trace path
1. Create a `TemporaryTracePath` drawable: an animated dashed polyline that draws itself progressively from point to point
2. Add tool schema: `show_trace_path(points[], duration_ms?, color?)`
3. Implement progressive draw animation: each segment appears sequentially with a configurable pace
4. Add tests for multi-point traces and animation timing

#### Comparison overlay
1. Add a `show_comparison(original_object, transformed_object_or_params, duration_ms?)` tool call
2. Render the original as a ghosted (semi-transparent, dashed) copy alongside the current version
3. Implement fade-in for the ghost and auto-dismiss after duration
4. Add tests for comparison with translation, rotation, and scaling transforms

#### Region spotlight
1. Add a `spotlight(object_names[], duration_ms?)` tool call
2. Render a semi-transparent dark overlay covering the entire canvas, with cutouts (full opacity) for the spotlighted objects
3. Auto-dismiss after duration or on next user interaction
4. Add tests for single-object and multi-object spotlight

#### Step marker sequence
1. Create a `TemporaryStepMarker` drawable: a numbered circle badge positioned at or near a canvas object
2. Add tool schema: `show_step_markers(steps[{target, label?}])` — places all markers at once with sequential numbers
3. Renderer support: fixed pixel-size badges that don't scale with zoom (screen-offset mode)
4. Add tests for marker placement, ordering, and auto-expiry

#### Measurement callout
1. Create a `TemporaryMeasurement` drawable: a dimension line between two points with a centered value label
2. Add tool schema: `show_measurement(from, to, value, unit?, duration_ms?)` for distances; `show_angle_measurement(vertex, p1, p2, value, duration_ms?)` for angles
3. Renderer support: arrow-ended line with text centered above, styled distinctly from permanent labels
4. Add tests for distance and angle measurements

#### Error indicator
1. Create a `TemporaryErrorMarker` drawable: a warning icon placed at or near a problematic object, with an optional tooltip message
2. Add tool schema: `show_error(target, message, duration_ms?)`
3. Renderer support: fixed-size icon with red accent color, rendered above the object
4. Add tests for error marker on degenerate shapes and invalid functions

#### Progress breadcrumb
1. Create a `TemporaryBreadcrumb` set: faded/ghosted previews of upcoming construction steps, rendered with low opacity and dashed outlines
2. Add tool schema: `show_progress(upcoming_steps[{type, params}])` — each step describes a drawable to preview
3. Implement progressive reveal: as the AI completes each step, the corresponding breadcrumb transitions from ghost to solid
4. Add tests for breadcrumb rendering and transition on step completion

### Data & Tabular Workflows

#### Tabular workflow
1. Define a `DataTable` model: named columns (numeric, symbolic, string), row storage, formula support per cell
2. Create a `DataTableManager` on the client side to hold table instances
3. Add tool schemas: `create_table(name, columns[])`, `add_rows(table, rows[])`, `query_table(table, filter?, sort?, columns?)`
4. Add tool schemas: `update_cell(table, row, column, value)`, `add_column(table, name, formula?)`, `delete_table(name)`
5. Implement a HUD-linked table preview panel: a small scrollable table view docked below or beside the canvas
6. Wire table data into the canvas state summary so the AI can reference table contents
7. Add tests for table CRUD, formula evaluation, and state serialization

#### CSV workflow
1. Add a `/csv import` slash command and `import_csv(file)` tool call that accepts an uploaded CSV file
2. Implement CSV parsing with delimiter auto-detection (comma, tab, semicolon), header inference, and type guessing
3. Add validation feedback: report row count, detected columns, type mismatches, missing values
4. Add `export_csv(table, filename?)` tool call that generates a CSV download from a DataTable
5. Handle encoding issues (UTF-8 BOM, Latin-1 fallback)
6. Add tests for various CSV formats, edge cases, and round-trip fidelity

### Plotting Suite

#### Scatter plots
1. Create a `ScatterPlot` drawable: a collection of (x, y) points rendered as small markers
2. Add tool schema: `plot_scatter(x_data[], y_data[], marker?, color?, label?)`
3. Renderer support: batch-render markers efficiently; handle overlap with optional jitter or transparency
4. Add tests for scatter creation, styling, and large dataset performance

#### Histogram
1. Create a `Histogram` drawable: auto-binned bars from a single data array
2. Add tool schema: `plot_histogram(data[], bins?, color?, normalize?)`
3. Implement binning algorithms (Sturges, Scott, or user-specified bin count/edges)
4. Renderer support: render as adjacent bars with optional outline
5. Add tests for binning correctness, normalization, and edge cases

#### Box plots
1. Create a `BoxPlot` drawable: whiskers, box (Q1–Q3), median line, outlier markers
2. Add tool schema: `plot_boxplot(data[], label?, color?, show_outliers?)`
3. Compute quartiles, IQR, whisker bounds, and outlier detection
4. Renderer support: render box, whiskers, median, and outlier dots
5. Add tests for quartile computation and outlier identification

#### Polar plots
1. Create a `PolarPlot` drawable: function r = f(θ) sampled over an angle range
2. Add tool schema: `plot_polar(expression, theta_min?, theta_max?, color?)`
3. Sample the function in polar coordinates and convert to Cartesian for rendering
4. Ensure polar grid (existing `PolarGrid`) is activated when polar plots are present
5. Add tests for common polar functions (rose curves, limaçons, spirals)

#### Implicit plots
1. Create an `ImplicitPlot` drawable: contour at f(x, y) = 0 using marching squares
2. Add tool schema: `plot_implicit(expression, x_range?, y_range?, color?)`
3. Implement marching squares algorithm over a grid within the visible bounds
4. Adaptive refinement near the zero contour for smoother curves
5. Add tests for circles, ellipses, hyperbolas, and non-algebraic implicit curves

#### Plotting from table columns
1. Add tool schema: `plot_from_table(table, x_column, y_column, plot_type?, color?)`
2. Wire into existing plot drawables by extracting column data from `DataTableManager`
3. Auto-label axes from column names
4. Add tests for plotting from tables with various column types

### Geometry & Math Tooling

#### ~~Geometric construction toolkit~~ [done]
#### ~~Relation inspector~~ [done]
#### ~~Transform workflows~~ [done]

#### Linear algebra workflows
1. Add tool schema: `compute_rref(matrix)` — compute reduced row echelon form with step-by-step row operations
2. Add tool schema: `compute_eigenvalues(matrix)` — return eigenvalues and eigenvectors
3. Add tool schema: `compute_decomposition(matrix, type)` — support LU, QR, SVD decompositions
4. Format results with LaTeX notation and step-by-step explanations in the AI response
5. Optionally visualize eigenvectors as arrows on the canvas for 2x2 matrices
6. Add tests for correctness against known matrix decompositions

#### Inequality/system region generation
1. Add tool schema: `shade_inequality(expression, bound_type, x_range?, y_range?, color?)`
2. Implement region sampling: evaluate the inequality on a grid and fill qualifying regions
3. Support systems of inequalities: intersect multiple regions and shade the feasible area
4. Add tool schema: `shade_feasible_region(inequalities[], color?)`
5. Renderer support: semi-transparent fill with optional boundary line
6. Add tests for linear inequalities, quadratic regions, and multi-constraint feasibility

#### Conic and advanced curve workflows
1. Add tool schema: `plot_conic(type, params)` — support ellipse, hyperbola, parabola from standard-form parameters
2. Add tool schema: `identify_conic(expression)` — classify and decompose a general second-degree equation
3. Visualize focus points, directrix, asymptotes, and eccentricity as canvas objects
4. Add parameter exploration: tool call to vary a parameter and show how the conic changes
5. Add tests for each conic type and degenerate cases

### Interaction & Exploration Tools

#### Tracing: root/extrema/intersection discovery
1. Add tool schema: `find_roots(function_name, x_range?)` — locate zeros using bisection/Newton's method
2. Add tool schema: `find_extrema(function_name, x_range?)` — locate local min/max via derivative sign changes
3. Add tool schema: `find_intersections(function1, function2, x_range?)` — locate crossing points
4. Mark discovered points on the canvas with labeled temporary markers
5. Return exact (or numeric) coordinates in the tool result for the AI to narrate
6. Add tests for polynomials, trig functions, and edge cases (tangent intersections, flat extrema)

#### Parameter sweeps and dynamic sliders
1. Define a `ParameterBinding` model: a named parameter, its current value, min, max, and step size
2. Add tool schema: `create_parameter(name, min, max, default, step?)`
3. Add tool schema: `sweep_parameter(name, values[])` — evaluate a function/construction at each value and show results as animation or overlay
4. Store parameter bindings in canvas state for serialization
5. Add tests for parameter creation, sweep execution, and state persistence

#### Interactive sliders
1. Create a slider UI component: a draggable horizontal slider docked in a panel near the canvas
2. Wire slider value changes to the bound parameter in real time; trigger canvas re-render on change
3. Add tool schema: `bind_slider(parameter_name, target_expression_or_object)`
4. Support multiple simultaneous sliders with labeled names and value readouts
5. Add tests for slider creation, value propagation, and multi-slider scenarios

### Statistics Workflows

#### Summaries and distributions
1. Add tool schema: `plot_distribution(type, params, x_range?, shade_from?, shade_to?)` — extend with: uniform, exponential, chi-squared, t-distribution, F-distribution, beta, gamma, Poisson
2. Implement PDF/PMF computation for each new distribution type
3. Add tool schema: `compute_probability(distribution, type, params, from, to)` — calculate P(a < X < b)
4. Add tool schema: `compute_inverse_probability(distribution, type, params, probability)` — find the quantile
5. Add tests for each distribution against known values

#### Residual/regression diagnostics
1. Add tool schema: `plot_residuals(regression_name)` — residual plot (residuals vs fitted values) for an existing regression
2. Add tool schema: `compare_regressions(regression_names[])` — display R², AIC/BIC, and residual statistics side by side
3. Render residual plots as scatter plots on a new or existing canvas area
4. Format comparison results as a table in the AI response
5. Add tests for residual computation and comparison metrics

### Canvas Tab System

#### Tab infrastructure
1. Define a `CanvasTab` model: unique ID, name, type (math/image), own `Canvas` instance (or image reference), coordinate state, drawable set
2. Create a `TabManager` class: ordered list of `CanvasTab` instances, active tab tracking, create/close/switch/rename/reorder
3. Refactor `Canvas` initialization so it can be instantiated multiple times without singleton assumptions
4. Ensure `DrawableManager`, `CoordinateMapper`, `UndoRedoManager` are scoped per tab
5. Add tests for tab lifecycle (create, switch, close, rename) and per-tab state isolation

#### Tab bar UI
1. Add a tab bar HTML element above the canvas area with tab buttons showing name and close icon
2. Implement click-to-switch, drag-to-reorder, double-click-to-rename interactions
3. Add a "+" button to create a new tab (defaults to math canvas)
4. Style the active tab distinctly; show tab type icon
5. Add tests for UI interactions (switching triggers correct canvas swap, close removes tab)

#### Math canvas tab
1. Wrap the existing canvas behavior inside a `CanvasTab` with `type="math"`
2. Ensure the default startup creates one math tab (backward compatible)
3. Verify all existing functionality works identically within a tab context
4. Run full client test suite to confirm no regressions

#### Image canvas tab
1. Define an `ImageCanvasTab` subtype: holds an image source as the base layer
2. Render the image as a fixed background, scaled to fit with aspect ratio preserved
3. Add a transparent drawing overlay on top of the image (reuse renderer infrastructure)
4. Implement coordinate mapping between image pixel space and the overlay's math space
5. Support pan/zoom on the image tab (zoom into image regions, overlay stays aligned)
6. Add tests for image loading, overlay drawing, coordinate mapping, and zoom behavior

#### AI tab management
1. Add tool schemas: `create_tab(type, name?, source?)`, `switch_tab(name_or_id)`, `close_tab(name_or_id)`, `rename_tab(name_or_id, new_name)`, `list_tabs()`
2. Wire tool calls through to `TabManager` methods
3. Include active tab info in the canvas state summary sent to the AI
4. Add chat narration support: when the AI switches tabs, include a brief note in the response
5. Add tests for each tool call and edge cases (close last tab, switch to nonexistent tab)

#### Cross-tab references
1. Define a `CrossTabReference` model: source tab + object/value, target tab + object/value, reference type
2. Add tool schema: `reference_from_tab(source_tab, source_object, target_tab, target_name?)`
3. Display cross-tab references in the canvas state so the AI can narrate provenance
4. Add tests for reference creation, value propagation, and cleanup on tab close

#### Tab type extensibility
1. Define a `TabType` registry/protocol that new tab types can implement (required methods: render, get_state, restore_state, supported_tools)
2. Document the tab type API for future types (comparison, 3D, notebook, data table)
3. Add tests for registry and fallback behavior when an unknown tab type is encountered

---

## P2 — Advanced Capabilities & Expansion

### Construction & Automation

#### Construction history timeline
1. Extend `UndoRedoManager` to store a named history of construction steps (a linear timeline with labels)
2. Add tool schema: `show_construction_history()` — return a numbered list of all steps taken
3. Add tool schema: `rewind_to_step(step_number)` — restore canvas state to a specific point
4. Add tool schema: `replay_from_step(step_number)` — animate forward from a step with a delay between each
5. Implement branch exploration: when rewinding and making a new change, fork the timeline
6. Add a timeline UI element (optional, docked below canvas) with clickable entries
7. Add tests for rewind, replay, branching, and timeline serialization

#### Reusable macros
1. Define a `Macro` model: named sequence of tool calls with parameterized arguments
2. Add tool schema: `record_macro(name)` / `stop_recording()` — capture tool calls into a macro
3. Add tool schema: `run_macro(name, params?)` — execute a stored macro with optional parameter substitution
4. Add tool schema: `list_macros()` / `delete_macro(name)`
5. Store macros in workspace state for persistence
6. Add tests for recording, playback, parameterization, and error handling

#### Reusable analysis templates
1. Define a template format: a macro with structured input/output contracts
2. Add tool schema: `create_template(name, description, input_schema, macro_name)`
3. Add tool schema: `run_template(name, inputs)` — execute with validated inputs
4. Ship 3–5 built-in templates (regression comparison, descriptive stats summary, construction check)
5. Add tests for template creation, input validation, and execution

### Visualization

#### Vector field generation
1. Create a `VectorField` drawable: a grid of arrows representing F(x, y) = (Fx, Fy)
2. Add tool schema: `plot_vector_field(fx_expression, fy_expression, x_range?, y_range?, density?)`
3. Sample the field on a regular grid; scale arrow length by magnitude
4. Optional color coding by magnitude
5. Add tests for constant fields, radial fields, and curl/divergence visualization

#### Contour/heatmap generation
1. Create a `Heatmap` drawable: a colored grid representing f(x, y) values
2. Add tool schema: `plot_heatmap(expression, x_range?, y_range?, color_map?, resolution?)`
3. Add tool schema: `plot_contour(expression, levels?, x_range?, y_range?, color?)`
4. Implement color map options (viridis, plasma, grayscale, diverging)
5. Add tests for heatmap rendering, contour level computation, and color mapping

#### Animation controls
1. Define an `Animation` model: a sequence of frames as canvas state mutations
2. Add tool schema: `create_animation(parameter, start, end, steps, target_expression)`
3. Add tool schema: `animate_locus(parameter, start, end, steps, point_expression)` — trace the locus of a point
4. Implement playback controls: play/pause/stop/speed/loop via tool calls and optional UI buttons
5. Render frame by frame using `requestAnimationFrame`
6. Add tests for animation creation, playback, and locus tracing

#### Numerical methods visualization
1. Add tool schema: `visualize_newton_method(function, x0, iterations?)` — show each step: tangent line, x-intercept, convergence
2. Add tool schema: `visualize_euler_method(ode_expression, x0, y0, h, steps)` — show each Euler step on the slope field
3. Render each iteration using the P1 progress breadcrumb or step marker system
4. Add tests for convergence, divergence, and step-by-step accuracy

### Graph Theory

#### Graph-theory workflow
1. Add tool schema: `generate_graph(type, n, params?)` — generate complete, random, bipartite, or tree graphs
2. Add tool schema: `edit_graph(graph_name, operation, params)` — add/remove vertices and edges
3. Add tool schema: `show_adjacency_matrix(graph_name)` / `show_incidence_matrix(graph_name)`
4. Add tool schema: `playback_algorithm(graph_name, algorithm, params?)` — step through BFS/DFS/Dijkstra/Kruskal with highlighted edges using AI visual communication toolkit
5. Add tests for graph generation, matrix computation, and algorithm step correctness

### Workspace & Productivity

#### Workspace orchestration
1. Define layout modes: single (current), side-by-side (two tabs visible), stacked (top/bottom)
2. Add tool schema: `set_layout(mode, left_tab?, right_tab?)`
3. Add tool schema: `pin_tab(name_or_id, pane)` — pin a tab so it stays visible
4. Implement CSS grid/flexbox layout switching for the canvas container
5. Handle resize events: both panes re-render on layout change
6. Add tests for layout switching, tab pinning, and resize behavior

#### Session snapshots and diff/restore
1. Define a `SessionSnapshot` model: full canvas state (all tabs) + chat history + timestamp + label
2. Add tool schema: `take_snapshot(label?)` / `list_snapshots()` / `restore_snapshot(label_or_index)`
3. Implement state diff: compare two snapshots and list added/removed/modified objects per tab
4. Format diffs as human-readable summaries
5. Add tests for snapshot creation, restoration, and diff accuracy

#### "What changed" summaries
1. After each AI turn with tool calls, auto-generate a brief diff summary
2. Add tool schema: `summarize_changes(since?)` — natural-language summary of changes since a snapshot or turn
3. Format as a collapsible section in the chat
4. Add tests for summary generation with various change types

#### Command palette and shortcuts
1. Implement a command palette modal (Ctrl+K / Cmd+K): searchable list of actions
2. Populate from: slash commands, common tool calls, tab management, layout controls
3. Add configurable keyboard shortcuts for frequent actions
4. Store shortcut customizations in localStorage
5. Add tests for palette search, shortcut binding, and action execution

#### Reproducible command/run logs
1. Extend the action trace system to export a full session log as JSON
2. Add a CLI command: `python -m cli.main replay <session_log.json>` — replay and verify
3. Integrate replay into CI for regression detection
4. Add tests for export format, replay fidelity, and regression detection

### 3D (staged after 2D maturity)

#### 3D graphing and shape workflows
1. Extend WebGL renderer with a 3D camera (perspective/orthographic, orbit controls)
2. Implement 3D coordinate mapper (math-space XYZ to screen-space)
3. Add basic 3D drawables: Point3D, Line3D, Plane, Surface (parametric mesh)
4. Add tool schema: `plot_3d_surface(expression, x_range?, y_range?, color_map?)`
5. Add tool schema: `plot_3d_parametric(x_expr, y_expr, z_expr, t_range?, color?)`
6. Add tool schemas for 3D point/line/plane creation
7. Ensure 3D lives in a dedicated 3D tab type
8. Add tests for 3D rendering, projection correctness, and tool call execution

#### 3D object/scene constraints
1. Implement distance/angle computation in 3D
2. Add tool schemas for 3D intersections (line-plane, plane-plane, line-sphere)
3. Add 3D snap targets (vertex, edge midpoint, face centroid)
4. Add tests for 3D constraint resolution

#### 3D camera/navigation
1. Add tool schema: `set_3d_view(azimuth, elevation, distance)`
2. Add tool schema: `orbit_to(target_object)` — center camera on an object
3. Add tool schema: `set_projection(type)` — switch perspective/orthographic
4. Implement smooth animated transitions between camera positions
5. Add tests for camera positioning and transitions

---

## P3 — Image Analysis & Real-World Math Extraction

### Image Segmentation Integration (Meta SAM)

#### Integrate Meta SAM (server-side)
1. Add SAM 2 (or SAM 3) as a Python dependency; download checkpoint on first use or via setup script
2. Create a `SegmentationService` class that loads the model and exposes `segment(image, prompts?) -> masks[]`
3. Optimize for memory: load model lazily, unload when idle; support ONNX runtime as lighter alternative
4. Add tests for model loading, basic segmentation on test images, and memory cleanup

#### Server-side SAM pipeline
1. Add a Flask route: `POST /api/segment` — accepts image file and optional prompt (point, box, or text)
2. Run SAM inference; return JSON with mask polygons, bounding boxes, confidence scores, and mask image
3. Add caching: store recent results keyed by image hash to avoid re-running
4. Add tests for the route with various image formats and prompt types

#### Interactive segment selection
1. On image canvas tab, render SAM mask boundaries as semi-transparent colored overlays
2. Implement click-to-select: user clicks inside a mask region, that mask is highlighted and selected
3. Implement box-select: user drags a rectangle, SAM re-runs with the box as a prompt
4. Show selected mask metadata (area, bounding box, centroid) in a tooltip
5. Add tests for click detection within masks, box prompt re-segmentation, and multi-mask selection

#### Text-prompted segmentation (SAM 3)
1. Add tool schema: `segment_image(image_tab, text_prompt)` — SAM 3 returns matching masks
2. Wire the text prompt through to `SegmentationService`
3. Display returned masks as highlighted overlays on the image tab with labels
4. Add tests with various text prompts against test images

#### Segment-to-canvas projection
1. For a selected mask, extract the contour polygon (simplify with Douglas-Peucker)
2. Fit geometric primitives: circle (Taubin), ellipse, rectangle, triangle; pick best by residual error
3. Create the corresponding canvas drawable on the math tab with computed dimensions
4. Add tool schema: `extract_shape(image_tab, mask_id, target_tab?)`
5. Add tests for contour extraction, primitive fitting accuracy, and drawable creation

### Mathematical Extraction from Images

#### Shape detection pipeline
1. Implement edge detection (Canny) and contour extraction (OpenCV `findContours`) as preprocessing
2. Classify detected contours: line (Hough), circle (Hough circles), ellipse (`fitEllipse`), polygon (`approxPolyDP`)
3. Convert classified shapes to canvas drawables with estimated dimensions
4. Add tool schema: `detect_shapes(image_tab, types_filter?)`
5. Render detected shapes as overlays on the image tab for review before committing
6. Add tests for detection accuracy on synthetic and real-world images

#### OCR for math
1. Integrate an OCR model: Tesseract for printed text, or specialized math OCR (e.g., pix2tex for LaTeX)
2. Add a Flask route: `POST /api/ocr` — accepts an image region, returns recognized text or LaTeX
3. Add tool schema: `extract_math(image_tab, region?)` — run OCR on full image or selected region
4. Parse recognized LaTeX into symbolic expressions and inject into chat as a solvable prompt
5. Handle common OCR errors: post-process with expression validator
6. Add tests for printed equations, handwritten equations, and edge cases

#### Measurement estimation
1. Add tool schema: `set_reference_scale(image_tab, point1, point2, real_distance, unit)` — calibrate image
2. Compute pixels-per-unit from the reference scale
3. Add tool schema: `measure_distance(image_tab, point1, point2)` — return estimated real-world distance
4. Add tool schema: `measure_area(image_tab, mask_id_or_polygon)` — return estimated area
5. Render measurement overlays on the image tab
6. Add tests for calibration accuracy and measurement computation

#### Curve fitting from images
1. Add tool schema: `trace_curve(image_tab, points[])` — user or AI clicks points along a curve
2. Alternatively, extract curve points from a SAM mask edge or detected contour
3. Fit candidate functions: polynomial, conic, exponential, logarithmic, sinusoidal
4. Return best-fit with R² and plot on the math tab
5. Add tool schema: `fit_curve_from_image(image_tab, mask_id_or_contour, function_type?)`
6. Add tests for fitting accuracy against known curves

### AI Image Annotation & Show-and-Tell Mode

#### AI image annotation mode
1. Add a per-image-tab flag: `annotation_mode: bool`
2. Add tool schema: `enable_annotation_mode(image_tab)` / `disable_annotation_mode(image_tab)`
3. Ensure all P1 temporary drawable tools work on image tab overlays (using image-space coordinates)
4. Add tests for annotation creation on image tabs and coordinate correctness

#### "Look at this" workflow
1. Define the multi-step orchestration: AI creates image tab -> annotates -> narrates -> switches to math tab -> constructs
2. Add a convenience tool schema: `show_and_tell(image_source, annotations[], math_constructions[])` for batch orchestration
3. Alternatively test that chaining individual tool calls works reliably
4. Add integration tests for the full flow with a sample image

#### Annotation layer on images
1. Ensure the image tab's transparent overlay supports the full renderer pipeline
2. Verify all P1 visual communication drawables render correctly on top of images
3. Add an optional semi-transparent backdrop behind annotations for readability on busy backgrounds
4. Add tests for annotation visibility on dark, light, and busy images

#### Image-to-math pipeline narration
1. Define a structured narration format: AI interleaves tab switches with chat messages
2. Ensure tab switch tool calls produce brief chat annotations
3. Test with a full pipeline: upload -> segmentation -> extraction -> derivation -> solution
4. Add integration tests verifying the narrative flow

#### Pinned image reference
1. When in side-by-side layout, allow an image tab to be pinned in one pane
2. Add tool schema: `pin_image_reference(image_tab)` — activates side-by-side with image locked
3. Math tab interactions happen in the other pane; image annotations still work on the pinned tab
4. Add tests for pinned layout activation and interaction isolation

### Image-Grounded AI Analysis

#### "Analyze this image" workflow
1. Define orchestration: user uploads image -> AI creates image tab -> runs detection + SAM -> summarizes
2. Add tool schema: `analyze_image(image_source)` — convenience wrapper for full pipeline
3. Return structured summary: detected shapes, symmetries, angles, areas with confidence
4. Render detected shapes as overlays on the image tab
5. Add tests for end-to-end pipeline on sample images

#### Side-by-side view
1. When AI runs `analyze_image`, auto-activate side-by-side layout
2. Link extracted math-tab objects to image-tab counterparts (matching colors, numbered labels)
3. Add tests for layout activation and cross-tab visual linking

#### Iterative refinement
1. Add tool schema: `refit_shape(image_tab, mask_id, target_type)` — re-classify a shape
2. Add tool schema: `set_real_dimension(image_tab, mask_id, dimension, value, unit)` — override with known value
3. Propagate recalibrations to dependent measurements and math-tab objects
4. Add tests for refinement accuracy and cascading updates

#### Image background layer
1. Render the image at the base z-layer, below all drawable overlays
2. Support opacity control via tool schema: `set_image_opacity(image_tab, opacity)`
3. Add tests for opacity rendering and z-layer ordering

### Physics & Real-World Modeling from Images

#### Trajectory analysis
1. Add tool schema: `fit_trajectory(image_tab, points[])` — fit projectile motion equations
2. Compute initial velocity, launch angle, predicted landing point
3. Overlay fitted parabola on image tab; plot equations on math tab
4. Add tests for fitting accuracy against known trajectories

#### Structural geometry
1. Add tool schema: `analyze_structure(image_tab)` — detect edges, vertices, joints
2. Compute angles at joints, segment lengths (with calibration), identify load-bearing triangles
3. Overlay structural elements on image tab; display measurements
4. Add tests for detection on simple truss and frame images

#### Symmetry and pattern detection
1. Add tool schema: `detect_symmetry(image_tab, region?)` — analyze for reflective, rotational, translational symmetry
2. Classify symmetry type; visualize axes/centers on image tab
3. For periodic patterns, detect repeating unit cell and tile type
4. Add tests for known patterns and expected classifications

#### Perspective geometry
1. Add tool schema: `detect_vanishing_points(image_tab)` — find vanishing points from parallel lines
2. Add tool schema: `compute_perspective_correction(image_tab, vanishing_points[])` — compute homography
3. Overlay vanishing points, horizon line, and perspective grid on image tab
4. Use homography to convert image measurements to real-world proportions
5. Add tests for detection and correction accuracy

---

## P4 — Platform Expansion & UX

### Export & Sharing

#### Canvas export (PNG/SVG/PDF)
1. Add a `download_canvas(format)` tool call and a UI export button
2. PNG export: use `canvas.toDataURL()` or serialize SVG to canvas; trigger browser download
3. SVG export: serialize current SVG DOM to standalone `.svg` with embedded styles
4. PDF export: render SVG to PDF server-side (e.g., `cairosvg` or `weasyprint`)
5. Support choosing which tab to export
6. Add tests for each format and visual fidelity

#### LaTeX/document export
1. Collect AI explanations and derivations from current session chat history
2. Format as LaTeX document with sections per turn, embedded equations, and canvas screenshots
3. Alternatively export as Markdown with embedded images
4. Add tool schema: `export_session(format, filename?)`
5. Add tests for document structure and LaTeX compilation

#### Shareable workspace links
1. Add `share_workspace()` tool call: serialize workspace and upload to server endpoint
2. Generate unique URL slug; endpoint serves the workspace JSON
3. Opening the link loads MatHud with the shared workspace (read-only or clone mode)
4. Add tests for serialization, URL generation, and loading from link

#### Notebook-style cell mode
1. Define a notebook tab type: ordered cells (prose, math, code, canvas snapshot)
2. Add tool schemas: `create_cell(type, content)`, `edit_cell(index, content)`, `delete_cell(index)`, `reorder_cells(new_order[])`
3. Render as a vertical scrollable document with editable cells
4. Support re-executing math/code cells and updating downstream cells
5. Add tests for cell CRUD, rendering, and re-execution

### Collaboration

#### Real-time collaborative sessions
1. Add WebSocket support to Flask backend (via `flask-socketio` or similar)
2. Implement OT or CRDT for canvas state synchronization
3. Broadcast tool call executions and canvas mutations to all clients
4. Show remote cursors and active user indicators
5. Handle conflict resolution for simultaneous edits
6. Add tests for multi-client sync, conflicts, and disconnection

#### Teacher/student mode
1. Define roles: teacher (full edit) and student (view + restricted edit)
2. Broadcast teacher annotations and tool calls to students in real time
3. Students create objects in a separate layer without affecting teacher's canvas
4. Add role switcher and session invite mechanism
5. Add tests for role-based permissions and broadcasting

### AI Enhancements

#### Conversation persistence across sessions
1. Serialize chat history (messages, tool calls, results) to JSON alongside workspace data
2. Auto-save on workspace save
3. Restore chat history into the chat interface on workspace load
4. Truncate or summarize older history to stay within AI context limits
5. Add tests for serialization, restoration, and truncation

#### AI-initiated suggestions
1. After AI completes a response, optionally run a secondary prompt for a suggested next step
2. Display as a dismissible chip below the AI's response
3. Clicking the chip sends it as a new user prompt
4. Add toggle to enable/disable proactive suggestions
5. Add tests for suggestion generation and UI interaction

#### Multi-turn planning
1. Add tool schema: `propose_plan(steps[])` — AI sends planned steps before executing
2. Render as a numbered checklist with approve/modify/reject buttons per step
3. On approve-all, execute sequentially; on modify, let user edit and re-submit; on reject, cancel
4. Add tests for plan rendering, approval flow, and partial modification

#### Voice input/output
1. Integrate Web Speech API: add microphone button next to chat input
2. Transcribe speech and insert into chat input; send on voice command or silence timeout
3. Add "read aloud" button on AI messages using speech synthesis
4. Use math-aware pronunciation
5. Add tests for speech recognition integration and synthesis triggering

### UX Polish

#### Dark mode / theme support
1. Define CSS custom properties for all colors
2. Create light and dark theme value sets
3. Add theme toggle in UI header (light / dark / system-auto)
4. Store preference in localStorage; apply on load before first paint
5. Ensure all renderers respect theme colors
6. Add tests for theme switching and persistence

#### Touch and stylus input
1. Add touch event handlers (`touchstart`, `touchmove`, `touchend`) in `CanvasEventHandler`
2. Implement pinch-to-zoom and two-finger pan gestures
3. Map stylus pressure to line weight (for future freehand annotation)
4. Add tap-to-select, long-press for context menu
5. Test on tablet devices; fix layout issues
6. Add tests for gesture recognition and coordinate mapping

#### PWA support
1. Create `manifest.json` with app name, icons, theme color, `display: standalone`
2. Implement service worker for offline caching of static assets
3. Cache main HTML template; show offline banner when server unreachable
4. Add install prompt for "Add to Home Screen"
5. Test offline behavior; AI chat shows offline message
6. Add tests for service worker registration and offline fallback

#### Step-by-step animation mode
1. Add a `step_mode: bool` toggle in the AI interface
2. When enabled, pause after each tool call; show "next step" button
3. During pause, use P1 progress breadcrumb to preview upcoming steps
4. On "next step", execute next tool call and advance breadcrumb
5. Add "skip to end" button for executing all remaining steps
6. Add tests for step mode toggling, pause/resume, and skip-to-end

### Educational

#### Guided lesson/tutorial system
1. Define a lesson format: JSON with steps (instruction text, expected tool calls, success criteria, hints)
2. Create a `LessonRunner` that loads lessons, displays instructions, monitors tool calls
3. Add tool schema: `start_lesson(name)` / `list_lessons()`
4. Ship 3–5 starter lessons
5. Provide per-step feedback: correct or hint
6. Add tests for lesson loading, step evaluation, and success detection

#### Quiz/challenge mode
1. Define a quiz format: JSON with questions, expected answers, and scoring
2. Create a `QuizRunner` that poses questions, evaluates canvas actions, scores results
3. Add tool schema: `start_quiz(name)` / `list_quizzes()`
4. Ship 3–5 starter quizzes
5. Display score summary with review of incorrect answers
6. Add tests for quiz loading, answer evaluation, and scoring

---

## P5 — Smart Glasses & Wearable HUD (long-term vision)

This tier represents the full realization of the "Mathematics Heads-Up Display" concept: MatHud running on AR smart glasses, analyzing the real world through cameras, and presenting mathematical insights as a transparent overlay.

### AR Canvas Overlay

#### World-anchored canvas objects
1. Research AR frameworks (WebXR, ARCore/ARKit via bridge, or native SDK)
2. Implement spatial anchor API: create anchor at detected surface, bind canvas object
3. Update object positions each frame based on head tracking / SLAM
4. Handle anchor persistence across sessions
5. Add tests for anchor stability under head movement

#### Ghosted solution overlay
1. Render semi-transparent "ideal" shape aligned to detected real-world object
2. Compute alignment using detected vs ideal shape parameters
3. Display deviation metrics as annotation bubbles
4. Add tests for alignment accuracy

#### Dynamic measurement lines
1. Render dimension lines between spatial anchors with live-updating values
2. Update in real time as user moves or anchors refine
3. Support angle measurement between three anchors
4. Add tests for measurement accuracy

#### Construction guides
1. Project helper lines (level, plumb, angle bisector) from spatial anchors into AR view
2. Update guide positions in real time
3. Add snapping to guide lines
4. Add tests for projection accuracy

### Continuous Scene Understanding

#### Live camera feed analysis
1. Capture camera frames at configurable frequency
2. Run lightweight edge detection and shape classification per frame
3. Register detected shapes as transient canvas objects; merge when confident
4. Adaptive frequency: increase when scene changes, decrease when stable
5. Add tests for detection latency and object stability

#### Real-time SAM segmentation
1. Run SAM on camera frames using optimized mobile model
2. Cache results; re-run only when scene changes significantly
3. Overlay mask boundaries in AR view as colored outlines
4. Add tests for segmentation speed and mask consistency

#### Material/surface recognition
1. Classify detected surfaces: flat, curved, periodic/textured, organic
2. For curved surfaces, fit mathematical models (sphere, cylinder, paraboloid)
3. For periodic textures, detect repeating pattern parameters
4. Add tests for classification accuracy

### Voice-First Interaction

#### Always-listening wake word
1. Implement on-device wake word detection using small keyword-spotting model
2. On detection, activate full speech-to-text
3. Provide audio feedback on activation
4. Add tests for detection accuracy and false positive rate

#### Spatial audio responses
1. Integrate spatial audio rendering
2. Position audio cues relative to objects being described
3. Implement directional reference phrases
4. Add tests for audio positioning

#### Gaze-anchored context
1. Integrate eye-tracking data from glasses SDK
2. Map gaze direction to canvas/world coordinates
3. Use gaze as implicit prompt target for deictic references ("that", "this")
4. Add tests for gaze-to-coordinate mapping

#### Dictation mode
1. Extend voice input with math-aware grammar recognition
2. Map spoken phrases to LaTeX
3. Display parsed expression in HUD for confirmation
4. Add tests for expression recognition accuracy

### Live Analysis Modes

#### Structural analysis mode
1. Detect linear structural elements in camera view
2. Model as 2D truss; compute forces and moment diagrams
3. Overlay force arrows and labels in AR
4. Add tests for force computation accuracy

#### Trajectory mode
1. Track moving object across frames (SAM 2 video tracking or optical flow)
2. Fit trajectory data to kinematic equations in real time
3. Predict future positions and overlay path
4. Display velocity, acceleration, time-to-landing
5. Add tests for tracking accuracy and prediction error

#### Surveying mode
1. Use stereo vision or depth sensor for distance estimation
2. Implement triangulation from multiple viewpoints
3. Compute elevation differences and slopes
4. Display overlay with uncertainty bounds
5. Add tests for distance estimation accuracy

#### Symmetry/pattern detector
1. Analyze patterns using autocorrelation or Fourier analysis
2. Classify symmetry type (reflective, rotational, translational, wallpaper group)
3. Overlay symmetry axes, centers, and unit cells
4. Add tests for classification accuracy

#### Optical mode
1. Detect reflective/transparent objects (lenses, mirrors)
2. Compute and overlay ray diagrams
3. Display focal points, optical centers, magnification
4. Add tests for ray tracing accuracy

### Educational AR

#### "Explain what I see"
1. On command, capture current view, run full analysis
2. Generate structured explanation narrated by AI
3. Overlay annotations on each identified feature
4. Add tests for explanation completeness and annotation placement

#### Interactive proofs in the real world
1. Detect hand-drawn shape on paper via camera + shape detection
2. Run construction proof step by step, overlaying on real drawing
3. Use P1 step markers and breadcrumbs for pacing
4. Add tests for detection and overlay alignment

#### Physics sandbox
1. Detect physical setups (inclined plane, pendulum, spring)
2. Overlay free-body diagrams, energy charts, governing equations
3. Animate predicted motion alongside real object
4. Add tests for setup recognition and computation accuracy

#### Nature math
1. Detect natural mathematical patterns (spirals, branching, petals, honeycomb)
2. Classify pattern type and overlay mathematical model
3. Display computed parameters (golden ratio, fractal dimension, symmetry count)
4. Add tests for pattern classification accuracy

### Practical / Professional AR

#### Navigation math
1. Integrate GPS/compass data with AR view
2. Compute bearing angles, distances, shortest paths to waypoints
3. Overlay direction arrows and distance markers
4. Add tests for bearing accuracy

#### Cooking/chemistry
1. Detect measuring vessels and labels via OCR and shape detection
2. Compute volume from vessel geometry
3. Display unit conversions, scaling ratios, recipe adjustments
4. Add tests for vessel detection and volume estimation

#### Fitness/sports
1. Track motion using pose estimation and object tracking
2. Compute launch angles, trajectories, spin rates
3. Overlay ideal vs actual trajectories with deviation metrics
4. Add tests for motion tracking and angle computation

#### Construction/DIY
1. Detect walls, floors, edges via plane detection
2. Compute angles, surface areas, material quantities
3. Project level lines, plumb lines, alignment guides
4. Add tests for measurement accuracy

#### Astronomy
1. Integrate star catalog and GPS/compass
2. Overlay constellation outlines, orbital paths, angular distances
3. Compute rise/set times and positions
4. Add tests for star identification and angular distance

### Social / Multi-User AR

#### Shared HUD sessions
1. Implement spatial anchor sharing protocol (cloud anchors)
2. Synchronize canvas overlays across users
3. Show remote user gaze indicators
4. Add tests for synchronization latency and consistency

#### Teacher/student AR mode
1. Extend P4 teacher/student model to AR
2. Broadcast teacher spatial annotations to student devices
3. Align annotations to each student's viewpoint via shared anchors
4. Add tests for annotation alignment across viewpoints

#### Remote expert
1. Stream local camera feed to remote collaborator
2. Remote can place canvas objects in shared spatial frame
3. Local user sees annotations anchored in their world
4. Add tests for stream quality, latency, and anchor accuracy

### Technical Architecture for AR

#### Edge compute pipeline
1. Profile model inference on target hardware
2. Partition workloads: lightweight on-device, heavy on phone/cloud
3. Implement async task queue with local caching
4. Add tests for latency and graceful degradation

#### Spatial anchor system
1. Select framework (ARCore Cloud Anchors, ARKit, or cross-platform)
2. Implement anchor create/save/load/delete lifecycle
3. Handle drift correction and re-localization
4. Add tests for persistence and re-localization accuracy

#### Gaze/gesture input layer
1. Integrate eye tracking SDK
2. Map gaze to canvas/world coordinates
3. Implement hand gesture recognition (pinch, swipe, palm)
4. Add tests for accuracy and false positive rejection

#### Low-latency streaming vision
1. Implement frame capture pipeline with configurable FPS/resolution
2. Add scene-change detection to trigger analysis only when needed
3. Buffer and batch frames for efficient inference
4. Add tests for latency, detection accuracy, and battery impact

#### Battery-aware modes
1. Define power profiles: full, balanced, minimal
2. Implement automatic switching based on battery thresholds
3. Expose user control to override
4. Add tests for switching and threshold behavior

#### Privacy controls
1. Camera only active when explicitly enabled (opt-in)
2. Local-only processing mode: all inference on-device
3. Face detection and blurring before cloud sends
4. Text/PII redaction before cloud sends
5. Add tests for blurring, PII detection, and local-only enforcement

---

## Parked (revisit when core AI workflows mature)

- Differential equations solver and direction field visualization
- Numerical ODE solvers
- Truth tables and boolean algebra workflows
- Financial mathematics (compound interest, annuities, amortization)
- Hypothesis testing (z-test, t-test, chi-square, F-test), ANOVA, confidence intervals
- Probability plots (Q-Q, P-P) and cumulative frequency tables

---

## Milestones

| Milestone | Focus | Tier |
|-----------|-------|------|
| **1 — Stability** | Complete remaining P0 reliability and execution foundations (Android pipeline, WebGL parity) | P0 |
| **2 — AI Interaction Core** | Coordinate confirmations, AI-to-user visual communication, tabular + CSV flows | P1 |
| **3 — AI Plotting + Stats** | Plotting suite, distribution/probability/stat-summary workflows, regression diagnostics | P1 |
| **4 — AI Geometry + Transforms** | Linear algebra, inequality regions, conic curves | P1 |
| **5 — Canvas Tab System** | Tab infrastructure, math/image canvas tabs, AI tab management, cross-tab references | P1 |
| **6 — AI Exploratory Workflow** | Tracing, sliders, interactive graph workflows, algorithm playback | P1–P2 |
| **7 — Image Analysis Foundation** | SAM integration, shape detection, OCR for math, AI image annotation mode | P3 |
| **8 — Real-World Math Extraction** | Measurement estimation, structural geometry, physics extraction, image-to-math pipeline | P3 |
| **9 — Advanced Canvas** | Construction history, macros, animations, vector fields, heatmaps, workspace orchestration, 3D staged rollout | P2 |
| **10 — Platform & Collaboration** | Export, sharing, collaborative sessions, conversation persistence, voice I/O, PWA, dark mode | P4 |
| **11 — Educational** | Guided lessons, quiz mode, step-by-step animation | P4 |
| **12 — Wearable HUD Prototype** | AR overlay proof-of-concept, spatial anchors, voice-first interaction, live camera analysis | P5 |
| **13 — Full AR HUD** | Live analysis modes, educational AR, practical/professional modes, social multi-user AR | P5 |
