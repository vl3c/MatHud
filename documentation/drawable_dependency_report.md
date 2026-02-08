# Drawable Dependency Architecture Report

Date: 2026-02-08  
Branch: `feat/drawable-dependency-audit`

## 1. Scope and Objective
This report documents how drawable dependencies are modeled, registered, traversed, and cleaned up across MatHud. It also identifies interaction/cascade behaviors and edge cases to guide hardening work.

Primary sources:
- `static/client/managers/drawable_dependency_manager.py`
- `static/client/managers/*.py` (creation/deletion/update paths)
- `static/client/drawables/*.py`

## 2. Core Dependency Model

## 2.1 Data structures
Dependency graph state is maintained by `DrawableDependencyManager`:
- `_parents: Dict[int, Set[int]]` maps child ID -> direct parent IDs
- `_children: Dict[int, Set[int]]` maps parent ID -> direct child IDs
- `_object_lookup: Dict[int, Drawable]` maps object ID -> object reference

## 2.2 Edge semantics
- `register_dependency(child, parent)` means: child depends on parent.
- Graph traversal utilities:
  - `get_parents`, `get_children` for direct relations
  - `get_all_parents`, `get_all_children` for transitive closure
- `resolve_dependency_order(drawables)` performs parent-first topological-style ordering (cycle-tolerant via visited set).

## 2.3 Validation and guards
Current behavior includes:
- point->point edge suppression (`_should_skip_point_point_dependency`)
- `None` endpoint guard in `register_dependency`
- callable `get_class_name` guard in `register_dependency`
- orphan pruning in `unregister_dependency` (`_prune_lookup_if_orphaned`)

## 3. Drawable Relationship Catalog

Class inventory from `static/client/drawables` (`get_class_name()`):
- `Angle`, `Bar`, `BarsPlot`, `Circle`, `CircleArc`, `ClosedShapeColoredArea`, `ColoredArea`, `ContinuousPlot`, `Decagon`, `DirectedGraph`, `DiscretePlot`, `Ellipse`, `Function`, `FunctionSegmentBoundedColoredArea`, `FunctionsBoundedColoredArea`, `GenericPolygon`, `Graph`, `Heptagon`, `Hexagon`, `Label`, `Nonagon`, `Octagon`, `ParametricFunction`, `Pentagon`, `PiecewiseFunction`, `Plot`, `Point`, `Quadrilateral`, `Rectangle`, `Segment`, `SegmentsBoundedColoredArea`, `Tree`, `Triangle`, `UndirectedGraph`, `Vector`.

## 3.1 Explicitly dependency-managed classes
### Primitive and geometric structure
- `Point`: no parents.
- `Segment`: parents = `point1`, `point2`.
- `Vector`: parent = `segment`.
- `Triangle`: parents = `segment1..segment3`.
- `Rectangle`: parents = `segment1..segment4`.
- `Circle`: parent = `center`.
- `Ellipse`: parent = `center`.
- `Angle`: parents = `segment1`, `segment2`, `vertex_point`, `arm1_point`, `arm2_point`.
- `CircleArc`: parents = `point1`, `point2`, optionally `circle`.

### Colored area variants
- `SegmentsBoundedColoredArea`: parents = `segment1`, `segment2`.
- `FunctionSegmentBoundedColoredArea`: parents = `func`, `segment`.
- `FunctionsBoundedColoredArea`: parents = `func1`, `func2`.
- `ColoredArea`: parents = optional `function` and iterable `segments`.
- `ClosedShapeColoredArea`: parents = iterable `segments`, optional `circle`, optional `ellipse`, optional `chord_segment`.
- Any class ending with `ColoredArea`: generic fallback path (`function` + `segments` attributes if present).

### Graph family
- `Graph`, `DirectedGraph`, `UndirectedGraph`, `Tree`: parents collected from `_segments`, `_vectors`, `_isolated_points`.

## 3.2 Drawable classes currently not explicitly dependency-managed
These classes are valid drawables but have no dedicated dependency branch in `analyze_drawable_for_dependencies`:
- `Bar`, `BarsPlot`, `ContinuousPlot`, `DiscretePlot`, `Plot`
- `ParametricFunction`, `PiecewiseFunction`
- polygon family type labels: `Quadrilateral`, `Pentagon`, `Hexagon`, `Heptagon`, `Octagon`, `Nonagon`, `Decagon`, `GenericPolygon`
- `Label`

Interpretation:
- For many of these, “no dependencies” is currently correct (pure renderables or expression-only objects).
- For polygon family types, dependency is represented at concrete `Triangle`/`Rectangle` level and via segments/points, not by those subtype labels.

## 4. Registration and Cleanup Paths by Manager

## 4.1 Registration paths
- Automatic analyzer registration (common pattern):
  - `SegmentManager.create_segment*`
  - `CircleManager.create_circle`
  - `EllipseManager.create_ellipse`
  - `PolygonManager.create_polygon`
  - `ColoredAreaManager.create_*` (except expression-only region case)
  - `GraphManager.create_graph`
  - `BarManager`, `StatisticsManager`, `UndoRedoManager` rebuild paths
- Explicit registration:
  - `AngleManager.create_angle` (five parent links)
  - `ArcManager.create_circle_arc` (point/circle parents)
  - `SegmentManager._split_segments_with_point` (new child segments depend on original split segment + propagated segment ancestors)

## 4.2 Cleanup/cascade paths
- Full dependency cleanup (`remove_drawable`) is called in some delete flows:
  - `SegmentManager.delete_segment`
  - `AngleManager.delete_angle`
  - `ArcManager.delete_circle_arc`
  - `BarManager` / `StatisticsManager` selected paths
- Partial/manual cleanup exists in other flows:
  - direct `drawables.remove(...)` in several managers without corresponding `dependency_manager.remove_drawable(...)`
  - dependency traversal used for cascades (`PointManager`, `SegmentManager`, `TransformationsManager`)

## 4.3 Relationship-aware cascades
- Segment deletion:
  - deletes dependent angles
  - optionally cascades to child segments via dependency graph
  - can cascade to parent segments when requested
  - removes vectors/triangles/rectangles touching the segment
- Point deletion:
  - deletes dependent angles and circle arcs
  - deletes segments/vectors/circles/ellipses that contain the point
  - preserves specific parent segments in split-chain scenarios
- Transformations:
  - gathers dependency children and refreshes formulas/caches for dependent drawables

## 5. Known Edge Cases and Risk Analysis

## 5.1 Stale dependency edges after some deletions (high priority)
Several delete paths remove objects from `DrawablesContainer` but do not always call `dependency_manager.remove_drawable(...)`. Since `DrawablesContainer.remove` does not touch dependency state, stale `_parents/_children/_object_lookup` entries can persist.

Likely affected flows include delete operations for:
- points
- circles
- ellipses
- colored areas
- graphs
- some triangle/rectangle removals that are direct container removals

Potential effects:
- false positives in “solitary” checks (`get_parents/get_children`)
- unexpected cascade behavior on later operations
- retention of dead object references in dependency maps

## 5.2 Expression-generated closed regions are unbound to source drawables (medium)
`ColoredAreaManager._create_from_expression` creates `ClosedShapeColoredArea` from sampled points and does not register upstream drawable dependencies. This is intentional for sampled-region semantics but means auto-cascade deletion from source shape changes is not available unless managed separately.

## 5.3 Mixed manual + analyzer registration paths (medium)
Some managers use explicit `register_dependency`, others rely on analyzer inference. This is workable but raises consistency risk when drawables evolve (new attrs/type names) and analyzer branches are not updated.

## 5.4 Circular dependencies (handled but should remain tested)
Traversal and ordering are cycle-tolerant (visited set), but cycle semantics should continue to be tested for stability in edit/delete flows.

## 5.5 Point->Point suppression behavior (intentional policy)
Point-to-point edges are skipped by policy. This prevents graph pollution from coincident/alias point relations but should be validated against any future feature that might require explicit point linkage semantics.

## 6. Interaction Matrix (Creation/Deletion)

## 6.1 Objects that create dependency edges on creation
- Segment, Vector, Triangle, Rectangle, Circle, Ellipse, Angle, CircleArc
- Graph/DirectedGraph/UndirectedGraph/Tree
- Segments/Function/Functions/ClosedShape colored areas

## 6.2 Objects currently treated as edge-neutral
- Function, PiecewiseFunction, ParametricFunction
- Plot/ContinuousPlot/DiscretePlot/BarsPlot
- Bar
- Label
- Point

## 6.3 Objects that trigger multi-object cascades on deletion
- Point
- Segment
- Circle
- Ellipse
- CircleArc
- Graph families

## 7. Recommended Edge-Case Test Matrix

Priority test additions for dependency integrity:
1. Delete each dependency-managed drawable class and assert dependency graph cleanup (`_parents/_children/_object_lookup`) is consistent.
2. Validate solitary edit-policy checks after delete/recreate cycles (point/circle/ellipse).
3. Validate segment split chains + point deletion preserve intended ancestor segment behavior.
4. Validate graph deletion clears graph->edge/point dependency references without leaving stale nodes.
5. Validate colored area deletions clear all registered dependency edges (including closed-shape variants).
6. Validate expression-based region areas intentionally remain decoupled from source drawables.

## 8. Current Hardening Applied on This Branch
Implemented across commits `e03cb6d`, `c601ec3`, `ee1202d`, `614edc4`:
- guarded `register_dependency` against `None` and non-callable `get_class_name`
- hardened point-point guard callability checks
- added orphan pruning on `unregister_dependency`
- normalized interactive delete flows to call `dependency_manager.remove_drawable(...)` across point/circle/ellipse/graph/colored-area/function/parametric/piecewise/polygon/vector and direct triangle/rectangle removals
- added dependency graph invariant assertions and lifecycle coverage (delete/recreate for point/circle/ellipse)
- codified edge-neutral drawable behavior tests (`Label`, plot/bar families, `ParametricFunction`, `PiecewiseFunction`)

Status: dependency cleanup consistency and coverage hardening are now in place for the primary interactable surfaces.

## 9. Task List (Interactive Drawable Dependency Hardening)
These are implementation tasks I can execute next, prioritized to improve dependency correctness for user-visible/interactable drawables while minimizing break risk.

### Priority A (low-risk, high-impact integrity)
1. [Done] Normalize dependency cleanup in delete flows for interactive drawables.
   - Scope: ensure `dependency_manager.remove_drawable(...)` is called whenever an interactable drawable is removed from `DrawablesContainer`.
   - Targets: point, circle, ellipse, colored area, graph deletion paths, and direct triangle/rectangle removals.
   - Why: prevents stale graph edges that affect future interaction rules and cascades.

2. [Done] Add invariant checks around dependency graph state in tests.
   - Scope: add helper assertions for “no stale IDs in `_object_lookup` without edges” and “no edges to deleted drawables”.
   - Targets: delete/update tests for point/segment/circle/ellipse/graph/colored-area.
   - Why: catches regressions early when manager logic changes.

3. [Done - initial] Add focused lifecycle tests for interactable drawable deletion sequences.
   - Scope: create->link->delete->recreate cycles for point, segment, circle, ellipse, angle, circle arc, graph, closed-shape area.
   - Why: verifies dependencies are recreated cleanly and solitary-edit policies stay correct.

### Priority B (coverage of missing/implicit dependency behavior)
4. [Done] Explicitly document and test edge-neutral drawable classes.
   - Scope: `Label`, `Bar`, `Plot`, `BarsPlot`, `ContinuousPlot`, `DiscretePlot`, `ParametricFunction`, `PiecewiseFunction`.
   - Action: add tests that assert they intentionally register no drawable-parent edges.
   - Why: turns implicit behavior into a stable contract and prevents accidental coupling.

5. [Done] Decide and codify dependency behavior for parametric/piecewise interactions.
   - Scope: if these are intended to participate in area/tool interactions like `Function`, define parent linkage rules; otherwise explicitly lock them as edge-neutral.
   - Why: closes ambiguity before adding new tool workflows.

### Priority C (structural consistency and safe refactors)
6. [Partial] Centralize drawable removal through a dependency-aware helper.
   - Scope: add a manager-level helper that performs: remove from container + dependency cleanup + optional redraw.
   - Why: removes duplicated delete logic and reduces future stale-edge bugs.

7. [Pending] Add dependency lifecycle logging for debug mode.
   - Scope: lightweight logs for register/unregister/remove with class/name pairs.
   - Why: makes edge-case triage much faster without changing runtime semantics.

### Priority D (interactive behavior quality)
8. [Done - targeted] Verify cascade semantics for graph drawables from user perspective.
   - Scope: when deleting segments/points participating in graphs, ensure graph objects stay internally consistent and user-visible outcomes are deterministic.
   - Why: graph objects are high-cascade surfaces with mixed internal/external edges.

9. [Done - targeted] Validate colored-area dependency behavior by area type.
   - Scope: ensure shape-bound areas cascade with shape deletion; keep expression-sampled areas intentionally decoupled unless product wants live coupling.
   - Why: avoid unexpected area disappearance or stale areas in interactive workflows.

10. [Done] Add a dependency policy matrix to developer docs.
   - Scope: one table listing each interactable drawable and its parent/child rules + delete/update cascade expectations.
   - Why: gives a stable reference for future feature work and reviews.
