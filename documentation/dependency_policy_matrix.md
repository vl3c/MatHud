# Drawable Dependency Policy Matrix

Date: 2026-02-08
Branch: `feat/drawable-dependency-audit`

## Interactable Drawable Rules

| Drawable | Parent Dependencies | Delete Behavior | Notes |
|---|---|---|---|
| Point | none | Deleting point cascades to dependent segment/vector/circle/ellipse/angle/arc paths via managers and dependency graph | Point->Point edges are intentionally suppressed |
| Segment | `point1`, `point2` | Deleting segment removes segment and dependency node, cascades to dependents (angle, vector, polygon shapes, split-chain children) | Split chains preserve selected ancestor semantics |
| Vector | `segment` | Deleting vector removes drawable and dependency node | Vector lifespan is tied to segment |
| Triangle | `segment1`, `segment2`, `segment3` | Removed during segment/point cascades; dependency node removed on deletion paths | Treated as interactable polygon |
| Rectangle | `segment1`, `segment2`, `segment3`, `segment4` | Removed during segment/point cascades; dependency node removed on deletion paths | Treated as interactable polygon |
| Circle | `center` | Deleting circle removes drawable and dependency node, plus dependent region areas | Lifecycle/recreate invariants are covered |
| Ellipse | `center` | Deleting ellipse removes drawable and dependency node, plus dependent region areas | Lifecycle/recreate invariants are covered |
| Angle | `segment1`, `segment2`, `vertex_point`, `arm1_point`, `arm2_point` | Deleting angle removes drawable and dependency node | Explicit dependency registration |
| CircleArc | `point1`, `point2`, optional `circle` | Deleting arc removes drawable and dependency node; point/circle deletion cascades | Explicit dependency registration |
| Graph / DirectedGraph / UndirectedGraph / Tree | internal `_segments`, `_vectors`, `_isolated_points` | Deleting graph removes drawable and dependency node; parent removals notify graph internals | Graph child references are pruned by manager callbacks |
| FunctionsBoundedColoredArea | `func1`, `func2` | Deleting area removes drawable and dependency node | Function deletion also clears associated areas |
| FunctionSegmentBoundedColoredArea | `func`, `segment` | Deleting area removes drawable and dependency node | Cascades through function/segment deletion |
| SegmentsBoundedColoredArea | `segment1`, `segment2` | Deleting area removes drawable and dependency node | Cascades through segment deletion |
| ClosedShapeColoredArea | boundary `segments`, optional `circle`/`ellipse`/`chord_segment` | Deleting area removes drawable and dependency node; shape deletion cascades | Expression-sampled regions are intentionally decoupled |

## Edge-Neutral Contract

These drawables intentionally register no parent dependency edges in `DrawableDependencyManager.analyze_drawable_for_dependencies`:

- `Label`
- `Bar`
- `Plot`
- `BarsPlot`
- `ContinuousPlot`
- `DiscretePlot`
- `ParametricFunction`
- `PiecewiseFunction`

Rationale: these are either pure renderables/composites or expression-only drawables without user-visible parent handles in the current interaction model.

## Invariants

Dependency graph should satisfy all of the following:

1. Every ID in `_parents` and `_children` exists in `_object_lookup`.
2. Parent/child maps are reciprocal (`child -> parent` implies `parent -> child`).
3. `_object_lookup` only contains IDs that participate in at least one edge.
4. Deleting a drawable removes its ID from `_object_lookup`, `_parents`, `_children`, and all edge sets.

These invariants are now exercised by client tests (`test_drawable_dependency_manager.py`, `test_canvas.py`).
