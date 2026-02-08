# Phase 0 Baseline and Guardrails (Refactor, No Logic Change)

Date: 2026-02-08
Branch: `refactor/composable-architecture-phase1`

## 1) Oversized Hotspot Inventory (Ranked by impact/risk)

Threshold signal used for triage:
- Large file: `>= 1000 LOC`
- Large class: `>= 800 LOC`
- Large function: `>= 100 LOC`

Ranked production hotspots (tests excluded from ranking):

| Rank | File | LOC | Primary concern | Impact | Risk |
|---|---|---:|---|---|---|
| 1 | `static/client/canvas.py` | 1819 | Monolithic orchestration surface (drawing, zoom, object ops, state) | Very High | Very High |
| 2 | `static/client/workspace_manager.py` | 1383 | Stateful restore/load flow with many object dependency paths | Very High | High |
| 3 | `static/client/ai_interface.py` | 2141 | AI orchestration/UI coupling and response processing breadth | High | High |
| 4 | `static/client/managers/drawable_manager.py` | 1153 | Central object lifecycle and manager delegation | High | High |
| 5 | `static/client/slash_command_handler.py` | 835 | Command routing and user-facing behavior branching | High | Medium |
| 6 | `static/client/managers/statistics_manager.py` | 934 | Large manager with heavy procedural methods | Medium | Medium |
| 7 | `static/client/managers/colored_area_manager.py` | 891 | Geometry/path edge-case handling in long methods | Medium | Medium |
| 8 | `static/client/rendering/svg_renderer.py` | 1014 | Renderer orchestration and cache/planning interaction | Medium | Medium |
| 9 | `static/client/rendering/svg_primitive_adapter.py` | 893 | Primitive application and DOM mutation pathways | Medium | Medium |
| 10 | `static/client/rendering/canvas2d_renderer.py` | 861 | Plan reuse + compositing orchestration | Medium | Medium |

Large pure-utility files to defer behind orchestrators:
- `static/client/utils/math_utils.py` (`2766 LOC`, low-moderate behavior risk, high split opportunity)
- `static/client/utils/graph_utils.py` (`1506 LOC`, medium risk due graph semantics)
- `static/client/utils/geometry_utils.py` (`1219 LOC`, medium risk due numerical edge cases)

## 2) Size Thresholds and Exception Rules

Target thresholds for this refactor stream:
- File target: prefer `<= 900 LOC` for orchestrators/managers.
- Class target: prefer `<= 600 LOC`.
- Function target: prefer `<= 80 LOC` in orchestration code and `<= 120 LOC` for numerically dense utilities.

Exception rules (must be explicitly documented in PR/work log):
- Numerically dense pure utility methods may exceed function threshold if split would reduce clarity.
- Backward-compatibility adapter/shim files may exceed file threshold temporarily.
- Generated or mirrored compatibility files are out of scope.
- Test aggregators and test fixture hubs are not phase-0 split targets.

## 3) Baseline Behavior Tests and Golden Outputs

Baseline test command:

```bash
/home/user/code/MatHud/venv/bin/python -m pytest -q \
  server_tests/test_canvas_state_tool_schema.py \
  server_tests/test_workspace_management.py \
  server_tests/test_tool_search_service.py \
  server_tests/client_renderer/test_renderer_support_plan.py
```

Baseline result:
- `91 passed in 0.29s`

Stored artifacts:
- Test log: `documentation/baselines/phase0_baseline_pytest.txt`
- Golden snapshot: `documentation/baselines/phase0_golden_outputs.json`
- Golden snapshot SHA256: `bc689f0a4500b7775e3d1a490a15f3f30fdcbb966ea86286fd434645b9b447ea`

Golden snapshot captures deterministic anchors used for parity checks:
- tool count and strict schema shape for `get_current_canvas_state`
- workspace schema version constant
- canonical rectangle output for a fixed diagonal input
- polygon type tokens used by workspace/polygon flows

## 4) Refactor Review Checklist (No Logic Change Gate)

Every refactor PR touching phase targets must pass this checklist:

- [ ] No public API contract change (function names, required args, return shapes).
- [ ] No behavior change in targeted baseline tests.
- [ ] Existing tests updated only when structurally necessary (no expectation drift without explicit approval).
- [ ] Orchestration extracted into composable helpers with unchanged side-effect order.
- [ ] Error messages/status contracts preserved for externally consumed paths.
- [ ] State schema keys and token values unchanged.
- [ ] Added helper-level tests for each extracted pure helper.
- [ ] Added parity note in PR summary: "no behavior change observed".
- [ ] Attached test evidence command + result counts.
- [ ] Any threshold exceptions documented with rationale and file/function references.
