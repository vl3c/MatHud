# Refactor Plan: Oversized Classes/Functions (No Logic Change)

## Non-Negotiable Constraint
- Do not change runtime behavior or public API semantics.
- All work is structural refactoring only (extract/split/rename/reorganize), with 1:1 behavior preserved.

## Objective
- Refactor oversized classes/functions into smaller composable units.
- Improve observability, readability, and structural clarity.
- Maintain exact behavior parity.

## Definition of Done
- Oversized hotspots are reduced to agreed thresholds or have documented exceptions.
- Extracted helpers have direct unit tests.
- Existing behavior/parity tests pass unchanged.
- Full client/server suites pass.
- Observability checkpoints exist at orchestration boundaries.

## Tracking Conventions
- Status values: `todo`, `in_progress`, `blocked`, `done`.
- Each completed item must include:
  - commit SHA
  - test evidence
  - parity note ("no behavior change observed")

## Phase Plan

### Phase 0 - Baseline and Guardrails
- [x] `done` Inventory oversized files/classes/methods (ranked by impact/risk)
- [x] `done` Establish size thresholds and exception rules
- [x] `done` Capture baseline behavior tests/golden outputs for top hotspots
- [x] `done` Add/confirm refactor review checklist (no logic change gate)

### Phase 1 - Core Orchestrators
- [ ] `todo` Complete remaining Canvas decomposition work
- [ ] `in_progress` Complete remaining WorkspaceManager decomposition work
- [ ] `in_progress` Add helper-level unit tests for extracted pure functions
- [ ] `in_progress` Add orchestration-phase tests for dependency order and error handling

### Phase 2 - Managers and Services
- [ ] `in_progress` Refactor oversized manager classes into composable steps
- [ ] `in_progress` Extract side-effect boundaries and isolate pure decision logic
- [ ] `in_progress` Add observability hooks (phase start/end/failure, counts, timings)
- [ ] `in_progress` Expand unit tests for newly extracted helpers

### Phase 3 - Tool Execution Pipelines
- [ ] `todo` Split long tool execution/parsing paths into deterministic helpers
- [ ] `todo` Add replay-friendly tracing checkpoints
- [ ] `todo` Add parity tests for representative command workflows
- [ ] `todo` Verify identical output/state for baseline scenarios

### Phase 4 - Coverage Expansion for New Elements
- [ ] `todo` Inventory all newly extracted helpers/components introduced during refactor phases
- [ ] `in_progress` Add focused unit tests for each new helper/component boundary
- [ ] `todo` Add negative/error-path tests for new orchestration and observability paths
- [ ] `todo` Track per-module coverage deltas and close major gaps

### Phase 5 - Final Hardening and Sign-Off
- [ ] `todo` Run full regression matrix (client + server)
- [ ] `todo` Run targeted high-risk end-to-end scenarios
- [ ] `todo` Close/document any threshold exceptions
- [ ] `todo` Final sign-off: "No behavior changes introduced"

## Work Log

### Entry Template
- Date:
- Scope:
- Status:
- Commits:
- Tests Run:
- Parity Evidence:
- Notes/Blockers:

## Sessions

### Session Template
- Session ID:
- Date:
- Goal:
- Planned Scope:
- Actual Scope:
- Status: `todo` | `in_progress` | `blocked` | `done`
- Related Commits:
- Validation:
- Parity Result:
- Follow-ups:

### Session Entries
- Session ID: S-2026-02-08-01
  - Date: 2026-02-08
  - Goal: Expand unit-test coverage for PR #16 extracted helpers
  - Planned Scope: Canvas/WorkspaceManager helper tests + suite wiring
  - Actual Scope: Completed as planned
  - Status: `done`
  - Related Commits: `b2174b0`
  - Validation: `venv\Scripts\python -m cli.main test client` (2313 run, 0 failures)
  - Parity Result: no behavior change observed
  - Follow-ups: continue Phase 1 decomposition tracking

- Session ID: S-2026-02-08-02
  - Date: 2026-02-08
  - Goal: Complete Phase 0 baseline/guardrails for no-logic-change refactor plan
  - Planned Scope: hotspot inventory + thresholds + baseline evidence + checklist
  - Actual Scope: Completed as planned
  - Status: `done`
  - Related Commits: `pending`
  - Validation: `/home/user/code/MatHud/venv/bin/python -m pytest -q server_tests/test_canvas_state_tool_schema.py server_tests/test_workspace_management.py server_tests/test_tool_search_service.py server_tests/client_renderer/test_renderer_support_plan.py` (91 passed)
  - Parity Result: no behavior change observed
  - Follow-ups: begin Phase 1 decomposition (Canvas/WorkspaceManager)

- Session ID: S-2026-02-08-03
  - Date: 2026-02-08
  - Goal: Start Phase 1 with WorkspaceManager orchestration decomposition
  - Planned Scope: extract save-workspace request pipeline into composable helpers + add orchestration/error tests
  - Actual Scope: extracted `save_workspace` to helper pipeline (`_execute_sync_json_request`, payload/parse helpers); added orchestration-order and error-path tests
  - Status: `in_progress`
  - Related Commits: `pending`
  - Validation: `py_compile` on modified files; targeted orchestration parity script (`phase1_workspace_orchestration_checks: PASS`); `/home/user/code/MatHud/venv/bin/python -m pytest -q server_tests/test_workspace_management.py server_tests/test_canvas_state_tool_schema.py` (22 passed)
  - Parity Result: no behavior change observed
  - Follow-ups: continue WorkspaceManager decomposition, then Canvas decomposition + broader client-suite validation

- Session ID: S-2026-02-08-04
  - Date: 2026-02-08
  - Goal: Continue Phase 1 decomposition and run full test sweep
  - Planned Scope: extract shared load/list/delete response parsing helpers; execute all server/client tests
  - Actual Scope: added shared workspace response parser path + success-message helpers; expanded orchestration/response test coverage; ran full server suite and client CLI test command
  - Status: `in_progress`
  - Related Commits: `pending`
  - Validation: `/home/user/code/MatHud/venv/bin/python -m pytest -q server_tests` (2 failed, 752 passed, 22 skipped; failures tied to occupied port 5000 environment); `/home/user/code/MatHud/venv/bin/python -m cli.main test client --start-server --port 5055 --timeout 120` (failed: server startup timeout in environment)
  - Parity Result: no behavior change observed in workspace/canvas-related regression subsets; full-suite failures are environment-related CLI server startup checks
  - Follow-ups: continue Phase 1 Canvas decomposition and re-run full suite in a clean port/runtime environment

- Session ID: S-2026-02-08-05
  - Date: 2026-02-08
  - Goal: Move to Phase 2 manager/service decomposition with no behavior change
  - Planned Scope: refactor `DrawableManager` region-capable lookup into composable decision helpers + add direct unit coverage
  - Actual Scope: extracted `_region_capable_lookup_chain` and `_first_region_capable_match`; added `TestDrawableManagerRegionLookup` coverage for priority ordering/fallback
  - Status: `in_progress`
  - Related Commits: `pending`
  - Validation: `py_compile` on touched files; targeted parity script (`phase2_drawable_manager_checks: PASS`); `/home/user/code/MatHud/venv/bin/python -m pytest -q server_tests/test_canvas_state_tool_schema.py server_tests/test_workspace_management.py server_tests/test_tool_search_service.py server_tests/client_renderer/test_renderer_support_plan.py` (91 passed)
  - Parity Result: no behavior change observed
  - Follow-ups: continue Phase 2 extractions in `DrawableManager` and adjacent managers; add observability hooks

- Session ID: S-2026-02-08-06
  - Date: 2026-02-08
  - Goal: Continue Phase 2 manager decomposition with shared delegation helpers
  - Planned Scope: refactor repeated colored-area deletion wrappers into a composable delegation boundary + expand unit tests
  - Actual Scope: introduced `_delete_colored_areas_for_target` and routed six wrapper methods through it; added delegation coverage in `TestDrawableManagerColoredAreaDelegation`
  - Status: `in_progress`
  - Related Commits: `pending`
  - Validation: `py_compile` on touched files; targeted parity script (`phase2_colored_area_delegation_checks: PASS`); baseline subset command (91 passed)
  - Parity Result: no behavior change observed
  - Follow-ups: continue Phase 2 manager/service decomposition and add light observability hooks at orchestration boundaries

- Session ID: S-2026-02-08-07
  - Date: 2026-02-08
  - Goal: Continue Phase 2 with observability hooks at manager orchestration boundaries
  - Planned Scope: add start/end/failure timing hooks to `StatisticsManager` public operations + add helper-level observability tests
  - Actual Scope: added optional logger-based hooks for `plot_distribution`, `plot_bars`, `fit_regression`, and `delete_plot`; expanded statistics manager tests for start/end/failure hook emission
  - Status: `in_progress`
  - Related Commits: `pending`
  - Validation: `py_compile` on touched files; `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test server -q` (756 passed, 22 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test client --start-server --port 5000 --timeout 240` (2315 run, 0 failures)
  - Parity Result: no behavior change observed
  - Follow-ups: continue Phase 2 observability rollout across remaining manager orchestration boundaries

- Session ID: S-2026-02-08-08
  - Date: 2026-02-08
  - Goal: Continue Phase 2 manager decomposition for dependency analysis paths
  - Planned Scope: split repetitive dependency collection/register logic in `DrawableDependencyManager` into composable helpers + add helper-level tests
  - Actual Scope: extracted `_append_and_register_dependency`, `_append_attr_dependency_if_present`, `_append_iterable_attr_dependencies`, and `_append_segment_attrs`; migrated `analyze_drawable_for_dependencies` branches to helper pipeline; added direct helper tests
  - Status: `in_progress`
  - Related Commits: `pending`
  - Validation: `py_compile` on touched files; `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test server -q` (756 passed, 22 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test client --start-server --port 5000 --timeout 240` (2317 run, 0 failures)
  - Parity Result: no behavior change observed
  - Follow-ups: continue Phase 2 decomposition for remaining manager/service hotspots

- Session ID: S-2026-02-08-09
  - Date: 2026-02-08
  - Goal: Start Phase 3 tool execution/parsing decomposition
  - Planned Scope: split chat-completions streaming/tool-call parsing path into deterministic helpers + add helper coverage
  - Actual Scope: extracted request-prep and streaming delta parsing helpers in `OpenAIChatCompletionsAPI`; decomposed tool-call accumulator path into composable helper methods; added tests for dict-shaped deltas, malformed entries, and request-prep branching
  - Status: `in_progress`
  - Related Commits: `pending`
  - Validation: `py_compile` on touched files; `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m pytest -q server_tests/test_openai_completions_api.py` (14 passed, 3 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test server -q` (759 passed, 22 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test client --start-server --port 5000 --timeout 240` (2317 run, 0 failures)
  - Parity Result: no behavior change observed
  - Follow-ups: continue Phase 3 decomposition in remaining provider/tool-call paths; begin Phase 4 coverage inventory once Phase 3 slices stabilize

- Session ID: S-2026-02-08-10
  - Date: 2026-02-08
  - Goal: Start Phase 4 coverage expansion for newly extracted helpers
  - Planned Scope: add focused unit tests for new helper boundaries in `OpenAIChatCompletionsAPI`
  - Actual Scope: added coverage for malformed chunk handling, content extraction guard behavior, and tool-call index normalization helper
  - Status: `in_progress`
  - Related Commits: `pending`
  - Validation: `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m pytest -q server_tests/test_openai_completions_api.py` (17 passed, 3 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test server -q` (762 passed, 22 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test client --start-server --port 5000 --timeout 240` (2317 run, 0 failures)
  - Parity Result: no behavior change observed
  - Follow-ups: continue Phase 4 across remaining newly extracted helper boundaries (statistics/dependency/openai stream helpers)

- Session ID: S-2026-02-08-11
  - Date: 2026-02-08
  - Goal: Continue Phase 3 decomposition and Phase 4 helper coverage in Responses API stream path
  - Planned Scope: extract shared tool-call accumulator merge helpers in `OpenAIResponsesAPI` and add direct unit coverage
  - Actual Scope: extracted `_get_or_create_tool_call_entry`, `_upsert_tool_call_entry`, and `_create_tool_call_entry_if_missing`; rewired function-call item/delta/completed extraction through helper boundaries; added tests for helper merging and non-overwrite behavior
  - Status: `in_progress`
  - Related Commits: `pending`
  - Validation: `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m pytest -q server_tests/test_openai_responses_api.py` (27 passed, 3 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test server -q` (764 passed, 22 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test client --start-server --port 5000 --timeout 240` (2317 run, 0 failures)
  - Parity Result: no behavior change observed
  - Follow-ups: continue Phase 3 decomposition for remaining provider/tool pipeline branches and expand Phase 4 helper/error-path coverage matrix

- Session ID: S-2026-02-08-12
  - Date: 2026-02-08
  - Goal: Continue Phase 4 coverage expansion for observability helper guard paths
  - Planned Scope: add direct unit coverage for `StatisticsManager` observability helper no-op/error-swallow behavior
  - Actual Scope: added tests for logger-absent and logger-failure guard paths in statistics manager observability helper
  - Status: `in_progress`
  - Related Commits: `pending`
  - Validation: `py_compile` on touched files; `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test server -q` (764 passed, 22 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test client --start-server --port 5000 --timeout 240` (2319 run, 0 failures)
  - Parity Result: no behavior change observed
  - Follow-ups: continue Phase 4 negative/error-path coverage across remaining extracted helper boundaries

### Entries
- 2026-02-08
  - Scope: PR #16 helper extraction coverage expansion (Canvas/WorkspaceManager tests)
  - Status: done
  - Commits: `b2174b0`
  - Tests Run: `venv\Scripts\python -m cli.main test client` (2313 run, 0 failures)
  - Parity Evidence: no behavior change observed; test suite remained green
  - Notes/Blockers: none

- 2026-02-08
  - Scope: Phase 0 completion artifacts (inventory, thresholds, baseline tests, no-logic-change checklist)
  - Status: done
  - Commits: `pending`
  - Tests Run: `/home/user/code/MatHud/venv/bin/python -m pytest -q server_tests/test_canvas_state_tool_schema.py server_tests/test_workspace_management.py server_tests/test_tool_search_service.py server_tests/client_renderer/test_renderer_support_plan.py` (91 passed)
  - Parity Evidence: no behavior change observed; baseline snapshot stored at `documentation/baselines/phase0_golden_outputs.json`
  - Notes/Blockers: local venv initially missing `pytest`, `python-dotenv`, and `openai`; installed for baseline execution

- 2026-02-08
  - Scope: Phase 1 kickoff - WorkspaceManager save pipeline decomposition + orchestration/error test coverage
  - Status: in_progress
  - Commits: `pending`
  - Tests Run: `py_compile` for touched files; targeted orchestration parity script (`PASS`); `/home/user/code/MatHud/venv/bin/python -m pytest -q server_tests/test_workspace_management.py server_tests/test_canvas_state_tool_schema.py` (22 passed)
  - Parity Evidence: no behavior change observed in validated baseline subset
  - Notes/Blockers: direct Brython client test collection requires `browser` runtime; used targeted parity script for new orchestration assertions in this environment

- 2026-02-08
  - Scope: Phase 1 continuation - shared load/list/delete response decomposition + full suite execution
  - Status: in_progress
  - Commits: `pending`
  - Tests Run: `/home/user/code/MatHud/venv/bin/python -m pytest -q server_tests` (2 failed, 752 passed, 22 skipped); `/home/user/code/MatHud/venv/bin/python -m cli.main test client --start-server --port 5055 --timeout 120` (server startup timeout)
  - Parity Evidence: targeted workspace regression subset remained green; remaining failures map to environment port/startup constraints rather than refactor logic
  - Notes/Blockers: port `127.0.0.1:5000` occupied in runtime, impacting CLI server-start test expectations

- 2026-02-08
  - Scope: Phase 2 kickoff - `DrawableManager` region-capable lookup decomposition and helper tests
  - Status: in_progress
  - Commits: `pending`
  - Tests Run: `py_compile` for touched files; targeted parity script (`phase2_drawable_manager_checks: PASS`); baseline subset command (91 passed)
  - Parity Evidence: no behavior change observed in lookup ordering/fallback behavior
  - Notes/Blockers: direct Brython package-style client test invocation still requires `browser` runtime in this environment

- 2026-02-08
  - Scope: Phase 2 continuation - colored-area deletion delegation helper extraction + test coverage expansion
  - Status: in_progress
  - Commits: `pending`
  - Tests Run: `py_compile` for touched files; targeted parity script (`phase2_colored_area_delegation_checks: PASS`); baseline subset command (91 passed)
  - Parity Evidence: no behavior change observed in colored-area deletion routing/arguments
  - Notes/Blockers: same environment limitation for direct Brython package test harness

- 2026-02-08
  - Scope: Phase 2 continuation - statistics manager observability hook extraction (start/end/failure + timing) with helper tests
  - Status: in_progress
  - Commits: `pending`
  - Tests Run: `py_compile` for touched files; `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test server -q` (756 passed, 22 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test client --start-server --port 5000 --timeout 240` (2315 run, 0 failures)
  - Parity Evidence: no behavior change observed; observability hooks are optional and logger-dependent
  - Notes/Blockers: direct pytest invocation for Brython client package tests requires browser runtime import path

- 2026-02-08
  - Scope: Phase 2 continuation - drawable dependency analyzer decomposition into composable helper boundaries + helper tests
  - Status: in_progress
  - Commits: `pending`
  - Tests Run: `py_compile` for touched files; `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test server -q` (756 passed, 22 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test client --start-server --port 5000 --timeout 240` (2317 run, 0 failures)
  - Parity Evidence: no behavior change observed in dependency registration, traversal, and graph-child cleanup workflows
  - Notes/Blockers: none

- 2026-02-08
  - Scope: Phase 3 kickoff - chat-completions stream/tool-call parsing decomposition + helper-level tests; add explicit coverage expansion phase
  - Status: in_progress
  - Commits: `pending`
  - Tests Run: `py_compile` for touched files; `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m pytest -q server_tests/test_openai_completions_api.py` (14 passed, 3 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test server -q` (759 passed, 22 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test client --start-server --port 5000 --timeout 240` (2317 run, 0 failures)
  - Parity Evidence: no behavior change observed in streamed token emission, final events, or tool-call serialization
  - Notes/Blockers: none

- 2026-02-08
  - Scope: Phase 4 kickoff - focused unit coverage for newly extracted chat-completions helper boundaries
  - Status: in_progress
  - Commits: `pending`
  - Tests Run: `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m pytest -q server_tests/test_openai_completions_api.py` (17 passed, 3 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test server -q` (762 passed, 22 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test client --start-server --port 5000 --timeout 240` (2317 run, 0 failures)
  - Parity Evidence: no behavior change observed; tests only exercise helper boundaries and guard branches
  - Notes/Blockers: none

- 2026-02-08
  - Scope: Phase 3/4 continuation - responses stream tool-call accumulator decomposition + helper-level coverage
  - Status: in_progress
  - Commits: `pending`
  - Tests Run: `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m pytest -q server_tests/test_openai_responses_api.py` (27 passed, 3 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test server -q` (764 passed, 22 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test client --start-server --port 5000 --timeout 240` (2317 run, 0 failures)
  - Parity Evidence: no behavior change observed in response streaming, tool-call JSON assembly, and final finish-reason handling
  - Notes/Blockers: none

- 2026-02-08
  - Scope: Phase 4 continuation - statistics observability helper guard/error-path coverage
  - Status: in_progress
  - Commits: `pending`
  - Tests Run: `py_compile` for touched files; `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test server -q` (764 passed, 22 skipped); `/home/user/code/MatHud/workspaces/refactor-composable-architecture-phase1/venv/bin/python -m cli.main test client --start-server --port 5000 --timeout 240` (2319 run, 0 failures)
  - Parity Evidence: no behavior change observed; tests only validate helper guard behavior
  - Notes/Blockers: none
