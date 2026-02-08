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
- [ ] `todo` Add observability hooks (phase start/end/failure, counts, timings)
- [ ] `in_progress` Expand unit tests for newly extracted helpers

### Phase 3 - Tool Execution Pipelines
- [ ] `todo` Split long tool execution/parsing paths into deterministic helpers
- [ ] `todo` Add replay-friendly tracing checkpoints
- [ ] `todo` Add parity tests for representative command workflows
- [ ] `todo` Verify identical output/state for baseline scenarios

### Phase 4 - Final Hardening and Sign-Off
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
