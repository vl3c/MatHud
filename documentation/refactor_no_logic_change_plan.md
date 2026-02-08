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
- [ ] `todo` Inventory oversized files/classes/methods (ranked by impact/risk)
- [ ] `todo` Establish size thresholds and exception rules
- [ ] `todo` Capture baseline behavior tests/golden outputs for top hotspots
- [ ] `todo` Add/confirm refactor review checklist (no logic change gate)

### Phase 1 - Core Orchestrators
- [ ] `todo` Complete remaining Canvas decomposition work
- [ ] `todo` Complete remaining WorkspaceManager decomposition work
- [ ] `todo` Add helper-level unit tests for extracted pure functions
- [ ] `todo` Add orchestration-phase tests for dependency order and error handling

### Phase 2 - Managers and Services
- [ ] `todo` Refactor oversized manager classes into composable steps
- [ ] `todo` Extract side-effect boundaries and isolate pure decision logic
- [ ] `todo` Add observability hooks (phase start/end/failure, counts, timings)
- [ ] `todo` Expand unit tests for newly extracted helpers

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

### Entries
- 2026-02-08
  - Scope: PR #16 helper extraction coverage expansion (Canvas/WorkspaceManager tests)
  - Status: done
  - Commits: `b2174b0`
  - Tests Run: `venv\Scripts\python -m cli.main test client` (2313 run, 0 failures)
  - Parity Evidence: no behavior change observed; test suite remained green
  - Notes/Blockers: none
