# Canvas Prompt Summary Rollout

This document captures the implementation and operational model for canvas-state prompt normalization, telemetry, and filtered state retrieval.

## 1. Scope Delivered

1. Server-side canvas-state summarizer (`static/canvas_state_summarizer.py`) with deterministic pruning and comparison metrics.
2. Prompt normalization pipeline in `static/openai_api_base.py` with mode controls:
   - `off`
   - `hybrid` (default)
   - `summary_only`
3. Hybrid small-scene fast path: skip summarization entirely when raw `canvas_state` bytes are already below threshold.
4. Telemetry logging (`canvas_prompt_telemetry`) with structured JSON payloads.
5. Filtered `get_current_canvas_state` tool contract (drawable-type and object-name filters).
6. Dev-only comparison endpoint and browser helper for side-by-side inspection.

## 2. Key Files

1. `static/canvas_state_summarizer.py`
2. `static/openai_api_base.py`
3. `static/functions_definitions.py`
4. `static/client/canvas.py`
5. `static/client/function_registry.py`
6. `static/client/ai_interface.py`
7. `static/routes.py`
8. `scripts/canvas_prompt_telemetry_report.py`
9. `server_tests/test_canvas_state_summarizer.py`
10. `server_tests/test_openai_api_base.py`
11. `server_tests/test_canvas_state_tool_schema.py`

## 3. Runtime Controls

```env
AI_CANVAS_SUMMARY_MODE=hybrid
AI_CANVAS_HYBRID_FULL_MAX_BYTES=6000
AI_CANVAS_SUMMARY_TELEMETRY=0
```

Mode behavior:
1. `off`: no prompt mutation.
2. `hybrid`: if `canvas_state` size is `<= AI_CANVAS_HYBRID_FULL_MAX_BYTES`, keep full state and return unchanged prompt; otherwise attach `canvas_state_summary` and remove full state.
3. `summary_only`: always use `canvas_state_summary` and remove full state.

## 4. Telemetry Payload

Logged from `_log_canvas_summary_telemetry` as:

`canvas_prompt_telemetry { ...json... }`

Core fields:
1. `mode`
2. `prompt_kind` (`text` or `multimodal`)
3. `normalize_elapsed_ms`
4. `input_bytes`, `normalized_prompt_bytes`, `output_payload_bytes`
5. `input_estimated_tokens`, `normalized_prompt_estimated_tokens`, `output_payload_estimated_tokens`
6. `reduction_pct`
7. `includes_full_state`
8. `summary_metrics` (when summary envelope is present)

Notes:
1. `normalized_prompt_bytes` is the post-normalization text payload before image injection.
2. `output_payload_bytes` reflects actual payload sent to provider (`str` or multimodal list serialization).

## 5. Dev Inspection Workflow

1. Start server in development mode.
2. In browser console, run:

```javascript
window.compareCanvasState()
```

3. Inspect:
   - full state object
   - summary state object
   - full/summary bytes + estimated tokens + reduction percentage

Debug endpoint used by helper:
`POST /api/debug/canvas-state-comparison`

## 6. Log Reporting Utility

Generate aggregate reports from session logs:

```bash
python scripts/canvas_prompt_telemetry_report.py
python scripts/canvas_prompt_telemetry_report.py --mode hybrid --csv-out /tmp/canvas.csv
python scripts/canvas_prompt_telemetry_report.py --json-out /tmp/canvas_summary.json
```

The script:
1. extracts `canvas_prompt_telemetry` rows,
2. groups by `mode`, `prompt_kind`, and `includes_full_state`,
3. emits averages for token sizes, reduction %, and normalize latency.

## 7. Benchmarks (Post Small-Scene Optimization)

Representative A/B measurements (off vs hybrid):
1. Small scene: 555 -> 555 tokens (0 delta; full state retained)
2. Medium scene: 3782 -> 2047 tokens (-1735; -45.87%)
3. Large scene: 14174 -> 7444 tokens (-6730; -47.48%)

Interpretation:
1. Small-scene fast path preserves clarity and avoids extra envelope overhead.
2. Medium/large scenes get substantial context reduction with summary mode.

## 8. Follow-up Guidance

1. Keep hybrid as default during rollout.
2. Use telemetry reports to tune `AI_CANVAS_HYBRID_FULL_MAX_BYTES` for your environment.
3. Keep `get_current_canvas_state` filter semantics backward-compatible (empty filters == full state behavior).
