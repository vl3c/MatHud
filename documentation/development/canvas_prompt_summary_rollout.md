# Canvas Prompt Summary Rollout

This document captures the implementation and operational model for how canvas state reaches the model: the canvas formats, per-batch change reports, prompt normalization for the legacy JSON format, telemetry, and filtered state retrieval.

## 1. Scope Delivered

1. Canvas state formatter (`static/canvas_state_formatter.py`): renders `get_canvas_state()` output as text (one object per line with computed facts), compact JSON, or a per-object delta; applies a token budget.
2. Canvas formats selected by `MATHUD_CANVAS_FORMAT` (`text` default, `min_json`, `json`), used by every provider including LocalAgent.
3. `[canvas changes]` appended to the last tool result of each tool batch, so the model is not left with a stale canvas during multi-step turns.
4. `get_current_canvas_state` results rendered on the server in the configured format.
5. Server-side canvas-state summarizer (`static/canvas_state_summarizer.py`) with deterministic pruning and comparison metrics, used by the `json` format:
   - `off`
   - `hybrid` (default)
   - `summary_only`
6. Hybrid small-scene fast path: skip summarization entirely when raw `canvas_state` bytes are already below threshold.
7. Telemetry logging (`canvas_prompt_telemetry`) with structured JSON payloads.
8. Filtered `get_current_canvas_state` tool contract (drawable-type and object-name filters).
9. Dev-only comparison endpoint and browser helper for side-by-side inspection.

## 2. Key Files

1. `static/canvas_state_formatter.py`
2. `static/token_estimation.py`
3. `static/canvas_state_summarizer.py`
4. `static/openai_api_base.py`
5. `static/providers/local/__init__.py`
6. `static/functions_definitions.py`
7. `static/client/canvas.py`
8. `static/client/function_registry.py`
9. `static/client/ai_interface.py`
10. `static/routes.py`
11. `scripts/canvas_prompt_telemetry_report.py`
12. `server_tests/test_canvas_state_formatter.py` (golden outputs for the captured scenes in `server_tests/fixtures/canvas_states/`)
13. `server_tests/test_canvas_state_prompts.py` (provider integration)
14. `server_tests/test_canvas_state_summarizer.py`
15. `server_tests/test_openai_api_base.py`
16. `server_tests/test_canvas_state_tool_schema.py`

## 3. Canvas Formats

```env
MATHUD_CANVAS_FORMAT=text          # text | min_json | json
MATHUD_CANVAS_BUDGET_TOKENS=4000   # 0 = unlimited; unset = provider default
```

The client sends the prompt JSON (`canvas_state`, `user_message`, `tool_call_results`, `use_vision`, `ai_model`) with every user message, and `state_after` as `canvas_state` after each tool batch. What the providers do with it:

| Format | User message | After a tool batch | `get_current_canvas_state` result |
|---|---|---|---|
| `text` (default) | `<canvas>` block with one object per line, then the user's text | `[canvas changes]` appended to the batch's last tool message | same text rendering |
| `min_json` | `<canvas>` block with compact JSON, then the user's text | same `[canvas changes]` lines | compact JSON |
| `json` | the whole prompt JSON (summarized per `AI_CANVAS_SUMMARY_MODE`); LocalAgent sends only the user text plus a `[Canvas: 3 Points, ...]` count line | nothing | raw state JSON |

Defaults: `text` for every provider. `min_json` stays a supported fallback alongside `text`: it lists the same objects (without the lengths, areas and angles `text` computes) and scored the same in the canvas-format benchmark (`scripts/benchmark_canvas_formats.py`), so it is the switch to try if a model reads JSON better than the text lines. It also parses with `json.loads`, but it is lossy (rounded numbers, defaults and render-only fields dropped, lists trimmed over the budget); code that needs the exact state should read `get_canvas_state()` output, the same JSON workspaces save. LocalAgent previously never saw coordinates, names or formulas unless it called `get_current_canvas_state`; cloud providers received the prompt JSON with float noise, render-only fields and envelope keys. The `json` format sends the same canvas payload as before (the prompt JSON, summarized per `AI_CANVAS_SUMMARY_MODE`, and the old system-prompt sentence about canvas state). It is not a byte-for-byte replay of the old requests: the summarizer's size metrics (the hybrid `metrics` block) are no longer placed in the prompt (telemetry still logs them); in search tool mode (the default, `MATHUD_TOOL_EXPOSURE=search`) the system prompt also carries the search-first tool-loading paragraph; and each tool call's result goes into its own tool message instead of all results landing in the batch's last one.

### 3.1 Text format

```
view x [-6, 6] y [-3.043, 3.043]; grid 1
A = (0, 0)
AB = Segment(A, B)  len 4
ABC = Triangle(A, B, C)  scalene right; sides AB=4 BC=3 CA=5; area 6
angle_BAC = Angle(AB, AC)  vertex A, 36.8699 deg
A(5) = Circle(center A, r 5)  area 78.5398; passes through C
f(x) = x^2 - 1  on [-5, 5]
G1 = Graph(undirected weighted; vertices A B C D E F G H)
  edges (12): A-B 4, A-C 2, ...
sales = BarChart(Mon 12, Tue 19, Wed 7, Thu 15, Fri 22)  x_start -12
```

1. Order follows dependencies: points, segments/vectors, polygons, circles/ellipses/arcs, angles, functions/curves, shaded areas, graphs, plots, labels, other objects, computations.
2. Facts are computed from the coordinates: segment/vector lengths, polygon sides, area and type, circle/ellipse area and the named points a circle passes through, angle size at the shared vertex, arc sweep and length, graph edges with weights (graph segments are shown as edges, not as separate segments). Facts are never computed from duplicated point names.
3. Numbers keep at most 6 significant digits; float noise (`199.20000000000002`, `1.2e-14`) is snapped. Long decimals inside expressions (regression fits) are shortened.
4. Only non-default styles are shown (e.g. `color purple`, `opacity 0.5`, `hidden`).
5. Unknown drawable buckets and unknown args fall back to compact JSON, so new drawable fields are never silently dropped.
6. Duplicate names get one warning line (`! duplicate names, tools cannot tell these apart: F x25 (points)`).

### 3.2 Budget

`MATHUD_CANVAS_BUDGET_TOKENS` caps the user-message canvas block (default 4000 estimated tokens for cloud providers, 1500 for local ones, `0` = unlimited). When a scene is larger: points are packed several per line, then groups are shrunk tier by tier (labels/computations first; then points/segments/vectors/angles together; then areas/plots/circles; functions, graphs and polygons last), each with a line such as `... 112 more points omitted; call get_current_canvas_state with object_names to see them`. `min_json` keeps the same fraction of every object list and adds `"omitted"` counts plus a `"note"`. `get_current_canvas_state` results get twice the canvas budget and are trimmed the same way beyond it. Tool-batch deltas (`[canvas changes]`) are text lines in both formats. Token counts use `estimate_tokens_from_text`, which counts one token per digit (the default local model's tokenizer does) and is within -5%..+11% of the real Qwen tokenizer on the captured scenes; CJK characters count one token each.

### 3.3 Changes after tool calls

Each provider remembers the last canvas state it showed the model (cleared by `reset_conversation`). When the client returns a tool batch, the results are first written into their tool messages (matched by tool-call id), then the difference between the remembered state and `state_after` is appended to the batch's last tool message:

```
[canvas changes]
+ A(5) = Circle(center A, r 5)  area 78.5398; passes through C
~ I = (-1, 0.5)  ->  (-1, -0.5)
- EF (segment) removed
```

Objects are compared by their rendered line, so moving a point also reports the lengths, areas and angles that changed with it. Nothing is appended when the canvas did not change. When the delta would be larger than the full rendering (e.g. after `clear_canvas`), or no state was shown yet, `[canvas now]` and the full canvas are appended instead. For Anthropic the note stays inside the `tool_result` block.

### 3.4 History

The `<canvas>` block stays on the latest user message (tool results describe changes against it) and is stripped by marker from older user messages. With the Responses API and `previous_response_id`, OpenAI keeps earlier turns (and their canvas blocks) server-side.

### 3.5 Measurements

Qwen tokens (tokenizer extracted from the local GGUF) for the user message about each captured scene, and for a `get_current_canvas_state` result:

| Scene | Cloud `json` (hybrid) | Cloud `text` | LocalAgent `json` | LocalAgent `text` | Tool result `json` | Tool result `text` |
|---|---|---|---|---|---|---|
| triangle + circle | 464 | 184 | 31 | 184 | 455 | 167 |
| mixed medium | 1169 | 399 | 55 | 399 | 1160 | 382 |
| weighted graph | 2189 | 269 | 29 | 269 | 2180 | 252 |
| regression, 30 points (duplicate names) | 1370 | 625 | 27 | 625 | 1361 | 608 |
| synthetic large (130 points, 65 segments) | 11151 | 3942 (budget) | 58 | 1458 (budget) | 16740 | 5102 |

LocalAgent `json` is small only because it carried object counts, not the scene. The `[canvas changes]` note for the mixed-medium edit above is 63 tokens.

## 4. Legacy JSON Format Controls

```env
AI_CANVAS_SUMMARY_MODE=hybrid
AI_CANVAS_HYBRID_FULL_MAX_BYTES=6000
AI_CANVAS_SUMMARY_TELEMETRY=0
```

Mode behavior (only with `MATHUD_CANVAS_FORMAT=json`):
1. `off`: no prompt mutation.
2. `hybrid`: if `canvas_state` size is `<= AI_CANVAS_HYBRID_FULL_MAX_BYTES`, keep full state and return unchanged prompt; otherwise attach `canvas_state_summary` and remove full state.
3. `summary_only`: always use `canvas_state_summary` and remove full state.

## 5. Telemetry Payload

Logged from `_log_canvas_summary_telemetry` as:

`canvas_prompt_telemetry { ...json... }`

Core fields:
1. `canvas_format`
2. `mode` (summary mode; meaningful for the `json` format)
3. `prompt_kind` (`text` or `multimodal`)
4. `normalize_elapsed_ms`
5. `input_bytes`, `normalized_prompt_bytes`, `output_payload_bytes`
6. `input_estimated_tokens`, `normalized_prompt_estimated_tokens`, `output_payload_estimated_tokens`
7. `reduction_pct`
8. `includes_full_state`
9. `summary_metrics` (when the summarizer ran; never sent to the model)

Notes:
1. `normalized_prompt_bytes` is the post-normalization text payload before image injection (for `text`/`min_json`, the canvas block plus user text).
2. `output_payload_bytes` reflects actual payload sent to provider (`str` or multimodal list serialization).

## 6. Dev Inspection Workflow

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

## 7. Log Reporting Utility

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

## 8. Benchmarks (JSON format, post small-scene optimization)

Representative A/B measurements (off vs hybrid):
1. Small scene: 555 -> 555 tokens (0 delta; full state retained)
2. Medium scene: 3782 -> 2047 tokens (-1735; -45.87%)
3. Large scene: 14174 -> 7444 tokens (-6730; -47.48%)

Interpretation:
1. Small-scene fast path preserves clarity and avoids extra envelope overhead.
2. Medium/large scenes get substantial context reduction with summary mode.

## 9. Follow-up Guidance

1. Keep `text` as the default; use `MATHUD_CANVAS_FORMAT=json` to compare against the original canvas payload (see section 3 for how it differs from the old requests).
2. Run a comprehension benchmark (questions about lengths, names, graph edges and changes after tool calls) per format and provider once models are reachable, and tune `MATHUD_CANVAS_BUDGET_TOKENS` for the local model's context size.
3. Client-side state gaps limit the text format: `Function.get_state` omits the curve color, `Point`/`Segment` states omit colors, and graph states omit isolated points.
4. Keep `get_current_canvas_state` filter semantics backward-compatible (empty filters == full state behavior).
