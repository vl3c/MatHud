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
10. View notes (section 3.5): after a tool batch, one `View note:` line when something the model just drew is too small, too flat or outside the view, so it offers the user a zoom; the app never moves the view on its own.

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
17. `static/canvas_view_note.py` (view notes) and `static/client/prompt_canvas_state.py` (the canvas size and curve extents in the prompt's state)
18. `server_tests/test_canvas_view_note.py` and `static/client/client_tests/test_prompt_canvas_state.py`

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

### 3.5 View notes

The app never moves or zooms the view on its own, and the user's view, and anything the user drew or loaded, is the user's business. The feature does one thing: tell the model when something it just drew can't be seen. After each of the model's tool batches, the `[canvas changes]` end with one `View note:` line when an object that batch created or geometrically changed is too small, too flat or outside the view. The system prompt (`VIEW_NOTE_GUIDANCE` in `static/openai_api_base.py`) tells the model to mention it in one short sentence at the end of its reply and offer the suggested zoom, and never to change the view because of a note unless the user asks for a view change or agrees. The motivating case: at the default view (about +-400 units) a triangle at (0,0), (6,0), (2,4) is a speck of a few pixels at the origin, and the model said nothing.

```
[canvas changes]
+ A = (0, 0)
...
+ ABC = Triangle(A, B, C)  scalene; sides AB=6 BC=5.65685 CA=4.47214; area 12
View note: the new ABC spans only ~6x4 px on screen (x 0..6, y 0..4; view x -628..628, y -481.5..481.5). Offer to zoom to about x -0.8..6.8, y -0.9..4.9 (zoom center_x=3, center_y=2, range_val=3.8, range_axis=x); don't change the view unless the user agrees.
```

A missed note is far better than a false or useless one: every rule stays silent when it is unsure.

1. What is new: the objects of the batch's canvas that the canvas before it (the one the model last saw) lacks or has with a different geometry, by bucket, name and args. Style args (colour, label text or visibility, font) are no change, and an object's geometry includes the positions of the points it is drawn through, so moving a point changes its segments and polygons. A batch of only `undo` or `redo` calls gets no note: it brings back what was there.
2. Rules (`static/canvas_view_note.py`), one note per batch:
   1. Outside: a new object lies entirely outside the view (`the new B(100) is outside the view`); a point that a segment on screen is drawn through is on screen. The suggestion shows the new objects: the outside ones and any the batch drew on screen with them (fitted data points partly above the view zoom out to all of them).
   2. Tiny: the batch's new shapes with a real extent (not points or labels) span fewer than 16 px on screen (2% of the view's smaller side without a canvas size). The rule stays silent when a small new shape touches or shares a point with an earlier shape that is itself tiny or point-like at this view, or is a ring centred on an earlier point: the batch builds on a scene the user already sees at this scale, or declined to zoom on (a circumcircle after a declined tiny triangle, a diagonal after the user zoomed out, a ring around an existing point). A large earlier shape around the new one (a big circle, a long diagonal's box, a huge polygon around the view), an earlier label, or a shared vertex (the user's point O, then a speck triangle through O) does not make a speck visible, so those still get the note. Small shapes attached to a readable shape drawn in the same batch are left out too: centred within their own size of one of its corners, or lying on its outline (angle arcs, right-angle squares, highlight circles, tick marks, a tangent on a graph). The edges of a new polygon are named by the polygon. When the canvas has a graph the client did not measure (only new curves are), the rule stays silent.
   3. Flat: a new function graph that waves (its samples change direction at least three times) varies by fewer than 16 px vertically, has no spike or listed vertical asymptote, and is squeezed on screen: the samples miss its shape (it is not resolved) and its period is under 16 px, like `sin(x)` at the default view. A wave already drawn as a wave (`0.1*sin(x)` at +-10, `sin(x)` at +-100) gets no note, because a zoom keeps the aspect ratio and would only show a straight piece of it; nor does a wave whose suggested zoom would show less than one period (`0.001*sin(x)`), or still squeeze each period under 16 px (`sin(100x)` at the default view). Lines, constants, a single bump, `1/x` and `tan` are never flat. The suggested view makes the variation fill a quarter of the view's height.
3. Measurement. With the results of a tool batch the client adds `canvas_size_px` (the canvas size in CSS pixels) and `curve_extents` to the prompt's copy of the state (`static/client/prompt_canvas_state.py`; never to user messages, saved workspaces or traces). The view is a uniform linear map (`CoordinateMapper.math_to_screen`), so screen sizes follow exactly from the view bounds and the canvas size. `curve_extents` holds, for each graph or curve the batch created or redefined (found by comparing the batch's states, so the same curves are measured whatever else is on the canvas), the box of 64 samples, plus whether it is `clipped` (no bounds: sampled over the visible x range), `waves`, `spiky` (a few samples far beyond the rest, or a blow-up halfway between the samples next to the biggest values) and, for a wave, `resolved` (values between samples, at half and at 0.31 of a step, match the straight line between them) with, when it is not, an estimated `period` (the step is halved until two consecutive short runs of samples resolve it and agree; a resolved run with fewer than two turns ends the search, so a square or sawtooth wave, which jumps rather than turns, costs a few runs, not 24). Asymptote and discontinuity lists are left out when telling a redefined curve from an unchanged one. At most 20 curves are measured, in canvas order; if measuring runs past 100 ms, no extents are sent at all, so a slow canvas means no note rather than a different one. A failure never stops the prompt. The server removes both keys before the canvas reaches the model, in every format.
4. Extents below the grid's finest spacing (`min_tick_spacing`, 1e-6) or 1e-9 of the coordinates' magnitude count as a point; coordinates from 1e15 on are written in short scientific notation; view bounds are rounded like the canvas view line.
5. Wording: the note names the new objects (whole shapes before their edges and points, at most three), with the verb agreeing with the names as printed (`the new ABC spans`, `the new P, Q are`), and "new or changed" when a named object existed before the batch.
6. Placement: the last line of `[canvas changes]` in text and min_json (a header line, or a `"view_note"` key in min_json, when the full canvas is sent instead), and the last line of the batch's last tool message in json. User messages never carry a note.
7. History: when the user sends a new message, view notes older than the previous turn are removed from the tool messages (the `View note:` lines, and the `"view_note"` key of a min_json `[canvas now]`); the previous turn keeps its notes, since the new message may accept the offer ("yes, zoom"). The guidance speaks of the `[canvas changes]` after the tool calls made since the user's latest message, so a reply without tool calls does not offer again. With the Responses API and `previous_response_id`, OpenAI keeps earlier turns, and their notes, on its side. There is no memory across turns: if the user declines and the model later draws another separate speck, the new speck gets its own note.
8. Scenario CV-08 (`scenarios/canvas.json`) draws the motivating triangle; replay checks that the view is unchanged and no view tool ran, and live mode also checks that the reply mentions zooming.

### 3.6 Measurements

Qwen tokens (tokenizer extracted from the local GGUF) for the user message about each captured scene, and for a `get_current_canvas_state` result (measured before view notes; a scene that gets one, such as the triangle + circle captured at the default view, adds about 110 estimated tokens):

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
4. View notes stay silent on purpose in unsure cases: shaded areas and bar charts are not measured, a tiny shape drawn in the same batch as a readable one counts as part of it, a small shape touching an earlier small shape builds on a scene the user already sees, a small shape on a canvas with a graph the batch did not draw is left alone, and a low wave the screen already draws as a wave gets no note.
5. Keep `get_current_canvas_state` filter semantics backward-compatible (empty filters == full state behavior).
