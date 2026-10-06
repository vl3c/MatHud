# Agentic Scenario Testing

Status: phases 1 to 7 of section 7 are built: the client hooks, the loader and check engine, the catalogue (`scenarios/*.json`), the replay runner, the reports, live mode with retrace and failure classification (phase 6), and the CI replay job (phase 7). Bug fixing (phase 8) goes on. Written 2026-09-26 against `main` at `e6996fa`; updated the same day to match the implementation (field names, hook options, the check language as built, K20 and K22 fixed, K25 and CV-05 added). Updated 2026-09-27 after merging `main` with vl3c/MatHud#72 and #73: K1, K2, K6, K7, K18 and K21 are fixed and their marks and waivers removed; K26, K27 and CV-06 added. Updated again on `claude/fix-view-undo-noops`: K4, K25 and K26 are fixed and their marks and waivers removed. Updated again on `claude/fix-names-persistence` (vl3c/MatHud#77): K3, K5, K10, K15 and K27 are fixed and their marks and waivers removed. Updated again on `claude/fix-batch-7`: K23 is fixed; K28 to K32 are added and fixed, with scenarios FN-10, FN-11, GEO-17, GEO-18, NM-07, ST-05 and WS-04. Updated again on `claude/scenario-live-mode` (phase 6): live mode, `--mode retrace` and failure classification; deviations from the design are listed under phase 6 in section 7.

This document proposes a testing framework that closes the agentic loop. A real model gets a natural-language request that needs several chained tool calls. The real app runs those calls in the browser, and the framework checks the resulting canvas. The same scenarios also run with no model at all: each one carries a reference tool-call sequence that the harness replays directly. The main goal is catching app bugs in how objects are created, related, named, updated, transformed, deleted, undone and persisted, which unit tests miss. The second goal is measuring how well models chain tool calls.

The catalogue in section 5 is the main deliverable. Every reference sequence in it was checked against `static/functions_definitions.py` with the server's own `ToolArgumentValidator`, and every one was executed through the app's real tool path in headless Chrome while this design was written (see 4.10). That prototype run found the bugs in section 6.

## How to run

```
python -m cli.main test scenarios --mode replay --start-server            # the whole catalogue, about 40 s
python -m cli.main test scenarios --mode replay --smoke --start-server    # the 11 smoke scenarios, about 25 s
python -m cli.main test scenarios --ids GEO-04,CV --tags undo --port 5000 # a running server; filters combine
python -m cli.main test scenarios --dry-run                               # validate every file and print the plan
python -m cli.main test scenarios --regrade logs/scenario_runs/<time>/results.json
python -m cli.main test scenarios --mode live --smoke                     # the local model on llama-server
python -m cli.main test scenarios --mode live --models ID --repeats 3 --ids GEO,CON
python -m cli.main test scenarios --mode live --provider openrouter --models deepseek/deepseek-v4.1-flash --smoke --dry-run
python -m cli.main test scenarios --mode retrace logs/scenario_runs/<time>/results.json
python -m cli.main test scenarios --mode live --attach-desktop 9333 --ids GEO-01   # in the open desktop window
```

1. `--start-server` starts `app.py` on `--port` (or the next free port) with `MATHUD_WORKSPACES_DIR` pointing at a temporary directory, and stops it afterwards. With `--port` alone the server must already run; scenarios that save or load workspaces are then skipped unless `--allow-workspace-writes` is given, because they would use that server's workspace directory.
2. `--ids` (alias `--only`) takes scenario ids or areas (`GEO-04`, `CV`), `--tags` takes tags; both repeat or take commas.
3. Output goes to `--out` (default `logs/scenario_runs/<time>/`): `results.jsonl` (started fresh by each run, one record per step as it finishes; every record carries its `attempt`, and a retried scenario's abandoned attempt is followed by an `attempt_discarded` record), then `results.json` (final attempts only) and `summary.md` (also on Ctrl+C), and `failures/<scenario>__<step>.json` and `.png` for every step with an unexpected failure (`--known-artifacts` adds them for expected failures). Artifact paths are stored relative to the output directory. `--json` prints the summary as JSON. `--regrade` honours `--ids`, `--tags` and `--smoke`.
4. Check statuses: `pass`, `fail`, `error` (a check that could not be evaluated; also when it is marked `known`, since a crash is not the bug's failure), `xfail` (a failing check marked `known`, or an invariant failure excused by a waiver), `xpass` (a `known` check that passed: "fixed? K<n>"), `warn` (I1's cross-bucket name clash), `skip` (answer checks in replay) and `unrecorded` (the check needs a function sample the run did not store, which can happen in `--regrade` after a check was edited; rerun the replay). Scenario statuses: `fail` (any `fail` or `error`), else `xpass`, else `xfail` (a check marked `known` failed), else `waived` (only waived invariants failed), else `pass`. A waiver that excused nothing in the run is listed as "waiver I5:K1 (global) unused" (or with the scenario id for a scenario waiver), an xpass-style hint that the bug may be fixed.
5. Exit codes: 0 with no unexpected failures, 1 with any `fail` or `error` (including a scenario the browser could not finish), 2 for invalid scenario files or a missing server, 130 when interrupted. Expected failures, waived failures, xpasses, unused waivers and unrecorded checks never fail the run.
6. Adding a scenario: add it to its area file, run it with `--ids`, and give every check that fails because of an app bug `"known": "K<n>"`, with the bug in section 6 and in `scenarios/known_bugs.json`. `server_tests/test_cli/test_scenario_catalogue.py` loads and validates every file in the server suite.
7. Latest replay (2026-09-27, `claude/fix-batch-7` on `main` at `cc57b41`; Windows 11, headless Chrome, `--start-server`): 81 scenarios, 81 pass, 0 xfail, 0 waived, 0 xpass, 0 unexpected failures, no unused waivers; checks 1,643 pass, 0 xfail, 5 warn, 10 skip; about 30 s including Chrome start-up. No scenario check is marked known: every bug with a scenario is fixed and its checks are regression guards. Each scenario added on this branch was also replayed against the code before its fix and failed there.
8. Live mode (`--mode live`) always starts its own server, with the settings of 4.6 pinned and recorded in the run config (`server_env`): `--tool-exposure` (default `search`), `--canvas-format` (`text`), `--canvas-budget` (the provider default), `--tool-search-mode` (`hybrid` for local, `local` for OpenRouter), `--local-reasoning-effort` (`medium`), `MATHUD_WORKSPACES_DIR` (a temporary directory), `REQUIRE_AUTH=false`, `MATHUD_OPENROUTER_MAX_RETRIES=0`, and every provider key the run does not use set to an empty string, which python-dotenv never overrides. Before the first message the harness reads `/api/available_models` and aborts (exit 2) unless every model is registered under the provider (`local_agent`; `openrouter_paid` or `openrouter_free`). Without `--models` a local run uses every model llama-server serves; OpenRouter needs `--models`.
   - Each scenario starts with a reset session and a `POST /new_conversation`; setup and `do` steps run as replay. A turn sends the prompt with `sendMatHudMessage`, polls `getMatHudTurnStatus` every 0.2 s and stops the turn with `stopMatHudTurn` after `timeout_s` (default `--turn-timeout` 300). The turn's request cap `max_requests` (default `--turn-max-requests` 8) goes to the client with the message (`sendMatHudMessage` option `max_requests`), which ends the turn with outcome `max_requests` instead of sending a request beyond it, so a turn sends at most `max_requests` requests; the harness stops a turn still running past the cap only as a backstop. The message also raises the client's own response timeouts (60 s for a first reply, 300 s after tool results) to `timeout_s` + 30 s (option `response_timeout_ms`), so a local model may think longer than 60 s. A scenario's `limits` override both, and `limits.max_tool_calls` becomes a `max_tool_calls` check graded only live. The turn record holds the prompt, every executed batch from the action traces (calls with arguments and results, plus the trace's `state_delta`), the assistant texts (`final_text` is the last non-empty one), the turn metrics, the turn outcome (`stop`, `error` with its `error_source`, `truncated`, `filtered`, `max_requests`, `abandoned`, or the harness's `timeout`, `request_cap`, `not_started`) and the signals of 4.7 (calls against the reference, tool errors and the ones recovered from, dropped calls). Live runs are serial and never retry a scenario. After a stopped, timed-out or failed turn (whose last request is counted as sent), and before every scenario, the harness waits (up to 120 s, `drain_s` in the turn record) until `GET /api/requests_in_flight` reports no model request running: the server drops a stopped turn's late reply (see the conversation generation in `static/openai_api_base.py`), and the wait keeps that request from sharing the model server with the next turn.
   - `--provider openrouter` adds the spend guards: the plan (turns times each turn's request cap, times models and `--repeats`) must fit `--max-requests` (default 250) before anything is sent; each turn's client limit is also capped at the requests the run has left, so the run's total is hard too, and the turn that uses up the cap ends the run (exit 1); a request in flight counts as sent, also when a turn breaks off on a browser or hook error. `--dry-run` prints the plan and, for OpenRouter, a cost ceiling from `static/model_prices.py`.
   - A failing scenario, and every scenario with a turn of more than one batch of counted calls, is retraced at once (unless `--no-retrace`): its executed calls run again batch by batch on a fresh session and the canvases are compared after every step, which classifies its failures (4.7). When the retrace reproduces the live canvas, its per-batch I4 and I5 verdicts replace the live turn-level ones (`judged_by: retrace`). A scenario with a turn that ended in `timeout`, `request_cap` or `not_started`, in an `error` from the provider or the transport, or whose action traces could not be read, is `infra` from that turn on, is not retraced and is left out of the pass rates. An `error` raised by the app itself (`error_source` `client`: an exception in the browser while handling the reply; `server`: an exception in the route) is `app`, and so is a turn the server `abandoned`: the harness's tab is the only client of its server and stops turns only through the client, so an abandoned turn points at the app's turn bookkeeping. The live command exits 1 on `app` or `nondeterministic` failures, on a stop at the request cap, and when more than `--max-infra-rate` (default 0.5) of the runs, or all of them, are `infra`; model mistakes are measurements. `summary.md` adds a per-model table (pass rates, failure classes, mean turn time, first token, requests and tokens, calls against the reference, tool errors, dropped calls) and, with `--repeats` or several models, passes per scenario.
9. `--mode retrace RESULTS` retraces every run stored in a live `results.json` on a server of its own and writes its results to `retrace-<time>/` next to it, with a table of the live runs, their classes and whether each was reproduced; runs with an infrastructure failure are left out, and `--dry-run` lists what would be retraced. `--regrade` also works on live results (answer checks use the stored final text), and regrades the stored retrace steps too.
10. `--attach-desktop N` runs `--mode replay` or `--mode live` in the desktop window opened with `python mathud_desktop.py --automation-port N`, with `--pace` seconds after each step (default 1.5). It asks before resetting the window unless `--yes`, pins nothing (the app's `.env` applies; see the attach-mode note under phase 6 in section 7) and keeps the model guard.

## 1. Summary

1. **Drive the real app in headless Chrome** through the existing Selenium setup (`cli/browser.py`). The canvas state that matters is the one the Brython client holds, so the harness reads it from the page. A server-only loop would test a stand-in for the canvas, not the canvas.
2. **Add six small `window.getMatHud*`-style hooks** to the client: canvas state (with an inspection view of attributes the state omits), running a tool batch exactly as a model batch runs, sending a chat message, turn status, stopping a turn, and resetting the session.
3. **Scenarios are JSON files** under `scenarios/`, one file per area. Each scenario has setup, one or more chat turns, each with a reference tool-call sequence, optional scripted steps (undo, save, load) and checks.
4. **Checks select objects by type and geometry**, not by names the model invents. There are two kinds: outcome checks (did the request get done?) and invariants that must hold after every step (unique names, no dangling references, derived fields consistent with geometry, truthful tool results, one undo entry per batch).
5. **Two run modes.** `replay` runs the reference calls with no model: deterministic, about 3 minutes for the whole catalogue, and suitable for CI. `live` sends the prompts to a model: LocalAgent on llama-server by default, OpenRouter behind the canvas benchmark's spend guards.
6. **Failures are classified** as app bug, model mistake, non-deterministic, infrastructure or known bug. Two tools do the classifying: invariants, and re-executing a live run's actual tool calls on a fresh canvas.
7. **Integration:** `python -m cli.main test scenarios --mode replay|live`, with `results.jsonl`, `summary.md`, failure screenshots and saved states, and `--regrade`. Replay runs in CI, non-blocking at first, with known bugs as expected failures.

## 2. Goals and non-goals

Goals:

1. Catch app bugs in object lifecycles: creation with the right attributes, relations between objects, naming and collisions, updates and dependent updates, transforms of composite objects, cascading deletes, undo and redo, and workspace round-trips.
2. Catch mismatches between what a tool reports and what is drawn.
3. Measure chained tool use on the local model: outcome pass rate, tool calls compared with the reference, tool errors, dropped calls, latency and tokens. This fits the Roadmap's "Benchmark suite (CLI)" and "Local-model tuning" items (A1, Model workbench, items 2 and 4).
4. Keep scenarios cheap to write: a scenario author writes only the arguments that matter, and the loader fills the rest with `null` from the schema.

Non-goals:

1. Replacing the Brython unit tests (about 2,950) or the server pytest suite.
2. Pixel-level rendering checks. A screenshot is saved for every failure, but nothing is graded from pixels.
3. Grading free-form prose. Answer checks look only for numbers or names in the final message, and they count toward model quality, never toward app bugs.

## 3. What exists today

| Building block | Where | How the framework uses it |
|---|---|---|
| Headless Chrome through Selenium, with Chrome and chromedriver resolution for Windows, Linux and ARM | `cli/browser.py` (`BrowserAutomation`) | Reused as the driver. `wait_for_app_ready` polls for `window.startMatHudTests`. |
| Server lifecycle, port selection, venv interpreter | `cli/server.py` (`ServerManager`), `cli/config.py` | Reused. `--start-server` works like `test client`. |
| Turn metrics | `static/client/turn_metrics.py`; `window.getMatHudLastTurnMetrics()`, `getMatHudTurnMetricsHistory()`, `clearMatHudTurnMetrics()` (`ai_interface.py:162-178`) | Reported per turn: model, wall time, time to first token, tokens, tokens/s, requests, `tool_calls` (emitted by the model) and `tool_executions` (run by the client). |
| Action traces: every executed call with arguments, result, `is_error` and duration, plus before and after states on the newest trace | `managers/action_trace_collector.py`; `window.getActionTraces()`, `getLastActionTrace()`, `clearActionTraces()`, `replayLastTrace()` (`ai_interface.py:132-151`) | The tool-call log of a live turn. Its `state_delta` is always empty today (K20), so the harness diffs states itself. |
| The model's tool-execution path | `ProcessFunctionCalls.get_results_traced` → `ResultProcessor.get_results_traced` (`result_processor.py:101-176`) | Replay runs batches through this same function, so the replay and live paths differ only in who chose the calls. |
| Canvas state | `Canvas.get_canvas_state()` (`canvas.py:571-620`): buckets such as `Points` or `Segments`, each a list of `{name, args}`, plus `Cartesian_System_Visibility` and `coordinate_system` | The primary object of every check. It omits colours and some cached values, hence the inspection view (4.3). |
| Canvas-state fixtures | `server_tests/fixtures/canvas_states/*.json` | Same shape as a workspace `state`. They can seed scenarios. |
| Canvas-format benchmark conventions | `scripts/benchmark_canvas_formats.py` | Reused: the `--provider` switch (`openrouter` or `local`), local model discovery from `/v1/models`, pinned environment variables, `--max-requests` checked before anything is sent, no SDK retries, `results.jsonl` appended as results arrive, `results.json` and `summary.md` also written on Ctrl+C, `--regrade`, and `--dry-run`. |
| Tool-discovery dataset | `server_tests/data/tool_discovery_cases.yaml` (JSON-compatible YAML) | Precedent for data files that need no PyYAML: the scenario files use plain JSON. |
| Tool argument validation | `static/tool_argument_validator.py` | The loader validates every reference call with it. |

Two things in the existing CLI do not work and must not be built on:

1. `cli/browser.py:306-359` (`get_canvas_state`, `call_canvas_method`, `call_function_registry`) reads `window._canvas`, but nothing sets it. `main.py:140,193` keeps `_canvas` as a Brython module global. As a result:
   - `python -m cli.main canvas state` prints `{}`.
   - `canvas clear` and `canvas zoom` report success without doing anything. `canvas exec` says the function ran with no return value. `canvas undo` and `canvas redo` always say there is nothing to undo or redo.
   - Canvas has no `get_state` or `reset_view` method either.
   - This is K22. The hooks below give these commands a working backend.
2. `start_new_conversation` (`ai_interface.py:1059-1074`) saves a workspace file as a side effect. A reset between scenarios needs its own hook.

## 4. Architecture

### 4.1 How a scenario runs

```
 harness (CPython)                          browser (Brython app)                  Flask + provider
 ─────────────────                          ─────────────────────                  ────────────────
 load scenarios/*.json, validate
 start server (pinned env) ───────────────────────────────────────────────────────▶ app.py --port N
 open headless Chrome, wait for app ──────▶ page loads (≈3 s)
 per scenario:
   resetMatHudSession(fixture?) ──────────▶ clear canvas, undo stacks, names,
                                            traces, metrics, chat; POST
                                            /new_conversation ─────────────────────▶ reset history
   setup batches: runMatHudToolCalls ─────▶ same path as a model batch
   per step:
     replay: runMatHudToolCalls(reference)
     live:   sendMatHudMessage(prompt) ───▶ ai_interface.send_user_message ──────▶ model ⇄ search_tools
             poll getMatHudTurnStatus ◀──── tool batches run in the client ◀──────── tool calls
             (stop after the timeout or the cap)
     scripted: runMatHudToolCalls(do)       (undo, save/load, ...)
     read getMatHudCanvasState({inspect})
          getActionTraces(), getMatHudLastTurnMetrics()
     run checks (pure Python) and invariants
     on failure: screenshot + state saved
   append one record per step to results.jsonl
 write results.json and summary.md (also on Ctrl+C)
```

Each scenario starts from a reset session rather than a page reload. The prototype measured about 2.9 s for a reload, and a reset should take well under a second. A reload is still used after a failed or timed-out step, in case the page is in a bad state. One prototype run hung once after about 60 reloads, and a rerun passed in 6 s, so every step needs a timeout and the harness must be able to restart the browser.

### 4.2 Why drive the browser rather than a server-only loop

| | Browser-driven (proposed) | Server-only loop |
|---|---|---|
| Where tools run | In the real Brython client, through `ProcessFunctionCalls`, the managers, undo archiving and the dependency manager | Nowhere. The canvas lives only in the browser, so a server-only loop must either fake tool execution or re-implement the canvas in CPython. |
| What is checked | The canvas the user sees, plus the exact state the next request sends to the model | A simulation. It would miss every app bug in section 6: all of them except the CLI bug K22 live in `static/client/`. |
| Model conversation | The app's own: streaming, the `search_tools` interception and injection (`routes.py:187-247`), `[canvas changes]`, per-call results keyed by tool-call id, the continuation loop in `ai_interface._on_stream_final` | It would have to reproduce `ai_interface`'s loop, so it would drift from it. |
| Cost | About 3 s per page load (measured). Chrome must be installed, as it already is for `test client` and CI. | Faster, with no browser. |
| Determinism | The canvas code gave the same result for the same call sequence wherever the prototype repeated one. Layouts that may use randomness (force-directed graphs) need a check per layout. | Deterministic, but it tests the wrong thing. |

The server-only loop is fine for comprehension benchmarks, which is why `benchmark_canvas_formats.py` uses one. For this framework, the canvas in the browser is the thing under test, so the browser is not optional. The server still contributes: its logs keep `response_metrics` lines and the calls it dropped for not being loaded (`routes.py:322-349`), and the harness records both.

### 4.3 Client hooks

These follow the existing pattern: functions on `window` that take and return JSON strings, like `getMatHudTestResults`. They live in one new Brython module, `static/client/scenario_hooks.py`, registered from `main.py` next to `startMatHudTests`. Its logic sits in pure-Python helpers so the Brython runner can test it.

| Hook | Returns | Notes |
|---|---|---|
| `getMatHudCanvasState(optionsJson)` | `{"state": <get_canvas_state()>, "inspection": {...}?}` | `{"inspect": true}` adds, per drawable: class, name, `color`, the attached label (text, visible), derived values (angle `angle_degrees` and its vertex and arms, polygon vertices in order) and, for functions with listed vertical asymptotes, `asymptote_probes` (`[a, f(a - h), f(a + h)]`). It also adds `undo_depth`, `redo_depth`, `coordinate_mode`, `grid_visible` (cartesian, polar, active), `polar_radial_spacing` and the name-generator hints. These are the things `get_canvas_state` does not show (K5, K17, K25). `samples` (`{name or "*": [x, ...]}`) and `t_samples` add function values computed by the app's own evaluator. |
| `runMatHudToolCalls(callsJson)` | `{"traced": [...], "undoable": [...], "state": ..., "trace_id", "undo_depth_before", "undo_depth_after", "redo_depth_before", "redo_depth_after"}` | Runs one batch through `AIInterface.execute_tool_batch`, the method both model response paths now use: `ProcessFunctionCalls.get_results_traced(calls, ai.available_functions, ai.undoable_functions, canvas)`, result storage, the current turn's metrics and an action trace, so replay and live leave identical traces. Calls are `{"function_name", "arguments"}` or the scenario form `{"tool", "args"}`. The prototype used `__BRYTHON__.runPythonSource` to do this; the hook replaces that trick. |
| `sendMatHudMessage(text, modelId)` | `{"status": "started"}`, `"busy"` or `{"status": "error", ...}` | Sets `#ai-model-selector` to `modelId` and fails if that option is missing. Turns vision off unless asked. Then calls `ai_interface.send_user_message(text)`. |
| `getMatHudTurnStatus()` | `{"processing": bool, "completed_turns": n, "last_outcome": ..., "tool_batches": k, "requests": r, "tool_calls": c}` | "Done" means `processing` is false and `completed_turns` went up. `requests` and `tool_batches` let the harness enforce per-turn caps while a turn runs. The client itself has no tool-loop cap. |
| `stopMatHudTurn()` | `{"status": ...}` | `ai_interface.stop_ai_processing()`. Used for caps and timeouts. The client's own timeout is 60 s for the first response and 300 s after tool results (`constants.py:66-67`). |
| `fitMatHudView(optionsJson?)` | `{"status": "ok", "fitted": bool, "extent", "view"}` or `"busy"` | Zooms the view (through `Canvas.zoom`, like a mouse zoom: no undo entry, redo stack and drawables untouched, coordinate mode kept) to the bounding box of every drawable plus 12% padding on each side: points, segment, vector and polygon vertices, circles, arcs and rotated ellipses by their extent, labels, bars, parametric curves by samples, functions over their bounds with Tukey-fenced samples so an asymptote does not stretch the view. An empty canvas leaves the view alone; a single point gets a window of about +-1. Added for attach mode. Only the automation paths call it (the hook, attach mode, `desktop prompt` and `desktop fit`): in regular use the app never pans or zooms the canvas on its own. |
| `resetMatHudSession(optionsJson?)` | `{"status": ...}` | Clears the drawables without archiving. Clears the undo and redo stacks, computations, the name-generator state, view, coordinate system (cartesian), grid visibility, both grids' zoom-adapted spacing (through `Canvas.reset`, which resets the polar grid's too since the K25 fix), traces and metrics. Options: `fixture` restores a canvas state through `WorkspaceManager`'s restore phases and then clears the undo stack; `chat` (default true) clears the chat DOM; `conversation` (default false) POSTs `/new_conversation`. Replay leaves it off because every POST starts a new server session log and `log_manager.py` keeps only the newest 50. It never saves a workspace. |

`runMatHudToolCalls` answers `{"status": "busy"}` while a chat turn runs, so scripted calls never mix into a user turn's canvas or metrics. `sendMatHudMessage` switches vision off only for the request it sends and restores the user's toggle afterwards. Tracing is diagnostic: if building or storing the action trace fails, `execute_tool_batch` logs a warning and returns `trace: None`, and the turn goes on (tool log, stop handling, the next request) without a trace summary.

The existing `getActionTraces()` / `clearActionTraces()` and `getMatHudLastTurnMetrics()` are used as they are; `state_delta` now reads real states (K20 fixed). The CLI's canvas commands use the hooks (K22 fixed): `canvas state` reads `getMatHudCanvasState` (`--inspect` adds the inspection view), and `canvas exec`, `clear`, `reset`, `undo`, `redo` and `zoom` run tools through `runMatHudToolCalls`.

One small server change was also needed, and is done: a `MATHUD_WORKSPACES_DIR` override (`static/config.py`, `get_workspaces_dir`). Workspace tools write into `./workspaces` of the checkout (`static/workspace_manager.py:59-66`), so without the override a scenario run would share that directory with the user's own workspaces. The harness points it at a temporary directory.

### 4.4 Scenario files

Scenarios are plain JSON under `scenarios/`, one file per area (`scenarios/geometry.json`, `scenarios/constructions.json`, ...). JSON is enough: `tool_discovery_cases.yaml` already shows the repo avoids a PyYAML dependency. Example (GEO-04 from the catalogue):

```json
{
  "schema": 1,
  "scenarios": [
    {
      "id": "GEO-04",
      "title": "Point on a segment splits it; undo is one step",
      "tags": ["points", "segments", "undo"],
      "smoke": false,
      "known": ["K1"],
      "setup": {
        "fixture": null,
        "calls": [
          {"tool": "create_segment", "args": {"x1": 0, "y1": 0, "x2": 4, "y2": 0}},
          {"tool": "create_segment", "args": {"x1": 4, "y1": 0, "x2": 0, "y2": 3}},
          {"tool": "create_segment", "args": {"x1": 0, "y1": 3, "x2": 0, "y2": 0}}
        ]
      },
      "steps": [
        {"snapshot": "before"},
        {
          "user": "Mark a point M on segment AB at (2, 0).",
          "reference": [{"tool": "create_point", "args": {"x": 2, "y": 0, "name": "M"}}],
          "limits": {"max_tool_calls": 3, "timeout_s": 300},
          "checks": [
            {"bind": "M", "select": {"type": "Point", "at": [2, 0]}},
            {"check": "exists", "select": {"type": "Segment", "ends": [[0, 0], [4, 0]]}},
            {"check": "relation", "relation": "collinear",
             "select": [{"type": "Segment", "ends": [[2, 0], [0, 0]]}, {"type": "Segment", "ends": [[2, 0], [4, 0]]}]}
          ]
        },
        {"do": [{"tool": "undo", "args": {}}]},
        {"checks": [{"check": "state_equals", "snapshot": "before", "known": "K1"}]}
      ]
    }
  ]
}
```

Rules:

1. `args` holds only the arguments that matter. The loader fills every other required argument with `null`, as a strict-schema model sends it, and then validates the call with `ToolArgumentValidator`. A pytest gate fails if any reference call has an unknown tool, an unknown argument or an invalid value. Filling in nulls matters: `create_colored_area` fails on `color: null` (K10), and only a reference that sends what a real model sends finds that.
2. Step kinds:
   - `user`: a chat turn. In live mode `user` is sent to the model; in replay mode `reference` runs instead.
   - `do`: scripted calls, run the same way in both modes and never shown to the model. Used for undo, save, clear and load, and for anything else that checks the app rather than the model.
   - `snapshot`: stores the current state and inspection view under a name.
   - `checks`: checks with no action attached.
3. `setup.fixture` names a canvas-state file, such as one from `server_tests/fixtures/canvas_states/`. `setup.calls` are replayed tool calls. Setup always runs as replay, even in live mode.
4. `limits` per turn: `max_tool_calls` (a model-quality check), `timeout_s` and `max_requests` (a hard stop through `stopMatHudTurn`).
5. `known` on a check (`"K<n>"`) makes it an expected failure. When it passes, the report shows XPASS ("fixed? K<n>"), so a fixed bug gets its marker removed. `known` on the scenario (a string or a list) lists every bug the scenario touches; the loader requires it to include every check-level mark and waiver. `invariants` (`{"I4": "K3"}`) waives an invariant for that scenario only; global waivers live in `scenarios/known_bugs.json`.
6. Every scenario gets two snapshots for free: `start` (after the reset and any fixture) and `setup` (after the setup calls). Steps are named `t1`, `t2` (turns), `do1` (scripted), `chk1` (checks only) and `snap1`, unless they set `id`.
7. Files also carry `"schema": 1` and `"area"`; a scenario may set `tol` (a number or `{"abs", "rel"}`), `targets` (what it aims at) and `allow_large` (switches off I7's magnitude limit).

### 4.5 The check language

Checks are pure functions of stored data: the state and inspection view after each step, that step's tool calls and results, turn metrics and the final assistant text. That keeps them testable with pytest and makes `--regrade` possible without a browser.

**Geometry view.** The checker first builds a normalized view of a state:

- points resolved to coordinates;
- every segment, vector, polygon, circle, ellipse, arc and angle resolved to coordinates through its point references;
- functions as numeric samples, either taken in the browser through the inspection hook or computed by an evaluator in CPython (open question 8.2).

**Selectors** pick objects without relying on names the model chose:

| Selector | Matches |
|---|---|
| `{"type": "Point", "name": "A"}` | By name, only when the user named the object in the prompt |
| `{"type": "Point", "at": [x, y]}` | A point within tolerance |
| `{"type": "Segment", "ends": [[x1, y1], [x2, y2]]}` | A segment with those endpoints, in either order |
| `{"type": "Triangle", "vertices": [[...], [...], [...]]}` | A polygon with that vertex set, in any rotation or orientation |
| `{"type": "Circle", "center": [x, y], "radius": r}` | Either field may be left out |
| `{"type": "Function", "samples": [[x, y], ...]}` | A function whose values match at those x |
| `{"type": "Circle", "only": true}` | The only object of that type; fails if there are zero or several |
| `{"type": "Point", "new_since": "before"}` | Objects added since a snapshot |
| `"$M"` | An object bound earlier with `{"bind": "M", "select": ...}`. Bindings follow the object by its bucket and name across steps. |
| `{"type": "Segment", "contains": [x, y]}`, `"through"`, `"slope"` | A segment containing a point, whose line passes through a point, or with that slope |
| `{"type": "Vector", "origin": [x, y], "tip": [x, y]}` | Vectors by either end (`ends` is ordered for vectors) |
| `{"type": "Point", "within": [xmin, ymin, xmax, ymax]}` | Objects whose position lies in a box |
| `{"type": "Label", "where": {"args.text": "hello", "inspect.color": "red"}}` | Any state or inspection field, by dotted path |
| `{"coords": [x, y]}` | A literal location, for relations |

`type` is the bucket name without its final `s` (`Point`, `Segment`, `CircleArc`, `FunctionsBoundedColoredArea`, ...), or a family: `Polygon`, `AnyFunction`, `ColoredArea`, `Graph`, `Plot`. Every selector may set its own `tol`.

**Outcome checks.** A failure in live mode is charged to the model, unless the replay fails too:

| Check | Meaning |
|---|---|
| `exists` / `absent` / `count` | Object selection with `==`, `<=` or `>=` |
| `point_at` / `moved` | Position, or displacement relative to a snapshot (`{"by": [dx, dy]}`) |
| `relation` | `point_on_circle`, `point_on_segment`, `point_on_line`, `point_on_function`, `collinear`, `parallel`, `perpendicular`, `midpoint_of`, `equal_length`, `equal_angles`, `tangent_to` (line to circle or function), `distance`, `length`, `slope`, `direction`, `angle_deg`, `area`, `function_value`, `inside` |
| `attribute` | A state or inspection field (`path`, e.g. `args.label.visible` or `inspect.color`) of a selected object, or of `"target": "view"`, `"state"` or `"inspection"`. Comparisons: `eq` (with `mod`), `ne`, `lt`, `le`, `gt`, `ge`, `contains`, `not_contains`, `set_eq`, `in`, `matches`, `has_numbers`, `len`, `is_null`; `same_as` compares with the same field at a snapshot |
| `state_equals` | Equal to a snapshot within tolerance, optionally with `"inspect": true`, `"ignore": [...]` and `"match_names": false`. Used for undo, redo, round-trips and rotate-and-back. |
| `unchanged_except` | Nothing but the listed selectors changed since a snapshot. Catches side effects such as K9 and K15. |
| `tool_called` / `tool_not_called` / `max_tool_calls` | Over the turn's executed calls, not counting `search_tools` |
| `no_tool_errors` / `tool_error` | No call failed / a given call failed (an `Error...` string, a flagged call or an `{"error": ...}` dict) |
| `tool_result` | A result matches a number (with tolerance), a set of names, or a JSON path, e.g. `analyze_graph` `path == [A,B,C,D]` or `cost == 4`. It checks the tool's last call, or call `index`; with `any: true`, any one of its calls may match. |
| `answer_mentions` | Numbers or names in the final assistant text. Model quality only. |

**Invariants.** They run after every step of every scenario. A failure is always an app bug, whoever made the calls:

| Id | Invariant | Would have caught |
|---|---|---|
| I1 | Names are unique within a bucket. Clashes across buckets, such as a segment and a vector both named `AB`, are warnings. | K7 name clash |
| I2 | No dangling references: segment, polygon, circle, arc and angle endpoints exist, and areas, plots and graphs name existing objects | orphans after deletes |
| I3 | Derived fields agree with geometry: `circle_formula` and `ellipse_formula`, `_p1_coords`, triangle and quadrilateral `types`, the angle's degrees (inspection), function asymptotes (a listed vertical asymptote must blow up when sampled, and a translated function's asymptotes must move) | K8, K11, K12, K16, K17, K24 |
| I4 | Tool results tell the truth: a create or delete reported as a success changed the state; a call that did nothing, or failed, did not report success; an error changed nothing; a created object's real name appears in the result | K2, K3 |
| I5 | Undo accounting, judged by what happened rather than by what the calls reported: a batch that changed the canvas adds exactly one undo entry, a batch that changed nothing (failed, refused or truthful no-op calls) adds zero, and an undo or redo moves the stack by one. "Changed" is judged as in I4: drawables, the view, the coordinate mode and inspection-only fields such as colour, labels and grid visibility all count. An explicit `load_workspace` that restores the canvas it replaced may add one entry. | K1 |
| I6 | Errors are flagged: a result shaped `{"error": ...}` counts as an error even though `ResultProcessor` does not flag it | K21 |
| I7 | Numbers are sane: no NaN or infinity, and no coordinate above 1e12 in magnitude unless the scenario allows it | K13 |

Some known bugs break an invariant almost everywhere; K1 breaks I5 in nearly every scenario. Such waivers live in one file, `scenarios/known_bugs.json` (`"invariant_waivers": {"I5": "K1"}`), and the report counts those failures as known rather than as app failures. A waived invariant that holds is reported as a pass, never as XPASS, because an invariant can hold in one step and break in the next. When the bug is fixed, the waiver is removed and the invariant becomes a regression guard.

As built, I3 also checks that a circle arc's endpoints lie on its circle and that a function's bounds are ordered, and the loader rejects keys a check type does not accept (so a typo such as `kown` fails the catalogue gate), required keys that are missing (`point_at` without `at`, `moved` without `by`, `tangent_to` anything but a literal Circle selector without `x`, bindings included), relations with the wrong number of selectors, and relation parameters that are missing (`direction` needs `parallel_to` or `perpendicular_to`; `distance`, `length`, `slope`, `area` and `angle_deg` need `value`). I4 checks three things: a batch whose calls all failed changed no drawables; in a batch that changed nothing (drawables, view, mode, colours and grid all as before), every successful mutating call is judged on its own: a bare success result ("Call successful!", True or nothing) fails, and any other result must admit the no-op in so many words (`no`, `not`, `nothing`, `already`, `unchanged` or `empty`, as in "Point 'C' already exists ..." or "Nothing to undo ..."), so one truthful call cannot excuse another's bare success; and a create call whose requested name no object got names the object it did create in its result, or, when it created nothing, explains itself. The naming rule is skipped for batches that also delete, clear, load, undo or redo, since an object created and removed in one batch leaves no trace.

**Tolerances.**

- The default is absolute `1e-9` plus relative `1e-9` for app-computed geometry.
- For values the model computes itself (hexagon vertices, polar conversions), the scenario sets a looser tolerance such as `1e-3`, so a model that rounds to three decimals still passes.
- Sampling checks use at least five sample points, away from any asymptote.

### 4.6 Run modes

**Replay (`--mode replay`).**

- Runs setup, then every step's `reference` and `do` calls through `runMatHudToolCalls`, and every check and invariant.
- No model, no GPU and no network other than localhost.
- It has two jobs:
  - It is a standalone app regression suite: invariants and outcome checks over realistic call chains.
  - It validates the scenarios themselves. A new scenario must replay green, or carry a `known` mark for each failing check, before it is merged. That proves the checks are correct for a known-good call sequence.
- Measured with the prototype: 2.2 to 3.2 s per scenario including a page reload, and 29 s for 11 scenarios including Chrome start-up. The whole catalogue (72 scenarios) takes about 3 minutes, and less once resets replace reloads.

**Live (`--mode live`).**

- Sends each `user` prompt through `sendMatHudMessage` and waits for the turn to finish. The limits from 4.4 apply, and every stop is recorded as an outcome.
- Runs the same checks and invariants.

**Providers** in live mode:

- `--provider local` is the default. The model is whatever llama-server reports at `/v1/models`, using the same code as the benchmark (`LocalAgentAPI.fetch_models`), and `--models` can pick one. `MATHUD_LOCAL_REASONING_EFFORT` is pinned through `--local-reasoning-effort`, with the benchmark's choices and default.
  - Before the first message, the harness checks that the model id appears under `local_agent` in `/api/available_models`.
  - This check is a spend guard: an unregistered id falls back to the OpenAI provider (`routes.py:528-532`), so a stale id could otherwise reach a paid API.
- `--provider openrouter` must be asked for explicitly and needs `OPENROUTER_API_KEY`. It uses the benchmark's guards:
  - `--max-requests`, checked before anything is sent against scenarios × turns × `max_requests` per turn;
  - a hard stop when the running total of `requests` from the turn metrics reaches the cap;
  - no retries, and a `--dry-run` that prints the plan and a cost estimate with the benchmark's `PRICES_PER_MTOK` table (moved into a shared module).
- The harness starts its own server with pinned settings, as the benchmark does:
  - `MATHUD_TOOL_EXPOSURE` (default `search`);
  - `MATHUD_CANVAS_FORMAT` (default `text`);
  - `MATHUD_CANVAS_BUDGET_TOKENS`, `TOOL_SEARCH_MODE`, `MATHUD_WORKSPACES_DIR`, and `REQUIRE_AUTH=false`.
  - These pins let a local `.env` change nothing except `LOCAL_AGENT_BASE_URL` and the key.
  - They are recorded in the run config, so tool exposure and canvas format can be compared across runs.
- Live runs are serial: the server keeps one global conversation (`app.ai_api`), and llama-server serves one model at a time.

**Trace re-execution (`--mode retrace RESULTS`).** Takes the executed calls of a live run, batch by batch from its action traces, and replays them on a fresh session. No model is involved. This is the tool for classifying failures.

### 4.7 Classifying failures

| Class | Rule |
|---|---|
| `app` | Any invariant fails, in any mode; or an outcome check fails in replay and is not marked `known`. |
| `model` | An outcome check fails live, replay passes, and re-executing the live trace reproduces the same failing state. The model's own calls produce the wrong canvas. |
| `nondeterministic` | Re-executing the live trace does not reproduce the live state, or repeated runs of the same trace differ. This points to app non-determinism (for example force layouts) or to harness timing, and needs investigating. |
| `known` | A failing check marked `known`, or a waived invariant. When a marked check passes: `XPASS`. |
| `infra` | Turn outcome `error` or `timeout`, llama-server unreachable, provider errors, a browser crash. Not counted in accuracy, like the benchmark's request errors. |
| `check` | A replay failure right after a scenario was added or edited, caught by the pytest gate that requires the catalogue to replay green. |

Two more signals are reported but not classified:

1. **Dropped calls.** Turn metrics `tool_calls` minus `tool_executions`: calls the server dropped because the model used a tool it had not loaded through `search_tools` (`routes.py:322-349`). The client executes `search_tools` too, so it counts on both sides.
2. **Efficiency.** Executed calls compared with the reference's call count, and tool errors the model recovered from (CON-01 expects one refused `update_point` and a switch to `translate_object`).

### 4.8 Reporting

Output goes to `logs/scenario_runs/<time>/`, following `logs/canvas_format_benchmark/`:

1. `results.jsonl`: one record per step, appended as it finishes, containing:
   - scenario and step ids, mode, provider, model;
   - the prompt, the executed calls with arguments and results, and the final assistant text;
   - the turn metrics (wall time, time to first token, requests, prompt and completion tokens, tokens/s, `tool_calls`, `tool_executions`, `tool_errors`, outcome);
   - the state and inspection view after the step, and each check with pass/fail, class, expected and actual values.
2. `results.json` and `summary.md` at the end, also on Ctrl+C.
   - The summary has a pass/fail table per scenario and per area, per model.
   - It also shows outcome and invariant pass rates, known and XPASS counts, the smoke subset on its own, mean latency and tokens per turn, and calls against the reference.
   - It ends with a list of failures giving the check, expected and actual values, class, and a link to the saved state and screenshot.
3. `failures/<scenario>__<step>.png` (a full-page screenshot through `capture_screenshot`) and `failures/<scenario>__<step>.json` (the state and inspection view).
4. `--regrade RESULTS_JSON` re-runs every check against the stored states with the current checker and scenario files, with no browser, and writes `results_regraded.json` and `summary_regraded.md`. Useful after a checker fix or a tolerance change.
5. `--dry-run` loads and validates all scenarios, prints the plan (scenarios, turns, estimated requests and cost for OpenRouter) and sends nothing.

### 4.9 Integration

1. CLI:
   ```
   python -m cli.main test scenarios --mode replay [--smoke] [--only GEO-04 CV-02] [--tags undo]
   python -m cli.main test scenarios --mode live --provider local [--models ID] [--repeats 3] [--smoke]
   python -m cli.main test scenarios --mode live --provider openrouter --models deepseek/deepseek-v4.1-flash --max-requests 150
   python -m cli.main test scenarios --mode retrace logs/scenario_runs/<time>/results.json
   python -m cli.main test scenarios --regrade logs/scenario_runs/<time>/results.json
   ```
   `--start-server`, `--port` and `--json` behave as in `test client`.
2. CI: add a `scenario-replay` job to `.github/workflows/tests.yml` alongside `client-tests`. It uses the same venv, Chrome setup and `--start-server`.
   - It runs the full replay (about 3 to 4 minutes including set-up) on pushes and pull requests.
   - It is non-blocking at first, with `continue-on-error`, like the client tests.
   - Once known bugs are marked and the job is stable, it should block. At that point, known bugs count as passes, and XPASS fails the job until the marker is removed.
   - The pytest gates (the scenario loader, argument validation and the pure check engine) join the blocking `server-tests` job straight away.
3. Live runs never run in CI: there are no keys by policy (`tests.yml:15`) and no GPU. They run by hand on the workstation.
   - They need llama-server with the model loaded on the GPU; the harness itself needs no GPU.
   - A local turn takes roughly 20 to 90 s: one to three requests (`search_tools`, the calls, the final answer) at medium reasoning effort. This is an estimate to replace with the first measured run.
   - That puts the smoke subset (18 turns) at about 6 to 27 minutes and the full catalogue (103 turns) at about 35 to 155 minutes per repeat.
   - `--repeats 3` gives a pass rate per scenario instead of a single sample.

### 4.10 Prototype evidence

While writing this design, a throwaway harness drove the real app (port 5032, `venv-ci`, headless Chrome) with the catalogue's reference calls, using `__BRYTHON__.runPythonSource` in place of `runMatHudToolCalls`.

- All reference calls passed `ToolArgumentValidator` after null filling.
- The first full pass stalled at WS-01, the 60th scenario, after about 60 page reloads. WS-01 and the scenarios after it then ran in two shorter passes, and WS-01 passed in 6 s (see 4.1).
- Observations from that run are quoted in section 6, marked "(observed)". Findings from reading code only are marked "(code)".

## 5. Scenario catalogue

Conventions:

1. **Reference calls** show only the arguments that matter, with values written as JSON (`true`, `false`, `null`). Every other required argument is `null`, filled in by the loader. Tool and argument names are checked against `static/functions_definitions.py`.
2. **Names in references** are the ones the app assigns on a fresh canvas, for example the first circle centre is `A`, so its circle is `A(3)`. In live mode the model finds names from the canvas; checks select by geometry.
3. **Setup (scripted)** runs as replay in both modes. **Scripted steps** are harness actions that are never shown to the model.
4. **Invariants I1 to I7** run after every step of every scenario and are not repeated in the checks.
5. **Known bugs** refer to section 6. Each such check is an expected failure until the bug is fixed. Bugs fixed since (K1, K2, K6, K7, K18 and K21 by vl3c/MatHud#72 and #73; K20 and K22 on this branch; K3, K5, K10, K15 and K27 by vl3c/MatHud#77; K4, K25 and K26 on `claude/fix-view-undo-noops`; K11, K12, K13, K14 and K24 on `claude/fix-functions`; K8, K9, K16, K17 and K19 on `claude/fix-stale-geometry`; K23 and K28 to K32 on `claude/fix-batch-7`) no longer carry marks in `scenarios/*.json`: their checks are regression guards, and `scenarios/known_bugs.json` lists them under `fixed`, which the loader refuses as a mark. The per-scenario "Known bugs" lines below keep the original design's list.

| Area | Scenarios | Smoke |
|---|---|---|
| Points, segments, vectors, polygons, circles, ellipses, arcs, angles and labels (GEO) | 18 | GEO-01 |
| Constructions (CON) | 6 | CON-01 |
| Functions, piecewise and parametric curves, tangents and normals (FN) | 11 | FN-01 |
| Coloured areas and regions (AR) | 4 | - |
| Transforms (TR) | 6 | TR-01 |
| Graph theory (GR) | 6 | GR-01 |
| Statistics, plots and regression (ST) | 5 | ST-01 |
| Math tools feeding the canvas (MC) | 4 | MC-01 |
| Canvas operations: view, coordinate systems, undo and redo (CV) | 6 | CV-02 |
| Workspaces (WS) | 4 | WS-01 |
| Naming, editing and deleting (NM) | 7 | NM-03 |
| Multi-turn follow-ups (MT) | 4 | MT-01 |
| **Total** | **81** | **11** |

The smoke subset has 11 scenarios (18 turns), one for each area except coloured areas, which FN-01 already exercises: GEO-01, CON-01, FN-01, TR-01, GR-01, ST-01, MC-01, CV-02, WS-01, NM-03 and MT-01. It runs in about 40 s in replay and about 6 to 27 minutes live on the local model.

### 5.1 Points, segments, vectors, polygons, circles, ellipses, arcs, angles and labels (GEO)

#### GEO-01: Three segments close into a triangle (smoke)

- Turn 1: "Draw segments from (0,0) to (4,0), from (4,0) to (0,3) and from (0,3) back to the origin."
  - Reference: `create_segment(x1=0, y1=0, x2=4, y2=0)`; `create_segment(x1=4, y1=0, x2=0, y2=3)`; `create_segment(x1=0, y1=3, x2=0, y2=0)`
- Checks:
  - count Points == 3, Segments == 3
  - exists Triangle with vertices {(0,0),(4,0),(0,3)} (auto-created); its types include `right`
  - every segment's endpoints are existing points (invariant I2); names unique (I1)
  - tool_calls <= 4 (excluding search_tools); no tool errors
- Targets: auto-construction when segments close a loop (`create_drawables_from_new_connections`).

#### GEO-02: Requested point name already taken

- Setup (scripted): `create_point(x=0, y=0, name="A")`
- Turn 1: "Add another point called A at (7, 7)."
  - Reference: `create_point(x=7, y=7, name="A")`
- Checks:
  - exactly one point named A, still at (0,0)
  - a point exists at (7,7) (any name); names unique
  - invariant I4: the name actually given to the (7,7) point appears in the tool result
  - model check: the final answer tells the user the point got another name (answer mentions it)
- Targets: silent renaming; tool result hides the real name.
- Known bugs: K3 (expected to fail until fixed; see section 6).

#### GEO-03: Creating a point on an occupied spot

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}], polygon_type="triangle", name="ABC")`
- Turn 1: "Put a point Z at (0, 3)."
  - Reference: `create_point(x=0, y=3, name="Z")`
- Checks:
  - Points count unchanged (3); no point named Z; C still at (0,3)
  - invariant I4: the result does not claim a new point was created (it should name C)
  - invariant I5: the refused create adds no undo entry
  - scripted `undo()` afterwards: removes the setup batch, so the state equals the `start` snapshot (with K1 it only pops the no-op entry and the triangle stays)
- Targets: silent reuse of an existing point, dropped name, no-op undo entries.
- Known bugs: K2, K3, K1 (expected to fail until fixed; see section 6).

#### GEO-04: Point on a segment splits it; undo is one step

- Setup (scripted): `create_segment(x1=0, y1=0, x2=4, y2=0)`; `create_segment(x1=4, y1=0, x2=0, y2=3)`; `create_segment(x1=0, y1=3, x2=0, y2=0)`
- Turn 1: "Mark a point M on segment AB at (2, 0)."
  - Reference: `create_point(x=2, y=0, name="M")`
- Scripted step: `undo()`
- Checks:
  - after the turn: M at (2,0); AB still exists; child segments MA and MB exist and are collinear with AB
  - after the scripted undo: state equals the setup snapshot (no MA left behind)
- Targets: segment splitting side effects; undo landing on a half-finished state.
- Known bugs: K1 (expected to fail until fixed; see section 6).

#### GEO-05: Huge and tiny coordinates, then zoom to them

- Turn 1: "Create a point named Far at (1000000000, -0.000000001) and a point at (1e-9, 1e-9), then zoom so the far point is in view with about 10 units on each side."
  - Reference: `create_point(x=1000000000, y=-1e-09, name="Far")`; `create_point(x=1e-09, y=1e-09)`; `zoom(center_x=1000000000, center_y=0, range_val=10, range_axis="x")`
- Checks:
  - a point exists at (1e9, -1e-9) with relative tolerance 1e-12 and one at (1e-9, 1e-9) with absolute tolerance 1e-15
  - the two points are distinct objects (no merging by a coarse coordinate tolerance)
  - view: left_bound <= 1e9 <= right_bound and right_bound - left_bound is about 20
  - invariant I7: no NaN or infinity anywhere in the state
  - a `create_point` result says that F was created instead of the requested name Far (`tool_result` with `any: true` and `matches`; a point name is one capital letter with optional primes, so `Far` is invalid; the result must say which name was used)
- Targets: numeric precision of storage, point matching and the view; multi-letter name coerced to a letter.
- Known bugs: K3 (fixed; see section 6).

#### GEO-06: Segment label: create, then hide

- Turn 1: "Draw a red segment from (-3,-3) to (3,3) with the label 'diag'."
  - Reference: `create_segment(x1=-3, y1=-3, x2=3, y2=3, color="red", label_text="diag", label_visible=true)`
- Turn 2: "Hide that label."
  - Reference: `update_segment(name="AB", new_label_visible=false)`
- Checks:
  - select the segment by endpoints {(-3,-3),(3,3)} (not by name)
  - after turn 1: label text `diag`, visible; color red (inspection hook)
  - after turn 2: label text still `diag`, visible false; color unchanged
- Targets: update paths that drop unrelated attributes.

#### GEO-07: Segment and vector on the same endpoints; delete the vector

- Turn 1: "Draw a segment from (0,0) to (3,0) and also a vector from (0,0) to (3,0)."
  - Reference: `create_segment(x1=0, y1=0, x2=3, y2=0)`; `create_vector(origin_x=0, origin_y=0, tip_x=3, tip_y=0)`
- Turn 2: "Now delete the vector but keep the segment."
  - Reference: `delete_vector(origin_x=0, origin_y=0, tip_x=3, tip_y=0)`
- Checks:
  - after turn 2: no Vectors; exactly one segment with endpoints (0,0),(3,0); both points remain
  - no tool errors (the call currently fails with `maximum recursion depth exceeded`)
  - invariant I1 (warning level): a segment and a vector both named AB
- Targets: mutual recursion between vector and segment deletion; cross-type name clash.
- Known bugs: K7 (expected to fail until fixed; see section 6).

#### GEO-08: Square given by exact corners

- Turn 1: "Draw the square ABCD with corners (0,0), (2,0), (2,2) and (0,2)."
  - Reference: `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 2, "y": 0}, {"x": 2, "y": 2}, {"x": 0, "y": 2}], polygon_type="quadrilateral", subtype="square", name="ABCD")`
- Checks:
  - one Rectangles entry whose vertex set equals the four given corners within 1e-12
  - types include `square`; four sides of length 2
  - vertex names follow the given order: B is at (2,0) (before the K19 fix, (0,2): vertices were re-ordered)
- Targets: subtype handling moving or re-ordering the given coordinates (before the K19 fix, 1.9999999999999996).
- Known bugs: K19, fixed on `claude/fix-stale-geometry`; the checks are regression guards.

#### GEO-09: Regular hexagon from a description

- Turn 1: "Draw a regular hexagon centred at the origin with circumradius 3, one vertex on the positive x-axis."
  - Reference: `create_polygon(vertices=[{"x": 3, "y": 0}, {"x": 1.5, "y": 2.598076211353}, {"x": -1.5, "y": 2.598076211353}, {"x": -3, "y": 0}, {"x": -1.5, "y": -2.598076211353}, {"x": 1.5, "y": -2.598076211353}], polygon_type="hexagon")`
- Checks:
  - one hexagon; all six vertices at distance 3 from (0,0) (tol 1e-6); a vertex at (3,0)
  - all sides equal (tol 1e-6)
- Targets: model computation plus polygon creation; mostly a model-quality scenario.

#### GEO-10: Move a free circle

- Turn 1: "Draw a circle centred at (2, 1) with radius 3."
  - Reference: `create_circle(center_x=2, center_y=1, radius=3)`
- Turn 2: "Move it so its centre is at (-1, 4)."
  - Reference: `update_circle(name="A(3)", new_center_x=-1, new_center_y=4)`
- Checks:
  - select `the only circle`
  - after turn 2: centre (-1,4), radius 3; circle name matches `<centre>(3)`
  - invariant I3: circle_formula agrees with centre and radius
- Targets: update path of circles, derived formula refresh.

#### GEO-11: Translating a triangle whose vertex is a circle centre

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}], polygon_type="triangle", name="ABC")`; `create_circle(center_x=0, center_y=0, radius=5)`
- Turn 1: "Move triangle ABC 10 units to the right."
  - Reference: `translate_object(name="ABC", x_offset=10, y_offset=0)`
- Checks:
  - A, B, C moved by exactly (10,0); the circle's centre is A, so it moved too (pinned: shared points move)
  - invariant I3: circle_formula reads `(x - 10.0)**2 + ...` (before the K8 fix, still `(x - 0.0)`)
  - unchanged_except: nothing but A, B, C, their segments, the triangle and the circle changed
- Targets: stale cached derived fields after a shared point moves.
- Known bugs: K8, fixed on `claude/fix-stale-geometry`; the checks are regression guards.

#### GEO-12: Ellipse rotation accumulates

- Turn 1: "Draw an ellipse centred at (-8, 5) with radii 3 and 1.5, rotated 30 degrees."
  - Reference: `create_ellipse(center_x=-8, center_y=5, radius_x=3, radius_y=1.5, rotation_angle=30)`
- Turn 2: "Rotate it by another 60 degrees."
  - Reference: `rotate_object(name="A(3, 1.5)", angle=60)`
- Checks:
  - select `the only ellipse`; after turn 2 rotation_angle is 90 (mod 180), centre and radii unchanged
  - invariant I3: ellipse_formula consistent with rotation 90
- Targets: rotation bookkeeping; ellipse names ignore the requested name (Roadmap known issue).

#### GEO-13: Arc endpoints must not drag existing points

- Setup (scripted): `create_segment(x1=0, y1=0, x2=3, y2=0)`
- Turn 1: "Draw the minor arc of the circle centred at the origin with radius 5, from (5,0) to (0,5)."
  - Reference: `create_circle_arc(point1_x=3, point1_y=0, point2_x=0, point2_y=5, center_x=0, center_y=0, radius=5, use_major_arc=false)`
- Checks:
  - B stays at (3,0) and segment AB keeps length 3
  - the arc's endpoints lie on the circle (distance 5 from the origin)
  - reference deliberately passes point1 = (3,0), a nearby existing point, as a sloppy model would
- Targets: constructions mutating unrelated existing points.
- Known bugs: K9, fixed on `claude/fix-stale-geometry`; the checks are regression guards.

#### GEO-14: Removing an angle marker keeps the triangle

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}], polygon_type="triangle", name="ABC")`
- Turn 1: "Mark the angle at A."
  - Reference: `create_angle(vx=0, vy=0, p1x=4, p1y=0, p2x=0, p2y=3)`
- Turn 2: "Remove that angle marker again."
  - Reference: `delete_angle(name="angle_BAC")`
- Checks:
  - after turn 1: one angle of 90 degrees at (0,0)
  - after turn 2: no angles; triangle ABC and all three sides still exist
- Targets: over-eager cascade deletes.
- Known bugs: K6 (expected to fail until fixed; see section 6).

#### GEO-15: Reflex angle

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}], polygon_type="triangle", name="ABC")`
- Turn 1: "Show the reflex angle at B."
  - Reference: `create_angle(vx=4, vy=0, p1x=0, p1y=0, p2x=0, p2y=3, is_reflex=true)`
- Checks:
  - one angle with is_reflex true at vertex (4,0); its size (geometry) is 360 - 36.87 = 323.13 degrees
  - inspection: cached angle_degrees equals the geometric value
- Targets: reflex handling in create and in cached values.

#### GEO-16: Labels with the same name, then edit one

- Turn 1: "Put a label 'x' at (1,1) called L and a label 'y' at (2,2) also called L."
  - Reference: `create_label(x=1, y=1, text="x", name="L")`; `create_label(x=2, y=2, text="y", name="L")`
- Turn 2: "Change the first label's text to 'hello' and tilt it by 15 degrees."
  - Reference: `update_label(name="L", new_text="hello", new_rotation_degrees=15)`
- Checks:
  - two labels with unique names; the one at (1,1) reads `hello`, rotation 15; the one at (2,2) is untouched
- Targets: label naming and update targeting.

#### GEO-17: Recolour an arc and switch it to the major arc

- Setup (scripted): `create_circle_arc(point1_x=5, point1_y=0, point2_x=0, point2_y=5, center_x=0, center_y=0, radius=5, use_major_arc=false)` (the arc is `ArcMin_AB`)
- Turn 1: "Make the arc red."
  - Reference: `update_circle_arc(name="ArcMin_AB", new_color="red")`
- Turn 2: "Now draw the major arc instead."
  - Reference: `update_circle_arc(name="ArcMin_AB", use_major_arc=true)`
- Turn 3: "Make sure it is the major arc."
  - Reference: `update_circle_arc(name="ArcMin_AB", use_major_arc=true)`
- Checks:
  - after turns 1 and 2: no tool errors (before the fix every call failed with "unexpected keyword argument 'point1_name'"); the arc is red, then the major arc and still red
  - after turn 3: the result says nothing changed
- Targets: the update path of arcs.
- Known bugs: K29, fixed on `claude/fix-batch-7`; the checks are regression guards.

#### GEO-18: Optional numbers sent as null take their defaults

- Turn 1: "Draw an ellipse centred at (2, -1) with radii 4 and 2."
  - Reference: `create_ellipse(center_x=2, center_y=-1, radius_x=4, radius_y=2)` (the loader sends `rotation_angle: null`)
- Turn 2: "Plot the parametric curve x = t, y = t^2."
  - Reference: `draw_parametric_function(x_expression="t", y_expression="t^2")` (with `t_min: null`, `t_max: null`)
- Checks: no tool errors; one ellipse with rotation_angle 0; one parametric curve with t_min 0
- Targets: create tools that crash on null optional numbers, the K10 pattern (before the fix the ellipse batch broke the hook with "must be real number, not NoneType").
- Known bugs: K30, fixed on `claude/fix-batch-7`; the checks are regression guards.

### 5.2 Constructions (CON)

#### CON-01: Midpoint is a static snapshot (documented) (smoke)

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 6, "y": 0}, {"x": 2, "y": 4}], polygon_type="triangle", name="ABC")`
- Turn 1: "Construct the midpoint of AB."
  - Reference: `construct_midpoint(segment_name="AB")`
- Turn 2: "Now move A to (1, 1)."
  - Reference: `translate_object(name="A", x_offset=1, y_offset=1)`
- Checks:
  - after turn 1: a new point at (3,0) (select by geometry, any name)
  - after turn 2: A at (1,1); AB and CA follow A (shared reference); the midpoint stays at (3,0) (pinned: Reference Manual says constructions are static)
  - model check: the model moved A with translate_object (update_point refuses referenced points) and needed at most one failed call
- Targets: dependent-object update semantics; model recovery from the update_point refusal.

#### CON-02: Perpendicular bisector undoes in one step

- Turn 1: "Draw AB from (0,0) to (4,0) and its perpendicular bisector."
  - Reference: `create_segment(x1=0, y1=0, x2=4, y2=0)`; `construct_perpendicular_bisector(segment_name="AB")`
- Scripted step: `undo()`
- Checks:
  - after the turn: a segment through (2,0) perpendicular to AB, length 6 (default)
  - after one scripted undo: only the batch's first call is undone at most; strictly, state equals the empty canvas (one batch = one step)
- Targets: nested archives in constructions.
- Known bugs: K1 (expected to fail until fixed; see section 6).

#### CON-03: Circumcircle and incircle

- Turn 1: "Draw the triangle with vertices (0,0), (6,0) and (2,4), then its circumcircle and incircle."
  - Reference: `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 6, "y": 0}, {"x": 2, "y": 4}], polygon_type="triangle")`; `construct_circumcircle(triangle_name="ABC")`; `construct_incircle(triangle_name="ABC")`
- Checks:
  - two circles; one passes through all three vertices (centre (3,1), r = sqrt(10))
  - the other is tangent to all three sides: distance from its centre to each side equals its radius (tol 1e-9), centre inside the triangle
  - names unique; tool_calls <= 4
- Targets: construction math; naming of auto-created centres.

#### CON-04: Perpendicular from a point

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 6, "y": 0}, {"x": 2, "y": 4}], polygon_type="triangle", name="ABC")`
- Turn 1: "Drop a perpendicular from C to AB."
  - Reference: `construct_perpendicular_from_point(point_name="C", segment_name="AB")`
- Checks:
  - exactly one new point, at (2,0), lying on AB
  - a new segment from C to that point, perpendicular to AB
- Targets: construction math; single-step undo (this one suspends archiving correctly).

#### CON-05: Parallel line and angle bisector

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 6, "y": 0}, {"x": 2, "y": 4}], polygon_type="triangle", name="ABC")`
- Turn 1: "Draw a line through C parallel to AB, and bisect the angle at A."
  - Reference: `construct_parallel_line(segment_name="AB", point_name="C")`; `construct_angle_bisector(vertex_name="A", p1_name="B", p2_name="C")`
- Checks:
  - a segment through (2,4) parallel to AB (length 6)
  - a segment from A whose direction makes equal angles with AB and AC (tol 1e-9)
- Targets: construction math.

#### CON-06: Construction asked to reuse an existing name

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 6, "y": 0}, {"x": 2, "y": 4}], polygon_type="triangle", name="ABC")`
- Turn 1: "Construct the midpoint of BC and call it A."
  - Reference: `construct_midpoint(segment_name="BC", name="A")`
- Checks:
  - A still at (0,0); a new point at (4,2) with a different name; names unique
  - invariant I4: the tool result names the point actually created
- Targets: name collisions in constructions.
- Known bugs: K3 (expected to fail until fixed; see section 6).

### 5.3 Functions, piecewise and parametric curves, tangents and normals (FN)

#### FN-01: Function with an asymptote and a shaded area (smoke)

- Turn 1: "Plot f(x) = 1/(x-1) from -5 to 5 and shade the area between f and the x-axis from x=2 to x=4."
  - Reference: `draw_function(function_string="1/(x-1)", name="f", left_bound=-5, right_bound=5)`; `create_colored_area(drawable1_name="f", left_bound=2, right_bound=4)`
- Checks:
  - function f with vertical_asymptotes [1]; f(3) = 0.5 by sampling
  - a FunctionsBoundedColoredArea for f and x_axis with bounds [2,4]
  - no tool errors (the reference passes `color: null` as a strict-schema model does; this currently fails)
- Targets: schema says color is optional but null is rejected.
- Known bugs: K10 (expected to fail until fixed; see section 6).

#### FN-02: Shifting a function moves its asymptotes

- Setup (scripted): `draw_function(function_string="1/x", name="f", left_bound=-10, right_bound=10)`
- Turn 1: "Shift f two units right and three up."
  - Reference: `translate_object(name="f", x_offset=2, y_offset=3)`
- Checks:
  - sampling: f(3) = 4, f(4) = 3.5
  - invariant I3: vertical_asymptotes == [2], horizontal asymptote 3 (currently [0] and 0)
  - bounds shifted to [-8, 12]
- Targets: cached analysis not updated by transforms.
- Known bugs: K11 (expected to fail until fixed; see section 6).

#### FN-03: Redefining an existing function name

- Setup (scripted): `draw_function(function_string="1/x", name="f", left_bound=-10, right_bound=10)`
- Turn 1: "Actually, make f(x) = x^2 instead."
  - Reference: `draw_function(function_string="x^2", name="f", left_bound=-10, right_bound=10)`
- Checks:
  - exactly one function f; f(2) = 4
  - invariant I3: no vertical_asymptotes listed (currently keeps [0])
- Targets: update-in-place path skipping re-analysis.
- Known bugs: K12 (expected to fail until fixed; see section 6).

#### FN-04: Piecewise function

- Turn 1: "Define g(x) = x^2 for x < 0 and g(x) = x for x >= 0."
  - Reference: `draw_piecewise_function(name="g", pieces=[{"expression": "x^2", "left": null, "right": 0, "left_inclusive": false, "right_inclusive": false, "undefined_at": null}, {"expression": "x", "left": 0, "right": null, "left_inclusive": true, "right_inclusive": false, "undefined_at": null}])`
- Checks:
  - one piecewise function; g(-2) = 4, g(3) = 3, g(0) = 0 (right piece is inclusive at 0)
- Targets: piecewise parsing and interval semantics.

#### FN-05: Tangent to a parametric circle

- Turn 1: "Draw the unit circle as a parametric curve x=cos(t), y=sin(t) for t from 0 to 2*pi, then the tangent at t = pi/4."
  - Reference: `draw_parametric_function(x_expression="cos(t)", y_expression="sin(t)", name="p", t_min=0, t_max=6.28318530718)`; `draw_tangent_line(curve_name="p", parameter=0.785398163397)`
- Checks:
  - a tangent segment whose midpoint is (0.7071, 0.7071) (tol 1e-6) and whose direction is perpendicular to (1,1)
- Targets: tangent math on parametric curves.

#### FN-06: Tangent at a vertical asymptote is refused

- Setup (scripted): `draw_function(function_string="tan(x)", name="t", left_bound=-4, right_bound=4)`
- Turn 1: "Draw the tangent to t at x = pi/2."
  - Reference: `draw_tangent_line(curve_name="t", parameter=1.570796326795)`
- Checks:
  - no new segment is created, or the call errors; either way the model tells the user the tangent does not exist
  - invariant I7: no coordinate with magnitude above 1e12 (currently a segment reaches y = 1.6e16)
- Targets: missing asymptote guard in tangent computation.
- Known bugs: K13 (expected to fail until fixed; see section 6).

#### FN-07: Tangent and normal to a parabola, then move the parabola

- Turn 1: "Plot y = x^2, then draw its tangent and normal lines at x = 1."
  - Reference: `draw_function(function_string="x^2", name="sq")`; `draw_tangent_line(curve_name="sq", parameter=1)`; `draw_normal_line(curve_name="sq", parameter=1)`
- Turn 2: "Shift the parabola 2 units to the right."
  - Reference: `translate_object(name="sq", x_offset=2, y_offset=0)`
- Checks:
  - after turn 1: tangent through (1,1) with slope 2; normal through (1,1) with slope -1/2
  - after turn 2: sq(3) = 1; the tangent and normal stay where they were (pinned: static like constructions)
- Targets: tangent math; static-construction semantics.

#### FN-08: Reversed bounds

- Turn 1: "Plot y = x from 5 to -5."
  - Reference: `draw_function(function_string="x", name="g", left_bound=5, right_bound=-5)`
- Checks:
  - either the call errors and the model retries with swapped bounds, or the stored bounds are [-5, 5]
  - invariant: a function reported as drawn has left_bound < right_bound
- Targets: missing validation (update_function validates, draw_function does not).
- Known bugs: K12 (expected to fail until fixed; see section 6).

#### FN-09: Removable discontinuity

- Turn 1: "Plot (x^2 - 1)/(x - 1) and mark that it has a hole at x = 1."
  - Reference: `draw_function(function_string="(x^2 - 1)/(x - 1)", name="h", undefined_at=[1])`
- Checks:
  - point_discontinuities contains 1; h(2) = 3; h(0) = 1
  - invariant I3: x = 1 is not listed as a vertical asymptote (currently it is)
- Targets: hole bookkeeping; asymptote detection from the expression text.
- Known bugs: K24 (expected to fail until fixed; see section 6).

#### FN-10: A quotient with abs() draws and its limits answer

- Turn 1: "Plot the sign function as abs(x)/x from -5 to 5."
  - Reference: `draw_function(function_string="abs(x)/x", name="s", left_bound=-5, right_bound=5)`
- Turn 2: "What is the limit of abs(x)/x as x approaches 0, and as x goes to infinity?"
  - Reference: `limit(expression="abs(x)/x", variable="x", value_to_approach="0")`; `limit(expression="abs(x)/x", variable="x", value_to_approach="inf")`
- Checks:
  - after turn 1: no tool errors (before the fix the turn never finished); s(2) = 1, s(-3) = -1; horizontal asymptotes [-1, 1]; no vertical asymptote; a point discontinuity at 0
  - after turn 2: the limit at 0 is an error saying the two-sided limit does not exist; the limit at infinity says the expression approaches 1
- Targets: nerdamer's `limit()` looping forever on quotients with `abs()`.
- Known bugs: K28, fixed on `claude/fix-batch-7`; the checks are regression guards.

#### FN-11: Asymptote lists without false or duplicate entries

- Turn 1: "Plot x/tan(x) from -2 to 2."
  - Reference: `draw_function(function_string="x/tan(x)", name="c", left_bound=-2, right_bound=2)`
- Turn 2: "Plot 1/(1 - cos(x)) from -1 to 1."
  - Reference: `draw_function(function_string="1/(1 - cos(x))", name="r", left_bound=-1, right_bound=1)`
- Turn 3: "Plot 1/x from 2 to 5."
  - Reference: `draw_function(function_string="1/x", name="q", left_bound=2, right_bound=5)`
- Checks:
  - c lists no vertical asymptote (before the fix ±π/2 and dozens of asymptotes outside the bounds; I3 flags ±π/2 as not blowing up) and three point discontinuities (-π/2, 0, π/2)
  - r lists the asymptote 0 once (before the fix 0, 1.4e-9, 6.2e-9 and 9.7e-9, and asymptotes outside the bounds)
  - q lists no asymptote (before the fix 0, outside [2, 5]); q(2) = 0.5, q(4) = 0.25
- Targets: asymptote detection from the expression text.
- Known bugs: K32, fixed on `claude/fix-batch-7`; the checks are regression guards.

### 5.4 Coloured areas and regions (AR)

#### AR-01: Shaded area follows its function

- Setup (scripted): `draw_function(function_string="x^2", name="f", left_bound=-3, right_bound=3)`; `create_colored_area(drawable1_name="f", left_bound=0, right_bound=2, color="orange", opacity=0.4)`
- Turn 1: "Move f three units to the right, shading included."
  - Reference: `translate_object(name="f", x_offset=3, y_offset=0)`
- Checks:
  - f(3) = 0; the area's bounds are [3, 5] (currently still [0, 2])
- Targets: dependents of functions not updated by transforms.
- Known bugs: K14 (expected to fail until fixed; see section 6).

#### AR-02: Area between two segments has no side effects

- Setup (scripted): `create_segment(x1=0, y1=0, x2=4, y2=0)`; `create_segment(x1=1, y1=2, x2=3, y2=2)`
- Turn 1: "Shade the region between segments AB and CD."
  - Reference: `create_colored_area(drawable1_name="AB", drawable2_name="CD", color="pink", opacity=0.3)`
- Checks:
  - one SegmentsBoundedColoredArea
  - unchanged_except: no new points or segments (currently E, F and five split segments appear)
- Targets: hidden geometry created by an area tool.
- Known bugs: K15 (expected to fail until fixed; see section 6).

#### AR-03: Fill a square and report its area

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 2, "y": 0}, {"x": 2, "y": 2}, {"x": 0, "y": 2}], polygon_type="quadrilateral", subtype="square", name="ABCD")`
- Turn 1: "Fill the square pink and tell me its area."
  - Reference: `create_region_colored_area(rectangle_name="ABCD", color="pink")`; `calculate_area(expression="ABCD")`
- Turn 2: "Move the square 5 units right."
  - Reference: `translate_object(name="ABCD", x_offset=5, y_offset=0)`
- Checks:
  - turn 1: a ClosedShapeColoredArea over ABCD; calculate_area result 4 (tol 1e-9); answer mentions 4
  - turn 2: the fill's geometry_snapshot moved by (5,0) (works today; pins it)
- Targets: region areas bound to polygons.

#### AR-04: Circle minus triangle

- Setup (scripted): `create_circle(center_x=0, center_y=0, radius=5)`; `create_polygon(vertices=[{"x": -1, "y": -1}, {"x": 2, "y": -1}, {"x": 0, "y": 2}], polygon_type="triangle")`
- Turn 1: "Shade the part of the circle outside the triangle and tell me its area."
  - Reference: `create_region_colored_area(expression="A(5) - BCD")`; `calculate_area(expression="A(5) - BCD")`
- Checks:
  - calculate_area result 25*pi - 4.5 = 74.0398 (tol 1e-3)
  - one region area; no tool errors (the reference passes `color: null`; this currently fails); answer mentions about 74.04
- Targets: boolean region expressions; names in expressions.
- Known bugs: K10 (expected to fail until fixed; see section 6).

### 5.5 Transforms (TR)

#### TR-01: Translate one of two triangles that share an edge (smoke)

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 4}], polygon_type="triangle", name="ABC")`; `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 4}, {"x": 0, "y": 4}], polygon_type="triangle", name="ACD")`
- Turn 1: "Move triangle ABC up by 2."
  - Reference: `translate_object(name="ABC", x_offset=0, y_offset=2)`
- Checks:
  - A (0,2), B (4,2), C (4,6): each shared point moved exactly once
  - D unchanged at (0,4); triangle ACD now uses the moved A and C (pinned: shared points)
  - invariant I3: triangle `types` recomputed for ACD
- Targets: double-moving shared vertices; stale derived data.
- Known bugs: K16, fixed on `claude/fix-stale-geometry`; the checks are regression guards.

#### TR-02: Rotate about a point and back

- Setup (scripted): `create_polygon(vertices=[{"x": 1, "y": 0}, {"x": 3, "y": 0}, {"x": 1, "y": 2}], polygon_type="triangle", name="ABC")`
- Turn 1: "Rotate triangle ABC by 90 degrees around the origin."
  - Reference: `rotate_object(name="ABC", angle=90, center_x=0, center_y=0)`
- Turn 2: "Rotate it back."
  - Reference: `rotate_object(name="ABC", angle=-90, center_x=0, center_y=0)`
- Checks:
  - after turn 1: A (0,1), B (0,3), C (-2,1) (tol 1e-9)
  - after turn 2: state equals the setup snapshot (tol 1e-9)
- Targets: rotation math; round-trip drift.

#### TR-03: Reflect across a named segment

- Setup (scripted): `create_polygon(vertices=[{"x": 1, "y": 0}, {"x": 3, "y": 0}, {"x": 1, "y": 2}], polygon_type="triangle", name="ABC")`; `create_segment(x1=-5, y1=-5, x2=5, y2=5)`
- Turn 1: "Reflect triangle ABC across segment DE."
  - Reference: `reflect_object(name="ABC", axis="segment", segment_name="DE")`
- Checks:
  - A (0,1), B (0,3), C (2,1) (mirror in y = x)
  - segment DE unchanged
- Targets: reflection math; axis resolution.

#### TR-04: Non-uniform scaling of a circle is refused cleanly

- Setup (scripted): `create_circle(center_x=0, center_y=0, radius=2)`
- Turn 1: "Stretch the circle horizontally by a factor of 2."
  - Reference: `scale_object(name="A(2)", sx=2, sy=1, cx=0, cy=0)`
- Checks:
  - the circle is unchanged; the tool result is an error; no undo entry was added by the failed call (I5)
  - model check: the answer suggests an ellipse, or the model draws create_ellipse(radius_x=4, radius_y=2)
- Targets: failed calls leaving undo entries; error wording.
- Known bugs: K1 (expected to fail until fixed; see section 6).

#### TR-05: Shear updates angle markers and triangle types

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 2, "y": 3.464101615138}], polygon_type="triangle", name="ABC", subtype="equilateral")`; `create_angle(vx=4, vy=0, p1x=0, p1y=0, p2x=2, p2y=3.464101615138)`
- Turn 1: "Shear triangle ABC horizontally with factor 1 about the origin."
  - Reference: `shear_object(name="ABC", axis="horizontal", factor=1, cx=0, cy=0)`
- Checks:
  - C moved to (5.4641, 3.4641); A and B unchanged
  - invariant I3: the triangle's types no longer include `equilateral` (before the K16 fix, still listed)
  - invariant I3 (inspection): the angle's degrees equal the new geometric angle at B (before the K17 fix, still 60)
- Targets: stale cached classifications and angle values.
- Known bugs: K16 and K17, fixed on `claude/fix-stale-geometry`; the checks are regression guards.

#### TR-06: Transforming an unsupported type fails without side effects

- Setup (scripted): `create_circle_arc(point1_x=20, point1_y=0, point2_x=10, point2_y=10, center_x=10, center_y=0, radius=10, use_major_arc=false)`
- Turn 1: "Move the arc up by one unit."
  - Reference: `translate_object(name="ArcMin_AB", x_offset=0, y_offset=1)`
- Checks:
  - either the arc (and its endpoints) moved by (0,1), or the call errors and nothing changed
  - invariant I5: a failed call adds no undo entry (currently two)
- Targets: archive before validation.
- Known bugs: K1 (expected to fail until fixed; see section 6).

### 5.6 Graph theory (GR)

#### GR-01: Weighted graph and shortest path (smoke)

- Turn 1: "Create an undirected weighted graph G1 with vertices A(0,0), B(4,0), C(4,4), D(0,4) and edges A-B (1), B-C (2), A-C (4), C-D (1). What is the shortest path from A to D?"
  - Reference: `generate_graph(name="G1", graph_type="graph", directed=false, vertices=[{"name": "A", "x": 0, "y": 0, "color": null, "label": null}, {"name": "B", "x": 4, "y": 0, "color": null, "label": null}, {"name": "C", "x": 4, "y": 4, "color": null, "label": null}, {"name": "D", "x": 0, "y": 4, "color": null, "label": null}], edges=[{"source": 0, "target": 1, "weight": 1, "name": null, "color": null, "directed": null}, {"source": 1, "target": 2, "weight": 2, "name": null, "color": null, "directed": null}, {"source": 0, "target": 2, "weight": 4, "name": null, "color": null, "directed": null}, {"source": 2, "target": 3, "weight": 1, "name": null, "color": null, "directed": null}])`; `analyze_graph(graph_name="G1", operation="shortest_path", params={"start": "A", "goal": "D", "root": null, "a": null, "b": null, "new_root": null, "x": null, "y": null})`
- Checks:
  - one UndirectedGraph with 4 vertices and 4 segments whose label texts are the weights
  - tool result: path [A,B,C,D], cost 4; the answer mentions cost 4
- Targets: graph construction and analysis agree with the canvas.

#### GR-02: DAG and topological order

- Turn 1: "Make a DAG D1 with vertices P, Q, R and edges P->Q, P->R, R->Q, then topologically sort it."
  - Reference: `generate_graph(name="D1", graph_type="dag", directed=true, vertices=[{"name": "P", "x": 10, "y": 0, "color": null, "label": null}, {"name": "Q", "x": 14, "y": 0, "color": null, "label": null}, {"name": "R", "x": 12, "y": 3, "color": null, "label": null}], edges=[{"source": 0, "target": 1, "weight": null, "name": null, "color": null, "directed": null}, {"source": 0, "target": 2, "weight": null, "name": null, "color": null, "directed": null}, {"source": 2, "target": 1, "weight": null, "name": null, "color": null, "directed": null}])`; `analyze_graph(graph_name="D1", operation="topological_sort")`
- Checks:
  - a DirectedGraph with 3 vectors; the order in the result respects every edge (P before R before Q)
- Targets: directed graph construction.

#### GR-03: Undirected graph from an adjacency matrix

- Turn 1: "Build an undirected graph M from the adjacency matrix [[0,1,0],[1,0,2],[0,2,0]] placed in the box x 20..26, y 0..6."
  - Reference: `generate_graph(name="M", graph_type="graph", directed=false, layout="circular", placement_box={"x": 20, "y": 0, "width": 6, "height": 6}, vertices=[{"name": null, "x": null, "y": null, "color": null, "label": null}, {"name": null, "x": null, "y": null, "color": null, "label": null}, {"name": null, "x": null, "y": null, "color": null, "label": null}], edges=[], adjacency_matrix=[[0, 1, 0], [1, 0, 2], [0, 2, 0]])`
- Turn 2: "What is the shortest path between the two end vertices?"
  - Reference: `analyze_graph(graph_name="M", operation="shortest_path", params={"start": "A", "goal": "C", "root": null, "a": null, "b": null, "new_root": null, "x": null, "y": null})`
- Checks:
  - 3 vertices inside the placement box; 2 undirected edges (segments), no vectors (currently 4 vectors, segments [])
  - shortest path has cost 3
- Targets: adjacency-matrix edges forced to directed.
- Known bugs: K18 (expected to fail until fixed; see section 6).

#### GR-04: Deleting a graph keeps pre-existing points

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}], polygon_type="triangle")`
- Turn 1: "Create graph G with a vertex X at (0,0) and a vertex Y at (8,8), joined by an edge of weight 1."
  - Reference: `generate_graph(name="G", graph_type="graph", directed=false, vertices=[{"name": "X", "x": 0, "y": 0, "color": null, "label": null}, {"name": "Y", "x": 8, "y": 8, "color": null, "label": null}], edges=[{"source": 0, "target": 1, "weight": 1, "name": null, "color": null, "directed": null}])`
- Turn 2: "Delete graph G."
  - Reference: `delete_graph(name="G")`
- Checks:
  - after turn 1: the vertex at (0,0) is the existing triangle vertex; the result lists its real name
  - after turn 2: the triangle and its three sides still exist; Y and the edge are gone
- Targets: graph deletion cascading into unrelated objects.
- Known bugs: K18 (expected to fail until fixed; see section 6).

#### GR-05: Tree levels

- Turn 1: "Draw a tree T rooted at R with children A and B; A has children C and D. Then give me the levels."
  - Reference: `generate_graph(name="T", graph_type="tree", directed=false, vertices=[{"name": "R", "x": 0, "y": 10, "color": null, "label": null}, {"name": "A", "x": -3, "y": 7, "color": null, "label": null}, {"name": "B", "x": 3, "y": 7, "color": null, "label": null}, {"name": "C", "x": -4, "y": 4, "color": null, "label": null}, {"name": "D", "x": -2, "y": 4, "color": null, "label": null}], edges=[{"source": 0, "target": 1, "weight": null, "name": null, "color": null, "directed": null}, {"source": 0, "target": 2, "weight": null, "name": null, "color": null, "directed": null}, {"source": 1, "target": 3, "weight": null, "name": null, "color": null, "directed": null}, {"source": 1, "target": 4, "weight": null, "name": null, "color": null, "directed": null}], root="R")`; `analyze_graph(graph_name="T", operation="levels", params={"start": null, "goal": null, "root": "R", "a": null, "b": null, "new_root": null, "x": null, "y": null})`
- Checks:
  - a tree with 5 vertices and 4 edges; levels: R at 0, A and B at 1, C and D at 2
- Targets: tree construction and analysis.

#### GR-06: Minimum spanning tree weight

- Setup (scripted): `generate_graph(name="G1", graph_type="graph", directed=false, vertices=[{"name": "A", "x": 0, "y": 0, "color": null, "label": null}, {"name": "B", "x": 4, "y": 0, "color": null, "label": null}, {"name": "C", "x": 4, "y": 4, "color": null, "label": null}, {"name": "D", "x": 0, "y": 4, "color": null, "label": null}], edges=[{"source": 0, "target": 1, "weight": 1, "name": null, "color": null, "directed": null}, {"source": 1, "target": 2, "weight": 2, "name": null, "color": null, "directed": null}, {"source": 0, "target": 2, "weight": 4, "name": null, "color": null, "directed": null}, {"source": 2, "target": 3, "weight": 1, "name": null, "color": null, "directed": null}])`
- Turn 1: "Find the minimum spanning tree of G1 and its total weight."
  - Reference: `analyze_graph(graph_name="G1", operation="mst")`
- Checks:
  - result edges AB, BC, CD; the answer's total weight is 4
- Targets: analysis over label weights.

### 5.7 Statistics, plots and regression (ST)

#### ST-01: Normal distribution with shading, then remove it (smoke)

- Turn 1: "Plot a normal distribution with mean 0 and sigma 1 and shade from -1 to 1."
  - Reference: `plot_distribution(name="nd", representation="continuous", distribution_type="normal", distribution_params={"mean": 0, "sigma": 1}, shade_bounds={"left_bound": -1, "right_bound": 1})`
- Turn 2: "Remove the plot."
  - Reference: `delete_plot(name="nd")`
- Scripted step: `undo()`
- Checks:
  - turn 1: a ContinuousPlot, its pdf function with value 0.39894 at 0 (tol 1e-4), a fill area with bounds [-1,1]
  - turn 2: plot, function and fill all gone
  - after one scripted undo: all three are back and consistent (fill_area_name resolves)
- Targets: composite plot bookkeeping; atomic undo.
- Known bugs: K1 (expected to fail until fixed; see section 6).

#### ST-02: Bar chart

- Turn 1: "Plot a bar chart of sales: Mon 3, Tue 5, Wed 2, starting at x = 30."
  - Reference: `plot_bars(name="sales", values=[3, 5, 2], labels_below=["Mon", "Tue", "Wed"], x_start=30, y_base=0)`
- Checks:
  - a BarsPlots entry with values [3,5,2] and those labels; derived Bars are pruned from the state
- Targets: plot composites.

#### ST-03: Linear regression over an existing point

- Setup (scripted): `create_point(x=1, y=3, name="A")`
- Turn 1: "Fit a line to the data (1,3), (2,5), (3,7), (4,9) and show the points."
  - Reference: `fit_regression(name="fit1", x_data=[1, 2, 3, 4], y_data=[3, 5, 7, 9], model_type="linear", show_points=true)`
- Checks:
  - tool result: m = 2, b = 1, r_squared = 1; function fit1 with fit1(0) = 1
  - a point at each data coordinate; point_names in the result match the points at those coordinates
  - A is reused for (1,3) and the result says so: `reused_point_names` is [A] and `created_point_names` is [B, C, D]
- Targets: regression math; reuse of existing points.
- Known bugs: K27, formerly filed under K2 (fixed; see section 6).

#### ST-04: Descriptive statistics answer

- Turn 1: "Give me the mean, median and standard deviation of 10, 20, 30, 40, 50."
  - Reference: `compute_descriptive_statistics(data=[10, 20, 30, 40, 50])`
- Checks:
  - tool result: mean 30, median 30, standard_deviation 14.1421 (population); canvas unchanged
  - answer mentions 30, 30 and 14.14 (model quality)
- Targets: non-canvas tool; model reporting.

#### ST-05: Moving a bar chart as a whole is refused, not faked

- Setup (scripted): `plot_bars(name="sales", values=[3, 5, 2], labels_below=["a", "b", "c"])`
- Turn 1: "Move the sales chart 4 units to the right."
  - Reference: `translate_object(name="sales", x_offset=4, y_offset=0)`
- Turn 2: "Rotate the sales chart by 90 degrees."
  - Reference: `rotate_object(name="sales", angle=90)`
- Checks: both calls are errors (the translate error says to plot again with `x_start`), and the canvas equals the setup snapshot (before the fix both answered "Call successful!" and nothing moved)
- Targets: transforms of composite plot records.
- Known bugs: K31, fixed on `claude/fix-batch-7`; the checks are regression guards.

### 5.8 Math tools feeding the canvas (MC)

#### MC-01: Solve, mark the roots, plot the parabola (smoke)

- Turn 1: "Solve x^2 - 5x + 6 = 0, mark the roots on the x-axis and plot the parabola."
  - Reference: `solve(equation="x^2 - 5*x + 6", variable="x")`; `create_point(x=2, y=0)`; `create_point(x=3, y=0)`; `draw_function(function_string="x^2 - 5*x + 6", name="p")`
- Checks:
  - points at (2,0) and (3,0); function p passes through both (relation point_on_function)
  - solve result contains 2 and 3; tool_calls <= 6
- Targets: chaining a CAS result into drawing.

#### MC-02: System of equations and the intersection

- Turn 1: "Solve x + y = 3 and x - y = 1, draw both lines and mark the intersection."
  - Reference: `solve_system_of_equations(equations=["x + y = 3", "x - y = 1"])`; `draw_function(function_string="3 - x", name="l1")`; `draw_function(function_string="x - 1", name="l2")`; `create_point(x=2, y=1, name="I")`
- Checks:
  - a point at (2,1) lying on both functions
- Targets: chaining; function naming.

#### MC-03: Derivative and tangent agree

- Turn 1: "What is the derivative of x^3 at x = 1? Draw the curve and its tangent there."
  - Reference: `derive(expression="x^3", variable="x")`; `draw_function(function_string="x^3", name="cub")`; `draw_tangent_line(curve_name="cub", parameter=1)`
- Checks:
  - derive result equivalent to 3*x^2; tangent through (1,1) with slope 3; answer mentions 3
- Targets: CAS and numeric derivative consistency.

#### MC-04: Polar grid and a point given in polar form

- Turn 1: "Switch to the polar grid and put a point at r = 2, theta = 90 degrees."
  - Reference: `set_coordinate_system(mode="polar")`; `convert_coordinates(coord1=2, coord2=1.570796326795, from_system="polar", to_system="cartesian")`; `create_point(x=0, y=2)`
- Checks:
  - coordinate_system.mode == polar
  - a point at (0,2) within 1e-9 (the conversion result has x = 1.2e-16; a model that copies it must still land within tolerance)
  - model check: the model converted degrees to radians
- Targets: polar mode; units.

### 5.9 Canvas operations: view, coordinate systems, undo and redo (CV)

#### CV-01: Undo a zoom

- Turn 1: "Zoom to x between -2 and 2 around the origin."
  - Reference: `zoom(center_x=0, center_y=0, range_val=2, range_axis="x")`
- Turn 2: "Undo that."
  - Reference: `undo()`
- Checks:
  - after turn 1: left_bound -2, right_bound 2
  - after turn 2: the view equals the view before turn 1 (currently unchanged: undo snapshots drawables only)
- Targets: view state outside the undo snapshot.
- Known bugs: K4 (expected to fail until fixed; see section 6).

#### CV-02: Undo a multi-object construction in one step (smoke)

- Turn 1: "Draw triangle ABC with vertices (0,0), (6,0), (2,4) and its circumcircle."
  - Reference: `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 6, "y": 0}, {"x": 2, "y": 4}], polygon_type="triangle", name="ABC")`; `construct_circumcircle(triangle_name="ABC")`
- Turn 2: "Undo that."
  - Reference: `undo()`
- Scripted step: `redo()`
- Checks:
  - after turn 2: state equals the empty canvas (one batch = one undo step)
  - after the scripted redo: state equals the state after turn 1
- Targets: undo stack out of sync with tool batches.
- Known bugs: K1 (expected to fail until fixed; see section 6).

#### CV-03: Clear, then undo

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}], polygon_type="triangle", name="ABC")`; `create_circle(center_x=10, center_y=10, radius=2)`; `zoom(center_x=0, center_y=0, range_val=10, range_axis="x")`
- Turn 1: "Clear the canvas."
  - Reference: `clear_canvas()`
- Turn 2: "Oops, bring it all back."
  - Reference: `undo()`
- Checks:
  - after turn 1: no drawables
  - after turn 2: every object back (state equals the setup snapshot); the view is the zoomed one
- Targets: clear resets the view outside undo.
- Known bugs: K4 (expected to fail until fixed; see section 6).

#### CV-04: Undo when there is nothing to undo

- Turn 1: "Undo."
  - Reference: `undo()`
- Checks:
  - invariant I4: the result says nothing was undone (currently `Call successful!`)
- Targets: untruthful tool results.
- Known bugs: K2 (expected to fail until fixed; see section 6).

#### CV-05: Clearing after a zoom resets the polar grid too

Added by the first replay run (not part of the original 72).

- Turn 1: "Zoom to x between -2 and 2 around the origin."
  - Reference: `zoom(center_x=0, center_y=0, range_val=2, range_axis="x")`
- Turn 2: "Clear the canvas and switch to the polar grid."
  - Reference: `clear_canvas()`; `set_coordinate_system(mode="polar")`
- Checks:
  - after turn 2: the view is the start-up view; the mode is polar
  - the polar grid's ring spacing equals its start-up value (currently 0.2 instead of 50)
- Targets: grid state that `Canvas.reset` leaves behind.
- Known bugs: K25 (expected to fail until fixed; see section 6).

#### CV-06: Repeating a change that is already in effect is a no-op

Added after the K1 and K2 fixes merged (not part of the original 72).

- Setup (scripted): `create_circle(center_x=0, center_y=0, radius=2, color="red")`; `zoom(center_x=0, center_y=0, range_val=2, range_axis="x")`
- Turn 1: "Make the circle red."
  - Reference: `update_circle(name="A(2)", new_color="red")`
- Turn 2: "Zoom to x between -2 and 2 around the origin."
  - Reference: `zoom(center_x=0, center_y=0, range_val=2, range_axis="x")`
- Checks:
  - after each turn: state and inspection view (colour, view) equal the setup snapshot
  - the undo depth is unchanged; the result is not a bare "Call successful!"
  - invariants I4 and I5 (waived for K26 until its fix)
- Targets: success claims and undo entries for no-op changes.
- Known bugs: K26 (expected to fail until fixed; see section 6).

### 5.10 Workspaces (WS)

#### WS-01: Round-trip of every object type (smoke)

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}], polygon_type="triangle", name="ABC", color="green")`; `create_circle(center_x=10, center_y=10, radius=2, color="purple")`; `create_ellipse(center_x=-8, center_y=5, radius_x=3, radius_y=1.5, rotation_angle=30, color="orange")`; `create_vector(origin_x=-5, origin_y=-5, tip_x=-2, tip_y=-1, color="red")`; `create_label(x=2, y=8, text="Hello", color="blue", font_size=18, rotation_degrees=15)`; `draw_function(function_string="1/(x-1)", name="f", left_bound=-5, right_bound=5, color="red", undefined_at=[1])`; `draw_parametric_function(x_expression="cos(t)", y_expression="sin(t)", name="pc", t_min=0, t_max=6.28318530718, color="brown")`; `create_angle(vx=0, vy=0, p1x=4, p1y=0, p2x=0, p2y=3, angle_name="alpha", is_reflex=true)`; `create_circle_arc(point1_x=12, point1_y=10, point2_x=10, point2_y=12, circle_name="D(2)", use_major_arc=true, color="navy")`; `create_colored_area(drawable1_name="f", left_bound=2, right_bound=4, color="yellow", opacity=0.5)`; `generate_graph(name="G", graph_type="graph", directed=false, vertices=[{"name": "U", "x": 20, "y": 0, "color": null, "label": null}, {"name": "V", "x": 24, "y": 0, "color": null, "label": null}, {"name": "W", "x": 22, "y": 3, "color": null, "label": null}], edges=[{"source": 0, "target": 1, "weight": 2, "name": null, "color": null, "directed": null}, {"source": 1, "target": 2, "weight": 5, "name": null, "color": null, "directed": null}])`; `plot_bars(name="sales", values=[3, 5, 2], labels_below=["Mon", "Tue", "Wed"], x_start=30, y_base=0)`; `set_coordinate_system(mode="polar")`; `set_grid_visible(visible=false)`
- Turn 1: "Save this workspace as scn_roundtrip."
  - Reference: `save_workspace(name="scn_roundtrip")`
- Scripted step: `clear_canvas()`; `set_grid_visible(visible=true)` (still in polar mode, so the load has to hide the polar grid again); `set_coordinate_system(mode="cartesian")`
- Turn 2: "Load the workspace scn_roundtrip."
  - Reference: `load_workspace(name="scn_roundtrip")`
- Scripted step: `delete_workspace(name="scn_roundtrip")`
- Checks:
  - after the load: state equals the snapshot taken before the save (get_canvas_state, the angle included)
  - inspection equals too: colors, angle name `alpha`, grid visibility
- Targets: persistence gaps invisible to get_canvas_state.
- Known bugs: K5 (fixed; see section 6).

#### WS-02: Undo right after a load

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}], polygon_type="triangle", name="ABC")`; `save_workspace(name="scn_undo_load")`; `clear_canvas()`; `create_point(x=9, y=9, name="Z")`
- Turn 1: "Load scn_undo_load."
  - Reference: `load_workspace(name="scn_undo_load")`
- Turn 2: "Undo the load."
  - Reference: `undo()`
- Scripted step: `delete_workspace(name="scn_undo_load")`
- Checks:
  - after turn 2: state equals the pre-load state (only Z)
- Targets: load pushing one undo entry per restored object.
- Known bugs: K1 (expected to fail until fixed; see section 6).

#### WS-03: Workspace name the user types with spaces

- Setup (scripted): `create_point(x=1, y=1, name="A")`
- Turn 1: "Save my work as 'my project v2'."
  - Reference: `save_workspace(name="my_project_v2")`
- Turn 2: "Which workspaces do I have?"
  - Reference: `list_workspaces()`
- Scripted step: `delete_workspace(name="my_project_v2")`
- Checks:
  - save_workspace was called with a valid name (matches ^[\w-]+$) and succeeded
  - list result contains that name; the answer tells the user the adjusted name
- Targets: name validation; model adaptation.

#### WS-04: A load name with URL characters loads nothing

- Setup (scripted): `create_point(x=1, y=1, name="A")`; `save_workspace(name="scn_url")`
- Scripted step: `clear_canvas()`
- Turn 1: "Load the workspace called scn_url&x=1."
  - Reference: `load_workspace(name="scn_url&x=1")`
- Turn 2: "Then load scn_url#2."
  - Reference: `load_workspace(name="scn_url#2")`
- Scripted steps: `load_workspace(name="scn_url")`, then `delete_workspace(name="scn_url")`
- Checks:
  - after each turn: the load is an error and the canvas stays empty (before the fix both loaded `scn_url`: the name went into the URL unencoded, so `&x=1` became a second parameter and `#2` a fragment)
  - after the scripted load: the point at (1,1) is back, so the workspace itself loads
- Targets: the load URL.
- Known bugs: K23, fixed on `claude/fix-batch-7`; the checks are regression guards.

### 5.11 Naming, editing and deleting (NM)

#### NM-01: Rename and recolour a free point

- Setup (scripted): `create_point(x=1, y=1, name="P")`
- Turn 1: "Rename P to Q and make it red."
  - Reference: `update_point(point_name="P", new_name="Q", new_color="red")`
- Checks:
  - one point, named Q, at (1,1), color red (inspection)
- Targets: edit policy on solitary points.

#### NM-02: Renaming a triangle vertex is refused

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}], polygon_type="triangle", name="ABC")`
- Turn 1: "Rename vertex B to P."
  - Reference: `update_point(point_name="B", new_name="P")`
- Checks:
  - canvas unchanged (state equals setup); exactly one tool error, and the answer explains why
  - invariant I5: the refused call added no undo entry
- Targets: refusal path; model explanation.
- Known bugs: K1 (expected to fail until fixed; see section 6).

#### NM-03: Delete a point used by segments, then undo (smoke)

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}], polygon_type="triangle", name="ABC")`; `create_segment(x1=4, y1=0, x2=6, y2=2)`
- Turn 1: "Delete point B."
  - Reference: `delete_point(x=4, y=0)`
- Scripted step: `undo()`
- Checks:
  - after the turn: no point at (4,0); segments AB, BC and the extra one from B are gone; triangle gone; A, C, CA and the point at (6,2) remain
  - after one scripted undo: state equals the setup snapshot (currently the triangle does not come back)
- Targets: cascade delete plus undo consistency.
- Known bugs: K1 (expected to fail until fixed; see section 6).

#### NM-04: Deleting something that does not exist

- Setup (scripted): `create_circle(center_x=0, center_y=0, radius=2)`
- Turn 1: "Delete the circle Z(9)."
  - Reference: `delete_circle(name="Z(9)")`
- Checks:
  - canvas unchanged; invariant I4: the result is not a success message
  - model check: the answer says there is no such circle
- Targets: untruthful results for no-op deletes.
- Known bugs: K2 (expected to fail until fixed; see section 6).

#### NM-05: Name hint after undo

- Turn 1: "Create point K at (1,1)."
  - Reference: `create_point(x=1, y=1, name="K")`
- Turn 2: "Undo."
  - Reference: `undo()`
- Turn 3: "Create point K at (3,3)."
  - Reference: `create_point(x=3, y=3, name="K")`
- Checks:
  - after turn 3: a point named K at (3,3) (currently named A)
- Targets: name generator state not rewound.
- Known bugs: K3 (expected to fail until fixed; see section 6).

#### NM-06: Deleting one of two triangles that share an edge

- Setup (scripted): `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 4}], polygon_type="triangle", name="ABC")`; `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 4}, {"x": 0, "y": 4}], polygon_type="triangle", name="ACD")`
- Turn 1: "Delete triangle ABC."
  - Reference: `delete_polygon(polygon_type="triangle", name="ABC", vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 4}])`
- Checks:
  - triangle ACD and its three sides remain (currently CA and ACD are deleted)
  - sides AB and BC are gone
- Targets: cascade deletes through shared edges.
- Known bugs: K6 (expected to fail until fixed; see section 6).

#### NM-07: Label edits and a move that change nothing say so

- Setup (scripted): `create_segment(x1=0, y1=0, x2=4, y2=0, name="AB", label_text="base", label_visible=false)`; `create_label(x=1, y=3, text="note", name="L1")`
- Turn 1: "Hide the label of AB." Reference: `update_segment(name="AB", new_label_visible=false)`
- Turn 2: "Set AB's label text to 'base'." Reference: `update_segment(name="AB", new_label_text="base")`
- Turn 3: "Change the text of L1 to 'note'." Reference: `update_label(name="L1", new_text="note")`
- Turn 4: "Move AB by (0, 0)." Reference: `translate_object(name="AB", x_offset=0, y_offset=0)`
- Turn 5: "Show AB's label." Reference: `update_segment(name="AB", new_label_visible=true)`
- Checks:
  - turns 1 to 4: the result says nothing changed (before the fix "Call successful!", which I4 flags); after turn 4 the canvas equals the setup snapshot
  - turn 5: no tool errors and the label is visible
- Targets: the K26 no-op checks for label fields and zero translations.
- Known bugs: K31, fixed on `claude/fix-batch-7`; the checks are regression guards.

### 5.12 Multi-turn follow-ups (MT)

#### MT-01: Build, move, measure, undo (smoke)

- Turn 1: "Draw a triangle with vertices A(0,0), B(4,0) and C(0,3)."
  - Reference: `create_polygon(vertices=[{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 0, "y": 3}], polygon_type="triangle", name="ABC")`
- Turn 2: "Now move A to (1,1)."
  - Reference: `translate_object(name="A", x_offset=1, y_offset=1)`
- Turn 3: "What is the area of the triangle now?"
  - Reference: `calculate_area(expression="ABC")`
- Turn 4: "Undo the move."
  - Reference: `undo()`
- Checks:
  - turn 2: A at (1,1), B and C unchanged
  - turn 3: calculate_area result 2.5 (|3*2 - (-1)*(-1)|/2 with A at (1,1)); answer mentions 2.5
  - turn 4: A back at (0,0) (the calculate_area turn added no undo entry)
- Targets: multi-turn reference resolution; undo accounting across read-only turns.

#### MT-02: Function, derivative, shading, deletion

- Turn 1: "Plot sin(x)."
  - Reference: `draw_function(function_string="sin(x)", name="f")`
- Turn 2: "Add its derivative in red."
  - Reference: `draw_function(function_string="cos(x)", name="g", color="red")`
- Turn 3: "Shade between them from 0 to pi."
  - Reference: `create_colored_area(drawable1_name="f", drawable2_name="g", left_bound=0, right_bound=3.14159265359, color="lightgreen", opacity=0.3)`
- Turn 4: "Delete the derivative."
  - Reference: `delete_function(name="g")`
- Checks:
  - turn 3: area between f and g over [0, pi]
  - turn 4: g gone, and the area that referenced it gone too (no dangling reference, I2); f remains
- Targets: dependent cleanup on delete.

#### MT-03: Graph edit changes the answer

- Setup (scripted): `generate_graph(name="G1", graph_type="graph", directed=false, vertices=[{"name": "A", "x": 0, "y": 0, "color": null, "label": null}, {"name": "B", "x": 4, "y": 0, "color": null, "label": null}, {"name": "C", "x": 4, "y": 4, "color": null, "label": null}, {"name": "D", "x": 0, "y": 4, "color": null, "label": null}], edges=[{"source": 0, "target": 1, "weight": 1, "name": null, "color": null, "directed": null}, {"source": 1, "target": 2, "weight": 2, "name": null, "color": null, "directed": null}, {"source": 0, "target": 2, "weight": 4, "name": null, "color": null, "directed": null}, {"source": 2, "target": 3, "weight": 1, "name": null, "color": null, "directed": null}])`
- Turn 1: "Shortest path from A to D in G1?"
  - Reference: `analyze_graph(graph_name="G1", operation="shortest_path", params={"start": "A", "goal": "D", "root": null, "a": null, "b": null, "new_root": null, "x": null, "y": null})`
- Turn 2: "Remove the edge between B and C and ask again."
  - Reference: `delete_segment(x1=4, y1=0, x2=4, y2=4)`; `analyze_graph(graph_name="G1", operation="shortest_path", params={"start": "A", "goal": "D", "root": null, "a": null, "b": null, "new_root": null, "x": null, "y": null})`
- Checks:
  - turn 1: cost 4 via B, C; turn 2: segment BC gone, graph G1 still exists, path A-C-D with cost 5
- Targets: graph and canvas staying in sync after a segment delete.

#### MT-04: Pronouns across turns

- Turn 1: "Draw a circle at the origin with radius 2."
  - Reference: `create_circle(center_x=0, center_y=0, radius=2)`
- Turn 2: "Make it red."
  - Reference: `update_circle(name="A(2)", new_color="red")`
- Turn 3: "Move it to (3,3)."
  - Reference: `update_circle(name="A(2)", new_center_x=3, new_center_y=3)`
- Turn 4: "And double its size."
  - Reference: `scale_object(name="A(2)", sx=2, sy=2, cx=3, cy=3)`
- Checks:
  - final: one circle, centre (3,3), radius 4, color red (inspection); name `<centre>(4)`
  - no tool errors; tool_calls <= 6 over the four turns
- Targets: multi-turn reference resolution; renamed circles after scaling.


## 6. Known bugs found while designing

Evidence:

- **(observed)**: reproduced in the running app on 2026-09-26 during the prototype replay, or during exploration with the same tool path.
- **(code)**: verified by reading the code only.

All paths are relative to `static/client/` unless stated otherwise.

| Id | Bug | Where | Evidence | Scenarios |
|---|---|---|---|---|
| K1 | **Fixed in vl3c/MatHud#73; its marks are removed.** **A tool batch is not one undo step.** `ResultProcessor` archives once per batch and says it suspends archiving, but it never does. Every manager then archives again, so do nested creates, and so do calls that fail or change nothing. One `undo` lands on half-finished states. | `result_processor.py:83-86` and `:131-134` (comment versus code); `managers/point_manager.py:149` (archives before the "already exists" return at `:152-154`); `managers/segment_manager.py:178`; `managers/transformations_manager.py:189` (archives before a translate that then fails); `workspace_manager.py:1203-1204` (`canvas.clear()` archives, then every restored object archives) | (observed) 3 `create_segment` calls left 17 undo entries. `create_segment` then `undo` leaves a lone point A. A point on AB then `undo` leaves `MA` but not `MB`. Triangle plus circumcircle then `undo` removes only the circle. `delete_point` B then `undo` brings back B and `BD` but not AB, BC or the triangle. `delete_plot` then `undo` brings back the pdf and plot but not the fill. A load pushed 92 entries, and `undo` after it removed only the triangle. A failed `translate_object` or `scale_object` and a refused `update_point` each add entries. | GEO-03, GEO-04, CON-02, TR-04, TR-06, ST-01, CV-02, WS-02, NM-02, NM-03 |
| K2 | **Fixed in vl3c/MatHud#73; its marks are removed.** **Tool results say "Call successful!" when nothing happened**, and create results never say what was created. For undoable tools, any result that is not a small string or dict becomes the success message. | `result_processor.py:289-291`, `:304-313`; e.g. `managers/circle_manager.py:202-204` returns False; `point_manager.py:197-199` returns False; `canvas.undo()` returns False | (observed) `delete_circle(name="nope")`, `delete_point` at an empty spot, `delete_segment` of a non-existent segment and `undo` with an empty stack all report success. `create_point` on an occupied spot returns success and creates nothing. | GEO-03, ST-03, CV-04, NM-04 |
| K3 | **Fixed on `claude/fix-names-persistence`; its marks are removed.** **Requested names are silently replaced, and the replacement is not reported.** Point names are letters only: a hint is upper-cased, split into letters, then primes are tried. Per-hint state is never rewound after undo or delete. | `name_generator/point.py:203-239` (hint parsing `:150-156`, index kept `:234`) | (observed) `name="A"` when A exists gives another letter (`B` on a canvas holding only A). `name="Big"` gives `B'`. A segment named `s1` becomes `SA` with points S and A. `name="K"` after an undo of K gives `A`. `construct_midpoint(name="A")` gives `D`. All report "Call successful!". **Fix:** a create tool's result names what it produced, e.g. "Created Point 'B' instead of the requested name 'A'." (`creation_report.py`, used by `ResultProcessor`), including "Used the existing Segment 'AB'; ..." for a reused object (which then adds no undo entry), "Updated the existing Function 'f' to ..." for a redefined one, "centred on 'O'" for circles and ellipses, and one entry per part for a composite construction; the point-name generator reuses a hint's letters once they are free again, so `K`, segment `PQ` or triangle `XYZ` after an undo get their names back. A point name stays one capital letter with optional primes: `Far` becomes `F` and the result says so, which GEO-05 now checks instead of expecting a point named `Far`. | GEO-02, GEO-05, CON-06, NM-05 |
| K4 | **Fixed on `claude/fix-view-undo-noops`; its marks and waiver are removed.** Undo entries now hold the view (zoom and pan), the coordinate mode, grid visibility and grid spacing, and undo and redo restore the computations; `set_coordinate_system` and `set_grid_visible` are undoable tools. Mouse panning and zooming still add no entry. Undo restores only the view parts the undone batch changed (zoom and pan as one, the mode, each grid's visibility), and a batch that leaves the canvas as it found it (a clear of an empty canvas, a zoom and back) adds no entry. **Undo only snapshots drawables.** Zoom and view are listed as undoable but are not in the snapshot. `clear_canvas` resets the view, and undo does not restore it. `undo`/`redo` push computations but do not restore them. | `managers/undo_redo_manager.py:85-90`, `:135-138`, `:166-169`; `function_registry.py:309`; `canvas.py:391-396` | (observed) `zoom` to ±2 then `undo`: the view stays ±2 and the undo entry is spent. Clear then undo restores the objects with the default view. | CV-01, CV-03 |
| K5 | **Fixed on `claude/fix-names-persistence`; its marks are removed.** **Workspace round-trips lose attributes that `get_canvas_state` does not show**, so a state comparison passes while data is lost. Lost: colours of points, segments, polygons, circles, ellipses, vectors and functions; a custom angle name; grid visibility. | colours: `drawables/point.py:81-83`, `drawables/circle.py:72-79` (no colour in `get_state`); angle name: `drawables/angle.py:334` and `managers/angle_manager.py:577-581`; grid: `managers/coordinate_system_manager.py:129` | (observed) A red point and a green circle come back black. Green triangle sides, the orange ellipse, the red vector and the red function come back default. Angle `alpha` comes back as `angle_BAC_reflex`. The state diff is empty. **Fix:** points, segments, vectors, circles, ellipses, functions, piecewise functions and every polygon class write a non-default colour into `args.color` (`Drawable._add_color_to_state`), and the load passes it back; angles load and undo with their saved name; the coordinate system state records `polar_grid_visible: false`, and a load restores it and the Cartesian `visible` flag. A load gives each segment its saved colour again after the polygons, so an edge recoloured on its own keeps its colour. `__deepcopy__` (undo) keeps the name of angles, circles, ellipses and polygons. Old saves without these fields load with the defaults. Non-default colours, and a hidden polar grid in polar mode, now show in the `text` and `min_json` canvas formats. | WS-01 |
| K6 | **Fixed in vl3c/MatHud#72; its marks are removed.** **Deletes cascade into objects that only share parts.** `delete_angle` deletes both arm segments unconditionally, which deletes the triangle; the tool description promises "if they are no longer part of other drawables". `delete_polygon` deletes every edge, so a triangle sharing an edge is deleted too. | `managers/angle_manager.py:373-380`; `managers/polygon_manager.py:261-267`; `managers/segment_manager.py:549-563` | (observed) Triangle ABC plus the angle at A, then `delete_angle`: only BC and the points remain. Triangles ABC and ACD, then `delete_polygon` ABC: ACD and CA are gone. | GEO-14, NM-06 |
| K7 | **Fixed in vl3c/MatHud#72; its marks are removed.** **`delete_vector` recurses forever when a segment has the same endpoints.** The vector deletes "its" segment by coordinates, and deleting that segment deletes vectors by coordinates. A segment and a vector can also both be named `AB`. | `managers/vector_manager.py:265-272` ↔ `managers/segment_manager.py:545-546` | (observed) `Error: maximum recursion depth exceeded`; the undo stack jumped from 6 to 358 entries. | GEO-07 |
| K8 | **Fixed on `claude/fix-stale-geometry`; its marks and the I3 waiver are removed.** **A cached circle formula goes stale when the centre moves through another object.** Polygon transforms refresh segment formulas, not circles centred on a moved vertex. Fix: `circle_formula` and `ellipse_formula` are properties computed from the current centre, radii and rotation, so no move path can leave them stale; translating a point now also refreshes its segments and dependants. | `drawables/circle.py:59`, `:77`; `managers/transformations_manager.py:431-447` | (observed) After translating triangle ABC by (10,0), circle A(5) centred on A still reports `(x - 0.0)**2 + (y - 0.0)**2 = 5**2`. This formula reaches the model in the `json` canvas format and through `get_current_canvas_state`. | GEO-11 |
| K9 | **Fixed on `claude/fix-stale-geometry`; its marks are removed.** **`create_circle_arc` moves existing points onto the circle.** Endpoints reused from existing points are projected in place, bypassing the edit policy that forbids moving referenced points. Fix: an existing point is never moved. One already on the circle is used as is; one off the circle stays put, the arc gets a point at its projection (reusing a point already there), and the tool result names both points. Points the call creates from reference coordinates are still snapped (onto an existing point when one sits at the projection). Endpoints that land on the same spot are refused, and a failed call removes the points it created. | `managers/arc_manager.py:425-431` | (observed) With segment A(0,0)-B(3,0), an arc of radius 5 given the reference point (3,0) moves B to (5,0), and AB grows to length 5. | GEO-13 |
| K10 | **Fixed on `claude/fix-names-persistence`; its marks are removed.** **`create_colored_area` and `create_region_colored_area` reject `color: null`**, which the strict schema requires models to send when they have no colour. The Python default applies only when the argument is omitted. | `managers/colored_area_manager.py:139` → `utils/style_utils.py:216`; default at `canvas.py:1770`, `:1790` | (observed) `Error: Invalid CSS color: None` for both tools. **Fix:** a None colour, opacity or resolution means the default. | FN-01, AR-04 |
| K11 | **Fixed on claude/fix-functions; its marks are removed.** A translation now shifts the vertical asymptotes and point discontinuities by dx and the horizontal asymptotes by dy. **Translating a function leaves its asymptotes behind.** Bounds, holes and the expression move; `vertical_asymptotes` and `horizontal_asymptotes` do not. The renderer splits the curve at these values. | `drawables/function.py:165-225` | (observed) `1/x` shifted by (2,3) becomes `(1/(x - 2)) + 3` with `vertical_asymptotes: [0]`, `horizontal_asymptotes: [0, 0]`. | FN-02 |
| K12 | **Fixed on claude/fix-functions; its marks are removed.** A redraw redefines the function (expression, bounds, holes) and recomputes its analysis; `update_function` recomputes it after a bounds change; reversed bounds are swapped and the tool result says so, equal bounds are an error. **`draw_function` on an existing name updates in place without re-analysing**, and `draw_function` never validates its bounds (`update_function` does). | `managers/function_manager.py:143-169`; validation only in `update_function` (`:305-310`) | (observed) `f = 1/x` redrawn as `x^2` keeps `vertical_asymptotes: [0]`. `left_bound=5, right_bound=-5` reports success. | FN-03, FN-08 |
| K13 | **Fixed on claude/fix-functions; its marks are removed.** A tangent or normal is refused (an error, nothing created) where f or f' is undefined or infinite, within two derivative steps of a listed vertical asymptote, where f(x) is larger than 1e12 in magnitude (too large to draw), or where f'(x) is larger than 1e12 within 1e-3 of a listed asymptote (too steep to draw). **No guard for a tangent at a vertical asymptote.** Only NaN and a missing derivative are rejected, so a huge finite value at x = π/2 gets through. | `managers/tangent_manager.py:166-180` | (observed) The tangent to `tan(x)` at π/2 creates a segment with y ≈ 1.63e16 and reports success. | FN-06 |
| K14 | **Fixed on claude/fix-functions; its marks are removed.** Translating a function shifts the explicit bounds of areas between it and the x-axis or a constant; an area between two functions keeps its bounds when one moves. **Coloured-area bounds do not follow a translated function.** | `managers/transformations_manager.py:153-154` (cache invalidation only); `drawables/functions_bounded_colored_area.py` | (observed) f = x² with the area on [0,2], shifted 3 right: f's bounds are [0,6], and the area stays on [0,2]. | AR-01 |
| K15 | **Fixed on `claude/fix-names-persistence`; its marks are removed.** **An area between two segments adds points and splits segments.** Its helper points are created with `extra_graphics=True`. | `managers/colored_area_manager.py:188-201` → `managers/segment_manager.py:565-604` | (observed) Points E(1,0) and F(3,0) and segments EA, EB, FA, FB, FE appear; the undo depth goes from 7 to 27. **Fix:** the helper points are no longer created; the renderer already shades the overlap of the two x-ranges. | AR-02 |
| K16 | **Fixed on `claude/fix-stale-geometry`; its marks and the I3 waivers are removed.** **Polygon type flags are computed only in the constructor.** Rotation without an explicit centre also skips the dependency refresh. Fix: `Polygon.get_type_flags()` classifies the current vertices on each call (triangles, quadrilaterals and n-gons); a rectangle's `rectangle` label comes from the flags, so a sheared rectangle drops it; every rotation refreshes dependants. The flags are memoised on the vertex coordinates. Copying a quadrilateral (undo snapshots) no longer re-runs the constructor checks, so a sheared Rectangle no longer breaks every later batch; saves store rectangle vertices in cyclic order, a restore brings a no-longer-rectangular Rectangle back as a Quadrilateral, and each restored item is restored on its own. | `drawables/triangle.py:72` (quadrilateral and n-gons alike); `managers/transformations_manager.py:271-272` | (observed) An equilateral triangle sheared by factor 1 still lists `equilateral`. | TR-01, TR-05 |
| K17 | **Fixed on `claude/fix-stale-geometry`; its marks are removed.** **Cached angle values are never refreshed.** `angle_degrees` is set at creation. `handle_segment_updated`, the only refresher, is never called. The renderer draws from the cache. The text canvas format recomputes, so the model and the drawing disagree. Fix: `raw_angle_degrees` and `angle_degrees` are properties computed from the current vertex and arm points; the dead refresher is removed. | `drawables/angle.py:219-222`; `managers/angle_manager.py:472` (no callers); `rendering/helpers/angle_renderer.py:227` | (observed) The angle at B stays 60° after the shear. | TR-05 |
| K18 | **Fixed in vl3c/MatHud#72; its marks are removed.** **Graph bugs.** Adjacency-matrix edges are always created as directed, so an undirected graph gets vectors in both directions and `segments: []`. `delete_graph` deletes every vertex point, including points that existed before and were reused as vertices, and cascades into their triangles. | `managers/graph_manager.py:176-186`; `managers/graph_manager.py:258-264` with reuse at `point_manager.py:152-154` | (observed) The matrix `[[0,1,0],[1,0,2],[0,2,0]]` gives 4 vectors, and `shortest_path` returns `{"path": None}`. A graph with a vertex at an existing triangle vertex (name X dropped, reused as A): `delete_graph` deletes A, AB, CA and the triangle. | GR-03, GR-04 |
| K19 | **Fixed on `claude/fix-stale-geometry`; its marks are removed.** **Polygon subtypes move and re-order the given vertices.** Canonicalisation rebuilds the vertices from the subtype. Fix: `create_polygon` keeps the vertices in the given order (corners out of cyclic order are refused, never re-ordered). A subtype (and the rectangle and square polygon types) is fitted by `utils/polygon_subtype_checks.py`: vertices that already form it (type-flag tolerances) are used verbatim; vertices off by at most 2% of the average side length are adjusted, moving as few vertices as possible and never an existing point, and the tool result says which point was placed where; anything else is refused with the labelled sides and angles, the tolerances and, for equilateral and right isosceles triangles, a vertex that would work. The triangle and non-rectangle quadrilateral canonicalizers are removed. | `managers/polygon_manager.py:144-178` (`canonicalize_rectangle` / `canonicalize_triangle` / `canonicalize_quadrilateral`) | (observed) Square (0,0),(2,0),(2,2),(0,2) is stored as A(0,0) B(0,2) C(2,2) D(2,0) with coordinates `1.9999999999999996`. The equilateral triangle vertices get about 1e-11 of noise. | GEO-08 |
| K20 | **Action-trace `state_delta` is always empty.** The delta expects buckets of `{name: state}` dicts, but real states hold lists. The unit tests use the invented dict shape. | `managers/action_trace_collector.py:275-287`; `client_tests/test_action_trace_collector.py:17-18` | (observed) Adding point Q gives `{"added": [], "removed": [], "modified": []}`. The server logs this delta with every batch. **Fixed** in phase 1: the delta reads list buckets, and a name used by two buckets is keyed `Bucket:name`. | (harness) |
| K21 | **Fixed in vl3c/MatHud#73; its marks are removed.** **Errors returned as `{"error": ...}` dicts are not flagged as errors** in traces or turn metrics (`tool_errors`), so the footer and benchmarks under-count tool errors. | `result_processor.py:161`; `turn_metrics.py:75-76`, `:126` | (observed) `analyze_graph` on a missing graph and `inspect_relation` on a missing segment both give `is_error: false`. | GR-03, invariant I6 |
| K22 | **The CLI's canvas commands do nothing.** They use `window._canvas`, which is never set, and call `get_state`/`reset_view`, which do not exist. | `cli/browser.py:306-359`; `cli/canvas.py:65-295`; `main.py:140`, `:193` | (code) There is no `window._canvas` assignment anywhere in `static/` or `templates/`. `canvas state` returns `{}`, and `canvas clear` prints "Canvas cleared" without clearing anything. **Fixed** in phase 1: the commands use the scenario hooks. | (harness) |
| K23 | **Fixed on `claude/fix-batch-7`.** The name is percent-encoded with `encodeURIComponent` (`WorkspaceManager._load_workspace_url`); save and delete send it in a JSON body and list takes none, so the load URL was the only one affected. **The load URL is not encoded.** `name="a&x=1"` loads workspace `a`. Low severity. | `workspace_manager.py:1440` | (observed) WS-04: `load_workspace(name="scn_url&x=1")` answered `Workspace "scn_url&x=1" loaded successfully.` and loaded `scn_url`; `#2` did the same. Names are limited to `[\w-]` on the server, so an encoded name is simply rejected. | WS-04 |
| K24 | **Fixed on claude/fix-functions; its marks are removed.** A denominator zero is an asymptote only if f grows beside it on either side, judged on the differences between samples at shrinking offsets (so a constant offset changes nothing, and sampling starts closer than any neighbouring pole); otherwise it is listed as a point discontinuity (a hole, a jump, or a bounded oscillation such as sin(1/x)). **A removable discontinuity is reported as a vertical asymptote.** Every zero of a denominator is taken as an asymptote, without checking for cancellation. The Roadmap's "adaptive plotting" item plans to replace this string-based detection. | `utils/math_utils.py:2794-2796` | (observed) `(x^2 - 1)/(x - 1)` lists `vertical_asymptotes: [1]`. | FN-09 |
| K25 | **Fixed on `claude/fix-view-undo-noops`; its mark is removed.** `Canvas.reset` now resets the polar grid's spacing, and the reset hook no longer does it itself. **`Canvas.reset` does not reset the polar grid.** It resets the Cartesian grid but never calls `PolarGrid.reset()`, so the zoom-adapted ring spacing survives `clear_canvas`, `reset_canvas` and workspace loads (which clear first). After zooming in and clearing, polar mode draws its rings at the old spacing across the default view: thousands of circles per frame. | `canvas.py:422-426` (`_reset_drawables_state`); `polar_grid.py:149-151` (`reset`, no callers) | (observed by the first replay run) After CV-01's zoom to ±2, WS-01's `load_workspace` in polar mode took 30 to 57 s instead of 2 to 5 s; the spacing stays 0.2 instead of 50. The reset hook now resets it, so scenarios stay isolated. | CV-05 |
| K26 | **Fixed on `claude/fix-view-undo-noops`; its marks and waivers are removed.** The `update_*` tools (colour for every type the tools cover except coloured areas; also point name and position, circle and ellipse centre), `zoom`, `set_coordinate_system` and `set_grid_visible` return a `NoChangeResult` ("Circle 'A(2)' already has color red; nothing changed.") when nothing would change, and add no undo entry. **A change that is already in effect reports success and adds an undo entry.** `update_circle` with the colour the circle already has, and `zoom` to the view already shown, answer "Call successful!" and archive an undo entry, although nothing changed. | the update and zoom paths archive before comparing (found on `main` after vl3c/MatHud#73) | (observed) CV-06: the undo depth grows by one per repeated call and I4 flags the bare success. | CV-06 |
| K27 | **Fixed on `claude/fix-names-persistence`; its marks are removed.** **`fit_regression` lists reused points as if it created them.** With `show_points`, a data point on an existing point reuses it, and `point_names` reports it like a new one; `point_color` is not applied to it. | `managers/statistics_manager.py:486-502` with reuse at `managers/point_manager.py:149-151` | (observed) ST-03: `point_names` is `[A, B, C, D]` with A the pre-existing point, and the result says nothing about reuse. This was finding 5 below; it was marked K2 until K2's fix showed it is separate. **Fix:** when points were reused, the result adds `reused_point_names`, `created_point_names` and a note; the tool description says to delete only the created points then. | ST-03 |
| K28 | **Fixed on `claude/fix-batch-7`.** Horizontal asymptotes are estimated numerically (`MathUtils._limit_at_infinity`: f at ±9.73·4^k up to the first overflow, settled by Aitken extrapolation of the trailing samples or a shrinking spread), with no nerdamer call; the `limit` tool runs nerdamer under a step and time budget (`static/nerdamer_guard.js`, 2000 steps or 1.5 s, the clock also checked in nerdamer's arithmetic, simplify and printing, so a limit stops within a few hundred milliseconds of the budget; worst measured about 1.75 s; derive, integrate, simplify, expand, factor and solve get 5 s) and answers an abandoned limit with an error that gives the numeric estimate (one-sided values at a finite point). **`draw_function("abs(x)/x")` hangs the browser tab.** Every draw computes horizontal asymptotes with nerdamer's `limit()`, whose L'Hôpital loop never ends when the derivative gives the quotient back (d/dx abs(x) = abs(x)/x). The `limit` tool hangs the same way. | `utils/math_utils.py` (`calculate_horizontal_asymptotes`, `limit`); `static/vendor/nerdamer/1.1.13/Calculus.js:2229-2246` (`Limit.divide`) | (observed, node probe of the vendored nerdamer) `limit` never returns for `abs(x)/x`, `sqrt(x^2)/x`, `x/(abs(x)+1)`, `abs(x^3)/x^3`, `(abs(x)+1)/x` at ±∞ and for `abs(x)/x` at 0. It also gives wrong values: `floor(x)/x` → 0, `sqrt(x^2+1)/x` → 0, `x/sqrt(x^2+1)` → Infinity. `solve` and `diff`, the other nerdamer calls on the draw path, finished for every probed expression. | FN-10 |
| K29 | **Fixed on `claude/fix-batch-7`.** The canvas wrapper passes only `new_color` and `use_major_arc`, the tool's schema; moving the endpoints stays unsupported (the tool description says to recreate the arc, which never moves existing points). Selecting the sweep the arc already has is a `NoChangeResult`. **`update_circle_arc` always fails.** `Canvas.update_circle_arc` forwards `point1_*`/`point2_*` keyword arguments (all None) that `ArcManager.update_circle_arc` does not accept. | `canvas.py:2061-2090` (`_build_update_circle_arc_kwargs`); `managers/arc_manager.py:553` | (observed) GEO-17: every call answers "Error: update_circle_arc() got an unexpected keyword argument 'point1_name'". | GEO-17 |
| K30 | **Fixed on `claude/fix-batch-7`.** A null `rotation_angle` means 0 and a null `t_min` means 0 (`t_max` already defaulted to 2π). An audit of every create, draw, construct, plot and update tool, each called with all its optional arguments null (`TestNullArguments`), found no other crash; `update_colored_area` on a region area fails for another reason ("Edit policy for ClosedShapeColoredArea is not configured", not filed). **`create_ellipse` with `rotation_angle: null` fails**, as does `draw_parametric_function` with `t_min: null`; strict-schema models send null for every optional argument. | `canvas.py:1335` (`create_ellipse`), `:1463` (`draw_parametric_function`) | (observed) GEO-18: the ellipse is stored with rotation None and the batch then fails with "must be real number, not NoneType"; the parametric call fails with "float() argument must be a string or a real number, not 'NoneType'". | GEO-18 |
| K31 | **Fixed on `claude/fix-batch-7`.** `no_op_tools.py` also compares `update_segment`'s `new_label_text` and `new_label_visible` and `update_label`'s `new_text`, and wraps `translate_object` so an offset of (0, 0) is a `NoChangeResult`; translating or rotating a plot is an error raised before archiving; `Function.translate` re-raises a parse failure after restoring its bounds. **Calls that change nothing still report "Call successful!"** (the K26 family): hiding an already hidden segment label, setting a label's current text, translating by (0, 0), and translating or rotating a bars, continuous or discrete plot, whose `translate` and `rotate` do nothing (`rotate_object` excludes only the base class name `Plot`). A function whose shifted expression fails to parse keeps its old expression, and the failure is only printed. | `no_op_tools.py`; `drawables/plot.py:63-69`; `managers/transformations_manager.py:175`; `drawables/function.py:215` | (observed) NM-07 and ST-05: I4 flags each bare success. | NM-07, ST-05 |
| K32 | **Fixed on `claude/fix-batch-7`.** tan() poles are classified like denominator zeros, each one (a 3-sample quick check first, the full test when it is unclear), candidates closer than 1e-6 relative are merged keeping the one nearest zero (one sort, one pass), and every asymptote and discontinuity is filtered to the bounds. The name special case in `FunctionsBoundedColoredArea._has_asymptote_at` is removed; the function's own analysis covers tan(x/100). **Vertical asymptotes that are not there.** The tan rule marks every `tan(` in the text, so `x/tan(x)` and `1/tan(x)` list ±π/2, where both tend to 0; nerdamer's near-duplicate roots of a denominator (0, 1.4e-9, 6.2e-9, 9.7e-9 for `1 - cos(x)`) are each listed; log and denominator asymptotes are not filtered to the bounds (`1/x` on [2, 5] lists 0); and the coloured-area renderer gives any function named `f3` the asymptotes of tan(x/100). | `utils/math_utils.py` (`_vertical_asymptotes_and_discontinuities`); `drawables/functions_bounded_colored_area.py:284-297` | (observed) FN-11: I3 flags x/tan(x)'s ±π/2 as not blowing up. | FN-11 |

Other findings from reading the code only, not yet reproduced in the app, each with a scenario that would catch it:

1. **Renaming a circle centre leaves its arcs pointing at the old circle name** (`drawables/circle_arc.py:63`, `managers/arc_manager.py:611`). A later `delete_circle` leaves the arc orphaned, and saving and reloading drops it. Add: "rename the centre, delete the circle, check the arc is gone" (I2).
2. **Vector endpoints can be renamed while the vector keeps its old name.** The vector is registered only against its internal segment (`managers/drawable_dependency_manager.py:477-478`). Add to NM.
3. **The edit policy needs a solitary point even for a colour change** (`managers/edit_policy.py:53-58`), so a triangle vertex cannot be recoloured. The tool text says only "solitary point". NM-02 covers the rename case.
4. **Creating an ellipse that already exists ignores `rotation_angle`** (`managers/ellipse_manager.py:111-116`). An ellipse with the same centre and radii but a new angle returns the old one unchanged.
5. (Filed as K27, now fixed.) **`fit_regression` with `show_points` reuses existing points and lists them as if they were new.** `point_color` is not applied to them (`managers/statistics_manager.py:459`). ST-03 has the setup; its check on `point_names` catches this.
6. **Coloured-area names are not unique** (`area_between_{f}_and_{g}`), so `delete_colored_area` removes only the first of two such areas.
7. **For a function and a segment, `left_bound`/`right_bound` are ignored** (`managers/colored_area_manager.py:207`), although the schema says they are used.
8. **A continuous `plot_distribution` that fails after drawing its pdf leaves the function behind** (`managers/statistics_manager.py:182-183`).
9. **`translate_object` has no type check**, so arcs, angles, graphs and areas fail only after archiving. TR-06 shows this for arcs.
10. **Tangents at a kink are wrong:** the central difference returns 0 at x = 0 for `abs(x)` (`utils/math_utils.py:3170-3185`). `draw_tangent_line` also does not search piecewise functions (`managers/tangent_manager.py:107-121`).
11. **Each `search_tools` call runs the search twice.** The server intercepts it (`static/routes.py:224-243`). The client then also executes it and POSTs `/search_tools` (`function_registry.py:223-293`, `static/routes.py:1146`). In `api` or `hybrid` mode each search can call the model, so one search can cost two model requests. Live runs will show this in `requests` per turn.

The client has no cap on tool loops: a model that keeps calling tools never finishes a turn. This is a design gap rather than a bug. The harness enforces its own cap (4.3), and the app could use one too (open question 8.4).

## 7. Implementation plan

Sizes are rough: S is under a day, M is 1 to 3 days, L is 3 to 5 days.

| Phase | Work | Size | Depends on |
|---|---|---|---|
| 1. Client hooks | `static/client/scenario_hooks.py`: the six hooks, pure helpers for the inspection view, registration in `main.py`, Brython tests in `client_tests/test_scenario_hooks.py` (registered in `tests.py`). Fix K20 so action traces carry real deltas. Point `cli/browser.py` and `cli/canvas.py` at the hooks (K22). Add `MATHUD_WORKSPACES_DIR` to `static/config.py`. | M | - |
| 2. Loader and check engine | Package `cli/scenarios/`: `model.py` (dataclasses, loader, null filling, `ToolArgumentValidator`), `geometry.py` (normalized view, selectors, tolerances), `checks.py` (outcome checks and invariants I1 to I7, all pure). Pytest suites in `server_tests/test_cli/`: the checker on hand-built states, including states copied from section 6's observations, and a gate that validates every file in `scenarios/`. | L | - |
| 3. Scenario files | Port section 5 into `scenarios/*.json` (12 files), with `known` markers. The prototype's `catalogue.py` data converts almost mechanically. | M | 2 |
| 4. Replay runner | `cli/scenarios/runner.py`: server start with pinned env, browser session, reset, setup, steps, snapshots, per-step timeout, browser restart, screenshots. `test scenarios --mode replay`, `--smoke`, `--only`, `--tags`. First full replay: every non-known check green. | M | 1, 2, 3 |
| 5. Reporting | `results.jsonl`, `results.json`, `summary.md`, failure artefacts, `--regrade`, `--dry-run`. Move the benchmark's price table and `ResultSink` into a shared module. | M | 4 |
| 6. Live runner | `sendMatHudMessage` loop, turn caps and stop, local model discovery and the `local_agent` check, the OpenRouter guards, `--repeats`, `retrace` mode and failure classification. The first live smoke run on the local model gives real timings for 4.9. | L | 4, 5 |
| 7. CI | A `scenario-replay` job in `tests.yml`, non-blocking at first. The pytest gates join `server-tests` straight away. Make the job blocking once it is stable, with XPASS failing it. | S | 4 |
| 8. Bug fixing | Work through K1 to K24 (K1, K2, K5, K6 and K10 first: they affect every model interaction), removing `known` markers as fixes land. | ongoing | 4 |

A reasonable first milestone is phases 1 to 4 with the smoke subset. That already turns section 6 into regression tests.

Status: phases 1 to 5 are done, with the whole catalogue rather than only the smoke subset. Differences from the plan: the benchmark's price table and `ResultSink` were not moved into a shared module (the scenario reports have their own sink; the price table is needed only by live mode); a `grade.py` module shares step grading between the runner and `--regrade`; and the runner resets the session between scenarios, restarting Chrome only after a hang (it retries the scenario once).

Phase 7 status: the `scenario-replay` job in `.github/workflows/tests.yml` replays the whole catalogue on pushes and pull requests with `--start-server`, non-blocking (`continue-on-error`) like `client-tests`; on a failure it uploads `logs/scenario_runs/ci` as the `scenario-replay-report` artifact. The pytest gates (`server_tests/test_cli/test_scenario_*.py`: loader, argument validation, catalogue and check engine) already run in the blocking `server-tests` job. Still to do: make the job blocking once it is stable, with XPASS failing it.

Phase 6 is done on `claude/scenario-live-mode` (`cli/scenarios/live.py`, `live_config.py`, `classify.py`; usage in "How to run", items 8 and 9). Differences from the plan:

1. Local models are discovered through the server's `/api/available_models`, which calls `LocalAgentAPI.fetch_models` against the server's own `LOCAL_AGENT_BASE_URL`, rather than by the harness calling `fetch_models` itself: the harness and the app then always agree on the llama-server, and discovery is also the `local_agent` registration check.
2. More spend guards than planned: provider keys the run does not use are set to empty strings in the server's environment; OpenRouter needs `--models` and searches tools locally (`api` and `hybrid` search send model requests that turn metrics do not count, so they are refused); "no retries" needed an app setting, `MATHUD_OPENROUTER_MAX_RETRIES` (`static/providers/openrouter_api.py`), which the harness pins to 0.
3. The price table moved to `static/model_prices.py` and the reasoning-effort choices to `static/providers/local` (`REASONING_EFFORT_CHOICES`); the benchmark imports both. `ResultSink` is still not shared.
4. Action traces keep each batch's calls and `state_delta`, but not its canvas or undo depths, so every live turn with more than one batch of counted calls is retraced, and the retrace's per-batch I4 and I5 (each batch on its own canvas and undo depths) replace the turn's when the retrace reproduces the live canvas. Without a retrace (`--no-retrace`, or one that did not reproduce) the turn is judged as a whole: I5 accepts between one entry per batch whose delta names a changed drawable and one entry per batch, and skips turns with undo or redo; I4 keeps only "every call failed but the canvas changed" and the naming rule, since batches can cancel out (zoom, then undo). The trace export cuts results to 500 characters; I4's naming rule skips such results.
5. Request caps are enforced by the client: `sendMatHudMessage` passes the turn's cap (or what the run's cap leaves, if less), and the client ends the turn instead of sending a request beyond it. Polling alone could not do it: the client sends the next request in the same task that records the previous one, and with a model answering in 20 ms a turn capped at 4 sent 7. The harness still stops a turn that runs past its cap, as a backstop. The client checks the cap before it runs a batch whose results would need a request beyond it, so the batch does not run, and it abandons the server's turn (as a stop does), so the batch's tool calls are not left awaiting results in the history. This is a deliberate change in grading: the model's last batch within the cap used to run (its results were never sent back), and now it does not run at all, so a turn that hits the cap is graded on the canvas as the model last saw it. A turn sends at most N requests for a cap of N; the plan and the cost ceiling count N per turn. The request in flight when a turn is stopped, times out, fails or breaks off on an error is counted as sent.
6. Dropped calls come from the turn metrics (`tool_calls` minus `tool_executions`), not from the server logs. Turn outcomes `request_cap` and `not_started`, provider and transport errors, and a turn whose action traces could not be read, count as `infra` (with every later turn of the run); client and server errors, and turns the server `abandoned` (nothing but the harness's own tab talks to its server, so a dropped reply is a bookkeeping bug), count as `app` (a stream request answered with an HTTP 4xx or 5xx status is a `server` error; only a request that got no answer is `transport`); `max_requests`, `truncated` and `filtered` count as `model`. `--max-infra-rate` keeps an app regression that breaks every turn from passing as infrastructure. A retrace stands in for its live run only when it reproduces both the canvas and how far each step moved the undo stack.
7. One client change: `getActionTraces()` writes values JSON cannot express as strings, as the scenario hooks do. Brython's `json` rejects tuples, so one result with a tuple (`analyze_graph`'s MST edges) made the whole trace list unreadable.
8. Server and client changes for abandoned turns: a stopped or timed-out turn's request runs on in the server (the browser only aborts its fetch), and a reply ending in tool calls appended them, with their "Awaiting result..." placeholders, to the next conversation. Each provider now keeps a conversation generation, bumped by a new or restored conversation, by each new user message and by `/save_partial_response`, which the client sends on a stop and now also on its own timeout. Every reply stream starts with a `turn` event; the client sends that turn back with the stop, and the server abandons only that turn while it is current, so a stop arriving after the next message drops nothing. A reply from an older generation never reaches the history (the check and the update share a lock with stops and resets), runs no tool calls, and the stream ends with a final event whose finish reason is `abandoned`, which the client treats as a quiet end of turn (New Conversation, a workspace load, a stop from another tab). The providers stop and close the model stream, record the request's metrics with that outcome, and abandoning a turn unloads the tools its `search_tools` calls loaded. New Conversation stops a running turn first. This also fixes the app's Stop button and its response timeout. `GET /api/requests_in_flight` reports the model requests still running.
9. Follow-ups from the last review of item 8:
   - The client sends its turn with every tool-results request too. When that turn is no longer current (a stop, a timeout, the next message or a new conversation was processed first), the route answers with only the `abandoned` final event: no model call, nothing appended, no searched tools loaded. Tool results sent without a turn are dropped after a stop or reset until the next user message.
   - The route claims the request's generation (`begin_turn` or `continue_turn`) and hands it to the provider, which adds the prompt to the history at once, under the history lock, before the route sends the `turn` event; the model is called when the stream is read. A stop that follows the turn event therefore always finds the prompt in the history, and the provider never takes a newer generation than the route.
   - Abandoning a turn answers the tool calls still awaiting results with "No result: the turn ended before this result was returned.", so no "Awaiting result..." placeholder is left behind.
   - A stop or timeout sends as its partial text only what streamed since the last final event: after a tool batch the chat still shows the previous request's text, which the history already holds.
   - A turn stopped before any text leaves its user message unanswered, so the next user message follows it. The history keeps both; the request merges consecutive user messages into one (chat completions, OpenRouter, LocalAgent and Anthropic), since several chat templates and APIs require alternation.
   - A tab whose turn was abandoned from elsewhere keeps its partial text but adds a short note, "This reply was stopped because the conversation changed elsewhere." A tab's own stops (and the harness's, which go through `stopMatHudTurn`) abort the stream first and show no note.
   - `--regrade` uses a `--max-infra-rate` given on its command line instead of the stored one, and a summary that fails without unexpected failures (too many infra runs) no longer prints "No unexpected failures." in green.

The live runner was checked end to end with a scripted stand-in for llama-server (it answers each prompt with the scenario's reference calls, after a `search_tools` batch, then "Done."): the smoke subset graded all 18 turns, and its only failures were two answer checks, classified `model`, with retraces that reproduced the live canvas; timeouts, the per-turn request cap and `--mode retrace` were exercised the same way. Over the whole catalogue it found one app problem the replay cannot see: in a live turn the `analyze_graph` MST result could not be serialized ("Circular reference detected"), so sending the tool results back to the model failed and GR-06's turn ended in `error`; vl3c/MatHud#89 fixes that at the source. A stand-in that answers in 20 ms showed the per-turn cap holds (4 requests for a cap of 4), one that answers after 70 s showed its late reply stays out of the next turn's history, and a stop sent after the next message no longer drops it. The first run on the local model is still to come; its timings replace the estimates in 4.9.

**Attach mode** (`--attach-desktop N`, on `claude/desktop-automation`): replay and live runs can play out in the user's desktop window instead of a server and headless Chrome of their own, so a person can watch each reset, setup batch, prompt, canvas and reply. `python mathud_desktop.py --automation-port N` gives the window's WebView2 a DevTools endpoint on 127.0.0.1:N; `cli/cdp.py` drives it (`Runtime.evaluate`, `Page.captureScreenshot`, `Page.reload`), and the runner's `BrowserSession` works through an `AppBrowser` interface that both backends implement, so the live and replay runners are unchanged. Differences from a run of its own:

1. Nothing is pinned. The desktop app runs with the user's `.env`, so `server_env` cannot be applied: the pin options are refused, and the run config records `attached_desktop: true`, `pins_applied: false` and the `unpinned_settings` (tool exposure, canvas format and budget, search mode, reasoning effort, OpenRouter retries, workspace directory, provider keys); `summary.md` and the summary JSON repeat it. Results of an attached run are therefore not comparable with pinned runs unless the `.env` matches the pins.
2. The model guard holds: every model must be registered under the provider in the app's `/api/available_models`, every message names its model, local is the default, and OpenRouter still needs `--models`, the request cap and its dry run. The search-mode and no-retry pins cannot be set on the app's server, so the app reports them (`GET /api/automation_settings`: the effective `tool_search_mode`, `openrouter_max_retries` and `tool_exposure`), and an attached OpenRouter run (like `desktop prompt --provider openrouter`) is refused unless the search mode is `local` and retries are 0; an app too old to report them is refused too. A live run records the reported settings as `server_settings`. A model the server registers but the window's dropdown lacks (the page loaded before llama-server) is added by refreshing the window's model list once (`window.refreshMatHudModels()`).
3. Every scenario resets the window's canvas, undo history and chat (live: also the server conversation; a replay run resets the conversation once at its start, since replay never adds to it), so the run always prints what it will do, asks first unless `--yes`, and refuses while a turn runs in the window. Workspace scenarios are always skipped and `--allow-workspace-writes` is refused (the app's workspace directory is the user's); while the run drives the window, `setMatHudAutomationGuards` makes the window's workspace tools answer with an error, so neither a model nor a retrace can save, load or delete the user's workspaces. The block is applied whenever the app becomes ready and again before every hook that can run tools (a reload, also one from outside, lifts it), and lifted when the run ends. It is a 90 s lease, renewed by those hooks and every 20 s by a turn's status polls: if the CLI crashes or is killed, the window lifts it by itself, and meanwhile the refusal says an automated run blocked the tools and how to lift the block (wait, or reload the window). After a recovery reload the run waits (up to 10 s) for the window's model list to load, and a send refused for a model missing from the dropdown refreshes the list and is retried once; such refusals count no request. Each fitted step's record keeps the fitted view (`fitted_view`), so `--regrade` rebases exactly as the run did: the attached replay of the whole catalogue (77 scenarios) regrades with the same 1493 passing checks. The window is never navigated on open; it is reloaded only to recover from a hang, and closing the session only disconnects. Ctrl+C stops the window's turn before the reports are written; a hook missing after an outside reload waits for the app once and retries; and when the DevTools port refuses connections (the window was closed) the run stops instead of erroring on every remaining scenario.
4. `--pace S` pauses after each step (default 1.5 s attached, 0 otherwise) so each canvas can be seen before the next reset.
5. `--fit-view` (the default when attached; `--no-fit-view` turns it off; refused without `--attach-desktop`, so headless runs never fit) calls `fitMatHudView` after setup and after each step, once the step is graded, recorded and its failure artifacts saved, then pauses. The scenarios draw within a few units while the default view spans about +-400, so without it the drawings are too small to see. The fit changes only the view, but the next step starts from it, so the grader's baseline is rebased (`ScenarioGrader.rebase_view`): the invariants' previous canvas and any snapshot taken at the step just graded become the fitted canvas (same drawables, fitted view). Taking the baseline after the fit, rather than leaving the view out of comparisons, keeps I4 and I5 (which count a view change as a change) and `state_equals ... view: true` exact: without the rebase, a batch that changes nothing after a fit fails I5. Undo restores only the view parts a step changed, so an undone step keeps the fitted view, consistent with the rebased snapshots. Scenarios that set or check the view (the `view` tag, a `zoom`, `set_coordinate_system` or `set_grid_visible` call, or a check that reads the view: CV-01, CV-03, CV-05, CV-06, GEO-05) are never fitted, because a fit would change what they test (CV-06 expects a repeated zoom to be a no-op; CV-03 expects undo to restore its own zoom). So are scenarios whose steps call tools that read the view: `find_function_features` without both bounds (it searches the visible x range), `generate_graph` with vertices lacking coordinates and no `placement_box` (it lays them out in the visible view), and checks of a label's `reference_scale_factor` (the zoom at the label's creation). No catalogue scenario has such a call today. In a live run the model then sees the fitted view in its canvas summary; the run config records `fit_view: true` and `summary.md` says so. With fitting on, the attached replay of the whole catalogue passed (77 scenarios, 4 workspace scenarios skipped) and so did the smoke subset.

Checked end to end on Windows 11 with a window of its own (`--port 5110 --automation-port 9301`, scratch `LOCALAPPDATA`) and a scripted stand-in for llama-server: the DevTools endpoint listened on 127.0.0.1 only; `desktop prompt` took 1.2 s for a three-request turn and the prompt, reply and point showed in the window; an attached replay of the whole catalogue with `--pace 0` ran 77 scenarios (4 workspace scenarios skipped) in 6.5 s, all passing, against about 30 s for headless Chrome; an attached live run of GEO-01, CON-01 and MT-01 to MT-04 graded every turn (1.2 s each with the stand-in). An attached live run on the real local model is still to come.

## 8. Open questions

1. **Tolerance for vertex re-ordering (K19).** Should the fix keep the user's order and exact coordinates, or should the check accept any orientation as long as the letter order is documented? The catalogue assumes the former.
2. **Sampling functions in the checker.** Sampling needs an evaluator that matches the client's expression handling (`^`, implicit multiplication, `pi`). There are two options: record samples in the browser through the inspection hook (the same evaluator as the renderer), or re-implement evaluation in CPython. The first is more faithful; the second makes `--regrade` fully offline. A hybrid is possible: sample in the browser at the x values each check needs, and store the samples. **Resolved:** the hybrid is built. The runner grades each step once as a dry run, fetches the samples its checks ask for through `getMatHudCanvasState`, and stores them with the step, so `--regrade` stays offline for the stored x values.
3. **Should constructions become live** (recomputed when their parents move)? The Reference Manual documents them as static snapshots, and CON-01 and FN-07 pin that. If live constructions are wanted, those checks flip.
4. **A tool-loop cap in the app itself,** e.g. at most N tool batches per user turn, so a looping model cannot run forever outside the harness.
5. **Cross-type name uniqueness.** A segment and a vector can both be `AB`, and `translate_object` then takes the first match in layering order. Should names be unique across all types? I1 reports this as a warning until that is decided.
