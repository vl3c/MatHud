# MatHud Architectural Review — 2026-03-24

## Overall Assessment

The project has strong foundations: good test coverage (~3,745 tests), strict mypy/ruff enforcement, clean manager-based decomposition, and well-documented conventions. The main issues are **god classes**, **copy-paste patterns across managers**, **inconsistent error handling**, and **missing CI/CD automation**.

---

## CRITICAL: God Classes

These files have accumulated too many responsibilities and need decomposition:

### 1. `static/client/ai_interface.py` (2,342 lines, 30+ instance attributes)

The single largest architectural problem. This class mixes:
- HTTP/AJAX communication protocol
- Streaming response buffering (text + reasoning + tool calls — 3 independent state machines)
- Chat UI DOM manipulation (message containers, image attachments, scroll behavior)
- TTS playback coordination
- Action trace collection
- Test execution state

**Suggested extraction:**
- `ChatUIManager` — DOM manipulation, message rendering, image attachment UI
- `StreamingResponseHandler` — stream buffering, chunk assembly, timeout tracking
- `ReasoningHandler` — reasoning token streaming/display
- `ToolCallLogger` — tool call log entries and summary

This would reduce AIInterface to ~600-800 lines of orchestration + AJAX transport.

### 2. `static/client/canvas.py` (2,194 lines, 80+ public methods)

Canvas acts as a universal coordinator. It should delegate more aggressively:
- Lines 297-356: Visibility culling logic — extract to `VisibilityManager`
- Lines 225-241: Frame batching logic — belongs in the renderer layer
- Lines 168-199: Legacy drawable registration — indicates incomplete migration
- Lines 360-403: Zoom displacement calculations — should use `coordinate_mapper` directly

Target: reduce to ~500 lines focused on initialization, public API delegation, and state archiving.

### 3. `static/routes.py` — `send_message_stream()` (194 lines) and `send_message()` (167 lines)

Both route handlers mix request parsing, provider selection, vision capture, tool search interception, streaming, error recovery, and tool injection/reset. They share significant duplicated logic:
- Tool reset logic duplicated 3x (lines 891-897, 909-915, 1146-1152)
- Provider model setting duplicated (lines 811-812 vs 1108-1109)

**Suggested extraction:**
- `_validate_and_parse_message_request()` — shared request parsing
- `ProviderManager` — unified interface over `ai_api`, `responses_api`, and custom providers
- `ToolInjectionManager` — search/inject/reset tool lifecycle
- `VisionManager` — vision capture, WebDriver init, image handling

---

## HIGH: Manager Copy-Paste Pattern

All 10+ drawable managers (`PointManager`, `SegmentManager`, `CircleManager`, `EllipseManager`, `ArcManager`, `AngleManager`, etc.) reimplement identical patterns:

```python
def __init__(self, canvas, drawables_container, name_generator, dependency_manager, ...):
    self.canvas = canvas
    self.drawables = drawables_container
    self.name_generator = name_generator
    self.dependency_manager = dependency_manager
    self._edit_policy = get_drawable_edit_policy(...)
```

Every manager also reimplements deletion-with-dependency-cleanup identically.

**Fix:** Create `BaseDrawableManager` with:
- Common `__init__` accepting shared dependencies
- Standard `create()`, `get_by_name()`, `delete()` contract
- Built-in edit policy setup
- `remove_drawable_with_dependencies()` as inherited method

---

## HIGH: Inconsistent Error Handling

Three different error handling strategies coexist:

| Pattern | Where | Problem |
|---|---|---|
| `print()` | `routes.py` lines 103, 177, 183, 905 | Not captured in logs |
| `logging.error()` | `tool_call_processor.py` lines 63-75 | Correct approach |
| Bare `except Exception: pass` | `routes.py` line 886, `workspace_manager.py` line 137, `drawables_container.py` lines 77-80 | Silently swallows errors |
| `except Exception: return` | `routes.py` line 239 | Returns potentially invalid state |

Additionally, `static/client/utils/math_utils.py` (lines 2363, 2414, 2422) silently skips asymptote/limit calculations on failure.

**Fix:** Standardize on Python `logging` module throughout. Replace all bare `except:` blocks with specific exception types. Create per-module loggers.

---

## HIGH: Production Debug Code

1. **Test error trigger in routes.py** (lines 855-858):
   ```python
   # TEMPORARY TEST TRIGGER - REMOVE AFTER TESTING
   if "TEST_ERROR_TRIGGER_12345" in message:
       raise ValueError("Test error triggered for message recovery testing")
   ```

2. **Debug print statements in expression_evaluator.py** (lines 50, 72, 83, 90):
   ```python
   print(f"Evaluated numeric expression: {expression} = {result}")  # DEBUG
   ```

---

## MEDIUM: Workspace Serialization Issues

### State mutation during serialization
`Segment.get_state()` calls `_sync_label_position()` which mutates internal state. Serialization should be read-only.

### No schema versioning on drawables
Each drawable's `get_state()` returns `Dict[str, Any]` with no version field. Format varies by type. No validation on deserialization.

### Client workspace_manager.py (1,421 lines)
Restoration logic uses per-type methods that are largely boilerplate. Should use a factory pattern where each drawable type owns its own `from_state()` class method.

---

## MEDIUM: Missing Abstractions

### Provider management
Code repeatedly operates on `app.ai_api`, `app.responses_api`, and `app.providers` dict separately. A `ProviderManager` would centralize model resolution, tool injection, and conversation lifecycle.

### DrawableManagerProxy
Uses `__getattr__` reflection to break circular initialization. Adds runtime overhead and defeats IDE type narrowing. Consider restructuring the dependency graph.

### Naming inconsistency
`DrawablesContainer` uses `add()`/`remove()` while all managers use `create_*()`/`delete_*()`.

---

## MEDIUM: Configuration Scattered

Constants that should be centralized are spread across files:

| Constant | Location | Should be in |
|---|---|---|
| `AI_RESPONSE_TIMEOUT_MS = 60000` | `ai_interface.py:69` | `constants.py` |
| `REASONING_TIMEOUT_MS = 300000` | `ai_interface.py:70` | `constants.py` |
| `MAX_ATTACHED_IMAGES = 5` | `ai_interface.py:72` | `constants.py` |
| `IMAGE_SIZE_WARNING_BYTES = 10MB` | `ai_interface.py:73` | `constants.py` |
| `_MAX_TRACES = 100` | `action_trace_collector.py` | `constants.py` |
| `WORKSPACES_DIR = "workspaces"` | `workspace_manager.py:22` | `config.py` (server) |
| `CANVAS_SNAPSHOT_DIR` | `routes.py:47` | `config.py` (server) |

---

## MEDIUM: Renderer Telemetry Duplication

`SvgTelemetry` and `Canvas2DTelemetry` are near-identical classes. Extract a `BaseRendererTelemetry` class.

---

## LOW: CI/CD and Infrastructure Gaps

| Gap | Impact |
|---|---|
| No automated test execution on PR/push | Regressions can be merged undetected |
| No code coverage measurement | Can't track coverage trends |
| No linting checks in CI (ruff/mypy) | Type errors can be merged |
| 7 unpinned dependencies in requirements.txt | Reproducibility risk |
| No Dependabot or dependency scanning | Security vulnerability blind spot |

---

## LOW: Test Coverage Gaps

- `result_validator.py` (102 lines) — no dedicated tests
- `command_autocomplete.py` (504 lines) — minimal testing
- `canvas_event_handler.py` (622 lines) — only 2 test files for complex state machine
- `functions_definitions.py` (2,731 lines) — manual JSON schemas with no code generation

---

## Refactoring Roadmap

### Phase 1: Quick Wins — COMPLETE (PR #49)
1. ~~Remove debug code (test trigger in routes.py, print statements in expression_evaluator.py)~~
2. ~~Centralize scattered constants~~
3. ~~Standardize error handling (replace `print()` with `logging`, eliminate bare `except:`)~~
4. ~~Pin all dependencies in requirements.txt~~
5. ~~Extract shared env loading into env_config.py~~
6. ~~Extract route helpers (tool reset/provider deduplication)~~

### Phase 2: Structural — COMPLETE (PR #49)
7. ~~Create `BaseDrawableManager` — 9 managers migrated~~
8. ~~Extract `BaseRendererTelemetry` from SVG/Canvas2D~~
9. ~~Extract `VisibilityManager` from Canvas~~
10. ~~Make `get_state()` side-effect-free (Segment fix)~~
11. ~~Decompose AIInterface (2,339 → 1,078 lines) into 5 classes:~~
    - ~~ToolCallLogManager, MessageMenuManager, ImageAttachmentManager, TTSUIManager, ChatUIManager~~
12. ~~Add tests for all new modules (35 server + 74 client)~~

### Phase 3: Architecture (higher risk, long-term value) — TODO
13. **Add CI/CD pipeline** — GitHub Actions workflow running server tests (`pytest`), client tests (Selenium via CLI), ruff, and mypy on every PR. High value: protects all refactoring work going forward.
14. **Implement drawable state schema versioning** — Add version field to each drawable's `get_state()` output. Create `from_state()` factory class methods on each drawable for deserialization. Validate schema on workspace load.
15. **Restructure dependency graph to eliminate `DrawableManagerProxy`** — The proxy uses `__getattr__` reflection to break circular initialization. Restructure so managers receive specific interfaces (Protocols) rather than a proxy. This improves IDE support and type safety.
16. **Add workspace format migration support** — With schema versioning in place, add migration functions that upgrade workspace JSON from version N to N+1. Enables breaking changes to serialization format without data loss.
17. **Extract route handler logic into service classes** — Move business logic from Flask route functions into testable service classes. Routes become thin HTTP adapters. Enables testing without Flask test client.
18. **Further decompose AIInterface** — The remaining 1,078 lines still mix request building, streaming orchestration, timeout management, and test execution. Candidates for extraction: `RequestBuilder` (~200 lines for payload/vision/send), `StreamingOrchestrator` (~200 lines for _on_stream_final/error/timeout).
19. **Further decompose Canvas** — Canvas is still ~2,100 lines after VisibilityManager extraction. Candidates: `CanvasRenderingCoordinator` (frame batching, render dispatch), `ZoomController` (zoom displacement calculations).
