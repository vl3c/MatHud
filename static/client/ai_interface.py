"""
MatHud AI Communication Interface

Handles bidirectional communication between the client-side canvas and backend AI services.
Manages user interactions, function call processing, and visual presentation of AI responses.

Key Features:
    - AJAX-based communication with backend AI API
    - Function call execution and result aggregation
    - Markdown parsing and MathJax rendering for AI responses
    - Chat interface management with styled messages
    - SVG state transmission for AI vision capabilities
    - Computation history integration with canvas state
    - Testing framework integration

Communication Flow:
    1. User input → JSON payload creation with canvas state
    2. Backend AI processing with function calls
    3. Function execution and result collection
    4. Response rendering with markdown and math support
    5. Canvas state updates with computation results

Dependencies:
    - browser: DOM manipulation and AJAX requests
    - function_registry: Available AI function mappings
    - process_function_calls: Function execution coordination
    - workspace_manager: File persistence operations
    - chat_ui_manager: Chat message rendering and streaming display
"""

from __future__ import annotations

import json
import traceback
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, cast

from browser import document, ajax, window, console, aio
from constants import (
    AI_RESPONSE_TIMEOUT_MS,
    REASONING_TIMEOUT_MS,
)
from function_registry import FunctionRegistry
from process_function_calls import ProcessFunctionCalls
from workspace_manager import WorkspaceManager
from tool_call_log_manager import ToolCallLogManager
from message_menu_manager import MessageMenuManager
from image_attachment_manager import ImageAttachmentManager
from canvas_snapshot import CanvasSnapshotter
from slash_command_handler import SlashCommandHandler
from command_autocomplete import CommandAutocomplete
from tts_ui_manager import TTSUIManager
from chat_ui_manager import ChatUIManager
from chat_persistence_manager import ChatPersistenceManager
from turn_metrics import TurnMetricsCollector, short_model_name, turn_outcome
from managers.action_trace_collector import ActionTraceCollector

if TYPE_CHECKING:
    from canvas import Canvas

# Shown in a tab whose turn the server dropped because of a stop, a new message or a
# new conversation in another tab: its partial reply stays here, but not on the server.
ABANDONED_ELSEWHERE_NOTE = "This reply was stopped because the conversation changed elsewhere."
# Shown when the model is changed while a reply runs: the turn keeps its model.
MODEL_CHANGE_DEFERRED_NOTE = "{new} will answer from your next message; this reply finishes with {current}."
# Shown when the model is changed back to the turn's own model before the reply ends.
MODEL_CHANGE_UNDONE_NOTE = "{current} will keep answering after this reply."


class AIInterface:
    """Communication bridge between the mathematical canvas and backend AI services.

    Orchestrates the complete interaction cycle from user input to AI response processing,
    including function call execution, state management, and visual presentation.

    Attributes:
        canvas (Canvas): Mathematical canvas for visualization and state management
        workspace_manager (WorkspaceManager): Handles workspace persistence operations
        is_processing (bool): Tracks whether an AI request is currently being processed
        available_functions (dict): Registry of all functions available to the AI
        undoable_functions (tuple): Functions that support undo/redo operations
        _chat_ui (ChatUIManager): Manages chat message rendering and streaming display
    """

    # The server's id of the running turn (its first stream event); sent back with a stop or
    # client timeout so the server abandons that turn only, never the next one.
    _server_turn: Optional[str] = None
    # Per-turn request cap (None: no cap) and requests sent in the turn so far; the
    # scenario harness sets the cap so a looping model cannot send more.
    _turn_request_limit: Optional[int] = None
    _turn_requests_sent: int = 0
    # Per-turn response timeout overriding AI_RESPONSE_TIMEOUT_MS and REASONING_TIMEOUT_MS.
    _turn_timeout_ms: Optional[int] = None
    # Model the running turn started with; every request of the turn goes to it, so a
    # model picked mid-turn answers from the next message.
    _turn_model: Optional[str] = None
    # Model the last mid-turn note named (None: no note yet), so a note is only added
    # when the model that will answer next changes.
    _announced_model: Optional[str] = None

    def __init__(self, canvas: "Canvas") -> None:
        """Initialize the AI interface with canvas integration and function registry.

        Sets up all necessary components for AI communication including function mappings,
        workspace management, and markdown processing capabilities.

        Args:
            canvas (Canvas): The mathematical canvas instance to interact with
        """
        self.canvas: "Canvas" = canvas
        self.workspace_manager: WorkspaceManager = WorkspaceManager(canvas)
        self.is_processing: bool = False  # Track whether we're processing an AI response
        self._stop_requested = False
        self._tests_running = False
        self._stop_tests_requested = False
        self.available_functions: Dict[str, Any] = FunctionRegistry.get_available_functions(
            canvas, self.workspace_manager, self
        )
        self.undoable_functions: tuple[str, ...] = FunctionRegistry.get_undoable_functions()
        # Slash command handler for local commands
        self.slash_command_handler: SlashCommandHandler = SlashCommandHandler(canvas, self.workspace_manager, self)
        # Command autocomplete popup (initialized lazily when DOM is ready)
        self.command_autocomplete: Optional[CommandAutocomplete] = None
        # Tool call log state (delegated to ToolCallLogManager)
        self._tool_call_log = ToolCallLogManager()
        # Timeout state
        self._response_timeout_id: Optional[int] = None
        # TTS UI (delegated to TTSUIManager)
        self._tts_ui = TTSUIManager(on_system_message=self._print_system_message_in_chat)
        # Chat message menu (delegated to MessageMenuManager)
        self._message_menu = MessageMenuManager(
            on_read_aloud=self._tts_ui.handle_read_aloud,
            on_tts_settings=self._tts_ui.show_settings_modal,
        )
        # Image attachment state (delegated to ImageAttachmentManager)
        self._image_attachment = ImageAttachmentManager(
            on_system_message=self._print_system_message_in_chat,
        )
        # Browser-side canvas capture for vision requests
        self._canvas_snapshotter = CanvasSnapshotter()
        # Bumped by each new user send and by Stop, so a late snapshot callback can tell it is stale
        self._send_token: int = 0
        # Chat UI (delegated to ChatUIManager)
        self._chat_ui = ChatUIManager(
            message_menu=self._message_menu,
            tool_call_log=self._tool_call_log,
            on_image_click=self._image_attachment.show_modal,
            on_start_timeout=lambda use_reasoning: self._start_response_timeout(use_reasoning_timeout=use_reasoning),
        )
        # Chat saved with workspaces and restored on user loads (delegated to ChatPersistenceManager)
        self._chat_persistence = ChatPersistenceManager(
            self._chat_ui,
            get_model_id=lambda: str(document["ai-model-selector"].value),
        )
        self.workspace_manager.set_chat_persistence(self._chat_persistence)
        # Message recovery state
        self._last_user_message: str = ""  # Buffered message for recovery on error
        # Action trace collector for deterministic tool-execution logs
        self._trace_collector: ActionTraceCollector = ActionTraceCollector()
        self._register_trace_js_api()
        # Per-turn response metrics (model, latency, tokens/s, tool calls) for the chat footer and benchmarks
        self._register_metrics_js_api()

    def _register_trace_js_api(self) -> None:
        """Expose trace inspection functions on the browser ``window`` object."""

        def _safe_json_to_js(data: Any) -> Any:
            """Serialize Python data to a JS object, returning error dict on failure.

            Values JSON cannot express become strings, as in the scenario hooks: Brython's
            json rejects tuples (analyze_graph's MST edges), which would otherwise make
            the whole trace list unreadable.
            """
            try:
                return window.JSON.parse(json.dumps(data, default=str))
            except Exception as exc:
                return window.JSON.parse(json.dumps({"error": str(exc)}))

        window.getActionTraces = lambda: _safe_json_to_js(self._trace_collector.export_traces_json())
        window.getLastActionTrace = lambda: _safe_json_to_js(self._trace_collector.get_last_trace_json())
        window.clearActionTraces = lambda: self._trace_collector.clear()
        window.replayLastTrace = lambda: _safe_json_to_js(
            self._trace_collector.replay_last_trace(
                self.available_functions,
                self.undoable_functions,
                self.canvas,
            )
        )

    @property
    def _turn_metrics(self) -> TurnMetricsCollector:
        """Per-turn response metrics collector, created on first use."""
        collector = getattr(self, "_turn_metrics_collector", None)
        if collector is None:
            collector = TurnMetricsCollector(clock_ms=lambda: float(window.performance.now()))
            self._turn_metrics_collector = collector
        return cast(TurnMetricsCollector, collector)

    def _register_metrics_js_api(self) -> None:
        """Expose turn metrics on ``window`` as JSON strings, like ``getMatHudTestResults``.

        - ``getMatHudLastTurnMetrics()``: the last completed turn, or ``null``.
        - ``getMatHudTurnMetricsHistory()``: recent completed turns, oldest first.
        - ``clearMatHudTurnMetrics()``: forget the history.
        """

        def _to_json(data: Any) -> str:
            try:
                return json.dumps(data)
            except Exception as exc:
                return json.dumps({"error": str(exc)})

        window.getMatHudLastTurnMetrics = lambda: _to_json(self._turn_metrics.last_turn())
        window.getMatHudTurnMetricsHistory = lambda: _to_json(self._turn_metrics.history())
        window.clearMatHudTurnMetrics = lambda: self._turn_metrics.clear()

    def run_tests(self) -> Dict[str, Any]:
        """Run unit tests for the AIInterface class and return results to the AI as the function result."""
        try:
            from test_runner import TestRunner

            test_runner = TestRunner(self.canvas, self.available_functions, self.undoable_functions)

            # Run tests and get formatted results in one step
            results = test_runner.run_tests()
            return cast(Dict[str, Any], test_runner.format_results_for_ai(results))
        except ImportError as e:
            print(f"Test runner not available: {e}")
            return cast(
                Dict[str, Any],
                {
                    "tests_run": 0,
                    "failures": 0,
                    "errors": 1,
                    "failing_tests": [],
                    "error_tests": [{"test": "Test Runner Import", "error": f"Could not import test runner: {e}"}],
                },
            )

    def compare_canvas_state(self) -> None:
        """Send current canvas state to debug endpoint and log full-vs-summary output."""
        try:
            payload = json.dumps({"canvas_state": self.canvas.get_canvas_state()})
            req = ajax.ajax(timeout=20000)
            req.bind("complete", self._on_compare_canvas_state_complete)
            req.bind("error", self._on_compare_canvas_state_error)
            req.open("POST", "/api/debug/canvas-state-comparison", True)
            req.set_header("content-type", "application/json")
            req.send(payload)
            console.log("[MatHud] Requested canvas-state comparison...")
        except Exception as e:
            console.error(f"[MatHud] Failed to request canvas-state comparison: {e}")

    def _on_compare_canvas_state_complete(self, req: Any) -> None:
        """Handle debug canvas-state comparison response."""
        try:
            if int(getattr(req, "status", 0) or 0) >= 400:
                console.error(f"[MatHud] compareCanvasState failed: HTTP {req.status} {req.text}")
                return

            raw = json.loads(str(getattr(req, "text", "")))
            data = raw.get("data", {}) if isinstance(raw, dict) else {}
            metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
            full_state = data.get("full", {}) if isinstance(data, dict) else {}
            summary_state = data.get("summary", {}) if isinstance(data, dict) else {}

            console.log("=== Canvas State Comparison ===")
            console.log(
                "Full state:",
                f"{metrics.get('full_bytes', 0)} bytes (~{metrics.get('full_estimated_tokens', 0)} tokens)",
            )
            console.log(
                "Summary:",
                f"{metrics.get('summary_bytes', 0)} bytes (~{metrics.get('summary_estimated_tokens', 0)} tokens)",
            )
            console.log("Reduction:", f"{metrics.get('reduction_pct', 0.0)}%")
            console.log("Full state object:")
            console.log(full_state)
            console.log("Summary state object:")
            console.log(summary_state)
        except Exception as e:
            console.error(f"[MatHud] Failed to parse canvas-state comparison response: {e}")

    def _on_compare_canvas_state_error(self, req: Any) -> None:
        """Handle debug canvas-state comparison request error."""
        try:
            status = getattr(req, "status", "unknown")
            text = getattr(req, "text", "")
            console.error(f"[MatHud] compareCanvasState request error ({status}): {text}")
        except Exception:
            console.error("[MatHud] compareCanvasState request error")

    async def run_tests_async(
        self,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Run unit tests asynchronously, yielding to browser between test classes.

        Args:
            should_stop: Optional callback that returns True if tests should be stopped.
        """
        try:
            from test_runner import TestRunner

            test_runner = TestRunner(self.canvas, self.available_functions, self.undoable_functions)

            # Run tests asynchronously and get formatted results
            results = await test_runner.run_tests_async(should_stop=should_stop)
            return cast(Dict[str, Any], test_runner.format_results_for_ai(results))
        except ImportError as e:
            print(f"Test runner not available: {e}")
            return cast(
                Dict[str, Any],
                {
                    "tests_run": 0,
                    "failures": 0,
                    "errors": 1,
                    "failing_tests": [],
                    "error_tests": [{"test": "Test Runner Import", "error": f"Could not import test runner: {e}"}],
                },
            )

    def initialize_autocomplete(self) -> None:
        """Initialize the command autocomplete popup.

        Should be called after the DOM is ready (chat-input element exists).
        This is typically called from main.py after event bindings are set up.
        """
        try:
            if "chat-input" in document:
                input_element = document["chat-input"]
                self.command_autocomplete = CommandAutocomplete(
                    input_element,
                    self.slash_command_handler,
                )
        except Exception as e:
            print(f"Error initializing command autocomplete: {e}")

    def initialize_image_attachment(self) -> None:
        """Initialize image attachment functionality (delegates to ImageAttachmentManager)."""
        self._image_attachment.initialize()

    def trigger_file_picker(self) -> None:
        """Programmatically trigger the file picker for image attachment."""
        self._image_attachment.trigger_file_picker()

    def _store_results_in_canvas_state(self, call_results: Dict[str, Any]) -> None:
        """Store valid function call results in the canvas state, skipping special cases and formatting values."""
        if not ProcessFunctionCalls.validate_results(call_results):
            return

        for key, value in call_results.items():
            # Skip storing workspace management functions and test results in computations
            if (
                key.startswith("list_workspaces")
                or key.startswith("save_workspace")
                or key.startswith("load_workspace")
                or key.startswith("run_tests")
            ):
                continue

            if not ProcessFunctionCalls.is_successful_result(value):
                continue

            # DISABLED: Saving basic calculations to canvas state (takes up too many tokens, not useful info to store)
            # self.canvas.add_computation(
            #     expression=key,  # The key is already the expression
            #     result=formatted_value
            # )

    def _print_ai_message_in_chat(self, ai_message: str) -> None:
        """Print an AI message to the chat history (delegates to ChatUIManager)."""
        self._chat_ui.print_ai_message(ai_message)

    def _on_stream_log(self, event_obj: Any) -> None:
        """Handle a server log event: output to browser console with appropriate level."""
        try:
            event = self._normalize_stream_event(event_obj)
            level = event.get("level", "info")
            message = event.get("message", "")
            source = event.get("source", "")

            prefix = f"[Server{':' + source if source else ''}]"
            full_message = f"{prefix} {message}"

            if level == "error":
                console.error(full_message)
            elif level == "warning":
                console.warn(full_message)
            else:
                console.log(full_message)
        except Exception as e:
            print(f"Error handling server log event: {e}")

    def _on_stream_reasoning(self, text: str) -> None:
        """Handle a reasoning token (delegates to ChatUIManager)."""
        self._chat_ui.on_stream_reasoning(text)

    def _on_stream_token(self, text: str) -> None:
        """Handle a streamed token (delegates to ChatUIManager)."""
        self._chat_ui.on_stream_token(text)

    def _finalize_stream_message(self, final_message: Optional[str] = None) -> None:
        """Finalize the streamed message (delegates to ChatUIManager)."""
        self._chat_ui.finalize_stream(final_message)

    def _remove_empty_response_container(self) -> None:
        """Remove empty response container (delegates to ChatUIManager)."""
        self._chat_ui.remove_empty_container()

    def _on_stream_final(self, event_obj: Any, turn_token: Optional[int] = None) -> None:
        """Handle the final event from the streaming response.

        ``turn_token`` is the metrics turn that sent the request; a late event of
        an earlier, stopped or timed-out turn is ignored.
        """
        if self._is_stale_response(turn_token, "final event"):
            return
        try:
            event = self._normalize_stream_event(event_obj)

            finish_reason = event.get("finish_reason", "stop")
            ai_tool_calls = event.get("ai_tool_calls", [])
            ai_message = event.get("ai_message", "")
            error_details = event.get("error_details", "")
            self._turn_metrics.record_request(event.get("metrics"), turn_token)

            # Log error details to console for debugging
            if finish_reason == "error":
                console.error(f"[AI Error] {error_details or ai_message}")
            if finish_reason == "abandoned":
                self._end_abandoned_turn(turn_token)
                return

            if not self._should_run_tools(finish_reason, ai_tool_calls):
                if not self._chat_ui.stream_buffer and ai_message:
                    self._chat_ui.stream_buffer = ai_message
                outcome = turn_outcome(finish_reason)
                error_source = (event.get("error_source") or "provider") if outcome == "error" else None
                self._chat_ui.pending_turn_metrics = self._turn_metrics.finish_turn(outcome, turn_token, error_source)
                self._finalize_stream_message(ai_message or None)
                # Restore user message on error so they can retry
                if finish_reason == "error":
                    self._restore_user_message_on_error()
                else:
                    # Clear recovery buffer on successful completion
                    self._last_user_message = ""
                self._enable_send_controls()
                return

            # The batch's results would need one more request than the turn's cap allows:
            # end the turn before running it, so the canvas matches what the model was told.
            if self._request_limit_reached():
                self._end_turn_at_request_limit(turn_token)
                return

            # Processing tool calls - keep the "Thinking..." container visible
            # It will be removed/updated when the final response arrives
            try:
                batch = self.execute_tool_batch(ai_tool_calls, turn_token)
                call_results = batch["call_results"]
                if self._chat_ui.stream_container is None:
                    self._chat_ui.ensure_stream_element()
                self._tool_call_log.ensure_element(self._chat_ui.stream_container, self._chat_ui.stream_content)
                self._tool_call_log.add_entries(ai_tool_calls, call_results)

                if self._stop_requested:
                    self._turn_metrics.finish_turn("stopped", turn_token)
                    self._finalize_stream_message()
                    self._print_system_message_in_chat("Generation stopped.")
                    self._enable_send_controls()
                    return

                trace_summary = self._trace_summary(batch["trace"])

                # Reset timeout with extended duration - AI needs time to process tool results
                self._start_response_timeout(use_reasoning_timeout=True)
                # Mark that we need a newline separator before the next text
                if self._chat_ui.stream_buffer.strip():
                    self._chat_ui.needs_continuation_separator = True
                self._send_prompt_to_ai(
                    None,
                    ProcessFunctionCalls.serialize_tool_call_results(ai_tool_calls, batch["traced_calls"]),
                    canvas_state=batch["state_after"],
                    action_trace=trace_summary,
                )
            except Exception as e:
                print(f"Error processing streamed tool calls: {e}")
                self._turn_metrics.finish_turn("error", turn_token, "client")
                self._enable_send_controls()
        except Exception as e:
            print(f"Error handling stream final: {e}")
            self._turn_metrics.finish_turn("error", turn_token, "client")
            self._enable_send_controls()

    def _end_abandoned_turn(self, turn_token: Optional[int]) -> None:
        """End a turn whose reply the server dropped (stopped elsewhere, conversation reset).

        The text streamed so far stays in this chat although the server did not keep
        it, so a short note says why the reply ended. This tab's own stops and
        timeouts abort the stream first and never get here.
        """
        self._turn_metrics.finish_turn("abandoned", turn_token)
        self._finalize_stream_message()
        if not self._stop_requested:
            self._print_system_message_in_chat(ABANDONED_ELSEWHERE_NOTE)
        self._enable_send_controls()

    def _on_stream_turn(self, event_obj: Any, turn_token: Optional[int] = None) -> None:
        """Remember the server's id of the running turn (sent back with a stop or timeout)."""
        if self._is_stale_response(turn_token, "turn event"):
            return
        try:
            turn = self._normalize_stream_event(event_obj).get("turn")
            if isinstance(turn, str) and turn:
                self._server_turn = turn
        except Exception as e:
            print(f"Error reading the turn event: {e}")

    def execute_tool_batch(self, tool_calls: Any, turn_token: Optional[int] = None) -> Dict[str, Any]:
        """Run one batch of tool calls the way a model's batch runs.

        Executes the calls through ``ProcessFunctionCalls.get_results_traced``, stores
        the results, records them in the current turn's metrics and stores an action
        trace, also when execution raises (the exception is then re-raised). The
        scenario hook ``runMatHudToolCalls`` calls this same method, so replayed and
        model batches take one code path.

        Tracing is diagnostic: a failure to build or store the trace is logged and
        leaves ``trace`` as None; it never fails the batch or the turn.

        Returns:
            Dict with ``call_results``, ``traced_calls``, ``state_after`` and ``trace``.
        """
        state_before = self.canvas.get_canvas_state()
        t0 = window.performance.now()
        traced_calls: list[Dict[str, Any]] = []
        try:
            call_results, traced_calls = ProcessFunctionCalls.get_results_traced(
                tool_calls,
                self.available_functions,
                self.undoable_functions,
                self.canvas,
            )
            self._store_results_in_canvas_state(call_results)
            self._turn_metrics.record_tool_results(traced_calls, turn_token)
        except Exception:
            try:
                self._store_batch_trace(state_before, self.canvas.get_canvas_state(), traced_calls, t0)
            except Exception:
                pass
            raise
        state_after = self.canvas.get_canvas_state()
        trace = self._store_batch_trace(state_before, state_after, traced_calls, t0)
        return {
            "call_results": call_results,
            "traced_calls": traced_calls,
            "state_after": state_after,
            "trace": trace,
        }

    def _store_batch_trace(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        traced_calls: list[Dict[str, Any]],
        t0: float,
    ) -> Optional[Dict[str, Any]]:
        """Build and store the action trace of a batch that started at ``t0``; None if tracing fails."""
        try:
            total_ms = window.performance.now() - t0
            trace = self._trace_collector.build_trace(state_before, state_after, traced_calls, total_ms)
            self._trace_collector.store(trace)
            return cast(Dict[str, Any], trace)
        except Exception as exc:
            try:
                console.warn(f"[MatHud] Could not store the action trace: {exc}")
            except Exception:
                pass
            return None

    def _trace_summary(self, trace: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Compact trace summary for the server log, or None when there is no trace or it cannot be built."""
        if trace is None:
            return None
        try:
            return cast(Dict[str, Any], self._trace_collector.build_compact_summary(trace))
        except Exception as exc:
            try:
                console.warn(f"[MatHud] Could not summarize the action trace: {exc}")
            except Exception:
                pass
            return None

    def _on_stream_error(self, err: Any, turn_token: Optional[int] = None) -> None:
        """Handle streaming errors and re-enable controls."""
        if self._is_stale_response(turn_token, "stream error"):
            return
        error_message = self._format_stream_error(err)
        print(f"Streaming error: {error_message}")
        try:
            console.error("Streaming error", err)
        except Exception:
            pass
        self._turn_metrics.finish_turn("error", turn_token, self._stream_error_source(err))
        self._restore_user_message_on_error()
        self._enable_send_controls()

    @staticmethod
    def _stream_error_source(err: Any) -> str:
        """``server`` when the route answered with an HTTP error status, else ``transport``.

        The stream helper passes ``{"message", "status"}`` for a response that is not
        OK; a network failure has no status.
        """
        try:
            status = err.get("status") if isinstance(err, dict) else getattr(err, "status", None)
            if isinstance(status, (int, float)) and 400 <= status < 600:
                return "server"
        except Exception:
            pass
        return "transport"

    def _is_stale_response(self, turn_token: Optional[int], kind: str) -> bool:
        """True when a response belongs to an earlier turn or arrives after its turn ended.

        The non-streaming request cannot be aborted, and any late callback would
        otherwise finalize the chat, run tool calls or re-enable the controls of
        the turn in progress. Callers without a token (tests, hooks) are never stale.
        """
        if turn_token is None:
            return False
        if turn_token == self._turn_metrics.turn_token and self.is_processing:
            return False
        print(f"Ignoring a late {kind} from a turn that already ended.")
        return True

    def _format_stream_error(self, err: Any) -> str:
        """Convert a streaming error object into a readable string."""
        try:
            if err is None:
                return "Unknown error"
            if isinstance(err, str):
                return err
            # Brython-wrapped JS Error objects expose message and stack
            message = getattr(err, "message", None)
            stack = getattr(err, "stack", None)
            if message:
                if stack and message not in stack:
                    return cast(str, f"{message} | stack: {stack}")
                return cast(str, message)
            # Fall back to toString() if available
            to_string = getattr(err, "toString", None)
            if callable(to_string):
                return cast(str, to_string())
            return str(err)
        except Exception as format_exc:
            return f"Error while formatting streaming error: {format_exc}"

    def _restore_user_message_on_error(self) -> None:
        """Restore the last user message to the input field after an error.

        This allows the user to retry sending the same message without
        having to retype it. Also applies a brief visual flash to indicate
        the message was restored due to an error.
        """
        if not self._last_user_message:
            return
        try:
            chat_input = document["chat-input"]
            chat_input.value = self._last_user_message
            # Apply visual error feedback
            chat_input.classList.add("error-flash")
            window.setTimeout(lambda: chat_input.classList.remove("error-flash"), 2000)
        except Exception as e:
            print(f"Error restoring user message: {e}")

    def _normalize_stream_event(self, event_obj: Any) -> Dict[str, Any]:
        """Convert JS objects or dicts into plain Python dicts."""
        try:
            if event_obj is None:
                return {}
            if isinstance(event_obj, dict):
                return event_obj
            try:
                return cast(Dict[str, Any], json.loads(window.JSON.stringify(event_obj)))
            except Exception:
                pass
            result = {}
            for key in [
                "type",
                "text",
                "ai_message",
                "ai_tool_calls",
                "finish_reason",
                "error_details",
                "error_source",
                "turn",
                "metrics",
                "level",
                "message",
                "source",
            ]:
                try:
                    result[key] = getattr(event_obj, key)
                except Exception:
                    pass
            return cast(Dict[str, Any], result)
        except Exception:
            return {}

    def _print_user_message_in_chat(
        self,
        user_message: str,
        images: Optional[list[str]] = None,
        record: bool = False,
    ) -> None:
        """Print a user message to the chat history (delegates to ChatUIManager).

        ``record`` keeps it in the transcript saved with workspaces (messages sent to the AI).
        """
        self._chat_ui.print_user_message(user_message, images, record=record)

    def _print_system_message_in_chat(self, message: str) -> None:
        """Print a system message to the chat history (delegates to ChatUIManager)."""
        self._chat_ui.print_system_message(message)

    def _debug_log_ai_response(self, ai_message: str, ai_function_calls: Any, finish_reason: str) -> None:
        """Log debug information about the AI response."""
        print(f"### AI message: {ai_message}")
        print(f"### AI function calls: {ai_function_calls}")
        print(f"### AI finish reason: {finish_reason}")

    @staticmethod
    def _should_run_tools(finish_reason: Any, tool_calls: Any) -> bool:
        """Whether a reply's tool calls run and the tool loop continues.

        Only a reply that ends in tool calls and carries some runs tools; every other
        ending (stop, error, length/truncated, refusal/filtered) is a final message.
        """
        return bool(tool_calls) and finish_reason in ("tool_calls", "function_call")

    def _disable_send_controls(self) -> None:
        """Switch send button to stop mode while processing and start a timeout."""
        try:
            self.is_processing = True
            self._stop_requested = False
            if "send-button" in document:
                btn = document["send-button"]
                if "disabled" in btn.attrs:
                    del btn.attrs["disabled"]
                btn.text = "Stop"
                btn.classList.add("stop-mode")
            self._start_response_timeout()
        except Exception as e:
            print(f"Error disabling send controls: {e}")

    def _enable_send_controls(self) -> None:
        """Restore send button to normal mode after processing and cancel the timeout."""
        try:
            self._cancel_response_timeout()
            self.is_processing = False
            self._stop_requested = False
            if "send-button" in document:
                btn = document["send-button"]
                btn.disabled = False
                btn.text = "Send"
                btn.classList.remove("stop-mode")
        except Exception as e:
            print(f"Error enabling send controls: {e}")

    def _start_response_timeout(self, use_reasoning_timeout: bool = False) -> None:
        """Start a timeout that will re-enable controls if no response is received.

        Args:
            use_reasoning_timeout: If True, use extended timeout for reasoning models
        """
        try:
            # Cancel any existing timeout first
            self._cancel_response_timeout()
            timeout_ms = REASONING_TIMEOUT_MS if use_reasoning_timeout else AI_RESPONSE_TIMEOUT_MS
            if self._turn_timeout_ms is not None:
                timeout_ms = self._turn_timeout_ms
            self._response_timeout_id = window.setTimeout(self._on_response_timeout, timeout_ms)
        except Exception as e:
            print(f"Error starting response timeout: {e}")

    def _cancel_response_timeout(self) -> None:
        """Cancel the response timeout if one is pending."""
        try:
            if self._response_timeout_id is not None:
                window.clearTimeout(self._response_timeout_id)
                self._response_timeout_id = None
        except Exception as e:
            print(f"Error cancelling response timeout: {e}")

    def _on_response_timeout(self) -> None:
        """Handle timeout - abort the stream, re-enable controls and show error message."""
        try:
            self._response_timeout_id = None
            if self.is_processing:
                print("AI response timeout - aborting stream and re-enabling send controls")
                self._turn_metrics.finish_turn("timeout")
                self._abort_current_stream()
                # The request goes on in the server: abandon it there, as a stop does, so
                # its late reply never reaches the conversation.
                self._save_partial_response(self._chat_ui.unsaved_reply_text() or "")
                self._print_ai_message_in_chat(
                    "⚠️ Request timed out. The AI is taking too long to respond. Please try again."
                )
                self._enable_send_controls()
        except Exception as e:
            print(f"Error handling response timeout: {e}")

    def _abort_current_stream(self) -> None:
        """Abort the current streaming connection if one is active."""
        try:
            if hasattr(window, "abortCurrentStream"):
                window.abortCurrentStream()
        except Exception as e:
            print(f"Error aborting stream: {e}")

    def stop_ai_processing(self) -> None:
        """Stop the current AI processing, abort the stream, and restore UI controls."""
        self._stop_requested = True
        self._send_token += 1  # drops a vision request still waiting for its snapshot
        self._turn_metrics.finish_turn("stopped")
        self._abort_current_stream()
        self._cancel_response_timeout()
        # Always notify the backend so it can clear stale conversation state
        # (e.g. previous_response_id pointing to unanswered tool calls).
        # The backend handles empty text gracefully.
        self._save_partial_response(self._chat_ui.unsaved_reply_text() or "")
        self._finalize_stream_message()
        self._print_system_message_in_chat("Generation stopped.")
        self._enable_send_controls()

    def _save_partial_response(self, partial_message: str) -> None:
        """Save interrupted partial response to the backend conversation history."""
        try:
            body: Dict[str, Any] = {"partial_message": partial_message}
            if self._server_turn:
                # Only this turn is abandoned, even if the next message is already on its way.
                body["turn"] = self._server_turn
            payload = json.dumps(body)
            ajax.post(
                "/save_partial_response",
                data=payload,
                headers={"Content-Type": "application/json"},
                oncomplete=lambda req: None,
                onerror=lambda req: print(f"Failed to save partial response: {req.status}"),
            )
        except Exception as e:
            print(f"Error saving partial response: {e}")

    def _process_ai_response(
        self, ai_message: str, tool_calls: Any, finish_reason: str, turn_token: Optional[int] = None
    ) -> None:
        self._debug_log_ai_response(ai_message, tool_calls, finish_reason)

        if not self._should_run_tools(finish_reason, tool_calls):
            outcome = turn_outcome(finish_reason)
            error_source = "provider" if outcome == "error" else None
            turn_metrics = self._turn_metrics.finish_turn(outcome, turn_token, error_source)
            self._chat_ui.print_ai_message(ai_message, turn_metrics=turn_metrics, record=True)
            self._enable_send_controls()
        else:
            # Text sent with the calls (e.g. a cut-off note) is shown before they run.
            if isinstance(ai_message, str) and ai_message.strip():
                self._chat_ui.print_ai_message(ai_message, record=True)
            if self._request_limit_reached():
                self._end_turn_at_request_limit(turn_token)
                return
            try:
                batch = self.execute_tool_batch(tool_calls, turn_token)
                trace_summary = self._trace_summary(batch["trace"])

                self._send_prompt_to_ai(
                    None,
                    ProcessFunctionCalls.serialize_tool_call_results(tool_calls, batch["traced_calls"]),
                    canvas_state=batch["state_after"],
                    action_trace=trace_summary,
                )
            except Exception as e:
                print(f"Error processing tool calls: {e}")
                traceback.print_exc()
                self._turn_metrics.finish_turn("error", turn_token, "client")
                self._enable_send_controls()  # Enable controls if there's an error

    def _on_error(self, request: Any, turn_token: Optional[int] = None) -> None:
        """Handle request errors and ensure send controls are re-enabled."""
        if self._is_stale_response(turn_token, "request error"):
            return
        print(f"Error: {request.status}, {request.text}")
        self._turn_metrics.finish_turn("error", turn_token, self._error_source_of(request))
        self._enable_send_controls()

    @staticmethod
    def _error_source_of(request: Any) -> str:
        """``server`` when the route reported an exception of its own, else ``transport``."""
        try:
            data = (request.json or {}).get("data") or {}
            if isinstance(data, dict) and data.get("error_source") == "server":
                return "server"
        except Exception:
            pass
        return "transport"

    def _on_complete(self, request: Any, turn_token: Optional[int] = None) -> None:
        """Handle request completion and process AI response."""
        if self._is_stale_response(turn_token, "response"):
            return
        try:
            if request.status == 200 or request.status == 0:
                # Extract data from the proper response structure
                response_data = request.json.get("data")
                if not response_data:
                    error_msg = request.json.get("message", "Invalid response format")
                    print(f"Error: {error_msg}")
                    self._turn_metrics.finish_turn("error", turn_token)
                    document["ai-response"].text = error_msg
                    self._enable_send_controls()
                    return

                ai_message = response_data.get("ai_message")
                ai_function_calls = response_data.get("ai_tool_calls")
                finish_reason = response_data.get("finish_reason")
                if isinstance(response_data.get("turn"), str):
                    self._server_turn = response_data.get("turn")
                if finish_reason == "abandoned":
                    self._end_abandoned_turn(turn_token)
                    return
                self._turn_metrics.record_request(response_data.get("metrics"), turn_token)

                # Parse the AI's response and create / delete drawables as needed
                self._process_ai_response(ai_message, ai_function_calls, finish_reason, turn_token)
            else:
                self._on_error(request, turn_token)
        except Exception as e:
            print(f"Error processing AI response: {e}")
            traceback.print_exc()
            self._turn_metrics.finish_turn("error", turn_token, "client")
            self._enable_send_controls()

    def _create_request_payload(
        self,
        prompt: Optional[str],
        action_trace: Optional[Dict[str, Any]] = None,
        turn: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create the JSON payload for the request.

        The vision snapshot, when there is one, travels inside the prompt JSON
        (``canvas_snapshot``), next to ``use_vision``. ``turn`` is the server's turn
        that tool results continue: the server drops them once that turn ended.
        """
        payload: Dict[str, Any] = {"message": prompt}
        if action_trace is not None:
            payload["action_trace"] = action_trace
        if turn:
            payload["turn"] = turn
        renderer_mode = getattr(self.canvas, "renderer_mode", None)
        if isinstance(renderer_mode, str):
            payload["renderer_mode"] = renderer_mode
        return payload

    def _make_request(self, payload: Dict[str, Any]) -> None:
        """Send an AJAX request with the given payload."""
        turn_token = self._turn_metrics.turn_token
        req = ajax.ajax()
        req.bind("complete", lambda request: self._on_complete(request, turn_token))
        req.bind("error", lambda request: self._on_error(request, turn_token))
        req.open("POST", "/send_message", True)
        req.set_header("content-type", "application/json")
        req.send(json.dumps(payload))

    def _start_streaming_request(self, payload: Dict[str, Any]) -> None:
        """Start a streaming request using a JS helper for Fetch streaming."""
        try:
            payload_json = json.dumps(payload)
            payload_js = window.JSON.parse(payload_json)
            # Don't reset any state here - all state management is done in _send_prompt_to_ai
            # This preserves intermediary text and reasoning content across tool call continuations
            # Call JS streaming helper with reasoning and log callbacks
            turn_token = self._turn_metrics.turn_token
            window.sendMessageStream(
                payload_js,
                self._on_stream_token,
                lambda event_obj: self._on_stream_final(event_obj, turn_token),
                lambda err: self._on_stream_error(err, turn_token),
                self._on_stream_reasoning,
                self._on_stream_log,
                lambda event_obj: self._on_stream_turn(event_obj, turn_token),
            )
        except Exception as e:
            print(f"Falling back to non-streaming request due to error: {e}")
            self._make_request(payload)

    def _send_request(
        self,
        prompt: Optional[str],
        action_trace: Optional[Dict[str, Any]] = None,
        turn: Optional[str] = None,
    ) -> None:
        payload = self._create_request_payload(prompt, action_trace=action_trace, turn=turn)
        self._start_streaming_request(payload)

    def _send_prompt_to_ai(
        self,
        user_message: Optional[str] = None,
        tool_call_results: Optional[str] = None,
        attached_images: Optional[list[str]] = None,
        canvas_state: Optional[Dict[str, Any]] = None,
        action_trace: Optional[Dict[str, Any]] = None,
    ) -> None:
        if tool_call_results is not None and self._request_limit_reached():
            # Backstop: the tool-call handlers end the turn before running a batch at the cap.
            self._end_turn_at_request_limit()
            return
        self._turn_requests_sent += 1
        if canvas_state is None:
            canvas_state = self.canvas.get_canvas_state()

        # Only use vision when we have a user message and no tool call results
        use_vision = document["vision-toggle"].checked and user_message is not None and tool_call_results is None

        prompt_json: Dict[str, Any] = {
            "canvas_state": canvas_state,
            "user_message": user_message,
            "tool_call_results": tool_call_results,
            "use_vision": use_vision,
            "ai_model": self.turn_model_id(),
        }

        # Include attached images if provided (works independently of vision toggle)
        if attached_images:
            prompt_json["attached_images"] = attached_images

        # For new user messages, reset all state including containers and buffers
        # For tool call results, preserve everything to keep intermediary text visible
        if user_message is not None and tool_call_results is None:
            self._chat_ui.request_start_time = window.Date.now()
            self._chat_ui.reset_streaming_state()
            self._turn_metrics.start_turn(user_message)

        if use_vision:
            # The snapshot may need an async image decode, so the request is sent from the callback.
            send_token = self._send_token
            self._canvas_snapshotter.capture(
                lambda snapshot: self._send_prompt_json_if_current(prompt_json, snapshot, action_trace, send_token)
            )
            return
        # Tool results continue the server's turn; the server drops them once it ended.
        turn = self._server_turn if tool_call_results is not None else None
        self._send_prompt_json(prompt_json, None, action_trace, turn)

    def turn_model_id(self) -> str:
        """The model this turn's requests go to: the one it started with, else the selected one.

        Turn tokens belong to a provider instance, so a turn's follow-ups must reach the
        model that started it; a model picked mid-turn answers from the next message.
        """
        if self.is_processing and self._turn_model:
            return self._turn_model
        return self._selected_model_id()

    @staticmethod
    def _selected_model_id() -> str:
        """The model chosen in the selector."""
        try:
            return str(document["ai-model-selector"].value)
        except Exception:
            return ""

    def on_model_selected(self, event: Any = None) -> None:
        """Tell the user which model answers their next message when it changes mid-turn.

        Switching back to the turn's own model says so, so an earlier note is not left
        promising a model that will not answer.
        """
        try:
            selected = self._selected_model_id()
            turn_model = self._turn_model
            if not self.is_processing or not turn_model:
                return
            if selected == (self._announced_model or turn_model):
                return
            self._announced_model = selected
            current = self._model_label(turn_model)
            if selected == turn_model:
                note = MODEL_CHANGE_UNDONE_NOTE.format(current=current)
            else:
                note = MODEL_CHANGE_DEFERRED_NOTE.format(new=self._model_label(selected), current=current)
            self._print_system_message_in_chat(note)
        except Exception as e:
            print(f"Error handling the model change: {e}")

    @staticmethod
    def _model_label(model_id: str) -> str:
        """The selector's label for a model, without the "(text only)" suffix."""
        try:
            for option in document["ai-model-selector"].options:
                if str(option.value) == model_id:
                    return str(option.text).replace(" (text only)", "")
        except Exception:
            pass
        return str(short_model_name(model_id))

    def _request_limit_reached(self) -> bool:
        """True when the turn has sent as many requests as its cap allows."""
        limit = self._turn_request_limit
        return limit is not None and self._turn_requests_sent >= limit

    def _end_turn_at_request_limit(self, turn_token: Optional[int] = None) -> None:
        """End the turn instead of sending one more request than its cap allows.

        The server's turn is abandoned like a stop's, so the tool calls of the last
        reply are not left awaiting results that will never be sent.
        """
        print(f"Request limit reached ({self._turn_request_limit} requests); ending the turn")
        self._turn_metrics.finish_turn("max_requests", turn_token)
        self._cancel_response_timeout()
        self._save_partial_response("")
        self._finalize_stream_message()
        self._print_system_message_in_chat("Stopped: the turn reached its request limit.")
        self._enable_send_controls()

    def _send_prompt_json_if_current(
        self,
        prompt_json: Dict[str, Any],
        canvas_snapshot: Optional[str],
        action_trace: Optional[Dict[str, Any]],
        send_token: int,
    ) -> None:
        """Send a vision request once its snapshot arrives, unless the turn was stopped or superseded.

        Sending a stale request would also abort the stream of the message that replaced it.
        """
        if send_token != self._send_token or self._stop_requested or not self.is_processing:
            print("Dropping a vision request whose turn was stopped before its snapshot was ready.")
            return
        self._send_prompt_json(prompt_json, canvas_snapshot, action_trace)

    def _send_prompt_json(
        self,
        prompt_json: Dict[str, Any],
        canvas_snapshot: Optional[str],
        action_trace: Optional[Dict[str, Any]],
        turn: Optional[str] = None,
    ) -> None:
        """Serialize the prompt (with the vision snapshot, if any) and send it."""
        if canvas_snapshot:
            prompt_json["canvas_snapshot"] = canvas_snapshot
        self._send_request(json.dumps(prompt_json), action_trace=action_trace, turn=turn)

    def send_user_message(
        self,
        message: str,
        request_limit: Optional[int] = None,
        response_timeout_ms: Optional[int] = None,
    ) -> None:
        """Sends a message as if the user typed it.

        If the message is a slash command (starts with "/"), it is executed
        locally without sending to the AI backend.

        Allows sending with just attached images (empty message).

        ``request_limit`` caps the model requests of this turn (the first one
        included): the turn ends instead of sending more. ``response_timeout_ms``
        replaces the client's response timeouts for this turn. Both are for the
        scenario harness; the chat sends with neither.
        """
        has_text = bool(message.strip())
        has_images = len(self._image_attachment.images) > 0

        # Need either text or images to send
        if self.is_processing or (not has_text and not has_images):
            return

        # Check for slash command (only if there's text)
        if has_text and self.slash_command_handler.is_slash_command(message):
            self._print_user_message_in_chat(message)
            result = self.slash_command_handler.execute(message)
            self._print_system_message_in_chat(result.message)
            return

        # Capture attached images before clearing
        images_to_send = list(self._image_attachment.images) if self._image_attachment.images else None

        # Use a default message for image-only sends
        display_message = message if has_text else "[Image attached]"
        ai_message = message if has_text else "What do you see in this image?"

        # Display the user message with images in chat
        self._print_user_message_in_chat(display_message, images=images_to_send, record=True)

        # Clear attached images after displaying (not after successful send)
        self._image_attachment.clear()

        # Regular AI flow
        self._turn_request_limit = request_limit
        self._turn_requests_sent = 0
        self._turn_timeout_ms = response_timeout_ms
        self._server_turn = None
        self._send_token += 1
        self._turn_model = self._selected_model_id()
        self._announced_model = None
        self._disable_send_controls()
        self._send_prompt_to_ai(ai_message, attached_images=images_to_send)

    def run_tests_action(self, event: Any) -> None:
        """Trigger the test suite directly on the client side (TEMPORARY).

        See documentation/development/removing_run_tests_button.md for removal instructions.
        """
        # If tests are running, stop them
        if self._tests_running:
            self._stop_tests_requested = True
            return

        # Start tests
        self._stop_tests_requested = False
        aio.run(self._execute_tests_async())

    async def _execute_tests_async(self) -> None:
        """Execute tests asynchronously using browser.aio."""
        run_tests_btn = document["run-tests-button"]

        try:
            self._tests_running = True

            # Switch to "Stop Tests" mode
            run_tests_btn.text = "Stop Tests"
            run_tests_btn.classList.add("stop-mode")

            # Disable Send button while tests run
            if "send-button" in document:
                document["send-button"].disabled = True

            self._print_user_message_in_chat("Run tests (direct execution)")

            # Run tests asynchronously with stop callback
            results = await self.run_tests_async(should_stop=lambda: self._stop_tests_requested)

            # Check if tests were stopped
            was_stopped = results.get("stopped", False)

            if was_stopped:
                summary = (
                    f"### Test Results (Stopped)\n\n"
                    f"- **Tests Run:** {results.get('tests_run', 0)}\n"
                    f"- **Failures:** {results.get('failures', 0)}\n"
                    f"- **Errors:** {results.get('errors', 0)}\n"
                    f"\n*Tests were stopped by user.*"
                )
            else:
                summary = (
                    f"### Test Results\n\n"
                    f"- **Tests Run:** {results.get('tests_run', 0)}\n"
                    f"- **Failures:** {results.get('failures', 0)}\n"
                    f"- **Errors:** {results.get('errors', 0)}\n"
                )

            if results.get("failing_tests"):
                summary += "\n#### Failures:\n"
                for fail in results["failing_tests"]:
                    summary += f"- **{fail['test']}**: {fail['error']}\n"

            if results.get("error_tests"):
                summary += "\n#### Errors:\n"
                for err in results["error_tests"]:
                    summary += f"- **{err['test']}**: {err['error']}\n"

            self._print_ai_message_in_chat(summary)

        except Exception as e:
            error_msg = f"Error running tests: {str(e)}"
            print(error_msg)
            self._print_ai_message_in_chat(error_msg)

        finally:
            # Restore buttons and reset stop flag
            self._tests_running = False
            self._stop_tests_requested = False
            run_tests_btn.text = "Run Tests"
            run_tests_btn.classList.remove("stop-mode")
            if "send-button" in document:
                document["send-button"].disabled = False

    def interact_with_ai(self, event: Any) -> None:
        if self.is_processing:
            self.stop_ai_processing()
            return

        # Get the user's message from the input field
        user_message = document["chat-input"].value.strip()
        has_images = len(self._image_attachment.images) > 0

        # Allow sending if there's text OR attached images
        if user_message or has_images:
            # Buffer message for recovery on error before clearing
            self._last_user_message = user_message
            document["chat-input"].value = ""
            self.send_user_message(user_message)

    def start_new_conversation(self, event: Any) -> None:
        """Saves the current workspace, resets the canvas and chat, and starts a new backend session."""
        # 0. Stop a turn still running, so its reply is not shown in the new chat
        if self.is_processing:
            self.stop_ai_processing()

        # 1. Save the current workspace (canvas and chat) automatically
        self.workspace_manager.save_workspace()

        # 2. Reset the client-side canvas
        self.canvas.clear()

        # 3. Clear the chat history UI and its transcript
        self._chat_ui.clear_chat()

        # 4. Call the backend to reset the AI conversation state
        req = ajax.ajax()
        req.open("POST", "/new_conversation", True)
        req.set_header("content-type", "application/json")
        req.send()
