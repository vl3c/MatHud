"""ActionTraceCollector — structured trace storage for AI tool-call batches.

Builds, stores, and exports action traces that record per-call results,
timing, and canvas state deltas.  Traces are kept in a bounded FIFO
(default 100 entries) with full canvas snapshots retained only on the
most recent trace to keep memory usage predictable.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from browser import window

if TYPE_CHECKING:
    TracedCall = Dict[str, Any]
    StateDelta = Dict[str, List[str]]
    ActionTrace = Dict[str, Any]

_MAX_TRACES = 100
_MAX_RESULT_STR_LEN = 500

# Functions that are not safe to replay (side-effects outside canvas state).
_NON_REPLAYABLE_FUNCTIONS = frozenset({
    "save_workspace", "load_workspace", "delete_workspace", "run_tests",
})


class ActionTraceCollector:
    """Collects and manages action traces for AI tool-call batches."""

    def __init__(self) -> None:
        self._traces: List["ActionTrace"] = []
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Building traces
    # ------------------------------------------------------------------

    def build_trace(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        traced_calls: List["TracedCall"],
        total_duration_ms: float,
    ) -> "ActionTrace":
        """Construct an ActionTrace dict from execution data.

        Args:
            state_before: Canvas state snapshot taken before execution.
            state_after: Canvas state snapshot taken after execution.
            traced_calls: Per-call trace records from ResultProcessor.
            total_duration_ms: Wall-clock time for the full batch.

        Returns:
            A fully populated ActionTrace dict (not yet stored).
        """
        self._counter += 1
        trace_id = f"{int(window.Date.now())}-{self._counter}"
        timestamp = window.Date.new().toISOString()

        state_delta = self.compute_state_delta(state_before, state_after)

        return {
            "trace_id": trace_id,
            "timestamp": timestamp,
            "tool_calls": traced_calls,
            "state_delta": state_delta,
            "total_duration_ms": round(total_duration_ms, 2),
            "canvas_state_before": state_before,
            "canvas_state_after": state_after,
        }

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(self, trace: "ActionTrace") -> None:
        """Append *trace* to the in-memory store (FIFO, capped at _MAX_TRACES).

        Full canvas snapshots are stripped from all but the latest trace.
        """
        # Strip snapshots from the previous latest trace
        if self._traces:
            prev = self._traces[-1]
            prev.pop("canvas_state_before", None)
            prev.pop("canvas_state_after", None)

        self._traces.append(trace)

        # Enforce FIFO cap
        if len(self._traces) > _MAX_TRACES:
            self._traces = self._traces[-_MAX_TRACES:]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_traces(self) -> List["ActionTrace"]:
        """Return the full list of stored traces."""
        return list(self._traces)

    def get_last_trace(self) -> Optional["ActionTrace"]:
        """Return a shallow copy of the most recent trace, or None if empty."""
        return dict(self._traces[-1]) if self._traces else None

    def clear(self) -> None:
        """Remove all stored traces."""
        self._traces.clear()
        self._counter = 0

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_traces_json(self) -> List[Dict[str, Any]]:
        """Return a JSON-serializable list with large result values truncated."""
        exported: List[Dict[str, Any]] = []
        for trace in self._traces:
            exported.append(self._make_exportable_trace(trace))
        return exported

    def get_last_trace_json(self) -> Optional[Dict[str, Any]]:
        """Return the latest trace in export-safe form, or None."""
        last = self.get_last_trace()
        if last is None:
            return None
        return self._make_exportable_trace(last)

    # ------------------------------------------------------------------
    # Compact summary (sent to server)
    # ------------------------------------------------------------------

    def build_compact_summary(self, trace: "ActionTrace") -> Dict[str, Any]:
        """Build a compact trace summary suitable for server logging.

        Includes per-call function name/timing/error but not full results
        or canvas state snapshots.
        """
        calls = trace.get("tool_calls", [])
        return {
            "trace_id": trace["trace_id"],
            "tool_count": len(calls),
            "error_count": sum(1 for c in calls if c.get("is_error")),
            "total_duration_ms": trace["total_duration_ms"],
            "state_delta": trace["state_delta"],
            "calls": [
                {
                    "function_name": c["function_name"],
                    "duration_ms": c["duration_ms"],
                    "is_error": c.get("is_error", False),
                }
                for c in calls
            ],
        }

    # ------------------------------------------------------------------
    # State delta computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_state_delta(
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> "StateDelta":
        """Compare two canvas state snapshots and return a StateDelta.

        The delta contains lists of drawable names that were added, removed,
        or modified between the two snapshots.
        """
        before_map = ActionTraceCollector._extract_drawable_map(before)
        after_map = ActionTraceCollector._extract_drawable_map(after)

        before_names = set(before_map.keys())
        after_names = set(after_map.keys())

        added = sorted(after_names - before_names)
        removed = sorted(before_names - after_names)

        modified: List[str] = []
        for name in sorted(before_names & after_names):
            before_json = json.dumps(before_map[name], sort_keys=True)
            after_json = json.dumps(after_map[name], sort_keys=True)
            if before_json != after_json:
                modified.append(name)

        return {"added": added, "removed": removed, "modified": modified}

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay_trace(
        self,
        trace: "ActionTrace",
        available_functions: Dict[str, Any],
        undoable_functions: Tuple[str, ...],
        canvas: Any,
    ) -> Dict[str, Any]:
        """Re-execute tool calls from *trace* and compare results.

        Skips non-idempotent functions (workspace I/O, tests).
        Uses the public ResultProcessor.get_results API per call.

        Returns:
            Dict with ``match_report`` (list of per-call dicts) and
            ``skipped`` (list of skipped function names).
        """
        from result_processor import ResultProcessor

        match_report: List[Dict[str, Any]] = []
        skipped: List[str] = []

        for tc in trace.get("tool_calls", []):
            fn = tc["function_name"]
            if fn in _NON_REPLAYABLE_FUNCTIONS:
                skipped.append(fn)
                continue

            args = dict(tc.get("arguments", {}))
            call = {"function_name": fn, "arguments": args}

            is_error = False
            new_result: Any = None
            try:
                results = ResultProcessor.get_results(
                    [call], available_functions, undoable_functions, canvas,
                )
                # Get the single result value (first entry in the dict)
                if results:
                    new_result = next(iter(results.values()))
                if isinstance(new_result, str) and new_result.startswith("Error"):
                    is_error = True
            except Exception as e:
                new_result = f"Error: {e}"
                is_error = True

            original = tc.get("result")
            matched = self._results_match(original, new_result)
            match_report.append({
                "function_name": fn,
                "matched": matched,
                "original_result": self._truncate(original),
                "new_result": self._truncate(new_result),
                "is_error": is_error,
            })

        return {"match_report": match_report, "skipped": skipped}

    def replay_last_trace(
        self,
        available_functions: Dict[str, Any],
        undoable_functions: Tuple[str, ...],
        canvas: Any,
    ) -> Dict[str, Any]:
        """Convenience: replay the most recent trace."""
        last = self.get_last_trace()
        if last is None:
            return {"error": "No traces stored"}
        return self.replay_trace(last, available_functions, undoable_functions, canvas)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_drawable_map(state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract {drawable_name: serialized_state} from a canvas state dict."""
        result: Dict[str, Any] = {}
        if not isinstance(state, dict):
            return result
        for category, drawables in state.items():
            if not isinstance(drawables, dict):
                continue
            for name, drawable_state in drawables.items():
                if isinstance(drawable_state, dict):
                    result[name] = drawable_state
        return result

    @staticmethod
    def _truncate(value: Any) -> Any:
        """Truncate string values to _MAX_RESULT_STR_LEN for export."""
        if isinstance(value, str) and len(value) > _MAX_RESULT_STR_LEN:
            return value[:_MAX_RESULT_STR_LEN] + "..."
        return value

    @staticmethod
    def _results_match(original: Any, new: Any) -> bool:
        """Loose comparison of two results for replay matching."""
        try:
            return json.dumps(original, sort_keys=True) == json.dumps(new, sort_keys=True)
        except (TypeError, ValueError):
            return str(original) == str(new)

    def _make_exportable_trace(self, trace: "ActionTrace") -> Dict[str, Any]:
        """Create an export-safe copy of a trace with truncated results."""
        exported: Dict[str, Any] = {
            "trace_id": trace["trace_id"],
            "timestamp": trace["timestamp"],
            "state_delta": trace["state_delta"],
            "total_duration_ms": trace["total_duration_ms"],
        }
        exported_calls: List[Dict[str, Any]] = []
        for tc in trace.get("tool_calls", []):
            ec = dict(tc)
            ec["result"] = self._truncate(tc.get("result"))
            exported_calls.append(ec)
        exported["tool_calls"] = exported_calls

        # Include full snapshots only if present (latest trace)
        if "canvas_state_before" in trace:
            exported["canvas_state_before"] = trace["canvas_state_before"]
        if "canvas_state_after" in trace:
            exported["canvas_state_after"] = trace["canvas_state_after"]

        return exported
