"""Base telemetry class shared by SVG and Canvas 2D renderers.

Provides common timing, counting, and snapshot logic so that each renderer's
telemetry subclass only needs to define its per-drawable bucket schema.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from browser import window


class BaseRendererTelemetry:
    """Performance telemetry collector base for rendering backends.

    Tracks timing metrics for plan building and application, cache statistics,
    and per-drawable performance data. Subclasses override ``_new_drawable_bucket``
    to add renderer-specific counters.

    Attributes:
        _phase_totals: Cumulative timing for each rendering phase.
        _phase_counts: Operation counts for each phase.
        _per_drawable: Per-drawable type timing breakdown.
        _adapter_events: Event counts from the primitive adapter.
        _frames: Total frames rendered since last reset.
    """

    def __init__(self) -> None:
        """Initialize telemetry with zeroed counters."""
        self.reset()

    def reset(self) -> None:
        """Reset all telemetry counters to zero."""
        self._phase_totals: Dict[str, float] = {
            "plan_build_ms": 0.0,
            "plan_apply_ms": 0.0,
            "cartesian_plan_build_ms": 0.0,
            "cartesian_plan_apply_ms": 0.0,
        }
        self._phase_counts: Dict[str, int] = {
            "plan_build_count": 0,
            "plan_apply_count": 0,
            "cartesian_plan_count": 0,
            "plan_miss_count": 0,
            "plan_skip_count": 0,
        }
        self._per_drawable: Dict[str, Dict[str, float]] = {}
        self._adapter_events: Dict[str, int] = {}
        self._frames: int = 0
        self._max_batch_depth: int = 0

    def begin_frame(self) -> None:
        """Signal the start of a new frame for counting purposes."""
        self._frames += 1

    def end_frame(self) -> None:
        """Signal the end of a frame (currently no-op)."""
        pass

    def _now(self) -> float:
        """Get current timestamp in milliseconds using performance.now() if available."""
        try:
            perf = getattr(window, "performance", None)
            if perf is not None:
                return float(perf.now())
        except Exception:
            pass
        return time.time() * 1000.0

    def mark_time(self) -> float:
        """Record and return the current timestamp for duration measurement."""
        return self._now()

    def elapsed_since(self, start: float) -> float:
        """Calculate milliseconds elapsed since a marked timestamp."""
        return max(self._now() - start, 0.0)

    # ------------------------------------------------------------------
    # Per-drawable bucket
    # ------------------------------------------------------------------

    def _new_drawable_bucket(self) -> Dict[str, float]:
        """Return a fresh per-drawable counters dict.

        Subclasses may override to add renderer-specific keys.
        """
        return {
            "plan_build_ms": 0.0,
            "plan_apply_ms": 0.0,
            "plan_build_count": 0,
            "plan_apply_count": 0,
            "plan_miss_count": 0,
            "plan_skip_count": 0,
        }

    def _drawable_bucket(self, name: str) -> Dict[str, float]:
        """Get or create the telemetry bucket for a drawable type."""
        bucket = self._per_drawable.get(name)
        if bucket is None:
            bucket = self._new_drawable_bucket()
            self._per_drawable[name] = bucket
        return bucket

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record_plan_build(self, name: str, duration_ms: float, *, cartesian: bool = False) -> None:
        """Record time spent building a render plan."""
        self._phase_totals["plan_build_ms"] += duration_ms
        self._phase_counts["plan_build_count"] += 1
        bucket = self._drawable_bucket(name)
        bucket["plan_build_ms"] += duration_ms
        bucket["plan_build_count"] += 1
        if cartesian:
            self._phase_totals["cartesian_plan_build_ms"] += duration_ms
            self._phase_counts["cartesian_plan_count"] += 1

    def record_plan_apply(self, name: str, duration_ms: float, *, cartesian: bool = False) -> None:
        """Record time spent applying a render plan."""
        self._phase_totals["plan_apply_ms"] += duration_ms
        self._phase_counts["plan_apply_count"] += 1
        bucket = self._drawable_bucket(name)
        bucket["plan_apply_ms"] += duration_ms
        bucket["plan_apply_count"] += 1
        if cartesian:
            self._phase_totals["cartesian_plan_apply_ms"] += duration_ms

    def record_plan_miss(self, name: str) -> None:
        """Record when a drawable could not be rendered via a plan."""
        self._phase_counts["plan_miss_count"] += 1
        bucket = self._drawable_bucket(name)
        bucket["plan_miss_count"] += 1

    def record_plan_skip(self, name: str) -> None:
        """Record when a plan was skipped due to being off-screen."""
        self._phase_counts["plan_skip_count"] += 1
        bucket = self._drawable_bucket(name)
        bucket["plan_skip_count"] += 1

    def record_adapter_event(self, name: str, amount: int = 1) -> None:
        """Record an event from the primitive adapter."""
        self._adapter_events[name] = self._adapter_events.get(name, 0) + amount

    def track_batch_depth(self, depth: int) -> None:
        """Track the maximum nested batch depth seen."""
        if depth > self._max_batch_depth:
            self._max_batch_depth = depth

    # ------------------------------------------------------------------
    # Snapshot / drain
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Get a copy of all current telemetry data without resetting."""
        adapter_events = dict(self._adapter_events)
        if self._max_batch_depth:
            adapter_events["max_batch_depth"] = self._max_batch_depth
        per_drawable = {name: dict(bucket) for name, bucket in self._per_drawable.items()}
        phase = dict(self._phase_totals)
        phase.update(self._phase_counts)
        return {
            "frames": self._frames,
            "phase": phase,
            "per_drawable": per_drawable,
            "adapter_events": adapter_events,
        }

    def drain(self) -> Dict[str, Any]:
        """Get all telemetry data and reset counters to zero."""
        snapshot = self.snapshot()
        self.reset()
        return snapshot
