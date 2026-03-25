"""Tests for BaseRendererTelemetry, Canvas2DTelemetry, and SvgTelemetry."""

from __future__ import annotations

import unittest

from rendering.base_telemetry import BaseRendererTelemetry
from rendering.canvas2d_renderer import Canvas2DTelemetry
from rendering.svg_renderer import SvgTelemetry


class TestBaseTelemetryInit(unittest.TestCase):
    """Verify that initialization calls reset and zeroes all counters."""

    def test_init_calls_reset(self) -> None:
        tel = BaseRendererTelemetry()
        self.assertEqual(tel._frames, 0)
        self.assertEqual(tel._max_batch_depth, 0)
        self.assertIsInstance(tel._phase_totals, dict)
        self.assertIsInstance(tel._phase_counts, dict)
        self.assertIsInstance(tel._per_drawable, dict)
        self.assertIsInstance(tel._adapter_events, dict)

    def test_init_phase_totals_zero(self) -> None:
        tel = BaseRendererTelemetry()
        for key in ("plan_build_ms", "plan_apply_ms",
                     "cartesian_plan_build_ms", "cartesian_plan_apply_ms"):
            self.assertEqual(tel._phase_totals[key], 0.0, f"{key} should be 0.0")

    def test_init_phase_counts_zero(self) -> None:
        tel = BaseRendererTelemetry()
        for key in ("plan_build_count", "plan_apply_count",
                     "cartesian_plan_count", "plan_miss_count", "plan_skip_count"):
            self.assertEqual(tel._phase_counts[key], 0, f"{key} should be 0")

    def test_init_per_drawable_empty(self) -> None:
        tel = BaseRendererTelemetry()
        self.assertEqual(len(tel._per_drawable), 0)

    def test_init_adapter_events_empty(self) -> None:
        tel = BaseRendererTelemetry()
        self.assertEqual(len(tel._adapter_events), 0)


class TestBaseTelemetryReset(unittest.TestCase):
    """Verify reset() zeroes all accumulated data."""

    def test_reset_clears_phase_totals(self) -> None:
        tel = BaseRendererTelemetry()
        tel._phase_totals["plan_build_ms"] = 42.0
        tel.reset()
        self.assertEqual(tel._phase_totals["plan_build_ms"], 0.0)

    def test_reset_clears_phase_counts(self) -> None:
        tel = BaseRendererTelemetry()
        tel._phase_counts["plan_build_count"] = 7
        tel.reset()
        self.assertEqual(tel._phase_counts["plan_build_count"], 0)

    def test_reset_clears_per_drawable(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_build("Circle", 1.0)
        self.assertIn("Circle", tel._per_drawable)
        tel.reset()
        self.assertEqual(len(tel._per_drawable), 0)

    def test_reset_clears_adapter_events(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_adapter_event("draw_circle", 5)
        self.assertIn("draw_circle", tel._adapter_events)
        tel.reset()
        self.assertEqual(len(tel._adapter_events), 0)

    def test_reset_clears_frames(self) -> None:
        tel = BaseRendererTelemetry()
        tel.begin_frame()
        tel.begin_frame()
        self.assertEqual(tel._frames, 2)
        tel.reset()
        self.assertEqual(tel._frames, 0)

    def test_reset_clears_max_batch_depth(self) -> None:
        tel = BaseRendererTelemetry()
        tel.track_batch_depth(5)
        self.assertEqual(tel._max_batch_depth, 5)
        tel.reset()
        self.assertEqual(tel._max_batch_depth, 0)


class TestBaseTelemetryBeginFrame(unittest.TestCase):
    """Verify begin_frame increments frame counter."""

    def test_begin_frame_increments(self) -> None:
        tel = BaseRendererTelemetry()
        self.assertEqual(tel._frames, 0)
        tel.begin_frame()
        self.assertEqual(tel._frames, 1)
        tel.begin_frame()
        self.assertEqual(tel._frames, 2)

    def test_end_frame_is_noop(self) -> None:
        tel = BaseRendererTelemetry()
        tel.begin_frame()
        tel.end_frame()
        self.assertEqual(tel._frames, 1)


class TestBaseTelemetryTiming(unittest.TestCase):
    """Verify mark_time and elapsed_since return reasonable values."""

    def test_mark_time_returns_positive(self) -> None:
        tel = BaseRendererTelemetry()
        t = tel.mark_time()
        self.assertGreater(t, 0)

    def test_elapsed_since_non_negative(self) -> None:
        tel = BaseRendererTelemetry()
        t = tel.mark_time()
        elapsed = tel.elapsed_since(t)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_elapsed_since_far_past_is_positive(self) -> None:
        tel = BaseRendererTelemetry()
        elapsed = tel.elapsed_since(0.0)
        self.assertGreater(elapsed, 0.0)

    def test_elapsed_since_far_future_is_zero(self) -> None:
        tel = BaseRendererTelemetry()
        elapsed = tel.elapsed_since(1e15)
        self.assertEqual(elapsed, 0.0)


class TestBaseTelemetryRecordPlanBuild(unittest.TestCase):
    """Verify record_plan_build accumulation."""

    def test_accumulates_phase_totals(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_build("Point", 1.5)
        tel.record_plan_build("Point", 2.5)
        self.assertAlmostEqual(tel._phase_totals["plan_build_ms"], 4.0)

    def test_increments_phase_count(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_build("Point", 1.0)
        tel.record_plan_build("Segment", 2.0)
        self.assertEqual(tel._phase_counts["plan_build_count"], 2)

    def test_creates_per_drawable_bucket(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_build("Circle", 3.0)
        self.assertIn("Circle", tel._per_drawable)
        bucket = tel._per_drawable["Circle"]
        self.assertAlmostEqual(bucket["plan_build_ms"], 3.0)
        self.assertEqual(bucket["plan_build_count"], 1)

    def test_accumulates_in_same_bucket(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_build("Circle", 1.0)
        tel.record_plan_build("Circle", 2.0)
        bucket = tel._per_drawable["Circle"]
        self.assertAlmostEqual(bucket["plan_build_ms"], 3.0)
        self.assertEqual(bucket["plan_build_count"], 2)

    def test_does_not_affect_cartesian_by_default(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_build("Point", 5.0)
        self.assertEqual(tel._phase_totals["cartesian_plan_build_ms"], 0.0)
        self.assertEqual(tel._phase_counts["cartesian_plan_count"], 0)


class TestBaseTelemetryRecordPlanApply(unittest.TestCase):
    """Verify record_plan_apply accumulation."""

    def test_accumulates_phase_totals(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_apply("Point", 1.5)
        tel.record_plan_apply("Point", 2.5)
        self.assertAlmostEqual(tel._phase_totals["plan_apply_ms"], 4.0)

    def test_increments_phase_count(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_apply("Segment", 1.0)
        self.assertEqual(tel._phase_counts["plan_apply_count"], 1)

    def test_creates_per_drawable_bucket(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_apply("Triangle", 2.0)
        self.assertIn("Triangle", tel._per_drawable)
        bucket = tel._per_drawable["Triangle"]
        self.assertAlmostEqual(bucket["plan_apply_ms"], 2.0)
        self.assertEqual(bucket["plan_apply_count"], 1)

    def test_does_not_affect_cartesian_by_default(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_apply("Point", 5.0)
        self.assertEqual(tel._phase_totals["cartesian_plan_apply_ms"], 0.0)


class TestBaseTelemetryRecordPlanMiss(unittest.TestCase):
    """Verify record_plan_miss increments counters."""

    def test_increments_phase_count(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_miss("Circle")
        tel.record_plan_miss("Circle")
        self.assertEqual(tel._phase_counts["plan_miss_count"], 2)

    def test_increments_per_drawable_bucket(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_miss("Circle")
        bucket = tel._per_drawable["Circle"]
        self.assertEqual(bucket["plan_miss_count"], 1)


class TestBaseTelemetryRecordPlanSkip(unittest.TestCase):
    """Verify record_plan_skip increments counters."""

    def test_increments_phase_count(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_skip("Ellipse")
        self.assertEqual(tel._phase_counts["plan_skip_count"], 1)

    def test_increments_per_drawable_bucket(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_skip("Ellipse")
        tel.record_plan_skip("Ellipse")
        bucket = tel._per_drawable["Ellipse"]
        self.assertEqual(bucket["plan_skip_count"], 2)


class TestBaseTelemetryCartesian(unittest.TestCase):
    """Verify cartesian=True updates both regular and cartesian counters."""

    def test_record_plan_build_cartesian(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_build("Grid", 3.0, cartesian=True)
        self.assertAlmostEqual(tel._phase_totals["plan_build_ms"], 3.0)
        self.assertAlmostEqual(tel._phase_totals["cartesian_plan_build_ms"], 3.0)
        self.assertEqual(tel._phase_counts["plan_build_count"], 1)
        self.assertEqual(tel._phase_counts["cartesian_plan_count"], 1)

    def test_record_plan_apply_cartesian(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_apply("Grid", 2.0, cartesian=True)
        self.assertAlmostEqual(tel._phase_totals["plan_apply_ms"], 2.0)
        self.assertAlmostEqual(tel._phase_totals["cartesian_plan_apply_ms"], 2.0)
        self.assertEqual(tel._phase_counts["plan_apply_count"], 1)

    def test_cartesian_does_not_increment_cartesian_plan_count_on_apply(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_apply("Grid", 2.0, cartesian=True)
        # cartesian_plan_count is only incremented by record_plan_build
        self.assertEqual(tel._phase_counts["cartesian_plan_count"], 0)

    def test_mixed_cartesian_and_regular(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_build("Point", 1.0)
        tel.record_plan_build("Grid", 2.0, cartesian=True)
        self.assertAlmostEqual(tel._phase_totals["plan_build_ms"], 3.0)
        self.assertAlmostEqual(tel._phase_totals["cartesian_plan_build_ms"], 2.0)
        self.assertEqual(tel._phase_counts["plan_build_count"], 2)
        self.assertEqual(tel._phase_counts["cartesian_plan_count"], 1)


class TestBaseTelemetryAdapterEvent(unittest.TestCase):
    """Verify record_adapter_event initializes and accumulates."""

    def test_initializes_new_event(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_adapter_event("fill_rect")
        self.assertEqual(tel._adapter_events["fill_rect"], 1)

    def test_accumulates_events(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_adapter_event("fill_rect", 3)
        tel.record_adapter_event("fill_rect", 2)
        self.assertEqual(tel._adapter_events["fill_rect"], 5)

    def test_multiple_event_types(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_adapter_event("fill_rect", 2)
        tel.record_adapter_event("draw_line", 1)
        self.assertEqual(tel._adapter_events["fill_rect"], 2)
        self.assertEqual(tel._adapter_events["draw_line"], 1)

    def test_default_amount_is_one(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_adapter_event("stroke_path")
        tel.record_adapter_event("stroke_path")
        self.assertEqual(tel._adapter_events["stroke_path"], 2)


class TestBaseTelemetryBatchDepth(unittest.TestCase):
    """Verify track_batch_depth tracks maximum depth."""

    def test_tracks_maximum(self) -> None:
        tel = BaseRendererTelemetry()
        tel.track_batch_depth(3)
        tel.track_batch_depth(5)
        tel.track_batch_depth(2)
        self.assertEqual(tel._max_batch_depth, 5)

    def test_ignores_lower_depth(self) -> None:
        tel = BaseRendererTelemetry()
        tel.track_batch_depth(10)
        tel.track_batch_depth(3)
        self.assertEqual(tel._max_batch_depth, 10)

    def test_zero_depth_does_not_update(self) -> None:
        tel = BaseRendererTelemetry()
        tel.track_batch_depth(0)
        self.assertEqual(tel._max_batch_depth, 0)


class TestBaseTelemetrySnapshot(unittest.TestCase):
    """Verify snapshot returns correct dict and does NOT reset."""

    def test_snapshot_returns_expected_keys(self) -> None:
        tel = BaseRendererTelemetry()
        snap = tel.snapshot()
        self.assertIn("frames", snap)
        self.assertIn("phase", snap)
        self.assertIn("per_drawable", snap)
        self.assertIn("adapter_events", snap)

    def test_snapshot_frames_value(self) -> None:
        tel = BaseRendererTelemetry()
        tel.begin_frame()
        tel.begin_frame()
        snap = tel.snapshot()
        self.assertEqual(snap["frames"], 2)

    def test_snapshot_phase_merges_totals_and_counts(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_build("Point", 1.5)
        snap = tel.snapshot()
        phase = snap["phase"]
        self.assertIn("plan_build_ms", phase)
        self.assertIn("plan_build_count", phase)
        self.assertAlmostEqual(phase["plan_build_ms"], 1.5)
        self.assertEqual(phase["plan_build_count"], 1)

    def test_snapshot_per_drawable_present(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_build("Circle", 2.0)
        snap = tel.snapshot()
        self.assertIn("Circle", snap["per_drawable"])
        self.assertAlmostEqual(snap["per_drawable"]["Circle"]["plan_build_ms"], 2.0)

    def test_snapshot_adapter_events_present(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_adapter_event("draw_arc", 3)
        snap = tel.snapshot()
        self.assertEqual(snap["adapter_events"]["draw_arc"], 3)

    def test_snapshot_includes_max_batch_depth_when_nonzero(self) -> None:
        tel = BaseRendererTelemetry()
        tel.track_batch_depth(4)
        snap = tel.snapshot()
        self.assertEqual(snap["adapter_events"]["max_batch_depth"], 4)

    def test_snapshot_excludes_max_batch_depth_when_zero(self) -> None:
        tel = BaseRendererTelemetry()
        snap = tel.snapshot()
        self.assertNotIn("max_batch_depth", snap["adapter_events"])

    def test_snapshot_does_not_reset(self) -> None:
        tel = BaseRendererTelemetry()
        tel.begin_frame()
        tel.record_plan_build("Point", 1.0)
        tel.record_adapter_event("fill", 2)
        tel.track_batch_depth(3)

        snap1 = tel.snapshot()
        snap2 = tel.snapshot()

        self.assertEqual(snap1["frames"], snap2["frames"])
        self.assertEqual(snap1["phase"], snap2["phase"])
        self.assertEqual(tel._frames, 1)

    def test_snapshot_returns_copies(self) -> None:
        tel = BaseRendererTelemetry()
        tel.record_plan_build("Point", 1.0)
        snap = tel.snapshot()
        # Mutating the snapshot should not affect internal state
        snap["phase"]["plan_build_ms"] = 999.0
        snap["per_drawable"]["Point"]["plan_build_ms"] = 999.0
        self.assertAlmostEqual(tel._phase_totals["plan_build_ms"], 1.0)
        self.assertAlmostEqual(tel._per_drawable["Point"]["plan_build_ms"], 1.0)


class TestBaseTelemetryDrain(unittest.TestCase):
    """Verify drain returns correct dict and DOES reset."""

    def test_drain_returns_expected_keys(self) -> None:
        tel = BaseRendererTelemetry()
        result = tel.drain()
        self.assertIn("frames", result)
        self.assertIn("phase", result)
        self.assertIn("per_drawable", result)
        self.assertIn("adapter_events", result)

    def test_drain_returns_accumulated_data(self) -> None:
        tel = BaseRendererTelemetry()
        tel.begin_frame()
        tel.record_plan_build("Segment", 5.0)
        tel.record_adapter_event("stroke", 10)
        result = tel.drain()
        self.assertEqual(result["frames"], 1)
        self.assertAlmostEqual(result["phase"]["plan_build_ms"], 5.0)
        self.assertEqual(result["adapter_events"]["stroke"], 10)

    def test_drain_resets_counters(self) -> None:
        tel = BaseRendererTelemetry()
        tel.begin_frame()
        tel.record_plan_build("Segment", 5.0)
        tel.record_adapter_event("stroke", 10)
        tel.track_batch_depth(3)
        tel.drain()

        self.assertEqual(tel._frames, 0)
        self.assertEqual(tel._phase_totals["plan_build_ms"], 0.0)
        self.assertEqual(tel._phase_counts["plan_build_count"], 0)
        self.assertEqual(len(tel._per_drawable), 0)
        self.assertEqual(len(tel._adapter_events), 0)
        self.assertEqual(tel._max_batch_depth, 0)

    def test_drain_then_snapshot_is_empty(self) -> None:
        tel = BaseRendererTelemetry()
        tel.begin_frame()
        tel.record_plan_build("Point", 1.0)
        tel.drain()
        snap = tel.snapshot()
        self.assertEqual(snap["frames"], 0)
        self.assertEqual(snap["phase"]["plan_build_ms"], 0.0)
        self.assertEqual(len(snap["per_drawable"]), 0)


class TestBaseTelemetryNewDrawableBucket(unittest.TestCase):
    """Verify _new_drawable_bucket returns correct default keys."""

    def test_default_bucket_keys(self) -> None:
        tel = BaseRendererTelemetry()
        bucket = tel._new_drawable_bucket()
        expected_keys = {
            "plan_build_ms",
            "plan_apply_ms",
            "plan_build_count",
            "plan_apply_count",
            "plan_miss_count",
            "plan_skip_count",
        }
        self.assertEqual(set(bucket.keys()), expected_keys)

    def test_default_bucket_values_zero(self) -> None:
        tel = BaseRendererTelemetry()
        bucket = tel._new_drawable_bucket()
        for key, value in bucket.items():
            self.assertEqual(value, 0.0, f"{key} should be 0.0")

    def test_buckets_are_independent(self) -> None:
        tel = BaseRendererTelemetry()
        b1 = tel._new_drawable_bucket()
        b2 = tel._new_drawable_bucket()
        b1["plan_build_ms"] = 99.0
        self.assertEqual(b2["plan_build_ms"], 0.0)


class TestCanvas2DTelemetryOverride(unittest.TestCase):
    """Verify Canvas2DTelemetry adds extra keys to the drawable bucket."""

    def test_bucket_has_legacy_render_keys(self) -> None:
        tel = Canvas2DTelemetry()
        bucket = tel._new_drawable_bucket()
        self.assertIn("legacy_render_ms", bucket)
        self.assertIn("legacy_render_count", bucket)

    def test_bucket_retains_base_keys(self) -> None:
        tel = Canvas2DTelemetry()
        bucket = tel._new_drawable_bucket()
        base_keys = {
            "plan_build_ms",
            "plan_apply_ms",
            "plan_build_count",
            "plan_apply_count",
            "plan_miss_count",
            "plan_skip_count",
        }
        for key in base_keys:
            self.assertIn(key, bucket, f"Missing base key: {key}")

    def test_legacy_keys_default_zero(self) -> None:
        tel = Canvas2DTelemetry()
        bucket = tel._new_drawable_bucket()
        self.assertEqual(bucket["legacy_render_ms"], 0.0)
        self.assertEqual(bucket["legacy_render_count"], 0)

    def test_record_plan_build_creates_canvas2d_bucket(self) -> None:
        tel = Canvas2DTelemetry()
        tel.record_plan_build("Point", 1.0)
        bucket = tel._per_drawable["Point"]
        self.assertIn("legacy_render_ms", bucket)
        self.assertIn("legacy_render_count", bucket)

    def test_canvas2d_inherits_all_base_behavior(self) -> None:
        tel = Canvas2DTelemetry()
        tel.begin_frame()
        tel.record_plan_build("Circle", 2.0)
        tel.record_plan_apply("Circle", 1.0)
        tel.record_plan_miss("Circle")
        tel.record_plan_skip("Circle")
        tel.record_adapter_event("fill", 3)
        tel.track_batch_depth(2)

        snap = tel.snapshot()
        self.assertEqual(snap["frames"], 1)
        self.assertAlmostEqual(snap["phase"]["plan_build_ms"], 2.0)
        self.assertAlmostEqual(snap["phase"]["plan_apply_ms"], 1.0)
        self.assertEqual(snap["phase"]["plan_miss_count"], 1)
        self.assertEqual(snap["phase"]["plan_skip_count"], 1)
        self.assertEqual(snap["adapter_events"]["fill"], 3)
        self.assertEqual(snap["adapter_events"]["max_batch_depth"], 2)


class TestSvgTelemetry(unittest.TestCase):
    """Smoke tests for SvgTelemetry (empty subclass of BaseRendererTelemetry)."""

    def test_instantiation(self) -> None:
        tel = SvgTelemetry()
        self.assertIsInstance(tel, BaseRendererTelemetry)

    def test_has_base_methods(self) -> None:
        tel = SvgTelemetry()
        for method_name in ("reset", "begin_frame", "snapshot", "drain"):
            self.assertTrue(
                callable(getattr(tel, method_name, None)),
                f"SvgTelemetry should have callable '{method_name}'",
            )

    def test_new_drawable_bucket_matches_base(self) -> None:
        svg_tel = SvgTelemetry()
        base_tel = BaseRendererTelemetry()
        svg_bucket = svg_tel._new_drawable_bucket()
        base_bucket = base_tel._new_drawable_bucket()
        self.assertEqual(
            set(svg_bucket.keys()),
            set(base_bucket.keys()),
            "SvgTelemetry bucket should have exactly the base keys (no extras)",
        )

    def test_new_drawable_bucket_values_zero(self) -> None:
        tel = SvgTelemetry()
        bucket = tel._new_drawable_bucket()
        for key, value in bucket.items():
            self.assertEqual(value, 0.0, f"{key} should be 0.0")

    def test_snapshot_after_begin_frame(self) -> None:
        tel = SvgTelemetry()
        tel.begin_frame()
        snap = tel.snapshot()
        self.assertEqual(snap["frames"], 1)

    def test_drain_resets(self) -> None:
        tel = SvgTelemetry()
        tel.begin_frame()
        tel.record_plan_build("Point", 2.0)
        result = tel.drain()
        self.assertEqual(result["frames"], 1)
        self.assertEqual(tel._frames, 0)
        self.assertEqual(len(tel._per_drawable), 0)


__all__ = [
    "TestBaseTelemetryInit",
    "TestBaseTelemetryReset",
    "TestBaseTelemetryBeginFrame",
    "TestBaseTelemetryTiming",
    "TestBaseTelemetryRecordPlanBuild",
    "TestBaseTelemetryRecordPlanApply",
    "TestBaseTelemetryRecordPlanMiss",
    "TestBaseTelemetryRecordPlanSkip",
    "TestBaseTelemetryCartesian",
    "TestBaseTelemetryAdapterEvent",
    "TestBaseTelemetryBatchDepth",
    "TestBaseTelemetrySnapshot",
    "TestBaseTelemetryDrain",
    "TestBaseTelemetryNewDrawableBucket",
    "TestCanvas2DTelemetryOverride",
    "TestSvgTelemetry",
]
