from __future__ import annotations

import unittest
from types import SimpleNamespace

from rendering.shared_drawable_renderers import (
    _coerce_font_size,
    _compute_zoom_adjusted_font_size,
)


class TestCoerceFontSize(unittest.TestCase):
    def test_valid_positive_candidate(self) -> None:
        result = _coerce_font_size(16, 12)
        self.assertEqual(result, 16.0)

    def test_valid_string_number_candidate(self) -> None:
        result = _coerce_font_size("18", 12)
        self.assertEqual(result, 18.0)

    def test_zero_candidate_uses_fallback(self) -> None:
        result = _coerce_font_size(0, 14)
        self.assertEqual(result, 14.0)

    def test_negative_candidate_uses_fallback(self) -> None:
        result = _coerce_font_size(-5, 14)
        self.assertEqual(result, 14.0)

    def test_none_candidate_uses_fallback(self) -> None:
        result = _coerce_font_size(None, 12)
        self.assertEqual(result, 12.0)

    def test_nan_candidate_uses_fallback(self) -> None:
        result = _coerce_font_size(float("nan"), 10)
        self.assertEqual(result, 10.0)

    def test_infinity_candidate_uses_fallback(self) -> None:
        result = _coerce_font_size(float("inf"), 10)
        self.assertEqual(result, 10.0)

    def test_invalid_string_uses_fallback(self) -> None:
        result = _coerce_font_size("abc", 12)
        self.assertEqual(result, 12.0)

    def test_invalid_fallback_uses_default(self) -> None:
        result = _coerce_font_size(None, None)
        self.assertEqual(result, 14.0)

    def test_zero_fallback_uses_default(self) -> None:
        result = _coerce_font_size(None, 0)
        self.assertEqual(result, 14.0)

    def test_negative_fallback_uses_default(self) -> None:
        result = _coerce_font_size(None, -5)
        self.assertEqual(result, 14.0)

    def test_custom_default_value(self) -> None:
        result = _coerce_font_size(None, None, default_value=20.0)
        self.assertEqual(result, 20.0)


class TestComputeZoomAdjustedFontSize(unittest.TestCase):
    def test_same_scale_returns_base_size(self) -> None:
        label = SimpleNamespace(reference_scale_factor=1.0)
        mapper = SimpleNamespace(scale_factor=1.0)
        result = _compute_zoom_adjusted_font_size(16.0, label, mapper)
        self.assertEqual(result, 16.0)

    def test_zoomed_in_returns_base_size(self) -> None:
        label = SimpleNamespace(reference_scale_factor=1.0)
        mapper = SimpleNamespace(scale_factor=2.0)
        result = _compute_zoom_adjusted_font_size(16.0, label, mapper)
        self.assertEqual(result, 16.0)

    def test_zoomed_out_scales_font(self) -> None:
        label = SimpleNamespace(reference_scale_factor=1.0)
        mapper = SimpleNamespace(scale_factor=0.5)
        result = _compute_zoom_adjusted_font_size(16.0, label, mapper)
        self.assertEqual(result, 8.0)

    def test_extreme_zoom_out_returns_minimum(self) -> None:
        from constants import label_min_screen_font_px

        label = SimpleNamespace(reference_scale_factor=1.0)
        mapper = SimpleNamespace(scale_factor=0.01)
        result = _compute_zoom_adjusted_font_size(16.0, label, mapper)
        self.assertGreaterEqual(result, 0.0)
        if result > 0:
            self.assertGreaterEqual(result, label_min_screen_font_px)

    def test_vanish_threshold(self) -> None:
        from constants import label_vanish_threshold_px

        label = SimpleNamespace(reference_scale_factor=1.0)
        mapper = SimpleNamespace(scale_factor=0.001)
        result = _compute_zoom_adjusted_font_size(2.0, label, mapper)
        self.assertEqual(result, 0.0)

    def test_missing_reference_scale_uses_default(self) -> None:
        label = SimpleNamespace()
        mapper = SimpleNamespace(scale_factor=1.0)
        result = _compute_zoom_adjusted_font_size(14.0, label, mapper)
        self.assertEqual(result, 14.0)

    def test_missing_current_scale_uses_default(self) -> None:
        label = SimpleNamespace(reference_scale_factor=1.0)
        mapper = SimpleNamespace()
        result = _compute_zoom_adjusted_font_size(14.0, label, mapper)
        self.assertEqual(result, 14.0)

    def test_invalid_reference_scale_uses_default(self) -> None:
        label = SimpleNamespace(reference_scale_factor="invalid")
        mapper = SimpleNamespace(scale_factor=1.0)
        result = _compute_zoom_adjusted_font_size(14.0, label, mapper)
        self.assertEqual(result, 14.0)

    def test_zero_reference_scale_uses_default(self) -> None:
        label = SimpleNamespace(reference_scale_factor=0)
        mapper = SimpleNamespace(scale_factor=1.0)
        result = _compute_zoom_adjusted_font_size(14.0, label, mapper)
        self.assertEqual(result, 14.0)

    def test_negative_reference_scale_uses_default(self) -> None:
        label = SimpleNamespace(reference_scale_factor=-1.0)
        mapper = SimpleNamespace(scale_factor=1.0)
        result = _compute_zoom_adjusted_font_size(14.0, label, mapper)
        self.assertEqual(result, 14.0)


__all__ = ["TestCoerceFontSize", "TestComputeZoomAdjustedFontSize"]
