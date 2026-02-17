from __future__ import annotations

import math
import unittest
from typing import Any, Callable, Optional, Tuple

from constants import (
    label_line_wrap_threshold,
    label_min_screen_font_px,
    label_text_max_length,
    label_vanish_threshold_px,
)
from drawables.label import Label
from drawables.label_render_mode import LabelRenderMode
from managers.drawables_container import DrawablesContainer
from managers.label_manager import LabelManager
from coordinate_mapper import CoordinateMapper
from rendering.cached_render_plan import build_plan_for_drawable, _capture_map_state
from rendering.style_manager import get_renderer_style
from rendering.shared_drawable_renderers import render_label_helper
from name_generator.drawable import DrawableNameGenerator
from .simple_mock import SimpleMock


class TestLabel(unittest.TestCase):
    def _make_label_manager(
        self,
        name_generator_factory: Optional[Callable[[Any], Any]] = None,
    ) -> tuple[LabelManager, DrawablesContainer, SimpleMock]:
        canvas = SimpleMock(draw_enabled=True)
        canvas.draw = SimpleMock()
        canvas.undo_redo_manager = SimpleMock()
        canvas.undo_redo_manager.archive = SimpleMock()
        container = DrawablesContainer()

        def get_drawables_by_class_name(class_name: str) -> list[Any]:
            if class_name == "Label":
                return container.Labels
            return []

        canvas.get_drawables_by_class_name = get_drawables_by_class_name

        name_generator = name_generator_factory(canvas) if name_generator_factory else DrawableNameGenerator(canvas)

        dependency_manager = SimpleMock()
        proxy = SimpleMock()
        manager = LabelManager(canvas, container, name_generator, dependency_manager, proxy)
        return manager, container, canvas

    def test_wraps_text_over_threshold(self) -> None:
        long_word = "a" * (label_line_wrap_threshold + 5)
        label = Label(0, 0, f"intro {long_word}")
        lines = label.lines
        self.assertGreaterEqual(len(lines), 2)
        self.assertTrue(all(len(line) <= label_line_wrap_threshold for line in lines))

    def test_text_exceeding_max_length_raises(self) -> None:
        overlong_text = "x" * (label_text_max_length + 25)
        with self.assertRaises(ValueError):
            Label(0, 0, overlong_text)

    def test_label_manager_adds_and_removes_labels(self) -> None:
        draw_calls: list[None] = []
        undo_calls: list[None] = []

        def record_draw() -> None:
            draw_calls.append(None)

        def record_archive() -> None:
            undo_calls.append(None)

        canvas = SimpleMock(draw_enabled=True)
        canvas.draw = record_draw
        canvas.undo_redo_manager = SimpleMock()
        canvas.undo_redo_manager.archive = record_archive

        container = DrawablesContainer()
        name_generator = SimpleMock(
            generate_label_name=lambda preferred: (
                preferred.strip() if isinstance(preferred, str) and preferred.strip() else "demo_label"
            )
        )
        dependency_manager = SimpleMock()
        proxy = SimpleMock()
        manager = LabelManager(canvas, container, name_generator, dependency_manager, proxy)

        label = manager.create_label(1, 2, "demo text", name="demo", rotation_degrees=15.0)
        self.assertIn(label, container.Labels)
        self.assertIs(manager.get_label_by_name("demo"), label)
        self.assertTrue(math.isclose(label.rotation_degrees, 15.0))
        self.assertEqual(len(draw_calls), 1)
        self.assertEqual(len(undo_calls), 1)

        removed = manager.delete_label("demo")
        self.assertTrue(removed)
        self.assertFalse(container.Labels)
        self.assertEqual(len(draw_calls), 2)
        self.assertEqual(len(undo_calls), 2)

    def test_label_name_generator_produces_letter_sequence(self) -> None:
        manager, _, _ = self._make_label_manager(name_generator_factory=lambda canvas: DrawableNameGenerator(canvas))
        label_one = manager.create_label(0.0, 0.0, "auto")
        label_two = manager.create_label(1.0, 1.0, "auto2")
        label_three = manager.create_label(2.0, 2.0, "auto3")

        self.assertEqual(label_one.name, "label_A")
        self.assertEqual(label_two.name, "label_B")
        self.assertEqual(label_three.name, "label_C")

    def test_label_name_generator_handles_duplicate_preferred_names(self) -> None:
        manager, _, _ = self._make_label_manager(name_generator_factory=lambda canvas: DrawableNameGenerator(canvas))

        custom_one = manager.create_label(0.0, 0.0, "auto", name="CustomLabel")
        custom_two = manager.create_label(1.0, 1.0, "auto", name="CustomLabel")

        self.assertEqual(custom_one.name, "CustomLabel")
        self.assertEqual(custom_two.name, "CustomLabel_1")

    def test_update_label_allows_multiple_properties(self) -> None:
        manager, container, canvas = self._make_label_manager()
        label = manager.create_label(1.0, 2.0, "demo", name="lbl", color="#123456", rotation_degrees=15.0)

        result = manager.update_label(
            "lbl",
            new_text="updated text",
            new_x=5.0,
            new_y=-3.0,
            new_color="#654321",
            new_font_size=18.5,
            new_rotation_degrees=42.0,
        )

        self.assertTrue(result)
        self.assertEqual(label.text, "updated text")
        self.assertTrue(math.isclose(label.position.x, 5.0))
        self.assertTrue(math.isclose(label.position.y, -3.0))
        self.assertEqual(label.color, "#654321")
        self.assertTrue(math.isclose(label.font_size, 18.5))
        self.assertTrue(math.isclose(label.rotation_degrees, 42.0))
        self.assertEqual(len(canvas.draw.calls), 2)  # create + update
        self.assertEqual(len(canvas.undo_redo_manager.archive.calls), 2)
        self.assertIs(container.Labels[0], label)

    def test_update_label_requires_both_coordinates(self) -> None:
        manager, _, canvas = self._make_label_manager()
        manager.create_label(0.0, 0.0, "demo", name="lbl")
        initial_archives = len(canvas.undo_redo_manager.archive.calls)

        with self.assertRaises(ValueError):
            manager.update_label("lbl", new_x=5.0)

        self.assertEqual(len(canvas.undo_redo_manager.archive.calls), initial_archives)

    def test_update_label_rejects_empty_color(self) -> None:
        manager, _, canvas = self._make_label_manager()
        manager.create_label(0.0, 0.0, "demo", name="lbl")
        initial_archives = len(canvas.undo_redo_manager.archive.calls)

        with self.assertRaises(ValueError):
            manager.update_label("lbl", new_color="   ")

        self.assertEqual(len(canvas.undo_redo_manager.archive.calls), initial_archives)

    def test_update_label_rejects_overlong_text(self) -> None:
        manager, _, canvas = self._make_label_manager()
        manager.create_label(0.0, 0.0, "demo", name="lbl")
        initial_archives = len(canvas.undo_redo_manager.archive.calls)
        overlong_text = "x" * (label_text_max_length + 10)

        with self.assertRaises(ValueError):
            manager.update_label("lbl", new_text=overlong_text)

        self.assertEqual(len(canvas.undo_redo_manager.archive.calls), initial_archives)

    def test_render_label_helper_draws_all_lines(self) -> None:
        label = Label(1.0, 2.0, "line one line two line three", font_size=12, rotation_degrees=30.0)
        primitives = SimpleMock()
        draw_text_mock = SimpleMock()
        primitives.draw_text = draw_text_mock
        mapper = SimpleMock(math_to_screen=lambda x, y: (x * 10.0, y * 10.0))
        style = {"label_font_size": 12, "label_text_color": "#123456"}

        render_label_helper(primitives, label, mapper, style)

        calls = draw_text_mock.calls
        self.assertGreaterEqual(len(calls), len(label.lines))

        first_args: Tuple[Any, ...] = calls[0][0]
        self.assertEqual(first_args[0], label.lines[0])
        self.assertTrue(math.isclose(first_args[1][0], 10.0))
        self.assertTrue(math.isclose(first_args[1][1], 20.0))
        first_kwargs = calls[0][1]
        metadata = first_kwargs.get("metadata", {})
        label_meta = metadata.get("label", {})
        self.assertTrue(math.isclose(float(label_meta.get("rotation_degrees", 0.0)), 30.0))

        if len(label.lines) > 1:
            second_args: Tuple[Any, ...] = calls[1][0]
            self.assertGreater(second_args[1][1], first_args[1][1])

    def test_label_rotation_property_and_rotate(self) -> None:
        label = Label(0.0, 0.0, "rotate me", rotation_degrees=10.0)
        self.assertTrue(math.isclose(label.rotation_degrees, 10.0))
        label.rotation_degrees = 25.0
        self.assertTrue(math.isclose(label.rotation_degrees, 25.0))
        label.rotate(90.0)
        self.assertTrue(math.isclose(label.rotation_degrees, 115.0))

    def test_translate_updates_position(self) -> None:
        label = Label(5.0, -3.0, "move me")
        label.translate(2.5, -4.5)
        self.assertTrue(math.isclose(label.position.x, 7.5))
        self.assertTrue(math.isclose(label.position.y, -7.5))
        self.assertEqual(label.text, "move me")

    def test_text_normalization_and_wrapping(self) -> None:
        label = Label(0.0, 0.0, "  first line\r\nsecond line\rthird line  ")
        self.assertEqual(label.text, "first line\nsecond line\nthird line")
        self.assertEqual(label.lines, ["first line", "second line", "third line"])

    def test_font_size_validation(self) -> None:
        label = Label(0.0, 0.0, "size check")
        with self.assertRaises(ValueError):
            label.font_size = 0

    def test_single_long_word_wraps_across_lines(self) -> None:
        long_word = "a" * (label_line_wrap_threshold * 2 + 5)
        label = Label(0.0, 0.0, long_word)
        self.assertTrue(all(len(line) <= label_line_wrap_threshold for line in label.lines))
        self.assertEqual("".join(label.lines), label.text)

    def test_render_label_helper_scales_font_when_zoomed_out(self) -> None:
        label = Label(0.0, 0.0, "scaled label", font_size=20.0)
        label.update_reference_scale(2.0)

        primitives = SimpleMock()
        draw_text_mock = SimpleMock()
        primitives.draw_text = draw_text_mock
        mapper = SimpleMock(math_to_screen=lambda x, y: (x, y), scale_factor=1.0)
        style = {"label_font_size": 12, "label_text_color": "#abcdef"}

        render_label_helper(primitives, label, mapper, style)

        self.assertTrue(draw_text_mock.calls, "Expected label text to be drawn at least once")
        font_arg = draw_text_mock.calls[0][0][2]
        self.assertTrue(isinstance(font_arg.size, (int, float)))
        self.assertEqual(font_arg.size, 10)

    def test_render_label_helper_skips_when_below_visibility_threshold(self) -> None:
        label = Label(0.0, 0.0, "tiny label", font_size=10.0)
        label.update_reference_scale(1.0)

        primitives = SimpleMock()
        draw_text_mock = SimpleMock()
        primitives.draw_text = draw_text_mock
        vanish_scale = label_vanish_threshold_px / (label.font_size * 2.0)
        mapper = SimpleMock(math_to_screen=lambda x, y: (x, y), scale_factor=vanish_scale)
        style = {"label_font_size": 12, "label_text_color": "#abcdef"}

        render_label_helper(primitives, label, mapper, style)

        self.assertFalse(draw_text_mock.calls, "Expected label to be skipped when size falls below threshold")

    def test_label_plan_reprojects_font_size_on_zoom_out(self) -> None:
        label = Label(0.0, 0.0, "plan label", font_size=24.0)
        label.update_reference_scale(1.0)

        mapper = CoordinateMapper(800, 600)
        style = get_renderer_style()

        plan = build_plan_for_drawable(label, mapper, style, supports_transform=False)
        self.assertIsNotNone(plan)
        if plan is None or not plan.commands:
            self.fail("Optimized plan did not produce any commands for label")

        initial_font = plan.commands[0].args[2]
        initial_size = float(getattr(initial_font, "size", 0.0))
        self.assertTrue(math.isclose(initial_size, 24.0))

        mapper.scale_factor = 0.4
        new_state = _capture_map_state(mapper)
        plan.update_map_state(new_state)

        updated_font = plan.commands[0].args[2]
        updated_size = float(getattr(updated_font, "size", 0.0))
        expected_size = max(24.0 * 0.4, label_min_screen_font_px)
        self.assertTrue(updated_size <= initial_size)
        self.assertTrue(
            math.isclose(updated_size, expected_size),
            msg=f"Expected reprojected font size {expected_size}, got {updated_size}",
        )

        mapper.scale_factor = label_vanish_threshold_px / (label.font_size * 2.0)
        far_state = _capture_map_state(mapper)
        plan.update_map_state(far_state)

        far_font = plan.commands[0].args[2]
        far_size = float(getattr(far_font, "size", 0.0))
        self.assertEqual(far_size, 0.0)

    def test_label_plan_spacing_scales_with_font(self) -> None:
        label = Label(0.0, 0.0, "first line\nsecond line", font_size=20.0)
        label.update_reference_scale(1.0)

        mapper = CoordinateMapper(800, 600)
        style = get_renderer_style()

        plan = build_plan_for_drawable(label, mapper, style, supports_transform=False)
        self.assertIsNotNone(plan)
        if plan is None:
            self.fail("Optimized plan did not produce any commands for multi-line label")

        draw_commands = [cmd for cmd in plan.commands if cmd.op == "draw_text"]
        self.assertGreaterEqual(len(draw_commands), 2)

        initial_positions = [cmd.args[1][1] for cmd in draw_commands[:2]]
        initial_spacing = initial_positions[1] - initial_positions[0]
        self.assertTrue(initial_spacing > 0)

        mapper.scale_factor = 0.5
        updated_state = _capture_map_state(mapper)
        plan.update_map_state(updated_state)

        updated_positions = [cmd.args[1][1] for cmd in draw_commands[:2]]
        updated_spacing = updated_positions[1] - updated_positions[0]
        expected_spacing = initial_spacing * 0.5

        self.assertTrue(
            math.isclose(updated_spacing, expected_spacing, rel_tol=1e-6, abs_tol=1e-6),
            msg=f"Expected spacing {expected_spacing}, got {updated_spacing}",
        )

    def test_label_render_mode_serialization_roundtrip(self) -> None:
        state = {
            "kind": "screen_offset",
            "text_format": "text_with_anchor_coords",
            "coord_precision": 3,
            "offset_from_point_radius": True,
            "non_selectable": True,
            "font_size_source": "style",
            "font_size_key": "point_label_font_size",
            "font_family_key": "point_label_font_family",
            "offset_px_x": 0.0,
            "offset_px_y": 0.0,
        }
        restored = LabelRenderMode.from_state(state)
        self.assertEqual(restored.to_state(), state)

    def test_label_get_state_includes_render_mode(self) -> None:
        label = Label(1.0, 2.0, "demo")
        state = label.get_state()
        args = state.get("args", {})
        render_mode = args.get("render_mode")
        self.assertIsInstance(render_mode, dict)
        self.assertEqual(render_mode.get("kind"), "world")
