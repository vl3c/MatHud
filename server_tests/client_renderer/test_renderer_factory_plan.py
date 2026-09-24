from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

import importlib
import unittest
from typing import Callable, Dict, List

from rendering import factory


class TestRendererFactoryPlan(unittest.TestCase):
    def tearDown(self) -> None:
        importlib.reload(factory)

    def _install_loader(self, constructors: Dict[str, Callable[[], object]], loaded: List[str]) -> None:
        def fake_loader(module_path: str, attr: str):
            loaded.append(module_path)
            return constructors.get(module_path)

        factory._load_renderer = fake_loader  # type: ignore[assignment]

    def test_create_renderer_falls_back_to_next_available(self) -> None:
        attempts: list[str] = []
        sentinel = object()

        def failing_canvas() -> None:
            attempts.append("canvas2d")
            raise RuntimeError("Canvas unavailable")

        def svg_renderer() -> object:
            attempts.append("svg")
            return sentinel

        self._install_loader(
            {"rendering.canvas2d_renderer": failing_canvas, "rendering.svg_renderer": svg_renderer},
            [],
        )

        result = factory.create_renderer()

        self.assertIs(result, sentinel)
        self.assertEqual(
            attempts, ["canvas2d", "svg"], "Factory should skip failing constructors and stop at first success"
        )

    def test_preferred_renderer_short_circuits_fallback(self) -> None:
        calls: dict[str, int] = {"canvas2d": 0, "svg": 0}
        svg_instance = object()

        def canvas_constructor() -> object:
            calls["canvas2d"] += 1
            return object()

        def svg_constructor() -> object:
            calls["svg"] += 1
            return svg_instance

        self._install_loader(
            {"rendering.canvas2d_renderer": canvas_constructor, "rendering.svg_renderer": svg_constructor},
            [],
        )

        result = factory.create_renderer(preferred="svg")

        self.assertIs(result, svg_instance)
        self.assertEqual(calls["svg"], 1)
        self.assertEqual(calls["canvas2d"], 0)

    def test_renderer_modules_are_imported_lazily(self) -> None:
        loaded: list[str] = []
        self._install_loader({"rendering.canvas2d_renderer": object}, loaded)

        factory.create_renderer()

        self.assertEqual(loaded, ["rendering.canvas2d_renderer"])

    def test_default_chain_is_canvas2d_then_svg(self) -> None:
        self.assertEqual(factory._build_preference_chain(None), ["canvas2d", "svg"])

    def test_unknown_preferred_mode_falls_back_to_defaults(self) -> None:
        loaded: list[str] = []
        sentinel = object()
        self._install_loader({"rendering.canvas2d_renderer": lambda: sentinel}, loaded)

        self.assertIs(factory.create_renderer(preferred="webgl"), sentinel)
        self.assertEqual(loaded, ["rendering.canvas2d_renderer"])


__all__ = ["TestRendererFactoryPlan"]
