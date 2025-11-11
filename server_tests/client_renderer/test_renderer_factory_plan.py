from __future__ import annotations

import sys
from pathlib import Path

CLIENT_ROOT = Path(__file__).resolve().parents[2] / "static" / "client"
CLIENT_ROOT_STR = str(CLIENT_ROOT)
if CLIENT_ROOT.exists() and CLIENT_ROOT_STR not in sys.path:
    sys.path.insert(0, CLIENT_ROOT_STR)

import importlib
import unittest

from rendering import factory


class TestRendererFactoryPlan(unittest.TestCase):
    def tearDown(self) -> None:
        importlib.reload(factory)

    def test_create_renderer_falls_back_to_next_available(self) -> None:
        attempts: list[str] = []
        sentinel = object()

        def failing_canvas() -> None:
            attempts.append("canvas2d")
            raise RuntimeError("Canvas unavailable")

        def svg_renderer() -> object:
            attempts.append("svg")
            return sentinel

        def webgl_renderer() -> object:
            attempts.append("webgl")
            return object()

        factory.Canvas2DRenderer = failing_canvas  # type: ignore[attr-defined]
        factory.SvgRenderer = svg_renderer  # type: ignore[attr-defined]
        factory.WebGLRenderer = webgl_renderer  # type: ignore[attr-defined]

        result = factory.create_renderer()

        self.assertIs(result, sentinel)
        self.assertEqual(attempts, ["canvas2d", "svg"], "Factory should skip failing constructors and stop at first success")

    def test_preferred_renderer_short_circuits_fallback(self) -> None:
        calls: dict[str, int] = {"canvas2d": 0, "svg": 0, "webgl": 0}

        def make_factory(name: str):
            def constructor() -> object:
                calls[name] += 1
                return object()

            return constructor

        factory.Canvas2DRenderer = make_factory("canvas2d")  # type: ignore[attr-defined]
        factory.SvgRenderer = make_factory("svg")  # type: ignore[attr-defined]
        webgl_instance = object()

        def preferred_webgl() -> object:
            calls["webgl"] += 1
            return webgl_instance

        factory.WebGLRenderer = preferred_webgl  # type: ignore[attr-defined]

        result = factory.create_renderer(preferred="webgl")

        self.assertIs(result, webgl_instance)
        self.assertEqual(calls["webgl"], 1)
        self.assertEqual(calls["canvas2d"], 0)
        self.assertEqual(calls["svg"], 0)


__all__ = ["TestRendererFactoryPlan"]
