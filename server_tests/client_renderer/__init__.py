from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

import sys
from types import ModuleType, SimpleNamespace


def _install_browser_stub() -> None:
    if "browser" in sys.modules:
        return

    class _NerdamerStub:
        def __call__(self, *_args: object, **_kwargs: object) -> "_NerdamerStub":
            return self

        def text(self) -> str:
            return ""

        def evaluate(self) -> "_NerdamerStub":
            return self

        def coeffs(self, *_args: object, **_kwargs: object) -> "_NerdamerStub":
            return self

        def solveEquations(self, *_args: object, **_kwargs: object):  # pragma: no cover - only used for compatibility
            return []

    class _DocumentStub:
        def __init__(self) -> None:
            self._store: dict[str, SimpleNamespace] = {}

        def getElementById(self, element_id: str):  # pragma: no cover - availability only
            return self._store.get(element_id)

        def __getitem__(self, key: str) -> SimpleNamespace:
            return self._store.setdefault(key, SimpleNamespace(clear=lambda: None))

        def __le__(self, other: object) -> None:  # pragma: no cover - DOM append noop
            return None

    class _HtmlStub:
        def CANVAS(self, **_kwargs: object) -> SimpleNamespace:
            style = SimpleNamespace(
                width="0px",
                height="0px",
                position="absolute",
                top="0",
                left="0",
                pointerEvents="none",
                display="block",
                zIndex="0",
            )
            canvas = SimpleNamespace(attrs={}, style=style, width=0, height=0)

            def get_context(kind: str):  # pragma: no cover - compatibility only
                if kind == "2d":
                    return SimpleNamespace()
                return SimpleNamespace()

            canvas.getContext = get_context
            return canvas

    class _SvgStub:
        def g(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(attrs={}, style=SimpleNamespace())

    class _ConsoleStub:
        def log(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - noop
            pass

        warn = error = info = debug = log

    nerdamer_stub = _NerdamerStub()

    class _WindowStub:
        def __init__(self) -> None:
            self.performance = SimpleNamespace(now=lambda: 0.0)
            self.localStorage = SimpleNamespace(getItem=lambda _key: None)
            self.Float32Array = SimpleNamespace(new=lambda data: list(data))
            self.Math = SimpleNamespace()
            self.math = SimpleNamespace(
                format=lambda value: value,
                sqrt=lambda value: value**0.5,
                pow=lambda base, exp: base**exp,
                det=lambda _matrix: 0.0,
                evaluate=lambda _expr, _vars=None: 0.0,
            )
            self.nerdamer = nerdamer_stub

    browser = ModuleType("browser")
    browser.document = _DocumentStub()
    browser.html = _HtmlStub()
    browser.svg = _SvgStub()
    browser.window = _WindowStub()
    browser.console = _ConsoleStub()

    sys.modules["browser"] = browser


_install_browser_stub()

__all__: list[str] = []
