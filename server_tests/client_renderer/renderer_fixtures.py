from __future__ import annotations

from server_tests import python_path_setup  # noqa: F401

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple


class PlanStub:
    """Lightweight stand-in for optimized plans used by renderers."""

    def __init__(
        self,
        *,
        visible: bool = True,
        plan_key: str = "plan",
        usage_counts: Optional[Dict[str, int]] = None,
        supports_transform: bool = False,
        needs_apply: bool = True,
        uses_screen_space: bool = False,
    ) -> None:
        self.visible = visible
        self.plan_key = plan_key
        self._usage_counts = dict(usage_counts or {})
        self._supports_transform = supports_transform
        self._needs_apply = needs_apply
        self._uses_screen_space = uses_screen_space
        self.apply_calls: int = 0
        self.update_calls: int = 0
        self.mark_dirty_calls: int = 0

    def is_visible(self, width: int, height: int, margin: float = 0.0) -> bool:  # pragma: no cover - trivial
        return self.visible

    def apply(self, primitives: Any) -> None:
        self.apply_calls += 1

    def update_map_state(self, _map_state: Dict[str, float]) -> None:
        self.update_calls += 1

    def mark_dirty(self) -> None:
        self.mark_dirty_calls += 1

    def supports_transform(self) -> bool:
        return self._supports_transform

    def needs_apply(self) -> bool:
        return self._needs_apply

    def uses_screen_space(self) -> bool:
        return self._uses_screen_space

    def get_usage_counts(self) -> Dict[str, int]:
        return dict(self._usage_counts)


class TelemetryRecorder:
    """Telemetry double that tracks call counts for assertions."""

    def __init__(self) -> None:
        self.plan_build_events: List[Tuple[str, bool]] = []
        self.plan_apply_events: List[Tuple[str, bool]] = []
        self.plan_skip_events: List[str] = []
        self.plan_miss_events: List[str] = []

    def mark_time(self) -> object:  # pragma: no cover - trivial
        return object()

    def elapsed_since(self, _token: object) -> float:  # pragma: no cover - deterministic
        return 0.0

    def record_plan_build(self, drawable_name: str, _elapsed: float, cartesian: bool = False) -> None:
        self.plan_build_events.append((drawable_name, cartesian))

    def record_plan_apply(self, drawable_name: str, _elapsed: float, cartesian: bool = False) -> None:
        self.plan_apply_events.append((drawable_name, cartesian))

    def record_plan_skip(self, drawable_name: str) -> None:
        self.plan_skip_events.append(drawable_name)

    def record_plan_miss(self, drawable_name: str) -> None:
        self.plan_miss_events.append(drawable_name)


class PrimitiveRecorder:
    """Records primitive draw calls emitted by helpers."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []

    def fill_circle(
        self, center: Tuple[float, float], radius: float, fill: Any, stroke: Any = None, *, screen_space: bool = False
    ) -> None:
        self.calls.append(("fill_circle", (center, radius, fill, stroke, screen_space), {}))

    def draw_text(
        self,
        text: str,
        position: Tuple[float, float],
        font: str,
        color: str,
        alignment: str,
        style_overrides: Optional[Dict[str, Any]] = None,
        *,
        screen_space: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "style_overrides": dict(style_overrides or {}),
            "metadata": dict(metadata or {}),
            "screen_space": screen_space,
        }
        self.calls.append(("draw_text", (text, position, font, color, alignment), payload))


@dataclass
class Offset:
    x: float = 0.0
    y: float = 0.0


class CoordinateMapperStub:
    def __init__(
        self,
        *,
        scale_factor: float = 1.0,
        origin: Tuple[float, float] = (0.0, 0.0),
        offset: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.scale_factor = scale_factor
        self.origin = SimpleNamespace(x=origin[0], y=origin[1])
        self.offset = SimpleNamespace(x=offset[0], y=offset[1])

    def math_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        return x + self.offset.x, y + self.offset.y

    def scale_value(self, value: float) -> float:
        return value * self.scale_factor


class CanvasContextRecorder:
    def __init__(self) -> None:
        self.clear_rect_calls: List[Tuple[float, float, float, float]] = []
        self.draw_image_calls: List[Tuple[Any, float, float]] = []

    def clearRect(self, x: float, y: float, width: float, height: float) -> None:
        self.clear_rect_calls.append((x, y, width, height))

    def drawImage(self, source: Any, dx: float, dy: float) -> None:
        self.draw_image_calls.append((source, dx, dy))

    def getImageData(self, *_args: Any) -> Any:  # pragma: no cover - fallback path unused in tests
        raise NotImplementedError

    def putImageData(self, *_args: Any) -> None:  # pragma: no cover - fallback path unused in tests
        raise NotImplementedError
