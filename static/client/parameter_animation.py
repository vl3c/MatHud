"""Parameter animations on requestAnimationFrame: one frame loop for every running animation.

``ParameterAnimator.start(name, start, end, seconds, repeat)`` moves a parameter linearly in
time from start to end ("once"), again and again ("loop") or back and forth ("bounce").
Each frame sets every running value through ``ParameterManager.apply_live_value`` (no undo
entry), then draws once. A "once" run ends exactly on its end value; elapsed time comes
from the frame timestamps, so a hidden tab (frames paused) catches up when it is shown.

The frame scheduler and the clock are injected, so tests drive frames by hand.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from managers.parameter_manager import ParameterManager


class _Run:
    __slots__ = ("start", "end", "seconds", "repeat", "started_at")

    def __init__(self, start: float, end: float, seconds: float, repeat: str) -> None:
        self.start = start
        self.end = end
        self.seconds = seconds
        self.repeat = repeat
        self.started_at: Optional[float] = None

    def value_at(self, timestamp_ms: float) -> "tuple[float, bool]":
        """The value at a frame timestamp, and whether the run is over."""
        if self.started_at is None:
            self.started_at = timestamp_ms
        progress = max(0.0, (timestamp_ms - self.started_at) / (self.seconds * 1000.0))
        if self.repeat == "once":
            if progress >= 1.0:
                return self.end, True
            fraction = progress
        elif self.repeat == "loop":
            fraction = progress - math.floor(progress)
        else:  # bounce
            phase = progress - 2.0 * math.floor(progress / 2.0)
            fraction = phase if phase <= 1.0 else 2.0 - phase
        return self.start + (self.end - self.start) * fraction, False


def _browser_request_frame(callback: Callable[[float], None]) -> Any:
    from browser import window

    return window.requestAnimationFrame(callback)


class ParameterAnimator:
    """Runs parameter animations, one requestAnimationFrame loop for all of them."""

    def __init__(
        self,
        manager: "ParameterManager",
        draw: Callable[[], None],
        request_frame: Callable[[Callable[[float], None]], Any] = _browser_request_frame,
    ) -> None:
        self.manager = manager
        self._draw = draw
        self._request_frame = request_frame
        self._runs: Dict[str, _Run] = {}
        self._frame_pending = False

    @property
    def running(self) -> List[str]:
        """The names of the parameters animating now."""
        return list(self._runs)

    def start(self, name: str, start: float, end: float, seconds: float, repeat: str) -> None:
        """Start (or restart) the animation of one parameter; its first frame sets start."""
        self._runs[name] = _Run(float(start), float(end), float(seconds), repeat)
        self._schedule()

    def stop(self, name: str) -> bool:
        """Stop one animation; True if it was running."""
        return self._runs.pop(name, None) is not None

    def stop_all(self) -> None:
        self._runs.clear()

    def _schedule(self) -> None:
        if self._frame_pending or not self._runs:
            return
        self._frame_pending = True
        self._request_frame(self.frame)

    def frame(self, timestamp_ms: float) -> None:
        """One animation frame: set every running value, draw once, finish ended runs.

        A failure ends every animation cleanly (values kept, full analysis, panel updated)
        instead of leaving a loop that throws on each frame.
        """
        self._frame_pending = False
        if not self._runs:
            return
        try:
            finished = self._advance(float(timestamp_ms))
        except Exception as exc:
            print(f"Warning: animation frame failed, stopping animations: {exc}")
            finished = list(self._runs)
            self._runs.clear()
        for name in finished:
            try:
                self.manager.animation_finished(name)
            except Exception as exc:
                print(f"Warning: could not finish the animation of {name}: {exc}")
        self._schedule()

    def _advance(self, timestamp_ms: float) -> List[str]:
        """Set each running value and draw; returns the runs that ended (removed)."""
        finished: List[str] = []
        for name, run in list(self._runs.items()):
            value, done = run.value_at(timestamp_ms)
            if not self.manager.apply_live_value(name, value) or done:
                finished.append(name)  # ended, or the parameter is gone (cleared or deleted)
        for name in finished:
            self._runs.pop(name, None)
        self._draw()
        return finished
