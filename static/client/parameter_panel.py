"""The slider panel: one row per parameter over the canvas (name, slider, value, play/pause).

The panel is ``div#parameter-panel`` inside ``#math-container`` (templates/index.html),
hidden when there are no parameters. It is an accelerator: everything it does, the AI
does with set_parameter and animate_parameter.

- Rows are rebuilt when parameters are created, deleted, redefined or restored (the
  manager's change listeners), and their values are updated after every canvas draw.
- Dragging a slider: the first ``input`` of a drag stops the parameter's animation and
  archives once; each ``input`` sets the value live (snapped to the step in Python, the
  element itself takes any value) and asks for one draw per animation frame; ``change``
  (release, or a keyboard step) ends the drag with the full analysis.
- Play bounces the parameter between its slider ends over 4 s through the same path as
  animate_parameter; pause stops it.
- Pointer and wheel events on the panel never reach the canvas, so a drag never pans it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from browser import document, html, window

from drawables.parameter import format_value, snap_to_step

if TYPE_CHECKING:
    from canvas import Canvas
    from managers.parameter_manager import ParameterManager

PANEL_ID = "parameter-panel"
PLAY_SECONDS = 4.0
_PLAY = "▶"
_PAUSE = "❚❚"
_STOPPED_EVENTS = (
    "mousedown",
    "mousemove",
    "mouseup",
    "touchstart",
    "touchmove",
    "touchend",
    "wheel",
    "pointerdown",
    "dblclick",
    "click",
)


class _Row:
    __slots__ = ("name", "slider", "value_text", "play")

    def __init__(self, name: str, slider: Any, value_text: Any, play: Any) -> None:
        self.name = name
        self.slider = slider
        self.value_text = value_text
        self.play = play


class ParameterPanel:
    """Keeps the slider panel in step with a canvas's parameters."""

    def __init__(
        self,
        manager: "ParameterManager",
        canvas: "Canvas",
        element: Optional[Any] = None,
        request_frame: Optional[Callable[[Callable[[float], None]], Any]] = None,
    ) -> None:
        self.manager = manager
        self.canvas = canvas
        self.element = element if element is not None else self._find_or_create_element()
        self._request_frame = request_frame or (lambda callback: window.requestAnimationFrame(callback))
        self._rows: Dict[str, _Row] = {}
        self._layout: tuple = ()
        self._dragging: Optional[str] = None
        self._draw_pending = False

    @staticmethod
    def _find_or_create_element() -> Any:
        if PANEL_ID in document:
            return document[PANEL_ID]
        element = html.DIV(id=PANEL_ID, Class="parameter-panel")
        document["math-container"] <= element
        return element

    def attach(self) -> "ParameterPanel":
        """Listen to the manager and the canvas, stop events reaching the canvas, show the rows."""
        self.manager.add_change_listener(self.rebuild)
        self.canvas.add_draw_listener(self.update_values)
        for event in _STOPPED_EVENTS:
            self.element.bind(event, self._stop_event)
        self.rebuild()
        return self

    @staticmethod
    def _stop_event(event: Any) -> None:
        event.stopPropagation()

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------

    def rebuild(self) -> None:
        """Rebuild the rows when the parameters or their ranges changed; else update values."""
        parameters = self.manager.parameters
        layout = tuple((p.name, p.min, p.max, p.step) for p in parameters)
        display = "block" if parameters else "none"
        if self.element.style.display != display:
            self.element.style.display = display
        if layout == self._layout and len(self._rows) == len(parameters):
            self.update_values()
            return
        self._layout = layout
        self._rows = {}
        self.element.clear()
        for parameter in parameters:
            row = self._build_row(parameter)
            self._rows[parameter.name] = row
        self.update_values()

    def _build_row(self, parameter: Any) -> _Row:
        name = parameter.name
        label = html.SPAN(name, Class="parameter-name")
        slider = html.INPUT(type="range", Class="parameter-slider")
        slider.attrs["min"] = str(parameter.min)
        slider.attrs["max"] = str(parameter.max)
        slider.attrs["step"] = "any"
        slider.attrs["aria-label"] = f"parameter {name}"
        slider.value = str(parameter.value)
        value_text = html.SPAN("", Class="parameter-value")
        play = html.BUTTON(_PLAY, Class="parameter-play", type="button")
        play.attrs["title"] = f"Animate {name}"
        slider.bind("input", lambda event, n=name: self._on_input(n, event))
        slider.bind("change", lambda event, n=name: self._on_change(n))
        play.bind("click", lambda event, n=name: self._on_play(n))
        row = html.DIV(Class="parameter-row")
        row <= label
        row <= slider
        row <= value_text
        row <= play
        self.element <= row
        return _Row(name, slider, value_text, play)

    def update_values(self) -> None:
        """Show each parameter's value and play state (after every draw; cheap when unchanged)."""
        for name, row in self._rows.items():
            parameter = self.manager.get_parameter(name)
            if parameter is None:
                self._layout = ()
                self.rebuild()
                return
            text = f"= {format_value(parameter.value, parameter.step)}"
            if row.value_text.text != text:
                row.value_text.text = text
            if self._dragging != name:
                position = str(parameter.value)
                if row.slider.value != position:
                    row.slider.value = position
            symbol = _PAUSE if parameter.animation else _PLAY
            if row.play.text != symbol:
                row.play.text = symbol
                row.play.attrs["title"] = f"{'Pause' if parameter.animation else 'Animate'} {name}"

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _on_input(self, name: str, event: Any) -> None:
        parameter = self.manager.get_parameter(name)
        if parameter is None:
            return
        if self._dragging != name:
            if self._dragging is not None:
                self._on_change(self._dragging)
            self._dragging = name
            self.manager.begin_live_change(name)
        try:
            position = float(event.target.value)
        except (TypeError, ValueError):
            return
        value = snap_to_step(position, parameter.min, parameter.max, parameter.step)
        if self.manager.apply_live_value(name, value):
            self._schedule_draw()

    def _on_change(self, name: str) -> None:
        if self._dragging == name:
            self._dragging = None
        self.manager.settle([name])

    def _on_play(self, name: str) -> None:
        parameter = self.manager.get_parameter(name)
        if parameter is None:
            return
        try:
            if parameter.animation:
                self.manager.stop_animation(name)
            else:
                self.manager.animate_parameter(name, None, None, PLAY_SECONDS, "bounce")
        except ValueError as exc:
            print(f"Warning: {exc}")
        self.update_values()

    def _schedule_draw(self) -> None:
        if self._draw_pending:
            return
        self._draw_pending = True
        self._request_frame(self._draw_frame)

    def _draw_frame(self, _timestamp: float) -> None:
        self._draw_pending = False
        try:
            self.canvas.draw()
        except Exception as exc:
            print(f"Warning: slider redraw failed: {exc}")
