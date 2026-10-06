"""Tests for driving the desktop window from the CLI (cli/desktop_automation.py, `desktop` subcommands).

A fake window stands in for the CDP-attached page: no browser, no window, no model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cli.desktop_automation import (
    DesktopError,
    choose_prompt_model,
    format_prompt_result,
    run_prompt,
)
from cli.main import cli
from cli.scenarios.live_config import GuardError

AVAILABLE = {
    "local_agent": [{"id": "qwen-local"}],
    "openai": [{"id": "gpt-5"}],
    "openrouter_paid": [{"id": "deepseek/deepseek-v4.1-flash"}],
}


class FakeWindow:
    """The hooks a prompt uses, with a turn that runs for ``turn_polls`` status polls."""

    attached = True
    base_url = "http://127.0.0.1:5110"

    def __init__(self, turn_polls: int = 2, never_ends: bool = False, busy: bool = False) -> None:
        self.turn_polls = turn_polls
        self.never_ends = never_ends or busy  # a busy window's own turn runs on
        self.processing = busy
        self.completed = 4
        self.polls_left = 0
        self.texts = ["an earlier reply"]
        self.traces: list[dict[str, Any]] = [{"trace_id": "old-1", "tool_calls": [{"function_name": "zoom"}]}]
        self.sent: list[tuple[Any, ...]] = []
        self.stopped = False
        self.closed = False
        self.metrics: Optional[dict[str, Any]] = {"turn_id": 4, "outcome": "stop", "requests": 9}

    def call_hook(self, name: str, *args: Any, timeout: int = 30) -> dict[str, Any]:
        if name == "getMatHudTurnStatus":
            if self.processing and not self.never_ends:
                self.polls_left -= 1
                if self.polls_left <= 0:
                    self._finish()
            return {"processing": self.processing, "completed_turns": self.completed}
        if name == "sendMatHudMessage":
            self.sent.append(args)
            self.processing = True
            self.polls_left = self.turn_polls
            return {"status": "started"}
        if name == "stopMatHudTurn":
            self.stopped = True
            self.processing = False
            return {"status": "stopped"}
        raise AssertionError(name)

    def _finish(self) -> None:
        self.processing = False
        self.completed += 1
        self.texts.append("Created point A.")
        self.traces.append(
            {
                "trace_id": "new-1",
                "tool_calls": [
                    {"function_name": "create_point", "arguments": {"x": 2, "y": 3, "name": "A"}, "result": "ok"}
                ],
            }
        )
        self.metrics = {"turn_id": self.completed, "outcome": "stop", "requests": 2, "wall_time_s": 1.5}

    def execute_js(self, script: str, *args: Any, timeout: float = 30) -> Any:
        if "chat-message" in script:
            return list(self.texts)
        if "getActionTraces" in script:
            return json.dumps(self.traces)
        if "getMatHudLastTurnMetrics" in script:
            return json.dumps(self.metrics)
        raise AssertionError(script)

    def close(self) -> None:
        self.closed = True


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TestChooseModel:
    def test_explicit_local_model_must_be_registered(self) -> None:
        assert choose_prompt_model(AVAILABLE, "local", "qwen-local") == "qwen-local"
        with pytest.raises(GuardError, match="gpt-5"):
            choose_prompt_model(AVAILABLE, "local", "gpt-5")

    def test_default_is_the_single_local_model(self) -> None:
        assert choose_prompt_model(AVAILABLE, "local", None) == "qwen-local"

    def test_refuses_to_guess_between_several_local_models(self) -> None:
        available = {"local_agent": [{"id": "a"}, {"id": "b"}]}
        with pytest.raises(GuardError, match="several local models"):
            choose_prompt_model(available, "local", None)

    def test_no_local_model_is_an_error(self) -> None:
        with pytest.raises(GuardError, match="no local_agent model"):
            choose_prompt_model({"openai": [{"id": "gpt-5"}]}, "local", None)

    def test_openrouter_needs_an_explicit_registered_model(self) -> None:
        with pytest.raises(GuardError, match="needs --model"):
            choose_prompt_model(AVAILABLE, "openrouter", None)
        assert (
            choose_prompt_model(AVAILABLE, "openrouter", "deepseek/deepseek-v4.1-flash")
            == "deepseek/deepseek-v4.1-flash"
        )
        with pytest.raises(GuardError):
            choose_prompt_model(AVAILABLE, "openrouter", "qwen-local")


class TestRunPrompt:
    def test_sends_with_the_model_and_collects_only_this_turn(self) -> None:
        window, clock = FakeWindow(), FakeClock()
        result = run_prompt(window, "Create A", "qwen-local", clock=clock, sleep=clock.sleep)  # type: ignore[arg-type]

        text, model, options = window.sent[0]
        assert (text, model) == ("Create A", "qwen-local")
        assert json.loads(options)["max_requests"] == 8
        assert result["outcome"] == "stop"
        assert result["stop_reason"] is None
        assert result["final_text"] == "Created point A."
        assert result["assistant_texts"] == ["Created point A."]
        assert [c["function_name"] for c in result["tool_calls"]] == ["create_point"]
        assert result["metrics"]["requests"] == 2

    def test_timeout_stops_the_turn(self) -> None:
        window, clock = FakeWindow(never_ends=True), FakeClock()
        result = run_prompt(window, "Create A", "qwen-local", timeout_s=2, clock=clock, sleep=clock.sleep)  # type: ignore[arg-type]

        assert window.stopped
        assert result["outcome"] == "timeout"
        assert result["metrics"] is None  # no turn finished after the prompt

    def test_refuses_while_a_turn_runs(self) -> None:
        with pytest.raises(DesktopError, match="already running"):
            run_prompt(FakeWindow(busy=True), "Create A", "qwen-local")  # type: ignore[arg-type]

    def test_refuses_slash_commands_and_empty_prompts(self) -> None:
        with pytest.raises(DesktopError, match="slash"):
            run_prompt(FakeWindow(), "/clear", "qwen-local")  # type: ignore[arg-type]
        with pytest.raises(DesktopError, match="empty"):
            run_prompt(FakeWindow(), "  ", "qwen-local")  # type: ignore[arg-type]

    def test_format_lists_reply_calls_and_metrics(self) -> None:
        window, clock = FakeWindow(), FakeClock()
        result = run_prompt(window, "Create A", "qwen-local", clock=clock, sleep=clock.sleep)  # type: ignore[arg-type]
        text = format_prompt_result(result)
        assert "Created point A." in text
        assert 'create_point({"x": 2, "y": 3, "name": "A"}) -> ok' in text
        assert "requests 2" in text


class TestDesktopCommands:
    def test_help_lists_the_subcommands(self) -> None:
        result = CliRunner().invoke(cli, ["desktop", "--help"])
        assert result.exit_code == 0
        for name in ("prompt", "state", "screenshot", "--automation-port"):
            assert name in result.output

    def test_prompt_prints_json(self) -> None:
        window = FakeWindow()
        with (
            patch("cli.desktop_automation.connect_desktop", return_value=window) as connect,
            patch("cli.desktop_automation.available_models", return_value=AVAILABLE),
            patch("cli.desktop_automation.time.sleep"),
        ):
            result = CliRunner().invoke(cli, ["desktop", "prompt", "Create A", "--port", "9301", "--json"])

        assert result.exit_code == 0, result.output
        connect.assert_called_once_with(9301)
        payload = json.loads(result.output)
        assert payload["model"] == "qwen-local"
        assert payload["final_text"] == "Created point A."
        assert window.closed

    def test_prompt_refuses_an_unregistered_model_before_sending(self) -> None:
        window = FakeWindow()
        with (
            patch("cli.desktop_automation.connect_desktop", return_value=window),
            patch("cli.desktop_automation.available_models", return_value=AVAILABLE),
        ):
            result = CliRunner().invoke(cli, ["desktop", "prompt", "Create A", "--model", "gpt-5"])

        assert result.exit_code == 2
        assert "Not sent" in result.output
        assert window.sent == []
        assert window.closed

    def test_prompt_reports_a_missing_window(self) -> None:
        with patch("cli.desktop_automation.connect_desktop", side_effect=DesktopError("no DevTools endpoint")):
            result = CliRunner().invoke(cli, ["desktop", "prompt", "Create A"])
        assert result.exit_code == 2
        assert "no DevTools endpoint" in result.output

    def test_state_and_screenshot(self, tmp_path: Path) -> None:
        class StateWindow(FakeWindow):
            def get_canvas_snapshot(self, options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
                return {"state": {"Points": [{"name": "A"}]}, "inspection": {"undo_depth": 1}}

            def screenshot(self, path: str) -> bool:
                Path(path).write_bytes(b"png")
                return True

        with patch("cli.desktop_automation.connect_desktop", return_value=StateWindow()):
            state = CliRunner().invoke(cli, ["desktop", "state"])
            inspected = CliRunner().invoke(cli, ["desktop", "state", "--inspect"])
            shot = CliRunner().invoke(cli, ["desktop", "screenshot", "-o", str(tmp_path / "w.png")])

        assert json.loads(state.output) == {"Points": [{"name": "A"}]}
        assert json.loads(inspected.output)["inspection"] == {"undo_depth": 1}
        assert shot.exit_code == 0 and (tmp_path / "w.png").read_bytes() == b"png"
