"""Tests for attach mode (`test scenarios --attach-desktop N`), with fake windows (no browser, no model).

The CDP backend is replaced by the runner tests' fake browsers marked as
attached; the desktop app's HTTP endpoints are patched. Nothing reaches a
window, a server or a provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest
from click.testing import CliRunner

from cli.main import cli
from cli.scenarios import attach as attach_module
from cli.scenarios import command as command_module
from cli.scenarios import runner as runner_module
from cli.scenarios.attach import (
    ATTACHED_PACE_S,
    UNPINNED_SETTINGS,
    confirm,
    confirmation_text,
    default_pace,
    pin_conflicts,
)
from cli.scenarios.report import ResultSink
from cli.scenarios.runner import BrowserSession, ReplayOptions, ReplayRunner

from server_tests.test_cli.test_scenario_live import PROMPT, FakeChatBrowser
from server_tests.test_cli.test_scenario_runner import FakeBrowser, write_catalogue

AVAILABLE = {"local_agent": [{"id": "qwen-local"}], "openai": [{"id": "gpt-5"}]}


class AttachedWindow(FakeChatBrowser):
    """A fake chat browser that behaves like the CDP backend: attached, with the app's URL."""

    attached = True
    base_url = "http://127.0.0.1:5110"

    def __init__(self, busy: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.busy = busy
        self.closes = 0

    def call_hook(self, name: str, *args: Any, timeout: int = 30) -> dict[str, Any]:
        if name == "getMatHudTurnStatus" and self.busy:
            return {"processing": True, "completed_turns": 0, "requests": 0}
        return super().call_hook(name, *args, timeout=timeout)

    def close(self) -> None:
        self.closes += 1


@pytest.fixture
def scenarios_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "scenarios"
    directory.mkdir()
    write_catalogue(directory)
    return directory


@pytest.fixture
def window(monkeypatch: pytest.MonkeyPatch) -> AttachedWindow:
    fake = AttachedWindow()
    monkeypatch.setattr(attach_module, "_connect", lambda port: fake)
    monkeypatch.setattr(attach_module, "CDPBrowser", lambda port: fake)
    monkeypatch.setattr(command_module, "_available_models", lambda base_url: AVAILABLE)
    monkeypatch.setattr(command_module, "_conversation_resetter", lambda base_url: lambda: None)
    monkeypatch.setattr(command_module, "_idle_waiter", lambda base_url: lambda: 0.0)
    monkeypatch.setattr(command_module, "_start_own_server", _no_server)
    monkeypatch.setattr(command_module.ServerManager, "is_server_running", _no_server)
    return fake


def _no_server(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("attach mode must not start or probe a server of its own")


@pytest.fixture
def pauses(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    class FakeTime:
        def __getattr__(self, name: str) -> Any:
            import time

            return getattr(time, name)

        @staticmethod
        def sleep(seconds: float) -> None:
            recorded.append(seconds)

    monkeypatch.setattr(runner_module, "time", FakeTime())
    return recorded


def invoke(scenarios_dir: Path, out: Path, *args: str, stdin: Optional[str] = None) -> Any:
    command = ["test", "scenarios", "--scenarios-dir", str(scenarios_dir), "--out", str(out), *args]
    return CliRunner().invoke(cli, command, input=stdin)


class TestHelpers:
    def test_default_pace(self) -> None:
        assert default_pace(None, attached=True) == ATTACHED_PACE_S
        assert default_pace(None, attached=False) == 0.0
        assert default_pace(0.4, attached=True) == 0.4
        assert default_pace(0.4, attached=False) == 0.4

    def test_pin_conflicts(self) -> None:
        assert pin_conflicts([]) is None
        message = pin_conflicts(["tool_exposure", "canvas_budget"])
        assert message is not None
        assert "--tool-exposure, --canvas-budget cannot be applied with --attach-desktop" in message

    def test_confirm(self) -> None:
        def ask(text: str) -> bool:
            raise AssertionError("must not ask")

        assert confirm("x", yes=True, ask=ask, interactive=False)
        assert not confirm("x", yes=False, ask=ask, interactive=False)
        assert confirm("x", yes=False, ask=lambda text: True, interactive=True)
        assert not confirm("x", yes=False, ask=lambda text: False, interactive=True)

    def test_confirmation_text_names_the_resets_and_openrouter_caveats(self) -> None:
        replay = confirmation_text("http://127.0.0.1:5110", "replay", 3, None)
        assert "resets its canvas, undo history and chat" in replay and "server conversation" not in replay
        live = confirmation_text("http://127.0.0.1:5110", "live", 3, "openrouter")
        assert "server conversation" in live and "TOOL_SEARCH_MODE must be local" in live


class TestSessionAttached:
    def test_open_does_not_reload_an_attached_window_but_a_restart_does(self) -> None:
        browser = FakeBrowser()
        browser.attached = True
        session = BrowserSession(lambda: browser, 5)
        session.open()
        assert browser.reloads == 0
        session.restart()
        assert browser.reloads == 1
        session.kill()
        assert browser.cleaned  # close() only; an attached window just disconnects

    def test_a_browser_of_its_own_is_loaded_on_open(self) -> None:
        browser = FakeBrowser()
        BrowserSession(lambda: browser, 5).open()
        assert browser.reloads == 1

    def test_pace_pauses_after_each_step(self, tmp_path: Path, scenarios_dir: Path, pauses: list[float]) -> None:
        from cli.scenarios.model import load_catalogue

        catalogue = load_catalogue(scenarios_dir)
        sink = ResultSink(tmp_path / "out", {"mode": "replay"})
        runner = ReplayRunner(
            catalogue, BrowserSession(FakeBrowser, 5), sink, ReplayOptions(pace_s=0.7), log=lambda line: None
        )
        runner.run(catalogue.select(ids=["GEO-90"]))
        assert pauses == [0.7, 0.7]  # GEO-90 has no setup calls and two steps


class TestAttachedReplay:
    def test_runs_in_the_window_and_records_the_attachment(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, pauses: list[float]
    ) -> None:
        out = tmp_path / "out"
        result = invoke(scenarios_dir, out, "--attach-desktop", "9301", "--yes")

        assert result.exit_code == 0, result.output
        assert "Attached to the desktop window at http://127.0.0.1:5110" in result.output
        data = json.loads((out / "results.json").read_text())
        config = data["config"]
        assert config["attached_desktop"] is True and config["automation_port"] == 9301
        assert config["port"] == 5110 and config["server_started"] is False and config["pins_applied"] is False
        assert config["pace_s"] == ATTACHED_PACE_S
        statuses = {item["id"]: item.get("skipped_reason") or item["status"] for item in data["scenarios"]}
        assert statuses["GEO-90"] == "pass"
        assert "uses workspaces" in statuses["GEO-91"]
        assert data["summary"]["attached_desktop"]["url"] == "http://127.0.0.1:5110"
        assert "Attached to the desktop app at http://127.0.0.1:5110" in (out / "summary.md").read_text()
        assert pauses and set(pauses) == {ATTACHED_PACE_S}
        assert window.closes >= 1  # the session disconnected; nothing closed the window

    def test_workspace_scenarios_run_only_when_allowed(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, pauses: list[float]
    ) -> None:
        out = tmp_path / "out"
        result = invoke(
            scenarios_dir, out, "--attach-desktop", "9301", "--yes", "--allow-workspace-writes", "--pace", "0"
        )
        data = json.loads((out / "results.json").read_text())
        assert not any(item.get("skipped_reason") for item in data["scenarios"]), result.output
        assert pauses == []

    def test_asks_before_resetting_the_window(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: list[str] = []

        def ask(text: str) -> bool:
            asked.append(text)
            return False

        monkeypatch.setattr(attach_module, "_ask", ask)
        monkeypatch.setattr(attach_module, "_stdin_is_terminal", lambda: True)
        out = tmp_path / "out"
        result = invoke(scenarios_dir, out, "--attach-desktop", "9301")
        assert result.exit_code == 2
        assert "Not confirmed" in result.output
        assert asked and "resets its canvas, undo history and chat" in asked[0]
        assert not (out / "results.json").exists()

    def test_refuses_without_a_terminal_or_yes(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow
    ) -> None:
        result = invoke(scenarios_dir, tmp_path / "out", "--attach-desktop", "9301")
        assert result.exit_code == 2 and "Pass --yes" in result.output

    def test_refuses_while_a_turn_runs(self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow) -> None:
        window.busy = True
        result = invoke(scenarios_dir, tmp_path / "out", "--attach-desktop", "9301", "--yes")
        assert result.exit_code == 2 and "A turn is running" in result.output

    def test_reports_a_missing_window(
        self, tmp_path: Path, scenarios_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing(port: int) -> Any:
            raise RuntimeError("no DevTools endpoint at http://127.0.0.1:9301/json")

        monkeypatch.setattr(attach_module, "_connect", missing)
        result = invoke(scenarios_dir, tmp_path / "out", "--attach-desktop", "9301", "--yes")
        assert result.exit_code == 2 and "Could not attach" in result.output

    @pytest.mark.parametrize(
        "args, message",
        [
            (["--start-server"], "drop --start-server"),
            (["--mode", "retrace", "results.json"], "not retrace"),
            (["--mode", "live", "--tool-exposure", "full"], "--tool-exposure cannot be applied"),
            (["--mode", "live", "--local-reasoning-effort", "high"], "--local-reasoning-effort cannot be applied"),
        ],
    )
    def test_refused_combinations(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, args: list[str], message: str
    ) -> None:
        if "results.json" in args:
            (tmp_path / "results.json").write_text("{}")
            args = [str(tmp_path / "results.json") if a == "results.json" else a for a in args]
        result = invoke(scenarios_dir, tmp_path / "out", "--attach-desktop", "9301", "--yes", *args)
        assert result.exit_code == 2, result.output
        assert message in result.output


class TestAttachedLive:
    def test_runs_the_prompt_in_the_window_with_an_explicit_model(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow, pauses: list[float]
    ) -> None:
        out = tmp_path / "out"
        result = invoke(scenarios_dir, out, "--mode", "live", "--attach-desktop", "9301", "--yes", "--ids", "GEO-90")

        assert result.exit_code == 0, result.output
        assert window.sent == [(PROMPT, "qwen-local")]
        config = json.loads((out / "results.json").read_text())["config"]
        assert config["attached_desktop"] is True and config["models"] == ["qwen-local"]
        assert config["unpinned_settings"] == list(UNPINNED_SETTINGS)
        assert "server_env" not in config  # nothing was pinned

    def test_unregistered_model_aborts_before_any_message(
        self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow
    ) -> None:
        result = invoke(scenarios_dir, tmp_path / "out", "--mode", "live", "--attach-desktop", "9301", "--yes",
                        "--models", "gpt-5")  # fmt: skip
        assert result.exit_code == 2 and "Aborting before the first message" in result.output
        assert window.sent == []

    def test_openrouter_keeps_its_guards(self, tmp_path: Path, scenarios_dir: Path, window: AttachedWindow) -> None:
        base = ["--mode", "live", "--attach-desktop", "9301", "--yes", "--provider", "openrouter"]
        no_models = invoke(scenarios_dir, tmp_path / "a", *base)
        assert no_models.exit_code == 2 and "needs --models" in no_models.output
        over_cap = invoke(scenarios_dir, tmp_path / "b", *base, "--models", "x/y", "--max-requests", "1")
        assert over_cap.exit_code == 2 and "nothing was sent" in over_cap.output
        dry = invoke(scenarios_dir, tmp_path / "c", *base, "--models", "x/y", "--dry-run", "--json")
        assert dry.exit_code == 0, dry.output
        assert json.loads(dry.output)["attached_desktop"] == 9301
        assert window.sent == []
