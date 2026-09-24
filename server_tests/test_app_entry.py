"""Tests for the ``python app.py`` entry point's server-thread supervision."""

from __future__ import annotations

import signal
from types import ModuleType
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def app_module() -> Iterator[ModuleType]:
    """Import app.py once, restoring the SIGINT handler it installs."""
    previous_sigint = signal.getsignal(signal.SIGINT)
    try:
        with patch("builtins.print"):
            import app
        yield app
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


def _thread(*alive_states: bool) -> MagicMock:
    """A stand-in server thread whose is_alive() returns ``alive_states``, then the last one."""
    thread = MagicMock()
    thread.is_alive.side_effect = [*alive_states] + [alive_states[-1]] * 20
    return thread


class TestKeepServing:
    def test_server_that_fails_to_start_exits_non_zero(self, app_module: ModuleType) -> None:
        # e.g. the port is already taken: the server thread ends right away.
        with (
            patch("mathud_desktop.wait_for_server", return_value=False) as wait,
            patch("builtins.print") as mock_print,
        ):
            assert app_module._keep_serving(_thread(False), "http://127.0.0.1:1/") == 1

        assert wait.call_args.kwargs["alive"] is not None
        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
        assert "failed to start" in printed
        assert "is running" not in printed

    def test_server_that_stops_later_exits_non_zero(self, app_module: ModuleType) -> None:
        with (
            patch("mathud_desktop.wait_for_server", return_value=True),
            patch.object(app_module.time, "sleep") as sleep,
            patch("builtins.print") as mock_print,
        ):
            assert app_module._keep_serving(_thread(True, True, True, False), "http://127.0.0.1:1/") == 1

        assert sleep.call_count >= 1
        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
        assert "is running" in printed
        assert "stopped unexpectedly" in printed

    def test_slow_server_is_reported_but_kept(self, app_module: ModuleType) -> None:
        with (
            patch("mathud_desktop.wait_for_server", return_value=False),
            patch.object(app_module.time, "sleep"),
            patch("builtins.print") as mock_print,
        ):
            assert app_module._keep_serving(_thread(True, True, False), "http://127.0.0.1:1/") == 1

        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
        assert "did not respond" in printed
        assert "is running" in printed
