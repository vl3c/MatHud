"""Tests for encoding-safe console output (cli/console.py and the root CLI group).

A cp1252 stream stands in for a piped or redirected Windows console, which
cannot encode characters such as "≈" that model replies contain.
"""

from __future__ import annotations

import io
import sys
from unittest.mock import patch

import click
import pytest

from cli.console import make_stdio_safe
from cli.main import cli
from server_tests.test_cli.test_desktop_automation import AVAILABLE, FakeWindow


def cp1252_stream() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict", newline="\n")


def written(stream: io.TextIOWrapper) -> bytes:
    stream.flush()
    buffer = stream.buffer
    assert isinstance(buffer, io.BytesIO)
    return buffer.getvalue()


class ApproxReplyWindow(FakeWindow):
    """A window whose turn replies, and returns a tool result, with "≈" in it."""

    def _finish(self) -> None:
        super()._finish()
        self.texts[-1] = "sqrt(2) ≈ 1.414"
        self.traces[-1]["tool_calls"][0]["result"] = "length ≈ 1.414"


class TestMakeStdioSafe:
    def test_a_cp1252_stream_cannot_encode_approx_on_its_own(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            cp1252_stream().write("≈")

    def test_replaces_what_the_stream_cannot_encode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, err = cp1252_stream(), cp1252_stream()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)

        make_stdio_safe()
        click.echo("x ≈ 1 é")
        click.echo("y ≈ 2", err=True)

        assert out.encoding == "cp1252"  # the encoding is kept; only unencodable characters change
        assert written(out) == "x ? 1 é\n".encode("cp1252")
        assert written(err) == b"y ? 2\n"

    def test_leaves_streams_that_already_handle_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        err = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="backslashreplace", newline="\n")
        monkeypatch.setattr(sys, "stderr", err)
        make_stdio_safe()
        assert err.errors == "backslashreplace"


class TestCliOutputOnACp1252Console:
    def test_desktop_prompt_prints_a_reply_the_console_cannot_encode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out, err = cp1252_stream(), cp1252_stream()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        window = ApproxReplyWindow()
        with (
            patch("cli.desktop_automation.connect_desktop", return_value=window),
            patch("cli.desktop_automation.available_models", return_value=AVAILABLE),
            patch("cli.desktop_automation.time.sleep"),
        ):
            cli.main(["desktop", "prompt", "How long is the diagonal?", "--no-fit-view"], standalone_mode=False)

        text = written(out).decode("cp1252")
        assert "sqrt(2) ? 1.414" in text
        assert "-> length ? 1.414" in text
        assert window.closed
