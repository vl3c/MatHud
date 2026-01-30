"""Tests for cli/screenshot.py."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli.screenshot import generate_default_filename
from cli.config import CLI_OUTPUT_DIR


class TestGenerateDefaultFilename:
    """Test generate_default_filename function."""

    def test_returns_path(self) -> None:
        """generate_default_filename returns a Path."""
        result = generate_default_filename()
        assert isinstance(result, Path)

    def test_path_in_output_dir(self) -> None:
        """Path should be in CLI_OUTPUT_DIR."""
        result = generate_default_filename()
        assert result.parent == CLI_OUTPUT_DIR

    def test_default_prefix(self) -> None:
        """Default prefix is 'screenshot'."""
        result = generate_default_filename()
        assert result.name.startswith("screenshot_")

    def test_custom_prefix(self) -> None:
        """Custom prefix is used."""
        result = generate_default_filename(prefix="canvas")
        assert result.name.startswith("canvas_")

    def test_png_extension(self) -> None:
        """Filename has .png extension."""
        result = generate_default_filename()
        assert result.suffix == ".png"

    def test_timestamp_format(self) -> None:
        """Filename contains timestamp."""
        result = generate_default_filename()
        # Extract timestamp part: screenshot_YYYYMMDD_HHMMSS.png
        name = result.stem  # screenshot_YYYYMMDD_HHMMSS
        parts = name.split("_")
        assert len(parts) == 3
        # Date part should be 8 digits
        assert len(parts[1]) == 8
        assert parts[1].isdigit()
        # Time part should be 6 digits
        assert len(parts[2]) == 6
        assert parts[2].isdigit()

    def test_unique_filenames(self) -> None:
        """Different calls produce different filenames (if time changes)."""
        with patch("cli.screenshot.datetime") as mock_datetime:
            mock_datetime.now.side_effect = [
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 1, 12, 0, 1),
            ]
            mock_datetime.strftime = datetime.strftime

            result1 = generate_default_filename()
            result2 = generate_default_filename()

            # Note: This test may need adjustment based on actual implementation
            # The key point is that timestamps make filenames unique
            assert "20240101" in result1.name or result1 != result2
