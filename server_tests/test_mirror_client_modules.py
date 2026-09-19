"""
Tests for the mirror client modules utilities.

Tests the _mirror_if_stale helper for mirroring client-side modules into the
server package and the public ensure_polygon_subtypes_available helper, using
only files under pytest's tmp_path.
"""

import importlib
import os
from pathlib import Path
from unittest import mock

import pytest

from static.mirror_client_modules import (
    POLYGON_SUBTYPES_MODULE,
    _mirror_if_stale,
    ensure_polygon_subtypes_available,
)


class TestMirrorIfStale:
    """Tests for the _mirror_if_stale helper."""

    def test_creates_missing_destination_with_source_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Creates a missing destination with the source's exact bytes."""
        invalidate_caches = mock.Mock()
        monkeypatch.setattr(importlib, "invalidate_caches", invalidate_caches)
        source = tmp_path / "source.py"
        source.write_bytes(b"print('constants')")
        destination = tmp_path / "destination.py"

        _mirror_if_stale(source, destination)

        assert destination.read_bytes() == b"print('constants')"
        assert invalidate_caches.call_count == 1

    def test_creates_destination_parent_folder(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Creates the destination's parent folder when it is missing."""
        invalidate_caches = mock.Mock()
        monkeypatch.setattr(importlib, "invalidate_caches", invalidate_caches)
        source = tmp_path / "source.py"
        source.write_bytes(b"source bytes")
        destination = tmp_path / "a" / "b" / "destination.py"

        _mirror_if_stale(source, destination)

        assert destination.read_bytes() == b"source bytes"
        assert invalidate_caches.call_count == 1

    def test_overwrites_destination_when_source_is_newer(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Overwrites a destination with an older mtime with the source bytes."""
        invalidate_caches = mock.Mock()
        monkeypatch.setattr(importlib, "invalidate_caches", invalidate_caches)
        source = tmp_path / "source.py"
        destination = tmp_path / "destination.py"
        source.write_bytes(b"source bytes")
        destination.write_bytes(b"destination bytes")
        os.utime(source, (2000, 2000))
        os.utime(destination, (1000, 1000))

        _mirror_if_stale(source, destination)

        assert destination.read_bytes() == b"source bytes"
        assert invalidate_caches.call_count == 1

    def test_leaves_destination_unchanged_when_destination_is_newer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Leaves a destination with a newer mtime unchanged."""
        invalidate_caches = mock.Mock()
        monkeypatch.setattr(importlib, "invalidate_caches", invalidate_caches)
        source = tmp_path / "source.py"
        destination = tmp_path / "destination.py"
        source.write_bytes(b"source bytes")
        destination.write_bytes(b"destination bytes")
        os.utime(source, (1000, 1000))
        os.utime(destination, (2000, 2000))

        _mirror_if_stale(source, destination)

        assert destination.read_bytes() == b"destination bytes"
        assert invalidate_caches.call_count == 0

    def test_leaves_destination_unchanged_when_mtime_is_equal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Leaves a destination with an equal mtime unchanged."""
        invalidate_caches = mock.Mock()
        monkeypatch.setattr(importlib, "invalidate_caches", invalidate_caches)
        source = tmp_path / "source.py"
        destination = tmp_path / "destination.py"
        source.write_bytes(b"source bytes")
        destination.write_bytes(b"destination bytes")
        os.utime(source, (1500, 1500))
        os.utime(destination, (1500, 1500))

        _mirror_if_stale(source, destination)

        assert destination.read_bytes() == b"destination bytes"
        assert invalidate_caches.call_count == 0

    def test_raises_runtime_error_when_source_is_missing(self, tmp_path: Path) -> None:
        """Raises a RuntimeError when the source file is missing."""
        source = tmp_path / "missing.py"
        destination = tmp_path / "destination.py"

        with pytest.raises(RuntimeError) as exc_info:
            _mirror_if_stale(source, destination)

        message = str(exc_info.value)
        assert message.startswith("Client module not found")

    def test_raises_runtime_error_when_copy_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises a RuntimeError containing the source name when copying fails."""
        source = tmp_path / "special_source.py"
        source.write_bytes(b"data")
        destination = tmp_path / "destination.py"
        destination.write_bytes(b"old data")
        os.utime(source, (2000, 2000))
        os.utime(destination, (1000, 1000))

        def failing_copy(_source: Path, _destination: Path) -> None:
            raise OSError("disk full")

        monkeypatch.setattr("static.mirror_client_modules.shutil.copyfile", failing_copy)

        with pytest.raises(RuntimeError) as exc_info:
            _mirror_if_stale(source, destination)

        message = str(exc_info.value)
        assert "Failed to synchronize client module" in message
        assert "special_source.py" in message


class TestEnsurePolygonSubtypesAvailable:
    """Tests for the ensure_polygon_subtypes_available helper."""

    def test_makes_polygon_subtypes_module_importable(self) -> None:
        """Mirrors polygon subtypes so the module can be imported by name."""
        ensure_polygon_subtypes_available()

        module = importlib.import_module(POLYGON_SUBTYPES_MODULE)
        assert module is not None
