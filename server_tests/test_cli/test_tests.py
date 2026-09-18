"""Tests for cli/tests.py."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cli.main import cli
from cli.tests import install_pre_commit_hook

HOOK_CONTENT = "#!/bin/sh\necho hook\n"


def _make_project(root: Path) -> Path:
    """Create a project root containing hooks/pre-commit."""
    (root / "hooks").mkdir(parents=True)
    (root / "hooks" / "pre-commit").write_text(HOOK_CONTENT)
    return root


def _git_result(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


class TestInstallPreCommitHook:
    """Test install_pre_commit_hook function."""

    def test_relative_hooks_path_resolved_against_project_root(self, tmp_path: Path) -> None:
        """A relative git path (main checkout) is resolved against PROJECT_ROOT and created."""
        project = _make_project(tmp_path / "project")
        with (
            patch("cli.tests.PROJECT_ROOT", project),
            patch("cli.tests.subprocess.run", return_value=_git_result(".git/hooks\n")) as mock_run,
        ):
            assert install_pre_commit_hook() is True

        assert mock_run.call_args.args[0] == ["git", "rev-parse", "--git-path", "hooks"]
        assert mock_run.call_args.kwargs["cwd"] == str(project)
        assert (project / ".git" / "hooks" / "pre-commit").read_text() == HOOK_CONTENT

    def test_absolute_hooks_path_used_as_is(self, tmp_path: Path) -> None:
        """An absolute git path (linked worktree or core.hooksPath) is used directly."""
        project = _make_project(tmp_path / "worktree")
        shared_hooks = tmp_path / "main" / ".git" / "hooks"
        with (
            patch("cli.tests.PROJECT_ROOT", project),
            patch("cli.tests.subprocess.run", return_value=_git_result(f"{shared_hooks.as_posix()}\n")),
        ):
            assert install_pre_commit_hook() is True

        assert (shared_hooks / "pre-commit").read_text() == HOOK_CONTENT

    def test_git_unavailable_returns_false(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A missing git executable reports a clean error instead of raising."""
        project = _make_project(tmp_path / "project")
        with (
            patch("cli.tests.PROJECT_ROOT", project),
            patch("cli.tests.subprocess.run", side_effect=FileNotFoundError("git not found")),
        ):
            assert install_pre_commit_hook() is False

        assert "Could not run git" in capsys.readouterr().err

    def test_git_error_returns_false(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A failing git command reports git's stderr."""
        project = _make_project(tmp_path / "project")
        with (
            patch("cli.tests.PROJECT_ROOT", project),
            patch(
                "cli.tests.subprocess.run",
                return_value=_git_result(stderr="fatal: not a git repository\n", returncode=128),
            ),
        ):
            assert install_pre_commit_hook() is False

        assert "fatal: not a git repository" in capsys.readouterr().err

    def test_missing_source_returns_false(self, tmp_path: Path) -> None:
        """Without hooks/pre-commit nothing is installed and git is not queried."""
        with (
            patch("cli.tests.PROJECT_ROOT", tmp_path),
            patch("cli.tests.subprocess.run") as mock_run,
        ):
            assert install_pre_commit_hook() is False

        mock_run.assert_not_called()

    def test_sets_executable_bits_on_unix(self, tmp_path: Path) -> None:
        """On non-Windows platforms the installed hook is made executable."""
        project = _make_project(tmp_path / "project")
        with (
            patch("cli.tests.PROJECT_ROOT", project),
            patch("cli.tests.subprocess.run", return_value=_git_result(".git/hooks\n")),
            patch("cli.tests.sys.platform", "linux"),
            patch.object(Path, "chmod") as mock_chmod,
        ):
            assert install_pre_commit_hook() is True

        mode = mock_chmod.call_args.args[0]
        assert mode & 0o111 == 0o111

    def test_skips_chmod_on_windows(self, tmp_path: Path) -> None:
        """On Windows the executable bits are not touched."""
        project = _make_project(tmp_path / "project")
        with (
            patch("cli.tests.PROJECT_ROOT", project),
            patch("cli.tests.subprocess.run", return_value=_git_result(".git/hooks\n")),
            patch("cli.tests.sys.platform", "win32"),
            patch.object(Path, "chmod") as mock_chmod,
        ):
            assert install_pre_commit_hook() is True

        mock_chmod.assert_not_called()

    @pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
    def test_installs_into_shared_hooks_from_linked_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """From a real linked worktree the hook lands in the main repository's hooks dir."""
        # Isolate from user/system git config (e.g. a global core.hooksPath).
        empty_config = tmp_path / "gitconfig"
        empty_config.write_text("")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_config))
        monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

        main = tmp_path / "main"
        main.mkdir()
        _git(main, "init", "-q")
        _git(
            main,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "init",
        )
        worktree = tmp_path / "wt"
        _git(main, "worktree", "add", "-q", "--detach", str(worktree))
        _make_project(worktree)
        assert (worktree / ".git").is_file()

        with patch("cli.tests.PROJECT_ROOT", worktree):
            assert install_pre_commit_hook() is True

        assert (main / ".git" / "hooks" / "pre-commit").read_text() == HOOK_CONTENT


@patch("cli.tests.install_pre_commit_hook")
def test_lint_install_hook_exit_codes(mock_install: MagicMock) -> None:
    """`test lint --install-hook` exits 0 on success and 1 on failure."""
    runner = CliRunner()
    mock_install.return_value = True
    assert runner.invoke(cli, ["test", "lint", "--install-hook"]).exit_code == 0
    mock_install.return_value = False
    assert runner.invoke(cli, ["test", "lint", "--install-hook"]).exit_code == 1
