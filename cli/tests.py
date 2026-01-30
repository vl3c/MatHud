"""Test execution commands for the MatHud CLI.

Provides commands for running server tests (pytest) and client tests (Brython via browser).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click

from cli.browser import BrowserAutomation
from cli.config import (
    CLIENT_TEST_POLL_INTERVAL,
    CLIENT_TEST_TIMEOUT,
    DEFAULT_PORT,
    PROJECT_ROOT,
    get_python_path,
)
from cli.server import ServerManager


def run_server_tests(
    test_path: Optional[str] = None,
    keyword: Optional[str] = None,
    with_auth: bool = False,
    verbose: bool = True,
    extra_args: Optional[list[str]] = None,
) -> int:
    """Run server-side pytest tests.

    Args:
        test_path: Specific test file or directory to run.
        keyword: Keyword expression for test filtering (-k).
        with_auth: Enable authentication during tests.
        verbose: Enable verbose output.
        extra_args: Additional arguments to pass to pytest.

    Returns:
        pytest exit code (0 for success).
    """
    python_path = get_python_path()
    if not python_path.exists():
        click.echo(click.style(f"Python not found at {python_path}", fg="red"), err=True)
        return 1

    # Build pytest command
    cmd = [str(python_path), "-m", "pytest"]

    # Test path
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("server_tests")

    # Verbose flag
    if verbose:
        cmd.append("-v")

    # Keyword filter
    if keyword:
        cmd.extend(["-k", keyword])

    # Extra args
    if extra_args:
        cmd.extend(extra_args)

    # Set up environment
    env = os.environ.copy()
    if not with_auth:
        env["REQUIRE_AUTH"] = "false"

    # Add client path to PYTHONPATH for pure-Python tests
    client_root = PROJECT_ROOT / "static" / "client"
    if client_root.exists():
        existing = env.get("PYTHONPATH", "")
        sep = ";" if sys.platform == "win32" else ":"
        if existing:
            env["PYTHONPATH"] = f"{client_root}{sep}{existing}"
        else:
            env["PYTHONPATH"] = str(client_root)

    click.echo(f"Running: {' '.join(cmd)}")
    if not with_auth:
        click.echo("Test mode: authentication disabled")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return result.returncode


def run_client_tests(
    port: int = DEFAULT_PORT,
    timeout: int = CLIENT_TEST_TIMEOUT,
    headless: bool = True,
    start_server: bool = False,
) -> dict:
    """Run client-side Brython tests via headless browser.

    Args:
        port: Server port number.
        timeout: Maximum time to wait for tests.
        headless: Run browser in headless mode.
        start_server: Start server if not running.

    Returns:
        Test results dictionary.
    """
    manager = ServerManager(port=port)
    server_was_started = False

    # Check if server is running
    if not manager.is_server_running():
        if start_server:
            click.echo("Starting server...")
            success, message = manager.start(wait=True)
            if not success:
                return {"status": "error", "error": f"Failed to start server: {message}"}
            server_was_started = True
        else:
            return {
                "status": "error",
                "error": f"Server is not running on port {port}. Start it with 'mathud server start' or use --start-server.",
            }

    try:
        with BrowserAutomation(port=port, headless=headless) as browser:
            click.echo(f"Navigating to http://127.0.0.1:{port}/...")

            if not browser.navigate_to_app():
                return {"status": "error", "error": "Failed to navigate to application"}

            click.echo("Waiting for application to be ready...")
            if not browser.wait_for_app_ready():
                return {"status": "error", "error": "Application did not become ready in time"}

            click.echo("Starting tests...")
            start_result = browser.start_tests()
            if start_result.get("status") == "error":
                return start_result

            click.echo(f"Waiting for tests to complete (timeout: {timeout}s)...")
            results = browser.poll_test_results(timeout=timeout, poll_interval=CLIENT_TEST_POLL_INTERVAL)

            return results

    finally:
        # Stop server if we started it
        if server_was_started:
            click.echo("Stopping server...")
            manager.stop()


@click.group()
def test() -> None:
    """Run MatHud tests."""
    pass


@test.command("server")
@click.argument("test_path", required=False)
@click.option("-k", "--keyword", help="Only run tests matching keyword expression")
@click.option("--with-auth", is_flag=True, help="Enable authentication during tests")
@click.option("-q", "--quiet", is_flag=True, help="Decrease verbosity")
@click.option("--extra", "-e", multiple=True, help="Extra arguments to pass to pytest")
def server_cmd(
    test_path: Optional[str],
    keyword: Optional[str],
    with_auth: bool,
    quiet: bool,
    extra: tuple[str, ...],
) -> None:
    """Run server-side pytest tests.

    Optionally specify a TEST_PATH to run specific tests.
    """
    exit_code = run_server_tests(
        test_path=test_path,
        keyword=keyword,
        with_auth=with_auth,
        verbose=not quiet,
        extra_args=list(extra) if extra else None,
    )
    raise SystemExit(exit_code)


@test.command("client")
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
@click.option(
    "--timeout",
    "-t",
    default=CLIENT_TEST_TIMEOUT,
    type=int,
    help=f"Test timeout in seconds (default: {CLIENT_TEST_TIMEOUT})",
)
@click.option("--no-headless", is_flag=True, help="Show browser window")
@click.option("--start-server", is_flag=True, help="Start server if not running")
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON")
def client_cmd(
    port: int,
    timeout: int,
    no_headless: bool,
    start_server: bool,
    as_json: bool,
) -> None:
    """Run client-side Brython tests via headless Chrome."""
    results = run_client_tests(
        port=port,
        timeout=timeout,
        headless=not no_headless,
        start_server=start_server,
    )

    if as_json:
        click.echo(json.dumps(results, indent=2))
    else:
        status = results.get("status")
        if status == "error":
            click.echo(click.style(f"Error: {results.get('error', 'Unknown error')}", fg="red"), err=True)
            raise SystemExit(1)
        elif status == "timeout":
            click.echo(click.style(f"Timeout: {results.get('error', 'Tests timed out')}", fg="yellow"), err=True)
            raise SystemExit(1)
        else:
            # Display test results
            tests_run = results.get("tests_run", 0)
            failures = results.get("failures", 0)
            errors = results.get("errors", 0)

            click.echo()
            click.echo(click.style("=== Test Results ===", bold=True))
            click.echo(f"Tests run: {tests_run}")

            if failures == 0 and errors == 0:
                click.echo(click.style(f"Failures: {failures}", fg="green"))
                click.echo(click.style(f"Errors: {errors}", fg="green"))
                click.echo(click.style("\nAll tests passed!", fg="green", bold=True))
            else:
                click.echo(click.style(f"Failures: {failures}", fg="red" if failures else "green"))
                click.echo(click.style(f"Errors: {errors}", fg="red" if errors else "green"))

                if results.get("failing_tests"):
                    click.echo(click.style("\nFailing tests:", fg="red"))
                    for fail in results["failing_tests"]:
                        click.echo(f"  - {fail.get('test', 'Unknown')}: {fail.get('error', '')}")

                if results.get("error_tests"):
                    click.echo(click.style("\nError tests:", fg="red"))
                    for err in results["error_tests"]:
                        click.echo(f"  - {err.get('test', 'Unknown')}: {err.get('error', '')}")

                raise SystemExit(1)


@test.command("all")
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    type=int,
    help=f"Server port (default: {DEFAULT_PORT})",
)
@click.option("--with-auth", is_flag=True, help="Enable authentication for server tests")
@click.option("--start-server", is_flag=True, help="Start server for client tests if not running")
def all_cmd(port: int, with_auth: bool, start_server: bool) -> None:
    """Run both server and client test suites."""
    click.echo(click.style("=== Running Server Tests ===", bold=True))
    server_exit = run_server_tests(with_auth=with_auth)

    if server_exit != 0:
        click.echo(click.style("\nServer tests failed. Skipping client tests.", fg="red"))
        raise SystemExit(server_exit)

    click.echo()
    click.echo(click.style("=== Running Client Tests ===", bold=True))
    results = run_client_tests(port=port, start_server=start_server)

    status = results.get("status")
    if status == "error":
        click.echo(click.style(f"Error: {results.get('error', 'Unknown error')}", fg="red"), err=True)
        raise SystemExit(1)

    failures = results.get("failures", 0)
    errors = results.get("errors", 0)

    if failures > 0 or errors > 0:
        click.echo(click.style(f"\nClient tests: {failures} failures, {errors} errors", fg="red"))
        raise SystemExit(1)

    click.echo(click.style(f"\nAll tests passed! (Server + {results.get('tests_run', 0)} client tests)", fg="green", bold=True))
