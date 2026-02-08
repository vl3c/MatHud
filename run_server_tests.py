#!/usr/bin/env python
"""
MatHud Server-Side Test Runner

Simple script to run server tests with pytest integration.
Provides command-line interface for running tests with various options.
Automatically disables authentication for testing environment.

Dependencies:
    - pytest: Testing framework
    - subprocess: Process execution
    - server_tests/: Test modules directory

Usage:
    python run_server_tests.py            - Run all server tests
    python run_server_tests.py test_file  - Run tests in a specific file
    python run_server_tests.py -k keyword - Run tests matching keyword
    python run_server_tests.py --help     - Show pytest help
    python run_server_tests.py --with-auth - Run tests with authentication enabled
"""

from __future__ import annotations

import os
import subprocess
import sys
import platform
from pathlib import Path


def run_tests() -> int:
    """Run the server tests with pytest.

    Processes command line arguments and executes pytest with appropriate
    test paths and options. Handles OS-specific Python interpreter paths.
    Automatically disables authentication for testing unless --with-auth is specified.

    Returns:
        int: pytest exit code (0 for success, non-zero for failure)
    """
    # Default test path and arguments
    test_path: str = "server_tests"
    extra_args: list[str] = []
    with_auth: bool = False

    # Process command line arguments
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]

        # Handle --help flag
        if arg == "--help" or arg == "-h":
            show_help()
            return 0

        # Handle --with-auth flag
        elif arg == "--with-auth":
            with_auth = True
            i += 1
            continue

        # Handle -k for keyword filtering
        elif arg == "-k" and i + 1 < len(sys.argv):
            extra_args.extend(["-k", sys.argv[i+1]])
            i += 2
            continue

        # If it's not a recognized flag, treat it as a test path
        elif not arg.startswith("-"):
            # If it's a path without extension, assume it's in server_tests
            if not arg.endswith(".py") and not os.path.exists(arg):
                possible_path = os.path.join("server_tests", arg)
                if os.path.exists(possible_path):
                    test_path = possible_path
                elif os.path.exists(possible_path + ".py"):
                    test_path = possible_path + ".py"
                else:
                    test_path = arg
            else:
                test_path = arg

        # Pass through any other arguments to pytest
        else:
            extra_args.append(arg)

        i += 1

    # Set test environment - disable authentication by default for testing
    if not with_auth:
        os.environ['REQUIRE_AUTH'] = 'false'
        print("Test mode: authentication disabled for testing")
    else:
        print("Test mode: authentication enabled (--with-auth)")

    # Determine the Python interpreter based on the OS
    is_windows: bool = platform.system() == "Windows"

    if is_windows:
        python_cmd: str = "venv\\Scripts\\python"
    else:
        python_cmd = "./venv/bin/python"

    # Build the command
    cmd: list[str] = [python_cmd, "-m", "pytest", test_path, "-v"] + extra_args

    env = os.environ.copy()
    client_root = Path(__file__).resolve().parent / "static" / "client"
    if client_root.exists():
        existing_pythonpath = env.get("PYTHONPATH", "")
        path_sep = ";" if is_windows else ":"
        client_root_str = str(client_root)
        if existing_pythonpath:
            if client_root_str not in existing_pythonpath.split(path_sep):
                env["PYTHONPATH"] = f"{client_root_str}{path_sep}{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = client_root_str

    print(f"Running tests: {' '.join(cmd)}")

    # Run the command
    result = subprocess.run(cmd, env=env)

    # Return the exit code
    return result.returncode

def show_help() -> None:
    """Show help information about this script and pytest options.

    Displays usage information for both this wrapper script and
    common pytest command-line options.
    """
    print(__doc__)
    print("Test runner specific options:")
    print("  --with-auth         Enable authentication during testing (default: disabled)")
    print()
    print("Common pytest options:")
    print("  -v, --verbose       Increase verbosity")
    print("  -k EXPRESSION       Only run tests which match the given substring expression")
    print("  -x, --exitfirst     Exit instantly on first error or failed test")
    print("  --no-header         Disable header")
    print("  --no-summary        Disable summary")
    print("  -q, --quiet         Decrease verbosity")
    print("\nFor more options: python run_server_tests.py -- --help")

if __name__ == "__main__":
    sys.exit(run_tests())
