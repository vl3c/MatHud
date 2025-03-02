#!/usr/bin/env python
"""
Simple script to run the server tests.

Usage:
    python run_server_tests.py            - Run all server tests
    python run_server_tests.py test_file  - Run tests in a specific file
    python run_server_tests.py -k keyword - Run tests matching keyword
    python run_server_tests.py --help     - Show pytest help
"""

import os
import sys
import subprocess
import platform

def run_tests():
    """Run the server tests with pytest."""
    # Default test path and arguments
    test_path = "ServerTests"
    extra_args = []
    
    # Process command line arguments
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        # Handle --help flag
        if arg == "--help" or arg == "-h":
            show_help()
            return 0
            
        # Handle -k for keyword filtering
        elif arg == "-k" and i + 1 < len(sys.argv):
            extra_args.extend(["-k", sys.argv[i+1]])
            i += 2
            continue
            
        # If it's not a recognized flag, treat it as a test path
        elif not arg.startswith("-"):
            # If it's a path without extension, assume it's in ServerTests
            if not arg.endswith(".py") and not os.path.exists(arg):
                possible_path = os.path.join("ServerTests", arg)
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
    
    # Determine the Python interpreter based on the OS
    is_windows = platform.system() == "Windows"
    
    if is_windows:
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "./venv/bin/python"
    
    # Build the command
    cmd = [python_cmd, "-m", "pytest", test_path, "-v"] + extra_args
    
    print(f"Running tests: {' '.join(cmd)}")
    
    # Run the command
    result = subprocess.run(cmd)
    
    # Return the exit code
    return result.returncode

def show_help():
    """Show help information about this script and pytest options."""
    print(__doc__)
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