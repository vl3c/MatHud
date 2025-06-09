#!/usr/bin/env python3
"""
MatHud Diagram Generation Launcher

Convenience script to run diagram generation from the root directory.
This script calls the actual generator in diagrams/scripts/

Usage:
    python generate_diagrams_launcher.py [options]
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Path to the actual generator script
    script_path = Path("diagrams/scripts/generate_diagrams.py")
    
    if not script_path.exists():
        print("Error: Diagram generation script not found!")
        print(f"   Expected: {script_path}")
        print("   Make sure you're running this from the MatHud root directory.")
        sys.exit(1)
    
    print("MatHud Diagram Generation Launcher")
    print(f"   Calling: {script_path}")
    print("-" * 50)
    
    # Change to the scripts directory and run the generator
    original_cwd = os.getcwd()
    try:
        os.chdir("diagrams/scripts")
        
        # Forward all command line arguments to the actual script
        cmd = [sys.executable, "generate_diagrams.py"] + sys.argv[1:]
        result = subprocess.run(cmd)
        
        sys.exit(result.returncode)
        
    except Exception as e:
        print(f"Error running diagram generator: {e}")
        sys.exit(1)
    
    finally:
        os.chdir(original_cwd)


if __name__ == '__main__':
    main() 