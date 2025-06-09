#!/usr/bin/env python3
"""
MatHud Diagram Tools Setup Script

Installs all required dependencies for automated diagram generation.
Handles platform-specific installations and verifies tool availability.

Usage:
    python setup_diagram_tools.py
"""

import subprocess
import sys
import platform
import os
from pathlib import Path

class DiagramToolsSetup:
    def __init__(self):
        self.system = platform.system().lower()
        self.python_packages = [
            'pylint',           # for pyreverse
            'graphviz',         # Python graphviz bindings
            'diagrams',         # for architecture diagrams
            'pydeps',           # for dependency analysis
            'pycallgraph2',     # for call graph generation
        ]
        
    def run_command(self, cmd, description):
        """Run a command and handle errors gracefully."""
        print(f"🔧 {description}...")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            print(f"   ✓ Success")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Failed: {e.stderr.strip()}")
            return False
    
    def install_graphviz_system(self):
        """Install system-level Graphviz based on the operating system."""
        if self.system == 'windows':
            print("📋 For Windows, please manually install Graphviz:")
            print("   1. Download from: https://graphviz.org/download/")
            print("   2. Install and add to PATH")
            print("   3. Or use: winget install graphviz")
            return True
        elif self.system == 'darwin':  # macOS
            return self.run_command('brew install graphviz', 'Installing Graphviz via Homebrew')
        elif self.system == 'linux':
            # Try different package managers
            if subprocess.run(['which', 'apt'], capture_output=True).returncode == 0:
                return self.run_command('sudo apt update && sudo apt install -y graphviz', 'Installing Graphviz via apt')
            elif subprocess.run(['which', 'yum'], capture_output=True).returncode == 0:
                return self.run_command('sudo yum install -y graphviz', 'Installing Graphviz via yum')
            elif subprocess.run(['which', 'dnf'], capture_output=True).returncode == 0:
                return self.run_command('sudo dnf install -y graphviz', 'Installing Graphviz via dnf')
            elif subprocess.run(['which', 'pacman'], capture_output=True).returncode == 0:
                return self.run_command('sudo pacman -S graphviz', 'Installing Graphviz via pacman')
            else:
                print("⚠️  Please manually install Graphviz for your Linux distribution")
                return False
    
    def install_python_packages(self):
        """Install required Python packages."""
        print("🐍 Installing Python packages...")
        
        # Upgrade pip first
        self.run_command(f'{sys.executable} -m pip install --upgrade pip', 'Upgrading pip')
        
        # Install packages
        for package in self.python_packages:
            success = self.run_command(
                f'{sys.executable} -m pip install {package}', 
                f'Installing {package}'
            )
            if not success:
                print(f"⚠️  Failed to install {package} - you may need to install it manually")
    
    def verify_installation(self):
        """Verify that all tools are properly installed."""
        print("\n🔍 Verifying installation...")
        
        tools_to_check = [
            ('pyreverse', 'pyreverse --help'),
            ('dot', 'dot -V'),
            ('python', f'{sys.executable} --version'),
        ]
        
        success_count = 0
        for tool, cmd in tools_to_check:
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
                print(f"   ✓ {tool} is available")
                success_count += 1
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"   ✗ {tool} is not available")
        
        # Check Python packages
        python_success = 0
        for package in self.python_packages:
            try:
                subprocess.run([sys.executable, '-c', f'import {package}'], 
                             check=True, capture_output=True)
                print(f"   ✓ Python package {package} is available")
                python_success += 1
            except subprocess.CalledProcessError:
                print(f"   ✗ Python package {package} is not available")
        
        total_expected = len(tools_to_check) + len(self.python_packages)
        total_success = success_count + python_success
        
        print(f"\n📊 Installation Summary: {total_success}/{total_expected} tools available")
        
        if total_success == total_expected:
            print("🎉 All tools installed successfully!")
            return True
        else:
            print("⚠️  Some tools are missing. Please check the installation instructions above.")
            return False
    
    def setup(self):
        """Run the complete setup process."""
        print("🚀 MatHud Diagram Tools Setup")
        print("=" * 50)
        
        print(f"🖥️  Detected OS: {platform.system()} {platform.release()}")
        print(f"🐍 Python version: {sys.version}")
        print()
        
        # Install system dependencies
        print("Step 1: Installing system dependencies")
        self.install_graphviz_system()
        print()
        
        # Install Python packages
        print("Step 2: Installing Python packages")
        self.install_python_packages()
        print()
        
        # Verify installation
        print("Step 3: Verifying installation")
        success = self.verify_installation()
        print()
        
        if success:
            print("🎯 Setup completed successfully!")
            print("\nNext steps:")
            print("1. Run: python generate_diagrams.py")
            print("2. Check the 'diagrams/' directory for generated files")
            print("3. Customize generate_diagrams.py as needed")
        else:
            print("❌ Setup completed with some issues.")
            print("Please resolve the missing dependencies manually.")
        
        print("\n" + "=" * 50)


def main():
    setup = DiagramToolsSetup()
    setup.setup()


if __name__ == '__main__':
    main() 