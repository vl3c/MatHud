#!/usr/bin/env python3
"""
MatHud Diagram Generation Script

Automatically generates various diagrams from the Python codebase:
- Class diagrams
- Package dependency diagrams
- Module interaction diagrams
- Architecture overview diagrams

Dependencies:
    pip install pylint graphviz diagrams
    
Usage:
    python generate_diagrams.py [--png-dir ../generated_png] [--svg-dir ../generated_svg] [--format png,svg]
"""

import os
import subprocess
import argparse
import sys
from pathlib import Path

class DiagramGenerator:
    def __init__(self, png_dir="../generated_png", svg_dir="../generated_svg", formats=["png", "svg"]):
        # Convert to absolute paths to avoid issues with cwd changes
        self.png_dir = Path(png_dir).resolve()
        self.svg_dir = Path(svg_dir).resolve()
        self.formats = formats
        
        # Create output directories
        if "png" in formats:
            self.png_dir.mkdir(exist_ok=True)
        if "svg" in formats:
            self.svg_dir.mkdir(exist_ok=True)
    
    def get_output_dir(self, fmt):
        """Get the appropriate output directory for the given format."""
        if fmt == "png":
            return self.png_dir
        elif fmt == "svg":
            return self.svg_dir
        else:
            # Fallback for other formats like DOT
            return self.png_dir
        
    def check_dependencies(self):
        """Check if required tools are installed."""
        tools = {
            'pyreverse': 'pylint',
            'dot': 'graphviz',
            'python': 'python'
        }
        
        missing = []
        for tool, package in tools.items():
            # Use 'where' on Windows, 'which' on Unix-like systems
            cmd = 'where' if sys.platform == 'win32' else 'which'
            try:
                result = subprocess.run([cmd, tool], capture_output=True, text=True)
                if result.returncode != 0:
                    missing.append(f"{tool} (install: pip install {package})")
            except FileNotFoundError:
                missing.append(f"{tool} (install: pip install {package})")
        
        if missing:
            print("Missing dependencies:")
            for dep in missing:
                print(f"  - {dep}")
            print("\nNote: You may need to restart your shell or add tools to PATH")
            sys.exit(1)
    
    def generate_class_diagrams(self):
        """Generate UML class diagrams for the entire project."""
        print("Generating class diagrams...")
        
        for fmt in self.formats:
            output_dir = self.get_output_dir(fmt)
            cmd = [
                'pyreverse',
                '-o', fmt,
                '-p', 'MatHud',
                '--output-directory', str(output_dir),
                'static/',
                'app.py'
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, cwd='../..')
                print(f"  ✓ Class diagram generated: {output_dir}/classes_MatHud.{fmt}")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Error generating class diagram: {e.stderr}")
    
    def generate_package_diagrams(self):
        """Generate package dependency diagrams."""
        print("Generating package diagrams...")
        
        for fmt in self.formats:
            output_dir = self.get_output_dir(fmt)
            cmd = [
                'pyreverse',
                '-o', fmt,
                '-p', 'MatHud_packages',
                '--output-directory', str(output_dir),
                '--show-associated', '1',
                '--show-ancestors', '1',
                'static/'
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, cwd='../..')
                print(f"  ✓ Package diagram generated: {output_dir}/packages_MatHud_packages.{fmt}")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Error generating package diagram: {e.stderr}")
    
    def generate_module_specific_diagrams(self):
        """Generate diagrams for specific important modules."""
        print("Generating module-specific diagrams...")
        
        important_modules = [
            ('static/app_manager.py', 'AppManager'),
            ('static/openai_api.py', 'OpenAI_API'),
            ('static/webdriver_manager.py', 'WebDriver'),
            ('static/workspace_manager.py', 'Workspace'),
            ('static/routes.py', 'Routes')
        ]
        
        for module_path, name in important_modules:
            # Use absolute path for checking existence
            abs_module_path = Path('../..') / module_path
            if abs_module_path.exists():
                for fmt in self.formats:
                    output_dir = self.get_output_dir(fmt)
                    cmd = [
                        'pyreverse',
                        '-o', fmt,
                        '-p', name,
                        '--output-directory', str(output_dir),
                        module_path  # Use relative path for pyreverse
                    ]
                    
                    try:
                        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd='../..')
                        print(f"  ✓ {name} diagram generated: {output_dir}/classes_{name}.{fmt}")
                    except subprocess.CalledProcessError as e:
                        print(f"  ✗ Error generating {name} diagram: {e.stderr}")
    
    def generate_architecture_diagram(self):
        """Generate a high-level architecture diagram using Python diagrams library."""
        print("Generating architecture diagram...")
        print("  ⚠ Architecture diagram generation skipped")
        print("    Note: You can manually create architecture diagrams using the 'diagrams' library")
        print("    See WORKFLOW_SUMMARY.md for examples")
    
    def generate_dependency_graph(self):
        """Generate a dependency graph using pydeps."""
        print("Generating dependency graph...")
        
        # Dependencies are typically SVG, so use SVG directory if available
        output_dir = self.svg_dir if "svg" in self.formats else self.png_dir
        
        try:
            cmd = [
                'pydeps',
                '--show-deps',
                '--max-bacon', '3',
                '--cluster',
                '--rankdir', 'TB',
                '--no-show',  # Prevent automatic opening of the generated file
                '-o', str(output_dir / 'dependencies.svg'),
                'static/'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd='../..')
            print(f"  ✓ Dependency graph generated: {output_dir}/dependencies.svg")
            
        except subprocess.CalledProcessError:
            print("  ✗ Error generating dependency graph")
            print("    Note: Install with 'pip install pydeps' for dependency graphs")
        except FileNotFoundError:
            print("  ⚠ pydeps not found - skipping dependency graph")
    
    def generate_call_graph(self):
        """Generate call graph using pycallgraph2."""
        print("Generating call graph...")
        
        try:
            import pycallgraph2
            from pycallgraph2 import PyCallGraph
            from pycallgraph2.output import GraphvizOutput
            
            # Use PNG directory for call graph output
            output_dir = self.png_dir
            output_file = output_dir / 'call_graph.png'
            
            print(f"  ⚠ Call graph generation is experimental")
            print(f"    This will trace app.py execution and may take time...")
            print(f"    Output: {output_file}")
            
            # Note: This would require running the actual application
            # For now, just show the command to run manually
            print("  💡 To generate call graph manually:")
            print("     cd to project root, then run:")
            print("     pycallgraph graphviz --output-file=diagrams/generated_png/call_graph.png -- python app.py")
            
        except ImportError:
            print("  ✗ pycallgraph2 not found")
            print("    Install with: pip install pycallgraph2")
        except Exception as e:
            print(f"  ✗ Error setting up call graph: {e}")
    
    def run(self):
        """Run all diagram generation tasks."""
        print(f"MatHud Diagram Generator")
        print(f"PNG output: {self.png_dir}")
        print(f"SVG output: {self.svg_dir}")
        print(f"Formats: {', '.join(self.formats)}")
        print("-" * 50)
        
        self.check_dependencies()
        
        self.generate_class_diagrams()
        self.generate_package_diagrams()
        self.generate_module_specific_diagrams()
        self.generate_architecture_diagram()
        self.generate_dependency_graph()
        self.generate_call_graph()
        
        print("-" * 50)
        print(f"✓ Diagram generation complete!")
        if "png" in self.formats:
            print(f"   PNG files: {self.png_dir}/")
        if "svg" in self.formats:
            print(f"   SVG files: {self.svg_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Generate diagrams for MatHud project')
    parser.add_argument('--png-dir', default='../generated_png', 
                       help='Output directory for PNG diagrams (default: ../generated_png)')
    parser.add_argument('--svg-dir', default='../generated_svg', 
                       help='Output directory for SVG diagrams (default: ../generated_svg)')
    parser.add_argument('--format', default='png,svg', 
                       help='Output formats: png,svg,dot (default: png,svg)')
    
    args = parser.parse_args()
    formats = [f.strip() for f in args.format.split(',')]
    
    generator = DiagramGenerator(args.png_dir, args.svg_dir, formats)
    generator.run()


if __name__ == '__main__':
    main() 