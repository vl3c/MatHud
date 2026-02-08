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

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from utils import (
    DIAGRAM_FONT,
    DIAGRAM_FONT_SIZE_STR,
    post_process_svg_fonts,
    setup_font_environment,
    setup_graphviz_path,
)


class DiagramGenerator:
    """Encapsulates all diagram generation routines for the project."""

    def __init__(
        self,
        png_dir: str = "../generated_png",
        svg_dir: str = "../generated_svg",
        formats: Sequence[str] | None = None,
    ) -> None:
        self.base_dir: Path = Path(__file__).parent.absolute()
        self.png_dir: Path = self.base_dir / png_dir
        self.svg_dir: Path = self.base_dir / svg_dir
        self.formats: List[str] = list(formats) if formats is not None else ["png", "svg"]
        self.font_updates_count: int = 0

        # Create output directories if they don't exist
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.svg_dir.mkdir(parents=True, exist_ok=True)

        # Setup Graphviz PATH for Windows
        setup_graphviz_path()

        # Setup font configuration for all diagrams
        setup_font_environment()



    def get_output_dir(self, fmt: str) -> Path:
        """Get the appropriate output directory for a format."""
        if fmt == 'png':
            return self.png_dir
        elif fmt == 'svg':
            return self.svg_dir
        else:
            # For other formats like dot, use svg directory
            return self.svg_dir

    def get_server_output_dir(self, fmt: str) -> Path:
        """Get the server-specific output directory for a format."""
        base_dir = self.get_output_dir(fmt)
        server_dir = base_dir / 'server'
        server_dir.mkdir(parents=True, exist_ok=True)
        return server_dir

    def _update_fonts_and_count(self, svg_file: Path) -> None:
        """Update SVG fonts and track count."""
        if post_process_svg_fonts(svg_file):
            self.font_updates_count += 1

    def check_dependencies(self) -> None:
        """Check if required tools are installed."""
        tools: Dict[str, str] = {
            'pyreverse': 'pylint',
            'dot': 'graphviz',
            'python': 'python'
        }

        missing: List[str] = []
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

    def generate_class_diagrams(self) -> None:
        """Generate UML class diagrams for the entire project."""
        print("Generating class diagrams...")

        # List of all Python files with classes
        class_files: List[str] = [
            'static/app_manager.py',
            'static/openai_api.py',
            'static/webdriver_manager.py',
            'static/workspace_manager.py',
            'static/ai_model.py',
            'static/log_manager.py',
            'static/tool_call_processor.py'
        ]

        for fmt in self.formats:
            output_dir = self.get_server_output_dir(fmt)
            cmd = [
                'pyreverse',
                '-o', fmt,
                '-p', 'MatHud_AllClasses',
                '--output-directory', str(output_dir),
                '--show-associated', '1',
                '--show-ancestors', '1'
            ] + class_files  # Add all class files explicitly

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, cwd='../..')
                print(f"  + Main class diagram generated: {output_dir}/classes_MatHud_AllClasses.{fmt}")

                # Post-process SVG files to use configured font
                if fmt == 'svg':
                    self._update_fonts_and_count(output_dir / f'classes_MatHud_AllClasses.{fmt}')

            except subprocess.CalledProcessError as e:
                print(f"  Error: Error generating main class diagram: {e.stderr}")

    def generate_package_diagrams(self) -> None:
        """Generate package dependency diagrams."""
        print("Generating package diagrams...")

        for fmt in self.formats:
            output_dir = self.get_server_output_dir(fmt)
            cmd = [
                'pyreverse',
                '-o', fmt,
                '-p', 'MatHud_packages',
                '--output-directory', str(output_dir),
                '--show-associated', '1',
                '--show-ancestors', '1',
                '-m', 'yes',  # Show module names
                'static/',  # Main application code
                'app.py',   # Entry point
                'run_server_tests.py'  # Test code
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, cwd='../..')
                print(f"  + Package diagram generated: {output_dir}/packages_MatHud_packages.{fmt}")

                # Post-process SVG files to use configured font
                if fmt == 'svg':
                    self._update_fonts_and_count(output_dir / f'packages_MatHud_packages.{fmt}')

            except subprocess.CalledProcessError as e:
                print(f"  Error: Error generating package diagram: {e.stderr}")

    def generate_module_specific_diagrams(self) -> None:
        """Generate diagrams for specific important modules."""
        print("Generating module-specific diagrams...")

        important_modules: List[Tuple[str, str]] = [
            ('static/app_manager.py', 'AppManager'),
            ('static/openai_api.py', 'OpenAI_API'),
            ('static/webdriver_manager.py', 'WebDriver'),
            ('static/workspace_manager.py', 'Workspace')
            # Note: routes.py is handled separately in generate_flask_routes_diagram()
        ]

        for module_path, name in important_modules:
            # Use absolute path for checking existence
            abs_module_path = Path('../..') / module_path
            if abs_module_path.exists():
                for fmt in self.formats:
                    output_dir = self.get_server_output_dir(fmt)
                    cmd = [
                        'pyreverse',
                        '-o', fmt,
                        '-p', name,
                        '--output-directory', str(output_dir),
                        module_path  # Use relative path for pyreverse
                    ]

                    try:
                        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd='../..')
                        print(f"  + {name} diagram generated: {output_dir}/classes_{name}.{fmt}")

                        # Post-process SVG files to use configured font
                        if fmt == 'svg':
                            self._update_fonts_and_count(output_dir / f'classes_{name}.{fmt}')

                    except subprocess.CalledProcessError as e:
                        print(f"  Error: Error generating {name} diagram: {e.stderr}")

    def generate_flask_routes_diagram(self) -> None:
        """Generate comprehensive Flask routes diagrams and function analysis."""
        print("Generating Flask routes visualization...")

        # Generate custom Flask routes diagram
        self._generate_custom_routes_diagram()

        # Try pyreverse for any class-like structures
        self._generate_pyreverse_routes_diagram()

        # Generate function call diagram for routes.py
        self._generate_function_call_diagram()

    def _generate_custom_routes_diagram(self) -> None:
        """Create a custom SVG diagram showing Flask routes and their relationships."""
        try:
            import re
            from pathlib import Path

            # Read routes.py content
            routes_file = Path('../../static/routes.py')
            if not routes_file.exists():
                print("  Warning: routes.py not found for custom analysis")
                return

            content = routes_file.read_text()

            # Extract route information using regex
            route_pattern = r"@app\.route\(['\"]([^'\"]+)['\"](?:,\s*methods=\[([^\]]+)\])?\)\s*\n\s*def\s+(\w+)"
            routes = re.findall(route_pattern, content)

            # Extract function definitions
            function_pattern = r"def\s+(\w+)\([^)]*\):"
            functions = re.findall(function_pattern, content)

            # Create custom SVG for routes
            svg_content = self._create_routes_svg(routes, functions)

            # Save SVG version
            if "svg" in self.formats:
                svg_output_dir = self.get_server_output_dir('svg')
                svg_output_file = svg_output_dir / 'flask_routes_custom.svg'
                svg_output_file.write_text(svg_content)
                print(f"  + Custom Flask routes diagram (SVG): {svg_output_file}")

                # Convert to PNG if needed
                if "png" in self.formats:
                    png_output_dir = self.get_server_output_dir('png')
                    png_output_file = png_output_dir / 'flask_routes_custom.png'
                    self._convert_svg_to_png(svg_output_file, png_output_file)

            # If only PNG format requested, create SVG first then convert
            elif "png" in self.formats:
                # Create temporary SVG
                temp_svg = Path('temp_flask_routes.svg')
                temp_svg.write_text(svg_content)

                png_output_dir = self.get_server_output_dir('png')
                png_output_file = png_output_dir / 'flask_routes_custom.png'
                self._convert_svg_to_png(temp_svg, png_output_file)

                # Clean up temporary file
                temp_svg.unlink()
                print(f"  + Custom Flask routes diagram (PNG): {png_output_file}")

        except Exception as e:
            print(f"  Warning: Custom routes diagram generation failed: {e}")

    def _convert_svg_to_png(self, svg_file: Path, png_file: Path) -> None:
        """Convert SVG file to PNG using cairosvg."""
        try:
            import cairosvg
            cairosvg.svg2png(url=str(svg_file), write_to=str(png_file))
            print(f"  + Converted to PNG: {png_file}")
        except ImportError:
            print("  Warning: cairosvg not available - trying alternative conversion methods...")
            self._convert_svg_to_png_fallback(svg_file, png_file)
        except Exception as e:
            print(f"  Warning: SVG to PNG conversion failed: {e}")
            self._convert_svg_to_png_fallback(svg_file, png_file)

    def _convert_svg_to_png_fallback(self, svg_file: Path, png_file: Path) -> None:
        """Fallback SVG to PNG conversion using dot command."""
        try:
            # Use dot (Graphviz) to convert SVG to PNG
            subprocess.run([
                'dot', '-Tpng', str(svg_file), '-o', str(png_file)
            ], capture_output=True, text=True, check=True)
            print(f"  + Converted to PNG (via dot): {png_file}")
        except subprocess.CalledProcessError as e:
            print(f"  Warning: Fallback conversion failed: {e}")
            print("    Consider installing cairosvg: pip install cairosvg")
        except FileNotFoundError:
            print("  Warning: No conversion tool available (dot command not found)")
            print("    Consider installing cairosvg: pip install cairosvg")

    def _create_routes_svg(
        self,
        routes: Sequence[Tuple[str, str, str]],
        functions: Sequence[str],
    ) -> str:
        """Create SVG content showing Flask routes."""
        # SVG header
        svg_width = max(600, len(routes) * 150)
        svg_height = max(400, len(routes) * 80 + 100)

        svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{svg_width}pt" height="{svg_height}pt" viewBox="0.00 0.00 {svg_width}.00 {svg_height}.00"
     xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(4 {svg_height-4})">
<title>Flask Routes</title>
<polygon fill="white" stroke="none" points="-4,4 -4,-{svg_height-4} {svg_width-4},-{svg_height-4} {svg_width-4},4 -4,4"/>
'''

        # Add title
        svg += f'''
<!-- Title -->
<text x="{svg_width/2}" y="-{svg_height-30}" text-anchor="middle" font-family="{DIAGRAM_FONT},sans-serif" font-size="16" font-weight="bold">MatHud Flask API Routes</text>
'''

        # Add routes
        y_offset = svg_height - 80
        for i, (path, methods, func_name) in enumerate(routes):
            methods_clean = methods.replace("'", "").replace('"', '') if methods else "GET"

            # Route box
            box_y = y_offset - (i * 70)
            svg += f'''
<!-- Route {i+1}: {path} -->
<g id="route_{i+1}" class="node">
<rect x="50" y="-{box_y+40}" width="300" height="60" fill="#e8f4fd" stroke="#1f77b4" stroke-width="2" rx="5"/>
<text x="60" y="-{box_y+15}" font-family="{DIAGRAM_FONT},sans-serif" font-size="12" font-weight="bold" fill="#1f77b4">{methods_clean}</text>
<text x="60" y="-{box_y}" font-family="{DIAGRAM_FONT},sans-serif" font-size="14" font-weight="bold">{path}</text>
<text x="60" y="-{box_y-15}" font-family="{DIAGRAM_FONT},sans-serif" font-size="11" fill="#666">{func_name}()</text>
</g>
'''

        # Add function summary
        func_y = 50
        svg += f'''
<!-- Functions Summary -->
<g id="functions" class="node">
<rect x="400" y="-{func_y+len(functions)*15+20}" width="180" height="{len(functions)*20+30}" fill="#f0f0f0" stroke="#666" stroke-width="1" rx="5"/>
<text x="410" y="-{func_y+len(functions)*15+5}" font-family="{DIAGRAM_FONT},sans-serif" font-size="12" font-weight="bold">All Functions:</text>
'''

        for i, func in enumerate(functions):
            svg += f'<text x="415" y="-{func_y+len(functions)*15-10-i*15}" font-family="{DIAGRAM_FONT},sans-serif" font-size="{DIAGRAM_FONT_SIZE_STR}" fill="#333">{func}()</text>\n'

        svg += '</g>\n'

        # Close SVG
        svg += '</g>\n</svg>'

        return svg

    def _generate_pyreverse_routes_diagram(self) -> None:
        """Try pyreverse analysis for any class-like structures in routes."""
        try:
            for fmt in self.formats:
                output_dir = self.get_server_output_dir(fmt)
                cmd = [
                    'pyreverse',
                    '-o', fmt,
                    '-p', 'FlaskRoutes',
                    '--output-directory', str(output_dir),
                    '--show-associated', '1',
                    'static/routes.py'
                ]

                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd='../..')
                    print(f"  + Pyreverse routes diagram: {output_dir}/classes_FlaskRoutes.{fmt}")

                    # Post-process SVG files to use configured font
                    if fmt == 'svg':
                        self._update_fonts_and_count(output_dir / f'classes_FlaskRoutes.{fmt}')

                except subprocess.CalledProcessError:
                    print("  Warning: Pyreverse routes minimal (functions only, no classes)")

        except Exception as e:
            print(f"  Warning: Pyreverse routes analysis failed: {e}")

    def _generate_function_call_diagram(self) -> None:
        """Generate function call relationships for routes.py."""
        try:
            # Use pydeps to analyze function calls in routes.py specifically
            output_dir = self.get_server_output_dir('svg') if "svg" in self.formats else self.get_server_output_dir('png')

            cmd = [
                'pydeps',
                '--show-deps',
                '--max-bacon', '2',  # Limited depth for function calls
                '--no-show',
                '-o', str(output_dir / 'routes_functions.svg'),
                'static/routes.py'
            ]

            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd='../..')
            print(f"  + Routes function calls: {output_dir}/routes_functions.svg")

            # Post-process SVG file to use configured font
            self._update_fonts_and_count(output_dir / 'routes_functions.svg')

        except subprocess.CalledProcessError:
            print("  Warning: Routes function call analysis failed")
        except FileNotFoundError:
            print("  Warning: pydeps not available for function analysis")

    def generate_function_analysis(self) -> None:
        """Generate enhanced function analysis for files with minimal class content."""
        print("Generating function analysis for enhanced visualization...")

        # Files that may benefit from function analysis
        files_to_analyze: List[Tuple[str, str]] = [
            ('static/functions_definitions.py', 'FunctionDefinitions'),
            ('app.py', 'AppMain'),
            ('run_server_tests.py', 'server_tests')
        ]

        for file_path, name in files_to_analyze:
            # Check if file exists
            abs_file_path = Path('../..') / file_path
            if abs_file_path.exists():
                self._generate_file_function_analysis(file_path, name)

    def _generate_file_function_analysis(self, file_path: str, name: str) -> None:
        """Generate function analysis for a specific file."""
        try:
            # Use pydeps for enhanced function call visualization
            output_dir = self.get_server_output_dir('svg') if "svg" in self.formats else self.get_server_output_dir('png')

            cmd = [
                'pydeps',
                '--show-deps',
                '--max-bacon', '2',
                '--cluster',
                '--rankdir', 'TB',
                '--no-show',
                '-o', str(output_dir / f'functions_{name.lower()}.svg'),
                file_path
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, cwd='../..')
                print(f"  + Function analysis for {name}: {output_dir}/functions_{name.lower()}.svg")

                # Post-process SVG file to use configured font
                self._update_fonts_and_count(output_dir / f'functions_{name.lower()}.svg')

            except subprocess.CalledProcessError:
                print(f"  Warning: Function analysis for {name} failed - file may have no dependencies")

        except Exception as e:
            print(f"  Warning: Function analysis error for {name}: {e}")

    def generate_architecture_diagram(self) -> None:
        """Generate comprehensive architecture diagrams using Python diagrams library."""
        print("Generating architecture diagrams (integrated mode)...")
        try:
            from generate_arch import ArchitectureDiagramGenerator

            # Create architecture diagram generator with same settings
            arch_generator = ArchitectureDiagramGenerator(
                png_dir=str(self.png_dir),
                svg_dir=str(self.svg_dir),
                formats=self.formats
            )

            # Generate all architecture diagrams (no cleaning in integrated mode)
            print("   Architecture diagrams will be saved to: architecture/ subfolder")
            arch_generator.generate_all_architecture_diagrams(clean_first=False)

        except ImportError as e:
            print(f"  Error: Missing architecture diagram generator: {e}")
            print("     Note: Make sure generate_arch.py is in the same directory")
        except Exception as e:
            print(f"  Error: Error generating architecture diagrams: {e}")
            print("     Note: Check that Graphviz is properly installed and accessible")

    def generate_dependency_graph(self) -> None:
        """Generate a dependency graph using pydeps."""
        print("Generating dependency graph...")

        # Dependencies are typically SVG, so use SVG directory if available
        output_dir = self.get_server_output_dir('svg') if "svg" in self.formats else self.get_server_output_dir('png')

        try:
            # Generate main project dependency graph
            cmd = [
                'pydeps',
                '--show-deps',
                '--max-bacon', '4',  # Increased depth
                '--cluster',
                '--rankdir', 'TB',
                '--no-show',  # Prevent automatic opening of the generated file
                '--include-missing',  # Show external dependencies
                '-o', str(output_dir / 'dependencies_main.svg'),
                'app.py'  # Start from main entry point
            ]

            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd='../..')
            print(f"  + Main dependency graph generated: {output_dir}/dependencies_main.svg")

            # Post-process SVG file to use configured font
            self._update_fonts_and_count(output_dir / 'dependencies_main.svg')

            # Generate static module dependencies
            cmd = [
                'pydeps',
                '--show-deps',
                '--max-bacon', '3',
                '--cluster',
                '--rankdir', 'LR',  # Left-to-right for better readability
                '--no-show',
                '-o', str(output_dir / 'dependencies_static.svg'),
                'static/'
            ]

            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd='../..')
            print(f"  + Static module dependencies generated: {output_dir}/dependencies_static.svg")

            # Post-process SVG file to use configured font
            self._update_fonts_and_count(output_dir / 'dependencies_static.svg')

        except subprocess.CalledProcessError as e:
            print("  Error: Error generating dependency graph")
            print(f"    Error: {e.stderr if e.stderr else 'Unknown error'}")
            print("    Note: Install with 'pip install pydeps' for dependency graphs")
        except FileNotFoundError:
            print("  Warning: pydeps not found - skipping dependency graph")

    def generate_call_graph(self) -> None:
        """Generate call graph using pycallgraph2."""
        print("Generating call graph...")

        try:
            import pycallgraph2
            from pycallgraph2 import PyCallGraph
            from pycallgraph2.output import GraphvizOutput

            # Use PNG directory for call graph output
            output_dir = self.get_server_output_dir('png')
            output_file = output_dir / 'call_graph.png'

            print("  Warning: Call graph generation is experimental")
            print("    This will trace app.py execution and may take time...")
            print(f"    Output: {output_file}")

            # Note: This would require running the actual application
            # For now, just show the command to run manually
            print("  Note: To generate call graph manually:")
            print("     cd to project root, then run:")
            print("     pycallgraph graphviz --output-file=diagrams/generated_png/server/call_graph.png -- python app.py")

        except ImportError:
            print("  Error: pycallgraph2 not found")
            print("    Install with: pip install pycallgraph2")
        except Exception as e:
            print(f"  Error: Error setting up call graph: {e}")

    def generate_brython_diagrams(self) -> None:
        """Generate comprehensive Brython client-side diagrams."""
        print("Generating Brython client-side diagrams (integrated mode)...")
        try:
            from generate_brython_diagrams import BrythonDiagramGenerator

            # Create Brython diagram generator with same settings
            brython_generator = BrythonDiagramGenerator(
                png_dir=str(self.png_dir),
                svg_dir=str(self.svg_dir),
                formats=self.formats
            )

            # Generate all Brython diagrams
            print("   Client diagrams will be saved to: client/ subfolder")
            brython_generator.generate_all_brython_diagrams()

        except ImportError as e:
            print(f"  Error: Missing Brython diagram generator: {e}")
            print("     Note: Make sure generate_brython_diagrams.py is in the same directory")
        except Exception as e:
            print(f"  Error: Error generating Brython diagrams: {e}")
            print("     Note: Check that Brython source code is available and Graphviz is properly installed")

    def run(self, include_brython: bool = False) -> None:
        """Run all diagram generation tasks."""
        print("MatHud Diagram Generator")
        print(f"PNG output: {self.png_dir}")
        print(f"SVG output: {self.svg_dir}")
        print(f"Formats: {', '.join(self.formats)}")
        print("-" * 50)

        self.check_dependencies()

        self.generate_class_diagrams()
        self.generate_package_diagrams()
        self.generate_module_specific_diagrams()
        self.generate_flask_routes_diagram()
        self.generate_function_analysis()
        self.generate_architecture_diagram()
        self.generate_dependency_graph()
        self.generate_call_graph()

        if include_brython:
            print()
            self.generate_brython_diagrams()

        print("-" * 50)
        print("+ Diagram generation complete!")
        if "png" in self.formats:
            print(f"   PNG files: {self.png_dir}/")
        if "svg" in self.formats:
            print(f"   SVG files: {self.svg_dir}/")
        if self.font_updates_count > 0:
            print(f"Fonts updated to {DIAGRAM_FONT} in {self.font_updates_count} SVG files")


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate diagrams for MatHud project')
    parser.add_argument('--png-dir', default='../generated_png',
                       help='Output directory for PNG diagrams (default: ../generated_png)')
    parser.add_argument('--svg-dir', default='../generated_svg',
                       help='Output directory for SVG diagrams (default: ../generated_svg)')
    parser.add_argument('--format', default='png,svg',
                       help='Output formats: png,svg,dot (default: png,svg)')

    # Create mutually exclusive group for Brython options
    brython_group = parser.add_mutually_exclusive_group()
    brython_group.add_argument('--include-brython', action='store_true',
                               help='Include comprehensive Brython client-side diagrams')
    brython_group.add_argument('--no-brython', action='store_true',
                               help='Explicitly disable Brython diagrams (overrides default)')

    args = parser.parse_args()
    formats: List[str] = [f.strip() for f in args.format.split(',') if f.strip()]

    # Determine if Brython should be included
    include_brython = args.include_brython and not args.no_brython

    generator = DiagramGenerator(args.png_dir, args.svg_dir, formats)
    generator.run(include_brython=include_brython)


if __name__ == '__main__':
    main()
