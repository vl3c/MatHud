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

import subprocess
import argparse
import sys
from pathlib import Path

# Import shared utilities
from utils import (
    setup_graphviz_path, 
    setup_font_environment, 
    post_process_svg_fonts,
    DIAGRAM_FONT,
    DIAGRAM_FONT_SIZE_STR
)

class DiagramGenerator:
    
    def __init__(self, png_dir="../generated_png", svg_dir="../generated_svg", formats=["png", "svg"]):
        # Convert to absolute paths to avoid issues with cwd changes
        self.png_dir = Path(png_dir).resolve()
        self.svg_dir = Path(svg_dir).resolve()
        self.formats = formats
        
        # Setup Graphviz PATH for Windows
        setup_graphviz_path()
        
        # Create output directories
        if "png" in formats:
            self.png_dir.mkdir(exist_ok=True)
        if "svg" in formats:
            self.svg_dir.mkdir(exist_ok=True)
            
        # Setup font configuration for all diagrams
        setup_font_environment()
    


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
        
        # List of all Python files with classes
        class_files = [
            'static/app_manager.py',
            'static/openai_api.py', 
            'static/webdriver_manager.py',
            'static/workspace_manager.py',
            'static/ai_model.py',
            'static/log_manager.py',
            'static/tool_call_processor.py'
        ]
        
        for fmt in self.formats:
            output_dir = self.get_output_dir(fmt)
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
                    post_process_svg_fonts(output_dir / f'classes_MatHud_AllClasses.{fmt}')
                    
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Error generating main class diagram: {e.stderr}")
    
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
                    post_process_svg_fonts(output_dir / f'packages_MatHud_packages.{fmt}')
                    
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Error generating package diagram: {e.stderr}")
    
    def generate_module_specific_diagrams(self):
        """Generate diagrams for specific important modules."""
        print("Generating module-specific diagrams...")
        
        important_modules = [
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
                        print(f"  + {name} diagram generated: {output_dir}/classes_{name}.{fmt}")
                        
                        # Post-process SVG files to use configured font
                        if fmt == 'svg':
                            post_process_svg_fonts(output_dir / f'classes_{name}.{fmt}')
                            
                    except subprocess.CalledProcessError as e:
                        print(f"  ✗ Error generating {name} diagram: {e.stderr}")
    
    def generate_flask_routes_diagram(self):
        """Generate comprehensive Flask routes diagrams and function analysis."""
        print("Generating Flask routes visualization...")
        
        # Generate custom Flask routes diagram
        self._generate_custom_routes_diagram()
        
        # Try pyreverse for any class-like structures
        self._generate_pyreverse_routes_diagram()
        
        # Generate function call diagram for routes.py
        self._generate_function_call_diagram()
    
    def _generate_custom_routes_diagram(self):
        """Create a custom SVG diagram showing Flask routes and their relationships."""
        try:
            import re
            from pathlib import Path
            
            # Read routes.py content
            routes_file = Path('../../static/routes.py')
            if not routes_file.exists():
                print("  ⚠ routes.py not found for custom analysis")
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
                svg_output_dir = self.get_output_dir('svg')
                svg_output_file = svg_output_dir / 'flask_routes_custom.svg'
                svg_output_file.write_text(svg_content)
                print(f"  + Custom Flask routes diagram (SVG): {svg_output_file}")
                
                # Convert to PNG if needed
                if "png" in self.formats:
                    png_output_dir = self.get_output_dir('png')
                    png_output_file = png_output_dir / 'flask_routes_custom.png'
                    self._convert_svg_to_png(svg_output_file, png_output_file)
            
            # If only PNG format requested, create SVG first then convert
            elif "png" in self.formats:
                # Create temporary SVG
                temp_svg = Path('temp_flask_routes.svg')
                temp_svg.write_text(svg_content)
                
                png_output_dir = self.get_output_dir('png')
                png_output_file = png_output_dir / 'flask_routes_custom.png'
                self._convert_svg_to_png(temp_svg, png_output_file)
                
                # Clean up temporary file
                temp_svg.unlink()
                print(f"  + Custom Flask routes diagram (PNG): {png_output_file}")
                    
        except Exception as e:
            print(f"  ⚠ Custom routes diagram generation failed: {e}")
    
    def _convert_svg_to_png(self, svg_file, png_file):
        """Convert SVG file to PNG using cairosvg."""
        try:
            import cairosvg
            cairosvg.svg2png(url=str(svg_file), write_to=str(png_file))
            print(f"  + Converted to PNG: {png_file}")
        except ImportError:
            print("  ⚠ cairosvg not available - trying alternative conversion methods...")
            self._convert_svg_to_png_fallback(svg_file, png_file)
        except Exception as e:
            print(f"  ⚠ SVG to PNG conversion failed: {e}")
            self._convert_svg_to_png_fallback(svg_file, png_file)
    
    def _convert_svg_to_png_fallback(self, svg_file, png_file):
        """Fallback SVG to PNG conversion using dot command."""
        try:
            # Use dot (Graphviz) to convert SVG to PNG
            result = subprocess.run([
                'dot', '-Tpng', str(svg_file), '-o', str(png_file)
            ], capture_output=True, text=True, check=True)
            print(f"  + Converted to PNG (via dot): {png_file}")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠ Fallback conversion failed: {e}")
            print(f"    Consider installing cairosvg: pip install cairosvg")
        except FileNotFoundError:
            print("  ⚠ No conversion tool available (dot command not found)")
            print(f"    Consider installing cairosvg: pip install cairosvg")

    def _create_routes_svg(self, routes, functions):
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
    
    def _generate_pyreverse_routes_diagram(self):
        """Try pyreverse analysis for any class-like structures in routes."""
        try:
            for fmt in self.formats:
                output_dir = self.get_output_dir(fmt)
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
                        post_process_svg_fonts(output_dir / f'classes_FlaskRoutes.{fmt}')
                        
                except subprocess.CalledProcessError:
                    print(f"  ⚠ Pyreverse routes minimal (functions only, no classes)")
                    
        except Exception as e:
            print(f"  ⚠ Pyreverse routes analysis failed: {e}")
    
    def _generate_function_call_diagram(self):
        """Generate function call relationships for routes.py."""
        try:
            # Use pydeps to analyze function calls in routes.py specifically
            output_dir = self.svg_dir if "svg" in self.formats else self.png_dir
            
            cmd = [
                'pydeps',
                '--show-deps',
                '--max-bacon', '2',
                '--cluster',
                '--rankdir', 'LR',
                '--no-show',
                '-o', str(output_dir / 'routes_functions.svg'),
                'static/routes.py'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True, cwd='../..')
            print(f"  + Routes function calls: {output_dir}/routes_functions.svg")
            
            # Post-process SVG file to use configured font
            post_process_svg_fonts(output_dir / 'routes_functions.svg')
            
        except subprocess.CalledProcessError:
            print("  ⚠ Routes function call analysis failed")
        except FileNotFoundError:
            print("  ⚠ pydeps not available for function analysis")
    
    def generate_function_analysis(self):
        """Generate enhanced function analysis for files with minimal class content."""
        print("Generating function analysis for enhanced visualization...")
        
        # Files that may benefit from function analysis
        files_to_analyze = [
            ('static/functions_definitions.py', 'FunctionDefinitions'),
            ('app.py', 'AppMain'),
            ('run_server_tests.py', 'ServerTests')
        ]
        
        for file_path, name in files_to_analyze:
            # Check if file exists
            abs_file_path = Path('../..') / file_path
            if abs_file_path.exists():
                self._generate_file_function_analysis(file_path, name)
    
    def _generate_file_function_analysis(self, file_path, name):
        """Generate function analysis for a specific file."""
        try:
            # Use pydeps for function dependency analysis
            output_dir = self.svg_dir if "svg" in self.formats else self.png_dir
            
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
                post_process_svg_fonts(output_dir / f'functions_{name.lower()}.svg')
                
            except subprocess.CalledProcessError:
                print(f"  ⚠ Function analysis for {name} failed - file may have no dependencies")
                
        except Exception as e:
            print(f"  ⚠ Function analysis error for {name}: {e}")
    
    def generate_architecture_diagram(self):
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
            print(f"  ❌ Missing architecture diagram generator: {e}")
            print("     💡 Make sure generate_arch.py is in the same directory")
        except Exception as e:
            print(f"  ❌ Error generating architecture diagrams: {e}")
            print("     💡 Check that Graphviz is properly installed and accessible")
    
    def generate_dependency_graph(self):
        """Generate a dependency graph using pydeps."""
        print("Generating dependency graph...")
        
        # Dependencies are typically SVG, so use SVG directory if available
        output_dir = self.svg_dir if "svg" in self.formats else self.png_dir
        
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
            post_process_svg_fonts(output_dir / 'dependencies_main.svg')
            
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
            post_process_svg_fonts(output_dir / 'dependencies_static.svg')
            
        except subprocess.CalledProcessError as e:
            print("  ✗ Error generating dependency graph")
            print(f"    Error: {e.stderr if e.stderr else 'Unknown error'}")
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
        self.generate_flask_routes_diagram()
        self.generate_function_analysis()
        self.generate_architecture_diagram()
        self.generate_dependency_graph()
        self.generate_call_graph()
        
        print("-" * 50)
        print(f"+ Diagram generation complete!")
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