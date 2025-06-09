#!/usr/bin/env python3
"""
MatHud Brython Client-Side Diagram Generation Script

Automatically generates comprehensive diagrams for the Brython client-side system:
- Drawable class hierarchy diagrams
- Manager pattern system diagrams  
- Core system architecture diagrams
- Client-server integration flow diagrams
- Package structure diagrams

Target Directory: static/Brython-3.11.3/Lib/site-packages/
Output: generated_png/brython/ and generated_svg/brython/

Dependencies:
    pip install pylint graphviz diagrams
    
Usage:
    python generate_brython_diagrams.py [--png-dir ../generated_png] [--svg-dir ../generated_svg] [--format png,svg]
"""

import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Set

# Import shared utilities
from utils import (
    setup_graphviz_path, 
    setup_font_environment, 
    post_process_svg_fonts,
    DIAGRAM_FONT,
    DIAGRAM_FONT_SIZE_STR
)


class BrythonDiagramGenerator:
    """
    Comprehensive diagram generator for the Brython client-side system.
    
    Generates documentation diagrams for:
    - All 10+ drawable classes (Point, Segment, Vector, Triangle, Rectangle, Circle, Ellipse, Angle, Function, ColoredArea)
    - All 15+ manager classes with relationships
    - Core system classes (Canvas, AIInterface, EventHandler, WorkspaceManager)
    - Utility systems (Expression evaluator, markdown parser, name generators)
    """
    
    def __init__(self, png_dir="../generated_png", svg_dir="../generated_svg", formats=["png", "svg"]):
        # Convert to absolute paths to avoid issues with cwd changes
        self.png_dir = Path(png_dir).resolve()
        self.svg_dir = Path(svg_dir).resolve()
        self.formats = formats
        
        # Setup Graphviz PATH for Windows
        setup_graphviz_path()
        
        # Create client-specific output directories
        self.brython_png_dir = self.png_dir / "client"
        self.brython_svg_dir = self.svg_dir / "client"
        
        if "png" in formats:
            self.brython_png_dir.mkdir(parents=True, exist_ok=True)
            # Create subdirectories for organization
            (self.brython_png_dir / "core").mkdir(exist_ok=True)
            (self.brython_png_dir / "drawables").mkdir(exist_ok=True)
            (self.brython_png_dir / "managers").mkdir(exist_ok=True)
            (self.brython_png_dir / "integration").mkdir(exist_ok=True)
            (self.brython_png_dir / "testing").mkdir(exist_ok=True)
            (self.brython_png_dir / "utilities").mkdir(exist_ok=True)
            
        if "svg" in formats:
            self.brython_svg_dir.mkdir(parents=True, exist_ok=True)
            # Create subdirectories for organization
            (self.brython_svg_dir / "core").mkdir(exist_ok=True)
            (self.brython_svg_dir / "drawables").mkdir(exist_ok=True)
            (self.brython_svg_dir / "managers").mkdir(exist_ok=True)
            (self.brython_svg_dir / "integration").mkdir(exist_ok=True)
            (self.brython_svg_dir / "testing").mkdir(exist_ok=True)
            (self.brython_svg_dir / "utilities").mkdir(exist_ok=True)
            
        # Setup font configuration for all diagrams
        setup_font_environment()
        
        # Track SVG font processing for summary
        self.svg_files_processed = 0
        
        # Define the Brython source directory
        self.brython_source_dir = Path("../../static/Brython-3.11.3/Lib/site-packages").resolve()
        
        # System component definitions
        self.drawable_classes = [
            'point.py', 'segment.py', 'vector.py', 'triangle.py', 'rectangle.py',
            'circle.py', 'ellipse.py', 'angle.py', 'function.py', 'colored_area.py',
            'functions_bounded_colored_area.py', 'function_segment_bounded_colored_area.py',
            'segments_bounded_colored_area.py', 'rotatable_polygon.py', 'drawable.py'
        ]
        
        self.manager_classes = [
            'drawable_manager.py', 'point_manager.py', 'segment_manager.py', 
            'vector_manager.py', 'triangle_manager.py', 'rectangle_manager.py',
            'circle_manager.py', 'ellipse_manager.py', 'angle_manager.py',
            'function_manager.py', 'colored_area_manager.py', 'drawable_dependency_manager.py',
            'drawable_manager_proxy.py', 'transformations_manager.py', 'undo_redo_manager.py',
            'drawables_container.py'
        ]
        
        self.core_system_files = [
            'canvas.py', 'ai_interface.py', 'canvas_event_handler.py',
            'workspace_manager.py', 'result_processor.py', 'process_function_calls.py'
        ]
        
        self.utility_files = [
            'expression_evaluator.py', 'expression_validator.py', 'markdown_parser.py',
            'function_registry.py', 'result_validator.py', 'constants.py', 'geometry.py'
        ]

    def get_brython_output_dir(self, fmt: str, subdir: str = "") -> Path:
        """Get the appropriate Brython output directory for the given format and subdirectory."""
        base_dir = self.brython_png_dir if fmt == "png" else self.brython_svg_dir
        return base_dir / subdir if subdir else base_dir

    def check_brython_source_exists(self) -> bool:
        """Check if the Brython source directory exists."""
        if not self.brython_source_dir.exists():
            print(f"Warning: Brython source directory not found: {self.brython_source_dir}")
            print("   Please ensure the Brython code is properly installed.")
            return False
        return True

    def _process_svg_font_and_count(self, svg_file):
        """Process SVG file for font optimization and track count."""
        if post_process_svg_fonts(svg_file):
            self.svg_files_processed += 1

    def generate_all_brython_diagrams(self):
        """Generate all Brython system diagrams."""
        print("GENERATING COMPREHENSIVE BRYTHON DIAGRAMS")
        print("=" * 60)
        
        if not self.check_brython_source_exists():
            return
            
        # Generate all diagram categories
        self.generate_core_system_diagrams()
        print()
        
        self.generate_drawable_system_diagrams()
        print()
        
        self.generate_manager_system_diagrams()
        print()
        
        self.generate_integration_diagrams()
        print()
        
        self.generate_utility_system_diagrams()
        print()
        
        self.generate_testing_diagrams()
        print()
        
        self.generate_package_structure_diagrams()
        print()
        
        print("Brython diagram generation completed!")
        print(f"   PNG diagrams: {self.brython_png_dir}")
        print(f"   SVG diagrams: {self.brython_svg_dir}")
        if "svg" in self.formats and self.svg_files_processed > 0:
            print(f"Fonts updated to {DIAGRAM_FONT} in {self.svg_files_processed} client SVG files")

    def generate_core_system_diagrams(self):
        """Generate diagrams for core system classes."""
        print("Generating Brython core system diagrams...")
        
        # Core system classes diagram
        core_files = [str(self.brython_source_dir / f) for f in self.core_system_files if (self.brython_source_dir / f).exists()]
        
        if core_files:
            for fmt in self.formats:
                output_dir = self.get_brython_output_dir(fmt, "core")
                cmd = [
                    'pyreverse',
                    '-o', fmt,
                    '-p', 'brython_core_classes',
                    '--output-directory', str(output_dir),
                    '--show-associated', '1',
                    '--show-ancestors', '1'
                ] + core_files
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"  + Core system diagram generated: {output_dir}/classes_brython_core_classes.{fmt}")
                    
                    if fmt == 'svg':
                        self._process_svg_font_and_count(output_dir / f'classes_brython_core_classes.{fmt}')
                        
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Error generating core system diagram: {e.stderr}")

        # Individual core component diagrams
        important_core_modules = [
            ('canvas.py', 'brython_canvas_system'),
            ('ai_interface.py', 'brython_ai_interface'),
            ('canvas_event_handler.py', 'brython_event_handling'),
            ('workspace_manager.py', 'brython_workspace_client'),
            ('result_processor.py', 'brython_result_processor'),
            ('process_function_calls.py', 'brython_function_execution')
        ]
        
        for module_file, diagram_name in important_core_modules:
            module_path = self.brython_source_dir / module_file
            if module_path.exists():
                for fmt in self.formats:
                    output_dir = self.get_brython_output_dir(fmt, "core")
                    cmd = [
                        'pyreverse',
                        '-o', fmt,
                        '-p', diagram_name,
                        '--output-directory', str(output_dir),
                        '--show-associated', '1',
                        str(module_path)
                    ]
                    
                    try:
                        subprocess.run(cmd, check=True, capture_output=True, text=True)
                        print(f"  + {diagram_name} diagram generated: {output_dir}/classes_{diagram_name}.{fmt}")
                        
                        if fmt == 'svg':
                            self._process_svg_font_and_count(output_dir / f'classes_{diagram_name}.{fmt}')
                            
                    except subprocess.CalledProcessError as e:
                        print(f"  ✗ Error generating {diagram_name} diagram: {e.stderr}")

    def generate_drawable_system_diagrams(self):
        """Generate diagrams for the drawable class hierarchy."""
        print("Generating Brython drawable system diagrams...")
        
        drawables_dir = self.brython_source_dir / "drawables"
        if not drawables_dir.exists():
            print(f"  Warning: Drawables directory not found: {drawables_dir}")
            return
            
        # Complete drawable hierarchy diagram
        drawable_files = [str(drawables_dir / f) for f in self.drawable_classes if (drawables_dir / f).exists()]
        
        if drawable_files:
            for fmt in self.formats:
                output_dir = self.get_brython_output_dir(fmt, "drawables")
                cmd = [
                    'pyreverse',
                    '-o', fmt,
                    '-p', 'brython_drawable_hierarchy',
                    '--output-directory', str(output_dir),
                    '--show-associated', '1',
                    '--show-ancestors', '1',
                    '--show-builtin', '0'
                ] + drawable_files
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"  + Drawable hierarchy diagram generated: {output_dir}/classes_brython_drawable_hierarchy.{fmt}")
                    
                    if fmt == 'svg':
                        self._process_svg_font_and_count(output_dir / f'classes_brython_drawable_hierarchy.{fmt}')
                        
                except subprocess.CalledProcessError as e:
                    print(f"  Error generating drawable hierarchy diagram: {e.stderr}")

        # Specific drawable type diagrams
        drawable_categories = [
            (['point.py', 'segment.py', 'vector.py', 'triangle.py', 'rectangle.py', 'circle.py', 'ellipse.py', 'angle.py'], 'brython_geometric_objects'),
            (['function.py'], 'brython_function_plotting'),
            (['colored_area.py', 'functions_bounded_colored_area.py', 'function_segment_bounded_colored_area.py', 'segments_bounded_colored_area.py'], 'brython_colored_areas'),
            (['rotatable_polygon.py', 'drawable.py', 'position.py'], 'brython_base_drawable_system')
        ]
        
        for category_files, diagram_name in drawable_categories:
            category_paths = [str(drawables_dir / f) for f in category_files if (drawables_dir / f).exists()]
            
            if category_paths:
                for fmt in self.formats:
                    output_dir = self.get_brython_output_dir(fmt, "drawables")
                    cmd = [
                        'pyreverse',
                        '-o', fmt,  
                        '-p', diagram_name,
                        '--output-directory', str(output_dir),
                        '--show-associated', '1',
                        '--show-ancestors', '1'
                    ] + category_paths
                    
                    try:
                        subprocess.run(cmd, check=True, capture_output=True, text=True)
                        print(f"  + {diagram_name} diagram generated: {output_dir}/classes_{diagram_name}.{fmt}")
                        
                        if fmt == 'svg':
                            self._process_svg_font_and_count(output_dir / f'classes_{diagram_name}.{fmt}')
                            
                    except subprocess.CalledProcessError as e:
                        print(f"  ✗ Error generating {diagram_name} diagram: {e.stderr}")

    def generate_manager_system_diagrams(self):
        """Generate diagrams for the manager pattern system."""
        print("Generating Brython manager system diagrams...")
        
        managers_dir = self.brython_source_dir / "managers"
        if not managers_dir.exists():
            print(f"  Warning: Managers directory not found: {managers_dir}")
            return
            
        # Complete manager orchestration diagram
        manager_files = [str(managers_dir / f) for f in self.manager_classes if (managers_dir / f).exists()]
        
        if manager_files:
            for fmt in self.formats:
                output_dir = self.get_brython_output_dir(fmt, "managers")
                cmd = [
                    'pyreverse',
                    '-o', fmt,
                    '-p', 'brython_manager_orchestration',
                    '--output-directory', str(output_dir),
                    '--show-associated', '1',
                    '--show-ancestors', '1',
                    '--show-builtin', '0'
                ] + manager_files
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"  + Manager orchestration diagram generated: {output_dir}/classes_brython_manager_orchestration.{fmt}")
                    
                    if fmt == 'svg':
                        self._process_svg_font_and_count(output_dir / f'classes_brython_manager_orchestration.{fmt}')
                        
                except subprocess.CalledProcessError as e:
                    print(f"  Error generating manager orchestration diagram: {e.stderr}")

        # Specific manager category diagrams
        manager_categories = [
            (['point_manager.py', 'segment_manager.py', 'vector_manager.py', 'triangle_manager.py', 'rectangle_manager.py', 'circle_manager.py', 'ellipse_manager.py', 'angle_manager.py'], 'brython_shape_managers'),
            (['function_manager.py', 'colored_area_manager.py'], 'brython_specialized_managers'),
            (['drawable_manager.py', 'drawable_dependency_manager.py', 'drawable_manager_proxy.py', 'drawables_container.py'], 'brython_core_managers'),
            (['transformations_manager.py', 'undo_redo_manager.py'], 'brython_system_managers')
        ]
        
        for category_files, diagram_name in manager_categories:
            category_paths = [str(managers_dir / f) for f in category_files if (managers_dir / f).exists()]
            
            if category_paths:
                for fmt in self.formats:
                    output_dir = self.get_brython_output_dir(fmt, "managers")
                    cmd = [
                        'pyreverse',
                        '-o', fmt,
                        '-p', diagram_name,
                        '--output-directory', str(output_dir),
                        '--show-associated', '1',
                        '--show-ancestors', '1'
                    ] + category_paths
                    
                    try:
                        subprocess.run(cmd, check=True, capture_output=True, text=True)
                        print(f"  + {diagram_name} diagram generated: {output_dir}/classes_{diagram_name}.{fmt}")
                        
                        if fmt == 'svg':
                            self._process_svg_font_and_count(output_dir / f'classes_{diagram_name}.{fmt}')
                            
                    except subprocess.CalledProcessError as e:
                        print(f"  ✗ Error generating {diagram_name} diagram: {e.stderr}")

    def generate_integration_diagrams(self):
        """Generate client-server integration flow diagrams."""
        print("Generating Brython integration diagrams...")
        
        # AJAX and AI integration components
        integration_files = [
            str(self.brython_source_dir / 'ai_interface.py'),
            str(self.brython_source_dir / 'result_processor.py'),
            str(self.brython_source_dir / 'process_function_calls.py'),
            str(self.brython_source_dir / 'workspace_manager.py')
        ]
        
        existing_integration_files = [f for f in integration_files if Path(f).exists()]
        
        if existing_integration_files:
            for fmt in self.formats:
                output_dir = self.get_brython_output_dir(fmt, "integration")
                cmd = [
                    'pyreverse',
                    '-o', fmt,
                    '-p', 'brython_ajax_communication',
                    '--output-directory', str(output_dir),
                    '--show-associated', '1',
                    '--show-ancestors', '1'
                ] + existing_integration_files
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"  + AJAX communication diagram generated: {output_dir}/classes_brython_ajax_communication.{fmt}")
                    
                    if fmt == 'svg':
                        self._process_svg_font_and_count(output_dir / f'classes_brython_ajax_communication.{fmt}')
                        
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Error generating AJAX communication diagram: {e.stderr}")

        # Function execution pipeline
        execution_files = [
            str(self.brython_source_dir / 'process_function_calls.py'),
            str(self.brython_source_dir / 'result_processor.py'),
            str(self.brython_source_dir / 'expression_evaluator.py'),
            str(self.brython_source_dir / 'result_validator.py')
        ]
        
        existing_execution_files = [f for f in execution_files if Path(f).exists()]
        
        if existing_execution_files:
            for fmt in self.formats:
                output_dir = self.get_brython_output_dir(fmt, "integration")
                cmd = [
                    'pyreverse',
                    '-o', fmt,
                    '-p', 'brython_function_execution_pipeline',
                    '--output-directory', str(output_dir),
                    '--show-associated', '1'
                ] + existing_execution_files
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"  + Function execution pipeline diagram generated: {output_dir}/classes_brython_function_execution_pipeline.{fmt}")
                    
                    if fmt == 'svg':
                        self._process_svg_font_and_count(output_dir / f'classes_brython_function_execution_pipeline.{fmt}')
                        
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Error generating function execution pipeline diagram: {e.stderr}")

    def generate_utility_system_diagrams(self):
        """Generate diagrams for utility systems."""
        print("Generating Brython utility system diagrams...")
        
        # Expression and validation utilities
        validation_files = [
            str(self.brython_source_dir / 'expression_evaluator.py'),
            str(self.brython_source_dir / 'expression_validator.py'),
            str(self.brython_source_dir / 'result_validator.py')
        ]
        
        existing_validation_files = [f for f in validation_files if Path(f).exists()]
        
        if existing_validation_files:
            for fmt in self.formats:
                output_dir = self.get_brython_output_dir(fmt, "utilities")
                cmd = [
                    'pyreverse',
                    '-o', fmt,
                    '-p', 'brython_expression_system',
                    '--output-directory', str(output_dir),
                    '--show-associated', '1'
                ] + existing_validation_files
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"  + Expression system diagram generated: {output_dir}/classes_brython_expression_system.{fmt}")
                    
                    if fmt == 'svg':
                        self._process_svg_font_and_count(output_dir / f'classes_brython_expression_system.{fmt}')
                        
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Error generating expression system diagram: {e.stderr}")

        # Content processing utilities
        content_files = [
            str(self.brython_source_dir / 'markdown_parser.py'),
            str(self.brython_source_dir / 'function_registry.py')
        ]
        
        existing_content_files = [f for f in content_files if Path(f).exists()]
        
        if existing_content_files:
            for fmt in self.formats:
                output_dir = self.get_brython_output_dir(fmt, "utilities")
                cmd = [
                    'pyreverse',
                    '-o', fmt,
                    '-p', 'brython_content_processing',
                    '--output-directory', str(output_dir),
                    '--show-associated', '1'
                ] + existing_content_files
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"  + Content processing diagram generated: {output_dir}/classes_brython_content_processing.{fmt}")
                    
                    if fmt == 'svg':
                        self._process_svg_font_and_count(output_dir / f'classes_brython_content_processing.{fmt}')
                        
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Error generating content processing diagram: {e.stderr}")

        # Name generator and utility modules
        utils_dir = self.brython_source_dir / "utils"
        name_gen_dir = self.brython_source_dir / "name_generator"
        
        if utils_dir.exists():
            utils_files = [str(f) for f in utils_dir.glob("*.py") if f.name != "__init__.py"]
            if utils_files:
                for fmt in self.formats:
                    output_dir = self.get_brython_output_dir(fmt, "utilities")
                    cmd = [
                        'pyreverse',
                        '-o', fmt,
                        '-p', 'brython_utils',
                        '--output-directory', str(output_dir),
                        str(utils_dir)
                    ]
                    
                    try:
                        subprocess.run(cmd, check=True, capture_output=True, text=True)
                        print(f"  + Utils system diagram generated: {output_dir}/classes_brython_utils.{fmt}")
                        
                        if fmt == 'svg':
                            self._process_svg_font_and_count(output_dir / f'classes_brython_utils.{fmt}')
                            
                    except subprocess.CalledProcessError as e:
                        print(f"  ✗ Error generating utils system diagram: {e.stderr}")

        if name_gen_dir.exists():
            for fmt in self.formats:
                output_dir = self.get_brython_output_dir(fmt, "utilities")
                cmd = [
                    'pyreverse',
                    '-o', fmt,
                    '-p', 'brython_name_generator',
                    '--output-directory', str(output_dir),
                    str(name_gen_dir)
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"  + Name generator diagram generated: {output_dir}/classes_brython_name_generator.{fmt}")
                    
                    if fmt == 'svg':
                        self._process_svg_font_and_count(output_dir / f'classes_brython_name_generator.{fmt}')
                        
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Error generating name generator diagram: {e.stderr}")

    def generate_testing_diagrams(self):
        """Generate diagrams for the testing framework."""
        print("Generating Brython testing framework diagrams...")
        
        tests_dir = self.brython_source_dir / "ClientTests"
        test_runner_file = self.brython_source_dir / "test_runner.py"
        
        # Test framework structure
        if tests_dir.exists():
            for fmt in self.formats:
                output_dir = self.get_brython_output_dir(fmt, "testing")
                cmd = [
                    'pyreverse',
                    '-o', fmt,
                    '-p', 'brython_test_framework',
                    '--output-directory', str(output_dir),
                    str(tests_dir)
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"  + Test framework diagram generated: {output_dir}/classes_brython_test_framework.{fmt}")
                    
                    if fmt == 'svg':
                        self._process_svg_font_and_count(output_dir / f'classes_brython_test_framework.{fmt}')
                        
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Error generating test framework diagram: {e.stderr}")

        # Test runner system
        if test_runner_file.exists():
            for fmt in self.formats:
                output_dir = self.get_brython_output_dir(fmt, "testing")
                cmd = [
                    'pyreverse',
                    '-o', fmt,
                    '-p', 'brython_test_runner',
                    '--output-directory', str(output_dir),
                    str(test_runner_file)
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"  + Test runner diagram generated: {output_dir}/classes_brython_test_runner.{fmt}")
                    
                    if fmt == 'svg':
                        self._process_svg_font_and_count(output_dir / f'classes_brython_test_runner.{fmt}')
                        
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Error generating test runner diagram: {e.stderr}")

    def generate_package_structure_diagrams(self):
        """Generate package structure overview diagrams."""
        print("Generating Brython package structure diagrams...")
        
        # Complete Brython system overview
        for fmt in self.formats:
            output_dir = self.get_brython_output_dir(fmt)
            cmd = [
                'pyreverse',
                '-o', fmt,
                '-p', 'brython_complete_system',
                '--output-directory', str(output_dir),
                '--show-associated', '1',
                '--show-ancestors', '1',
                '-m', 'yes',  # Show module names
                str(self.brython_source_dir)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"  + Complete system diagram generated: {output_dir}/packages_brython_complete_system.{fmt}")
                
                if fmt == 'svg':
                    self._process_svg_font_and_count(output_dir / f'packages_brython_complete_system.{fmt}')
                    
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Error generating complete system diagram: {e.stderr}")

        # Individual package diagrams
        packages = [
            ('drawables', 'brython_drawables_package'),
            ('managers', 'brython_managers_package'),
            ('utils', 'brython_utils_package'),
            ('name_generator', 'brython_name_generator_package'),
            ('ClientTests', 'brython_tests_package')
        ]
        
        for package_name, diagram_name in packages:
            package_dir = self.brython_source_dir / package_name
            if package_dir.exists():
                for fmt in self.formats:
                    output_dir = self.get_brython_output_dir(fmt)
                    cmd = [
                        'pyreverse',
                        '-o', fmt,
                        '-p', diagram_name,
                        '--output-directory', str(output_dir),
                        '-m', 'yes',
                        str(package_dir)
                    ]
                    
                    try:
                        subprocess.run(cmd, check=True, capture_output=True, text=True)
                        print(f"  + {package_name} package diagram generated: {output_dir}/packages_{diagram_name}.{fmt}")
                        
                        if fmt == 'svg':
                            self._process_svg_font_and_count(output_dir / f'packages_{diagram_name}.{fmt}')
                            
                    except subprocess.CalledProcessError as e:
                        print(f"  ✗ Error generating {package_name} package diagram: {e.stderr}")

    def run(self):
        """Main execution method."""
        self.generate_all_brython_diagrams()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Generate comprehensive Brython client-side diagrams")
    parser.add_argument('--png-dir', default='../generated_png', 
                       help='Output directory for PNG diagrams (default: ../generated_png)')
    parser.add_argument('--svg-dir', default='../generated_svg',
                       help='Output directory for SVG diagrams (default: ../generated_svg)')
    parser.add_argument('--format', default='png,svg',
                       help='Output formats (default: png,svg)')
    
    args = parser.parse_args()
    
    # Parse formats
    formats = [fmt.strip() for fmt in args.format.split(',')]
    
    # Create and run generator
    generator = BrythonDiagramGenerator(
        png_dir=args.png_dir,
        svg_dir=args.svg_dir,
        formats=formats
    )
    
    generator.run()


if __name__ == "__main__":
    main() 