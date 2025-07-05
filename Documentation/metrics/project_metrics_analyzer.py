#!/usr/bin/env python3
"""
MatHud Project Metrics Analyzer

This script analyzes the MatHud project codebase to generate comprehensive metrics
including lines of code, number of classes, methods, tests, and other useful statistics.

Usage:
    python project_metrics_analyzer.py

Output:
    - Detailed metrics printed to console
    - Summary table saved to project_overview_table.txt
    - Detailed report saved to detailed_project_metrics.txt
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class ProjectMetricsAnalyzer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.metrics = {
            'files': defaultdict(int),
            'lines': defaultdict(int),
            'classes': 0,
            'methods': 0,
            'functions': 0,
            'test_functions': 0,
            'ai_functions': 0,
            'drawable_classes': 0,
            'manager_classes': 0,
            'imports': 0,
            'comments': 0,
            'docstrings': 0,
            'docstring_lines': 0,
            'reference_manual_lines': 0,
            'unique_python_imports': set(),
            'python_dependencies': 0,
            'javascript_libraries': 0,
            'test_files': 0,
            'file_details': defaultdict(list)
        }
        
        # File extensions to analyze
        self.extensions = {
            '.py': 'Python',
            '.html': 'HTML',
            '.css': 'CSS',
            '.txt': 'Text',
            '.md': 'Markdown',
            # '.js': 'JavaScript',
            '.json': 'JSON'
        }
        
        # Directories to exclude
        self.exclude_dirs = {
            '__pycache__', '.git', 'venv', '.vscode', '.pytest_cache',
            'logs', 'workspaces', 'canvas_snapshots', 'generated_svg', 'generated_png'
        }
    
    def analyze_project(self):
        """Main analysis method that walks through the project directory."""
        print(f"Analyzing MatHud project at: {self.project_root}")
        print("=" * 60)
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            for file in files:
                file_path = Path(root) / file
                self.analyze_file(file_path)
        
        # Analyze dependencies after all files are processed
        self.analyze_dependencies()
        self.generate_reports()
    
    def analyze_file(self, file_path):
        """Analyze a single file for various metrics."""
        try:
            suffix = file_path.suffix.lower()
            if suffix not in self.extensions:
                return
            
            file_type = self.extensions[suffix]
            self.metrics['files'][file_type] += 1
            
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
            except UnicodeDecodeError:
                # Skip binary files
                return
            
            line_count = len(lines)
            
            # Handle Reference Manual separately (contains duplicated docstrings)
            is_reference_manual = file_path.name == 'Reference Manual.txt'
            if is_reference_manual:
                # Track but don't count toward documentation totals
                self.metrics['reference_manual_lines'] = line_count
            else:
                self.metrics['lines'][file_type] += line_count
            
            # Store file details
            relative_path = file_path.relative_to(self.project_root)
            file_info = {
                'path': str(relative_path),
                'lines': line_count,
                'size': file_path.stat().st_size,
                'is_reference_manual': is_reference_manual
            }
            
            # Analyze Python files in detail
            if suffix == '.py':
                py_metrics = self.analyze_python_file(content, file_path)
                file_info.update(py_metrics)
                
                # Subtract docstring lines from Python code lines
                docstring_lines = py_metrics.get('docstring_lines', 0)
                self.metrics['lines'][file_type] -= docstring_lines
                self.metrics['docstring_lines'] += docstring_lines
                file_info['lines'] -= docstring_lines
                file_info['docstring_lines'] = docstring_lines
            
            # Count test files from server_tests and client_tests directories
            relative_path = file_path.relative_to(self.project_root)
            if any(part in str(relative_path).lower() for part in ['server_tests', 'client_tests']):
                if file_path.suffix.lower() == '.py':  # Only count Python test files
                    self.metrics['test_files'] += 1
                    file_info['is_test_file'] = True
            
            self.metrics['file_details'][file_type].append(file_info)
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    def analyze_python_file(self, content, file_path):
        """Detailed analysis of Python files."""
        lines = content.split('\n')
        file_metrics = {
            'classes': 0,
            'methods': 0,
            'functions': 0,
            'test_functions': 0,
            'imports': 0,
            'comments': 0,
            'docstrings': 0,
            'docstring_lines': 0,
            'is_drawable': False,
            'is_manager': False,
            'is_test': False
        }
        
        in_multiline_string = False
        string_delimiter = None
        
        for line in lines:
            stripped = line.strip()
            
            # Handle multiline strings/docstrings
            if not in_multiline_string:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    string_delimiter = stripped[:3]
                    # Check if it's a single-line docstring
                    if stripped.endswith(string_delimiter) and len(stripped) > 6:
                        # Single-line docstring
                        file_metrics['docstrings'] += 1
                        self.metrics['docstrings'] += 1
                        file_metrics['docstring_lines'] += 1
                    else:
                        # Multi-line docstring starts
                        in_multiline_string = True
                        file_metrics['docstrings'] += 1
                        self.metrics['docstrings'] += 1
                        file_metrics['docstring_lines'] += 1
            else:
                # Inside multiline docstring
                file_metrics['docstring_lines'] += 1
                if string_delimiter in stripped:
                    in_multiline_string = False
                continue
            
            if in_multiline_string:
                continue
            
            # Class definitions
            if re.match(r'^class\s+\w+', stripped):
                file_metrics['classes'] += 1
                self.metrics['classes'] += 1
                
                # Check for specific class types
                if 'drawable' in file_path.name.lower() or any(
                    keyword in stripped.lower() for keyword in ['drawable', 'point', 'segment', 'circle', 'triangle', 'rectangle', 'ellipse', 'vector', 'angle', 'function']
                ):
                    file_metrics['is_drawable'] = True
                    if 'drawable' in stripped.lower():
                        self.metrics['drawable_classes'] += 1
                
                if 'manager' in stripped.lower():
                    file_metrics['is_manager'] = True
                    self.metrics['manager_classes'] += 1
            
            # Method definitions (inside classes) - use original line for indentation
            if re.match(r'^\s+def\s+\w+', line):
                file_metrics['methods'] += 1
                self.metrics['methods'] += 1
                
                # Test methods
                if 'def test_' in stripped:
                    file_metrics['test_functions'] += 1
                    self.metrics['test_functions'] += 1
            
            # Function definitions (at module level)
            elif re.match(r'^def\s+\w+', stripped):
                file_metrics['functions'] += 1
                self.metrics['functions'] += 1
                
                # Test functions
                if 'def test_' in stripped:
                    file_metrics['test_functions'] += 1
                    self.metrics['test_functions'] += 1
            
            # Import statements
            elif stripped.startswith('import ') or stripped.startswith('from '):
                file_metrics['imports'] += 1
                self.metrics['imports'] += 1
                
                # Track unique imports for dependency analysis
                import_module = self.extract_import_module(stripped)
                if import_module:
                    self.metrics['unique_python_imports'].add(import_module)
            
            # Comments
            elif stripped.startswith('#'):
                file_metrics['comments'] += 1
                self.metrics['comments'] += 1
        
        # Check if it's a test file
        if 'test' in file_path.name.lower():
            file_metrics['is_test'] = True
        
        # Special case: analyze functions_definitions.py for AI functions
        if file_path.name == 'functions_definitions.py':
            ai_functions = self.count_ai_functions(content)
            self.metrics['ai_functions'] = ai_functions
            file_metrics['ai_functions'] = ai_functions
        
        return file_metrics
    
    def extract_import_module(self, import_line):
        """Extract the main module name from an import statement."""
        try:
            # Handle 'import module' and 'from module import ...'
            if import_line.startswith('import '):
                module = import_line[7:].split('.')[0].split(' as ')[0].split(',')[0].strip()
            elif import_line.startswith('from '):
                module = import_line[5:].split('.')[0].split(' import')[0].strip()
            else:
                return None
            
            # Filter out relative imports and local modules
            if module and not module.startswith('.') and module.isidentifier():
                return module
            return None
        except:
            return None
    
    def analyze_dependencies(self):
        """Analyze Python and JavaScript dependencies."""
        # Analyze Python requirements files
        deps = set()
        
        # Main requirements.txt
        requirements_files = [
            self.project_root / 'requirements.txt',
            self.project_root / 'diagrams' / 'diagram_requirements.txt'
        ]
        
        for requirements_file in requirements_files:
            if requirements_file.exists():
                try:
                    with open(requirements_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
                        # Count dependencies (remove version specs)
                        for line in lines:
                            dep = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('!=')[0].strip()
                            if dep:
                                deps.add(dep)
                except:
                    pass
        
        self.metrics['python_dependencies'] = len(deps)
        
        # Analyze index.html for JavaScript libraries
        index_file = self.project_root / 'templates' / 'index.html'
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Count script tags and CDN libraries
                    js_libs = set()
                    
                    # Look for script src tags
                    import re
                    script_pattern = r'<script[^>]+src=["\']([^"\']+)["\']'
                    matches = re.findall(script_pattern, content, re.IGNORECASE)
                    
                    for src in matches:
                        if 'http' in src or 'cdn' in src:
                            # External library
                            lib_name = src.split('/')[-1].split('.')[0]
                            js_libs.add(lib_name)
                        elif '.js' in src:
                            # Local library
                            lib_name = src.split('/')[-1].split('.')[0]
                            js_libs.add(lib_name)
                    
                    # Also check for specific known libraries mentioned in text
                    if 'brython' in content.lower():
                        js_libs.add('brython')
                    if 'mathjax' in content.lower():
                        js_libs.add('mathjax')
                    if 'nerdamer' in content.lower():
                        js_libs.add('nerdamer')
                    
                    self.metrics['javascript_libraries'] = len(js_libs)
            except:
                pass
    
    def count_ai_functions(self, content):
        """Count AI function definitions in functions_definitions.py."""
        # Look for function definitions in the AI functions file
        function_pattern = r'"name":\s*"([^"]+)"'
        matches = re.findall(function_pattern, content)
        return len(matches)
    
    def generate_reports(self):
        """Generate both summary and detailed reports."""
        self.print_summary()
        self.save_overview_table()
        self.save_detailed_report()
    
    def print_summary(self):
        """Print summary metrics to console."""
        print("\n" + "="*60)
        print("MATHUD PROJECT METRICS SUMMARY")
        print("="*60)
        
        print(f"\nFILE STATISTICS:")
        total_files = sum(self.metrics['files'].values())
        total_lines = sum(self.metrics['lines'].values())
        print(f"   Total Files: {total_files:,}")
        print(f"   Total Lines: {total_lines:,}")
        
        print(f"\nFILES BY TYPE:")
        for file_type, count in sorted(self.metrics['files'].items()):
            lines = self.metrics['lines'][file_type]
            print(f"   {file_type:>10}: {count:>3} files, {lines:>6,} lines")
        
        print(f"\nCODE STRUCTURE:")
        print(f"   Classes:          {self.metrics['classes']:>4}")
        print(f"   Methods:          {self.metrics['methods']:>4}")
        print(f"   Functions:        {self.metrics['functions']:>4}")
        
        print(f"\nDEPENDENCIES:")
        print(f"   Python Dependencies:{self.metrics['python_dependencies']:>4}")
        print(f"   JavaScript Libraries:{self.metrics['javascript_libraries']:>3}")
        print(f"   Import Statements:{self.metrics['imports']:>4}")
        print(f"   Unique Python Imports:{len(self.metrics['unique_python_imports']):>3}")
        
        print(f"\nDOCUMENTATION:")
        print(f"   Comments:         {self.metrics['comments']:>4}")
        print(f"   Docstrings:       {self.metrics['docstrings']:>4}")
        print(f"   Docstring Lines:  {self.metrics['docstring_lines']:>4}")
        
        print(f"\nTESTING:")
        print(f"   Test Files:       {self.metrics['test_files']:>4}")
        print(f"   Test Functions:   {self.metrics['test_functions']:>4}")
        
        print(f"\nAI INTEGRATION:")
        print(f"   AI Functions:     {self.metrics['ai_functions']:>4}")
        
        print(f"\nSPECIALIZED CLASSES:")
        print(f"   Drawable Classes: {self.metrics['drawable_classes']:>4}")
        print(f"   Manager Classes:  {self.metrics['manager_classes']:>4}")
    
    def save_overview_table(self):
        """Save a formatted overview table to file."""
        output_file = self.project_root / 'Documentation' / 'Project Overview Table.txt'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("MatHud Project Overview Table\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Main metrics table
            f.write("PROJECT METRICS OVERVIEW\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Metric':<25} {'Count':<10} {'Details':<15}\n")
            f.write("-" * 50 + "\n")
            
            total_files = sum(self.metrics['files'].values())
            total_lines = sum(self.metrics['lines'].values())
            
            f.write(f"{'Total Files':<25} {total_files:<10,} {'All types':<15}\n")
            f.write(f"{'Total Lines of Code':<25} {total_lines:<10,} {'All files':<15}\n")
            f.write(f"{'Python Files':<25} {self.metrics['files']['Python']:<10} {self.metrics['lines']['Python']:,} lines\n")
            f.write(f"{'HTML Files':<25} {self.metrics['files']['HTML']:<10} {self.metrics['lines']['HTML']:,} lines\n")
            f.write(f"{'CSS Files':<25} {self.metrics['files']['CSS']:<10} {self.metrics['lines']['CSS']:,} lines\n")
            f.write(f"{'JavaScript Files':<25} {self.metrics['files']['JavaScript']:<10} {self.metrics['lines']['JavaScript']:,} lines\n")
            f.write(f"{'Documentation Files':<25} {self.metrics['files']['Text'] + self.metrics['files']['Markdown']:<10} {self.metrics['lines']['Text'] + self.metrics['lines']['Markdown']:,} lines\n")
            
            f.write("\n" + "-" * 50 + "\n")
            f.write("CODE ARCHITECTURE\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Classes':<25} {self.metrics['classes']:<10} {'OOP Design':<15}\n")
            f.write(f"{'Methods':<25} {self.metrics['methods']:<10} {'Class methods':<15}\n")
            f.write(f"{'Functions':<25} {self.metrics['functions']:<10} {'Module functions':<15}\n")
            f.write(f"{'Drawable Classes':<25} {self.metrics['drawable_classes']:<10} {'Geometric shapes':<15}\n")
            f.write(f"{'Manager Classes':<25} {self.metrics['manager_classes']:<10} {'System managers':<15}\n")
            
            f.write("\n" + "-" * 50 + "\n")
            f.write("DEPENDENCIES\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Python Dependencies':<25} {self.metrics['python_dependencies']:<10} {'External packages':<15}\n")
            f.write(f"{'JavaScript Libraries':<25} {self.metrics['javascript_libraries']:<10} {'Frontend libs':<15}\n")
            f.write(f"{'Import Statements':<25} {self.metrics['imports']:<10} {'All imports':<15}\n")
            f.write(f"{'Unique Python Imports':<25} {len(self.metrics['unique_python_imports']):<10} {'Distinct modules':<15}\n")
            
            f.write("\n" + "-" * 50 + "\n")
            f.write("TESTING\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Test Files':<25} {self.metrics['test_files']:<10} {'Test modules':<15}\n")
            f.write(f"{'Test Functions':<25} {self.metrics['test_functions']:<10} {'Unit tests':<15}\n")
            
            f.write("\n" + "-" * 50 + "\n")
            f.write("DOCUMENTATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Comments':<25} {self.metrics['comments']:<10} {'Code comments':<15}\n")
            f.write(f"{'Docstrings':<25} {self.metrics['docstrings']:<10} {'Documentation':<15}\n")
            f.write(f"{'Docstring Lines':<25} {self.metrics['docstring_lines']:<10} {'Doc content':<15}\n")
            
            f.write("\n" + "-" * 50 + "\n")
            f.write("AI INTEGRATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'AI Functions':<25} {self.metrics['ai_functions']:<10} {'Tool definitions':<15}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("PROJECT SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write("MatHud is a comprehensive mathematical visualization tool featuring:\n\n")
            f.write(f"• {total_lines:,} lines of code across {total_files} files\n")
            f.write(f"• {self.metrics['classes']} classes with {self.metrics['methods']} methods\n")
            f.write(f"• {self.metrics['drawable_classes']} drawable geometric shapes\n")
            f.write(f"• {self.metrics['manager_classes']} specialized system managers\n")
            f.write(f"• {self.metrics['ai_functions']} AI-powered mathematical functions\n")
            f.write(f"• {self.metrics['test_functions']} test functions for quality assurance\n")
            f.write(f"• Flask backend with Brython frontend architecture\n")
            f.write(f"• Interactive SVG canvas with real-time mathematical visualization\n\n")
            f.write(f"NOTE: Reference Manual.txt ({self.metrics['reference_manual_lines']:,} lines) contains duplicated docstring\n")
            f.write("content and is excluded from documentation totals to avoid double-counting.\n")
        
        print(f"\nOverview table saved to: {output_file}")
    
    def save_detailed_report(self):
        """Save detailed metrics report to file."""
        output_file = self.project_root / 'Documentation' / 'metrics' / 'detailed_project_metrics.txt'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("MatHud Project - Detailed Metrics Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Detailed file breakdown by type
            for file_type in sorted(self.metrics['file_details'].keys()):
                files = self.metrics['file_details'][file_type]
                if not files:
                    continue
                
                f.write(f"\n{file_type.upper()} FILES ({len(files)} files)\n")
                f.write("-" * 40 + "\n")
                
                # Sort files by line count (descending)
                files.sort(key=lambda x: x['lines'], reverse=True)
                
                for file_info in files:
                    f.write(f"{file_info['path']:<40} {file_info['lines']:>6} lines")
                    
                    if file_type == 'Python':
                        if file_info.get('classes', 0) > 0:
                            f.write(f" | {file_info['classes']} classes")
                        if file_info.get('methods', 0) > 0:
                            f.write(f" | {file_info['methods']} methods")
                        if file_info.get('functions', 0) > 0:
                            f.write(f" | {file_info['functions']} functions")
                        if file_info.get('test_functions', 0) > 0:
                            f.write(f" | {file_info['test_functions']} tests")
                        if file_info.get('ai_functions', 0) > 0:
                            f.write(f" | {file_info['ai_functions']} AI functions")
                    
                    # Mark Reference Manual as duplicated content
                    if file_info.get('is_reference_manual', False):
                        f.write(f" | {file_info['size']:,} bytes | EXCLUDED (duplicated docstrings)\n")
                    else:
                        f.write(f" | {file_info['size']:,} bytes\n")
                
                total_lines = sum(f['lines'] for f in files)
                total_size = sum(f['size'] for f in files)
                f.write(f"\nSubtotal: {total_lines:,} lines, {total_size:,} bytes\n")
        
        print(f"Detailed report saved to: {output_file}")

def main():
    """Main function to run the project analysis."""
    # Get the project root (parent directory of Documentation)
    script_path = Path(__file__).parent
    project_root = script_path.parent.parent
    
    print("MatHud Project Metrics Analyzer")
    print("=" * 40)
    print(f"Project Root: {project_root}")
    
    analyzer = ProjectMetricsAnalyzer(project_root)
    analyzer.analyze_project()
    
    print(f"\nAnalysis complete! Generated files:")
    print(f"   • Documentation/Project Overview Table.txt (summary)")
    print(f"   • Documentation/metrics/detailed_project_metrics.txt (detailed breakdown)")

if __name__ == "__main__":
    main() 