"""Shared utilities for diagram generation scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


# Font configuration constants
DIAGRAM_FONT = "Arial"
DIAGRAM_FONT_SIZE = "12"
DIAGRAM_FONT_SIZE_STR = "12"


def setup_graphviz_path() -> None:
    """Setup Graphviz PATH on Windows automatically."""
    try:
        # Check if dot is already available
        try:
            subprocess.run(['dot', '-V'], check=True, capture_output=True)
            print("  + Graphviz dot command is already available")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Only setup on Windows
        if sys.platform != 'win32':
            return
            
        # Common Graphviz installation paths on Windows
        potential_paths: List[Path] = [
            Path("C:/Program Files/Graphviz/bin"),
            Path("C:/Program Files (x86)/Graphviz/bin"),
            Path("C:/Graphviz/bin"),
            Path("C:/tools/Graphviz/bin"),
        ]
        
        # Find Graphviz installation
        graphviz_bin: str | None = None
        for path in potential_paths:
            if path.exists() and (path / "dot.exe").exists():
                graphviz_bin = str(path)
                break
        
        if graphviz_bin:
            # Add to PATH for this session
            current_path = os.environ.get('PATH', '')
            if graphviz_bin not in current_path:
                os.environ['PATH'] = f"{graphviz_bin};{current_path}"
                print(f"  + Added Graphviz to PATH: {graphviz_bin}")
            
            # Verify it works
            try:
                subprocess.run(['dot', '-V'], check=True, capture_output=True)
                print("  + Graphviz dot command is now available")
            except subprocess.CalledProcessError:
                print("  ⚠ Graphviz found but dot command still not working")
        else:
            print("  ⚠ Graphviz not found in common installation paths")
            print("    Please install Graphviz or add it to your PATH manually")
            print("    Install with: winget install graphviz")
            print("    Or download from: https://graphviz.org/download/")
            
    except Exception as e:
        print(f"  ⚠ Could not setup Graphviz PATH: {e}")


def setup_font_environment() -> None:
    """Setup environment variables to use configured font in Graphviz."""
    # Set Graphviz font preferences
    os.environ['FONTNAME'] = DIAGRAM_FONT
    os.environ['FONTSIZE'] = DIAGRAM_FONT_SIZE_STR
    # Some systems use different environment variables
    os.environ['GRAPHVIZ_DOT_FONTNAME'] = DIAGRAM_FONT
    os.environ['GRAPHVIZ_DOT_FONTSIZE'] = DIAGRAM_FONT_SIZE_STR


def post_process_svg_fonts(svg_file: Path, diagram_font: str = DIAGRAM_FONT) -> bool:
    """Post-process SVG file to ensure configured font is used."""
    try:
        if not svg_file.exists():
            return False
            
        content = svg_file.read_text(encoding='utf-8')
        
        # Replace common serif fonts with configured font
        font_replacements: Tuple[Tuple[str, str], ...] = (
            ("Times New Roman", diagram_font),
            ("Times", diagram_font),
            ("serif", f"{diagram_font},sans-serif"),
            ('font-family="Times-Roman"', f'font-family="{diagram_font}"'),
            ("font-family='Times-Roman'", f"font-family='{diagram_font}'"),
            ("Times-Roman", diagram_font),
            ("TimesNewRoman", diagram_font),
        )
        
        modified = False
        for old_font, new_font in font_replacements:
            if old_font in content:
                content = content.replace(old_font, new_font)
                modified = True
        
        # Add default font if no font-family is specified in text elements
        if 'font-family' not in content and '<text' in content:
            content = content.replace('<text', f'<text font-family="{diagram_font}"')
            modified = True
        
        if modified:
            svg_file.write_text(content, encoding='utf-8')
            return True
        return False
            
    except Exception as e:
        print(f"  Warning: Could not update fonts in {svg_file.name}: {e}")
        return False 