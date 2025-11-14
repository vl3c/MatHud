"""
MatHud Base Name Generation System

Abstract base class for systematic naming of mathematical objects.
Provides the foundation for consistent naming conventions across object types.

Key Features:
    - Common name utilities and validation
    - Drawable name retrieval by class type
    - String filtering for mathematical naming
    - Canvas integration for name conflict detection

Dependencies:
    - re: Regular expression pattern matching for name filtering
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import re

if TYPE_CHECKING:
    from canvas import Canvas


ALPHABET: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class NameGenerator:
    """Base class for name generation systems with common utilities.
    
    Provides shared functionality for systematic naming of mathematical objects
    including name filtering, drawable retrieval, and canvas integration.
    
    Attributes:
        canvas (Canvas): Canvas instance for accessing drawable objects
    """
    
    def __init__(self, canvas: "Canvas") -> None:
        """Initialize base name generator with canvas reference.
        
        Args:
            canvas (Canvas): Canvas instance for drawable object access
        """
        self.canvas: "Canvas" = canvas
    
    def get_drawable_names(self, class_name: str) -> List[str]:
        """Get sorted list of names for drawables of a specific class.
        
        Args:
            class_name (str): Class name to filter drawables
            
        Returns:
            list: Sorted list of drawable names for the specified class
        """
        drawables = self.canvas.get_drawables_by_class_name(class_name)
        drawable_names: List[str] = sorted([drawable.name for drawable in drawables])
        return drawable_names
    
    def filter_string(self, name: str) -> str:
        """Filter a string to keep only letters, apostrophes, digits, and parentheses.
        
        Args:
            name (str): Input string to filter
            
        Returns:
            str: Filtered string containing only valid mathematical naming characters
        """
        if not name:
            return ""
        pattern: str = r"[a-zA-Z0-9_'\(\)]+"
        matches: List[str] = re.findall(pattern, name)
        return ''.join(matches) 