"""
MatHud Function Name Generation System

Mathematical function naming system using standard notation conventions.
Provides systematic naming for mathematical functions and expressions.

Key Features:
    - Standard mathematical notation (f, g, h, f1, f2, ...)
    - Function name incrementation with numeric suffixes
    - Parenthesis handling for function expressions
    - Collision detection and resolution

Dependencies:
    - re: Regular expression pattern matching for name parsing
    - itertools.count: Infinite counting for name generation
    - name_generator.base: Base class functionality
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import re
from itertools import count
from .base import NameGenerator


class FunctionNameGenerator(NameGenerator):
    """Generates mathematical names for function objects.
    
    Implements systematic function naming using mathematical notation
    conventions with support for standard and extended naming patterns.
    
    Attributes:
        canvas (Canvas): Canvas instance for accessing drawable objects
    """
    
    def _extract_number_suffix(self, func_name: str) -> Tuple[str, Optional[int]]:
        """Extract a numeric suffix from a function name if present.
        
        Args:
            func_name (str): Function name to analyze
            
        Returns:
            tuple: (prefix, number) where number is None if no suffix found
        """
        match: Optional[re.Match[str]] = re.search(r'(?<=\w)(\d+)$', func_name)
        if match:
            number: int = int(match.group())
            prefix: str = func_name[:match.start()]
            return prefix, number
        return func_name, None
    
    def _increment_function_name(self, func_name: str) -> str:
        """Increment a function name by adding or incrementing a number suffix.
        
        Args:
            func_name (str): Function name to increment
            
        Returns:
            str: Incremented function name
        """
        func_name = self.filter_string(func_name)
        
        prefix: str
        number: Optional[int]
        prefix, number = self._extract_number_suffix(func_name)
        
        if number is not None:
            return prefix + str(number + 1)
        else:
            return func_name + '1'
    
    def _try_function_name(self, letter: str, number: int, existing_names: List[str]) -> Optional[str]:
        """Try a function name with the given letter and number.
        
        Args:
            letter (str): Function letter (f, g, h, etc.)
            number (int): Number suffix (0 for no suffix)
            existing_names (list): List of existing function names
            
        Returns:
            str or None: Function name if available, None otherwise
        """
        func_name: str = f"{letter}{number if number > 0 else ''}"
        if func_name not in existing_names:
            return func_name
        return None
    
    def _generate_unique_function_name(self) -> str:
        """Generate a unique function name using alphabetical sequence.
        
        Returns:
            str: Unique function name following mathematical conventions
            
        Raises:
            ValueError: If all function names are somehow taken (highly unlikely)
        """
        func_alphabet: str = 'fghijklmnopqrstuvwxyzabcde'
        function_names: List[str] = self.get_drawable_names('Function')
        
        for number in count():
            for letter in func_alphabet:
                name: Optional[str] = self._try_function_name(letter, number, function_names)
                if name:
                    return name
                    
        raise ValueError("All function names are taken")
    
    def _extract_function_name_before_parenthesis(self, preferred_name: str) -> str:
        """Extract the function name before any parenthesis.
        
        Args:
            preferred_name (str): Function name that may contain parentheses
            
        Returns:
            str: Function name without parentheses and arguments
        """
        match: Optional[re.Match[str]] = re.search(r'(?<=\w)(?=\()', preferred_name)
        if match:
            return preferred_name[:match.start()]
        return preferred_name
    
    def _find_available_function_name(self, preferred_name: str, function_names: List[str]) -> str:
        """Find an available function name based on the preferred name.
        
        Args:
            preferred_name (str): Preferred function name
            function_names (list): List of existing function names
            
        Returns:
            str: Available function name with incremented suffix if needed
        """
        if preferred_name not in function_names:
            return preferred_name
            
        # Try incrementing until we find an available name
        current_name: str = preferred_name
        while True:
            current_name = self._increment_function_name(current_name)
            if current_name not in function_names:
                return current_name
    
    def generate_function_name(self, preferred_name: Optional[str]) -> str:
        """Generate a unique function name, using preferred_name if possible.
        
        Args:
            preferred_name (str): Preferred function name
            
        Returns:
            str: Unique function name
        """
        if not preferred_name:
            return self._generate_unique_function_name()
            
        function_names: List[str] = self.get_drawable_names('Function')
        
        # Extract name before parenthesis if present
        clean_name: str = self._extract_function_name_before_parenthesis(preferred_name)
        
        # Find an available function name
        return self._find_available_function_name(clean_name, function_names)

    def reset_state(self) -> None:
        """Reset any internal state for function naming (if any in the future)."""
        # No complex state like used_letters_from_names currently, but good to have for consistency.
        pass 