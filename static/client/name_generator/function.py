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
    
    def _extract_number_suffix(self, func_name):
        """Extract a numeric suffix from a function name if present.
        
        Args:
            func_name (str): Function name to analyze
            
        Returns:
            tuple: (prefix, number) where number is None if no suffix found
        """
        match = re.search(r'(?<=\w)(\d+)$', func_name)
        if match:
            number = int(match.group())
            prefix = func_name[:match.start()]
            return prefix, number
        return func_name, None
    
    def _increment_function_name(self, func_name):
        """Increment a function name by adding or incrementing a number suffix.
        
        Args:
            func_name (str): Function name to increment
            
        Returns:
            str: Incremented function name
        """
        func_name = self.filter_string(func_name)
        
        prefix, number = self._extract_number_suffix(func_name)
        
        if number is not None:
            return prefix + str(number + 1)
        else:
            return func_name + '1'
    
    def _try_function_name(self, letter, number, existing_names):
        """Try a function name with the given letter and number.
        
        Args:
            letter (str): Function letter (f, g, h, etc.)
            number (int): Number suffix (0 for no suffix)
            existing_names (list): List of existing function names
            
        Returns:
            str or None: Function name if available, None otherwise
        """
        func_name = f"{letter}{number if number > 0 else ''}"
        if func_name not in existing_names:
            return func_name
        return None
    
    def _generate_unique_function_name(self):
        """Generate a unique function name using alphabetical sequence.
        
        Returns:
            str: Unique function name following mathematical conventions
            
        Raises:
            ValueError: If all function names are somehow taken (highly unlikely)
        """
        func_alphabet = 'fghijklmnopqrstuvwxyzabcde'
        function_names = self.get_drawable_names('Function')
        
        for number in count():
            for letter in func_alphabet:
                name = self._try_function_name(letter, number, function_names)
                if name:
                    return name
                    
        raise ValueError("All function names are taken")
    
    def _extract_function_name_before_parenthesis(self, preferred_name):
        """Extract the function name before any parenthesis.
        
        Args:
            preferred_name (str): Function name that may contain parentheses
            
        Returns:
            str: Function name without parentheses and arguments
        """
        match = re.search(r'(?<=\w)(?=\()', preferred_name)
        if match:
            return preferred_name[:match.start()]
        return preferred_name
    
    def _find_available_function_name(self, preferred_name, function_names):
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
        current_name = preferred_name
        while True:
            current_name = self._increment_function_name(current_name)
            if current_name not in function_names:
                return current_name
    
    def generate_function_name(self, preferred_name):
        """Generate a unique function name, using preferred_name if possible.
        
        Args:
            preferred_name (str): Preferred function name
            
        Returns:
            str: Unique function name
        """
        if not preferred_name:
            return self._generate_unique_function_name()
            
        function_names = self.get_drawable_names('Function')
        
        # Extract name before parenthesis if present
        clean_name = self._extract_function_name_before_parenthesis(preferred_name)
        
        # Find an available function name
        return self._find_available_function_name(clean_name, function_names)

    def reset_state(self):
        """Reset any internal state for function naming (if any in the future)."""
        # No complex state like used_letters_from_names currently, but good to have for consistency.
        pass 