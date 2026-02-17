"""
MatHud Point Name Generation System

Alphabetical naming system for geometric points following mathematical conventions.
Provides systematic progression through single and multiple letter combinations.

Key Features:
    - Alphabetical sequence progression (A, B, C, ..., Z, A', B', ...)
    - Mathematical naming conventions support
    - Name collision detection and resolution
    - Expression parsing for multi-point extraction

Dependencies:
    - re: Regular expression pattern matching for name parsing
    - name_generator.base: Base class functionality
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import re
from .base import ALPHABET, NameGenerator


class PointNameGenerator(NameGenerator):
    """Generates alphabetical names for geometric points.

    Implements systematic alphabetical progression for point naming
    following standard mathematical conventions with apostrophe extensions.

    Attributes:
        canvas (Canvas): Canvas instance for accessing drawable objects
        used_letters_from_names (dict): Tracks which letters have been used for each name
    """

    def __init__(self, canvas: Any) -> None:
        """Initialize point name generator with canvas reference.

        Args:
            canvas (Canvas): Canvas instance for drawable object access
        """
        super().__init__(canvas)
        self.used_letters_from_names: Dict[str, Dict[str, Any]] = {}  # Track which letters have been used for each name

    def reset_state(self) -> None:
        """Reset the internal tracking of used letters from names."""
        self.used_letters_from_names = {}

    def _init_tracking_for_expression(self, expression: str) -> Dict[str, Any]:
        """Initialize tracking for a point expression.

        Args:
            expression (str): Point expression to initialize tracking for

        Returns:
            dict: Tracking data for the expression
        """
        if expression not in self.used_letters_from_names:
            matches: List[str] = re.findall(r"[A-Z][\']*", expression)
            self.used_letters_from_names[expression] = {
                "letters": list(dict.fromkeys(matches)),  # All letters
                "next_index": 0,  # Next unused letter index
            }
        return self.used_letters_from_names[expression]

    def _get_next_letters(self, name_data: Dict[str, Any], n: int) -> List[str]:
        """Extract the next n letters from the name data.

        Args:
            name_data (dict): Name tracking data
            n (int): Number of letters to extract

        Returns:
            list: List of the next n letters
        """
        available_letters: List[str] = name_data["letters"]
        start_index: int = name_data["next_index"]

        result: List[str] = []
        for i in range(n):
            if start_index + i < len(available_letters):
                result.append(available_letters[start_index + i])
            else:
                result.append("")

        # Update the next index
        name_data["next_index"] = min(start_index + n, len(available_letters))
        return result

    def split_point_names(self, expression: Optional[str], n: int = 2) -> List[str]:
        """Split a point expression into individual point names.

        Args:
            expression (str): Point expression to split
            n (int): Number of point names to extract

        Returns:
            list: List of individual point names
        """
        if expression is None or len(expression) < 1:
            return [""] * n

        expression = self.filter_string(expression)
        expression = expression.upper()

        # Initialize tracking for this name
        name_data: Dict[str, Any] = self._init_tracking_for_expression(expression)

        # Get the next n letters
        return self._get_next_letters(name_data, n)

    def _generate_unique_point_name(self) -> str:
        """Generate a unique point name using alphabetical sequence with apostrophes.

        Returns:
            str: Unique point name following alphabetical progression
        """
        point_names: List[str] = self.get_drawable_names("Point")

        return self._find_available_name_from_alphabet(ALPHABET, point_names)

    def _find_available_name_from_alphabet(self, alphabet: str, existing_names: List[str]) -> str:
        """Find an available name from an alphabet, adding apostrophes as needed.

        Args:
            alphabet (str): Alphabet string to iterate through
            existing_names (list): List of existing names to avoid

        Returns:
            str: Available name with appropriate apostrophes
        """
        num_apostrophes: int = 0
        while True:
            for letter in alphabet:
                name: str = letter + "'" * num_apostrophes
                if name not in existing_names:
                    return name
            num_apostrophes += 1

    def _init_tracking_for_preferred_name(self, preferred_name: str) -> Dict[str, Any]:
        """Initialize tracking for a preferred point name.

        Args:
            preferred_name (str): Preferred name to initialize tracking for

        Returns:
            dict: Tracking data for the preferred name
        """
        if preferred_name not in self.used_letters_from_names:
            matches: List[str] = re.findall(r"[A-Z][\']*", preferred_name)
            self.used_letters_from_names[preferred_name] = {
                "letters": list(dict.fromkeys(matches)),  # All available letters with their apostrophes
                "next_index": 0,  # Next unused letter index
            }
        return self.used_letters_from_names[preferred_name]

    def _find_available_name_from_preferred(self, letter_with_apostrophes: str, point_names: List[str]) -> str:
        """Find an available name based on a preferred letter, adding apostrophes if needed.

        Args:
            letter_with_apostrophes (str): Preferred letter with any existing apostrophes
            point_names (list): List of existing point names

        Returns:
            str: Available name based on preferred letter
        """
        base_letter: str = letter_with_apostrophes[0]  # Get just the letter without apostrophes

        # Try the letter as is first
        if letter_with_apostrophes not in point_names:
            return letter_with_apostrophes

        # Try adding apostrophes
        result: Optional[str] = self._try_add_apostrophes(base_letter, point_names)
        return result if result is not None else base_letter

    def _try_add_apostrophes(
        self, base_letter: str, point_names: List[str], initial_count: int = 1, max_attempts: int = 5
    ) -> Optional[str]:
        """Try adding apostrophes to a base letter until finding an unused name.

        Args:
            base_letter (str): Base letter to modify
            point_names (list): List of existing point names
            initial_count (int): Starting number of apostrophes
            max_attempts (int): Maximum apostrophes to try

        Returns:
            str or None: Available name with apostrophes, or None if not found
        """
        num_apostrophes: int = initial_count

        while num_apostrophes <= max_attempts:
            name: str = base_letter + "'" * num_apostrophes
            if name not in point_names:
                return name
            num_apostrophes += 1

        return None  # Could not find an available name with reasonable apostrophes

    def generate_point_name(self, preferred_name: Optional[str]) -> str:
        """Generate a unique point name, using preferred_name if possible.

        Args:
            preferred_name (str): Preferred point name

        Returns:
            str: Unique point name
        """
        if not preferred_name:
            unique_name: str = self._generate_unique_point_name()
            return unique_name

        # Filter and uppercase the preferred name
        preferred_name = self.filter_string(preferred_name).upper()

        point_names: List[str] = self.get_drawable_names("Point")

        # Initialize tracking for this name
        name_data: Dict[str, Any] = self._init_tracking_for_preferred_name(preferred_name)

        available_letters: List[str] = name_data["letters"]
        start_index: int = name_data["next_index"]

        # Try each remaining letter from the preferred name
        for i in range(start_index, len(available_letters)):
            letter_with_apostrophes: str = available_letters[i]

            name: str = self._find_available_name_from_preferred(letter_with_apostrophes, point_names)

            if name:
                name_data["next_index"] = i + 1
                return name

        # If no letters from preferred name are available, generate a unique name
        unique_name = self._generate_unique_point_name()
        return unique_name
