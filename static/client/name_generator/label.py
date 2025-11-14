"""
MatHud Label Name Generation System

Provides deterministic label naming when users omit a name or when duplicates
must be resolved. Uses an alphabetical sequence (label_A, label_B, …) and
falls back to numbered suffixes when necessary.
"""

from __future__ import annotations

from typing import Optional

from .base import ALPHABET, NameGenerator


class LabelNameGenerator(NameGenerator):
    """Generates unique names for Label drawables."""

    def generate_label_name(self, preferred_name: Optional[str]) -> str:
        """Return a unique label name, honoring the preferred value when possible."""
        normalized = self._normalize_preferred_name(preferred_name)
        existing = set(self.get_drawable_names("Label"))

        if normalized:
            candidate = normalized
            suffix = 1
            while candidate in existing:
                candidate = f"{normalized}_{suffix}"
                suffix += 1
            return candidate

        for letter in ALPHABET:
            candidate = f"label_{letter}"
            if candidate not in existing:
                return candidate

        suffix = 1
        while True:
            candidate = f"label_{suffix}"
            if candidate not in existing:
                return candidate
            suffix += 1

    def _normalize_preferred_name(self, preferred_name: Optional[str]) -> str:
        if preferred_name is None:
            return ""
        trimmed = preferred_name.strip()
        if not trimmed:
            return ""
        filtered = self.filter_string(trimmed)
        return filtered or trimmed

