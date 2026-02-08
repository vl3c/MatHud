"""Name generator package."""

from __future__ import annotations

__all__: list[str] = []
"""
Name generator package for managing unique names of drawable objects.
"""

from .drawable import DrawableNameGenerator
from .base import NameGenerator
from .point import PointNameGenerator
from .function import FunctionNameGenerator
from .arc import ArcNameGenerator

__all__ = [
    'DrawableNameGenerator',
    'NameGenerator',
    'PointNameGenerator',
    'FunctionNameGenerator',
    'ArcNameGenerator',
]
