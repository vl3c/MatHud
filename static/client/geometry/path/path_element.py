"""
MatHud Path Element Abstract Base Class

Abstract base class for elements that compose a path.
A path element can be a straight line segment or a curved arc.

Key Features:
    - Start and end point accessors
    - Sampling for rendering (generates discrete points)
    - Reversal to traverse in opposite direction
    - Length calculation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple


class PathElement(ABC):
    """Abstract element forming part of a path (segment or arc).
    
    Path elements can be chained together to form a CompositePath.
    Each element defines its start and end points, and can generate
    sampled points for rendering.
    """
    
    @abstractmethod
    def start_point(self) -> Tuple[float, float]:
        """Return the starting point of this element."""
        raise NotImplementedError
    
    @abstractmethod
    def end_point(self) -> Tuple[float, float]:
        """Return the ending point of this element."""
        raise NotImplementedError
    
    @abstractmethod
    def sample(self, resolution: int = 32) -> List[Tuple[float, float]]:
        """Generate sampled points along this element.
        
        Args:
            resolution: Number of sample points for curved elements.
                       For line segments, this is ignored (returns just endpoints).
        
        Returns:
            List of (x, y) coordinate tuples.
        """
        raise NotImplementedError
    
    @abstractmethod
    def reversed(self) -> PathElement:
        """Return a new element traversing the same path in reverse direction."""
        raise NotImplementedError
    
    @abstractmethod
    def length(self) -> float:
        """Return the approximate length of this element."""
        raise NotImplementedError
    
    def connects_to(self, other: PathElement, tolerance: float = 1e-9) -> bool:
        """Check if this element's end connects to another element's start.
        
        Args:
            other: The next path element.
            tolerance: Maximum distance to consider points as connected.
        
        Returns:
            True if the elements connect within tolerance.
        """
        end = self.end_point()
        start = other.start_point()
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        return (dx * dx + dy * dy) <= tolerance * tolerance

