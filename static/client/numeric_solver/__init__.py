"""
Numeric Solver Package

Multi-start damped Newton-Raphson solver for systems of equations that
cannot be solved symbolically (transcendental, mixed nonlinear, 3+ variables).
"""

from .solver import solve_numeric

__all__ = ['solve_numeric']
