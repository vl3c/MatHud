"""
Utility functions for the numeric solver.

Provides initial guess generation and solution deduplication.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence

# Default seed for reproducible random guesses (can be overridden)
_RANDOM_SEED: Optional[int] = 42


def generate_initial_guesses(
    n_vars: int,
    user_guesses: Optional[Sequence[Sequence[float]]] = None,
) -> List[List[float]]:
    """Generate initial guesses for the solver.

    Produces structured guesses plus random points in [-10, 10].

    Args:
        n_vars: Number of variables.
        user_guesses: Optional user-provided starting points.

    Returns:
        List of initial guess vectors.
    """
    guesses: List[List[float]] = []

    # Add user-provided guesses first
    if user_guesses:
        for guess in user_guesses:
            if len(guess) == n_vars:
                guesses.append(list(guess))

    # Structured guesses
    # Origin
    guesses.append([0.0] * n_vars)

    # Unit vectors (positive and negative)
    for i in range(n_vars):
        pos = [0.0] * n_vars
        pos[i] = 1.0
        guesses.append(pos)

        neg = [0.0] * n_vars
        neg[i] = -1.0
        guesses.append(neg)

    # All ones
    guesses.append([1.0] * n_vars)

    # All negative ones
    guesses.append([-1.0] * n_vars)

    # Pi-related values (useful for trigonometric equations)
    guesses.append([math.pi / 6] * n_vars)
    guesses.append([math.pi / 4] * n_vars)
    guesses.append([math.pi / 3] * n_vars)
    guesses.append([math.pi / 2] * n_vars)
    guesses.append([math.pi] * n_vars)

    # Small values
    guesses.append([0.5] * n_vars)
    guesses.append([-0.5] * n_vars)
    guesses.append([0.1] * n_vars)

    # Generate random guesses in [-10, 10]
    # Use a fixed seed for reproducibility
    rng = random.Random(_RANDOM_SEED)
    n_random = 20
    for _ in range(n_random):
        guess = [rng.uniform(-10, 10) for _ in range(n_vars)]
        guesses.append(guess)

    return guesses


def deduplicate_solutions(
    solutions: Sequence[Sequence[float]],
    variables: Sequence[str],
    tolerance: float = 1e-6,
) -> List[Dict[str, float]]:
    """Remove near-duplicate solutions and format as list of dicts.

    Args:
        solutions: List of solution vectors.
        variables: List of variable names.
        tolerance: Tolerance for considering solutions equal.

    Returns:
        List of unique solutions as dictionaries {variable: value}.
    """
    if not solutions:
        return []

    unique: List[List[float]] = []

    for sol in solutions:
        is_duplicate = False
        for existing in unique:
            if _solutions_close(sol, existing, tolerance):
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(list(sol))

    # Round to 10 significant digits and format as dicts
    result: List[Dict[str, float]] = []
    for sol in unique:
        sol_dict: Dict[str, float] = {}
        for var, val in zip(variables, sol):
            sol_dict[var] = _round_to_significant(val, 10)
        result.append(sol_dict)

    return result


def _solutions_close(
    sol1: Sequence[float],
    sol2: Sequence[float],
    tolerance: float,
) -> bool:
    """Check if two solutions are within tolerance."""
    if len(sol1) != len(sol2):
        return False
    return all(abs(a - b) < tolerance for a, b in zip(sol1, sol2))


def _round_to_significant(value: float, digits: int) -> float:
    """Round a value to a specified number of significant digits."""
    if value == 0:
        return 0.0

    magnitude = math.floor(math.log10(abs(value)))
    factor = 10 ** (digits - 1 - magnitude)
    return round(value * factor) / factor
