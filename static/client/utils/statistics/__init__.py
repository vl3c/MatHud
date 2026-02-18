"""Statistics utilities for regression, distribution, and descriptive analysis.

This package provides statistical functions for curve fitting,
probability distribution generation, and descriptive statistics.

Key Features:
    - Multiple regression model types (linear, polynomial, exponential, etc.)
    - R-squared calculation for goodness of fit
    - Expression building for fitted models
    - Normal distribution PDF expression generation
    - Descriptive statistics (mean, median, mode, quartiles, etc.)
"""

from __future__ import annotations

from utils.statistics.descriptive import (
    DescriptiveStatisticsResult,
    compute_descriptive_statistics,
)
from utils.statistics.regression import (
    RegressionResult,
    SUPPORTED_MODEL_TYPES,
    fit_regression,
    fit_linear,
    fit_polynomial,
    fit_exponential,
    fit_logarithmic,
    fit_power,
    fit_logistic,
    fit_sinusoidal,
    calculate_r_squared,
    build_expression,
)
