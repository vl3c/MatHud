"""The hypothesis_test, confidence_interval and chi_squared_test tools.

The function registry binds each tool straight to its entry point here; they need no
canvas and draw nothing. Each turns arithmetic overflow into the same ValueError
``compute_distribution`` gives, so the model gets a clear refusal instead of a traceback.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from utils.statistics.chi_squared_tests import run_chi_squared_test
from utils.statistics.confidence_intervals import run_confidence_interval
from utils.statistics.hypothesis_tests import run_hypothesis_test


def _guarded(compute: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    try:
        return compute()
    except (OverflowError, ZeroDivisionError):
        raise ValueError("the inputs are too large or too small to compute") from None


def hypothesis_test_tool(
    test_type: str,
    sample1: Optional[Dict[str, Any]] = None,
    sample2: Optional[Dict[str, Any]] = None,
    hypothesized_value: Optional[float] = None,
    alternative: Optional[str] = None,
    alpha: Optional[float] = None,
    pooled: Optional[bool] = None,
) -> Dict[str, Any]:
    """z, t, two-sample, paired and proportion tests (see ``hypothesis_tests``)."""
    return _guarded(
        lambda: run_hypothesis_test(test_type, sample1, sample2, hypothesized_value, alternative, alpha, pooled)
    )


def confidence_interval_tool(
    interval_type: str,
    sample1: Optional[Dict[str, Any]] = None,
    sample2: Optional[Dict[str, Any]] = None,
    confidence_level: Optional[float] = None,
    pooled: Optional[bool] = None,
) -> Dict[str, Any]:
    """Intervals of means and proportions (see ``confidence_intervals``)."""
    return _guarded(lambda: run_confidence_interval(interval_type, sample1, sample2, confidence_level, pooled))


def chi_squared_test_tool(
    test_type: str,
    observed: Optional[list] = None,
    expected: Optional[list] = None,
    table: Optional[list] = None,
    df: Optional[float] = None,
    alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """Goodness of fit and independence (see ``chi_squared_tests``)."""
    return _guarded(lambda: run_chi_squared_test(test_type, observed, expected, table, df, alpha))
