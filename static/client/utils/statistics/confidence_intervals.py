"""Confidence intervals of means and proportions, like a calculator's ZInterval, TInterval,
2-SampZInt, 2-SampTInt, 1-PropZInt and 2-PropZInt, plus a paired t interval.

``run_confidence_interval`` returns the estimate, its standard error, the critical value
z* or t*, the margin of error and the interval. Proportion intervals are Wald intervals
(the sample proportions in the standard error), as on the TI-84, and are not clipped to
[0, 1]. Pure Python with no ``browser`` import.
"""

from __future__ import annotations

from typing import Any, Dict, List

from utils.statistics.hypothesis_tests import (
    df_value,
    reference_distribution,
    require_finite,
    rounded,
    symmetric_critical,
)
from utils.statistics.inference_samples import parse_confidence_level
from utils.statistics.inference_setup import setup_for

INTERVAL_NAMES = {
    "z": "ZInterval",
    "t": "TInterval",
    "two_sample_z": "2-SampZInt",
    "two_sample_t": "2-SampTInt",
    "paired_t": "paired TInterval",
    "one_proportion": "1-PropZInt",
    "two_proportion": "2-PropZInt",
}


def run_confidence_interval(
    interval_type: str,
    sample1: Any,
    sample2: Any,
    confidence_level: Any,
    pooled: Any,
) -> Dict[str, Any]:
    """Build one interval; raises ValueError for invalid or contradictory input."""
    if interval_type not in INTERVAL_NAMES:
        raise ValueError(f"Unsupported interval_type '{interval_type}'; use one of {', '.join(INTERVAL_NAMES)}")
    ignored: List[str] = []
    level = parse_confidence_level(confidence_level)
    setup = setup_for(interval_type, sample1, sample2, pooled, None, False, ignored)
    spec, params = reference_distribution(setup.df)
    critical = symmetric_critical(spec, params, (1.0 - level) / 2.0)
    margin = critical * setup.standard_error
    lower, upper = setup.estimate - margin, setup.estimate + margin
    require_finite(setup.estimate, setup.standard_error, setup.df, critical, lower, upper)
    result: Dict[str, Any] = {
        "interval": interval_type,
        "method": f"{INTERVAL_NAMES[interval_type]}: {setup.method}",
        "parameter": setup.parameter,
        "confidence_level": level,
        "estimate": setup.estimate,
        "standard_error": setup.standard_error,
    }
    if setup.df is not None:
        result["df"] = df_value(setup)
    result["critical_value"] = critical
    result["margin_of_error"] = margin
    result["lower"] = lower
    result["upper"] = upper
    result.update(setup.extras)
    result["samples"] = setup.samples
    notes = list(setup.notes)
    if setup.standard_error == 0.0:
        notes.append(
            "the standard error is 0 (every trial a success or every one a failure), so the interval has no width and is not reliable"
        )
    if notes:
        result["notes"] = notes
    if ignored:
        result["ignored_inputs"] = ignored
    final: Dict[str, Any] = rounded(result)
    return final
