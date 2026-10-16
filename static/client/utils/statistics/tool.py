"""The compute_descriptive_statistics tool: one list, or paired x and y lists.

The function registry binds the tool straight to this pure entry point; it
needs no canvas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils.statistics.descriptive import compute_descriptive_statistics
from utils.statistics.two_variable import compute_two_variable_statistics

SAMPLE_FIGURES_NOTE = "sample_standard_deviation and sample_variance need at least two values"


def compute_descriptive_statistics_tool(data: List[float], y_data: Optional[List[float]] = None) -> Dict[str, Any]:
    """Run the tool: one-variable statistics of data, or two-variable ones with y_data.

    A null or empty y_data means one list (local models send [] for an unused
    array). The result gains a note when a figure is null: the sample figures
    of a single value, or r of a list whose values are all equal.
    """
    if y_data is None or (isinstance(y_data, list) and len(y_data) == 0):
        result: Dict[str, Any] = dict(compute_descriptive_statistics(data))
        if result["sample_standard_deviation"] is None:
            result["note"] = SAMPLE_FIGURES_NOTE
        return result
    paired: Dict[str, Any] = dict(compute_two_variable_statistics(data, y_data))
    if paired["r"] is None:
        paired["note"] = _undefined_r_note(paired["x"], paired["y"])
    return paired


def _undefined_r_note(x: Dict[str, Any], y: Dict[str, Any]) -> str:
    equal = [name for name, stats in (("x", x), ("y", y)) if stats["min"] == stats["max"]]
    return (
        "r and r_squared are undefined because " + " and ".join(f"all {name} values" for name in equal) + " are equal"
    )
