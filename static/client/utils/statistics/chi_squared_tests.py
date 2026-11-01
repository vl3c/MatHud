"""χ² tests: goodness of fit and independence (or homogeneity) of a contingency table.

``run_chi_squared_test`` returns the statistic Σ(O - E)²/E, df, the upper-tail p-value,
the critical value at alpha, the decision, the expected counts and each category's or
cell's contribution, like a calculator's χ²GOF-Test and χ²-Test. No continuity
correction is applied (as on the TI-84).

Expected values for goodness of fit are counts or probabilities. Rounded textbook values
are accepted: probabilities summing to within 0.01 of 1 are rescaled, counts summing to
within 1% of the observed total are used as given; each gets a note. Pure Python with no
``browser`` import.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from utils.statistics.distributions import get_distribution
from utils.statistics.hypothesis_tests import decide, number_text, require_finite, rounded
from utils.statistics.inference_samples import parse_alpha, parse_counts, whole_number
from utils.statistics.special_functions import bisect_increasing

CHI_SQUARED_TEST_TYPES = ("goodness_of_fit", "independence")
# Expected counts below this make the χ² approximation poor.
MIN_EXPECTED_COUNT = 5.0
PROBABILITY_SUM_TOLERANCE = 0.01
COUNT_SUM_TOLERANCE = 0.01  # relative to the observed total
EXACT_SUM_TOLERANCE = 1e-9


def chi_squared_upper_critical(df: float, tail: float) -> float:
    """The x with P(X > x) = tail for χ² with ``df``, bisecting the upper tail directly."""
    spec = get_distribution("chi_squared")
    params = {"df": df}
    hi = max(1.0, df)
    for _ in range(2000):
        if spec.sf(hi, params) < tail:
            break
        hi *= 2.0
    return float(bisect_increasing(lambda x: -spec.sf(x, params), -tail, (0.0, hi)))


def _goodness_of_fit_expected(observed: List[float], expected: Any, notes: List[str]) -> List[float]:
    total = math.fsum(observed)
    k = len(observed)
    if expected is None:
        return [total / k] * k
    values: List[float] = parse_counts(expected, "expected")
    if len(values) != k:
        raise ValueError(f"expected has {len(values)} values but observed has {k}; give one per category")
    if any(value <= 0.0 for value in values):
        raise ValueError("every expected value must be > 0")
    given = math.fsum(values)
    near_one = abs(given - 1.0) <= PROBABILITY_SUM_TOLERANCE
    near_total = abs(given - total) <= COUNT_SUM_TOLERANCE * total
    if near_one and near_total:
        raise ValueError(
            "the expected values could be probabilities or counts (the observed total is close to 1); "
            "give expected counts whose sum differs clearly from 1, or rescale the observed counts"
        )
    if near_one:
        if abs(given - 1.0) > EXACT_SUM_TOLERANCE:
            notes.append(f"the expected probabilities sum to {number_text(given)}; they were rescaled to sum to 1")
        return [value / given * total for value in values]
    if near_total:
        if abs(given - total) > EXACT_SUM_TOLERANCE * total:
            notes.append(
                f"the expected counts sum to {number_text(given)} and the observed counts to {number_text(total)}; "
                "the expected counts were used as given"
            )
        return values
    if abs(given - 100.0) <= 1.0:
        raise ValueError(
            f"the expected values sum to {number_text(given)}, which looks like percentages; divide them by 100 "
            "to give probabilities"
        )
    raise ValueError(
        f"the expected counts sum to {number_text(given)} but the observed counts sum to {number_text(total)}; "
        f"give counts summing to {number_text(total)} or probabilities summing to 1"
    )


def _goodness_of_fit_df(df: Any, categories: int) -> int:
    if df is None:
        return categories - 1
    value: int = whole_number(df, "df", 1)
    if value > categories - 1:
        raise ValueError(f"df must be at most {categories - 1} (the number of categories minus 1)")
    return value


def _parse_table(table: Any) -> List[List[float]]:
    if not isinstance(table, list) or len(table) < 2:
        raise ValueError("table must be a list of at least 2 rows of counts, e.g. [[10, 20], [30, 40]]")
    rows = [parse_counts(row, f"table[{index}]") for index, row in enumerate(table)]
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("every row of table must have the same number of counts")
    for index, row in enumerate(rows):
        if math.fsum(row) <= 0.0:
            raise ValueError(f"row {index + 1} of table has no counts; every row total must be > 0")
    for column in range(width):
        if math.fsum(row[column] for row in rows) <= 0.0:
            raise ValueError(f"column {column + 1} of table has no counts; every column total must be > 0")
    return rows


def _independence_expected(rows: List[List[float]]) -> List[List[float]]:
    total = math.fsum(math.fsum(row) for row in rows)
    row_totals = [math.fsum(row) for row in rows]
    column_totals = [math.fsum(row[column] for row in rows) for column in range(len(rows[0]))]
    return [[row_total * column_total / total for column_total in column_totals] for row_total in row_totals]


def _contributions(observed: List[float], expected: List[float]) -> List[float]:
    return [(o - e) ** 2 / e for o, e in zip(observed, expected)]


def _refuse_foreign(test_type: str, observed: Any, expected: Any, table: Any, df: Any) -> None:
    if test_type == "goodness_of_fit":
        if table is not None:
            raise ValueError(
                "table is for test_type independence; goodness_of_fit takes observed (and expected), "
                "so send table as null or use independence"
            )
        if observed is None:
            raise ValueError("goodness_of_fit needs observed, the list of observed counts per category")
        return
    given = [name for name, value in (("observed", observed), ("expected", expected), ("df", df)) if value is not None]
    if given:
        raise ValueError(
            f"{', '.join(given)} {'is' if len(given) == 1 else 'are'} for test_type goodness_of_fit; "
            "independence takes table (rows x columns of counts) only"
        )
    if table is None:
        raise ValueError("independence needs table, the contingency table as rows of counts")


def run_chi_squared_test(
    test_type: str,
    observed: Any,
    expected: Any,
    table: Any,
    df: Any,
    alpha: Any,
) -> Dict[str, Any]:
    """Run one χ² test; raises ValueError for invalid or contradictory input."""
    if test_type not in CHI_SQUARED_TEST_TYPES:
        raise ValueError(f"Unsupported test_type '{test_type}'; use one of {', '.join(CHI_SQUARED_TEST_TYPES)}")
    _refuse_foreign(test_type, observed, expected, table, df)
    level = parse_alpha(alpha)
    notes: List[str] = []
    if test_type == "goodness_of_fit":
        statistic, degrees, details = _goodness_of_fit(observed, expected, df, notes)
        method = "χ²GOF-Test: chi-squared goodness of fit"
        h0, h1 = "the data follow the expected distribution", "the data do not follow the expected distribution"
    else:
        statistic, degrees, details = _independence(table, notes)
        method = "χ²-Test: chi-squared test of independence (or homogeneity)"
        if len(details["expected"]) == 2 and len(details["expected"][0]) == 2:
            method += ", no continuity correction"
        h0, h1 = "the row and column variables are independent", "the row and column variables are associated"
    spec = get_distribution("chi_squared")
    p = spec.sf(statistic, {"df": float(degrees)})
    critical = chi_squared_upper_critical(float(degrees), level)
    require_finite(statistic, p, critical)
    result: Dict[str, Any] = {
        "test": test_type,
        "method": method,
        "null_hypothesis": h0,
        "alternative_hypothesis": h1,
        "statistic_name": "chi_squared",
        "statistic": statistic,
        "df": degrees,
    }
    result.update(decide(p, level, h1))
    result["critical_value"] = critical
    result["tail_note"] = "chi-squared tests use the upper tail only"
    result.update(details)
    if notes:
        result["notes"] = notes
    final: Dict[str, Any] = rounded(result)
    return final


def _goodness_of_fit(observed: Any, expected: Any, df: Any, notes: List[str]) -> Tuple[float, int, Dict[str, Any]]:
    counts = parse_counts(observed, "observed")
    if math.fsum(counts) <= 0.0:
        raise ValueError("observed has no counts; the total must be > 0")
    expected_counts = _goodness_of_fit_expected(counts, expected, notes)
    degrees = _goodness_of_fit_df(df, len(counts))
    contributions = _contributions(counts, expected_counts)
    _note_small_expected(expected_counts, notes)
    return math.fsum(contributions), degrees, {"expected": expected_counts, "contributions": contributions}


def _independence(table: Any, notes: List[str]) -> Tuple[float, int, Dict[str, Any]]:
    rows = _parse_table(table)
    expected_rows = _independence_expected(rows)
    contributions = [_contributions(row, expected_row) for row, expected_row in zip(rows, expected_rows)]
    _note_small_expected([value for row in expected_rows for value in row], notes)
    statistic = math.fsum(value for row in contributions for value in row)
    degrees = (len(rows) - 1) * (len(rows[0]) - 1)
    return statistic, degrees, {"expected": expected_rows, "contributions": contributions}


def _note_small_expected(expected: List[float], notes: List[str]) -> None:
    smallest = min(expected)
    if smallest < MIN_EXPECTED_COUNT:
        notes.append(
            f"some expected counts are below {MIN_EXPECTED_COUNT:g} (smallest {number_text(smallest)}); "
            "the chi-squared approximation may be poor"
        )
