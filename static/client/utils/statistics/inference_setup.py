"""The estimate, standard error and reference distribution behind each test and interval.

``mean_setup``, ``one_proportion_setup`` and ``two_proportion_setup`` turn parsed samples into a ``Setup``: the parameter
in words, the point estimate, its standard error, the degrees of freedom (None for a z
statistic), the sample figures the result reports, and condition notes. Tests and
intervals share it; only the proportion standard errors differ between them (a test uses
the hypothesized or pooled proportion, an interval the sample proportions, as the TI-84).

Standard errors use ``math.hypot`` and Welch's df a scale-free form, so large inputs give
finite answers instead of inf or NaN. Pure Python with no ``browser`` import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.statistics.descriptive import _compute_mean
from utils.statistics.inference_samples import (
    KIND_PAIRED,
    KIND_PROPORTION,
    KIND_T,
    KIND_Z,
    Sample,
    parse_sample,
    refuse_extra_sample,
    sample_sd,
)

# Successes and failures (or n·p0 and n·(1 - p0)) below this make the normal approximation poor.
MIN_NORMAL_COUNT = 10.0


@dataclass
class Setup:
    """What a test or interval needs; ``df`` is None for a z (normal) statistic."""

    parameter: str
    method: str
    estimate: float
    standard_error: float
    df: Optional[float]
    samples: List[Dict[str, Any]]
    extras: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    fractional_df: bool = False


def mean_setup(kind: str, first: Sample, second: Optional[Sample], pooled: bool) -> Setup:
    """z, t, two_sample_z, two_sample_t (Welch or pooled) and paired_t."""
    if kind == "z":
        assert first.sigma is not None and first.mean is not None
        return Setup("mu", "one-sample z", first.mean, first.sigma / math.sqrt(first.n), None, [_mean_row(first)])
    if kind == "t":
        return _one_sample_t(first)
    if kind == "paired_t":
        assert second is not None
        return _paired_t(first, second)
    assert second is not None and first.mean is not None and second.mean is not None
    estimate = first.mean - second.mean
    rows = [_mean_row(first), _mean_row(second)]
    if kind == "two_sample_z":
        assert first.sigma is not None and second.sigma is not None
        error = math.hypot(first.sigma / math.sqrt(first.n), second.sigma / math.sqrt(second.n))
        return Setup("mu1 - mu2", "two-sample z", estimate, error, None, rows)
    return _two_sample_t(first, second, estimate, rows, pooled)


def _one_sample_t(sample: Sample) -> Setup:
    assert sample.sd is not None and sample.mean is not None
    if sample.sd == 0.0:
        raise ValueError(f"{sample.name} has no spread (sd 0), so the t statistic is undefined")
    return Setup(
        "mu", "one-sample t", sample.mean, sample.sd / math.sqrt(sample.n), float(sample.n - 1), [_mean_row(sample)]
    )


def _paired_t(first: Sample, second: Sample) -> Setup:
    assert first.values is not None and second.values is not None
    if len(first.values) != len(second.values):
        raise ValueError(
            f"paired_t needs lists of equal length; sample1.data has {len(first.values)} values "
            f"and sample2.data has {len(second.values)}"
        )
    if len(first.values) < 2:
        raise ValueError("paired_t needs at least 2 pairs")
    differences = [a - b for a, b in zip(first.values, second.values)]
    if not all(math.isfinite(d) for d in differences):
        raise ValueError("the differences are too large to compute")
    differences_sample = Sample(
        name="differences", n=len(differences), mean=_compute_mean(differences), sd=sample_sd(differences)
    )
    assert differences_sample.sd is not None and differences_sample.mean is not None
    if differences_sample.sd == 0.0:
        raise ValueError("every pair has the same difference (sd 0), so the t statistic is undefined")
    error = differences_sample.sd / math.sqrt(differences_sample.n)
    rows = [_mean_row(differences_sample)]
    return Setup(
        "mu_d", "paired t (sample1 - sample2)", differences_sample.mean, error, float(differences_sample.n - 1), rows
    )


def _two_sample_t(first: Sample, second: Sample, estimate: float, rows: List[Dict[str, Any]], pooled: bool) -> Setup:
    assert first.sd is not None and second.sd is not None
    if first.sd == 0.0 and second.sd == 0.0:
        raise ValueError("both samples have no spread (sd 0), so the t statistic is undefined")
    n1, n2 = first.n, second.n
    if pooled:
        pooled_sd = math.hypot(math.sqrt(n1 - 1.0) * first.sd, math.sqrt(n2 - 1.0) * second.sd) / math.sqrt(
            n1 + n2 - 2.0
        )
        error = pooled_sd * math.sqrt(1.0 / n1 + 1.0 / n2)
        setup = Setup("mu1 - mu2", "two-sample t, pooled variance", estimate, error, float(n1 + n2 - 2), rows)
        setup.extras["pooled_sd"] = pooled_sd
        return setup
    part1 = first.sd / math.sqrt(n1)
    part2 = second.sd / math.sqrt(n2)
    error = math.hypot(part1, part2)
    share1 = (part1 / error) ** 2
    share2 = (part2 / error) ** 2
    df = 1.0 / (share1 * share1 / (n1 - 1.0) + share2 * share2 / (n2 - 1.0))
    setup = Setup("mu1 - mu2", "two-sample t, Welch (unequal variances)", estimate, error, df, rows)
    setup.fractional_df = True
    return setup


def _mean_row(sample: Sample) -> Dict[str, Any]:
    row: Dict[str, Any] = {"name": sample.name, "n": sample.n, "mean": sample.mean}
    if sample.sigma is not None:
        row["sigma"] = sample.sigma
    if sample.sd is not None:
        row["sd"] = sample.sd
    return row


def one_proportion_setup(first: Sample, p0: Optional[float]) -> Setup:
    """p: a test (``p0`` given) uses p0 in its standard error, an interval (None) the sample p_hat."""
    p_hat = first.p_hat
    rows = [_proportion_row(first)]
    if p0 is not None:
        setup = Setup("p", "one-proportion z", p_hat, math.sqrt(p0 * (1.0 - p0) / first.n), None, rows)
        _note_expected_counts(setup, [(first.n * p0, first.n * (1.0 - p0))], "n*p0 and n*(1 - p0)")
        return setup
    setup = Setup(
        "p", "one-proportion z interval (Wald)", p_hat, math.sqrt(p_hat * (1.0 - p_hat) / first.n), None, rows
    )
    _note_observed_counts(setup, [first])
    return setup


def two_proportion_setup(first: Sample, second: Sample, for_test: bool) -> Setup:
    """p1 - p2: a test uses the pooled proportion, an interval the two sample proportions."""
    assert first.successes is not None and second.successes is not None
    p1, p2 = first.p_hat, second.p_hat
    rows = [_proportion_row(first), _proportion_row(second)]
    if for_test:
        pooled = (first.successes + second.successes) / (first.n + second.n)
        if pooled in (0.0, 1.0):
            raise ValueError("both samples have all successes or all failures, so the two-proportion test is undefined")
        error = math.sqrt(pooled * (1.0 - pooled) * (1.0 / first.n + 1.0 / second.n))
        setup = Setup("p1 - p2", "two-proportion z (pooled)", p1 - p2, error, None, rows)
        setup.extras["pooled_p_hat"] = pooled
        counts = [(s.n * pooled, s.n * (1.0 - pooled)) for s in (first, second)]
        _note_expected_counts(setup, counts, "n*p_pooled and n*(1 - p_pooled) in each sample")
        return setup
    error = math.hypot(math.sqrt(p1 * (1.0 - p1) / first.n), math.sqrt(p2 * (1.0 - p2) / second.n))
    setup = Setup("p1 - p2", "two-proportion z interval", p1 - p2, error, None, rows)
    _note_observed_counts(setup, [first, second])
    return setup


def _proportion_row(sample: Sample) -> Dict[str, Any]:
    return {"name": sample.name, "n": sample.n, "successes": sample.successes, "p_hat": sample.p_hat}


def _note_expected_counts(setup: Setup, counts: List[Tuple[float, float]], label: str) -> None:
    if any(count < MIN_NORMAL_COUNT for pair in counts for count in pair):
        setup.notes.append(
            f"{label} should be at least {MIN_NORMAL_COUNT:g}; the normal approximation may be poor here"
        )


def _note_observed_counts(setup: Setup, samples: List[Sample]) -> None:
    for sample in samples:
        assert sample.successes is not None
        failures = sample.n - sample.successes
        if sample.successes < MIN_NORMAL_COUNT or failures < MIN_NORMAL_COUNT:
            setup.notes.append(
                f"{sample.name} has {sample.successes} successes and {failures} failures; both should be at "
                f"least {MIN_NORMAL_COUNT:g}, so the normal approximation may be poor here"
            )


# What each test or interval type reads from its samples, and the hint for a stray sample2.
SAMPLE_KINDS: Dict[str, str] = {
    "z": KIND_Z,
    "t": KIND_T,
    "two_sample_z": KIND_Z,
    "two_sample_t": KIND_T,
    "paired_t": KIND_PAIRED,
    "one_proportion": KIND_PROPORTION,
    "two_proportion": KIND_PROPORTION,
}
ONE_SAMPLE_HINTS: Dict[str, str] = {
    "z": "for two samples use two_sample_z",
    "t": "for two samples use two_sample_t, or paired_t for paired values",
    "one_proportion": "for two samples use two_proportion",
}


def setup_for(
    kind: str,
    sample1: Any,
    sample2: Any,
    pooled: Any,
    p0: Optional[float],
    for_test: bool,
    ignored: List[str],
) -> Setup:
    """Parse the samples for ``kind`` and build its Setup (``p0``: a one-proportion test's p0)."""
    sample_kind = SAMPLE_KINDS.get(kind)
    if sample_kind is None:
        raise ValueError(f"Unsupported type '{kind}'; use one of {', '.join(SAMPLE_KINDS)}")
    if pooled is not None and kind != "two_sample_t":
        ignored.append("pooled")
    first = parse_sample(sample1, "sample1", sample_kind, ignored)
    second: Optional[Sample] = None
    if kind in ONE_SAMPLE_HINTS:
        refuse_extra_sample(sample2, "sample2", kind, ONE_SAMPLE_HINTS[kind])
    else:
        second = parse_sample(sample2, "sample2", sample_kind, ignored)
    if kind == "one_proportion":
        return one_proportion_setup(first, p0 if for_test else None)
    if kind == "two_proportion":
        assert second is not None
        return two_proportion_setup(first, second, for_test)
    return mean_setup(kind, first, second, bool(pooled))
