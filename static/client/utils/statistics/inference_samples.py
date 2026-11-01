"""Input parsing for the inference tools: samples, alpha, confidence levels and χ² counts.

A sample arrives as one object with the keys ``data``, ``mean``, ``sd``, ``sigma``, ``n``
and ``successes`` (every key nullable), the way a calculator takes "Data" or "Stats".
Each test kind reads its own keys and refuses the keys that belong to another kind, so a
call that picked the wrong test is refused with a hint instead of answered. Every
refusal is a ValueError whose message names what to send.

Pure Python with no ``browser`` import: the same code runs in Brython and under pytest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from utils.statistics.descriptive import _compute_mean, _spread_sums, _validate_data, _variance_and_deviation

# Sample kinds: what a test reads from each sample.
KIND_Z = "z"  # a mean with a known population sigma
KIND_T = "t"  # a mean with the sample standard deviation
KIND_PAIRED = "paired"  # a list of paired values
KIND_PROPORTION = "proportion"  # successes out of n trials

SAMPLE_KEYS = ("data", "mean", "sd", "sigma", "n", "successes")

DEFAULT_ALPHA = 0.05
DEFAULT_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class Sample:
    """One parsed sample. ``mean``/``sd``/``sigma`` are set for mean kinds, ``successes`` for proportions."""

    name: str
    n: int
    mean: Optional[float] = None
    sd: Optional[float] = None
    sigma: Optional[float] = None
    successes: Optional[int] = None
    values: Optional[List[float]] = None

    @property
    def p_hat(self) -> float:
        assert self.successes is not None
        return self.successes / self.n


def given_keys(raw: Any, name: str) -> Dict[str, Any]:
    """The non-null keys of a sample object (an absent or null sample has none)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object with data, mean, sd, sigma, n and successes, or null")
    unknown = sorted(key for key in raw if key not in SAMPLE_KEYS)
    if unknown:
        raise ValueError(f"{name} has unknown keys {', '.join(unknown)}; use {', '.join(SAMPLE_KEYS)}")
    return {key: value for key, value in raw.items() if not is_empty(value)}


def is_empty(value: Any) -> bool:
    """Null, or an empty list (small models send [] for an unused array)."""
    return value is None or (isinstance(value, list) and len(value) == 0)


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def whole_number(value: Any, label: str, minimum: int) -> int:
    number = finite_number(value, label)
    if number != math.floor(number):
        raise ValueError(f"{label} must be a whole number")
    if number < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return int(number)


def parse_sample(raw: Any, name: str, kind: str, ignored: List[str]) -> Sample:
    """Parse one sample for ``kind``; stray keys that change nothing are appended to ``ignored``."""
    fields = given_keys(raw, name)
    if not fields:
        raise ValueError(_missing_sample_message(name, kind))
    if kind == KIND_PROPORTION:
        return _proportion_sample(fields, name)
    if "successes" in fields:
        raise ValueError(
            f"{name}.successes is for proportion tests (one_proportion, two_proportion); "
            "a test of a mean takes data, or mean and n"
        )
    if kind == KIND_Z:
        return _z_sample(fields, name, ignored)
    if "sigma" in fields:
        raise ValueError(
            f"{name}.sigma is a known population standard deviation, which means a z test "
            "(z, two_sample_z); for a t test give the sample standard deviation sd, or data"
        )
    if kind == KIND_PAIRED:
        return _paired_sample(fields, name)
    return _t_sample(fields, name)


def _missing_sample_message(name: str, kind: str) -> str:
    if kind == KIND_PROPORTION:
        return f"{name} is needed: give successes and n"
    if kind == KIND_PAIRED:
        return f"{name} is needed: give its data list"
    if kind == KIND_Z:
        return f"{name} is needed: give data or mean and n, with sigma"
    return f"{name} is needed: give data, or mean, sd and n"


def _proportion_sample(fields: Dict[str, Any], name: str) -> Sample:
    foreign = [key for key in ("data", "mean", "sd", "sigma") if key in fields]
    if foreign:
        raise ValueError(
            f"a proportion test takes {name}.successes and {name}.n (counts); "
            f"{', '.join(foreign)} belong to tests of a mean (z, t)"
        )
    if "successes" not in fields or "n" not in fields:
        raise ValueError(
            f"{name} needs successes and n (counts); for a percentage only, successes = p_hat * n "
            "when that is a whole number"
        )
    n = whole_number(fields["n"], f"{name}.n", 1)
    raw_successes = finite_number(fields["successes"], f"{name}.successes")
    if 0.0 < raw_successes < 1.0:
        raise ValueError(
            f"{name}.successes is a count, not a proportion: for p-hat {raw_successes:g} of n = {n} send "
            f"{raw_successes * n:g} if that is a whole number, otherwise ask for the count"
        )
    successes = whole_number(raw_successes, f"{name}.successes", 0)
    if successes > n:
        raise ValueError(f"{name}.successes ({successes}) cannot exceed {name}.n ({n})")
    return Sample(name=name, n=n, successes=successes)


def _z_sample(fields: Dict[str, Any], name: str, ignored: List[str]) -> Sample:
    if "sigma" not in fields:
        if "sd" in fields:
            raise ValueError(
                f"a z test needs the known population standard deviation as {name}.sigma, and {name} gives sd: "
                f"if that value is the population's (known) standard deviation, send it as {name}.sigma; "
                "if it is the sample's standard deviation, use t (or two_sample_t)"
            )
        raise ValueError(f"a z test needs {name}.sigma, the known population standard deviation")
    sigma = finite_number(fields["sigma"], f"{name}.sigma")
    if sigma <= 0.0:
        raise ValueError(f"{name}.sigma must be > 0")
    if "sd" in fields:
        ignored.append(f"{name}.sd")
    if "data" in fields:
        values = _data_values(fields, name, check_sd=False)
        return Sample(name=name, n=len(values), mean=_compute_mean(values), sigma=sigma, values=values)
    n, mean = _summary_mean(fields, name, minimum_n=1)
    return Sample(name=name, n=n, mean=mean, sigma=sigma)


def _t_sample(fields: Dict[str, Any], name: str) -> Sample:
    if "data" in fields:
        values = _data_values(fields, name, check_sd=True)
        if len(values) < 2:
            raise ValueError(f"{name}.data needs at least 2 values for a t test")
        return Sample(name=name, n=len(values), mean=_compute_mean(values), sd=sample_sd(values), values=values)
    n, mean = _summary_mean(fields, name, minimum_n=2)
    if "sd" not in fields:
        raise ValueError(f"{name}.sd is needed: the sample standard deviation s (Sx)")
    sd = finite_number(fields["sd"], f"{name}.sd")
    if sd < 0.0:
        raise ValueError(f"{name}.sd must be >= 0")
    return Sample(name=name, n=n, mean=mean, sd=sd)


def _paired_sample(fields: Dict[str, Any], name: str) -> Sample:
    if "data" not in fields:
        raise ValueError(
            "paired_t needs both data lists (sample1.data and sample2.data); "
            "for summary statistics of the differences use test_type t on one sample"
        )
    values = _data_values(fields, name, check_sd=True)
    return Sample(name=name, n=len(values), values=values)


def _data_values(fields: Dict[str, Any], name: str, *, check_sd: bool) -> List[float]:
    conflicting = [key for key in ("mean", "sd") if key in fields and (check_sd or key == "mean")]
    if conflicting:
        raise ValueError(
            f"give {name}.data or the summary ({' and '.join(conflicting)}), not both; send {name}.data alone"
        )
    data = fields["data"]
    if not isinstance(data, list):
        raise ValueError(f"{name}.data must be a list of numbers")
    try:
        _validate_data(data, f"{name}.data")
    except TypeError as error:
        raise ValueError(str(error)) from None
    values = [float(value) for value in data]
    if "n" in fields:
        n = whole_number(fields["n"], f"{name}.n", 1)
        if n != len(values):
            raise ValueError(f"{name}.n is {n} but {name}.data has {len(values)} values; send n as null")
    return values


def _summary_mean(fields: Dict[str, Any], name: str, minimum_n: int) -> Tuple[int, float]:
    missing = [key for key in ("mean", "n") if key not in fields]
    if missing:
        raise ValueError(f"{name} needs data, or {' and '.join(missing)} with the other summary values")
    return whole_number(fields["n"], f"{name}.n", minimum_n), finite_number(fields["mean"], f"{name}.mean")


def sample_sd(values: List[float]) -> float:
    """The sample standard deviation (n - 1), the same routine as compute_descriptive_statistics."""
    scale, scaled = _spread_sums(values, _compute_mean(values), min(values) == max(values))
    return float(_variance_and_deviation(scale, scaled, len(values) - 1)[1])


def refuse_extra_sample(raw: Any, name: str, test_type: str, hint: str) -> None:
    """A sample a one-sample test does not read: refuse it, naming the test that would."""
    if given_keys(raw, name):
        raise ValueError(f"{name} is given but test_type {test_type} uses one sample; {hint}, or send {name} as null")


def parse_alpha(value: Any) -> float:
    """The significance level: null = 0.05; strictly between 0 and 0.5."""
    if value is None:
        return DEFAULT_ALPHA
    alpha = finite_number(value, "alpha")
    if alpha > 1.0:
        raise ValueError("alpha is a probability: write 0.05 for a 5% level")
    if alpha >= 0.5:
        raise ValueError(
            "alpha is the significance level, e.g. 0.05; a value like 0.95 is a confidence level (alpha = 1 - 0.95)"
        )
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0, e.g. 0.05")
    return alpha


def parse_confidence_level(value: Any) -> float:
    """The confidence level: null = 0.95; in (0, 1), or a percentage in (1, 100) such as 95."""
    if value is None:
        return DEFAULT_CONFIDENCE_LEVEL
    level = finite_number(value, "confidence_level")
    if 1.0 < level < 100.0:
        level /= 100.0
    elif not 0.0 < level < 1.0:
        raise ValueError(
            "confidence_level must be between 0 and 1 (e.g. 0.95) or a percentage between 1 and 100 (e.g. 95), "
            "exclusive"
        )
    if level < 0.5:
        raise ValueError(
            f"confidence_level {value:g} reads as {level:.4g}, below 50%; confidence levels are usually 0.90, 0.95 "
            "or 0.99 (a significance level alpha = 0.05 is the level 0.95)"
        )
    return level


def parse_counts(value: Any, label: str) -> List[float]:
    """A list of at least two non-negative finite counts."""
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of counts")
    if len(value) < 2:
        raise ValueError(f"{label} needs at least 2 categories")
    counts = [finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)]
    for index, count in enumerate(counts):
        if count < 0.0:
            raise ValueError(f"{label}[{index}] must be >= 0")
    return counts
