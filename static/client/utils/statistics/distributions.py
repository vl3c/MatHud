"""Probability distributions: the distribution table and its lookup.

``DISTRIBUTIONS`` maps each ``distribution_type`` the tools accept to a ``DistributionSpec``:
how to parse its parameters, its support, density (pmf for a discrete one), cdf, upper tail,
quantile, mean and variance, its density as a MatHud expression, and its default plot
bounds. ``get_distribution`` looks a type up. The functions themselves live in
``utils.statistics.distribution_functions``.

Every distribution reads its own keys from one flat parameter object shared by all of them
(``SHARED_PARAM_NAMES``, the ``distribution_params`` of the tool schemas). A missing key
counts as null; a null one takes its default or, for a needed one, is an error naming it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from utils.statistics import distribution_functions as fn
from utils.statistics.distribution_functions import (  # noqa: F401  (re-exported)
    _format_plain_number,
    _require_finite,
    default_normal_bounds,
    normal_pdf,
    normal_pdf_expression,
)

DEFAULT_DISTRIBUTION_TYPE = "normal"

# Every key of the shared distribution_params object, in the schema's order.
SHARED_PARAM_NAMES: Tuple[str, ...] = ("mean", "sigma", "lower", "upper", "rate", "df", "df2", "n", "p")

# Largest n (binomial), rate (poisson) and degrees of freedom accepted.
MAX_SIZE_PARAM = 1e6

Params = Mapping[str, float]


@dataclass(frozen=True)
class ParamRule:
    """One parameter: its key, its default (None = needed), how errors name it, and its check."""

    key: str
    default: Optional[float]
    label: str
    check: Callable[[float], Optional[str]]


@dataclass(frozen=True)
class DistributionSpec:
    """One distribution: its parameters and functions (see the module docstring).

    ``parse_params(raw, strict=False)`` turns a raw ``distribution_params`` value (a dict, or
    anything else for none) into the canonical parameters, in a fixed key order, and raises
    ValueError for invalid ones. A missing or null parameter takes its default; with
    ``strict=True`` a parameter that is present but null is an error instead. Keys of other
    distributions are ignored (``ignored_params`` names the non-null ones).
    """

    name: str
    discrete: bool
    params: Tuple[ParamRule, ...]
    support: Callable[[Params], Tuple[float, float]]
    pdf: Callable[[float, Params], float]
    cdf: Callable[[float, Params], float]
    sf: Callable[[float, Params], float]
    quantile: Callable[[float, Params], float]
    moments: Callable[[Params], Tuple[Optional[float], Optional[float]]]
    pdf_expression: Optional[Callable[[Params], str]]
    default_bounds: Callable[[Params], Tuple[float, float]]
    extra_check: Optional[Callable[[Params], Optional[str]]] = None
    mode: Optional[Callable[[Params], float]] = None  # discrete rows: the most likely value

    @property
    def param_names(self) -> Tuple[str, ...]:
        return tuple(rule.key for rule in self.params)

    def parse_params(self, raw: Any, strict: bool = False) -> Dict[str, float]:
        return _parse_params(self, raw, strict)

    def ignored_params(self, raw: Any) -> List[str]:
        """The non-null keys of ``raw`` this distribution does not read, in the shared order."""
        if not isinstance(raw, dict):
            return []
        own = set(self.param_names)
        known = [key for key in SHARED_PARAM_NAMES if key in raw]
        others = sorted(key for key in raw if key not in SHARED_PARAM_NAMES)
        return [key for key in known + others if key not in own and raw[key] is not None]


def _parse_params(spec: DistributionSpec, raw: Any, strict: bool) -> Dict[str, float]:
    params = raw if isinstance(raw, dict) else {}
    if strict:
        for rule in spec.params:
            if rule.key in params and params[rule.key] is None:
                raise ValueError(f"{rule.key} must be a number")
    missing = [rule for rule in spec.params if rule.default is None and params.get(rule.key) is None]
    if missing:
        raise ValueError(f"{spec.name} needs " + " and ".join(rule.label for rule in missing))
    parsed: Dict[str, float] = {}
    for rule in spec.params:
        value = params.get(rule.key)
        number = rule.default if value is None else float(value)
        assert number is not None
        if not math.isfinite(number):
            raise ValueError(f"{rule.key} must be finite")
        problem = rule.check(number)
        if problem:
            raise ValueError(f"{rule.key} must be {problem}")
        parsed[rule.key] = number
    if spec.extra_check is not None:
        problem = spec.extra_check(parsed)
        if problem:
            raise ValueError(problem)
    return parsed


def _positive(value: float) -> Optional[str]:
    return None if value > 0.0 else "> 0"


def _any(value: float) -> Optional[str]:
    return None


def _size(value: float) -> Optional[str]:
    return None if 0.0 < value <= MAX_SIZE_PARAM else f"> 0 and at most {MAX_SIZE_PARAM:g}"


def _probability(value: float) -> Optional[str]:
    return None if 0.0 < value < 1.0 else "between 0 and 1 (exclusive)"


def _trials(value: float) -> Optional[str]:
    if value != math.floor(value):
        return "a whole number"
    return None if 1.0 <= value <= MAX_SIZE_PARAM else f"between 1 and {MAX_SIZE_PARAM:g}"


def _uniform_order(params: Params) -> Optional[str]:
    return None if params["lower"] < params["upper"] else "lower must be less than upper"


def _sd(variance: Optional[float]) -> float:
    return math.sqrt(variance) if variance is not None and math.isfinite(variance) else 1.0


def _quantile_of(
    cdf: Callable[[float, Params], float],
    sf: Callable[[float, Params], float],
    support: Callable[[Params], Tuple[float, float]],
    center: Callable[[Params], float],
    scale: Callable[[Params], float],
) -> Callable[[float, Params], float]:
    def quantile(prob: float, params: Params) -> float:
        return 0.0 + float(  # 0.0 + turns -0.0 into 0.0
            fn.continuous_quantile(
                prob,
                lambda x: cdf(x, params),
                lambda x: sf(x, params),
                support(params),
                center(params),
                scale(params),
            )
        )

    return quantile


def _discrete_quantile_of(
    cdf: Callable[[float, Params], float],
    sf: Callable[[float, Params], float],
    support: Callable[[Params], Tuple[float, float]],
    moments: Callable[[Params], Tuple[Optional[float], Optional[float]]],
) -> Callable[[float, Params], float]:
    def quantile(prob: float, params: Params) -> float:
        lo, hi = support(params)
        mean, variance = moments(params)
        return float(
            fn.discrete_quantile(
                prob,
                lambda k: cdf(k, params),
                lambda k: sf(k, params),
                int(lo),
                int(hi) if math.isfinite(hi) else None,
                mean if mean is not None else lo,
                _sd(variance),
            )
        )

    return quantile


def _integer_bounds(spec_moments: Callable[[Params], Tuple[Optional[float], Optional[float]]]) -> Callable:
    """Default bars: the integers within mean ± 5 sd (the caller clips them to the support)."""

    def bounds(params: Params) -> Tuple[float, float]:
        mean, variance = spec_moments(params)
        assert mean is not None
        sd = _sd(variance)
        return float(math.floor(mean - 5.0 * sd)), float(math.ceil(mean + 5.0 * sd))

    return bounds


_ZERO_TO_INF = (0.0, math.inf)
_REAL_LINE = (-math.inf, math.inf)

_normal_quantile = _quantile_of(
    fn.normal_cdf, fn.normal_sf, lambda p: _REAL_LINE, lambda p: p["mean"], lambda p: p["sigma"]
)
_t_quantile = _quantile_of(fn.t_cdf, fn.t_sf, lambda p: _REAL_LINE, lambda p: 0.0, lambda p: 1.0)
_chi_squared_quantile = _quantile_of(
    fn.chi_squared_cdf, fn.chi_squared_sf, lambda p: _ZERO_TO_INF, lambda p: p["df"], lambda p: _sd(2.0 * p["df"])
)
_poisson_quantile = _discrete_quantile_of(
    fn.poisson_cdf, fn.poisson_sf, lambda params: (0.0, math.inf), fn.poisson_moments
)
_geometric_quantile = _discrete_quantile_of(
    fn.geometric_cdf, fn.geometric_sf, lambda params: (1.0, math.inf), fn.geometric_moments
)
_f_quantile = _quantile_of(fn.f_cdf, fn.f_sf, lambda p: _ZERO_TO_INF, lambda p: 1.0, lambda p: 1.0)


def _t_bounds(params: Params) -> Tuple[float, float]:
    half = min(10.0, max(4.0, _t_quantile(0.999, params)))
    return -half, half


def _poisson_bounds(params: Params) -> Tuple[float, float]:
    rate = params["rate"]
    upper = max(math.ceil(rate + 5.0 * math.sqrt(rate)), _poisson_quantile(0.999, params))
    return float(math.floor(rate - 5.0 * math.sqrt(rate))), float(upper)


def _binomial_support(params: Params) -> Tuple[float, float]:
    return 0.0, params["n"]


_NORMAL = DistributionSpec(
    name="normal",
    discrete=False,
    params=(
        ParamRule("mean", 0.0, "mean", _any),
        ParamRule("sigma", 1.0, "sigma", _positive),
    ),
    support=lambda params: _REAL_LINE,
    pdf=lambda x, params: normal_pdf(x, params["mean"], params["sigma"]),
    cdf=fn.normal_cdf,
    sf=fn.normal_sf,
    quantile=_normal_quantile,
    moments=fn.normal_moments,
    pdf_expression=lambda params: normal_pdf_expression(params["mean"], params["sigma"]),
    default_bounds=lambda params: default_normal_bounds(params["mean"], params["sigma"], k=4.0),
)

_UNIFORM = DistributionSpec(
    name="uniform",
    discrete=False,
    params=(
        ParamRule("lower", 0.0, "lower", _any),
        ParamRule("upper", 1.0, "upper", _any),
    ),
    support=lambda params: (params["lower"], params["upper"]),
    pdf=fn.uniform_pdf,
    cdf=fn.uniform_cdf,
    sf=fn.uniform_sf,
    quantile=fn.uniform_quantile,
    moments=fn.uniform_moments,
    pdf_expression=fn.uniform_pdf_expression,
    default_bounds=lambda params: (params["lower"], params["upper"]),
    extra_check=_uniform_order,
)

_EXPONENTIAL = DistributionSpec(
    name="exponential",
    discrete=False,
    params=(ParamRule("rate", 1.0, "rate (λ)", _positive),),
    support=lambda params: _ZERO_TO_INF,
    pdf=fn.exponential_pdf,
    cdf=fn.exponential_cdf,
    sf=fn.exponential_sf,
    quantile=fn.exponential_quantile,
    moments=fn.exponential_moments,
    pdf_expression=fn.exponential_pdf_expression,
    default_bounds=lambda params: (0.0, fn.exponential_quantile(0.999, params)),
)

_T = DistributionSpec(
    name="t",
    discrete=False,
    params=(ParamRule("df", None, "df (degrees of freedom)", _size),),
    support=lambda params: _REAL_LINE,
    pdf=fn.t_pdf,
    cdf=fn.t_cdf,
    sf=fn.t_sf,
    quantile=_t_quantile,
    moments=fn.t_moments,
    pdf_expression=fn.t_pdf_expression,
    default_bounds=_t_bounds,
)

_CHI_SQUARED = DistributionSpec(
    name="chi_squared",
    discrete=False,
    params=(ParamRule("df", None, "df (degrees of freedom)", _size),),
    support=lambda params: _ZERO_TO_INF,
    pdf=fn.chi_squared_pdf,
    cdf=fn.chi_squared_cdf,
    sf=fn.chi_squared_sf,
    quantile=_chi_squared_quantile,
    moments=fn.chi_squared_moments,
    pdf_expression=fn.chi_squared_pdf_expression,
    default_bounds=lambda params: (
        max(0.0, params["df"] - 6.0 * math.sqrt(2.0 * params["df"])),
        _chi_squared_quantile(0.999, params),
    ),
)

_F = DistributionSpec(
    name="f",
    discrete=False,
    params=(
        ParamRule("df", None, "df (numerator degrees of freedom)", _size),
        ParamRule("df2", None, "df2 (denominator degrees of freedom)", _size),
    ),
    support=lambda params: _ZERO_TO_INF,
    pdf=fn.f_pdf,
    cdf=fn.f_cdf,
    sf=fn.f_sf,
    quantile=_f_quantile,
    moments=fn.f_moments,
    pdf_expression=fn.f_pdf_expression,
    default_bounds=lambda params: (0.0, min(10.0, max(3.0, _f_quantile(0.99, params)))),
)

_BINOMIAL = DistributionSpec(
    name="binomial",
    discrete=True,
    params=(
        ParamRule("n", None, "n (number of trials)", _trials),
        ParamRule("p", None, "p (success probability)", _probability),
    ),
    support=_binomial_support,
    pdf=fn.binomial_pmf,
    cdf=fn.binomial_cdf,
    sf=fn.binomial_sf,
    quantile=_discrete_quantile_of(fn.binomial_cdf, fn.binomial_sf, _binomial_support, fn.binomial_moments),
    moments=fn.binomial_moments,
    pdf_expression=None,
    default_bounds=_integer_bounds(fn.binomial_moments),
    mode=lambda params: float(min(params["n"], math.floor((params["n"] + 1.0) * params["p"]))),
)

_POISSON = DistributionSpec(
    name="poisson",
    discrete=True,
    params=(ParamRule("rate", None, "rate (λ)", _size),),
    support=lambda params: (0.0, math.inf),
    pdf=fn.poisson_pmf,
    cdf=fn.poisson_cdf,
    sf=fn.poisson_sf,
    quantile=_poisson_quantile,
    moments=fn.poisson_moments,
    pdf_expression=None,
    default_bounds=_poisson_bounds,
    mode=lambda params: float(math.floor(params["rate"])),
)

_GEOMETRIC = DistributionSpec(
    name="geometric",
    discrete=True,
    params=(ParamRule("p", None, "p (success probability)", _probability),),
    support=lambda params: (1.0, math.inf),
    pdf=fn.geometric_pmf,
    cdf=fn.geometric_cdf,
    sf=fn.geometric_sf,
    quantile=_geometric_quantile,
    moments=fn.geometric_moments,
    pdf_expression=None,
    default_bounds=lambda params: (1.0, _geometric_quantile(0.999, params)),
    mode=lambda params: 1.0,
)

DISTRIBUTIONS: Dict[str, DistributionSpec] = {
    spec.name: spec for spec in (_NORMAL, _UNIFORM, _EXPONENTIAL, _T, _CHI_SQUARED, _F, _BINOMIAL, _POISSON, _GEOMETRIC)
}


def get_distribution(distribution_type: Any, *, default: Optional[str] = None) -> DistributionSpec:
    """The spec for a distribution type, matched without case or surrounding spaces.

    An empty or null type takes ``default`` when one is given. An unknown type, or an empty
    one with no default, raises ValueError naming the type as given and the valid ones.
    """
    key = str(distribution_type or "").strip().lower()
    if not key and default is not None:
        key = default
    spec = DISTRIBUTIONS.get(key)
    if spec is None:
        raise ValueError(f"Unsupported distribution_type '{distribution_type}'; use one of {', '.join(DISTRIBUTIONS)}")
    return spec
