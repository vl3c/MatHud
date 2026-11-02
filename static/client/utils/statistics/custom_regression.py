"""Custom-model regression: least-squares fits of a model in x with named constants.

``fit_custom_model(x_data, y_data, model, initial_values)`` finds the constants of a
``CustomModel`` (custom_model.read_model) that minimise the sum of squared residuals.

Starting values are the hard part of nonlinear fitting. The fit:

1. Splits the constants into linear ones (the model is affine in them jointly) and
   nonlinear ones, and solves the linear ones exactly for any nonlinear values
   (variable_projection), so only the nonlinear constants need starting values.
2. Screens candidate values of the nonlinear constants on a subsample (fit_starts).
3. Refines the best few candidates, and a start from any initial values, with
   Levenberg-Marquardt on at most COARSE_POINTS points, then the best FINAL_RUNS on
   every point, and keeps the lowest sum of squares. Among equally good fits, one whose
   curve does not swing between the data points wins, then the smoothest (aliasing).
4. Checks the result: a nonlinear constant the data cannot determine (A*exp(k*(x - x0)):
   x0 trades against A) is held at a plain value and the others re-solved; a negative
   constant whose negation fits as well (a Gaussian's width) is reported positive; notes
   say what the user should know.

Work is bounded by a count of model evaluations (EVALUATION_LIMIT), so a fit gives the
same answer on every machine.

Pure Python (no browser imports), so server-side pytest imports it directly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from utils.statistics.custom_model import CustomModel
from utils.statistics.fit_starts import Shape, half_cycles, screen, subsample, variation
from utils.statistics.levenberg_marquardt import EvaluationBudget, FitRun, levenberg_marquardt, undetermined_columns
from utils.statistics.regression import calculate_r_squared
from utils.statistics.variable_projection import Projector, linear_constants

MAX_DATA_POINTS = 1000
# Model evaluations (one per data point) a fit may spend: about 6 s in Brython
EVALUATION_LIMIT = 250_000
# Screening uses at least SCREEN_POINTS data points (more for oscillating data) and this
# share of the budget; every start is refined on at most COARSE_POINTS points, the best
# FINAL_RUNS of them on all points with FINAL_SHARE of the budget kept for them
SCREEN_POINTS = 40
SCREEN_SHARE = 0.3
FINAL_SHARE = 0.4
COARSE_POINTS = 100
REFINED_STARTS = 4
FINAL_RUNS = 2
# A run is refined on all points only when its sum of squares on the coarse points is
# within this factor of the best one's
FINALIST_RATIO = 1.5
# Fits whose sums of squares are within this fraction of the best fit the data equally well
TIE_TOLERANCE = 1e-6
# Relative step of the undetermined-constant probe, and the fraction of a constant's effect
# that must remain once the other constants have absorbed what they can
PROBE_STEP = 1e-5
DETERMINED_FRACTION = 1e-6
# Below this R² a note suggests initial values
POOR_FIT = 0.9


class CustomFit:
    """The fitted constants, the fit's figures and the notes for the user.

    Attributes:
        values: Constant name -> fitted value.
        held: Constants the data cannot determine, held at a plain value (name -> value).
        undetermined: Other constants the data cannot determine (left where the fit put them).
        alternative: Other constants that fit the data as well but draw a different curve.
        swings: The curve swings between the data points much more than through them.
        exact: As many data points as constants.
    """

    __slots__ = (
        "values",
        "predictions",
        "residuals",
        "sse",
        "r_squared",
        "rmse",
        "converged",
        "held",
        "undetermined",
        "alternative",
        "swings",
        "exact",
        "evaluations",
        "notes",
    )

    def __init__(self) -> None:
        self.values: Dict[str, float] = {}
        self.predictions: List[float] = []
        self.residuals: List[float] = []
        self.sse = 0.0
        self.r_squared = 0.0
        self.rmse = 0.0
        self.converged = True
        self.held: Dict[str, float] = {}
        self.undetermined: List[str] = []
        self.alternative: Optional[Dict[str, float]] = None
        self.swings = False
        self.exact = False
        self.evaluations = 0
        self.notes: List[str] = []


def fit_custom_model(
    x_data: Sequence[float],
    y_data: Sequence[float],
    model: CustomModel,
    initial_values: Optional[Mapping[str, float]] = None,
    evaluation_limit: int = EVALUATION_LIMIT,
) -> CustomFit:
    """Least-squares values of the model's constants for the data.

    Args:
        initial_values: Starting guesses for some constants (refined as well as the
            screened candidates; the best result wins).

    Raises:
        ValueError: Unusable data (lengths, non-finite values, too few or too many points),
            x^c with negative x, or no constant values tried let the model be evaluated at
            every data point.
    """
    xs, ys = _checked_data(x_data, y_data, model)
    guesses = initial_values or {}
    given = {k: float(guesses[name]) for k, name in enumerate(model.constants) if name in guesses}
    linear = linear_constants(model, xs)
    nonlinear = [k for k in range(len(model.constants)) if k not in linear]
    screen_budget = EvaluationBudget(int(evaluation_limit * SCREEN_SHARE))
    staged = len(xs) > COARSE_POINTS
    final_budget = EvaluationBudget(int(evaluation_limit * FINAL_SHARE) if staged else 0)
    budget = EvaluationBudget(evaluation_limit - screen_budget.limit - final_budget.limit)
    full = Projector(model, xs, ys, linear, nonlinear, final_budget if staged else budget)
    cycles = half_cycles(xs, ys)
    screener = full.on(*subsample(xs, ys, max(SCREEN_POINTS, cycles + 1)), screen_budget)
    screening = screen(screener, xs, given, cycles, REFINED_STARTS)
    coarse = full.on(*subsample(xs, ys, COARSE_POINTS), budget) if staged else full
    starts = list(screening.starts)
    given_start = coarse.project([given.get(k, 1.0) for k in nonlinear]) if given else None
    if given_start is not None:
        starts.insert(0, given_start[0])
    if not starts:
        raise ValueError(
            f"The model '{model.text}' could not be evaluated at every data point for any constant values "
            f"tried (a domain error such as ln or sqrt of a negative number, or a division by zero). "
            f"Check the model, or give initial_values that work."
        )

    runs = _refine(coarse, starts)
    if coarse is not full:
        runs = _refine(full, _finalists(coarse, runs)) or runs
    shape = Shape(model, xs)
    best, alternative = _choose(shape, runs)
    if shape.swings(best.params) and screening.steady is not None:
        best, alternative = _choose(shape, runs + _refine(full, [screening.steady]))

    fit = CustomFit()
    best, projector = _hold_undetermined(full, coarse, best, given, fit)
    best = _prefer_positive(projector, best)
    _describe(fit, model, xs, ys, best, alternative, shape)
    fit.evaluations = budget.used + screen_budget.used + final_budget.used
    return fit


def _checked_data(
    x_data: Sequence[float], y_data: Sequence[float], model: CustomModel
) -> Tuple[List[float], List[float]]:
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length.")
    xs, ys = [float(x) for x in x_data], [float(y) for y in y_data]
    if not all(math.isfinite(value) for value in xs + ys):
        raise ValueError("x_data and y_data must hold finite numbers.")
    count = len(model.constants)
    if len(xs) < count:
        raise ValueError(
            f"The model has {count} constants but there are only {len(xs)} data points; give at least {count} points."
        )
    if len(xs) > MAX_DATA_POINTS:
        raise ValueError(f"At most {MAX_DATA_POINTS} data points can be fitted; got {len(xs)}.")
    if model.power_constants and min(xs) < 0.0:
        names = ", ".join(sorted(model.power_constants))
        raise ValueError(
            f"The model raises x to the constant {names}, but some x are negative: x^n with negative x is defined "
            f"only for whole n. Write the power as a number (a*x^3), or shift the data."
        )
    return xs, ys


def _finalists(coarse: Projector, runs: List[FitRun]) -> List[List[float]]:
    """The best FINAL_RUNS coarse runs within FINALIST_RATIO of the best, each drawing its own curve.

    Two runs can reach the same curve with different constants (two peaks swapped); refining
    both on every point would only repeat the work.
    """
    shape = Shape(coarse.model, coarse.xs)
    chosen: List[List[float]] = []
    curves: List[List[float]] = []
    for run in runs:
        if len(chosen) >= FINAL_RUNS or run.sse > FINALIST_RATIO * runs[0].sse:
            break
        curve = shape.curve(run.params)
        coarse.budget.spend(len(shape.grid))
        if curve is None or all(_different_curves(curve, other) for other in curves):
            chosen.append(run.params)
            if curve is not None:
                curves.append(curve)
    return chosen


def _different_curves(first: List[float], second: List[float]) -> bool:
    extent = (max(second) - min(second)) or max(abs(value) for value in second) or 1.0
    return max(abs(first[i] - second[i]) for i in range(len(first))) > 1e-6 * extent


def _refine(projector: Projector, starts: List[List[float]]) -> List[FitRun]:
    """Levenberg-Marquardt from each start over the nonlinear constants; runs best first.

    The linear constants are solved exactly at every step, so the iteration never fights
    their correlation with the nonlinear ones (a*exp(b*x) at x near 2000).
    """
    runs: List[FitRun] = []
    for start in starts:
        if runs and projector.budget.exhausted:
            break
        run = _refine_one(projector, projector.theta(start))
        if run is not None:
            runs.append(run)
    runs.sort(key=lambda run: run.sse)
    return runs


def _refine_one(projector: Projector, theta: List[float]) -> Optional[FitRun]:
    if theta:
        run = levenberg_marquardt(projector.residuals, theta, projector.budget)
        if run is None:
            return None
        theta, iterations, converged = run.params, run.iterations, run.converged
    else:
        iterations, converged = 0, True
    projected = projector.project(theta)
    if projected is None:
        return None
    return FitRun(projected[0], [], projected[1], iterations, converged)


def _ties(sse: float, best: float) -> bool:
    return sse <= best * (1.0 + TIE_TOLERANCE) + 1e-300


def _choose(shape: Shape, runs: List[FitRun]) -> Tuple[FitRun, Optional[FitRun]]:
    """The best run, and another that fits the data as well but draws a different curve.

    Among runs that fit equally well, a curve that does not swing between the data points
    wins, then the smoothest (least total variation): a sine of a frequency beyond the
    sampling rate fits the same points as a slow one (aliasing), and the slow one is meant.
    """
    if not runs:
        raise ValueError("The model could not be evaluated at every data point.")
    runs = sorted(runs, key=lambda run: run.sse)
    best = runs[0]
    tied = [run for run in runs if _ties(run.sse, best.sse)]
    if len(tied) == 1:
        return best, None
    curves = []
    for run in tied:
        values = shape.curve(run.params)
        if values is not None:
            curves.append((shape.swings(run.params), variation(values), run, values))
    if not curves:
        return best, None
    curves.sort(key=lambda item: (item[0], item[1]))
    _, _, chosen, chosen_values = curves[0]
    extent = (max(chosen_values) - min(chosen_values)) or max(abs(v) for v in chosen_values) or 1.0
    for _, _, run, values in curves[1:]:
        if max(abs(values[i] - chosen_values[i]) for i in range(len(values))) > 1e-3 * extent:
            return chosen, run
    return chosen, None


# ---------------------------------------------------------------------------
# Checks of the result
# ---------------------------------------------------------------------------


def _undetermined(projector: Projector, params: List[float]) -> Tuple[List[int], List[int]]:
    """(linear, nonlinear) indices of constants the data cannot fix: others absorb their effect.

    A linear constant is undetermined when its column is a combination of the other
    linear ones. A nonlinear constant is undetermined when moving it changes the fitted
    curve (the linear constants re-solved) by less than DETERMINED_FRACTION of how it
    moves the curve with them held, beyond what the earlier nonlinear constants already
    do: a*b*x, where any b fits once a is re-solved. A constant that cannot be moved
    either way (a domain edge) is never called undetermined; one side is enough.
    """
    model = projector.model
    linear = projector.undetermined_linear(params)
    columns: List[List[float]] = []
    scales: List[float] = []
    probed: List[int] = []
    for k in projector.nonlinear:
        step = PROBE_STEP * (abs(params[k]) + 1e-3)
        sides: List[Tuple[float, List[float], List[float]]] = []
        for signed in (step, -step):
            moved = list(params)
            moved[k] += signed
            fitted = projector.residuals(projector.theta(moved))
            plain = model.predictions(projector.xs, moved)
            if fitted is not None and plain is not None:
                sides.append((signed, fitted, plain))
        if not sides:
            continue
        if len(sides) == 1:
            base = model.predictions(projector.xs, params)
            fitted_base = projector.residuals(projector.theta(params))
            if base is None or fitted_base is None:
                continue
            signed, fitted, plain = sides[0]
            sides = [(signed, fitted, plain), (0.0, fitted_base, base)]
        (h1, fitted1, plain1), (h2, fitted2, plain2) = sides
        width = h1 - h2
        columns.append([(fitted1[i] - fitted2[i]) / width for i in range(len(fitted1))])
        scales.append(math.sqrt(sum(((plain1[i] - plain2[i]) / width) ** 2 for i in range(len(plain1)))))
        probed.append(k)
    dependent = undetermined_columns(columns, DETERMINED_FRACTION, scales)
    return linear, [probed[index] for index in dependent]


def _hold_undetermined(
    full: Projector, coarse: Projector, best: FitRun, given: Mapping[int, float], fit: CustomFit
) -> Tuple[FitRun, Projector]:
    """Hold each undetermined nonlinear constant at a plain value and re-solve the others.

    Which constants are undetermined is found on the coarse points (it is a property of the
    model at these constants; the full data only costs more). The value is its initial
    value, else 0, else 1: the first at which the others still fit the data as well. Fills
    fit.held and fit.undetermined.
    """
    model = full.model
    linear, nonlinear = _undetermined(coarse, best.params)
    projector = full
    for k in nonlinear:
        for value in (given.get(k), 0.0, 1.0):
            if value is None:
                continue
            trial = projector.holding({k: float(value)})
            run = _refine_one(trial, trial.theta(best.params))
            if run is not None and _ties(run.sse, best.sse):
                best, projector = run, trial
                fit.held[model.constants[k]] = float(value)
                break
        else:
            fit.undetermined.append(model.constants[k])
    fit.undetermined.extend(model.constants[k] for k in linear)
    fit.undetermined = [name for name in model.constants if name in fit.undetermined]
    return best, projector


def _prefer_positive(projector: Projector, best: FitRun) -> FitRun:
    """Report a negative nonlinear constant positive when its negation fits as well (a width)."""
    for k in projector.nonlinear:
        if best.params[k] < 0.0:
            flipped = list(best.params)
            flipped[k] = -flipped[k]
            projected = projector.project(projector.theta(flipped))
            if projected is not None and _ties(projected[1], best.sse):
                best = FitRun(projected[0], [], projected[1], best.iterations, best.converged)
    return best


# ---------------------------------------------------------------------------
# Figures and notes
# ---------------------------------------------------------------------------


def _describe(
    fit: CustomFit,
    model: CustomModel,
    xs: List[float],
    ys: List[float],
    best: FitRun,
    alternative: Optional[FitRun],
    shape: Shape,
) -> None:
    fit.values = {name: best.params[k] for k, name in enumerate(model.constants)}
    fit.predictions = model.predictions(xs, best.params) or []
    fit.residuals = [ys[i] - fit.predictions[i] for i in range(len(ys))] if fit.predictions else []
    fit.sse = best.sse
    fit.r_squared = calculate_r_squared(ys, fit.predictions) if fit.predictions else 0.0
    fit.rmse = math.sqrt(best.sse / len(ys))
    fit.converged = best.converged
    fit.swings = shape.swings(best.params)
    fit.exact = len(xs) == len(model.constants)
    if alternative is not None:
        fit.alternative = {name: alternative.params[k] for k, name in enumerate(model.constants)}
    fit.notes = _notes(fit, model, xs)


def _number(value: float) -> str:
    return f"{value:.6g}"


def _notes(fit: CustomFit, model: CustomModel, xs: List[float]) -> List[str]:
    notes = []
    for name, value in fit.held.items():
        notes.append(
            f"The data cannot determine {name} separately from the other constants; it was held at "
            f"{_number(value)}, and any value fits equally well once they adjust."
        )
    if fit.undetermined:
        names = ", ".join(fit.undetermined)
        notes.append(
            f"The data cannot determine {names} separately from the other constants; "
            f"these values are one of many that fit equally well."
        )
    if fit.exact:
        notes.append(
            "There are as many data points as constants, so the curve passes through every point "
            "and R² = 1 says nothing about how well the model suits the data."
        )
    if fit.alternative is not None:
        listed = ", ".join(f"{name} = {_number(value)}" for name, value in fit.alternative.items())
        notes.append(
            f"Other constants fit these points equally well but draw a different curve ({listed}); "
            f"more points (per cycle) would tell them apart."
        )
    if fit.swings:
        notes.append(
            "The fitted curve swings between the data points much more than through them (too few points "
            "per cycle?); check it, or give initial_values near the expected constants."
        )
    if not fit.converged:
        notes.append(
            "The fit stopped at its work limit before converging; give initial_values near the expected constants."
        )
    offset = _offset_note(fit, model, xs)
    if offset:
        notes.append(offset)
    elif fit.r_squared < POOR_FIT and not fit.exact:
        notes.append(
            f"R² is {_number(fit.r_squared)}: if a better fit is expected, give initial_values near the "
            f"expected constants, or check that the model suits the data."
        )
    return notes


def _offset_note(fit: CustomFit, model: CustomModel, xs: List[float]) -> Optional[str]:
    """Suggest x - shift when x lies far from 0 (years) and the fit looks ill-conditioned."""
    low, high = min(xs), max(xs)
    span = high - low
    centre = (low + high) / 2.0
    if span <= 0.0 or abs(centre) <= 10.0 * span:
        return None
    extreme = any(value != 0.0 and not 1e-12 < abs(value) < 1e12 for value in fit.values.values())
    if fit.converged and fit.r_squared >= 0.99 and not extreme:
        return None
    unit = 10.0 ** math.floor(math.log10(span))
    shift = math.floor(low / unit) * unit
    shift_text = _number(shift)
    from compiled_expression import substitute_variable

    rewritten = substitute_variable(model.text, "x", f"(x - {shift_text})")
    return (
        f"x runs from {_number(low)} to {_number(high)}, far from 0, which makes this model hard to fit; "
        f"write it in x - {shift_text} instead, such as {rewritten}."
    )
