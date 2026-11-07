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
   x0 trades against A) is held at a plain value and the others re-solved; among constant
   sets that fit exactly as well by symmetry (a sine's phase plus π with its amplitude
   negated, a Gaussian's width negated), the plainest is reported; notes say what the
   user should know.

Work is bounded by a count of model evaluations (EVALUATION_LIMIT, plus
CANONICAL_EVALUATIONS for the plainest report and a SHIFT_SHARE-th of the limit to refit
a model in x - shift when x lies far from 0), so a fit gives the same answer on every
machine.

Pure Python (no browser imports), so server-side pytest imports it directly.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from utils.statistics.custom_model import CustomModel, read_model
from utils.statistics.fit_starts import MAX_HALF_CYCLES, Shape, half_cycle_counts, screen, subsample, variation
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
MAX_SCREEN_POINTS = 161
SCREEN_SHARE = 0.3
FINAL_SHARE = 0.4
COARSE_POINTS = 100
REFINED_STARTS = 4
FINAL_RUNS = 2
# A run is refined on all points only when its sum of squares on the coarse points is
# within this factor of the best one's
FINALIST_RATIO = 1.5
# Fits whose sums of squares are within this fraction of the best fit the data equally well,
# and, times the sum of y squared, the rounding an exact fit's sum of squares may hold (an
# argument near 100 in sin() keeps about 1e-12 of each value)
TIE_TOLERANCE = 1e-6
EXACT_SLACK = 1e-18
# Relative step of the undetermined-constant probe, and the fraction of a constant's effect
# that must remain once the other constants have absorbed what they can
PROBE_STEP = 1e-5
DETERMINED_FRACTION = 1e-6
# Below this R² a note suggests initial values
POOR_FIT = 0.9
# x far from 0: a fit below this R² (or unconverged, or with extreme values) is refitted in
# x - shift with a SHIFT_SHARE-th of the work, and that suggested when it fits SHIFT_GAIN better
SHIFT_FIT = 0.99
SHIFT_SHARE = 4
SHIFT_GAIN = 0.9
# Model evaluations spent, beyond EVALUATION_LIMIT (and at most a tenth of it), looking for
# the plainest equivalent report (symmetries), and how many of the plainest are checked on
# every point
CANONICAL_EVALUATIONS = 15_000
CANONICAL_CONFIRMED = 3


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
    fit, projector, best = _fit_constants(xs, ys, model, given, evaluation_limit)
    offset = _offset_note(fit, model, projector, best, evaluation_limit // SHIFT_SHARE)
    fit.notes = _notes(fit, offset)
    return fit


def _fit_constants(
    xs: List[float], ys: List[float], model: CustomModel, given: Dict[int, float], evaluation_limit: int
) -> Tuple[CustomFit, Projector, FitRun]:
    """The fit without its notes, with the projector and run it came from."""
    linear = linear_constants(model, xs)
    nonlinear = [k for k in range(len(model.constants)) if k not in linear]
    screen_budget = EvaluationBudget(int(evaluation_limit * SCREEN_SHARE))
    staged = len(xs) > COARSE_POINTS
    final_budget = EvaluationBudget(int(evaluation_limit * FINAL_SHARE) if staged else 0)
    canonical_budget = EvaluationBudget(min(CANONICAL_EVALUATIONS, evaluation_limit // 10))
    budget = EvaluationBudget(evaluation_limit - screen_budget.limit - final_budget.limit)
    full = Projector(model, xs, ys, linear, nonlinear, final_budget if staged else budget)
    counts = half_cycle_counts(xs, ys)
    # Screened on a point per half cycle tried (frequencies sampled), at least SCREEN_POINTS
    # and at most MAX_SCREEN_POINTS (a frequency near the sampling limit still fits the points
    # it is screened on; refining on more points tells it from its aliases); a model with no
    # sine or cosine on at most as many as MAX_HALF_CYCLES needs
    cycles = int(math.ceil(max(counts))) if counts else 1
    if not model.trig_arguments():
        cycles = min(cycles, MAX_HALF_CYCLES)
    screen_points = max(SCREEN_POINTS, min(cycles + 1, MAX_SCREEN_POINTS))
    screener = full.on(*subsample(xs, ys, screen_points), screen_budget)
    screening = screen(screener, xs, given, counts, REFINED_STARTS)
    # At least 4 points per estimated cycle (2 per half cycle), so refining does not alias
    coarse = full.on(*subsample(xs, ys, max(COARSE_POINTS, int(1.6 * cycles))), budget) if staged else full
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
    best = _canonical(projector, shape, best, canonical_budget)
    _describe(fit, model, projector, best, alternative, shape)
    fit.evaluations = budget.used + screen_budget.used + final_budget.used + canonical_budget.used
    return fit, projector, best


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
    """The best run, and another that fits the data as well but draws a different curve
    (not one swinging between the points when the best does not).

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
    steady, _, chosen, chosen_values = curves[0]
    extent = (max(chosen_values) - min(chosen_values)) or max(abs(v) for v in chosen_values) or 1.0
    for swings, _, run, values in curves[1:]:
        # A curve swinging between the points beside one that does not is an alias, which evenly
        # spaced data always has: not worth a note
        if swings and not steady:
            break
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
            if run is None or not _ties(run.sse, best.sse):
                continue
            if not _ties(best.sse, run.sse):
                # Holding it found a much better fit: best was stuck, not undetermined; free it again
                freed = _refine_one(projector, projector.theta(run.params))
                if freed is not None and freed.sse <= run.sse:
                    best = freed
                else:
                    best, projector = run, trial
                break
            best, projector = run, trial
            fit.held[model.constants[k]] = float(value)
            break
        else:
            fit.undetermined.append(model.constants[k])
    fit.undetermined.extend(model.constants[k] for k in linear)
    fit.undetermined = [name for name in model.constants if name in fit.undetermined]
    return best, projector


def _canonical(projector: Projector, shape: Shape, best: FitRun, budget: EvaluationBudget) -> FitRun:
    """The plainest of the constant sets that fit exactly as well as best (symmetries).

    A sine fits as well with its phase or shift moved by half a period (its amplitude
    changes sign), with frequency and phase both negated, and a width as well negated.
    Variants from negating one or two nonlinear constants and from moving one by half
    periods (_Phases) are explored on a subsample while they keep tying; the plainest
    (_Phases.plainness) that also ties on every point and swings no more than best wins.
    """
    if not projector.nonlinear:
        return best
    xs, ys = subsample(projector.xs, projector.ys, COARSE_POINTS)
    sample = projector.on(xs, ys, budget)
    start = sample.project(sample.theta(best.params))
    if start is None:
        return best
    phases = _Phases(projector, start[0])
    slack = EXACT_SLACK * sum(y * y for y in ys)
    target = start[1] * (1.0 + TIE_TOLERANCE) + slack
    seen = {_key(sample.theta(best.params))}
    frontier = [start[0]]
    tied = [start[0]]
    while frontier and not sample.budget.exhausted:
        params = frontier.pop(0)
        for theta in phases.variants(sample.theta(params)):
            key = _key(theta)
            if key in seen or sample.budget.exhausted:
                continue
            seen.add(key)
            projected = sample.project(theta)
            if projected is not None and projected[1] <= target:
                frontier.append(projected[0])
                tied.append(projected[0])
    tied.sort(key=phases.plainness)
    full_target = best.sse * (1.0 + TIE_TOLERANCE) + slack * len(projector.ys)
    full = projector.on(projector.xs, projector.ys, budget)
    for params in tied[:CANONICAL_CONFIRMED]:
        if params is start[0]:
            return best
        confirmed = full.project(full.theta(params))
        if confirmed is None or confirmed[1] > full_target:
            continue
        if shape.swings(confirmed[0]) and not shape.swings(best.params):
            continue
        return FitRun(confirmed[0], [], confirmed[1], best.iterations, best.converged)
    return best


class _Phases:
    """The trigonometric arguments of a model at fitted constants: what a plain report looks like.

    For each trigonometric argument (b*x + c, B*(x - C), 2*pi*(x - C)/T): its slope in x
    (the frequency's sign), and the constants it is affine in with their effect on it: a
    constant adding g to the argument per unit repeats the curve every pi/|g| (with the
    amplitude negated), so it can be moved by that half period. A phase (c) is kept within
    (-pi, pi], a shift of x (C) within one period from 0.
    """

    def __init__(self, projector: Projector, params: List[float]) -> None:
        self.projector = projector
        self.low, self.high = min(projector.xs), max(projector.xs)
        self.arguments = projector.model.trig_arguments()
        # constant -> (half period, whether it shifts x: B*(x - C), where it moves the argument
        # by minus its slope, rather than adding a phase)
        self.half_periods: Dict[int, Tuple[float, bool]] = {}
        for argument in self.arguments:
            slope = self._slope(argument, params) / ((self.high - self.low) or 1.0)
            for k in projector.nonlinear:
                effect = self._effect(argument, params, k)
                if effect is not None and effect != 0.0 and k not in self.half_periods:
                    shifts_x = abs(effect + slope) <= 1e-6 * abs(effect)
                    self.half_periods[k] = (math.pi / abs(effect), shifts_x)

    def variants(self, theta: List[float]) -> List[List[float]]:
        """theta with one or two values negated, or one moved by half periods towards its plain range."""
        nonlinear = self.projector.nonlinear
        variants = []
        for i in range(len(theta)):
            if theta[i] != 0.0:
                variants.append(_with(theta, {i: -theta[i]}))
            half = self.half_periods.get(nonlinear[i], (math.pi, False))[0]
            turns = math.floor(theta[i] / half)
            for shift in range(turns - 1, turns + 3):
                if shift != 0:
                    variants.append(_with(theta, {i: theta[i] - shift * half}))
            for j in range(i + 1, len(theta)):
                if theta[i] != 0.0 and theta[j] != 0.0:
                    variants.append(_with(theta, {i: -theta[i], j: -theta[j]}))
        return variants

    def plainness(self, params: List[float]) -> Tuple[int, int, float, int, float]:
        """Smaller is plainer: positive frequencies, positive amplitudes, phases in (-pi, pi] and
        shifts in [0, period), positive values, small values."""
        projector = self.projector
        backwards = sum(1 for argument in self.arguments if self._slope(argument, params) < 0.0)
        negative_linear = sum(1 for k in projector.linear if params[k] < 0.0)
        beyond = 0.0
        negative = 0
        for k in projector.nonlinear:
            value = params[k]
            half, shifts_x = self.half_periods.get(k, (math.inf, False))
            low, high = (0.0, 2.0 * half) if shifts_x else (-half, half)
            beyond += max(0.0, low - value, value - high) / (half if math.isfinite(half) else 1.0)
            if value < 0.0 and k not in self.half_periods:
                negative += 1
        return backwards, negative_linear, beyond, negative, sum(abs(params[k]) for k in projector.nonlinear)

    def _slope(self, argument: Callable[..., float], params: List[float]) -> float:
        try:
            return float(argument(self.high, *params) - argument(self.low, *params))
        except Exception:
            return 0.0

    def _effect(self, argument: Callable[..., float], params: List[float], k: int) -> Optional[float]:
        """d(argument)/d(constant k) when the argument is affine in it with the same effect at every x."""
        step = 1e-3 * (abs(params[k]) + 1.0)
        effects = []
        try:
            for x in (self.low, self.high):
                base = argument(x, *params)
                for multiple in (1.0, 2.0):
                    moved = list(params)
                    moved[k] += multiple * step
                    effects.append((argument(x, *moved) - base) / (multiple * step))
        except Exception:
            return None
        first = effects[0]
        if not all(math.isfinite(effect) and abs(effect - first) <= 1e-6 * abs(first) for effect in effects):
            return None
        return first


def _with(theta: List[float], changes: Dict[int, float]) -> List[float]:
    changed = list(theta)
    for index, value in changes.items():
        changed[index] = value
    return changed


def _key(theta: Sequence[float]) -> Tuple[float, ...]:
    return tuple(round(value, 9) for value in theta)


# ---------------------------------------------------------------------------
# Figures and notes
# ---------------------------------------------------------------------------


def _describe(
    fit: CustomFit,
    model: CustomModel,
    projector: Projector,
    best: FitRun,
    alternative: Optional[FitRun],
    shape: Shape,
) -> None:
    xs, ys = projector.xs, projector.ys
    fit.values = {name: best.params[k] for k, name in enumerate(model.constants)}
    fit.predictions = model.predictions(xs, best.params) or []
    fit.residuals = [ys[i] - fit.predictions[i] for i in range(len(ys))] if fit.predictions else []
    fit.sse = best.sse
    fit.r_squared = calculate_r_squared(ys, fit.predictions) if fit.predictions else 0.0
    fit.rmse = math.sqrt(best.sse / len(ys))
    fit.converged = best.converged
    fit.swings = shape.swings(best.params)
    fit.exact = len(xs) == len(model.constants)
    exact = EXACT_SLACK * sum(y * y for y in ys)
    if alternative is not None and alternative.sse <= best.sse * (1.0 + TIE_TOLERANCE) + exact:
        fit.alternative = {name: alternative.params[k] for k, name in enumerate(model.constants)}


def _number(value: float) -> str:
    return f"{value:.6g}"


def _notes(fit: CustomFit, offset: Optional[str]) -> List[str]:
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
    if offset:
        notes.append(offset)
    elif fit.r_squared < POOR_FIT and not fit.exact:
        notes.append(
            f"R² is {_number(fit.r_squared)}: if a better fit is expected, give initial_values near the "
            f"expected constants, or check that the model suits the data."
        )
    return notes


def _offset_note(
    fit: CustomFit, model: CustomModel, projector: Projector, best: FitRun, evaluation_limit: int
) -> Optional[str]:
    """Suggest x - shift when x lies far from 0 (years) and the fit looks ill-conditioned.

    The model is refitted in x - shift with evaluation_limit, and suggested with that fit
    when it fits better, or as well with plainer values. A model the shift merely rewrites
    (exp(b*x) becomes exp(b*(x - 2000)) with another a, checked by refitting its linear
    constants with the same nonlinear ones) is refitted only when the fit did not converge
    or has extreme values (a beyond the float range sticks the rate at its edge); a model
    the shift changes (K/(1+A*exp(-k*x)), whose A changes too) also when R² is below
    SHIFT_FIT. A model linear in every constant is solved exactly and never needs it.
    """
    xs = projector.xs
    low, high = min(xs), max(xs)
    span = high - low
    centre = (low + high) / 2.0
    if span <= 0.0 or abs(centre) <= 10.0 * span or not projector.nonlinear:
        return None
    troubled = not fit.converged or _extreme(fit)
    if not troubled and fit.r_squared >= SHIFT_FIT:
        return None
    unit = 10.0 ** math.floor(math.log10(span))
    shift_text = _number(math.floor(low / unit) * unit)
    from compiled_expression import substitute_variable

    shifted = f"(x - {shift_text})"
    rewritten = substitute_variable(model.text, "x", shifted).replace(f"({shifted})", shifted)
    rewrites = _rewrites(rewritten, model, projector, best)
    if rewrites is False or (rewrites and not troubled):
        return None
    refit = _shifted_fit(rewritten, xs, projector.ys, evaluation_limit)
    if refit is None:
        return None
    fit.evaluations += refit.evaluations
    slack = EXACT_SLACK * sum(y * y for y in projector.ys)
    better = refit.sse < SHIFT_GAIN * fit.sse
    plainer = (
        troubled and refit.converged and not _extreme(refit) and refit.sse <= fit.sse * (1.0 + TIE_TOLERANCE) + slack
    )
    if not (better or plainer):
        return None
    values = ", ".join(f"{name} = {_number(value)}" for name, value in refit.values.items())
    return (
        f"x runs from {_number(low)} to {_number(high)}, far from 0, which makes this model hard to fit; written in "
        f"x - {shift_text}, as {rewritten}, it fits with R² = {_number(refit.r_squared)} ({values})."
    )


def _extreme(fit: CustomFit) -> bool:
    return any(value != 0.0 and not 1e-12 < abs(value) < 1e12 for value in fit.values.values())


def _shifted_fit(text: str, xs: List[float], ys: List[float], evaluation_limit: int) -> Optional[CustomFit]:
    """The fit of the model written in x - shift (no notes), or None when it cannot be fitted."""
    try:
        shifted = read_model(text)
        return _fit_constants(xs, ys, shifted, {}, evaluation_limit)[0]
    except ValueError:
        return None


def _rewrites(text: str, model: CustomModel, projector: Projector, best: FitRun) -> Optional[bool]:
    """Whether the model text fits the data as well as model with the same nonlinear constants.

    None when it does not (the shift changes a nonlinear constant too), False when it cannot
    be read or names other constants.
    """
    try:
        shifted = read_model(text)
    except ValueError:
        return False
    if shifted.constants != model.constants:
        return False
    check = Projector(
        shifted,
        projector.xs,
        projector.ys,
        projector.linear,
        projector.nonlinear,
        EvaluationBudget(EVALUATION_LIMIT),
        projector.fixed,
    )
    projected = check.project(projector.theta(best.params))
    slack = EXACT_SLACK * sum(y * y for y in projector.ys)
    return True if projected is not None and projected[1] <= best.sse * (1.0 + TIE_TOLERANCE) + slack else None
