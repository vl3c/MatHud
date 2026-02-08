"""
Regression Analysis Module

Provides fitting algorithms for multiple regression model types:
- Linear: y = mx + b
- Polynomial: y = a0 + a1*x + a2*x^2 + ... + an*x^n
- Exponential: y = a * e^(bx)
- Logarithmic: y = a + b * ln(x)
- Power: y = a * x^b
- Logistic: y = L / (1 + e^(-k(x - x0)))
- Sinusoidal: y = a * sin(bx + c) + d

Each fitting function returns a RegressionResult with the fitted expression,
coefficients, R-squared value, and model type.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, TypedDict


class RegressionResult(TypedDict):
    """Result of a regression fit."""
    expression: str
    coefficients: Dict[str, float]
    r_squared: float
    model_type: str


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_data(x_data: List[float], y_data: List[float], min_points: int = 2) -> None:
    """Validate input data arrays."""
    if not isinstance(x_data, list) or not isinstance(y_data, list):
        raise TypeError("x_data and y_data must be lists")
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    if len(x_data) < min_points:
        raise ValueError(f"Need at least {min_points} data points for regression")
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError(f"Data point {i} contains non-numeric value")
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Data point {i} contains non-finite value")


def _validate_positive(values: List[float], name: str) -> None:
    """Validate that all values are positive."""
    for i, v in enumerate(values):
        if v <= 0:
            raise ValueError(f"{name}[{i}] = {v} must be positive")


# ---------------------------------------------------------------------------
# Pure Python matrix operations (no numpy dependency)
# ---------------------------------------------------------------------------

def _matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Multiply two matrices."""
    rows_A = len(A)
    cols_A = len(A[0]) if A else 0
    rows_B = len(B)
    cols_B = len(B[0]) if B else 0

    if cols_A != rows_B:
        raise ValueError("Matrix dimensions incompatible for multiplication")

    result: List[List[float]] = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result


def _matrix_transpose(A: List[List[float]]) -> List[List[float]]:
    """Transpose a matrix."""
    if not A:
        return []
    rows = len(A)
    cols = len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def _matrix_inverse(A: List[List[float]]) -> List[List[float]]:
    """Compute matrix inverse using Gauss-Jordan elimination."""
    n = len(A)
    if n == 0 or len(A[0]) != n:
        raise ValueError("Matrix must be square")

    # Create augmented matrix [A | I]
    aug: List[List[float]] = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row

        if max_val < 1e-12:
            raise ValueError("Matrix is singular or nearly singular")

        # Swap rows
        aug[col], aug[max_row] = aug[max_row], aug[col]

        # Eliminate column
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

    # Extract inverse
    return [row[n:] for row in aug]


def _solve_least_squares(X: List[List[float]], y: List[float]) -> List[float]:
    """
    Solve least squares problem: find coefficients c such that X @ c ≈ y.
    Uses normal equations: c = (X^T X)^(-1) X^T y
    """
    Xt = _matrix_transpose(X)
    XtX = _matrix_multiply(Xt, X)
    XtX_inv = _matrix_inverse(XtX)

    # Convert y to column vector
    y_col = [[yi] for yi in y]
    Xty = _matrix_multiply(Xt, y_col)

    # Compute coefficients
    c_col = _matrix_multiply(XtX_inv, Xty)
    return [c[0] for c in c_col]


# ---------------------------------------------------------------------------
# R-squared calculation
# ---------------------------------------------------------------------------

def calculate_r_squared(y_actual: List[float], y_predicted: List[float]) -> float:
    """
    Calculate coefficient of determination (R²).

    R² = 1 - SS_res / SS_tot
    where SS_res = sum((y_actual - y_predicted)^2)
          SS_tot = sum((y_actual - mean(y_actual))^2)
    """
    if len(y_actual) != len(y_predicted):
        raise ValueError("y_actual and y_predicted must have same length")
    if len(y_actual) == 0:
        raise ValueError("Cannot calculate R² for empty data")

    y_mean = sum(y_actual) / len(y_actual)
    ss_tot = sum((y - y_mean) ** 2 for y in y_actual)
    ss_res = sum((ya - yp) ** 2 for ya, yp in zip(y_actual, y_predicted))

    if ss_tot < 1e-12:
        # All y values are essentially the same
        return 1.0 if ss_res < 1e-12 else 0.0

    r_squared = 1.0 - (ss_res / ss_tot)
    # Clamp to [0, 1] to handle numerical errors
    return max(0.0, min(1.0, r_squared))


# ---------------------------------------------------------------------------
# Expression building
# ---------------------------------------------------------------------------

def _format_coefficient(value: float, precision: int = 6) -> str:
    """Format a coefficient for expression string.

    Avoids scientific notation since MatHud parser interprets 'e' as Euler's number.
    """
    if abs(value) < 1e-10:
        return "0"
    # Use fixed-point notation to avoid scientific notation (e.g., 1e-05)
    # which MatHud would interpret as 1 * euler_number - 05
    formatted = f"{value:.{precision}f}".rstrip('0').rstrip('.')
    # Ensure we don't return empty string or just a minus sign
    if not formatted or formatted == '-':
        return "0"
    return formatted


def build_expression(model_type: str, coefficients: Dict[str, float]) -> str:
    """
    Build a MatHud-compatible expression string from model type and coefficients.
    Uses '^' for exponentiation as per MatHud conventions.
    """
    if model_type == "linear":
        m = coefficients.get("m", 0.0)
        b = coefficients.get("b", 0.0)
        m_str = _format_coefficient(m)
        b_str = _format_coefficient(abs(b))
        sign = "+" if b >= 0 else "-"
        return f"({m_str})*x {sign} {b_str}"

    elif model_type == "polynomial":
        # Coefficients are a0, a1, a2, ... for a0 + a1*x + a2*x^2 + ...
        terms = []
        degree = 0
        while f"a{degree}" in coefficients:
            degree += 1

        for i in range(degree):
            coef = coefficients.get(f"a{i}", 0.0)
            if abs(coef) < 1e-10:
                continue
            coef_str = _format_coefficient(coef)
            if i == 0:
                terms.append(coef_str)
            elif i == 1:
                terms.append(f"({coef_str})*x")
            else:
                terms.append(f"({coef_str})*x^{i}")

        if not terms:
            return "0"
        return " + ".join(terms).replace("+ -", "- ")

    elif model_type == "exponential":
        a = coefficients.get("a", 1.0)
        b = coefficients.get("b", 0.0)
        a_str = _format_coefficient(a)
        b_str = _format_coefficient(b)
        return f"({a_str})*exp(({b_str})*x)"

    elif model_type == "logarithmic":
        a = coefficients.get("a", 0.0)
        b = coefficients.get("b", 0.0)
        a_str = _format_coefficient(a)
        b_str = _format_coefficient(abs(b))
        sign = "+" if b >= 0 else "-"
        return f"{a_str} {sign} ({b_str})*ln(x)"

    elif model_type == "power":
        a = coefficients.get("a", 1.0)
        b = coefficients.get("b", 0.0)
        a_str = _format_coefficient(a)
        b_str = _format_coefficient(b)
        return f"({a_str})*x^({b_str})"

    elif model_type == "logistic":
        L = coefficients.get("L", 1.0)
        k = coefficients.get("k", 1.0)
        x0 = coefficients.get("x0", 0.0)
        L_str = _format_coefficient(L)
        k_str = _format_coefficient(k)
        x0_str = _format_coefficient(abs(x0))
        x0_sign = "-" if x0 >= 0 else "+"
        return f"({L_str})/(1 + exp(-({k_str})*(x {x0_sign} {x0_str})))"

    elif model_type == "sinusoidal":
        a = coefficients.get("a", 1.0)
        b = coefficients.get("b", 1.0)
        c = coefficients.get("c", 0.0)
        d = coefficients.get("d", 0.0)
        a_str = _format_coefficient(a)
        b_str = _format_coefficient(b)
        c_str = _format_coefficient(abs(c))
        d_str = _format_coefficient(abs(d))
        c_sign = "+" if c >= 0 else "-"
        d_sign = "+" if d >= 0 else "-"
        return f"({a_str})*sin(({b_str})*x {c_sign} {c_str}) {d_sign} {d_str}"

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Model fitting functions
# ---------------------------------------------------------------------------

def fit_linear(x_data: List[float], y_data: List[float]) -> RegressionResult:
    """
    Fit linear model: y = mx + b

    Returns coefficients {m, b} and R-squared.
    """
    _validate_data(x_data, y_data)

    n = len(x_data)
    sum_x = sum(x_data)
    sum_y = sum(y_data)
    sum_xy = sum(x * y for x, y in zip(x_data, y_data))
    sum_xx = sum(x * x for x in x_data)

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        raise ValueError("Cannot fit linear model: x values have no variance")

    m = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - m * sum_x) / n

    # Calculate R-squared
    y_predicted = [m * x + b for x in x_data]
    r_squared = calculate_r_squared(y_data, y_predicted)

    coefficients = {"m": m, "b": b}
    expression = build_expression("linear", coefficients)

    return {
        "expression": expression,
        "coefficients": coefficients,
        "r_squared": r_squared,
        "model_type": "linear",
    }


def fit_polynomial(x_data: List[float], y_data: List[float], degree: int) -> RegressionResult:
    """
    Fit polynomial model: y = a0 + a1*x + a2*x^2 + ... + an*x^n

    Returns coefficients {a0, a1, ..., an} and R-squared.
    """
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("Polynomial degree must be a positive integer")

    min_points = degree + 1
    _validate_data(x_data, y_data, min_points=min_points)

    len(x_data)

    # Build Vandermonde matrix
    X: List[List[float]] = []
    for x in x_data:
        row = [x ** j for j in range(degree + 1)]
        X.append(row)

    # Solve least squares
    coeffs = _solve_least_squares(X, y_data)

    # Calculate R-squared
    y_predicted = []
    for x in x_data:
        y_pred = sum(coeffs[j] * (x ** j) for j in range(degree + 1))
        y_predicted.append(y_pred)
    r_squared = calculate_r_squared(y_data, y_predicted)

    coefficients = {f"a{i}": coeffs[i] for i in range(degree + 1)}
    expression = build_expression("polynomial", coefficients)

    return {
        "expression": expression,
        "coefficients": coefficients,
        "r_squared": r_squared,
        "model_type": "polynomial",
    }


def fit_exponential(x_data: List[float], y_data: List[float]) -> RegressionResult:
    """
    Fit exponential model: y = a * e^(bx)

    Linearized form: ln(y) = ln(a) + bx
    Requires all y values to be positive.

    Returns coefficients {a, b} and R-squared.
    """
    _validate_data(x_data, y_data)
    _validate_positive(y_data, "y_data")

    # Linearize: ln(y) = ln(a) + b*x
    ln_y = [math.log(y) for y in y_data]

    # Fit linear model to (x, ln(y))
    linear_result = fit_linear(x_data, ln_y)
    b = linear_result["coefficients"]["m"]
    ln_a = linear_result["coefficients"]["b"]
    a = math.exp(ln_a)

    # Calculate R-squared on original scale
    y_predicted = [a * math.exp(b * x) for x in x_data]
    r_squared = calculate_r_squared(y_data, y_predicted)

    coefficients = {"a": a, "b": b}
    expression = build_expression("exponential", coefficients)

    return {
        "expression": expression,
        "coefficients": coefficients,
        "r_squared": r_squared,
        "model_type": "exponential",
    }


def fit_logarithmic(x_data: List[float], y_data: List[float]) -> RegressionResult:
    """
    Fit logarithmic model: y = a + b * ln(x)

    Requires all x values to be positive.

    Returns coefficients {a, b} and R-squared.
    """
    _validate_data(x_data, y_data)
    _validate_positive(x_data, "x_data")

    # Transform x to ln(x)
    ln_x = [math.log(x) for x in x_data]

    # Fit linear model to (ln(x), y)
    linear_result = fit_linear(ln_x, y_data)
    b = linear_result["coefficients"]["m"]
    a = linear_result["coefficients"]["b"]

    # Calculate R-squared
    y_predicted = [a + b * math.log(x) for x in x_data]
    r_squared = calculate_r_squared(y_data, y_predicted)

    coefficients = {"a": a, "b": b}
    expression = build_expression("logarithmic", coefficients)

    return {
        "expression": expression,
        "coefficients": coefficients,
        "r_squared": r_squared,
        "model_type": "logarithmic",
    }


def fit_power(x_data: List[float], y_data: List[float]) -> RegressionResult:
    """
    Fit power model: y = a * x^b

    Linearized form: ln(y) = ln(a) + b*ln(x)
    Requires all x and y values to be positive.

    Returns coefficients {a, b} and R-squared.
    """
    _validate_data(x_data, y_data)
    _validate_positive(x_data, "x_data")
    _validate_positive(y_data, "y_data")

    # Linearize: ln(y) = ln(a) + b*ln(x)
    ln_x = [math.log(x) for x in x_data]
    ln_y = [math.log(y) for y in y_data]

    # Fit linear model to (ln(x), ln(y))
    linear_result = fit_linear(ln_x, ln_y)
    b = linear_result["coefficients"]["m"]
    ln_a = linear_result["coefficients"]["b"]
    a = math.exp(ln_a)

    # Calculate R-squared on original scale
    y_predicted = [a * (x ** b) for x in x_data]
    r_squared = calculate_r_squared(y_data, y_predicted)

    coefficients = {"a": a, "b": b}
    expression = build_expression("power", coefficients)

    return {
        "expression": expression,
        "coefficients": coefficients,
        "r_squared": r_squared,
        "model_type": "power",
    }


def fit_logistic(
    x_data: List[float],
    y_data: List[float],
    L_init: Optional[float] = None,
    k_init: Optional[float] = None,
    x0_init: Optional[float] = None,
    max_iterations: int = 100,
) -> RegressionResult:
    """
    Fit logistic model: y = L / (1 + e^(-k(x - x0)))

    Uses grid search followed by coordinate descent refinement for Brython
    compatibility (no scipy/numpy dependency).

    Performance note:
        This function uses iterative optimization which may be slower than
        analytical methods. Each iteration evaluates 27 candidate solutions.
        For typical datasets (<1000 points), fitting completes in <100ms.
        Very large datasets may benefit from subsampling before fitting.

    Parameters:
        x_data: List of x values
        y_data: List of y values
        L_init: Initial estimate for carrying capacity (default: max(y) * 1.1)
        k_init: Initial estimate for growth rate (default: 1.0)
        x0_init: Initial estimate for midpoint (default: mean of x range)
        max_iterations: Maximum refinement iterations (default: 100)

    Returns:
        RegressionResult with coefficients {L, k, x0} and R-squared.
    """
    _validate_data(x_data, y_data)

    min(y_data)
    y_max = max(y_data)
    x_min = min(x_data)
    x_max = max(x_data)

    # Initial estimates
    if L_init is None:
        L_init = y_max * 1.1 if y_max > 0 else 1.0
    if x0_init is None:
        x0_init = (x_min + x_max) / 2
    if k_init is None:
        k_init = 1.0

    def logistic(x: float, L: float, k: float, x0: float) -> float:
        exp_arg = -k * (x - x0)
        # Prevent overflow
        if exp_arg > 700:
            return 0.0
        if exp_arg < -700:
            return L
        return L / (1.0 + math.exp(exp_arg))

    def compute_sse(L: float, k: float, x0: float) -> float:
        sse = 0.0
        for x, y in zip(x_data, y_data):
            y_pred = logistic(x, L, k, x0)
            sse += (y - y_pred) ** 2
        return sse

    # Grid search for initial estimates
    best_L, best_k, best_x0 = L_init, k_init, x0_init
    best_sse = compute_sse(best_L, best_k, best_x0)

    L_range = [y_max * f for f in [0.9, 1.0, 1.1, 1.2, 1.5]]
    k_range = [0.1, 0.5, 1.0, 2.0, 5.0]
    x0_range = [x_min + (x_max - x_min) * f for f in [0.25, 0.5, 0.75]]

    for L_test in L_range:
        for k_test in k_range:
            for x0_test in x0_range:
                sse = compute_sse(L_test, k_test, x0_test)
                if sse < best_sse:
                    best_sse = sse
                    best_L, best_k, best_x0 = L_test, k_test, x0_test

    # Simple refinement using coordinate descent
    step_L = abs(best_L) * 0.1 if best_L != 0 else 0.1
    step_k = abs(best_k) * 0.1 if best_k != 0 else 0.1
    step_x0 = (x_max - x_min) * 0.1

    for _ in range(max_iterations):
        improved = False

        for delta_L in [-step_L, 0, step_L]:
            for delta_k in [-step_k, 0, step_k]:
                for delta_x0 in [-step_x0, 0, step_x0]:
                    if delta_L == delta_k == delta_x0 == 0:
                        continue
                    new_L = best_L + delta_L
                    new_k = best_k + delta_k
                    new_x0 = best_x0 + delta_x0
                    if new_L <= 0 or new_k <= 0:
                        continue
                    sse = compute_sse(new_L, new_k, new_x0)
                    if sse < best_sse:
                        best_sse = sse
                        best_L, best_k, best_x0 = new_L, new_k, new_x0
                        improved = True

        if not improved:
            step_L *= 0.5
            step_k *= 0.5
            step_x0 *= 0.5
            if step_L < 1e-8 and step_k < 1e-8 and step_x0 < 1e-8:
                break

    # Calculate R-squared
    y_predicted = [logistic(x, best_L, best_k, best_x0) for x in x_data]
    r_squared = calculate_r_squared(y_data, y_predicted)

    coefficients = {"L": best_L, "k": best_k, "x0": best_x0}
    expression = build_expression("logistic", coefficients)

    return {
        "expression": expression,
        "coefficients": coefficients,
        "r_squared": r_squared,
        "model_type": "logistic",
    }


def fit_sinusoidal(
    x_data: List[float],
    y_data: List[float],
    max_iterations: int = 100,
) -> RegressionResult:
    """
    Fit sinusoidal model: y = a * sin(bx + c) + d

    Uses period estimation via zero-crossing detection followed by coordinate
    descent refinement for Brython compatibility (no scipy/numpy dependency).

    Performance note:
        This function uses iterative optimization which may be slower than
        analytical methods. Each iteration evaluates 81 candidate solutions
        (3^4 combinations). For typical datasets (<1000 points), fitting
        completes in <200ms. Very large datasets may benefit from subsampling
        before fitting.

    Parameters:
        x_data: List of x values
        y_data: List of y values (must have at least 4 points)
        max_iterations: Maximum refinement iterations (default: 100)

    Returns:
        RegressionResult with coefficients {a, b, c, d} where:
        - a: amplitude
        - b: angular frequency (2*pi/period)
        - c: phase shift
        - d: vertical offset
    """
    _validate_data(x_data, y_data, min_points=4)

    y_mean = sum(y_data) / len(y_data)
    y_max = max(y_data)
    y_min = min(y_data)
    x_min = min(x_data)
    x_max = max(x_data)
    x_range = x_max - x_min

    # Initial estimates
    d_init = y_mean
    a_init = (y_max - y_min) / 2
    if a_init < 1e-10:
        a_init = 1.0

    # Estimate period by finding zero crossings in (y - mean)
    y_centered = [y - y_mean for y in y_data]
    crossings = []
    sorted_pairs = sorted(zip(x_data, y_centered))
    for i in range(len(sorted_pairs) - 1):
        x1, y1 = sorted_pairs[i]
        x2, y2 = sorted_pairs[i + 1]
        if y1 * y2 < 0:  # Sign change
            # Linear interpolation to find crossing
            x_cross = x1 - y1 * (x2 - x1) / (y2 - y1)
            crossings.append(x_cross)

    if len(crossings) >= 2:
        # Estimate period as twice average distance between crossings
        crossing_diffs = [crossings[i+1] - crossings[i] for i in range(len(crossings) - 1)]
        half_period = sum(crossing_diffs) / len(crossing_diffs)
        period = 2 * half_period
        b_init = 2 * math.pi / period if period > 1e-10 else 1.0
    else:
        # Fall back to assuming one period spans the data
        b_init = 2 * math.pi / x_range if x_range > 1e-10 else 1.0

    c_init = 0.0

    def sinusoidal(x: float, a: float, b: float, c: float, d: float) -> float:
        return a * math.sin(b * x + c) + d

    def compute_sse(a: float, b: float, c: float, d: float) -> float:
        sse = 0.0
        for x, y in zip(x_data, y_data):
            y_pred = sinusoidal(x, a, b, c, d)
            sse += (y - y_pred) ** 2
        return sse

    # Grid search for better initial values
    best_a, best_b, best_c, best_d = a_init, b_init, c_init, d_init
    best_sse = compute_sse(best_a, best_b, best_c, best_d)

    # Try different periods
    b_values = [b_init * f for f in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]]
    c_values = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi]

    for b_test in b_values:
        for c_test in c_values:
            sse = compute_sse(a_init, b_test, c_test, d_init)
            if sse < best_sse:
                best_sse = sse
                best_b, best_c = b_test, c_test

    # Coordinate descent refinement
    step_a = abs(best_a) * 0.1 if best_a != 0 else 0.1
    step_b = abs(best_b) * 0.1 if best_b != 0 else 0.1
    step_c = 0.1
    step_d = abs(best_d) * 0.1 if best_d != 0 else 0.1

    for _ in range(max_iterations):
        improved = False

        for delta_a in [-step_a, 0, step_a]:
            for delta_b in [-step_b, 0, step_b]:
                for delta_c in [-step_c, 0, step_c]:
                    for delta_d in [-step_d, 0, step_d]:
                        if delta_a == delta_b == delta_c == delta_d == 0:
                            continue
                        new_a = best_a + delta_a
                        new_b = best_b + delta_b
                        new_c = best_c + delta_c
                        new_d = best_d + delta_d
                        if new_b <= 0:
                            continue
                        sse = compute_sse(new_a, new_b, new_c, new_d)
                        if sse < best_sse:
                            best_sse = sse
                            best_a, best_b, best_c, best_d = new_a, new_b, new_c, new_d
                            improved = True

        if not improved:
            step_a *= 0.5
            step_b *= 0.5
            step_c *= 0.5
            step_d *= 0.5
            if step_a < 1e-8 and step_b < 1e-8 and step_c < 1e-8 and step_d < 1e-8:
                break

    # Calculate R-squared
    y_predicted = [sinusoidal(x, best_a, best_b, best_c, best_d) for x in x_data]
    r_squared = calculate_r_squared(y_data, y_predicted)

    coefficients = {"a": best_a, "b": best_b, "c": best_c, "d": best_d}
    expression = build_expression("sinusoidal", coefficients)

    return {
        "expression": expression,
        "coefficients": coefficients,
        "r_squared": r_squared,
        "model_type": "sinusoidal",
    }


# ---------------------------------------------------------------------------
# Main fitting dispatcher
# ---------------------------------------------------------------------------

SUPPORTED_MODEL_TYPES = (
    "linear",
    "polynomial",
    "exponential",
    "logarithmic",
    "power",
    "logistic",
    "sinusoidal",
)


def fit_regression(
    x_data: List[float],
    y_data: List[float],
    model_type: str,
    degree: Optional[int] = None,
) -> RegressionResult:
    """
    Fit a regression model to data.

    Args:
        x_data: List of x values
        y_data: List of y values
        model_type: One of "linear", "polynomial", "exponential",
                    "logarithmic", "power", "logistic", "sinusoidal"
        degree: Polynomial degree (required for polynomial model)

    Returns:
        RegressionResult with expression, coefficients, r_squared, model_type
    """
    model = model_type.strip().lower() if isinstance(model_type, str) else ""

    if model not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Supported: {', '.join(SUPPORTED_MODEL_TYPES)}"
        )

    if model == "linear":
        return fit_linear(x_data, y_data)
    elif model == "polynomial":
        if degree is None:
            raise ValueError("degree is required for polynomial regression")
        return fit_polynomial(x_data, y_data, degree)
    elif model == "exponential":
        return fit_exponential(x_data, y_data)
    elif model == "logarithmic":
        return fit_logarithmic(x_data, y_data)
    elif model == "power":
        return fit_power(x_data, y_data)
    elif model == "logistic":
        return fit_logistic(x_data, y_data)
    elif model == "sinusoidal":
        return fit_sinusoidal(x_data, y_data)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
