"""SymPy computations for the CAS tools: derive, limit, integrate, simplify, expand, factor,
solve, systems of equations, the solution sets of inequalities, whether two expressions are
equivalent (for the step checker), Taylor polynomials, partial fractions and sums.

SymPy is the app's only symbolic engine: the client asks the server and checks every
result numerically in the browser. Without SymPy the symbolic tools say so, and the numeric
ones (quadrature, the limit verdict, numeric roots, sign charts) still answer.

Expressions arrive as text in the client's notation (``^`` for powers, ``e``, ``pi``,
``i``, ``Infinity``). They are never evaluated: ``parse_expression`` reads them with
Python's ``ast`` module, accepts only numbers, names, arithmetic and calls to the
functions in ``_FUNCTIONS``, and builds the SymPy expression from that tree. Results are
printed back in the same notation, so math.js can read them.

This module runs in a worker process (``python -m static.sympy_cas``, managed by
``static.sympy_engine``) that is killed when a computation runs too long; it reads one
JSON request per line on stdin and writes one JSON response per line on stdout.
"""

from __future__ import annotations

import ast
import cmath
import json
import math
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Inputs longer than this are refused before parsing.
MAX_EXPRESSION_LENGTH = 1000
# Integer literals with more digits than this are refused.
MAX_INTEGER_DIGITS = 60
# A decimal is read as nsimplify's fraction only when they agree this closely (relative).
DECIMAL_AGREEMENT = 1e-14
# A power of two numbers is left unevaluated when the exponent is larger than this, so
# 10^10^10 cannot exhaust memory before the time limit stops the worker.
MAX_EVALUATED_EXPONENT = 10_000
# Periodic solutions are listed for n with |root| <= this.
SOLUTION_WINDOW = 2 * math.pi
# Names that are constants, in the client's notation.
_CONSTANTS = ("e", "E", "pi", "i", "Infinity", "inf", "oo")
# Variable names: identifiers (Greek letters included, as in Δx) of at most 31 characters,
# not starting with "_".


def _is_name(name: str) -> bool:
    return name.isidentifier() and not name.startswith("_") and len(name) <= 31


class ExpressionError(ValueError):
    """The text is not an expression this module accepts."""


def _function_table() -> Dict[str, Tuple[Callable[..., Any], Tuple[int, ...]]]:
    """name -> (SymPy builder, allowed argument counts)."""
    import sympy as sp

    def log(*args: Any) -> Any:
        return sp.log(args[0]) if len(args) == 1 else sp.log(args[0], args[1])

    one: Tuple[int, ...] = (1,)
    table: Dict[str, Tuple[Callable[..., Any], Tuple[int, ...]]] = {
        name: (getattr(sp, name), one)
        for name in (
            "sin",
            "cos",
            "tan",
            "sec",
            "csc",
            "cot",
            "asin",
            "acos",
            "atan",
            "acot",
            "asec",
            "acsc",
            "sinh",
            "cosh",
            "tanh",
            "coth",
            "sech",
            "csch",
            "asinh",
            "acosh",
            "atanh",
            "exp",
            "sqrt",
            "sign",
            "floor",
            "factorial",
            "gamma",
            "erf",
        )
    }
    table.update(
        {
            "log": (log, (1, 2)),
            "ln": (sp.log, one),
            "log10": (lambda a: sp.log(a, 10), one),
            "log2": (lambda a: sp.log(a, 2), one),
            "abs": (sp.Abs, one),
            "sgn": (sp.sign, one),
            "ceil": (sp.ceiling, one),
            "cbrt": (lambda a: sp.real_root(a, 3), one),
            "nthroot": (sp.real_root, (2,)),
            "atan2": (sp.atan2, (2,)),
            "mod": (sp.Mod, (2,)),
            "re": (sp.re, one),
            "im": (sp.im, one),
            "arg": (sp.arg, one),
            "conj": (sp.conjugate, one),
            "pow": (sp.Pow, (2,)),
            "min": (sp.Min, (2, 3, 4)),
            "max": (sp.Max, (2, 3, 4)),
        }
    )
    return table


_function_cache: Dict[str, Tuple[Callable[..., Any], Tuple[int, ...]]] = {}


def _functions() -> Dict[str, Tuple[Callable[..., Any], Tuple[int, ...]]]:
    if not _function_cache:
        _function_cache.update(_function_table())
    return _function_cache


def _prepared_text(text: str) -> str:
    """Client notation as Python syntax: ^ -> **, and the implicit products 2x, 2(x), (a)(b), (a)x."""
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(f"The expression is longer than {MAX_EXPRESSION_LENGTH} characters.")
    prepared = text.replace("^", "**").replace("·", "*").replace("×", "*")
    # 2x, 2.5x, 2(x+1) but not 1e5 or 1e-3
    prepared = re.sub(r"(?<![A-Za-z_\d.])(\d+\.?\d*)(?![eE][+-]?\d)\s*(?=[A-Za-z(])", r"\1*", prepared)
    prepared = re.sub(r"\)\s*(?=[A-Za-z\d(])", ")*", prepared)
    return prepared


def parse_expression(text: str, real: bool = False, names: Optional[Dict[str, Any]] = None) -> Any:
    """The SymPy expression for text, built from its syntax tree without evaluating it.

    Every name that is not a constant or a known function becomes a symbol (real when
    real is True); a name in names becomes the symbol it maps to, even a constant's name
    (a summation index i).

    Raises:
        ExpressionError: for syntax errors and anything but numbers, names, arithmetic and
            calls to known functions (attributes, subscripts, keywords, lambdas, ...).
    """
    try:
        tree = ast.parse(_prepared_text(str(text)).strip(), mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"Cannot read the expression {text!r}: {error.msg}.") from None
    return _Builder(real, names).build(tree.body)


class _Builder:
    """Turns an ast tree into a SymPy expression, refusing every node it does not know."""

    def __init__(self, real: bool, names: Optional[Dict[str, Any]] = None) -> None:
        import sympy as sp

        self.sp = sp
        self.real = real
        self.names = names or {}

    def build(self, node: ast.AST) -> Any:
        if isinstance(node, ast.BinOp):
            left, right = self.build(node.left), self.build(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return self._power(left, right)
            if isinstance(node.op, ast.Mod):
                return self.sp.Mod(left, right)  # floored, like Python's % and math.js mod
            raise ExpressionError(f"The operator {type(node.op).__name__} is not supported.")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = self.build(node.operand)
            return -operand if isinstance(node.op, ast.USub) else operand
        if isinstance(node, ast.Constant):
            return self._number(node.value)
        if isinstance(node, ast.Name):
            return self._name(node.id)
        if isinstance(node, ast.Call):
            return self._call(node)
        raise ExpressionError(f"{type(node).__name__} is not allowed in an expression.")

    def _power(self, base: Any, exponent: Any) -> Any:
        sp = self.sp
        if base.is_Number and exponent.is_Number and abs(exponent) > MAX_EVALUATED_EXPONENT:
            return sp.Pow(base, exponent, evaluate=False)
        return base**exponent

    def _number(self, value: Any) -> Any:
        sp = self.sp
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ExpressionError("Only numbers are allowed as literals.")
        if isinstance(value, int):
            if len(str(abs(value))) > MAX_INTEGER_DIGITS:
                raise ExpressionError(f"Integers have at most {MAX_INTEGER_DIGITS} digits.")
            return sp.Integer(value)
        if not math.isfinite(value):
            raise ExpressionError("Numbers must be finite.")
        if abs(value) >= 10.0**MAX_INTEGER_DIGITS:
            raise ExpressionError(f"Numbers have at most {MAX_INTEGER_DIGITS} digits before the point.")
        # A decimal such as 0.5 is read as the exact rational it stands for: nsimplify's
        # (0.333333333333333 as 1/3) when it is the decimal up to rounding and no longer than
        # it (1e-300 as 10^-300, where nsimplify gives a fraction of two 300-digit numbers;
        # 1.0000000001 as itself, where nsimplify gives 1).
        simplified = sp.nsimplify(sp.Float(repr(value)), rational=True)
        exact = sp.Rational(repr(value))
        close = abs(float(simplified) - value) <= DECIMAL_AGREEMENT * abs(value)
        return simplified if close and len(str(simplified)) <= len(str(exact)) else exact

    def _name(self, name: str) -> Any:
        sp = self.sp
        if name in self.names:
            return self.names[name]
        constants = {"e": sp.E, "E": sp.E, "pi": sp.pi, "i": sp.I, "Infinity": sp.oo, "inf": sp.oo, "oo": sp.oo}
        if name in constants:
            return constants[name]
        if not _is_name(name):
            raise ExpressionError(f"{name!r} is not a valid variable name.")
        if name in _functions():
            raise ExpressionError(f"{name} is a function; call it with parentheses.")
        return sp.Symbol(name, real=True) if self.real else sp.Symbol(name)

    def _call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("Only named functions can be called.")
        name = node.func.id
        entry = _functions().get(name)
        if entry is None:
            raise ExpressionError(f"Unknown function {name}().")
        if node.keywords:
            raise ExpressionError(f"{name}() takes no keyword arguments.")
        if any(isinstance(argument, ast.Starred) for argument in node.args):
            raise ExpressionError(f"{name}() takes no unpacked arguments.")
        builder, counts = entry
        if len(node.args) not in counts:
            raise ExpressionError(f"{name}() takes {' or '.join(str(count) for count in counts)} argument(s).")
        arguments = [self.build(argument) for argument in node.args]
        if name == "pow":
            return self._power(*arguments)  # the same guard as ^
        return builder(*arguments)


# ----- printing --------------------------------------------------------------------------


def _printer_class() -> Any:
    from sympy.printing.precedence import precedence
    from sympy.printing.str import StrPrinter

    class ClientPrinter(StrPrinter):
        """SymPy's str() in the client's notation: ^, e, i, Infinity, abs, ceil, mod, conj."""

        def _print_Pow(self, expr: Any, rational: bool = False) -> str:
            text = str(super()._print_Pow(expr, rational))
            if "**" not in text:
                return text  # sqrt(x), 1/x
            base = self.parenthesize(expr.base, precedence(expr), strict=False)
            exponent = self._print(expr.exp)
            if not (expr.exp.is_Symbol or (expr.exp.is_Integer and expr.exp >= 0)):
                exponent = f"({exponent})"
            return f"{base}^{exponent}"

        def _print_Exp1(self, expr: Any) -> str:
            return "e"

        def _print_ImaginaryUnit(self, expr: Any) -> str:
            return "i"

        def _print_Infinity(self, expr: Any) -> str:
            return "Infinity"

        def _print_NegativeInfinity(self, expr: Any) -> str:
            return "-Infinity"

        def _print_Abs(self, expr: Any) -> str:
            return f"abs({self._print(expr.args[0])})"

        def _print_ceiling(self, expr: Any) -> str:
            return f"ceil({self._print(expr.args[0])})"

        def _print_Mod(self, expr: Any) -> str:
            return f"mod({','.join(self._print(arg) for arg in expr.args)})"

        def _print_conjugate(self, expr: Any) -> str:
            return f"conj({self._print(expr.args[0])})"

        def _print_Min(self, expr: Any) -> str:
            return f"min({','.join(self._print(arg) for arg in expr.args)})"

        def _print_Max(self, expr: Any) -> str:
            return f"max({','.join(self._print(arg) for arg in expr.args)})"

        def _print_Float(self, expr: Any) -> str:
            value = float(expr)
            return repr(int(value)) if value.is_integer() and abs(value) < 1e15 else f"{value:.15g}"

    return ClientPrinter


_printer: List[Any] = []


def to_text(expression: Any) -> str:
    """expression in the client's notation, without spaces."""
    if not _printer:
        _printer.append(_printer_class()())
    return str(_printer[0].doprint(expression)).replace(" ", "")


def _is_usable(expression: Any) -> bool:
    """False for unevaluated integrals, limits and derivatives, NaN, complex infinity and bounds such as <-1, 1>."""
    import sympy as sp

    unevaluated = (sp.Integral, sp.Limit, sp.Derivative, sp.Subs)
    if expression is None or expression.has(*unevaluated, sp.nan, sp.zoo, sp.AccumBounds):
        return False
    return not expression.has(sp.Piecewise)


# A root whose exact form is longer than this (Cardano's formula for a cubic) is given in decimals.
MAX_EXACT_ROOT_LENGTH = 60


def _has_nested_radical(expression: Any) -> bool:
    """True for a radical of an expression holding another radical, e.g. (27/2+3*sqrt(93)/2)^(1/3)."""
    import sympy as sp

    def is_radical(node: Any) -> bool:
        return isinstance(node, sp.Pow) and not node.exp.is_Integer

    return any(
        is_radical(node) and any(is_radical(inner) for inner in sp.preorder_traversal(node.base))
        for node in sp.preorder_traversal(expression)
    )


def _displayed_root(root: Any) -> Any:
    """root, or its decimal value when it is a CRootOf or a long radical expression without parameters."""
    import sympy as sp

    if root.free_symbols:
        return root  # Cardano's formula in a parameter has no decimal value
    if not root.has(sp.CRootOf) and len(to_text(root)) <= MAX_EXACT_ROOT_LENGTH and not _has_nested_radical(root):
        return root
    value = sp.N(root, 15)
    real, imaginary = value.as_real_imag()
    if abs(imaginary) <= 1e-12 * max(1, abs(real)):
        return real
    return value


# ----- operations ------------------------------------------------------------------------


def _symbol(name: str, real: bool) -> Any:
    import sympy as sp

    if not _is_name(str(name)) or name in _CONSTANTS:
        raise ExpressionError(f"{name!r} is not a valid variable name.")
    return sp.Symbol(str(name), real=True) if real else sp.Symbol(str(name))


def compute_limit(expression: str, variable: str, target: str) -> Optional[str]:
    """lim expression as variable -> target, or None when there is no single (two-sided) value."""
    import sympy as sp

    expr = parse_expression(expression, real=True)
    x = _symbol(variable, real=True)
    point = parse_expression(target, real=True)
    if point in (sp.oo, -sp.oo):
        result = sp.limit(expr, x, point)
    else:
        right = sp.limit(expr, x, point, "+")
        left = sp.limit(expr, x, point, "-")
        if not (_is_usable(right) and _is_usable(left)) or sp.simplify(right - left) != 0:
            if right in (sp.oo, -sp.oo) and right == left:
                return to_text(right)
            return None
        result = right
    if not _is_usable(result) or not _is_real_value(result):
        return None
    return to_text(result)


def _is_real_value(value: Any) -> bool:
    """False for a value that is not real, or may not be: 2*(-1)^(1/3), -Infinity + i*pi."""
    import sympy as sp

    if value in (sp.oo, -sp.oo):
        return True
    return not value.has(sp.I, sp.oo, -sp.oo) and value.is_extended_real is not False


def compute_integral(
    expression: str, variable: str, lower: Optional[str] = None, upper: Optional[str] = None
) -> Optional[str]:
    """The antiderivative (no constant), or the definite integral over [lower, upper]; None if SymPy has none."""
    import sympy as sp

    expr = parse_expression(expression, real=True)
    x = _symbol(variable, real=True)
    if lower is None or upper is None:
        result = _without_piecewise(sp.integrate(expr, x, conds="none"), x)
    else:
        bounds = (x, parse_expression(lower, real=True), parse_expression(upper, real=True))
        result = sp.integrate(expr, bounds, conds="none")
        if not result.free_symbols and not _is_real_value(result):
            return None  # 1/x over [-Infinity, -1] gives -Infinity + i*pi
    if not _is_usable(result):
        return None
    if not result.free_symbols:
        result = _shorter(result, sp.simplify(result))
    return to_text(result)


def _without_piecewise(result: Any, x: Any) -> Any:
    """A two-branch Piecewise split at x = c (the antiderivative of abs(x) or sign(x)) written
    with sign: (A + B)/2 + sign(x - c)*(B - A)/2, which is A left of c and B right of it. Other
    results are returned as they are."""
    import sympy as sp

    if not isinstance(result, sp.Piecewise) or len(result.args) != 2 or result.args[1].cond != sp.true:
        return result
    (left, condition), (right, _) = result.args
    if not isinstance(condition, (sp.Le, sp.Lt)) or condition.lhs != x or condition.rhs.free_symbols:
        return result
    split = condition.rhs
    if sp.simplify(left.subs(x, split) - right.subs(x, split)) != 0:
        return result  # not continuous at the split: no single antiderivative
    joined = sp.factor((left + right) / 2) + sp.sign(x - split) * sp.factor((right - left) / 2)
    return _shorter(joined, sp.simplify(joined))


def _shorter(expression: Any, alternative: Any) -> Any:
    return alternative if len(to_text(alternative)) < len(to_text(expression)) else expression


def compute_derivative(expression: str, variable: str) -> Optional[str]:
    """d/d(variable) of expression, with the DiracDelta terms of abs/sign at their jumps dropped."""
    import sympy as sp

    x = _symbol(variable, real=True)
    result = sp.diff(parse_expression(expression, real=True), x)
    result = result.replace(sp.DiracDelta, lambda *args: sp.S.Zero)
    result = _shorter(result, sp.simplify(result))
    return to_text(result) if _is_usable(result) else None


def compute_simplify(expression: str) -> Optional[str]:
    """SymPy's simplify, with common factors of a quotient cancelled when simplify keeps them."""
    import sympy as sp

    result = sp.simplify(parse_expression(expression, real=True))
    cancelled = sp.cancel(result)
    if sp.count_ops(sp.denom(cancelled)) < sp.count_ops(sp.denom(result)):
        result = cancelled
    return to_text(result) if _is_usable(result) else None


def compute_solve(
    equation: str,
    variable: str,
    lower: Optional[str] = None,
    upper: Optional[str] = None,
    bounds: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The solution set of lhs = rhs, or None when it is not a list of roots.

    Returns {"roots": [...]} or {"identity": True, "roots": []}. A periodic solution set
    lists the real roots in [-window, window] (2*pi, or less when there are more than
    MAX_LISTED_ROOTS), with "window" set to that bound and "general" to its general forms
    (x = a*n + b for any integer n) when they are affine in n. A polynomial or rational
    equation is solved over the complex numbers, any other over the reals.

    With lower and upper (and bounds, see _interval), the real roots in that interval are
    listed: at most MAX_LISTED_ROOTS, the smallest first, with "total" set to their number
    when there are more.
    """
    import sympy as sp

    sides = str(equation).split("=")
    if len(sides) > 2:
        raise ExpressionError("An equation has at most one '='.")
    domain = _interval(lower, upper, bounds)
    if domain is not None:
        return _roots_in_interval(sides, variable, domain)
    x = _symbol(variable, real=False)
    difference = parse_expression(sides[0]) - (parse_expression(sides[1]) if len(sides) == 2 else 0)
    numerator, denominator = sp.fraction(sp.together(difference))
    rational = numerator.is_polynomial(x) and denominator.is_polynomial(x)
    if not rational:
        # Over the reals the variable is real, so sqrt(x^2) and the like simplify.
        real_x = _symbol(variable, real=True)
        difference = difference.subs(x, real_x)
        x = real_x
    domain = sp.S.Complexes if rational else sp.S.Reals
    solutions = sp.solveset(difference, x, domain=domain)
    if solutions == domain:
        return {"identity": True, "roots": []}
    if solutions == sp.S.EmptySet:
        return {"roots": []}
    if _is_parameter_roots(solutions):
        solutions = solutions.args[0] if isinstance(solutions.args[0], sp.FiniteSet) else solutions.args[1]
    if (
        isinstance(solutions, sp.ConditionSet)
        and isinstance(solutions.base_set, sp.FiniteSet)
        and x not in solutions.condition.free_symbols
        and _possible(solutions.condition)
    ):
        # |x| = a gives {-a, a} for a >= 0: the condition is on the parameters
        solutions = solutions.base_set
    if isinstance(solutions, sp.Complement) and isinstance(solutions.args[0], sp.FiniteSet):
        # {V/R} \ {0} for R = V/I: the poles left out are ones the roots avoid (the client drops poles)
        solutions = solutions.args[0]
    if not isinstance(solutions, sp.FiniteSet) and difference.free_symbols - {x}:
        solutions = _formula_roots(difference, x) or solutions
    if isinstance(solutions, sp.FiniteSet):
        listed = [_displayed_root(root) for root in solutions]
        if not all(_is_usable(root) for root in listed):
            return None
        return {"roots": [to_text(root) for root in _sorted_roots(listed)]}
    return _periodic_solution(solutions, variable)


# Values given to the parameters when a formula's roots are checked: positive, as in most formulas
# (T = 2*pi*sqrt(L/g)), small and large (d = sqrt((x-1)^2+4) has real roots only for d >= 2).
_PARAMETER_SAMPLES = (1.3, 6.7, 0.45, 0.1)


def _formula_roots(difference: Any, x: Any) -> Any:
    """The roots sp.solve gives for an equation with parameters that solveset cannot list as a
    set (sqrt(x) = a, a^x = b), each checked at sample parameter values; None when there are none
    or the equation is periodic in x (sp.solve gives sin(x) = a its principal values only)."""
    import sympy as sp
    from sympy.functions.elementary.trigonometric import TrigonometricFunction

    if any(node.has(x) for node in difference.atoms(TrigonometricFunction)):
        return None
    try:
        roots = sp.solve(difference, x)
    except (NotImplementedError, ValueError, TypeError):
        return None
    parameters = sorted(difference.free_symbols - {x}, key=str)
    checked = [root for root in roots if _holds_for_samples(difference, x, root, parameters)]
    return sp.FiniteSet(*checked) if checked else None


def _possible(condition: Any) -> bool:
    """True when a condition on the parameters holds at one or more sample values (a >= 0 for
    |x| = a; never -a^2 - 1 >= 0)."""
    import sympy as sp

    parameters = sorted(condition.free_symbols, key=str)
    for sample in _PARAMETER_SAMPLES + tuple(-value for value in _PARAMETER_SAMPLES):
        values = {name: sp.Float(sample + 0.37 * index) for index, name in enumerate(parameters)}
        if condition.subs(values) == sp.true:
            return True
    return False


def _holds_for_samples(difference: Any, x: Any, root: Any, parameters: Sequence[Any]) -> bool:
    """True when the equation holds at the root for some sample parameter values that make the root
    real (sqrt(x) = a - 5 has the root (a-5)^2 only for a >= 5)."""
    import sympy as sp

    for sample in _PARAMETER_SAMPLES:
        values = {name: sp.Float(sample + 0.37 * index) for index, name in enumerate(parameters)}
        try:
            point = complex(sp.N(root.subs(values), 15))
            if not (cmath.isfinite(point) and abs(point.imag) <= 1e-9 * max(1.0, abs(point.real))):
                continue
            residual = complex(sp.N(difference.subs(values).subs(x, point.real), 15))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if cmath.isfinite(residual) and abs(residual) <= 1e-8 * max(1.0, abs(point.real)):
            return True
    return False


def _is_parameter_roots(solutions: Any) -> bool:
    """True for {roots} ∩ Reals, solveset's answer when the roots hold other names (e^(k*x) = 2
    gives {log(2)/k} ∩ Reals): the roots are complete, and real for the parameter values that
    make them real."""
    import sympy as sp

    if not isinstance(solutions, sp.Intersection) or len(solutions.args) != 2:
        return False
    first, second = solutions.args
    return sp.S.Reals in (first, second) and any(
        isinstance(part, sp.FiniteSet) and part.free_symbols for part in (first, second)
    )


def _roots_in_interval(sides: Sequence[str], variable: str, domain: Any) -> Optional[Dict[str, Any]]:
    """The real roots of lhs = rhs in domain (an Interval), or None when they are not a finite list."""
    import sympy as sp

    x = _symbol(variable, real=True)
    difference = parse_expression(sides[0], real=True) - (
        parse_expression(sides[1], real=True) if len(sides) == 2 else 0
    )
    solutions = sp.solveset(difference, x, domain=domain)
    if solutions == domain:
        return {"identity": True, "roots": []}
    if solutions == sp.S.EmptySet:
        return {"roots": []}
    if not isinstance(solutions, sp.FiniteSet):
        return None  # a condition set (cos(x) = x) or roots solveset cannot list (sin(x^2) = 0)
    listed = [_displayed_root(root) for root in solutions]
    if not all(_is_usable(root) and _is_real_value(root) for root in listed):
        return None
    ordered = _sorted_roots(listed)
    result: Dict[str, Any] = {"roots": [to_text(root) for root in ordered[:MAX_LISTED_ROOTS]]}
    if len(ordered) > MAX_LISTED_ROOTS:
        result["total"] = len(ordered)
    return result


# At most this many roots of a periodic equation are listed (the window shrinks to fit).
MAX_LISTED_ROOTS = 60


def _periodic_solution(solutions: Any, variable: str) -> Optional[Dict[str, Any]]:
    """The roots near 0 of a union of finite sets and integer image sets, or None for other sets."""
    import sympy as sp

    parts = list(solutions.args) if isinstance(solutions, sp.Union) else [solutions]
    general: List[str] = []
    roots: List[Any] = []
    for part in parts:
        if isinstance(part, sp.FiniteSet):
            roots.extend(part)
        elif isinstance(part, sp.ImageSet) and part.base_sets[0] == sp.S.Integers:
            form = _general_form(part, variable)
            if form is not None:
                general.append(form)
            try:
                roots.extend(_roots_in_window(part))
            except _IncompleteSolution:
                return None
        else:
            return None  # an interval, a condition set, or something else not a list of roots
    if not all(_is_usable(root) for root in roots):
        return None
    listed, window = _nearest_roots(_distinct(roots))
    result: Dict[str, Any] = {"roots": [to_text(root) for root in _sorted_roots(listed)], "window": window}
    if general and len(general) == sum(isinstance(part, sp.ImageSet) for part in parts):
        result["general"] = general
    return result


def _general_form(image: Any, variable: str) -> Optional[str]:
    """ "x = a*n+b" for an image set n -> a*n + b, else None (sqrt(n) is not real for every n)."""
    import sympy as sp

    n = sp.Symbol("n", integer=True)
    expression = image.lamda.expr.subs(image.lamda.variables[0], n)
    if sp.diff(expression, n, 2) != 0 or sp.diff(expression, n).has(n):
        return None
    return f"{variable} = {to_text(expression)}"


def _roots_in_window(image: Any) -> List[Any]:
    """The real values of an integer image set within SOLUTION_WINDOW of 0 (at most a few hundred)."""
    import sympy as sp

    k = image.lamda.variables[0]
    expression = image.lamda.expr
    slope, offset = sp.diff(expression, k), expression.subs(k, 0)
    try:
        step, start = float(slope), float(offset)
        low = math.ceil((-SOLUTION_WINDOW - start) / step) if step else 0
        high = math.floor((SOLUTION_WINDOW - start) / step) if step else 0
        candidates = range(min(low, high) - 1, max(low, high) + 2)
    except (TypeError, ValueError, OverflowError):
        # Not affine in k: sample it, which finds every root in the window only when the
        # values leave it at both ends (sqrt(n*pi) does; 1/(n*pi) piles up at 0 and does not)
        if any(_inside_window(expression.subs(k, end)) for end in (-_SAMPLED_K, _SAMPLED_K)):
            raise _IncompleteSolution()
        candidates = range(-_SAMPLED_K, _SAMPLED_K + 1)
    if len(candidates) > 4 * MAX_LISTED_ROOTS:
        middle = -start / step if step else 0.0  # the k nearest 0
        candidates = range(int(middle) - 2 * MAX_LISTED_ROOTS, int(middle) + 2 * MAX_LISTED_ROOTS + 1)
    return [value for value in (expression.subs(k, index) for index in candidates) if _inside_window(value)]


_SAMPLED_K = 50


class _IncompleteSolution(Exception):
    """The roots in the window cannot all be listed."""


def _inside_window(value: Any) -> bool:
    """True for a real value within SOLUTION_WINDOW of 0."""
    try:
        number = complex(value.evalf())
    except (TypeError, ValueError):
        return False
    return abs(number.imag) < 1e-12 and abs(number.real) <= SOLUTION_WINDOW * (1 + 1e-12)


def _nearest_roots(roots: Sequence[Any]) -> Tuple[List[Any], str]:
    """The roots in the largest window [-w, w] (w <= 2*pi) holding at most MAX_LISTED_ROOTS, and w's text."""
    import sympy as sp

    if len(roots) <= MAX_LISTED_ROOTS:
        return list(roots), "2*pi"
    by_size = sorted(roots, key=lambda root: abs(float(root.evalf())))
    sizes = [abs(float(root.evalf())) for root in by_size]
    count = MAX_LISTED_ROOTS
    while count > 0 and sizes[count] - sizes[count - 1] <= 1e-12 * max(1.0, sizes[count]):
        count -= 1  # keep +r and -r together
    kept = by_size[:count]
    window = sp.nsimplify(sizes[count - 1], [sp.pi]) if count else sp.S.Zero
    return kept, to_text(window)


def _distinct(values: Sequence[Any]) -> List[Any]:
    """values without repeats, compared by their decimal values."""
    unique: List[Any] = []
    seen: set = set()
    for value in values:
        try:
            number = complex(value.evalf())
            key: Any = (round(number.real, 10), round(number.imag, 10))
        except (TypeError, ValueError):
            key = value
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def _sorted_roots(roots: Sequence[Any]) -> List[Any]:
    """Real roots by value first, then complex ones by real and imaginary part."""

    def key(root: Any) -> Tuple[int, float, float]:
        try:
            number = complex(root.evalf())
        except (TypeError, ValueError):
            return (2, 0.0, 0.0)
        is_complex = 1 if abs(number.imag) > 1e-12 * max(1.0, abs(number.real)) else 0
        return (is_complex, number.real, number.imag)

    return sorted(roots, key=key)


# ----- inequalities ----------------------------------------------------------------------

_RELATION_OPERATORS: Dict[type, str] = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
# A periodic inequality without an interval is solved on [-PERIODIC_WINDOW, PERIODIC_WINDOW].
PERIODIC_WINDOW = "2*pi"
# A periodic inequality is solved on one period and shifted across the window; when that
# takes more copies than this, the answer covers one period.
MAX_PERIOD_COPIES = 50
# Interval brackets: lower open?, upper open?
_BOUNDS = {"[]": (False, False), "[)": (False, True), "(]": (True, False), "()": (True, True)}


def parse_relation(text: str) -> List[Any]:
    """The SymPy relations (real symbols) of an inequality, or of a chain such as 1 < 2x+1 <= 5 (two relations).

    Raises:
        ExpressionError: unless the text is one or two comparisons with <, <=, > or >= that
            point the same way, between expressions parse_expression accepts.
    """
    import sympy as sp

    try:
        tree = ast.parse(_prepared_text(str(text)).strip(), mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"Cannot read the inequality {text!r}: {error.msg}.") from None
    node = tree.body
    if not isinstance(node, ast.Compare) or not all(type(op) in _RELATION_OPERATORS for op in node.ops):
        raise ExpressionError("An inequality compares two expressions with <, <=, > or >=.")
    operators = [_RELATION_OPERATORS[type(op)] for op in node.ops]
    if len(operators) > 2 or len({operator[0] for operator in operators}) > 1:
        raise ExpressionError("A chain of inequalities has two signs pointing the same way, e.g. 1 < 2x+1 <= 5.")
    builder = _Builder(real=True)
    operands = [builder.build(node.left)] + [builder.build(operand) for operand in node.comparators]
    relation = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}
    return [
        relation[operator](operands[index], operands[index + 1], evaluate=False)
        for index, operator in enumerate(operators)
    ]


def _interval(lower: Optional[str], upper: Optional[str], bounds: Optional[str]) -> Any:
    """The Interval from lower to upper (bounds "[]", "[)", "(]" or "()"), or None when neither end is given."""
    import sympy as sp

    if lower is None and upper is None:
        return None
    if lower is None or upper is None:
        raise ExpressionError("An interval needs both ends.")
    if (bounds or "[]") not in _BOUNDS:
        raise ExpressionError("The interval's brackets must be [], [), (] or ().")
    low, high = parse_expression(lower, real=True), parse_expression(upper, real=True)
    if low.free_symbols or high.free_symbols or not (_is_real_value(low) and _is_real_value(high)):
        raise ExpressionError("The interval's ends must be real numbers.")
    if not bool(low < high):
        raise ExpressionError("The interval's lower end must be below its upper end.")
    left_open, right_open = _BOUNDS[bounds or "[]"]
    return sp.Interval(low, high, left_open, right_open)


def compute_solve_inequality(
    inequality: str,
    variable: str,
    lower: Optional[str] = None,
    upper: Optional[str] = None,
    bounds: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The real solution set of an inequality in one variable, or None when SymPy has no usable answer.

    Returns {"intervals": [{"lower", "upper", "lower_closed", "upper_closed"}, ...],
    "window": text or None, "period": text or None}: the intervals in increasing order (an
    isolated point is an interval with equal closed ends), every end as text in the client's
    notation ("Infinity" for an infinite end). With lower and upper the set is restricted to
    that interval (bounds as in _interval), which "window" repeats. A periodic inequality
    (sin(x) > 1/2) is solved on one period [0, T) and the copies shifted across the interval,
    or across [-2*pi, 2*pi] without one; "period" is then T, and when the window would need
    more than MAX_PERIOD_COPIES copies "window" is one period, [a, a + T) (a is the
    interval's lower end, or 0).
    """
    import sympy as sp
    from sympy.calculus.util import periodicity

    x = _symbol(variable, real=True)
    relations = parse_relation(inequality)
    decimal = any(_DECIMAL_LITERAL.search(str(text)) for text in (inequality, lower, upper) if text is not None)
    for relation in relations:
        names = relation.free_symbols - {x}
        if names:
            raise ExpressionError(f"The inequality has other variables besides {variable}: {_names(names)}.")
    if not any(relation.has(x) for relation in relations):
        raise ExpressionError(f"The inequality does not contain {variable}.")
    domain = _interval(lower, upper, bounds)
    periods = {periodicity(relation.lhs - relation.rhs, x) for relation in relations}
    if periods == {None} or periods == {sp.S.Zero}:
        solution = _solve_relations(relations, x, domain if domain is not None else sp.S.Reals)
        window = domain
        period = None
    elif len(periods) == 1 and None not in periods:
        period = periods.pop()
        window = domain if domain is not None else _default_window()
        solution, window = _periodic_solution_set(relations, x, period, window, domain is not None)
    else:
        return None  # periodic and non-periodic parts together: left to the numeric sign chart
    if solution is None:
        return None
    intervals = _set_intervals(solution, decimal)
    if intervals is None:
        return None
    return {
        "intervals": intervals,
        "window": _interval_text(window, decimal) if window is not None else None,
        "period": to_text(period) if period is not None else None,
    }


def _names(symbols: Any) -> str:
    return ", ".join(sorted(str(symbol) for symbol in symbols))


def _default_window() -> Any:
    import sympy as sp

    bound = parse_expression(PERIODIC_WINDOW)
    return sp.Interval(-bound, bound)


def _solve_relations(relations: Sequence[Any], x: Any, domain: Any) -> Any:
    """The intersection of each relation's solution set in domain, or None when SymPy cannot solve one."""
    solution = domain
    for relation in relations:
        part = _solve_relation(relation, x, domain)
        if part is None:
            return None
        solution = solution.intersect(part)
    return solution


def _solve_relation(relation: Any, x: Any, domain: Any) -> Any:
    """solve_univariate_inequality's set in domain, or reduce_inequalities' when it cannot; None when neither can."""
    from sympy.solvers.inequalities import reduce_inequalities, solve_univariate_inequality

    try:
        return solve_univariate_inequality(relation, x, relational=False, domain=domain)
    except NotImplementedError:
        pass
    try:
        return reduce_inequalities(relation, x).as_set().intersect(domain)
    except NotImplementedError:
        return None


def _periodic_solution_set(relations: Sequence[Any], x: Any, period: Any, window: Any, given: bool) -> Tuple[Any, Any]:
    """(solution set in the window, the window) for relations with this period.

    SymPy answers a periodic inequality for one period, even on a larger domain
    (sin(x) > 1/2 gives (pi/6, 5*pi/6) on [-2*pi, 2*pi]), so the inequality is solved on
    [0, period) and the copies of that set are shifted across the window. The window shrinks
    to one period when it is unbounded or needs more than MAX_PERIOD_COPIES copies.
    """
    import sympy as sp

    base = _solve_relations(relations, x, sp.Interval.Ropen(0, period))
    if base is None or _set_intervals(base) is None:
        return None, window
    first, last = _copy_range(window, period)
    if first is None or last - first + 1 > MAX_PERIOD_COPIES:
        start = window.inf if given and window.inf.is_finite else sp.S.Zero
        window = sp.Interval.Ropen(start, start + period).intersect(window)
        first, last = _copy_range(window, period)
    assert first is not None and last is not None
    shifted = [sp.imageset(sp.Lambda(x, x + k * period), base) for k in range(first, last + 1)]
    return sp.Union(*shifted).intersect(window), window


def _copy_range(window: Any, period: Any) -> Tuple[Optional[int], int]:
    """(first, last) k such that the copies [k*T, (k+1)*T) cover the window; (None, 0) when it is unbounded."""
    import sympy as sp

    if not (window.inf.is_finite and window.sup.is_finite):
        return None, 0
    # Exact: with float division, pi / (pi/3) can come out just below 3 and lose the last copy
    return int(sp.floor(window.inf / period)), int(sp.floor(window.sup / period))


def _set_intervals(solution: Any, decimal: bool = False) -> Optional[List[Dict[str, Any]]]:
    """A union of intervals and points as sorted interval records, or None for any other set (ConditionSet, ...).

    With decimal (the input had decimals), terminating fractions are written as decimals (_end_text).
    """
    import sympy as sp

    if solution == sp.S.EmptySet:
        return []
    parts = list(solution.args) if isinstance(solution, sp.Union) else [solution]
    records: List[Tuple[float, Dict[str, Any]]] = []
    for part in parts:
        if part == sp.S.Reals:
            part = sp.Interval(-sp.oo, sp.oo)
        if isinstance(part, sp.Interval):
            ends = (part.start, part.end)
            closed = (not part.left_open, not part.right_open)
        elif isinstance(part, sp.FiniteSet):
            for point in part:
                record = _interval_record(point, point, True, True, decimal)
                if record is None:
                    return None
                records.append(record)
            continue
        else:
            return None
        record = _interval_record(ends[0], ends[1], closed[0], closed[1], decimal)
        if record is None:
            return None
        records.append(record)
    return [record for _, record in sorted(records, key=lambda item: item[0])]


def _interval_record(
    low: Any, high: Any, low_closed: bool, high_closed: bool, decimal: bool = False
) -> Optional[Tuple[float, Dict[str, Any]]]:
    """(sort key, {"lower", "upper", "lower_closed", "upper_closed"}), or None when an end is not a real number."""
    texts = []
    for end in (low, high):
        if not _is_usable(end) or not _is_real_value(end) or end.free_symbols:
            return None
        texts.append(_end_text(end, decimal))
    try:
        key = float(low)
    except (TypeError, ValueError):
        return None
    record = {"lower": texts[0], "upper": texts[1], "lower_closed": low_closed, "upper_closed": high_closed}
    return key, record


def _interval_text(interval: Any, decimal: bool = False) -> str:
    """[a, b), (a, b], ... in the client's notation."""
    left = "(" if interval.left_open else "["
    right = ")" if interval.right_open else "]"
    return f"{left}{_end_text(interval.start, decimal)}, {_end_text(interval.end, decimal)}{right}"


# A decimal number in the input: 1.5, .5, 2., 1e-300 (not part of a name such as x1)
_DECIMAL_LITERAL = re.compile(r"(?<![A-Za-z_\d.])(?:\d+\.\d*|\.\d+|\d+[eE][-+]?\d+)")
# A decimal end is written in scientific notation below this exponent (1e-7, not 0.0000001)
MIN_PLAIN_EXPONENT = -6


def _end_text(end: Any, decimal: bool) -> str:
    """An interval end as text: with decimal, a fraction whose decimal expansion ends (p/q with
    q = 2^a 5^b) as that decimal (123456789/100000000 as 1.23456789, 10^-300 as 1e-300);
    otherwise as to_text writes the displayed root."""
    import sympy as sp

    if decimal and isinstance(end, sp.Rational) and not end.is_Integer:
        text = _decimal_text(int(end.p), int(end.q))
        if text is not None:
            return text
    return to_text(_displayed_root(end))


def _decimal_text(numerator: int, denominator: int) -> Optional[str]:
    """numerator / denominator in decimals, or None when its expansion does not end."""
    rest, places = denominator, 0
    for prime in (2, 5):
        count = 0
        while rest % prime == 0:
            rest //= prime
            count += 1
        places = max(places, count)
    if rest != 1:
        return None
    sign = "-" if numerator < 0 else ""
    digits = str(abs(numerator) * (10**places // denominator))  # the value is digits * 10^-places
    exponent = len(digits) - 1 - places
    if exponent < MIN_PLAIN_EXPONENT:
        mantissa = digits[0] + ("." + digits[1:].rstrip("0") if digits[1:].rstrip("0") else "")
        return f"{sign}{mantissa}e{exponent}"
    if len(digits) <= places:
        digits = "0" * (places - len(digits) + 1) + digits
    return f"{sign}{digits[: len(digits) - places]}.{digits[len(digits) - places :]}"


# ----- equivalence ----------------------------------------------------------------------

# simplify runs only on a difference with at most this many operations (count_ops).
MAX_SIMPLIFY_OPS = 300
# A difference is returned only when its text has at most this many characters.
MAX_DIFFERENCE_LENGTH = 80
# A constant difference smaller than this (in 30-digit arithmetic) may be 0 SymPy cannot show.
ZERO_DIFFERENCE = 1e-25


def compute_equivalent(first: str, second: str) -> Dict[str, Optional[str]]:
    """Whether second - first is 0 for every real value of the variables (where both are defined).

    Returns {"verdict": "equivalent" | "not_equivalent" | "unknown", "difference": text or None}:

    - equivalent when cancel(together(second - first)) or expand(second - first) is 0, or
      else (for a difference of at most MAX_SIMPLIFY_OPS operations) simplify gives 0;
    - not_equivalent when the difference is a rational function with rational coefficients
      that does not cancel to 0, or simplifies to a nonzero number, with that difference;
    - unknown otherwise, with the simplified difference when its text has at most
      MAX_DIFFERENCE_LENGTH characters (x - abs(x) for x against sqrt(x^2)).

    The variables are real, so sqrt(x^2) is abs(x); a step that holds only where both sides
    are defined (log(x^2) and 2*log(x)) comes out unknown, which the client's numeric check
    decides.
    """
    import sympy as sp

    difference = parse_expression(second, real=True) - parse_expression(first, real=True)
    if difference == 0:
        return {"verdict": "equivalent", "difference": None}
    cancelled = sp.cancel(sp.together(difference))
    if cancelled == 0 or sp.expand(difference) == 0:
        return {"verdict": "equivalent", "difference": None}
    if _is_rational_over_q(cancelled) and _is_usable(cancelled):
        return {"verdict": "not_equivalent", "difference": _difference_text(cancelled)}
    simplified = sp.simplify(difference) if sp.count_ops(difference) <= MAX_SIMPLIFY_OPS else None
    if simplified is not None and simplified == 0:
        return {"verdict": "equivalent", "difference": None}
    shortest = cancelled if simplified is None else _shorter(cancelled, simplified)
    if not _is_usable(shortest):
        return {"verdict": "unknown", "difference": None}
    if shortest.is_number and _is_nonzero_number(shortest):
        return {"verdict": "not_equivalent", "difference": _difference_text(shortest)}
    return {"verdict": "unknown", "difference": _difference_text(shortest)}


def _is_rational_over_q(expression: Any) -> bool:
    """True for a ratio of polynomials in its variables with rational coefficients (x/2 - 1/x),
    where cancel is canonical: a nonzero result is a nonzero function."""
    import sympy as sp

    symbols = sorted(expression.free_symbols, key=str)
    if not symbols or expression.has(sp.I) or expression.atoms(sp.NumberSymbol):
        return False
    if not all(number.is_Rational for number in expression.atoms(sp.Number)):
        return False
    numerator, denominator = sp.fraction(expression)
    return bool(numerator.is_polynomial(*symbols) and denominator.is_polynomial(*symbols))


def _is_nonzero_number(value: Any) -> bool:
    """True when a number's 30-digit value is clearly not 0 (sqrt(2) - 1.41421356, not a zero simplify missed)."""
    import sympy as sp

    try:
        approximate = complex(sp.N(value, 30))
    except (TypeError, ValueError):
        return False
    return abs(approximate) > ZERO_DIFFERENCE


def _difference_text(difference: Any) -> Optional[str]:
    text = to_text(difference)
    return text if len(text) <= MAX_DIFFERENCE_LENGTH else None


# ----- Taylor polynomials ----------------------------------------------------------------

# The highest order taylor_polynomial gives: higher orders run past the worker's time limit
# on ordinary products such as exp(sin(x))*ln(1+x).
MAX_TAYLOR_ORDER = 10
# The coefficients are computed this many orders beyond the polynomial, for its first
# omitted term (where the series is needed, 5 takes order 10 of abs(x)*e^x past the time limit).
TAYLOR_LOOKAHEAD = 3
# The derivatives are given up for the series once one has more operations than this: they
# grow about tenfold per step on roots of quotients (sqrt((1-x)/(1+x))), and steadily on
# e^(-x^2)/(1+x), whose series is quicker; x^x at 2 needs about 600.
MAX_DERIVATIVE_OPS = 1000
# The derivatives are also given up after this many seconds: those of e^(x*sin(x)) at 0 stay
# under the operation limit but take about 4 s to order 13, its series 0.3 s.
DERIVATIVE_SECONDS = 2.0
# A polynomial up to this degree has its first omitted term found beyond the lookahead.
MAX_TAIL_DEGREE = 100


def compute_taylor(expression: str, variable: str, point: str, order: str) -> Dict[str, Any]:
    """The Taylor polynomial of expression at variable = point, up to (variable - point)^order.

    Returns {"polynomial", "center", "base", "order", "next_term", "exact", "flat",
    "one_sided", "lookahead"}: the polynomial in ascending powers of (x - a) ("1 + x + x^2/2",
    "1 + (x-1) - (x-1)^2/2"), a's text, the base of the powers (x or (x-1)), the order, the first nonzero omitted term up to
    (x - a)^(order + TAYLOR_LOOKAHEAD) (None when there is none), whether the function is a
    polynomial of degree at most order (the polynomial is the function), whether every
    coefficient up to that power is 0 although the function is not 0 (exp(-1/x^2) at 0),
    whether the two sides' series differ beyond the order (abs(x)^3 at 0 to order 2), and
    order + TAYLOR_LOOKAHEAD.

    The coefficients come from the derivatives at a when the series would be slow (a point
    other than 0, or an exponential of a non-polynomial), the derivatives are all finite
    real numbers and the function has no abs, sign, floor or the like; otherwise from the
    series on both sides of a, because a one-sided series hides a corner or a jump (abs(x)
    is x right of 0 and -x left of it). See _two_sided_coefficients.

    Raises:
        ExpressionError: for other variables, a point that is not a real number, an order
            outside 0..MAX_TAYLOR_ORDER, and a function that is not smooth at the point
            (a corner, a jump, a pole, a fractional power or a logarithm there).
    """
    import sympy as sp

    x = _symbol(variable, real=True)
    function = parse_expression(expression, real=True)
    others = function.free_symbols - {x}
    if others:
        raise ExpressionError(f"The function has other variables besides {variable}: {_names(others)}.")
    degree = _taylor_order(order)
    center = parse_expression(point, real=True)
    if center.free_symbols or center in (sp.oo, -sp.oo) or not _is_usable(center) or not _is_real_value(center):
        raise ExpressionError("The point must be a real number, such as 0, 1 or pi/4.")
    count = degree + 1 + TAYLOR_LOOKAHEAD
    right, left = _two_sided_coefficients(function, x, center, degree, count)
    if right is None or left is None or any(sp.simplify(right[k] - left[k]) != 0 for k in range(degree + 1)):
        raise ExpressionError(
            f"{expression} is not smooth at {variable} = {to_text(center)} (a corner, jump, pole, "
            "fractional power or logarithm there), so it has no Taylor polynomial at that point."
        )
    one_sided = any(sp.simplify(right[k] - left[k]) != 0 for k in range(degree + 1, len(left)))
    right = right + _polynomial_tail(function, x, center, count)
    base = _power_base(variable, center)
    later = [k for k in range(degree + 1, len(right)) if right[k] != 0]
    exact = bool(function.is_polynomial(x) and sp.degree(function, x) <= degree)
    return {
        "polynomial": _joined_terms([to_text(right[k] * base**k) for k in range(degree + 1) if right[k] != 0]),
        "center": to_text(center),
        "base": str(base),
        "order": degree,
        "next_term": to_text(right[later[0]] * base ** later[0]) if later and not one_sided else None,
        "exact": exact,
        "flat": not exact
        and not later
        and all(right[k] == 0 for k in range(len(right)))
        and _nonzero_near(function, x, center),
        "one_sided": one_sided,
        "lookahead": degree + TAYLOR_LOOKAHEAD,
    }


def _nonzero_near(function: Any, x: Any, center: Any) -> bool:
    """Whether the function is not 0 beside the point: floor(x) at 1/2 and
    sin(x)^2 + cos(x)^2 - 1 are 0 there, exp(-1/x^2) at 0 is not."""
    import sympy as sp

    if sp.simplify(function) == 0:
        return False
    step = sp.Rational(1, 1000)
    return any(sp.N(function.subs(x, center + side * step)) != 0 for side in (1, -1))


def _taylor_order(order: str) -> int:
    text = str(order).strip()
    if not text.isdigit() or int(text) > MAX_TAYLOR_ORDER:
        raise ExpressionError(f"The order must be a whole number from 0 to {MAX_TAYLOR_ORDER}.")
    return int(text)


def _two_sided_coefficients(
    function: Any, x: Any, center: Any, degree: int, count: int
) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    """The coefficients of (x - a)^0 .. (x - a)^(count-1) from the right of a and from the
    left (the left ones up to the order only, unless the function may break off beyond it).

    A function without abs, sign, floor and the like whose derivatives at a are all finite
    real numbers takes its coefficients from them, one list for both sides, where SymPy's
    series is slow: at a point other than 0 (e^(sin(x)) at pi/3), and for an exponential of
    a non-polynomial (e^(e^x), e^(sin(x))) at 0 too, where the derivatives may grow
    twice as large. Otherwise, as for most functions at 0 (where the series is quick, and
    the derivatives of e^(-x^2)/(1-x) grow steadily) or abs(x)^3, each side comes from its
    series."""
    import sympy as sp

    breaking = function.has(sp.Abs, sp.sign, sp.floor, sp.ceiling, sp.Min, sp.Max, sp.Mod)
    exponential = _has_exponential_of_non_polynomial(function, x)
    if not breaking and (center != 0 or exponential):
        limit = 2 * MAX_DERIVATIVE_OPS if exponential else MAX_DERIVATIVE_OPS
        coefficients = _derivative_coefficients(function, x, center, count, limit)
        if coefficients is not None:
            return coefficients, coefficients
    h = sp.Dummy("h", real=True)
    # e^(-(1+h)^2) as e^(-1) e^(-2h) e^(-h^2): SymPy's series of that is much quicker
    shifted = function.subs(x, center + h).replace(
        lambda part: isinstance(part, sp.exp), lambda part: sp.expand(sp.exp(sp.expand(part.args[0])))
    )
    right = _series_coefficients(shifted, h, count, "+")
    # The left side to the order only, unless the function has a part that may break off
    # beyond it (abs(x)^3 at 0 is smooth to order 2 only); the series is the slow step.
    left = _series_coefficients(shifted, h, count if breaking else degree + 1, "-")
    return right, left


def _has_exponential_of_non_polynomial(function: Any, x: Any) -> bool:
    """Whether function has e^g or b^g with g not a polynomial in x (e^(sin(x)), 2^(1/x))."""
    import sympy as sp

    if any(not part.args[0].is_polynomial(x) for part in function.atoms(sp.exp)):
        return True
    return any(part.exp.has(x) and not part.exp.is_polynomial(x) for part in function.atoms(sp.Pow))


def _derivative_coefficients(function: Any, x: Any, center: Any, count: int, limit: int) -> Optional[List[Any]]:
    """f^(k)(a)/k! for k = 0 .. count-1, or None when one of them is not a finite real
    number (0/0 in sin(x)/x at 0, a pole, sqrt(x) at 0), a derivative has more than limit
    operations, or they take more than DERIVATIVE_SECONDS."""
    import time

    import sympy as sp

    deadline = time.monotonic() + DERIVATIVE_SECONDS
    coefficients: List[Any] = []
    derivative = function
    for k in range(count):
        if k:
            derivative = sp.expand_mul(sp.diff(derivative, x))
            if sp.count_ops(derivative) > limit or time.monotonic() > deadline:
                return None
        value = derivative.subs(x, center)
        if value.has(sp.nan, sp.zoo, sp.oo, -sp.oo) or value.free_symbols:
            return None
        if not _is_usable(value) or not _is_real_value(value):
            return None
        coefficients.append(sp.expand(value / sp.factorial(k)))
    return coefficients


def _polynomial_tail(function: Any, x: Any, center: Any, count: int) -> List[Any]:
    """For a polynomial of degree at most MAX_TAIL_DEGREE, its coefficients of
    (x - a)^count .. (x - a)^degree, so the first omitted term of x^14 to order 10 is found;
    [] for anything else."""
    import sympy as sp

    if not function.is_polynomial(x):
        return []
    polynomial = sp.Poly(function, x)
    if polynomial.degree() < count or polynomial.degree() > MAX_TAIL_DEGREE:
        return []
    h = sp.Dummy("h")
    shifted = sp.Poly(sp.expand(function.subs(x, center + h)), h)
    return [sp.expand(shifted.coeff_monomial(h**k)) for k in range(count, polynomial.degree() + 1)]


def _series_coefficients(expression: Any, h: Any, count: int, direction: str) -> Optional[List[Any]]:
    """The coefficients of h^0 .. h^(count-1) in expression's series at h = 0 from one side,
    or None when the series is not a polynomial in h with real numbers as coefficients
    (sqrt(h), 1/h, log(h)) or there is a pole."""
    import sympy as sp

    try:
        series = sp.expand(expression.series(h, 0, count, dir=direction).removeO())
    except sp.PoleError:
        return None
    if not series.is_polynomial(h):
        return None
    polynomial = sp.Poly(series, h)
    coefficients = [polynomial.coeff_monomial(h**k) for k in range(count)]
    for coefficient in coefficients:
        if coefficient.free_symbols or not _is_usable(coefficient) or not _is_real_value(coefficient):
            return None
    return coefficients


def _power_base(variable: str, center: Any) -> Any:
    """The symbol printed as the base of the powers: x at 0, else (x-a), (x+1) or (x-(1-sqrt(2)))."""
    import sympy as sp

    if center == 0:
        return sp.Symbol(variable)
    if center.is_Add:
        return sp.Symbol(f"({variable}-({to_text(center)}))")
    text = to_text(center)
    return sp.Symbol(f"({variable}+{text[1:]})" if text.startswith("-") else f"({variable}-{text})")


def _joined_terms(texts: Sequence[str]) -> str:
    """Terms joined as a sum, "1 + x - x^3/6"; "0" for none."""
    if not texts:
        return "0"
    joined = texts[0]
    for text in texts[1:]:
        joined += f" - {text[1:]}" if text.startswith("-") else f" + {text}"
    return joined


# ----- partial fractions -----------------------------------------------------------------

# Partial fractions are found for denominators of at most this degree.
MAX_APART_DEGREE = 10


def compute_apart(expression: str, variable: str) -> Dict[str, Any]:
    """The partial fractions of a ratio of polynomials in variable, over the rationals.

    Returns {"fractions": text, "unchanged": bool}: the polynomial part first, then the
    fractions by their denominators ("x + 1 + 2/(x-1)", "1/(x-1) - 1/(x+1)"); irreducible
    quadratics stay as they are, as calculators leave them, and "unchanged" says the result is
    a single fraction (1/(x^2-2) over the rationals). {"fractions": None} when the expression
    is not a ratio of polynomials in variable alone with variable in the denominator
    (x^2 + 2x, sin(x)/x, (x+1)/(x*y)): the client expands it as usual. {"fractions": None,
    "refusal": text} for a denominator of degree above MAX_APART_DEGREE.
    """
    import sympy as sp

    x = _symbol(variable, real=True)
    function = parse_expression(expression, real=True)
    if function.free_symbols != {x} or not function.is_rational_function(x):
        return {"fractions": None, "unchanged": False}
    denominator = sp.denom(sp.cancel(sp.together(function)))
    if not denominator.has(x):
        return {"fractions": None, "unchanged": False}
    if sp.degree(denominator, x) > MAX_APART_DEGREE:
        refusal = f"partial fractions are found for denominators of degree up to {MAX_APART_DEGREE}."
        return {"fractions": None, "unchanged": False, "refusal": refusal}
    terms = sp.Add.make_args(sp.apart(function, x))
    polynomial = sp.Add(*[term for term in terms if not sp.denom(term).has(x)])
    powers = sp.Poly(polynomial, x).terms() if polynomial != 0 else []
    fractions = sorted((term for term in terms if sp.denom(term).has(x)), key=lambda term: _fraction_key(term, x))
    texts = [to_text(coefficient * x**k) for (k,), coefficient in powers] + [_fraction_text(term) for term in fractions]
    return {"fractions": _joined_terms(texts), "unchanged": polynomial == 0 and len(fractions) == 1}


def _fraction_key(term: Any, x: Any) -> Tuple[Any, ...]:
    """Fractions ordered by their denominator's factor: its degree, then its monic
    coefficients by size (x, x - 1, x + 1, x - 2), then the power (1/x before 1/x^2)."""
    import sympy as sp

    _, factors = sp.factor_list(sp.denom(term), x)
    with_x = [(base, power) for base, power in factors if base.has(x)]
    if len(with_x) != 1:
        return (MAX_APART_DEGREE + 1, sp.default_sort_key(term))
    base, power = with_x[0]
    coefficients = [complex(sp.N(coefficient)) for coefficient in sp.Poly(base, x).monic().all_coeffs()[1:]]
    sizes = tuple((abs(value), value.real, value.imag) for value in coefficients)
    return (sp.degree(base, x), sizes, power)


def _fraction_text(term: Any) -> str:
    """A fraction as text, with 1/x^2 for x^(-2)."""
    if term.is_Pow and term.exp.is_Integer and term.exp < -1:
        return f"1/{to_text(term.base**-term.exp)}"
    return to_text(term)


# ----- expand and factor -----------------------------------------------------------------

# An expanded or factored form longer than this is refused ((x+1)^200 has 201 long terms).
MAX_REWRITE_LENGTH = 5000


def compute_expand(expression: str) -> Optional[str]:
    """The expanded form; None when it is not usable (NaN, an unevaluated limit).

    Raises:
        ExpressionError: when it is longer than MAX_REWRITE_LENGTH characters.
    """
    import sympy as sp

    return _rewritten(sp.expand(parse_expression(expression, real=True)), "expanded")


def compute_factor(expression: str) -> Optional[str]:
    """The factored form over the rationals, as compute_expand gives the expanded one."""
    import sympy as sp

    parsed = parse_expression(expression, real=True)
    if parsed.is_Rational and parsed != 0:
        return _factored_number(parsed)
    return _rewritten(sp.factor(parsed), "factored")


def _factored_number(number: Any) -> str:
    """360 as 2^3*3^2*5, -3/4 as -3/2^2, 1 as 1."""
    import sympy as sp

    def product(value: int) -> str:
        factors = sp.factorint(value)
        return "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(factors.items())) or "1"

    numerator, denominator = abs(int(number.p)), int(number.q)
    text = product(numerator)
    if denominator != 1:
        bottom = product(denominator)
        text = f"{text}/({bottom})" if "*" in bottom else f"{text}/{bottom}"
    return f"-{text}" if number < 0 else text


def _rewritten(result: Any, form: str) -> Optional[str]:
    if not _is_usable(result):
        return None
    text = to_text(result)
    if len(text) > MAX_REWRITE_LENGTH:
        raise ExpressionError(f"The {form} form has more than {MAX_REWRITE_LENGTH} characters.")
    return text


# ----- systems of equations --------------------------------------------------------------

# A system has at most this many equations and unknowns.
MAX_SYSTEM_EQUATIONS = 10
MAX_SYSTEM_UNKNOWNS = 10
# The equations of a system travel as one argument, each at most MAX_EXPRESSION_LENGTH long.
MAX_SYSTEM_TEXT_LENGTH = MAX_SYSTEM_EQUATIONS * (MAX_EXPRESSION_LENGTH + 1)


def compute_solve_system(equations: str, variables: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The solutions of a system: equations separated by ";", each lhs = rhs or an expression = 0.

    The unknowns are variables (comma-separated), else x, y, z and t where the equations use
    them, plus other names (sorted) while there are fewer unknowns than equations; the other
    names are parameters. Returns, for a linear system: {"none": True} when it is inconsistent,
    {"solutions": [one {unknown: value}]} when the solution is unique, or {"infinite": True,
    "general": {unknown: value in the free unknowns}, "free": [names], "rank": r} when the
    equations are dependent. For any other system of ratios of polynomials: {"solutions":
    [...], "complex": n}, the real solutions sorted by their values (exact when short, else
    decimals) and the number of non-real ones left out, or {"infinite": True, "curves": [{unknown:
    value}, ...], "free": [names]} when the solutions form curves. None for other systems
    (sin(x) = y, where solve lists principal values only) and when SymPy cannot list the
    solutions.
    """
    import sympy as sp

    texts = str(equations).split(";")
    if len(texts) > MAX_SYSTEM_EQUATIONS:
        raise ExpressionError(f"A system has at most {MAX_SYSTEM_EQUATIONS} equations.")
    differences = [_equation_difference(text) for text in texts]
    unknowns = _system_unknowns(differences, variables)
    for text, difference in zip(texts, differences):
        if difference.free_symbols and not difference.free_symbols & set(unknowns):
            # xy = 6 beside x + y = 5: not a contradiction, but a name the unknowns do not include
            names = ", ".join(sorted(str(name) for name in difference.free_symbols))
            raise ExpressionError(
                f"The equation {text.strip()} has none of the unknowns ({', '.join(map(str, unknowns))}), "
                f"only {names}; write products with * (x*y), or name the unknowns."
            )
    if all(_is_linear(difference, unknowns) for difference in differences):
        return _linear_system(sp.linsolve(differences, unknowns), unknowns)
    if not all(difference.is_rational_function(*unknowns) for difference in differences):
        return None  # sin(x) = y: solve lists principal values only, not every solution
    shaped = _shape_basis_solutions(differences, unknowns)
    if shaped is not None:
        return _nonlinear_system(shaped, unknowns)
    return _nonlinear_system(sp.solve(differences, unknowns, dict=True), unknowns)


_PREFERRED_UNKNOWNS = ("x", "y", "z", "t")


def _system_unknowns(differences: Sequence[Any], variables: Optional[str]) -> List[Any]:
    if variables:
        unknowns = [_symbol(name.strip(), real=False) for name in str(variables).split(",")]
    else:
        names = sorted(set().union(*(difference.free_symbols for difference in differences)), key=str)
        preferred = [name for name in names if str(name) in _PREFERRED_UNKNOWNS]
        others = [name for name in names if str(name) not in _PREFERRED_UNKNOWNS]
        unknowns = preferred + others[: max(0, len(differences) - len(preferred))]
    if not unknowns:
        raise ExpressionError("The equations have no unknowns.")
    if len(unknowns) > MAX_SYSTEM_UNKNOWNS:
        raise ExpressionError(f"A system has at most {MAX_SYSTEM_UNKNOWNS} unknowns.")
    return unknowns


def _shape_basis_solutions(differences: Sequence[Any], unknowns: Sequence[Any]) -> Optional[List[Dict[Any, Any]]]:
    """Every solution of a polynomial system whose lex Groebner basis has the shape u1 - g1(un),
    ..., h(un) (the usual case for finitely many solutions), from the roots of h; None for other
    systems. sp.solve takes minutes on a quintic intersection that this does at once."""
    import sympy as sp

    numerators = [sp.fraction(sp.together(difference))[0] for difference in differences]
    if any(difference.free_symbols - set(unknowns) for difference in differences):
        return None  # parameters
    if not all(numerator.is_polynomial(*unknowns) for numerator in numerators):
        return None
    basis = list(sp.groebner(numerators, *unknowns, order="lex").exprs)
    if basis == [1]:
        return []
    last = unknowns[-1]
    if len(basis) != len(unknowns) or basis[-1].free_symbols - {last} or not basis[-1].free_symbols:
        return None
    explicit: Dict[Any, Any] = {}
    for unknown, polynomial in zip(unknowns[:-1], basis[:-1]):
        coefficient = sp.diff(polynomial, unknown)
        if not coefficient.is_number or coefficient == 0 or polynomial.free_symbols - {unknown, last}:
            return None
        explicit[unknown] = sp.expand(unknown - polynomial / coefficient)
        if explicit[unknown].has(unknown):
            return None
    solutions = []
    for root in _polynomial_roots(sp.Poly(basis[-1], last)):
        solution = {unknown: _value_at(value, last, root) for unknown, value in explicit.items()}
        solution[last] = root
        solutions.append(solution)
    denominators = [sp.fraction(sp.together(difference))[1] for difference in differences]
    return [solution for solution in solutions if all(_nonzero_at(d, solution) for d in denominators)]


def _polynomial_roots(polynomial: Any) -> List[Any]:
    """Every root of a polynomial: exact for factors of degree at most 2, else 20-digit decimals
    (sp.N of a CRootOf takes about half a second)."""
    import sympy as sp

    roots: List[Any] = []
    for factor, _ in polynomial.factor_list()[1]:
        if factor.degree() <= 2:
            roots.extend(sp.roots(factor, multiple=True))
        else:
            roots.extend(factor.nroots(n=20, maxsteps=200))
    return roots


def _value_at(value: Any, unknown: Any, root: Any) -> Any:
    import sympy as sp

    substituted = value.subs(unknown, root)
    if root.has(sp.Float):
        return sp.N(substituted, 20)
    return _shorter(substituted, sp.radsimp(sp.expand(substituted)))


def _nonzero_at(denominator: Any, solution: Dict[Any, Any]) -> bool:
    import sympy as sp

    if denominator.is_number:
        return bool(denominator != 0)
    value = complex(sp.N(denominator.subs(solution), 15))
    return abs(value) > 1e-12


def _equation_difference(text: str) -> Any:
    if not text.strip():
        raise ExpressionError("An equation of the system is empty.")
    sides = text.split("=")
    if len(sides) > 2:
        raise ExpressionError("An equation has at most one '='.")
    return parse_expression(sides[0]) - (parse_expression(sides[1]) if len(sides) == 2 else 0)


def _is_linear(difference: Any, unknowns: Sequence[Any]) -> bool:
    import sympy as sp

    if not difference.is_polynomial(*unknowns):
        return False
    return bool(difference == 0 or sp.Poly(difference, *unknowns).total_degree() <= 1)


def _linear_system(solution: Any, unknowns: Sequence[Any]) -> Dict[str, Any]:
    import sympy as sp

    if solution == sp.S.EmptySet:
        return {"none": True}
    values = list(solution)[0]
    free = sorted(set().union(*(sp.sympify(value).free_symbols for value in values)) & set(unknowns), key=str)
    if not free:
        return {"solutions": [{str(name): to_text(value) for name, value in zip(unknowns, values)}]}
    general = {str(name): to_text(value) for name, value in zip(unknowns, values) if value != name}
    return {"infinite": True, "general": general, "free": [str(name) for name in free], "rank": len(general)}


def _nonlinear_system(solutions: Any, unknowns: Sequence[Any]) -> Optional[Dict[str, Any]]:
    curves = _solution_curves(solutions, unknowns)
    if curves is not None:
        return curves
    found: List[Tuple[Tuple[float, ...], Dict[str, str]]] = []
    non_real = 0
    for solution in solutions:
        if set(solution) != set(unknowns) or any(solution[name].free_symbols for name in unknowns):
            return None
        real = _real_solution(solution, unknowns)
        if real is None:
            non_real += 1
        elif all(real[1] != texts for _, texts in found):
            found.append(real)
    found.sort(key=lambda item: item[0])
    return {"solutions": [texts for _, texts in found], "complex": non_real}


def _solution_curves(solutions: Any, unknowns: Sequence[Any]) -> Optional[Dict[str, Any]]:
    """{"infinite": True, "curves": ..., "free": ...} when every solution leaves some unknowns
    free (x^2 + y^2 = 1 twice gives x = ±sqrt(1 - y^2)), else None."""
    if not solutions:
        return None
    free: set = set()
    curves = []
    for solution in solutions:
        missing = set(unknowns) - set(solution)
        named = set().union(*(value.free_symbols for value in solution.values()))
        if not missing or named - set(unknowns) or not named <= missing:
            return None
        free |= missing
        curves.append({str(name): to_text(value) for name, value in solution.items()})
    return {"infinite": True, "curves": curves, "free": sorted(str(name) for name in free)}


def _real_solution(
    solution: Dict[Any, Any], unknowns: Sequence[Any]
) -> Optional[Tuple[Tuple[float, ...], Dict[str, str]]]:
    """(the values as floats, {unknown: text}) for a real solution, None for a non-real one.
    A value written with i or as a CRootOf is given in decimals."""
    import sympy as sp

    values: List[float] = []
    texts: Dict[str, str] = {}
    for name in unknowns:
        value = solution[name]
        approximate = sp.N(value, 15)
        real, imaginary = (float(part) for part in approximate.as_real_imag())
        if not math.isfinite(real) or abs(imaginary) > 1e-12 * max(1.0, abs(real)):
            return None
        shown = sp.re(approximate) if value.has(sp.I, sp.Float) else _displayed_root(value)
        values.append(real)
        texts[str(name)] = to_text(shown)
    return tuple(values), texts


# ----- sums ------------------------------------------------------------------------------

# A sum with numeric bounds and more terms than this is summed to a symbol N first, and
# the closed form evaluated at the upper bound: term by term, 1/k to 10^6 outlasts the
# worker's time limit and 2^k to 10^5 is too long an integer to print.
MAX_EXACT_TERMS = 1000
# The same for a term with other symbols beyond this many terms: SymPy takes 3 s to add
# k*x^k term by term up to 200.
MAX_SYMBOLIC_TERMS = 50
# A constant sum whose exact text is longer than this (1/k to 100, a 41-digit fraction) is
# given in decimals.
MAX_EXACT_SUM_LENGTH = 60
# A closed form is simplified only up to this many operations (count_ops): simplify takes
# 6 s on the 50 sines of sin(k*x) to 50, and 12 s on the 100 radicals of sqrt(k) to 100.
MAX_SUM_SIMPLIFY_OPS = 60
# A fraction longer than this comes with its decimal (6439880978201/512000000).
MAX_PLAIN_FRACTION_LENGTH = 12
# A closed form with symbols whose text is longer than this is not given.
MAX_SUM_FORM_LENGTH = 400
# A form SymPy gives that calculators lack (harmonic(n), zeta(3)) is named up to this length.
MAX_SYMPY_FORM_LENGTH = 80
_INFINITY_NAMES = ("Infinity", "inf", "oo")


def compute_summation(term: str, index: str, lower: str, upper: str) -> Dict[str, Any]:
    """The closed form of the sum of term for index = lower .. upper.

    The bounds are integers, Infinity (upper only), or c*n + d (c >= 1 and d integers) in
    one symbol n other than the index; a symbolic lower bound needs a symbolic or infinite
    upper one. The index is an integer and n an integer symbol, also in the term.

    Returns a dict with only the keys that apply:

    - "value": the closed form in the client's notation, with "decimal" (15 digits) when it
      is a real constant that is not rational, and "condition" (text such as
      "-1 < r < 1" or "r != 1"), "condition_symbol" and "condition_intervals" (records as
      solve_inequality gives them) when it holds for those values of a symbol only
      (a geometric sum with a symbolic ratio);
    - "value": "0" and "empty": True for numeric bounds with lower > upper (calculators'
      empty sum, not SymPy's Karr convention, which gives -29 for k^2 from 5 to 1);
    - "general" and "general_symbol" for more than MAX_EXACT_TERMS terms: the closed form
      in N, which "value" is at the upper bound, or which "decimal" is with "value": None and
      "evaluated": True for a constant closed form that is not a polynomial in N;
    - "value": None with "decimal" and "too_long": True for a constant whose text is longer
      than MAX_EXACT_SUM_LENGTH;
    - "diverges": True with "limit" ("Infinity", "-Infinity", or None when the partial
      sums do not approach a value) for an infinite sum;
    - "value": None with "converges": True (an infinite sum with no closed form that SymPy
      shows converges), or "reason": "no_closed_form" (with "sympy_form" when SymPy's form
      uses functions or constants calculators lack, harmonic(n) or zeta(3), and "decimal"
      when that form is a real constant), "too_many_terms" or "undefined_term" (a term is
      infinite or 0/0 in the range).

    A term with other symbols but not the index, summed to Infinity, is refused:
    summation(x, k, 1, Infinity) is most likely a typo for x^k.

    Raises:
        ExpressionError: for an index that is not a name, bounds of another form, or a term
            the parser refuses.
    """

    import sympy as sp

    k = _index_symbol(index)
    low, high, bound = _sum_bounds(lower, upper, index)
    names: Dict[str, Any] = {index: k}
    if bound is not None:
        names[str(bound)] = bound
    function = parse_expression(term, real=True, names=names)
    if high == sp.oo and k not in function.free_symbols and function.free_symbols - {bound}:
        raise ExpressionError(
            f"The term does not contain the index {index}, so the sum to Infinity has no value; "
            f"write the term in {index}, as in x^{index}."
        )
    if _undefined_in_range(function, k, low, high):
        return {"value": None, "reason": "undefined_term"}
    if low.is_Integer and high.is_Integer:
        if low > high:
            return {"value": "0", "empty": True}
        count = high - low + 1
        if count > MAX_EXACT_TERMS or (count > MAX_SYMBOLIC_TERMS and function.free_symbols - {k}):
            return _long_sum(function, k, low, high)
    closed = _closed_form(function, k, low, high, bound)
    if isinstance(closed, dict):
        return closed
    return _sum_record(closed[0], closed[1], bound, finite=bool(high.is_Integer))


def _index_symbol(index: str) -> Any:
    import sympy as sp

    name = str(index).strip()
    if not _is_name(name) or name in _INFINITY_NAMES or name in _functions():
        raise ExpressionError(f"The index must be a variable name such as k, not {index!r}.")
    return sp.Symbol(name, integer=True)


_BOUND_RULE = "The bounds must be whole numbers, a symbol such as n (or 2n, n-1), or Infinity for the upper bound."


def _sum_bounds(lower: str, upper: str, index: str) -> Tuple[Any, Any, Any]:
    """(lower, upper, the bound symbol or None), with the bound symbol an integer."""
    import sympy as sp

    low, high = parse_expression(lower, real=True), parse_expression(upper, real=True)
    symbols = low.free_symbols | high.free_symbols
    if any(str(symbol) == index for symbol in symbols):
        raise ExpressionError(f"The index {index} cannot appear in a bound.")
    if any(str(symbol).lower().startswith("inf") for symbol in symbols):
        raise ExpressionError(f"{_BOUND_RULE} Write an infinite upper bound as Infinity.")
    if len(symbols) > 1:
        raise ExpressionError(f"{_BOUND_RULE} The bounds have more than one symbol: {_names(symbols)}.")
    if low in (sp.oo, -sp.oo) or high == -sp.oo:
        raise ExpressionError(f"{_BOUND_RULE} The lower bound cannot be infinite.")
    if low.free_symbols and not (high.free_symbols or high == sp.oo):
        raise ExpressionError("A symbolic lower bound needs a symbolic or infinite upper bound, as in k from n to 2n.")
    bound = None
    if symbols:
        symbol = symbols.pop()
        bound = sp.Symbol(str(symbol), integer=True)
        low, high = low.subs(symbol, bound), high.subs(symbol, bound)
    for end in (low, high):
        if end != sp.oo and not _is_integer_bound(end, bound):
            raise ExpressionError(_BOUND_RULE)
    return low, high, bound


def _is_integer_bound(end: Any, bound: Any) -> bool:
    """Whether end is an integer, or c*n + d with integers c >= 1 and d."""
    import sympy as sp

    if not end.free_symbols:
        return bool(end.is_Integer)
    try:
        polynomial = sp.Poly(end, bound)
    except sp.PolynomialError:
        return False
    coefficients = polynomial.all_coeffs()
    return polynomial.degree() == 1 and all(c.is_Integer for c in coefficients) and coefficients[0] >= 1


def _undefined_in_range(function: Any, k: Any, low: Any, high: Any) -> bool:
    """Whether the term's denominator, a polynomial in the index alone, vanishes at an
    integer of the range (1/(k-3)^2 from 1 to Infinity, which SymPy sums as if it did not)."""
    import sympy as sp

    denominator = sp.denom(sp.together(function))
    if denominator.free_symbols != {k} or not denominator.is_polynomial(k) or not low.is_Integer:
        return False
    try:
        roots = sp.roots(sp.Poly(denominator, k), filter="Z")
    except (sp.PolynomialError, NotImplementedError):
        return False
    return any(root >= low and (not high.is_Integer or root <= high) for root in roots)


def _long_sum(function: Any, k: Any, low: Any, high: Any) -> Dict[str, Any]:
    """A sum of more than MAX_EXACT_TERMS terms through its closed form in a symbol N.

    The closed form is evaluated exactly at the upper bound when it is a polynomial in N
    or has other symbols; a constant otherwise gets its decimal only ("value": None,
    "decimal", "evaluated": True): 2^(N+1) at N = 10^9 takes seconds to compute exactly,
    and exp(-N) at N = 1000 is 0 times infinity in math.js."""
    import sympy as sp

    taken = {str(symbol) for symbol in function.free_symbols}
    name = next(candidate for candidate in ("N", "M", "L", "N1", "N2") if candidate not in taken)
    count = sp.Symbol(name, integer=True)
    closed = _closed_form(function, k, low, count, count)
    if isinstance(closed, dict):
        return {"value": None, "reason": "too_many_terms"}
    expression, condition = closed
    general = {"general": to_text(_sum_form(expression, count)), "general_symbol": name}
    if expression.free_symbols - {count} or expression.is_polynomial(count):
        try:
            record = _sum_record(expression.subs(count, high), condition, None, finite=True)
        except ValueError:  # an integer too long to print (x*2^k to 10^6)
            return {"value": None, "reason": "too_long"}
    else:
        value = expression.evalf(15, subs={count: high})
        if not _is_real_value(value):
            return {"value": None, "reason": "no_closed_form"}
        record = {"value": None, "decimal": _decimal(value), "evaluated": True}
    record.update(general)
    return record


def _closed_form(function: Any, k: Any, low: Any, high: Any, bound: Any) -> Any:
    """(closed form, condition or None) for the sum, or the record that says why there is none."""
    import sympy as sp

    result = sp.piecewise_fold(sp.summation(function, (k, low, high)))
    condition = None
    if result.has(sp.Piecewise):
        result, condition = _generic_branch(result)
        if result is None or (condition is not None and bound is not None and condition.has(bound)):
            return {"value": None, "reason": "no_closed_form"}
    if result.has(sp.Sum):  # before the test for oo, which an infinite Sum holds
        if high == sp.oo:
            convergent = _is_convergent(function, k, low)
            if convergent is False:
                return {"diverges": True, "limit": None}
            if convergent:
                return {"value": None, "converges": True}
        return {"value": None, "reason": "no_closed_form"}
    if high == sp.oo and result in (sp.oo, -sp.oo):
        return {"diverges": True, "limit": to_text(result)}
    if result.has(sp.zoo):
        return {"value": None, "reason": "undefined_term"}
    if result.has(sp.oo, -sp.oo, sp.nan):  # nan for 1/(16k^2-1) to Infinity, whose sum is 1/2 - pi/8
        return {"value": None, "reason": "no_closed_form"}
    if not _is_calculator_form(result):
        record: Dict[str, Any] = {"value": None, "reason": "no_closed_form"}
        text = to_text(result)
        if len(text) <= MAX_SYMPY_FORM_LENGTH:
            record["sympy_form"] = text
        if not result.free_symbols and _is_real_value(result):
            record["decimal"] = _decimal(result)  # zeta(3): exact to 15 digits, and checked in the browser
        return record
    return result, condition


def _generic_branch(result: Any) -> Tuple[Any, Any]:
    """The closed form of a Piecewise sum for the generic values of its symbol, with the
    condition under which it holds, or (None, None):

    - Piecewise((n+1, Eq(r, 1)), ((r^(n+1)-1)/(r-1), True)) gives the last branch, for r != 1;
    - Piecewise((1/(1-r), (r > -1) & (r < 1)), (Sum(...), True)) gives the first, for -1 < r < 1.
    """
    import sympy as sp

    if not isinstance(result, sp.Piecewise):
        return None, None  # a Piecewise inside a larger expression
    pieces = list(result.args)
    expression, condition = pieces[-1]
    if condition == sp.true and not expression.has(sp.Sum):
        earlier = [sp.Not(earlier) for _, earlier in pieces[:-1]]
        return expression, sp.And(*earlier) if earlier else None
    for expression, condition in pieces:
        if condition != sp.true and not isinstance(condition, sp.Eq) and not expression.has(sp.Sum):
            return expression, condition
    return None, None


def _is_convergent(function: Any, k: Any, low: Any) -> Optional[bool]:
    """SymPy's verdict on the infinite sum, None when it has none or cannot be trusted:
    it calls sin(k)/k and cos(k)/k divergent, and takes 10 s on abs(sin(k))/k^2."""
    import sympy as sp

    oscillating = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc, sp.Abs, sp.sign, sp.floor, sp.ceiling, sp.Mod)
    if any(applied.has(k) for applied in function.atoms(*oscillating)):
        return None
    try:
        verdict = sp.Sum(function, (k, low, sp.oo)).is_convergent()
    except Exception:  # SymPy's tests raise for terms they cannot classify
        return None
    if verdict == sp.true or verdict is True:
        return True
    if verdict == sp.false or verdict is False:
        return False
    return None


def _calculator_functions() -> Tuple[Any, ...]:
    """The SymPy function classes the client can evaluate: those the parser builds."""
    import sympy as sp

    table = {getattr(sp, name) for name in _functions() if isinstance(getattr(sp, name, None), type)}
    return tuple(table | {sp.log, sp.exp, sp.Abs, sp.sign, sp.ceiling, sp.Mod, sp.conjugate, sp.Min, sp.Max})


def _is_calculator_form(expression: Any) -> bool:
    """Whether every function in expression is one calculators have and every constant is pi or e."""
    import sympy as sp

    if not _is_usable(expression):
        return False
    functions = _calculator_functions()
    if not all(isinstance(applied, functions) for applied in expression.atoms(sp.Function)):
        return False  # harmonic, zeta, lerchphi, exp_polar, besseli, ...
    return all(constant in (sp.pi, sp.E) for constant in expression.atoms(sp.NumberSymbol))


def _sum_form(expression: Any, bound: Any) -> Any:
    """The shortest of the simplified closed form, its fraction with both signs flipped
    (1/(1-r), not -1/(r-1); e/(-1+e), not -e/(1-e)) and, for a closed form in the bound
    symbol, its factored form, which wins a tie (n*(n+1)*(2*n+1)/6, not n*(2*n^2+3*n+1)/6);
    only forms in calculator functions count. Without a bound symbol a polynomial stays expanded, as calculators
    give it (1 + x + ... + x^20, not a product of cyclotomic factors)."""
    import sympy as sp

    if sp.count_ops(expression) > MAX_SUM_SIMPLIFY_OPS:
        return expression  # simplify takes seconds on 50 sines
    simplified = sp.simplify(expression)
    if not _is_calculator_form(simplified):
        simplified = expression
    factored = sp.factor(simplified) if bound is not None and simplified.has(bound) else None
    candidates = [simplified]
    numerator, denominator = sp.fraction(sp.together(simplified))
    if denominator != 1:
        candidates.append((-numerator) / (-denominator))
    if factored is not None:
        candidates.append(factored)

    def rank(candidate: Any) -> Tuple[int, int, int]:
        """The shortest text, the factored form on a tie, then the fewest minus signs."""
        text = to_text(candidate)
        return len(text), 0 if candidate is factored else 1, text.count("-")

    return min((candidate for candidate in candidates if _is_calculator_form(candidate)), key=rank)


def _sum_record(expression: Any, condition: Any, bound: Any, finite: bool = False) -> Dict[str, Any]:
    """The response for a closed form (see compute_summation). The constant sum of finitely
    many terms is not simplified: simplify takes over 10 s on sums such as sqrt(k) or sin(k)
    to 100, and SymPy writes their terms out anyway."""
    constant = not expression.free_symbols
    if constant and _is_long_constant(expression):
        return _too_long(expression)
    form = expression if constant and finite else _sum_form(expression, bound)
    record: Dict[str, Any] = {"value": to_text(form)}
    if constant and len(record["value"]) > MAX_EXACT_SUM_LENGTH:
        return _too_long(form)
    if len(record["value"]) > MAX_SUM_FORM_LENGTH:
        return {"value": None, "reason": "too_long"}
    long_fraction = form.is_Rational and not form.is_Integer and len(record["value"]) > MAX_PLAIN_FRACTION_LENGTH
    if constant and (long_fraction or not form.is_Rational) and _is_real_value(form):
        record["decimal"] = _decimal(form)
    if condition is not None:
        described = _condition_record(condition)
        if described is None:
            return {"value": None, "reason": "no_closed_form"}
        record.update(described)
    return record


def _is_long_constant(value: Any) -> bool:
    """Whether a constant is too long to simplify or print exactly: a sum of hundreds of
    radicals (simplify takes over 10 s on sqrt(k) to 100), or a fraction of hundreds of digits."""
    import sympy as sp

    if isinstance(value, sp.Rational):
        return max(abs(int(value.p)).bit_length(), int(value.q).bit_length()) > 4 * MAX_EXACT_SUM_LENGTH
    return bool(sp.count_ops(value) > MAX_SUM_SIMPLIFY_OPS)


def _too_long(value: Any) -> Dict[str, Any]:
    if not _is_real_value(value):
        return {"value": None, "reason": "no_closed_form"}
    return {"value": None, "decimal": _decimal(value), "too_long": True}


def _decimal(value: Any) -> str:
    """value to 15 significant digits; in SymPy's notation (1.99800206661008e+30103) beyond a float's range."""
    import sympy as sp

    number = sp.N(value, 15)
    if math.isfinite(float(number)):
        return to_text(sp.Float(number, 15))
    return str(number)


def _condition_record(condition: Any) -> Optional[Dict[str, Any]]:
    """{"condition", "condition_symbol", "condition_intervals"} for a condition on one
    symbol whose set is one interval or the reals but one point; None for any other."""

    symbols = condition.free_symbols
    if len(symbols) != 1:
        return None
    symbol = next(iter(symbols))
    try:
        region = condition.as_set()
    except (NotImplementedError, TypeError, ValueError):
        return None
    text = _region_text(region, str(symbol))
    intervals = _set_intervals(region)
    if text is None or intervals is None:
        return None
    return {"condition": text, "condition_symbol": str(symbol), "condition_intervals": intervals}


def _region_text(region: Any, name: str) -> Optional[str]:
    """The condition as text: -1 < r < 1, -1 <= x < 1, r < 1 for an interval; r != -1 and
    r != 0 for the reals but finitely many points; x < -1 or x > 1 for other unions of
    intervals; None for any other set."""
    import sympy as sp

    parts = sorted(region.args, key=lambda part: float(part.inf)) if isinstance(region, sp.Union) else [region]
    if not all(isinstance(part, sp.Interval) for part in parts):
        return None
    removed = [first.sup for first, second in zip(parts, parts[1:]) if first.sup == second.inf]
    if len(parts) > 1 and len(removed) == len(parts) - 1 and parts[0].inf == -sp.oo and parts[-1].sup == sp.oo:
        return " and ".join(f"{name} != {to_text(point)}" for point in removed)
    return " or ".join(_condition_interval(part, name) for part in parts)


def _condition_interval(region: Any, name: str) -> str:
    """-1 < r < 1, -1 <= x < 1, r < 1, x >= 0 for one interval."""
    import sympy as sp

    left = "<" if region.left_open else "<="
    right = "<" if region.right_open else "<="
    if region.inf == -sp.oo:
        return f"{name} {right} {to_text(region.sup)}"
    if region.sup == sp.oo:
        return f"{name} {'>' if region.left_open else '>='} {to_text(region.inf)}"
    return f"{to_text(region.inf)} {left} {name} {right} {to_text(region.sup)}"


# Every value in a response is text, a bool, a small int or None: Brython's json.loads
# cannot read the exponent of a float such as 1e+16.
OPERATIONS: Dict[str, Callable[..., Any]] = {
    "derive": compute_derivative,
    "limit": compute_limit,
    "integrate": compute_integral,
    "simplify": compute_simplify,
    "expand": compute_expand,
    "factor": compute_factor,
    "solve": compute_solve,
    "solve_system": compute_solve_system,
    "solve_inequality": compute_solve_inequality,
    "equivalent": compute_equivalent,
    "taylor": compute_taylor,
    "apart": compute_apart,
    "summation": compute_summation,
}


def handle(request: Dict[str, Any]) -> Dict[str, Any]:
    """{"result": ...} for one request {"operation": ..., "args": {...}}, or {"error": message}."""
    operation = OPERATIONS.get(str(request.get("operation")))
    if operation is None:
        return {"error": f"Unknown operation {request.get('operation')!r}."}
    args = request.get("args") or {}
    if not isinstance(args, dict) or not all(isinstance(value, (str, type(None))) for value in args.values()):
        return {"error": "Arguments must be strings."}
    try:
        return {"result": operation(**args)}
    except ExpressionError as error:
        return {"error": str(error)}
    except (NotImplementedError, ValueError, TypeError, ZeroDivisionError, RecursionError) as error:
        # SymPy raises these for problems it cannot do; the caller answers numerically or says so
        return {"result": None, "note": f"{type(error).__name__}: {error}"[:300]}


def _limit_memory() -> None:
    """Cap the worker's address space (POSIX only) so a runaway computation fails instead of swapping."""
    try:
        import resource

        limit = 2 * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, ValueError, OSError):
        pass


def _exit_with_parent() -> None:
    """End the worker when the app that started it is gone, even in the middle of a computation.

    POSIX only in effect: there the worker is re-parented when the app dies, and getppid() changes.
    """
    import os
    import threading
    import time

    parent = os.getppid()

    def watch() -> None:
        while True:
            time.sleep(2)
            if os.getppid() != parent:
                os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def main() -> None:
    """Worker loop: one JSON request per stdin line, one JSON response per stdout line."""
    _limit_memory()
    _exit_with_parent()
    output = sys.stdout
    sys.stdout = sys.stderr  # stray prints must not corrupt the protocol
    import sympy  # noqa: F401  (imported before the ready line, so the first request is fast)

    _functions()
    output.write(json.dumps({"ready": True}) + "\n")
    output.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = handle(request) if isinstance(request, dict) else {"error": "Bad request."}
        except Exception as error:  # the worker must answer every line
            response = {"result": None, "note": f"{type(error).__name__}: {error}"[:300]}
        output.write(json.dumps(response) + "\n")
        output.flush()


if __name__ == "__main__":
    main()
