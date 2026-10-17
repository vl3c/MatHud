"""SymPy computations for the CAS tools: limit, integrate, solve and simplify (and derive,
where nerdamer has no answer), the solution sets of inequalities, whether two expressions
are equivalent (for the step checker), Taylor polynomials and partial fractions.

The browser's symbolic engine (nerdamer) is often wrong on these four (see
documentation/Roadmap.md, CAS reliability), so the client asks the server first and keeps
nerdamer as the fallback when the server or SymPy is unavailable.

Expressions arrive as text in the client's notation (``^`` for powers, ``e``, ``pi``,
``i``, ``Infinity``). They are never evaluated: ``parse_expression`` reads them with
Python's ``ast`` module, accepts only numbers, names, arithmetic and calls to the
functions in ``_FUNCTIONS``, and builds the SymPy expression from that tree. Results are
printed back in the same notation, so math.js and nerdamer can read them.

This module runs in a worker process (``python -m static.sympy_cas``, managed by
``static.sympy_engine``) that is killed when a computation runs too long; it reads one
JSON request per line on stdin and writes one JSON response per line on stdout.
"""

from __future__ import annotations

import ast
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


def parse_expression(text: str, real: bool = False) -> Any:
    """The SymPy expression for text, built from its syntax tree without evaluating it.

    Every name that is not a constant or a known function becomes a symbol (real when
    real is True).

    Raises:
        ExpressionError: for syntax errors and anything but numbers, names, arithmetic and
            calls to known functions (attributes, subscripts, keywords, lambdas, ...).
    """
    try:
        tree = ast.parse(_prepared_text(str(text)).strip(), mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"Cannot read the expression {text!r}: {error.msg}.") from None
    return _Builder(real).build(tree.body)


class _Builder:
    """Turns an ast tree into a SymPy expression, refusing every node it does not know."""

    def __init__(self, real: bool) -> None:
        import sympy as sp

        self.sp = sp
        self.real = real

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
    """root, or its decimal value when it is a CRootOf or a long radical expression."""
    import sympy as sp

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
        result = sp.integrate(expr, x, conds="none")
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
    if isinstance(solutions, sp.FiniteSet):
        listed = [_displayed_root(root) for root in solutions]
        if not all(_is_usable(root) for root in listed):
            return None
        return {"roots": [to_text(root) for root in _sorted_roots(listed)]}
    return _periodic_solution(solutions, variable)


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
# grow about tenfold per step on roots of quotients (sqrt((1-x)/(1+x))).
MAX_DERIVATIVE_OPS = 2000
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

    The coefficients come from the derivatives at a when they are all finite real numbers
    and the function has no abs, sign, floor or the like; otherwise from the series on both
    sides of a, because a one-sided series hides a corner or a jump (abs(x) is x right of 0
    and -x left of it).

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
    real numbers takes its coefficients from them, one list for both sides: the series at
    a point other than 0 can run past the time limit (e^(-x^2) at 1). Otherwise, as for
    sin(x)/x at 0 or abs(x)^3 at 0, each side comes from its series."""
    import sympy as sp

    breaking = function.has(sp.Abs, sp.sign, sp.floor, sp.ceiling, sp.Min, sp.Max, sp.Mod)
    if not breaking:
        coefficients = _derivative_coefficients(function, x, center, count)
        if coefficients is not None:
            return coefficients, coefficients
    h = sp.Dummy("h", real=True)
    shifted = function.subs(x, center + h)
    right = _series_coefficients(shifted, h, count, "+")
    # The left side to the order only, unless the function has a part that may break off
    # beyond it (abs(x)^3 at 0 is smooth to order 2 only); the series is the slow step.
    left = _series_coefficients(shifted, h, count if breaking else degree + 1, "-")
    return right, left


def _derivative_coefficients(function: Any, x: Any, center: Any, count: int) -> Optional[List[Any]]:
    """f^(k)(a)/k! for k = 0 .. count-1, or None when one of them is not a finite real
    number (0/0 in sin(x)/x at 0, a pole, sqrt(x) at 0) or a derivative grows past
    MAX_DERIVATIVE_OPS."""
    import sympy as sp

    coefficients: List[Any] = []
    derivative = function
    for k in range(count):
        if k:
            derivative = sp.expand_mul(sp.diff(derivative, x))
            if sp.count_ops(derivative) > MAX_DERIVATIVE_OPS:
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


# Every value in a response is text, a bool, a small int or None: Brython's json.loads
# cannot read the exponent of a float such as 1e+16.
OPERATIONS: Dict[str, Callable[..., Any]] = {
    "derive": compute_derivative,
    "limit": compute_limit,
    "integrate": compute_integral,
    "simplify": compute_simplify,
    "solve": compute_solve,
    "solve_inequality": compute_solve_inequality,
    "equivalent": compute_equivalent,
    "taylor": compute_taylor,
    "apart": compute_apart,
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
        # SymPy raises these for problems it cannot do; the caller falls back to nerdamer
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
