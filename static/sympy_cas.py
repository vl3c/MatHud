"""SymPy computations for the CAS tools: limit, integrate, solve and simplify (and derive,
where nerdamer has no answer).

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
        # A decimal such as 0.5 is read as the exact rational it stands for.
        return sp.nsimplify(sp.Float(repr(value)), rational=True)

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
        """SymPy's str() in the client's notation: ^, e, i, Infinity, abs, ceil."""

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


def compute_solve(equation: str, variable: str) -> Optional[Dict[str, Any]]:
    """The solution set of lhs = rhs, or None when it is not a list of roots.

    Returns {"roots": [...]} or {"identity": True, "roots": []}. A periodic solution set
    lists the real roots in [-window, window] (2*pi, or less when there are more than
    MAX_LISTED_ROOTS), with "window" set to that bound and "general" to its general forms
    (x = a*n + b for any integer n) when they are affine in n. A polynomial or rational
    equation is solved over the complex numbers, any other over the reals.
    """
    import sympy as sp

    sides = str(equation).split("=")
    if len(sides) > 2:
        raise ExpressionError("An equation has at most one '='.")
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


OPERATIONS: Dict[str, Callable[..., Any]] = {
    "derive": compute_derivative,
    "limit": compute_limit,
    "integrate": compute_integral,
    "simplify": compute_simplify,
    "solve": compute_solve,
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
