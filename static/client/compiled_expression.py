"""Compile user expressions once and evaluate them with named variables.

``compile_expression(text, variables, functions)`` fixes the text with
``ExpressionValidator.fix_math_expression`` (Python form), validates it, and returns a
``CompiledExpression`` whose ``bind()`` gives a plain callable taking the variables
positionally, in order:

    f = compile_expression("x^2 + 1").bind()                       # f(2.0) == 5.0
    r = compile_expression("2θ", ("θ",), FUNCTIONS_CURVE).bind()   # r(1.0) == 2.0
    F = compile_expression("x^2 + y^2 - 1", ("x", "y")).bind()     # F(1.0, 1.0) == 1.0

``functions`` picks the namespace: ``FUNCTIONS_ALL`` (every function, used for function
and piecewise plots) or ``FUNCTIONS_CURVE`` (the curve-safe rows, for parametric curves).
Names the expression reads that are neither functions, constants, the imaginary unit nor
variables are listed in ``free_names``; they are not rejected (``a*x`` parses and fails
per sample, as before).

``substitute_variable(text, name, replacement)`` rewrites one variable as a whole token,
for shifting a graph: substitute_variable("x^2 + max(x, 1)", "x", "(x - 2)").

Parameters (the canvas's sliders) live in one namespace per page, written only by the
namespace owner's ParameterManager (the page canvas) through ``set_parameter_values``. ``bind(parameters=True)``
compiles against "live" globals that hold them, so a free name such as ``a`` in
``a*sin(x)`` is read at call time: changing a value changes every such callable at once,
with no recompiling. Plain ``bind()`` never sees parameters, so the CAS checks keep their
meaning of a name.

Pure Python (no browser imports), so server-side pytest imports it directly.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, cast

import expression_functions as ef
from expression_validator import ExpressionValidator

FUNCTIONS_ALL = "all"
FUNCTIONS_CURVE = "curve"

# Names each namespace defines (functions and constants), without resolving MathUtils
_NAMESPACE_KEYS: Dict[str, FrozenSet[str]] = {
    FUNCTIONS_ALL: frozenset(ef.BY_NAME) | frozenset(ef.CONSTANTS),
    FUNCTIONS_CURVE: frozenset(ef.CURVE_NAMES) | frozenset(ef.CONSTANTS),
}
_NAMESPACE_BUILDERS: Dict[str, Callable[[], Dict[str, Any]]] = {
    FUNCTIONS_ALL: ef.build_python_namespace,
    FUNCTIONS_CURVE: ef.build_curve_namespace,
}
_NAMESPACES: Dict[str, Dict[str, Any]] = {}

# Parsed expressions per raw text (failures are never cached); cleared when full
_CACHE_LIMIT = 256


class _Parsed:
    """What compiling one text yields; independent of the variables and the namespace."""

    __slots__ = ("python_text", "tree", "names")

    def __init__(self, python_text: str, tree: ast.Expression, names: FrozenSet[str]) -> None:
        self.python_text = python_text
        self.tree = tree
        self.names = names


_CACHE: Dict[str, _Parsed] = {}
# Bound lambdas per (Python text, variables, functions), and their globals per function set
_LAMBDAS: Dict[Tuple[str, Tuple[str, ...], str, bool], Callable[..., float]] = {}
_LAMBDA_GLOBALS: Dict[str, Dict[str, Any]] = {}
# Globals of lambdas bound with parameters=True, per function set: the plain globals plus
# the current parameter values, updated in place by set_parameter_values
_LIVE_GLOBALS: Dict[str, Dict[str, Any]] = {}
# The current parameter values (name -> float); the single source the live globals copy
_PARAMETERS: Dict[str, float] = {}

# Identifiers (Unicode letters too, so θ and Δx are names) and number literals matched whole,
# so a variable is never found inside a longer name (max, x2, Δx) or a number (1e5)
_NAME_OR_NUMBER = re.compile(r"[^\W\d]\w*|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
# Checked variable tuples per (variables, functions) as given
_CHECKED_VARIABLES: Dict[Tuple[Tuple[str, ...], str], Tuple[str, ...]] = {}


def base_namespace(functions: str) -> Dict[str, Any]:
    """The functions and constants of one namespace kind, built once (do not mutate it)."""
    namespace = _NAMESPACES.get(functions)
    if namespace is None:
        namespace = _NAMESPACE_BUILDERS[_checked_functions(functions)]()
        _NAMESPACES[functions] = namespace
    return namespace


class CompiledExpression:
    """A validated expression and the variables it is evaluated with.

    Attributes:
        source: The text as given.
        python_text: The text after fix_math_expression (Python form).
        variables: The variable names, normalised (ϑ -> θ), in argument order.
        functions: FUNCTIONS_ALL or FUNCTIONS_CURVE.
    """

    __slots__ = ("source", "python_text", "variables", "functions", "_parsed")

    def __init__(self, source: str, parsed: _Parsed, variables: Tuple[str, ...], functions: str) -> None:
        self.source = source
        self.python_text = parsed.python_text
        self.variables = variables
        self.functions = functions
        self._parsed = parsed

    @property
    def free_names(self) -> Tuple[str, ...]:
        """Names read (not called) that the namespace and the variables leave undefined, sorted.

        Bare function names (sin + x) are bound, not free; i and j (the imaginary unit) are
        reserved, not free; a call of a function missing from the namespace (derive(t) in a
        curve) is not a name read.
        """
        defined = _NAMESPACE_KEYS[self.functions]
        return tuple(
            sorted(
                name
                for name in self._parsed.names
                if name not in defined and name not in self.variables and name not in ef.IMAGINARY_UNITS
            )
        )

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """The free names that are current parameters, sorted."""
        return tuple(name for name in self.free_names if name in _PARAMETERS)

    @property
    def undefined_names(self) -> Tuple[str, ...]:
        """The free names that are not current parameters, sorted (they raise NameError)."""
        return tuple(name for name in self.free_names if name not in _PARAMETERS)

    def bind(self, parameters: bool = False) -> Callable[..., float]:
        """A callable taking the variables positionally and returning a float.

        Calling it raises what evaluating the expression raises
        (NameError for a free name, ValueError for a domain error...). With parameters=True
        a free name that is a current parameter reads its value at call time
        (set_parameter_values); other free names still raise NameError.

        The callable is a lambda compiled once per (Python text, variables, functions)
        and shared by every bind of them: Brython's eval(code, namespace) translates the
        code to JavaScript on every call, a compiled lambda only once. The lambda is built
        from the validated tree (no text is re-parsed), and its globals are one namespace
        per function set, never written while evaluating.
        """
        key = (self.python_text, self.variables, self.functions, bool(parameters))
        evaluate = _LAMBDAS.get(key)
        if evaluate is None:
            evaluate = _compile_lambda(self._parsed, self.variables, self.functions, bool(parameters))
            if len(_LAMBDAS) >= _CACHE_LIMIT:
                _LAMBDAS.clear()
            _LAMBDAS[key] = evaluate
        return evaluate


# The name the lambda converts its result with: not an identifier, so no expression text
# and no variable can name it
_FLOAT = "<float>"


def _compile_lambda(
    parsed: _Parsed, variables: Tuple[str, ...], functions: str, parameters: bool = False
) -> Callable[..., float]:
    """lambda <variables>: float(<validated tree>), built from the tree, never from text.

    The tree itself compiled while parsing, and the lambda adds two nodes above it, well
    within the depth the validator allows. With parameters, its globals are the live ones,
    so later parameter values are read at call time (checked in Brython too: a lambda made
    by eval(code, g) sees later writes to g).
    """
    code = compile(_lambda_tree(parsed.tree, variables), "<string>", mode="eval")
    namespace = _live_globals(functions) if parameters else _lambda_globals(functions)
    return cast(Callable[..., float], eval(code, namespace))


def _lambda_tree(tree: ast.Expression, variables: Tuple[str, ...]) -> ast.Expression:
    body = tree.body
    # Brython compiles hand-built nodes only with explicit locations
    location: Dict[str, Any] = {
        "lineno": body.lineno,
        "col_offset": body.col_offset,
        "end_lineno": body.end_lineno,
        "end_col_offset": body.end_col_offset,
    }
    arguments = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=name, annotation=None, type_comment=None, **location) for name in variables],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )
    call = ast.Call(func=ast.Name(id=_FLOAT, ctx=ast.Load(), **location), args=[body], keywords=[], **location)
    return ast.Expression(body=ast.Lambda(args=arguments, body=call, **location))


def _lambda_globals(functions: str) -> Dict[str, Any]:
    """The globals of the compiled lambdas: a copy of the base namespace plus the converter."""
    namespace = _LAMBDA_GLOBALS.get(functions)
    if namespace is None:
        namespace = dict(base_namespace(functions))
        namespace[_FLOAT] = float
        _LAMBDA_GLOBALS[functions] = namespace
    return namespace


def _live_globals(functions: str) -> Dict[str, Any]:
    """The globals of lambdas bound with parameters: the plain globals plus the parameters."""
    namespace = _LIVE_GLOBALS.get(functions)
    if namespace is None:
        namespace = dict(_lambda_globals(functions))
        namespace.update(_PARAMETERS)
        _LIVE_GLOBALS[functions] = namespace
    return namespace


def set_parameter_values(values: Mapping[str, float]) -> None:
    """Replace the current parameter values (name -> number), in place in every live namespace.

    Names that are no longer parameters are removed, so callables reading them raise
    NameError again. Each name must be an identifier that no namespace defines; the
    ParameterManager's name rules guarantee it, and this check keeps a bad name from
    replacing a function such as sin.

    Raises:
        ValueError: A name is not an identifier, starts with "_" or is reserved, or a value
            is not a real number.
    """
    checked: Dict[str, float] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_") or name in ef.RESERVED_NAMES:
            raise ValueError(f"Invalid parameter name {name!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Parameter {name!r} must be a number, got {value!r}")
        checked[name] = float(value)
    if checked == _PARAMETERS:
        return
    removed = [name for name in _PARAMETERS if name not in checked]
    _PARAMETERS.clear()
    _PARAMETERS.update(checked)
    for namespace in _LIVE_GLOBALS.values():
        for name in removed:
            namespace.pop(name, None)
        namespace.update(checked)


def has_parameters() -> bool:
    """Whether any parameter exists (render caches skip their lookups when none does)."""
    return bool(_PARAMETERS)


def parameter_values() -> Dict[str, float]:
    """A copy of the current parameter values."""
    return dict(_PARAMETERS)


def parameter_value(name: str) -> Optional[float]:
    """The current value of one parameter, or None when it is not a parameter."""
    return _PARAMETERS.get(name)


def plain_number_text(value: float) -> str:
    """The shortest repr of a finite float written without an exponent (1e-05 -> 0.00001).

    Expression text reads 1e-05 as a number, but 2e-3 style is easy to misread next to
    Euler's e, so substituted values never use it.
    """
    text = repr(float(value))
    if "e" not in text and "E" not in text:
        return text
    # repr's exponent form has one non-zero digit before the point: 1.5e-10, 1e+20
    mantissa, exponent_text = text.lower().split("e")
    sign = ""
    if mantissa.startswith("-"):
        sign, mantissa = "-", mantissa[1:]
    whole, _, fraction = mantissa.partition(".")
    digits = whole + fraction
    point = len(whole) + int(exponent_text)
    if point <= 0:
        body = "0." + "0" * (-point) + digits
    elif point >= len(digits):
        body = digits + "0" * (point - len(digits))
    else:
        body = digits[:point] + "." + digits[point:]
    return sign + body


def with_parameter_values(text: str, names: Optional[Sequence[str]] = None) -> str:
    """The text with each parameter name (or each of the given ones) replaced by its value.

    Values go in parentheses, so a*x^2 with a = -2 becomes (-2.0)*x^2. Names that are not
    current parameters are left alone. Used where a text, not a callable, is analysed
    (a function's asymptotes at the current values).
    """
    result = text
    for name in names if names is not None else tuple(_PARAMETERS):
        value = _PARAMETERS.get(name)
        if value is not None:
            result = substitute_variable(result, name, f"({plain_number_text(value)})")
    return result


def compile_expression(
    text: str, variables: Sequence[str] = ("x",), functions: str = FUNCTIONS_ALL
) -> CompiledExpression:
    """Fix, validate and compile text for evaluation with the given variables.

    Args:
        text: The expression as typed or stored (math.js or Python form).
        variables: Variable names, in the order the bound callable takes them.
        functions: FUNCTIONS_ALL or FUNCTIONS_CURVE.

    Raises:
        ValueError: The text does not validate (the messages of
            ExpressionValidator.validate_expression_tree), or a variable name is not an
            identifier, starts with "_", is reserved (ef.RESERVED_NAMES) or repeats.
    """
    checked = _checked_variables(variables, functions)
    parsed = _CACHE.get(text)
    if parsed is None:
        parsed = _parse(text)
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.clear()
        _CACHE[text] = parsed
    return CompiledExpression(text, parsed, checked, functions)


def _parse(text: str) -> _Parsed:
    python_text = ExpressionValidator.fix_math_expression(text, python_compatible=True)
    tree = ExpressionValidator.parse_validated_tree(python_text)
    # Compiled here, as before, so an error only compile finds is still raised while parsing
    compile(tree, "<string>", mode="eval")
    return _Parsed(python_text, tree, _read_names(tree))


def _read_names(tree: ast.AST) -> FrozenSet[str]:
    """Names the tree reads, except those only called (sin in sin(x))."""
    nodes = list(ast.walk(tree))
    called = {id(node.func) for node in nodes if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    return frozenset(node.id for node in nodes if isinstance(node, ast.Name) and id(node) not in called)


def _checked_functions(functions: str) -> str:
    if functions not in _NAMESPACE_KEYS:
        raise ValueError(f"Unknown function set {functions!r}; expected {FUNCTIONS_ALL!r} or {FUNCTIONS_CURVE!r}")
    return functions


def _checked_variables(variables: Sequence[str], functions: str) -> Tuple[str, ...]:
    key: Optional[Tuple[Tuple[str, ...], str]] = None
    if not isinstance(variables, str):
        try:
            key = (tuple(variables), functions)
            cached = _CHECKED_VARIABLES.get(key)
        except TypeError:  # an unhashable name; reported below
            key, cached = None, None
        if cached is not None:
            return cached
    checked = _check_variables(variables, functions)
    if key is not None:
        if len(_CHECKED_VARIABLES) >= _CACHE_LIMIT:
            _CHECKED_VARIABLES.clear()
        _CHECKED_VARIABLES[key] = checked
    return checked


def _check_variables(variables: Sequence[str], functions: str) -> Tuple[str, ...]:
    _checked_functions(functions)
    if isinstance(variables, str):
        raise ValueError(f"Variables must be a sequence of names, not the string {variables!r}")
    checked: List[str] = []
    for given in variables:
        name = ExpressionValidator.normalize_unicode_name(given)
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"Invalid variable name {given!r}: not an identifier")
        if name.startswith("_"):
            raise ValueError(f"Invalid variable name {given!r}: names starting with '_' are reserved")
        if name in ef.RESERVED_NAMES:
            raise ValueError(
                f"Invalid variable name {given!r}: it names a function, constant, the imaginary unit or a keyword"
            )
        if name in checked:
            raise ValueError(f"Duplicate variable name {given!r}")
        checked.append(name)
    return tuple(checked)


def substitute_variable(text: str, name: str, replacement: str) -> str:
    """Replace each whole-token occurrence of the variable name in text.

    Tokens that only contain the name are left alone: other names (max, exp, x2, xy, Δx,
    θ1), number literals (1e5) and the name followed by "(" (a call, never a variable).
    The text is the stored form of fix_math_expression, so ϑ is already θ; the name is
    normalised the same way.

    On every stored function text this gives the same string as the regex that
    Function.translate and PiecewiseFunction.translate used before (a standalone x between
    word boundaries, function names protected). The two differ only where a digit precedes
    the name (2x, which stored text never holds, since fix_math_expression inserts "*" after
    numbers) or "(" follows it (x(x+1), which never parses and so is never stored).

    Args:
        text: The expression (stored, non-Python form).
        name: The variable to replace, such as "x" or "θ".
        replacement: The text put in its place, such as "(x - 2)".

    Returns:
        The rewritten text.
    """
    target = ExpressionValidator.normalize_unicode_name(name)

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token == target and text[match.end() : match.end() + 1] != "(":
            return replacement
        return token

    return _NAME_OR_NUMBER.sub(replace, text)
