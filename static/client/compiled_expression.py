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

Pure Python (no browser imports), so server-side pytest imports it directly.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple, cast

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

    __slots__ = ("python_text", "code", "names")

    def __init__(self, python_text: str, code: Any, names: FrozenSet[str]) -> None:
        self.python_text = python_text
        self.code = code
        self.names = names


_CACHE: Dict[str, _Parsed] = {}
# Bound lambdas per (Python text, variables, functions), and their globals per function set
_LAMBDAS: Dict[Tuple[str, Tuple[str, ...], str], Callable[..., float]] = {}
_LAMBDA_GLOBALS: Dict[str, Dict[str, Any]] = {}

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

    def bind(self) -> Callable[..., float]:
        """A callable taking the variables positionally and returning a float.

        It never raises here; calling it raises what evaluating the expression raises
        (NameError for a free name, ValueError for a domain error...).

        The callable is a lambda compiled once per (Python text, variables, functions)
        and shared by every bind of them: Brython's eval(code, namespace) translates the
        code to JavaScript on every call, a compiled lambda only once. Its globals are one
        namespace per function set, never written while evaluating.
        """
        key = (self.python_text, self.variables, self.functions)
        evaluate = _LAMBDAS.get(key)
        if evaluate is None:
            evaluate = _compile_lambda(self.python_text, self.variables, self.functions)
            if len(_LAMBDAS) >= _CACHE_LIMIT:
                _LAMBDAS.clear()
            _LAMBDAS[key] = evaluate
        return evaluate


def _compile_lambda(python_text: str, variables: Tuple[str, ...], functions: str) -> Callable[..., float]:
    # The text validated on its own, so it is one expression in parentheses too; the line
    # break ends a trailing comment. _float cannot be a variable (names starting with "_"
    # are rejected), so it always is float.
    source = f"lambda {', '.join(variables)}: _float(({python_text}\n))"
    return cast(Callable[..., float], eval(source, _lambda_globals(functions)))


def _lambda_globals(functions: str) -> Dict[str, Any]:
    """The globals of the compiled lambdas: a copy of the base namespace plus _float."""
    namespace = _LAMBDA_GLOBALS.get(functions)
    if namespace is None:
        namespace = dict(base_namespace(functions))
        namespace["_float"] = float
        _LAMBDA_GLOBALS[functions] = namespace
    return namespace


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
    code = compile(tree, "<string>", mode="eval")
    return _Parsed(python_text, code, _read_names(tree))


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
