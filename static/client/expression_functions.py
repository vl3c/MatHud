"""The functions expressions may call, one row per function.

ExpressionValidator (the whitelist, aliases, superscript notation and both Python
evaluation namespaces) and MathUtils (Python-only routing and the names math.js knows)
derive their tables from the rows below, so a function is described in one place.

Adding a function:
    1. Add a row to ``FUNCTIONS``. ``python`` is the callable, or the name of a MathUtils
       attribute as a string (resolved on first use, so this module never imports the
       browser). Set ``mathjs`` when math.js knows it under another name and ``None`` when
       math.js lacks it; mark it ``python_only`` when MathUtils.evaluate must use Python.
    2. When ``described`` is true (the default), also name it in the hand-written
       ``evaluate_expression`` description in ``static/functions_definitions.py``;
       ``server_tests/test_expression_functions.py`` fails until both agree.
    3. Set ``sympy`` when ``static/sympy_cas.py`` accepts it (the same test checks the
       SymPy table).
    4. A ``curve_safe`` row of exactly one argument needs sample points in
       ``client_tests/test_expression_functions.py``, which compares its Python value with
       math.js in the browser.

Pure Python (no browser imports), so server-side pytest imports it directly.
"""

from __future__ import annotations

import ast
import keyword
import math
import random
import re
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple, Type, Union, cast

PythonFunction = Union[Callable[..., Any], str]
Arity = Tuple[int, Optional[int]]


class ExpressionFunction:
    """One function expressions may call.

    Attributes:
        name: The canonical name, as fix_math_expression writes it.
        python: The callable, or the name of a MathUtils attribute (resolved lazily).
        arity: (min, max) argument counts, max None for variadic. Not enforced; tests
            check it against the callable's signature and the SymPy table.
        mathjs: The name math.js knows it by, or None when math.js lacks it.
        aliases: Spellings rewritten to name when "(" follows them (sine( -> sin().
        word_aliases: Spellings rewritten to name as words (ln -> log).
        curve_safe: Allowed in parametric curves (the curve namespace, FUNCTIONS_CURVE in
            compiled_expression). Function and piecewise plots still use the full namespace.
        superscript: sin²(x) means sin(x)^2 (the word aliases follow the row).
        inverse: The function a superscript ⁻¹ names (sin⁻¹ -> asin), if any.
        python_only: MathUtils.evaluate must use Python (math.js lacks it).
        sympy: The server's SymPy parser (sympy_cas) accepts it.
        described: Named in the evaluate_expression tool description.
    """

    __slots__ = (
        "name",
        "python",
        "arity",
        "mathjs",
        "aliases",
        "word_aliases",
        "curve_safe",
        "superscript",
        "inverse",
        "python_only",
        "sympy",
        "described",
    )

    def __init__(
        self,
        name: str,
        python: PythonFunction,
        arity: Arity,
        *,
        mathjs: Optional[str] = "",
        aliases: Tuple[str, ...] = (),
        word_aliases: Tuple[str, ...] = (),
        curve_safe: bool = False,
        superscript: bool = False,
        inverse: Optional[str] = None,
        python_only: bool = False,
        sympy: bool = False,
        described: bool = True,
    ) -> None:
        self.name = name
        self.python = python
        self.arity = arity
        # "" means math.js uses the same name
        self.mathjs: Optional[str] = name if mathjs == "" else mathjs
        self.aliases = aliases
        self.word_aliases = word_aliases
        self.curve_safe = curve_safe
        self.superscript = superscript
        self.inverse = inverse
        self.python_only = python_only
        self.sympy = sympy
        self.described = described

    def __repr__(self) -> str:
        return f"ExpressionFunction({self.name!r})"


# Reciprocal trigonometric functions and their inverses, which Python's math module lacks.
# They follow math.js: acot(x) = atan(1/x), so acot(0) = pi/2 and acot(-1) = -pi/4.
def _sec(x: float) -> float:
    return 1 / math.cos(x)


def _csc(x: float) -> float:
    return 1 / math.sin(x)


def _cot(x: float) -> float:
    return 1 / math.tan(x)


def _asec(x: float) -> float:
    return math.acos(1 / x)


def _acsc(x: float) -> float:
    return math.asin(1 / x)


def _acot(x: float) -> float:
    return math.pi / 2 if x == 0 else math.atan(1 / x)


def _randint(a: int, b: int) -> int:
    """A random integer in [a, b] (math.js has no inclusive equivalent)."""
    return random.randint(a, b)


# Calculator functions, following math.js 14.5.2 so plots and evaluate_expression agree.
def _mod(x: float, y: float) -> float:
    """The floored remainder (its sign is the divisor's): mod(-7, 3) = 2, mod(7, -3) = -2.

    A zero divisor gives x, as math.js does (Python's % raises instead).
    """
    return x if y == 0 else x % y


def _nthroot(x: float, n: float) -> float:
    """The real n-th root, as math.js nthRoot: nthroot(-8, 3) = -2, nthroot(16, -2) = 0.25.

    Raises:
        ValueError: for n = 0, and for a negative x unless n is an odd integer.
    """
    if n == 0:
        raise ValueError("nthroot: the root n must not be 0")
    if x < 0:
        if n != int(n) or int(n) % 2 == 0:
            raise ValueError("nthroot: the root must be odd when the value is negative")
        return -float((-x) ** (1 / n))
    return float(x ** (1 / n))


def _sign(x: Any) -> Any:
    """-1, 0 or 1 for a real number (NaN stays NaN); z/|z| for a complex number (0 at 0)."""
    if isinstance(x, complex):
        return x / abs(x) if x != 0 else 0.0
    if x != x:  # NaN
        return x
    return float((x > 0) - (x < 0))


# Parts of a complex number; a real number is its own real part, with angle 0 or pi.
def _re(z: Any) -> Any:
    return z.real if isinstance(z, complex) else z


def _im(z: Any) -> Any:
    return z.imag if isinstance(z, complex) else 0.0


def _arg(z: Any) -> float:
    """The angle of z, between -pi and pi: arg(-1) = pi."""
    if isinstance(z, complex):
        return math.atan2(z.imag, z.real)
    return math.atan2(0.0, z)


def _conj(z: Any) -> Any:
    return z.conjugate() if isinstance(z, complex) else z


_F = ExpressionFunction
_ONE: Arity = (1, 1)


def _elementary(name: str, python: PythonFunction, arity: Arity = _ONE, **options: Any) -> ExpressionFunction:
    """A row allowed in curves, with superscript notation, that SymPy accepts (sin, log, exp...)."""
    return ExpressionFunction(name, python, arity, curve_safe=True, superscript=True, sympy=True, **options)


FUNCTIONS: Tuple[ExpressionFunction, ...] = (
    _elementary("sin", math.sin, aliases=("sine",), inverse="asin"),
    _elementary("cos", math.cos, aliases=("cosine",), inverse="acos"),
    _elementary("tan", math.tan, aliases=("tangent",), inverse="atan"),
    _elementary("sqrt", "sqrt", aliases=("square root",)),
    _elementary("log", math.log, (1, 2), aliases=("logarithm",), word_aliases=("ln",)),
    _elementary("log10", math.log10, aliases=("logarithm10",)),
    _elementary("log2", math.log2, aliases=("logarithm2",)),
    _F("factorial", math.factorial, _ONE, aliases=("fact",), curve_safe=True, sympy=True),
    _elementary("asin", math.asin, aliases=("arcsine",)),
    _elementary("acos", math.acos, aliases=("arccosine",)),
    _elementary("atan", math.atan, aliases=("arctangent",)),
    _elementary("sinh", math.sinh, aliases=("hyperbolic sine",), inverse="asinh"),
    _elementary("cosh", math.cosh, aliases=("hyperbolic cosine",), inverse="acosh"),
    _elementary("tanh", math.tanh, aliases=("hyperbolic tangent",), inverse="atanh"),
    _elementary("sec", _sec, inverse="asec"),
    _elementary("csc", _csc, inverse="acsc"),
    _elementary("cot", _cot, inverse="acot"),
    _elementary("asec", _asec),
    _elementary("acsc", _acsc),
    _elementary("acot", _acot),
    _elementary("asinh", math.asinh),
    _elementary("acosh", math.acosh),
    _elementary("atanh", math.atanh),
    _elementary("exp", math.exp, aliases=("exponential",)),
    _elementary("abs", abs, aliases=("absolute",)),
    _F("pow", "pow", (2, 2), aliases=("power",), curve_safe=True, sympy=True),
    _F("bin", bin, _ONE, aliases=("binary",)),
    _F("det", "det", _ONE, aliases=("determinant",)),
    _F("arrangements", "arrangements", (2, 2), mathjs="permutations"),
    _F("permutations", "permutations", (1, 2)),
    _F("combinations", "combinations", (2, 2)),
    _F("limit", "limit", (3, 3), mathjs=None, aliases=("lim",), described=False),
    _F("derive", "derivative", (2, 2), mathjs=None, aliases=("derivative", "derivate"), described=False),
    _F("integrate", "integral", (2, 4), mathjs=None, aliases=("integral",), described=False),
    _F("simplify", "simplify", _ONE, described=False),
    _F("expand", "expand", _ONE, mathjs=None, described=False),
    _F("factor", "factor", _ONE, mathjs=None, described=False),
    _F("solve", "solve", (2, 2), mathjs=None, described=False),
    _F("random", "random", (0, 2), aliases=("rand",)),
    _F("round", "round", (1, 2), curve_safe=True),
    _F("gcd", "gcd", (1, None)),
    _F("lcm", "lcm", (1, None)),
    _F("is_prime", "is_prime", _ONE, mathjs=None, python_only=True),
    _F("prime_factors", "prime_factors", _ONE, mathjs=None, python_only=True),
    _F("mod_pow", "mod_pow", (3, 3), mathjs=None, python_only=True),
    _F("mod_inverse", "mod_inverse", (2, 2), mathjs=None, python_only=True),
    _F("next_prime", "next_prime", _ONE, mathjs=None, python_only=True),
    _F("prev_prime", "prev_prime", _ONE, mathjs=None, python_only=True),
    _F("totient", "totient", _ONE, mathjs=None, python_only=True),
    _F("divisors", "divisors", _ONE, mathjs=None, python_only=True),
    _F("mean", "mean", _ONE),
    _F("median", "median", _ONE),
    _F("mode", "mode", _ONE),
    _F("stdev", "stdev", _ONE, mathjs="std", aliases=("std",)),
    _F("variance", "variance", _ONE, aliases=("var",)),
    _F("ceil", math.ceil, _ONE, curve_safe=True, sympy=True),
    _F("floor", math.floor, _ONE, curve_safe=True, sympy=True),
    _F("trunc", math.trunc, _ONE, mathjs="fix", curve_safe=True),
    _F("max", max, (1, None), curve_safe=True, sympy=True),
    _F("min", min, (1, None), curve_safe=True, sympy=True),
    _F("sum", sum, (1, 2)),
    _F("mod", _mod, (2, 2), curve_safe=True),
    _F("nthroot", _nthroot, (2, 2), mathjs="nthRoot", aliases=("nthRoot",), curve_safe=True, sympy=True),
    _F("cbrt", math.cbrt, _ONE, curve_safe=True, sympy=True),
    _F("atan2", math.atan2, (2, 2), curve_safe=True, sympy=True),
    _F("sign", _sign, _ONE, aliases=("sgn",), curve_safe=True, sympy=True),
    _F("gamma", math.gamma, _ONE, curve_safe=True, sympy=True),
    _F("erf", math.erf, _ONE, curve_safe=True, sympy=True),
    _F("re", _re, _ONE, aliases=("real",)),
    _F("im", _im, _ONE, aliases=("imag",)),
    _F("arg", _arg, _ONE),
    _F("conj", _conj, _ONE, aliases=("conjugate",)),
    _F("randint", _randint, (2, 2), mathjs=None, python_only=True),
    _F("summation", "summation", (4, 4), mathjs=None, python_only=True),
    _F("product", "product", (4, 4), mathjs=None, python_only=True),
    _F("arithmetic_sum", "arithmetic_sum", (3, 3), mathjs=None, python_only=True),
    _F("geometric_sum", "geometric_sum", (3, 3), mathjs=None, python_only=True),
    _F("geometric_sum_infinite", "geometric_sum_infinite", (2, 2), mathjs=None, python_only=True),
    _F("ratio_test", "ratio_test", (2, 2), mathjs=None, python_only=True),
    _F("root_test", "root_test", (2, 2), mathjs=None, python_only=True),
    _F("p_series_test", "p_series_test", _ONE, mathjs=None, python_only=True),
)

BY_NAME: Dict[str, ExpressionFunction] = {row.name: row for row in FUNCTIONS}

# Constants in both evaluation namespaces (∞ is written inf for Python, Infinity by math.js)
CONSTANTS: Dict[str, float] = {"pi": math.pi, "e": math.e, "inf": math.inf, "Infinity": math.inf}

# The AST node types an expression may contain
ALLOWED_NODES: Tuple[Type[ast.AST], ...] = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Constant,
    ast.Name,
    ast.BinOp,
    ast.UnaryOp,
    ast.Expression,
    ast.Call,
    ast.List,
)

# Names a call may name. pi and e are listed too, as before; only calls consult the set.
ALLOWED_FUNCTIONS: FrozenSet[str] = frozenset(BY_NAME) | frozenset(("pi", "e"))

# Functions whose superscript applies to the result: sin²(x) = sin(x)^2
SUPERSCRIPT_NAMES: FrozenSet[str] = frozenset(
    name for row in FUNCTIONS if row.superscript for name in (row.name,) + row.word_aliases
)

# A superscript ⁻¹ on these names the inverse function: sin⁻¹(x) = asin(x)
INVERSE_NAMES: Dict[str, str] = {row.name: row.inverse for row in FUNCTIONS if row.inverse is not None}

# Aliases and the names they stand for: call aliases only before "(" (sine -> sin), word
# aliases anywhere (ln -> log); see replace_aliases
CALL_ALIASES: Dict[str, str] = {alias: row.name for row in FUNCTIONS for alias in row.aliases}
WORD_ALIASES: Dict[str, str] = {alias: row.name for row in FUNCTIONS for alias in row.word_aliases}

# Functions MathUtils.evaluate computes in Python because math.js lacks them
PYTHON_ONLY_NAMES: FrozenSet[str] = frozenset(row.name for row in FUNCTIONS if row.python_only)

# Names math.js knows under another name (arrangements -> permutations, stdev -> std, trunc -> fix,
# nthroot -> nthRoot)
MATHJS_RENAMES: Dict[str, str] = {
    row.name: row.mathjs for row in FUNCTIONS if row.mathjs is not None and row.mathjs != row.name
}

# Functions allowed in parametric curves
CURVE_NAMES: FrozenSet[str] = frozenset(row.name for row in FUNCTIONS if row.curve_safe)

# The imaginary unit: fix_math_expression writes i as j for Python (2i -> 2j, i alone -> j)
IMAGINARY_UNITS: FrozenSet[str] = frozenset(("i", "j"))

# Names a variable (or, later, a parameter) may not take: function names, word aliases
# (ln), constants, the imaginary unit and Python keywords (2in becomes 2*in, a syntax error).
# Call-only aliases (power, binary, var, fact, rand, lim, std, sine...) stay free: they
# name a function only when "(" follows them.
RESERVED_NAMES: FrozenSet[str] = (
    frozenset(BY_NAME) | frozenset(WORD_ALIASES) | frozenset(CONSTANTS) | IMAGINARY_UNITS | frozenset(keyword.kwlist)
)


def _resolve(row: ExpressionFunction) -> Callable[..., Any]:
    """The row's callable; a string names a MathUtils attribute (imported here, on first use)."""
    if isinstance(row.python, str):
        from utils.math_utils import MathUtils

        return cast(Callable[..., Any], getattr(MathUtils, row.python))
    return row.python


def _namespace(rows: Tuple[ExpressionFunction, ...]) -> Dict[str, Any]:
    namespace: Dict[str, Any] = {row.name: _resolve(row) for row in rows}
    namespace.update(CONSTANTS)
    return namespace


def build_python_namespace() -> Dict[str, Any]:
    """Every function and constant, for evaluating expressions in x (callers cache it)."""
    return _namespace(FUNCTIONS)


def build_curve_namespace() -> Dict[str, Any]:
    """The curve-safe functions and the constants, for parametric curves (callers cache it)."""
    return _namespace(tuple(row for row in FUNCTIONS if row.curve_safe))


# ----- Rewrites by whole tokens -----
# Identifiers and number literals (with an optional exponent), matched whole from left to
# right, so a name inside a longer identifier (kiln, x2ln, artifact) is never a match while
# one after a number (2ln(x), 1e3sine(x), 2.5ln(x)) is.
# Note: "[-+]" rather than "[+-]" -- Brython's re fails to match "e-5" with the latter.
_TOKEN = r"[a-zA-Z_][a-zA-Z_0-9]*|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
# Call aliases with a space ("hyperbolic sine") are tried, with their "(", before any
# identifier, so they match only where they start a token: after a name such as x their
# first word joins it, and "sine(" is then matched as an alias on its own.
# "(" is escaped by hand, since re.escape would also escape the space
_SPACED_CALL_ALIASES = sorted((alias for alias in CALL_ALIASES if " " in alias), key=lambda a: (-len(a), a))
_ALIAS_PATTERN = re.compile("|".join([alias + "\\(" for alias in _SPACED_CALL_ALIASES] + [_TOKEN]))


def _minimal_substrings(texts: List[str]) -> Tuple[str, ...]:
    """The texts that contain none of the others: any text occurs only where one of these does."""
    return tuple(text for text in texts if not any(other != text and other in text for other in texts))


# Text that must occur for replace_aliases to change anything ("sine(" also covers "cosine(")
_ALIAS_HINTS = _minimal_substrings(sorted([alias + "(" for alias in CALL_ALIASES] + list(WORD_ALIASES)))


def _replace_alias(match: re.Match[str]) -> str:
    token = match.group(0)
    if token[-1] == "(":  # a call alias with a space, matched with its "("
        return CALL_ALIASES[token[:-1]] + "("
    word_target = WORD_ALIASES.get(token)
    if word_target is not None:
        return word_target
    call_target = CALL_ALIASES.get(token)
    if call_target is not None and match.string[match.end() : match.end() + 1] == "(":
        return call_target
    return token


def replace_aliases(text: str) -> str:
    """Rewrite aliases as canonical names, each matched as a whole token.

    Call aliases change only when "(" follows: sine(x) -> sin(x), 2sine(x) -> 2sin(x).
    Word aliases change anywhere: ln(x) -> log(x), 2ln -> 2log. Names that merely contain an
    alias stay: kiln, ln2, x2ln(x), artifact(3), myvar(2).
    """
    if not any(hint in text for hint in _ALIAS_HINTS):
        return text
    return _ALIAS_PATTERN.sub(_replace_alias, text)


# Identifiers and number literals, matched whole (also used for implicit multiplication)
TOKEN_PATTERN = re.compile(_TOKEN)
# Text that must occur for rename_for_mathjs or calls_python_only to find anything
_RENAME_HINTS = tuple(sorted(name + "(" for name in MATHJS_RENAMES))
_PYTHON_ONLY_HINTS = _minimal_substrings(sorted(PYTHON_ONLY_NAMES))


def _rename_call(match: re.Match[str]) -> str:
    token = match.group(0)
    target = MATHJS_RENAMES.get(token)
    if target is not None and match.string[match.end() : match.end() + 1] == "(":
        return target
    return token


# The remainder operator with the spaces around it; math.js reads "mod" with the same precedence
_PERCENT = re.compile(r"\s*%\s*")


def rename_for_mathjs(text: str) -> str:
    """Rewrite text for math.js: calls of functions it knows under another name (trunc(x) ->
    fix(x)), and % as its mod operator.

    Every place that hands expression text to math.js calls this. Names are matched as whole
    tokens followed by "(", like the aliases, so xtrunc(x) stays. % always means the remainder,
    as in Python: math.js would read 7 % -3 as (7%) - 3, a percentage, so it becomes 7 mod -3.
    """
    if "%" in text:
        text = _PERCENT.sub(" mod ", text)
    if not any(hint in text for hint in _RENAME_HINTS):
        return text
    return TOKEN_PATTERN.sub(_rename_call, text)


def calls_python_only(text: str) -> bool:
    """True when text calls a function MathUtils.evaluate must compute in Python.

    The name must be a whole token followed by "(" (spaces allowed): is_prime(7) and
    randint (1, 2) do; cross_product(a, b) and a variable named product do not.
    """
    if not any(name in text for name in _PYTHON_ONLY_HINTS):
        return False
    for match in TOKEN_PATTERN.finditer(text):
        if match.group(0) in PYTHON_ONLY_NAMES and text[match.end() :].lstrip()[:1] == "(":
            return True
    return False
