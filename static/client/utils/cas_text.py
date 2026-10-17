"""Text rewrites that prepare an expression for nerdamer, the symbolic engine.

nerdamer reads a name it does not know as a variable, so ln(x) becomes ln*x and its
derivative is "ln"; log(x, 2) keeps only the natural log when differentiated. These
helpers rewrite the common spellings nerdamer lacks into ones it knows, and list the
function names it would misread so the caller can refuse them with a clear message.

The solve tool's inequalities and check_steps use seven more: split_relation (the expressions and signs of
an inequality or a chain), parse_interval ("[0, 2*pi)" and the like),
insert_letter_call_products (x(x+2) as x*(x+2), which SymPy and math.js read as a call),
insert_implicit_products ((x-1)(x+2) and (x+1)2 as products, as SymPy reads them),
split_top_level_terms (the signed terms of a sum), and insert_calculator_calls and
split_constant_products (sin x, sinx and xe^x read as calculators read them).

Pure Python (no browser imports), so server-side pytest covers it.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set, Tuple

# A name followed by "(". A digit may come before it (2ln(x) is 2*ln(x)); a letter, "_" or "."
# may not, and since matches are found left to right, a name with digits inside (log10, x2ln)
# is matched whole from its first letter.
_IDENTIFIER_CALL = re.compile(r"(?<![A-Za-z_.])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_INEQUALITY = re.compile(r"<|>|!=|≤|≥|≠")

# Spellings nerdamer reads as a variable times its argument, and their nerdamer form.
_RENAMED_FUNCTIONS = {"ln": "log", "sgn": "sign"}


def find_closing_parenthesis(text: str, open_index: int) -> int:
    """Index of the parenthesis closing text[open_index] == '(', or -1 when unbalanced."""
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


def split_top_level_commas(text: str) -> List[str]:
    """Split text on the commas that are not nested inside brackets."""
    parts: List[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def rewrite_for_nerdamer(expression: str) -> str:
    """Rewrite ln, sgn, log2(a) and two-argument log(a, b) into the forms nerdamer knows.

    ln(a) -> log(a) (nerdamer's log is natural), sgn(a) -> sign(a), log2(a) -> (log(a)/log(2))
    and log(a, b) -> (log(a)/log(b)). Nested calls are rewritten too; anything else is
    returned unchanged.
    """
    output: List[str] = []
    position = 0
    for match in _IDENTIFIER_CALL.finditer(expression):
        if match.start() < position:
            continue  # inside a call already rewritten
        name = match.group(1)
        if name not in _RENAMED_FUNCTIONS and name not in ("log", "log2"):
            continue
        open_index = match.end() - 1
        close_index = find_closing_parenthesis(expression, open_index)
        if close_index < 0:
            continue
        arguments = [
            rewrite_for_nerdamer(argument)
            for argument in split_top_level_commas(expression[open_index + 1 : close_index])
        ]
        output.append(expression[position : match.start()])
        output.append(_rewritten_call(name, arguments))
        position = close_index + 1
    output.append(expression[position:])
    return "".join(output)


def _rewritten_call(name: str, arguments: List[str]) -> str:
    if name == "log2" and len(arguments) == 1:
        return f"(log({arguments[0]})/log(2))"
    if name == "log" and len(arguments) == 2:
        return f"(log({arguments[0]})/log({arguments[1]}))"
    return f"{_RENAMED_FUNCTIONS.get(name, name)}({', '.join(arguments)})"


def unknown_function_names(expression: str, known_functions: Iterable[str]) -> List[str]:
    """Names called like functions that nerdamer does not define, in order of appearance.

    One-letter names are left out: nerdamer reads x(x+1) as x*(x+1), as a person would.
    """
    known = set(known_functions)
    unknown: List[str] = []
    for match in _IDENTIFIER_CALL.finditer(expression):
        name = match.group(1)
        if len(name) > 1 and name not in known and name not in unknown:
            unknown.append(name)
    return unknown


def has_inequality(expression: str) -> bool:
    """True when the text holds an inequality sign (<, >, <=, >=, !=, ≤, ≥, ≠)."""
    return bool(_INEQUALITY.search(expression))


def is_unevaluated(result: str) -> bool:
    """True when nerdamer handed back a call it could not carry out, e.g. integrate(sin(x^2),x)."""
    return bool(re.search(r"\b(?:integrate|diff|limit|defint)\(", result))


def _is_letter(char: str) -> bool:
    """True for an ASCII or Greek letter (the Greek and Coptic block, U+0370 to U+03FF)."""
    return ("a" <= char.lower() <= "z") or ("Ͱ" <= char <= "Ͽ")


def _is_name_char(char: str) -> bool:
    return _is_letter(char) or char.isdigit() or char == "_"


def insert_letter_call_products(text: str, function_names: Iterable[str]) -> str:
    """Write a one-letter name before "(" as a product: x(x+2) -> x*(x+2), 2a(b+1) -> 2a*(b+1).

    SymPy's parser and math.js both read x(x+2) as a call of a function x. The name must be
    a single ASCII or Greek letter (a longer name such as exp or x2 is left alone; digits
    before it are a coefficient) and not one of function_names.
    """
    functions = set(function_names)
    output: List[str] = []
    for index, char in enumerate(text):
        output.append(char)
        if not _is_letter(char) or char in functions or not text[index + 1 :].lstrip().startswith("("):
            continue
        start = index
        while start > 0 and _is_name_char(text[start - 1]):
            start -= 1
        while start < index and text[start].isdigit():
            start += 1  # 2a: a coefficient, not part of the name
        if start == index and not text[index - 1 : index] == ".":
            output.append("*")
    return "".join(output)


def insert_implicit_products(text: str) -> str:
    """Write the implicit products after ")" and after a number followed by spaces as explicit ones.

    (x-1)(x+2), (x+1)x, (x+1)2 and |x-1|(x+1) (as abs(x-1)(x+1)) get "*" after ")"; so does
    a number before a name or "(" across spaces, as in (x-2)^2 (x+1). SymPy's parser reads
    these as products (sympy_cas._prepared_text); math.js and the expression validator do
    not. A number directly before a name (2x) is left as it is, as is an exponent (1e5).
    """
    output: List[str] = []
    index = 0
    count = len(text)
    while index < count:
        char = text[index]
        if char == ")":
            following = _next_non_space(text, index + 1)
            output.append(")")
            if following < count and (
                _is_letter(text[following]) or text[following].isdigit() or text[following] == "("
            ):
                output.append("*")
                index = following
                continue
            index += 1
            continue
        if char.isdigit() and (index == 0 or not (_is_name_char(text[index - 1]) or text[index - 1] == ".")):
            end = _number_end(text, index)
            output.append(text[index:end])
            following = _next_non_space(text, end)
            if following > end and following < count and (_is_letter(text[following]) or text[following] == "("):
                output.append("*")
                index = following
                continue
            index = end
            continue
        output.append(char)
        index += 1
    return "".join(output)


def split_top_level_terms(text: str) -> List[Tuple[int, str]]:
    """The terms of a sum, each with its sign: "x^3 - 2x^2 + x" gives [(1, "x^3"), (-1, "2x^2"), (1, "x")].

    Only + and - outside brackets split, and only as binary operators: not in an exponent
    (1e-10), nor after an operator (2*-x, x^-1). A term's leading signs fold into its sign
    (-(x) - -1 gives [(-1, "(x)"), (1, "1")]). Text that does not read as a sum of
    non-empty terms is one term with sign 1.
    """
    terms: List[Tuple[int, str]] = []
    depth = 0
    sign = 1
    start = 0
    index = 0
    count = len(text)
    while index < count and text[index] in "+- ":
        if text[index] == "-":
            sign = -sign
        index += 1
    start = index
    while index < count:
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char in "+-" and depth == 0 and _ends_operand(text, start, index):
            term = text[start:index].strip()
            if not term:
                return [(1, text.strip())]
            terms.append((sign, term))
            sign = 1
            while index < count and text[index] in "+- ":
                if text[index] == "-":
                    sign = -sign
                index += 1
            start = index
            continue
        index += 1
    term = text[start:].strip()
    if not term or depth != 0:
        return [(1, text.strip())]
    terms.append((sign, term))
    return terms


def _ends_operand(text: str, start: int, index: int) -> bool:
    """Whether the sign at index follows an operand of the term starting at start (so it is
    binary): a name, a number other than an exponent's mantissa, ")" or "]"."""
    before = index - 1
    while before >= start and text[before] == " ":
        before -= 1
    if before < start:
        return False
    previous = text[before]
    exponent_digit = index + 1 < len(text) and text[index + 1].isdigit()
    if (
        previous in "eE"
        and exponent_digit
        and before == index - 1
        and before > start
        and text[before - 1] in "0123456789."
    ):
        number_start = before - 1
        while number_start > start and (text[number_start - 1].isdigit() or text[number_start - 1] == "."):
            number_start -= 1
        if number_start == start or not _is_name_char(text[number_start - 1]):
            return False  # 1e-10, 2.5E+3: an exponent
    return _is_name_char(previous) or previous in ")]." or previous in "²³!"


def _next_non_space(text: str, index: int) -> int:
    while index < len(text) and text[index] == " ":
        index += 1
    return index


def _number_end(text: str, start: int) -> int:
    """The end of the number starting at start: digits, a decimal part and an exponent (1.5e-3)."""
    index = start
    while index < len(text) and text[index].isdigit():
        index += 1
    if index < len(text) and text[index] == ".":
        index += 1
        while index < len(text) and text[index].isdigit():
            index += 1
    if index < len(text) and text[index] in "eE":
        exponent = index + 1
        if exponent < len(text) and text[exponent] in "+-":
            exponent += 1
        if exponent < len(text) and text[exponent].isdigit():
            index = exponent
            while index < len(text) and text[index].isdigit():
                index += 1
    return index


# Inequality signs as split_relation reads them, longest first: "=<" and "=>" are aliases.
_RELATION_SIGNS = (
    ("<=", "<="),
    (">=", ">="),
    ("=<", "<="),
    ("=>", ">="),
    ("≤", "<="),
    ("≥", ">="),
    ("<", "<"),
    (">", ">"),
)
_LOGICAL_WORD = re.compile(r"(?<![A-Za-z_])(?:and|or)(?![A-Za-z_])")


def split_relation(text: str) -> Tuple[List[str], List[str]]:
    """(the expressions, the signs) of an inequality, or of a chain of two signs pointing the same way.

    "1 < 2x+1 <= 5" gives (["1", "2x+1", "5"], ["<", "<="]). The signs are <, <=, > and >=;
    =< and => are read as <= and >=.

    Raises:
        ValueError: for !=, ==, a lone =, "and"/"or", more than two signs, signs pointing
            different ways or a missing side, with a message saying what to write instead.
    """
    if "!=" in text or "≠" in text:
        raise ValueError("solve does not take '!=' (not equal): solve the equation with '=' and exclude its roots.")
    if "==" in text:
        raise ValueError("write an equation with a single '=', or an inequality with <, <=, > or >=.")
    if _LOGICAL_WORD.search(text) or "&" in text:
        raise ValueError(
            "solve takes one inequality, or a chain such as 1 < 2x+1 <= 5; solve each part of an "
            "'and' or 'or' on its own and combine the solution sets."
        )
    parts: List[str] = []
    signs: List[str] = []
    start = 0
    index = 0
    while index < len(text):
        sign = next(((spelling, name) for spelling, name in _RELATION_SIGNS if text.startswith(spelling, index)), None)
        if sign is None:
            if text[index] == "=":
                raise ValueError("an inequality cannot also hold '='; solve an equation and an inequality separately.")
            index += 1
            continue
        parts.append(text[start:index].strip())
        signs.append(sign[1])
        index += len(sign[0])
        start = index
    parts.append(text[start:].strip())
    if not signs:
        raise ValueError("an inequality needs one of <, <=, > or >=.")
    if len(signs) > 2:
        raise ValueError("an inequality has at most two signs, as in the chain 1 < 2x+1 <= 5.")
    if len({sign[0] for sign in signs}) > 1:
        raise ValueError("the two signs of a chain must point the same way, as in 1 < 2x+1 <= 5.")
    if any(not part for part in parts):
        raise ValueError(f"an expression is missing on one side of {signs[0]}.")
    return parts, signs


def parse_interval(text: str) -> Tuple[str, str, bool, bool]:
    """(lower, upper, lower closed?, upper closed?) of "[a, b]", "(a, b)", "[a, b)", "(a, b]" or "a, b" (closed).

    Raises:
        ValueError: unless the text is two comma-separated ends, with matching brackets or none.
    """
    stripped = text.strip()
    lower_closed = upper_closed = True
    bracketed = stripped[:1] in ("[", "(") and stripped[-1:] in ("]", ")")
    if bracketed and stripped[0] == "(" and find_closing_parenthesis(stripped, 0) not in (-1, len(stripped) - 1):
        bracketed = False  # (1+2), 3: the parentheses belong to the first end
    if bracketed:
        lower_closed, upper_closed = stripped[0] == "[", stripped[-1] == "]"
        stripped = stripped[1:-1]
    elif stripped[:1] == "[" or stripped[-1:] == "]":
        raise ValueError(f"write the interval as [a, b], (a, b), [a, b) or (a, b], not {text!r}.")
    ends = split_top_level_commas(stripped)
    if len(ends) != 2 or not all(ends):
        raise ValueError(f"an interval has two ends separated by a comma, such as [0, 2*pi], not {text!r}.")
    return ends[0], ends[1], lower_closed, upper_closed


# Identifiers and numbers, matched whole (expression_functions.TOKEN_PATTERN)
_TOKEN = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_NUMBER = re.compile(r"\d+(?:\.\d+)?|\.\d+")
# A power on a function name before its operand: sin^2 x, sin^-1 x (asin x)
_FUNCTION_POWER = re.compile(r"\^(?:\d+|-1|\(-1\))")
# A power on a factor of the operand: sin x^2, sin x^y, sin x^(2y)
_FACTOR_POWER = re.compile(r"\^(?:\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9]*|(?=\())")
# A function glued to a one-letter operand, with an optional coefficient: sinx, sin2x
_GLUED_OPERAND = re.compile(r"(\d+(?:\.\d+)?)?([A-Za-z])$")
# One letter and e or pi written together: xe, ex, xpi, pix (not i, the imaginary unit)
_CONSTANT_PRODUCT = re.compile(r"(?:([a-df-hj-zA-Z])(e|pi)|(e|pi)([a-df-hj-zA-Z]))$")
_COEFFICIENTS = ("pi", "e")
# A digit after these is part of a name (log2, log10), so log2x is left unread
_LOGS = ("log", "ln")
# One letter, the longest function name (asinh), a number and one letter: longer names stay
_MAX_GLUED_LENGTH = 16


def glued_call(
    name: str, glued_names: Iterable[str], function_names: Iterable[str] = ()
) -> Optional[Tuple[str, str, str]]:
    """(prefix, function, operand) when name is a function glued to a one-letter operand,
    with at most one letter before it: sinx -> ("", "sin", "x"), sin2x -> ("", "sin", "2x"),
    xlnx -> ("x", "ln", "x"); None for any other name (costs, seconds, cos_theta), for a
    function name (sech is not sec(h)) and for a digit after a log (log2x: log2(x) or log(2x)?).
    The letter before is never the start of a function name: asinhx is not a*sinh(x).
    """
    known = set(function_names)
    if "_" in name or name in known or len(name) > _MAX_GLUED_LENGTH:
        return None
    functions = sorted(glued_names, key=len, reverse=True)
    for prefix in ("", name[:1]):
        if prefix and not prefix.isalpha():
            break
        body = name[len(prefix) :]
        for function in functions:
            operand = body[len(function) :]
            if not body.startswith(function) or not _GLUED_OPERAND.match(operand):
                continue
            if prefix and prefix + function in known:
                continue  # asinhx: not a*sinh(x)
            if function in _LOGS and operand[0].isdigit():
                return None
            return prefix, function, operand
    return None


# Greek letters spelled out: one-letter variables, not names written as words
_GREEK_NAMES = frozenset(
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron rho sigma tau"
    " upsilon phi chi psi omega".split()
)


def has_word_names(names: Iterable[str], glued_names: Iterable[str], function_names: Iterable[str] = ()) -> bool:
    """True when a name is written as a word (profit, revenue, cos_theta): three letters or
    more, and no function, Greek letter, glued call (cost) or constant product (xpi). Beside
    such names cost and secs are variables too, so glued calls are not read."""
    glued, functions = set(glued_names), set(function_names)
    for name in names:
        if sum(character.isalpha() for character in name) < 3 or name in functions or name in glued:
            continue
        if name in _GREEK_NAMES or _CONSTANT_PRODUCT.match(name) is not None:
            continue
        if glued_call(name, glued, functions) is None:
            return True
    return False


def insert_calculator_calls(
    text: str,
    unary_names: Iterable[str],
    glued_names: Iterable[str],
    keep: Iterable[str] = (),
    function_names: Iterable[str] = (),
) -> str:
    """Read a one-argument function written without parentheses as calculators do.

    sin x -> sin(x), sin 2x -> sin(2x), sin pi x -> sin(pi*x), sin 2.5 x -> sin(2.5*x),
    2 sin x cos x -> 2*sin(x)*cos(x), sin^2 x -> sin^2(x), sin^-1 x -> sin^-1(x) (asin),
    sin x^2 -> sin(x^2), x ln x -> x*ln(x), sin cos x -> sin(cos(x)). A name in glued_names
    may touch a one-letter operand, with one letter before it: sinx -> sin(x), sin2x ->
    sin(2x), xlnx -> x*ln(x), sinx/x -> sin(x)/x (but sinx(x+1) and sinx^-1 are left), except
    beside names written as words: in revenue - cost, cost is a variable (has_word_names). The
    operand is an implicit product (numbers, pi, e and names), with powers; it stops before
    another function. Where the textbook reading is not clear the text is left as it is, for
    the engines to refuse: sin x y, sin 2*x, sin x/2, sin x(x+1). So is text that already has its parentheses (sin(x), sin^2(x)), a name with
    "_", names in keep (a variable such as cost) and, as a whole, function_names (sech, coth:
    never sec(h) or cot(h)).
    """
    unary, glued, kept = set(unary_names), set(glued_names), set(keep)
    functions = set(function_names) | unary
    if not any(name in text for name in unary | glued):
        return text
    tokens = [match for match in _TOKEN.finditer(text) if not match.group(0)[0].isdigit()]
    if has_word_names([match.group(0) for match in tokens], glued, functions):
        glued = set()  # revenue - cost: cost is a variable too
    for match in reversed(tokens):
        text = _calculator_call(text, match.start(), match.group(0), unary, glued, kept, functions)
    return text


def _calculator_call(
    text: str, start: int, name: str, unary: Set[str], glued: Set[str], kept: Set[str], functions: Set[str]
) -> str:
    """text with the function name at start applied to its operand, or text unchanged."""
    if name in kept or "_" in name or (start > 0 and text[start - 1] == "."):
        return text
    end = start + len(name)
    if name in unary:
        power, operand_start = "", end
        power_match = _FUNCTION_POWER.match(text, end)
        if power_match is not None:
            power, operand_start = power_match.group(0), power_match.end()
        spaced = _next_non_space(text, operand_start)
        if spaced < len(text) and text[spaced] == "(":
            return text  # sin(x), sin (x), sin^2(x): already a call
        if spaced == operand_start and not power:
            return text  # sin, sin+1, sin*x: not a call
        operand = _operand(text, spaced, unary)
        if operand is None:
            return text
        call, operand_end = f"{name}{power}({operand[0]})", operand[1]
    else:
        parts = glued_call(name, glued, functions)
        if parts is None:
            return text
        prefix, function, glued_operand = parts
        operand_end = _power_end(text, end)
        if _unclear_after(text, operand_end, unary, glued=True):
            return text
        call = f"{function}({glued_operand}{text[end:operand_end]})"
        if prefix:
            call = f"{prefix}*{call}"
    return _joined_before(text, start, unary) + call + text[operand_end:]


def _operand(text: str, position: int, unary: Set[str]) -> Optional[Tuple[str, int]]:
    """(operand, end) for the implicit product at position, or None when there is none or its
    end is not clear (sin x y, sin 2*x). Factors after a space join with "*": sin pi x -> pi*x."""
    factors: List[str] = []
    index = position
    while True:
        factor_start = index
        number = _NUMBER.match(text, index)
        if number is not None:
            index = number.end()
        token = _TOKEN.match(text, index)
        name = token.group(0) if token is not None and not token.group(0)[0].isdigit() else ""
        name_end = token.end() if token is not None else index
        if name in unary:
            if index > factor_start or factors or text[name_end : name_end + 1] != "(":
                break  # a function: the operand ends before it (sin x cos x)
            index = _closing_index(text, name_end)
            if index < 0:
                return None
        elif name:
            index = name_end
        if index == factor_start:
            break
        index = _power_end(text, index)
        factors.append(text[factor_start:index])
        following = _next_non_space(text, index)
        if following == index or following >= len(text):
            break
        if not _starts_factor(text, following, unary):
            break
        if not (name in _COEFFICIENTS or (number is not None and not name)):
            return None  # sin x y: x times y, or sin(x) times y?
        index = following
    if not factors or _unclear_after(text, index, unary):
        return None
    return "*".join(factors), index


def _starts_factor(text: str, index: int, unary: Set[str]) -> bool:
    """True when a number or a name that is not a function starts at index."""
    token = _TOKEN.match(text, index)
    return token is not None and token.group(0) not in unary


def _unclear_after(text: str, index: int, unary: Set[str], glued: bool = False) -> bool:
    """True when the operand ending at index could go on: sin 2*x, sin x/2, sin x(x+1), sin x^-1.
    A glued operand ends with its letter, so only ( and ^ make it unclear: sinx/x is sin(x)/x."""
    following = _next_non_space(text, index)
    if following < len(text) and text[following] in "(^":
        return True  # sin x(x+1), sin x^-1
    if glued or following >= len(text) or text[following] not in "*/":
        return False
    after = _next_non_space(text, following + 1)
    return _starts_factor(text, after, unary) or text[after : after + 1] == "("


def _power_end(text: str, index: int) -> int:
    """The end of a power at index (x^2, x^y, x^(2y)), or index when there is none."""
    power = _FACTOR_POWER.match(text, index)
    if power is None:
        return index
    if power.group(0) != "^":
        return power.end()
    close = _closing_index(text, index + 1)
    return index if close < 0 else close


def _closing_index(text: str, open_index: int) -> int:
    """The index after the ")" matching the "(" at open_index, or -1."""
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def _joined_before(text: str, start: int, unary: Set[str]) -> str:
    """text before start, with the spaces before it written as "*" after a factor: x ln x ->
    x*ln(x). After a function name (sin cos x) or an operator the text is kept."""
    before = start
    while before > 0 and text[before - 1] == " ":
        before -= 1
    if before == start or before == 0 or not (_is_name_char(text[before - 1]) or text[before - 1] == ")"):
        return text[:start]
    previous = [match.group(0) for match in _TOKEN.finditer(text, 0, before)]
    if previous and text[before - 1] != ")" and previous[-1] in unary:
        return text[:start]
    return text[:before] + "*"


def split_constant_products(text: str, reserved: Iterable[str], keep: Iterable[str] = ()) -> str:
    """Write a letter and e or pi written together as a product, as calculators read them.

    xe^x -> x*e^x, 2xe^(x^2) -> 2x*e^(x^2), xpi -> x*pi, pix -> pi*x. Only two-part names
    (one letter other than e or i, and e or pi) change; reserved names (re), names in keep,
    longer names (time, rate) and calls (ex(2)) stay.
    """
    if "e" not in text and "pi" not in text:
        return text
    skip = set(reserved) | set(keep)
    output: List[str] = []
    last = 0
    for match in _TOKEN.finditer(text):
        token = match.group(0)
        parts = _CONSTANT_PRODUCT.match(token)
        if parts is None or token in skip or text[match.end() : match.end() + 1] == "(":
            continue
        if match.start() > 0 and text[match.start() - 1] == ".":
            continue
        first, second = (parts.group(1), parts.group(2)) if parts.group(1) else (parts.group(3), parts.group(4))
        output.append(text[last : match.start()] + f"{first}*{second}")
        last = match.end()
    return "".join(output) + text[last:]
