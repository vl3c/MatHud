"""Text rewrites that prepare an expression for nerdamer, the symbolic engine.

nerdamer reads a name it does not know as a variable, so ln(x) becomes ln*x and its
derivative is "ln"; log(x, 2) keeps only the natural log when differentiated. These
helpers rewrite the common spellings nerdamer lacks into ones it knows, and list the
function names it would misread so the caller can refuse them with a clear message.

The solve tool and check_steps use six more: split_relation (the expressions and signs of
an inequality or a chain), parse_interval ("[0, 2*pi)" and the like),
insert_letter_call_products (x(x+2) as x*(x+2), which SymPy and math.js read as a call),
insert_implicit_products ((x-1)(x+2) and (x+1)2 as products, as SymPy reads them),
split_top_level_terms (the signed terms of a sum), and misspellings (sin x, sinx and xe^x,
which the tools refuse with the standard form to write). taylor_polynomial and expand with
a variable also refuse an ambiguous_division (1/(x+1)(x+2)).

Pure Python (no browser imports), so server-side pytest covers it.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

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


def ambiguous_division(text: str, function_names: Iterable[str]) -> Optional[str]:
    """The divisor of the first "/" that another factor follows without "*", or None.

    In 1/(x+1)(x+2), 1/x(x+1), sin(2x)/2x and x/2e^x the divisor is followed directly by
    another factor: SymPy reads (1/(x+1))*(x+2), math.js divides by the product, and a
    person may mean either. A divisor is a bracket, a number, a name or a call to one of
    function_names (cos^2(x) included), with any powers (1/(x-1)^2(x+1) and 1/(x-1)²(x+1)
    are ambiguous too).
    """
    functions = set(function_names)
    index = text.find("/")
    while index >= 0:
        start = _next_non_space(text, index + 1)
        end = _powered_operand_end(text, start, functions)
        if end is not None:
            after = _next_non_space(text, end)
            if after < len(text) and (text[after] == "(" or text[after] == "." or _is_name_char(text[after])):
                return text[start:end]
        index = text.find("/", index + 1)
    return None


# Superscript exponents, as the symbol palette inserts them: (x+1)², x⁻¹
_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺"


def _powered_operand_end(text: str, start: int, functions: Set[str]) -> Optional[int]:
    """The end of the operand at start with its powers (x^2, (x+1)^-2, (x+1)²), or None."""
    end = _operand_end(text, start, functions)
    if end is None:
        return None
    return _powers_end(text, end, functions)


def _powers_end(text: str, end: int, functions: Set[str]) -> Optional[int]:
    """The end of the powers (^2, ^(n+1), ²) from end on: end itself when there are none."""
    while True:
        while end < len(text) and text[end] in _SUPERSCRIPTS:
            end += 1
        caret = _next_non_space(text, end)
        if caret >= len(text) or text[caret] != "^":
            return end
        exponent = _next_non_space(text, caret + 1)
        if exponent < len(text) and text[exponent] in "+-":
            exponent = _next_non_space(text, exponent + 1)
        following = _operand_end(text, exponent, functions)
        if following is None:
            return None
        end = following


def _operand_end(text: str, start: int, functions: Set[str]) -> Optional[int]:
    """The end of the bracket, number, name or function call at start, or None."""
    if start >= len(text):
        return None
    if text[start] == "(":
        closing = find_closing_parenthesis(text, start)
        return closing + 1 if closing >= 0 else None
    if text[start] in "0123456789.":
        end = start
        while end < len(text) and text[end] in "0123456789.":
            end += 1
        return end
    if not _is_letter(text[start]) and text[start] != "_":
        return None
    end = start
    # "²".isdigit() is true, but a superscript is a power, not part of the name
    while end < len(text) and _is_name_char(text[end]) and text[end] not in _SUPERSCRIPTS:
        end += 1
    if text[start:end] not in functions:
        # a longer name before "(" is a call to a function the engines may not know
        # (arctan(x)), not a product, which insert_letter_call_products reads only after one letter
        if end - start > 1 and end < len(text) and text[end] == "(":
            closing = find_closing_parenthesis(text, end)
            return closing + 1 if closing >= 0 else None
        return end
    # a call, with any power of the function between its name and "(": cos^2(x), sin²(x)
    powered = _powers_end(text, end, functions)
    call = _next_non_space(text, powered) if powered is not None else len(text)
    if call < len(text) and text[call] == "(":
        closing = find_closing_parenthesis(text, call)
        return closing + 1 if closing >= 0 else None
    return end


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


# ----- Spellings the tools refuse --------------------------------------------------------
# The tools take standard syntax only (sin(x), x*ln(x), x*e^x). Spellings a reader could take
# two ways (sin x, sinx, xlnx, xe^x) are never rewritten: misspellings finds them, so the
# caller can refuse them with the standard form to write.

# Identifiers and numbers, matched whole (expression_functions.TOKEN_PATTERN)
_TOKEN = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
# A power on a function name before its "(": sin^2(x), sin^-1(x)
_FUNCTION_POWER = re.compile(r"\^(?:\d+|-1|\(-1\))")
# The factors written after a function name without parentheses: sin x, sin 2x, sin pi x
_SPACED_OPERAND = re.compile(r"(?:\s*(?:\d+(?:\.\d+)?[A-Za-z]*|[A-Za-z][A-Za-z0-9]*))+")
# A power after e: xe^x, xe^(kt)
_POWER = re.compile(r"\^(?:\([^()]*\)|-?[A-Za-z0-9.]+)")
# A function glued to a one-letter operand, with an optional coefficient: sinx, sin2x
_GLUED_OPERAND = re.compile(r"(\d+(?:\.\d+)?)?([A-Za-z])$")
# One letter and e or pi written together: xe, ex, xpi, pix (not i, the imaginary unit)
_CONSTANT_PRODUCT = re.compile(r"(?:([a-df-hj-zA-Z])(e|pi)|(e|pi)([a-df-hj-zA-Z]))$")
# log2x and log10x: log2(x) or log(2x)?
_LOG_DIGITS = re.compile(r"^(log|ln)(\d+)([A-Za-z])$")
_LOGS = ("log", "ln")
# One letter, the longest function name (asinh), a number and one letter: longer names are words
_MAX_GLUED_LENGTH = 16
# Longer names are never split into pieces
_MAX_SPLIT_LENGTH = 40


class Misspelling(NamedTuple):
    """A spelling the tools refuse.

    Attributes:
        name: the name as written (sin, sinx, xlnx, xe).
        call: the standard form to write instead (sin(x), x*ln(x), x*e^x).
        reason: a sentence naming both.
        bare: True for a function written without parentheses (sin x, sin alone).
        letters: the one-letter variables a glued name holds (x for sinx, xlnx and xe).
        powered: True for e glued to a letter and raised to a power (xe^x).
    """

    name: str
    call: str
    reason: str
    bare: bool
    letters: Tuple[str, ...]
    powered: bool


def glued_call(
    name: str, glued_names: Iterable[str], function_names: Iterable[str] = ()
) -> Optional[Tuple[str, str, str]]:
    """(prefix, function, operand) when name looks like a function glued to a one-letter
    operand, with at most one letter before it: sinx -> ("", "sin", "x"), sin2x -> ("", "sin",
    "2x"), xlnx -> ("x", "ln", "x"); None for any other name (costs, seconds, cos_theta), for a
    function name (sech is not sec(h)) and for a digit after a log (log2x). The letter before is
    never the start of a function name (asinhx is asinh and x).
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
                continue
            if function in _LOGS and operand[0].isdigit():
                return None
            return prefix, function, operand
    return None


def misspellings(
    text: str,
    unary_names: Iterable[str],
    glued_names: Iterable[str],
    function_names: Iterable[str] = (),
    keep: Iterable[str] = (),
) -> List[Misspelling]:
    """The spellings in text the tools refuse, in order: a function in unary_names written
    without parentheses (sin x, sin 2x, x ln x, sin^2 x, sin alone), a name that looks like a
    function glued to a letter (sinx, sin2x, xlnx, asinhx), a letter and e or pi written
    together (xe^x, xpi), a log glued to digits (log2x), and any other name that starts with
    a function name (sqrt2, ln2, sinxe, sinpix, costs). Names in keep (the variable being
    solved for), in function_names, with "_" and after a "." are never misspellings; ln|x| and
    sin (x) are standard.
    """
    unary, glued, kept = set(unary_names), set(glued_names), set(keep)
    functions = set(function_names) | unary
    prefixes = sorted(functions | glued, key=len, reverse=True)
    found: List[Misspelling] = []
    for match in _TOKEN.finditer(text):
        name = match.group(0)
        if name[0].isdigit() or name in kept or (match.start() > 0 and text[match.start() - 1] == "."):
            continue
        if name in unary:
            misspelling = _bare_call(text, name, match.end(), functions)
        elif name in functions or text[match.end() : match.end() + 1] == "(":
            continue
        else:
            misspelling = _glued_name(text, name, match.end(), glued, functions)
            if misspelling is None:
                misspelling = _function_prefix(name, prefixes)
        if misspelling is not None:
            found.append(misspelling)
    return found


def _bare_call(text: str, name: str, end: int, functions: Set[str]) -> Optional[Misspelling]:
    """sin x, sin^2 x, sin alone: name, ending at end, has no "(" (or the bar of ln|x|) after
    it. The call in the hint takes the factors after it, up to the next function."""
    power = _FUNCTION_POWER.match(text, end)
    power_text = power.group(0) if power is not None else ""
    after = power.end() if power is not None else end
    if _next_character(text, after) in ("(", "|"):
        return None
    operand = _SPACED_OPERAND.match(text, after)
    factors: List[str] = []
    # re.findall, not str.split(): Brython's split() keeps a leading space here
    for factor in re.findall(r"[^\s]+", operand.group(0)) if operand is not None else []:
        if factor in functions:
            break
        factors.append(factor)
    following = _next_character(text, operand.end() if operand is not None else after)
    unclear = following in ("*", "/", "^", "(") or (len(factors) > 1 and not _is_coefficient(factors[0]))
    if factors and unclear:
        # sin x/2, sin 2*x, sin x y: sin(x)/2 or sin(x/2)? The writer says.
        call = f"{name}{power_text}(...)"
        return Misspelling(name, call, f"{name} needs parentheses around its argument; write {call}", True, (), False)
    call = f"{name}{power_text}({'*'.join(factors[:2]) or 'x'})"
    return Misspelling(name, call, f"{name} needs parentheses; write {call}", True, (), False)


def _next_character(text: str, index: int) -> str:
    """The first character at or after index that is not a space ("" at the end)."""
    rest = text[index:].lstrip()
    return rest[:1]


def _is_coefficient(factor: str) -> bool:
    """True for a number or pi, which come first in a product (sin 2 x, sin pi x)."""
    return factor == "pi" or factor[0].isdigit()


# The hint for a name that starts with a function name and is no glued call
NAME_HINT = "write it in standard syntax (e.g. sqrt(2), sin(x)*e^x) or rename it"


def _function_prefix(name: str, prefixes: Sequence[str]) -> Optional[Misspelling]:
    """sqrt2 (write sqrt(2)), sinxe, sinpix, costs: a name that starts with a function name in
    prefixes (longest first), as a Misspelling with no letters; None for other names. A call
    with digits after it (sqrt2) has its call in the hint; any other has call "" (costs may be
    a variable)."""
    if "_" in name:
        return None
    function = next((prefix for prefix in prefixes if len(name) > len(prefix) and name.startswith(prefix)), None)
    if function is None:
        return None
    rest = name[len(function) :]
    if rest.isdigit():
        call = f"{function}({rest})"
        return Misspelling(name, call, f"{name} is read as one variable; write {call}", False, (), False)
    return Misspelling(name, "", f"{name} is read as one variable; {NAME_HINT}", False, (), False)


def decomposes(name: str, pieces: Iterable[str], required: Iterable[str]) -> bool:
    """True when name is two or more of pieces and digit runs written together, at least one
    of them in required: sinxe (sin, x, e), sin2, exe, sinxcosx."""
    if len(name) > _MAX_SPLIT_LENGTH:
        return False
    parts, needed = sorted(set(pieces), key=len, reverse=True), set(required)
    # found[i]: (pieces in name[i:], whether one is required), or None; from the end
    found: Dict[int, Optional[Tuple[int, bool]]] = {len(name): (0, False)}
    for start in range(len(name) - 1, -1, -1):
        found[start] = None
        candidates = [part for part in parts if name.startswith(part, start)]
        digits = re.match(r"\d+", name[start:])
        if digits is not None:
            candidates.append(digits.group(0))
        for part in candidates:
            rest = found.get(start + len(part))
            if rest is None:
                continue
            option = (rest[0] + 1, rest[1] or part in needed)
            current = found[start]
            if current is None or (option[1] and not current[1]):
                found[start] = option
    result = found[0]
    return result is not None and result[0] >= 2 and result[1]


def _glued_name(text: str, name: str, end: int, glued: Set[str], functions: Set[str]) -> Optional[Misspelling]:
    """sinx, xlnx, xe^x, xpi, log2x as Misspellings; None for other names."""
    parts = glued_call(name, glued, functions)
    if parts is not None:
        prefix, function, operand = parts
        call = f"{prefix + '*' if prefix else ''}{function}({operand})"
        letters = tuple(dict.fromkeys(prefix + operand[-1]))
        return Misspelling(name, call, f"{name} is read as one variable; write {call}", False, letters, False)
    logs = _LOG_DIGITS.match(name)
    if logs is not None:
        function, digits, letter = logs.groups()
        # log2 and log10 are functions, ln2 is not
        second = f"ln({digits})*{letter}" if function == "ln" else f"{function}{digits}({letter})"
        call = f"{second} or {function}({digits}{letter})"
        return Misspelling(name, call, f"{name} is read as one variable; write {call}", False, (letter,), False)
    constant = _CONSTANT_PRODUCT.match(name)
    if constant is None:
        return None
    first, second = (constant.group(1), constant.group(2)) if constant.group(1) else constant.group(3, 4)
    power = _POWER.match(text, end) if second == "e" else None
    call = f"{first}*{second}{power.group(0) if power is not None else ''}"
    letter = second if first in ("e", "pi") else first
    reason = f"{name} is read as one variable; write {call}"
    return Misspelling(name, call, reason, False, (letter,), power is not None)
