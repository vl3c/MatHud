"""Text rewrites that prepare an expression for nerdamer, the symbolic engine.

nerdamer reads a name it does not know as a variable, so ln(x) becomes ln*x and its
derivative is "ln"; log(x, 2) keeps only the natural log when differentiated. These
helpers rewrite the common spellings nerdamer lacks into ones it knows, and list the
function names it would misread so the caller can refuse them with a clear message.

The solve tool's inequalities use five more: split_relation (the expressions and signs of
an inequality or a chain), parse_interval ("[0, 2*pi)" and the like),
insert_letter_call_products (x(x+2) as x*(x+2), which SymPy and math.js read as a call),
insert_implicit_products ((x-1)(x+2) and (x+1)2 as products, as SymPy reads them) and
split_top_level_terms (the signed terms of a sum).

Pure Python (no browser imports), so server-side pytest covers it.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

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
