"""Text rewrites that prepare an expression for nerdamer, the symbolic engine.

nerdamer reads a name it does not know as a variable, so ln(x) becomes ln*x and its
derivative is "ln"; log(x, 2) keeps only the natural log when differentiated. These
helpers rewrite the common spellings nerdamer lacks into ones it knows, and list the
function names it would misread so the caller can refuse them with a clear message.

Pure Python (no browser imports), so server-side pytest covers it.
"""

from __future__ import annotations

import re
from typing import Iterable, List

_IDENTIFIER_CALL = re.compile(r"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
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
