"""The check_steps tool: is each step of a derivation equivalent to the one before?

Each step is an expression in standard syntax. Steps go through the input pipeline the
solve tool uses (InequalitySolver.prepared: Unicode notation, sin^2(x), ln, |x| bars, x(x+2)
and (x-1)(x+2) as products), so SymPy and math.js read the same text. Calculator spellings
are never rewritten. Then, for each pair of neighbouring steps:

1. A function written without parentheses (sin x, sin alone) makes the pair undecided, with
   the call to write (cas_text.misspellings).
2. A step neither engine can read makes the pair undecided, with the parse message.
3. A pair with a name the engines misread is undecided, with a hint: another function name
   used as a variable (gamma), a function glued to more than one letter (sinxy, sinxcosx),
   or xy against x and y.
4. A name that looks like a glued call or constant (sinx, xlnx, xe, xpi, cost) or starts
   with a function name (sqrt2, sinxe, costs) is read as one variable. If the steps are
   equivalent that way they are equivalent however it is read, with a note for a glued
   call; otherwise the pair is undecided with the form to write (sin(x), x*ln(x), x*e^x,
   sqrt(2)). Such a pair is never not_equivalent, and neither is one with a name of two
   letters or more that only one step has (exe, mx): it is undecided with a hint.
5. The numeric check (utils.equivalence.compare, every variable varied) decides most
   pairs. "equivalent" is the answer, with a note when one step is undefined where the
   other is defined (log(x^2) and 2*log(x)), and SymPy is not asked.
6. Otherwise the server's SymPy engine is asked (operation "equivalent"), within a budget of
   SYMPY_BUDGET_S seconds for the whole derivation, and the two verdicts combine:

   | SymPy           | numeric                    | result                          |
   | equivalent      | undecided                  | equivalent                      |
   | equivalent      | not_equivalent             | undecided (the checks disagree) |
   | not_equivalent  | not_equivalent, undecided  | not_equivalent                  |
   | unknown or none | any                        | the numeric verdict             |

   A not_equivalent step's reason gives SymPy's difference when it has one
   ("step 2 - step 1 = -2*x (not 0)") and a readable point where the values differ.

The browser parts (math.js, SymPy requests) are imported when first used, so pytest
covers the module with Python stand-ins for them.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, cast

from utils.cas_text import NAME_HINT, has_inequality, split_top_level_terms
from utils.equivalence import (
    EQUIVALENT,
    NOT_EQUIVALENT,
    UNDECIDED,
    EquivalenceReport,
    Evaluator,
    Point,
    compare,
    distinct_value_texts,
    json_number,
    point_text,
)

MIN_STEPS = 2
MAX_STEPS = 12
MAX_STEP_LENGTH = 1000
# SymPy time for one derivation; the pairs after it are checked numerically only.
SYMPY_BUDGET_S = 12.0

# How each pair was decided
NUMERIC = "numeric"
SYMBOLIC = "symbolic"
BOTH = "symbolic+numeric"
NEITHER = "none"

_EQUATION_REFUSAL = (
    "check_steps takes expressions; equations and inequalities come next. For a chain a = b = c, pass a, b and c."
)
_SINGLE_LETTER = re.compile(r"^[A-Za-zͰ-Ͽ]$")


_CONSTANTS = ("e", "pi")
# Longer names are never split into factors or calls
_MAX_SPLIT_LENGTH = 40
# Greek letters spelled out: one-letter variables
_GREEK_NAMES = frozenset(
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron rho sigma tau"
    " upsilon phi chi psi omega".split()
)
# A letter and digits, such as x2: x squared or 2x when x is a variable too
_INDEXED = re.compile(r"^([A-Za-z])(\d+)$")


def _calculator_spellings() -> FrozenSet[str]:
    """The functions calculators apply without parentheses (sin, ln, sqrt): never a variable."""
    from expression_functions import CALCULATOR_SPELLINGS

    return frozenset(CALCULATOR_SPELLINGS)


def _function_spellings() -> FrozenSet[str]:
    """Every spelling of a function (sin, ln, sine, gamma, max)."""
    from expression_functions import BY_NAME, CALL_ALIASES, WORD_ALIASES

    names: Set[str] = set(BY_NAME) | set(_calculator_spellings()) | set(CALL_ALIASES) | set(WORD_ALIASES)
    return frozenset(names)


def _glued_names() -> List[str]:
    """The calculator functions a name may be glued to (sinx, sinxy, sinxcosx, sechx), longest
    first."""
    from expression_functions import GLUED_CALL_NAMES

    return sorted(GLUED_CALL_NAMES, key=len, reverse=True)


def _pieces(name: str, letters: Set[str]) -> Optional[List[str]]:
    """name split into letters, e and pi (xye -> [x, y, e]), or None (also for a name longer
    than _MAX_SPLIT_LENGTH)."""
    if len(name) > _MAX_SPLIT_LENGTH:
        return None
    parts = sorted(letters | set(_CONSTANTS), key=len, reverse=True)
    # found[i]: the pieces of name[i:], or None; worked from the end, so it takes linear time
    found: Dict[int, Optional[List[str]]] = {len(name): []}
    for start in range(len(name) - 1, -1, -1):
        found[start] = None
        for part in parts:
            rest = found.get(start + len(part)) if name.startswith(part, start) else None
            if rest is not None:
                found[start] = [part] + rest
                break
    return found[0]


def _call_pieces(name: str, letters: Set[str]) -> Optional[List[str]]:
    """name split into glued calls and letters, with a call among them (sinxcosx ->
    [sin(x), cos(x)], xsinxcosx -> [x, sin(x), cos(x)]), or None (also for a name longer than
    _MAX_SPLIT_LENGTH). Worked from the end, so a failed split takes linear time."""
    if len(name) > _MAX_SPLIT_LENGTH:
        return None
    # found[i]: (pieces of name[i:], whether a call is among them), or None
    found: Dict[int, Optional[Tuple[List[str], bool]]] = {len(name): ([], False)}
    for start in range(len(name) - 1, -1, -1):
        found[start] = None
        for function in _glued_names():
            after = start + len(function)
            if name.startswith(function, start) and name[after : after + 1].isalpha():
                rest = found.get(after + 1)
                if rest is not None:
                    found[start] = ([f"{function}({name[after]})"] + rest[0], True)
                    break
        if found[start] is None and name[start] in letters:
            rest = found.get(start + 1)
            if rest is not None and rest[1]:
                found[start] = ([name[start]] + rest[0], True)
    result = found[0]
    return result[0] if result is not None and result[1] else None


class PreparedStep:
    """One step as the engines read it.

    Attributes:
        number: its position, from 1.
        text: the step as given.
        prepared: the text both engines read.
        variables: the names math.js reads as variables, sorted ([] when unreadable).
        evaluator: math.js's value at a point, or None when math.js cannot read the step.
        scale: the sum of |term| over the step's terms, when it is a sum of several.
        read_error: why math.js cannot read it ("" when it can).
        bare: why a function in it has no parentheses (sin x), "" when none has.
        glued: the names in it that look like a glued call or constant (sinx, xe), as
            cas_text.Misspellings.
    """

    __slots__ = ("number", "text", "prepared", "variables", "evaluator", "scale", "read_error", "bare", "glued")

    def __init__(self, number: int, text: str) -> None:
        self.number = number
        self.text = text
        self.prepared = text
        self.variables: List[str] = []
        self.evaluator: Optional[Evaluator] = None
        self.scale: Optional[Evaluator] = None
        self.read_error = ""
        self.bare = ""
        self.glued: List[Any] = []


class _Budget:
    """SymPy time left for this derivation."""

    def __init__(self, seconds: float) -> None:
        self.left = seconds
        self.ran_out_at: Optional[int] = None
        self.unavailable_at: Optional[int] = None


class StepChecker:
    """Every method is static."""

    @staticmethod
    def check_steps(steps: Any) -> Any:
        """The verdict on each step of a derivation, as a dict, or "Error: ..." for bad input.

        Returns {"verdict", "first_wrong_step", "steps": [{"step", "verdict", "method",
        "reason"?, "counterexample"?, "note"?}, ...], "summary"} with a record for each step
        from the second on; "notes" lists what limited the check (SymPy's budget).
        """
        try:
            texts = StepChecker._validated(steps)
        except ValueError as error:
            return f"Error: {error}"
        prepared = [StepChecker._prepare(index + 1, text) for index, text in enumerate(texts)]
        budget = _Budget(SYMPY_BUDGET_S)
        records = [
            StepChecker._check_pair(prepared[index - 1], prepared[index], budget) for index in range(1, len(prepared))
        ]
        return StepChecker._result(records, budget)

    # ----- input --------------------------------------------------------------------------

    @staticmethod
    def _validated(steps: Any) -> List[str]:
        if isinstance(steps, str) or not isinstance(steps, (list, tuple)):
            raise ValueError("steps must be a list of expressions, one per step.")
        if len(steps) < MIN_STEPS:
            raise ValueError("a derivation needs at least 2 steps.")
        if len(steps) > MAX_STEPS:
            raise ValueError(f"check_steps takes at most {MAX_STEPS} steps; check a longer derivation in parts.")
        texts: List[str] = []
        for number, step in enumerate(steps, start=1):
            if not isinstance(step, str) or not step.strip():
                raise ValueError(f"step {number} must be a non-empty expression.")
            if len(step) > MAX_STEP_LENGTH:
                raise ValueError(f"step {number} is longer than {MAX_STEP_LENGTH} characters.")
            normalized = StepChecker._normalized(step)
            if "=" in normalized or has_inequality(normalized):
                raise ValueError(f"step {number} ({step.strip()}) is an equation or inequality. {_EQUATION_REFUSAL}")
            texts.append(step.strip())
        return texts

    @staticmethod
    def _normalized(text: str) -> str:
        """text with Unicode notation (≤, ≠, ...) rewritten, as the solve tool reads it."""
        from utils.math_utils import MathUtils

        return str(MathUtils._normalize_symbols(text))

    @staticmethod
    def _prepare(number: int, text: str) -> PreparedStep:
        step = PreparedStep(number, text)
        try:
            found = StepChecker._misspellings(text)
        except Exception:
            found = []
        step.bare = next((m.reason for m in found if m.bare), "")
        step.glued = [m for m in found if not m.bare]
        try:
            step.prepared = StepChecker._prepared_text(text)
            step.variables = StepChecker._variables(step.prepared)
        except Exception as error:
            step.read_error = _message(error)
            return step
        step.evaluator = StepChecker._evaluator(step.prepared)
        if step.evaluator is None:
            step.read_error = "math.js: it cannot be evaluated"
            return step
        terms = split_top_level_terms(step.prepared)
        if len(terms) > 1:
            step.scale = _sum_of_sizes([StepChecker._evaluator(term) for _, term in terms])
        return step

    @staticmethod
    def _misspellings(text: str) -> List[Any]:
        """The calculator spellings in text (InequalitySolver.misspellings)."""
        from utils.inequality_solver import InequalitySolver

        return list(InequalitySolver.misspellings(text))

    @staticmethod
    def _prepared_text(text: str) -> str:
        """The text both engines read (InequalitySolver.prepared)."""
        from utils.inequality_solver import InequalitySolver

        return str(InequalitySolver.prepared(text))

    @staticmethod
    def _variables(text: str) -> List[str]:
        from utils.symbolic_checks import SymbolicChecks

        return list(SymbolicChecks.expression_variables(text))

    @staticmethod
    def _evaluator(text: str) -> Optional[Evaluator]:
        from utils.symbolic_checks import SymbolicChecks

        return cast(Optional[Evaluator], SymbolicChecks.scope_function(text))

    @staticmethod
    def _ask_sympy(first: str, second: str) -> Optional[Dict[str, Any]]:
        """The server's answer for second - first ({"result"}, {"error"}, {"timeout"}), None when unavailable."""
        from utils.sympy_client import SympyClient

        return cast(Optional[Dict[str, Any]], SympyClient.compute("equivalent", first=first, second=second))

    # ----- one pair -----------------------------------------------------------------------

    @staticmethod
    def _check_pair(previous: PreparedStep, step: PreparedStep, budget: _Budget) -> Dict[str, Any]:
        record: Dict[str, Any] = {"step": step.number}
        misread = (
            previous.bare
            or step.bare
            or StepChecker._misread_name(previous, step)
            or StepChecker._misread_name(step, previous)
        )
        if misread:
            return dict(record, verdict=UNDECIDED, method=NEITHER, reason=misread)
        # the glued calls first: their hint names the call (cost: write cos(t))
        glued = sorted(previous.glued + step.glued, key=lambda misspelling: not misspelling.call)
        result = StepChecker._verdict(record, previous, step, budget, numeric_only=bool(glued))
        if result.get("verdict") == EQUIVALENT:
            return StepChecker._noted(result, [misspelling for misspelling in glued if misspelling.call])
        if glued:
            # sinx read as a variable: a spelling never makes a step wrong
            return dict(record, verdict=UNDECIDED, method=NEITHER, reason=glued[0].reason)
        lone = StepChecker._lone_name(previous, step)
        if lone is not None and result.get("verdict") == NOT_EQUIVALENT:
            # a name only one step has may be a spelling the checks do not know (exe, mx)
            return dict(
                record, verdict=UNDECIDED, method=NEITHER, reason=f"{lone} is read as one variable; {NAME_HINT}"
            )
        return result

    @staticmethod
    def _noted(result: Dict[str, Any], called: List[Any]) -> Dict[str, Any]:
        """An equivalent result, with a note on the first glued call read as a variable (sinx)."""
        if not called:
            return result
        name, call = called[0].name, called[0].call
        note = f"{name} is read as one variable (for the function, write {call}); the steps agree either way"
        return dict(result, note=f"{result['note']}; {note}" if result.get("note") else note)

    @staticmethod
    def _lone_name(previous: PreparedStep, step: PreparedStep) -> Optional[str]:
        """A name of two letters or more, without "_", that one step has and the other has not
        and that is no function or Greek letter (exe, mx, sqrt2), or None."""
        for first, second in ((previous, step), (step, previous)):
            for name in first.variables:
                if name in second.variables or "_" in name or name in _GREEK_NAMES:
                    continue
                if sum(character.isalpha() for character in name) >= 2 and name not in _function_spellings():
                    return str(name)
        return None

    @staticmethod
    def _verdict(
        record: Dict[str, Any], previous: PreparedStep, step: PreparedStep, budget: _Budget, numeric_only: bool
    ) -> Dict[str, Any]:
        """The record for a pair the engines read as written; numeric_only skips SymPy unless
        the numeric check cannot run."""
        numeric: Optional[EquivalenceReport] = None
        if previous.evaluator is not None and step.evaluator is not None:
            # a function name is never bound as a variable (it would hide the function)
            variables = sorted((set(previous.variables) | set(step.variables)) - _calculator_spellings())
            numeric = compare(previous.evaluator, step.evaluator, variables, previous.scale, step.scale)
            if numeric.verdict == EQUIVALENT:
                return StepChecker._equivalent(record, previous, step, numeric, NUMERIC)
            if numeric_only:
                return dict(record, verdict=numeric.verdict)
        sympy = StepChecker._sympy_verdict(previous, step, budget)
        if numeric is None:
            return StepChecker._without_numeric(record, previous, step, sympy)
        return StepChecker._combined(record, previous, step, numeric, sympy)

    @staticmethod
    def _misread_name(step: PreparedStep, other: PreparedStep) -> Optional[str]:
        """Why a name in step is not what the writer meant, or None.

        Misread, as math.js and SymPy read each of them as one variable: a calculator function
        read as a variable, another function name used as a variable (gamma, which neither
        engine reads as one), a function glued to more than one letter (sinxy, sinxcosx), and
        a name made of the other step's one-letter variables, e and pi (xy against x and y), x2
        beside x, and xy against yx. Names with "_" and longer names (costs, cos_theta,
        seconds) are variables.
        """
        others = set(other.variables)
        letters = {name for name in others | set(step.variables) if _SINGLE_LETTER.match(name)}
        glued = {misspelling.name for misspelling in step.glued if misspelling.call}
        for name in step.variables:
            if name in glued:
                continue  # the caller decides (sinx, xe, sqrt2)
            if name in _calculator_spellings():
                return f"{name} is a function, but it is read as a variable here; write {name}(x) with parentheses"
            if name in _function_spellings():
                return f"{name} is the name of a function, which the engines cannot read as a variable; rename it"
            if len(name) < 2 or name in others or "_" in name:
                continue
            indexed = _INDEXED.match(name)
            if indexed is not None and indexed.group(1) in letters:
                letter, digits = indexed.groups()
                return f"{name} is read as one variable; write {letter}^{digits} or {digits}*{letter}"
            reordered = next((o for o in others if o != name and sorted(o) == sorted(name)), None)
            if reordered is not None and name.isalpha() and len(name) <= 4:
                return f"{name} and {reordered} are each read as one variable; write {'*'.join(name)}"
            hint = StepChecker._glued_hint(name, letters)
            if hint is not None:
                return f"{name} is read as one variable; write {hint}"
            pieces = _pieces(name, {letter for letter in others if _SINGLE_LETTER.match(letter)})
            if pieces is None:
                continue
            factors = [piece for piece in pieces if piece not in _CONSTANTS]
            if len(factors) == len(pieces):
                return (
                    f"{name} is read as one variable, but the other step has {' and '.join(factors)}; "
                    f"write {'*'.join(pieces)}"
                )
            return f"{name} is read as one variable; write {'*'.join(pieces)}"
        return None

    @staticmethod
    def _glued_hint(name: str, letters: Set[str]) -> Optional[str]:
        """sin(x*y) for sinxy, sin(x)*cos(x) for sinxcosx: a calculator function glued to more
        letters than the input pipeline splits; None for other names."""
        calls = _call_pieces(name, letters | {name[-1]})
        if calls is not None and len(calls) > 1:
            return "*".join(calls)
        for function in _glued_names():
            if name.startswith(function):
                pieces = _pieces(name[len(function) :], letters)
                if pieces and len(pieces) > 1 and not set(pieces) & set(_CONSTANTS):
                    return f"{function}({'*'.join(pieces)})"
        return None

    @staticmethod
    def _sympy_verdict(previous: PreparedStep, step: PreparedStep, budget: _Budget) -> Dict[str, Any]:
        """{"verdict": equivalent | not_equivalent | unknown | none, "difference", "error"?, "note"?}."""
        if budget.unavailable_at is not None:
            return {"verdict": "none"}
        if budget.left <= 0:
            if budget.ran_out_at is None:
                budget.ran_out_at = step.number
            return {"verdict": "none", "note": "budget"}
        started = time.time()
        response = StepChecker._ask_sympy(previous.prepared, step.prepared)
        budget.left -= time.time() - started
        if response is None:
            budget.unavailable_at = step.number
            return {"verdict": "none"}
        if response.get("error"):
            return {"verdict": "none", "error": str(response["error"])}
        if response.get("timeout"):
            return {"verdict": "unknown", "note": "timeout"}
        result = response.get("result")
        if not isinstance(result, dict) or result.get("verdict") not in (EQUIVALENT, NOT_EQUIVALENT, "unknown"):
            return {"verdict": "unknown"}
        return {"verdict": result["verdict"], "difference": result.get("difference")}

    @staticmethod
    def _without_numeric(
        record: Dict[str, Any], previous: PreparedStep, step: PreparedStep, sympy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """A pair math.js cannot evaluate: SymPy's verdict, else undecided with the read error."""
        unreadable = previous if previous.evaluator is None else step
        if sympy["verdict"] == EQUIVALENT:
            return dict(record, verdict=EQUIVALENT, method=SYMBOLIC)
        if sympy["verdict"] == NOT_EQUIVALENT:
            return dict(
                record, verdict=NOT_EQUIVALENT, method=SYMBOLIC, reason=_difference_reason(previous, step, sympy)
            )
        reason = f"step {unreadable.number} cannot be read ({unreadable.read_error}"
        if sympy.get("error"):
            reason += f"; SymPy: {_trimmed(sympy['error'])}"
        return dict(record, verdict=UNDECIDED, method=NEITHER, reason=reason + ")")

    @staticmethod
    def _combined(
        record: Dict[str, Any],
        previous: PreparedStep,
        step: PreparedStep,
        numeric: EquivalenceReport,
        sympy: Dict[str, Any],
    ) -> Dict[str, Any]:
        """The verdict of a pair the numeric check did not call equivalent, with SymPy's."""
        symbolic = sympy["verdict"]
        if symbolic == EQUIVALENT:
            if numeric.verdict == NOT_EQUIVALENT:
                reason = "the checks disagree: SymPy finds the steps equivalent, but "
                return dict(
                    record, verdict=UNDECIDED, method=BOTH, reason=reason + _values_text(previous, step, numeric)
                )
            return StepChecker._equivalent(record, previous, step, numeric, SYMBOLIC)
        if symbolic == NOT_EQUIVALENT or numeric.verdict == NOT_EQUIVALENT:
            return StepChecker._not_equivalent(record, previous, step, numeric, sympy)
        return dict(
            record,
            verdict=UNDECIDED,
            method=NUMERIC if symbolic == "none" else BOTH,
            reason=_undecided_reason(previous, step, numeric, sympy),
        )

    @staticmethod
    def _not_equivalent(
        record: Dict[str, Any],
        previous: PreparedStep,
        step: PreparedStep,
        numeric: EquivalenceReport,
        sympy: Dict[str, Any],
    ) -> Dict[str, Any]:
        """A step one check proved wrong (and the other did not prove right): SymPy's difference,
        when it has one, and the numeric counterexample."""
        numeric_wrong = numeric.verdict == NOT_EQUIVALENT
        symbolic_wrong = sympy["verdict"] == NOT_EQUIVALENT
        parts = []
        if symbolic_wrong or sympy.get("difference"):
            parts.append(_difference_reason(previous, step, sympy))
        if numeric_wrong:
            parts.append(_values_text(previous, step, numeric))
        if numeric_wrong and (symbolic_wrong or sympy.get("difference")):
            method = BOTH
        else:
            method = NUMERIC if numeric_wrong else SYMBOLIC
        result = dict(record, verdict=NOT_EQUIVALENT, method=method, reason="; ".join(parts))
        if numeric_wrong and numeric.counterexample is not None:
            result["counterexample"] = _counterexample(numeric)
        return result

    @staticmethod
    def _equivalent(
        record: Dict[str, Any], previous: PreparedStep, step: PreparedStep, numeric: EquivalenceReport, method: str
    ) -> Dict[str, Any]:
        result = dict(record, verdict=EQUIVALENT, method=method)
        if numeric.undefined_point is not None:
            undefined = previous if numeric.undefined_side == 1 else step
            where = f" at {point_text(numeric.undefined_point)}" if numeric.undefined_point else ""
            result["note"] = f"equivalent where both are defined; step {undefined.number} is undefined{where}"
        return result

    # ----- the result ---------------------------------------------------------------------

    @staticmethod
    def _result(records: List[Dict[str, Any]], budget: _Budget) -> Dict[str, Any]:
        verdicts = [record["verdict"] for record in records]
        wrong = [record["step"] for record in records if record["verdict"] == NOT_EQUIVALENT]
        if wrong:
            verdict = NOT_EQUIVALENT
        elif UNDECIDED in verdicts:
            verdict = UNDECIDED
        else:
            verdict = EQUIVALENT
        result: Dict[str, Any] = {
            "verdict": verdict,
            "first_wrong_step": wrong[0] if wrong else None,
            "steps": records,
            "summary": " ".join(_sentence(record) for record in records),
        }
        notes = []
        if budget.ran_out_at is not None:
            notes.append(
                f"SymPy's time budget ({SYMPY_BUDGET_S:g} s) ran out: from step {budget.ran_out_at} on, "
                "the steps were checked numerically only."
            )
        if budget.unavailable_at is not None:
            notes.append(
                f"SymPy was not available: from step {budget.unavailable_at} on, the steps were checked numerically only."
            )
        if notes:
            result["notes"] = notes
        return result


def _sentence(record: Dict[str, Any]) -> str:
    number = record["step"]
    if record["verdict"] == EQUIVALENT:
        note = record.get("note")
        if note:
            return f"Step {number} follows from step {number - 1} ({note})."
        return f"Step {number} follows from step {number - 1}."
    if record["verdict"] == NOT_EQUIVALENT:
        return f"Step {number} is not equivalent to step {number - 1}: {record['reason']}."
    return f"Step {number} could not be checked against step {number - 1}: {record['reason']}."


def _undecided_reason(
    previous: PreparedStep, step: PreparedStep, numeric: EquivalenceReport, sympy: Dict[str, Any]
) -> str:
    reason = f"the numeric check could not decide ({_step_words(numeric.reason, previous, step)})"
    if sympy.get("note") == "timeout":
        return reason + " and SymPy ran out of time"
    if sympy.get("error"):
        return reason + f" and SymPy could not read a step ({_trimmed(sympy['error'])})"
    if sympy["verdict"] == "unknown":
        reason += " and SymPy could not show that the difference is 0"
        if sympy.get("difference"):
            reason += f" (step {step.number} - step {previous.number} = {sympy['difference']})"
    return reason


def _step_words(reason: str, previous: PreparedStep, step: PreparedStep) -> str:
    """The numeric check's reason with the steps named: "step 1 has no real value ..."."""
    for words, replacement in (
        ("the first expression", f"step {previous.number}"),
        ("the second expression", f"step {step.number}"),
        ("neither expression", "neither step"),
        ("the expressions", "the steps"),
    ):
        reason = reason.replace(words, replacement)
    return reason


def _difference_reason(previous: PreparedStep, step: PreparedStep, sympy: Dict[str, Any]) -> str:
    difference = sympy.get("difference")
    if not difference:
        return f"SymPy finds step {step.number} - step {previous.number} is not 0"
    return f"step {step.number} - step {previous.number} = {difference} (not 0)"


def _values_text(previous: PreparedStep, step: PreparedStep, numeric: EquivalenceReport) -> str:
    """ "at x = 1, step 1 = 3 and step 2 = 1" (without "at" when there are no variables)."""
    first, second = numeric.first_value, numeric.second_value
    if first is None or second is None:
        return "their values differ"
    texts = distinct_value_texts(first, second)
    values = f"step {previous.number} = {texts[0]} and step {step.number} = {texts[1]}"
    return f"at {point_text(numeric.counterexample)}, {values}" if numeric.counterexample else values


def _counterexample(numeric: EquivalenceReport) -> Dict[str, Any]:
    point: Point = numeric.counterexample or {}
    return {
        "at": {name: json_number(value) for name, value in point.items()},
        "previous": json_number(numeric.first_value) if numeric.first_value is not None else None,
        "this": json_number(numeric.second_value) if numeric.second_value is not None else None,
    }


def _sum_of_sizes(terms: Sequence[Optional[Evaluator]]) -> Optional[Evaluator]:
    """sum |term(point)| as an evaluator, or None when a term cannot be evaluated."""
    if any(term is None for term in terms):
        return None
    functions = [term for term in terms if term is not None]

    def scale(point: Point) -> Optional[float]:
        total = 0.0
        for function in functions:
            value = function(point)
            if value is None:
                return None
            total += abs(value)
        return total

    return scale


def _message(error: Exception) -> str:
    return _trimmed(str(getattr(error, "message", "") or error)) or type(error).__name__


def _trimmed(text: str) -> str:
    return str(text).strip().rstrip(".")
