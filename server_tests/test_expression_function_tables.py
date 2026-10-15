"""Goldens for the expression function tables of ExpressionValidator and MathUtils.

They record what the validator and MathUtils derive from their function lists: the
whitelists, both evaluation namespaces, the alias rewrites and the text MathUtils sends to
math.js. Refactors of those lists must keep every section equal; a deliberate behaviour
change regenerates the golden and says which entries changed.

Regenerate with ``python -m server_tests.test_expression_function_tables``.
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest
from types import SimpleNamespace
from typing import Any, Callable, Dict, List
from unittest import mock

from server_tests import client_renderer  # noqa: F401  (installs the browser stub)
from expression_validator import ExpressionValidator
from utils.math_utils import MathUtils

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "data", "expression_function_tables_golden.json")

# Modules whose callables are labelled by their namespace key, so moving a helper between
# them does not change the label
_LOCAL_MODULES = ("expression_validator", "expression_functions")
_CURVE_SAMPLE_POINTS = (0.25, 0.5, 2)

# Contexts every function name and alias is written in; N stands for the name
_ALIAS_CONTEXTS = ("N(x)", "2N(x)", "kN(x)", "_N(x)", "N2(x)", "N (x)", "N", "Nx", "x*N", "N(x)+N(y)")
_ALIAS_EXTRAS = (
    "kiln*2",
    "ln2",
    "lnx",
    "c2ln",
    "artifact(3)",
    "myvar(2)",
    "brand(1)",
    "xstd(1)",
    "nderivative(x)",
    "2sine(x)",
    "sine (x)",
    "ln (x)",
    "2ln(x)",
    "x_ln(x)",
    "x2ln(x)",
    "a1sine(x)",
    "1e3ln(x)",
    "θ2ln(x)",
    "2.5ln(x)",
    "2square root(x)",
    "xhyperbolic sine(1)",
    "hyperbolic sine(x)",
    "log(ln(x))",
    "ln(ln(x))",
    "2ln(x) + sine(x)",
)

# MathUtils.evaluate inputs (with their variables) whose routing and math.js text are pinned
_EVALUATE_CASES = (
    ("arrangements(5,2)", None),
    ("stdev([1,2])", None),
    ("trunc(2.5)", None),
    ("xtrunc(1)", None),
    ("is_prime(7)", None),
    ("randint(1,2)", None),
    ("summation('n','n',0,3)", None),
    ("2*product", {"product": 3}),
    ("sin(x)", {"x": 1}),
    ("std([1,2]) + var([1,2])", None),
)
# Inputs for the other places that hand expression text to math.js
_SITE_INPUTS = ("trunc(x)", "xtrunc(x)", "arrangements(x, 2)", "stdev([x, 1])", "sin(x)")


def _label(key: str, value: Any) -> str:
    """A stable name for a namespace value: math.sin, builtins.abs, MathUtils.derivative, local:sec."""
    if not callable(value):
        return f"const:{value!r}"
    module = getattr(value, "__module__", None)
    if module in _LOCAL_MODULES:
        return f"local:{key}"
    qualname = getattr(value, "__qualname__", repr(value))
    if module == "utils.math_utils":
        return str(qualname)
    return f"{module}.{qualname}"


def _outcome(function: Callable[..., Any], *args: Any) -> str:
    try:
        return repr(function(*args))
    except Exception as error:
        return "EXCEPTION " + type(error).__name__


def _fix_both(expression: str) -> List[str]:
    return [
        _outcome(ExpressionValidator.fix_math_expression, expression, True),
        _outcome(ExpressionValidator.fix_math_expression, expression, False),
    ]


def _tables() -> Dict[str, Any]:
    return {
        "allowed_functions": sorted(ExpressionValidator.ALLOWED_FUNCTIONS),
        "allowed_nodes": sorted(node.__name__ for node in ExpressionValidator.ALLOWED_NODES),
        "superscript_functions": sorted(ExpressionValidator._SUPERSCRIPT_FUNCTIONS),
        "inverse_functions": dict(sorted(ExpressionValidator._INVERSE_FUNCTIONS.items())),
        "function_replacements": dict(sorted(ExpressionValidator._get_function_replacements().items())),
        "python_only_functions": sorted(MathUtils._PYTHON_ONLY_FUNCTIONS),
    }


def _namespaces() -> Dict[str, Any]:
    x_namespace = ExpressionValidator._build_functions()
    t_namespace = ExpressionValidator._get_variables_and_functions_parametric(0)
    return {
        "x": {key: _label(key, value) for key, value in sorted(x_namespace.items())},
        "t": {key: _label(key, value) for key, value in sorted(t_namespace.items())},
    }


def _curve_values() -> Dict[str, List[str]]:
    """Values of every callable in the parametric namespace at a few points."""
    t_namespace = ExpressionValidator._get_variables_and_functions_parametric(0)
    values: Dict[str, List[str]] = {}
    for key, value in sorted(t_namespace.items()):
        if not callable(value):
            continue
        values[key] = [_outcome(value, point) for point in _CURVE_SAMPLE_POINTS]
        values[key + "/2"] = [_outcome(value, point, 2) for point in _CURVE_SAMPLE_POINTS]
    return values


def _alias_corpus() -> Dict[str, List[str]]:
    words = [old[:-1] for old in ExpressionValidator._get_function_replacements() if old.endswith("(")]
    names = sorted(ExpressionValidator.ALLOWED_FUNCTIONS) + sorted(words) + ["ln"]
    expressions = [context.replace("N", name) for name in names for context in _ALIAS_CONTEXTS]
    expressions.extend(_ALIAS_EXTRAS)
    return {expression: _fix_both(expression) for expression in expressions}


class _MathJsRecorder:
    """Stands in for window.math: records the text it is asked to evaluate or compile."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def evaluate(self, expression: Any, _scope: Any = None) -> float:
        self.calls.append("evaluate " + str(expression))
        return 1.0

    def compile(self, expression: Any) -> Any:
        self.calls.append("compile " + str(expression))
        return SimpleNamespace(evaluate=lambda _scope=None: 1.0)

    def format(self, value: Any) -> str:
        return str(value)

    def subtract(self, a: Any, b: Any) -> float:
        return float(a) - float(b)

    def abs(self, value: Any) -> float:
        return abs(float(value))

    def complex(self, real: float, imag: float) -> complex:
        return complex(real, imag)


def _recorded(call: Callable[[], Any]) -> Dict[str, Any]:
    """Run call with window.math replaced by a recorder; return what math.js was given."""
    import numeric_solver.expression_utils as expression_utils
    import utils.math_utils as math_utils

    recorder = _MathJsRecorder()
    fake_window = SimpleNamespace(math=recorder)
    with (
        mock.patch.object(math_utils, "window", fake_window),
        mock.patch.object(expression_utils, "window", fake_window),
    ):
        try:
            result = call()
            raised = None
        except Exception as error:
            result, raised = None, type(error).__name__
    returned_error = isinstance(result, str) and result.startswith("Error")
    return {"mathjs": recorder.calls, "raised": raised, "returned_error": returned_error}


def _mathjs_bridge() -> Dict[str, Any]:
    from numeric_solver.expression_utils import evaluate_residuals
    from utils.symbolic_checks import SymbolicChecks

    evaluate: Dict[str, Any] = {}
    for expression, variables in _EVALUATE_CASES:
        record = _recorded(lambda: MathUtils.evaluate(expression, variables))
        record["path"] = "mathjs" if record["mathjs"] else ("none" if record["returned_error"] else "python")
        evaluate[expression] = record

    def probe(expression: str) -> None:
        evaluator = MathUtils._mathjs_probe_evaluator(expression)
        if evaluator is not None:
            evaluator(1.0)

    def real_function(expression: str) -> None:
        function = SymbolicChecks.real_function(expression, "x")
        if function is not None:
            function(1.0)

    sites: Dict[str, Callable[[str], Any]] = {
        "probe": probe,
        "interior_singularity": lambda text: MathUtils._find_interior_singularity(text, "x", 0, 1),
        "tangent_asymptotes": lambda text: MathUtils._tangent_asymptotes(text, -1, 1),
        "root_status": lambda text: MathUtils._root_status([text, "1"], "x", complex(1, 0)),
        "equation_holds": lambda text: MathUtils._equation_holds([text, "1"], {"x": 1.0}),
        "degenerate_linear_system": lambda text: MathUtils._degenerate_linear_system([text + " = 1"]),
        "real_function": real_function,
        "numeric_value": SymbolicChecks.numeric_value,
        "evaluate_residuals": lambda text: evaluate_residuals([text], ["x"], [1.0]),
    }
    recorded_sites = {
        site: {text: _recorded(lambda: call(text)) for text in _SITE_INPUTS} for site, call in sites.items()
    }
    return {"evaluate": evaluate, "sites": recorded_sites}


def collect() -> Dict[str, Any]:
    """Every golden section, computed from the current code."""
    return {
        "tables": _tables(),
        "namespaces": _namespaces(),
        "curve_values": _curve_values(),
        "aliases": _alias_corpus(),
        "mathjs_bridge": _mathjs_bridge(),
    }


class TestExpressionFunctionTables(unittest.TestCase):
    """The derived tables, namespaces, aliases and math.js text match the recorded golden."""

    golden: Dict[str, Any] = {}

    @classmethod
    def setUpClass(cls) -> None:
        with open(GOLDEN_PATH, encoding="utf-8") as golden_file:
            cls.golden = json.load(golden_file)

    def _assert_section(self, name: str, actual: Any) -> None:
        expected = self.golden[name]
        if isinstance(expected, dict):
            self.assertEqual(sorted(actual), sorted(expected), f"{name}: keys differ")
            mismatches = {key: (actual[key], expected[key]) for key in expected if actual[key] != expected[key]}
            self.assertEqual(mismatches, {}, f"{name}: (actual, expected) per key")
        else:
            self.assertEqual(actual, expected)

    def test_tables(self) -> None:
        self._assert_section("tables", _tables())

    def test_namespaces(self) -> None:
        self._assert_section("namespaces", _namespaces())

    def test_curve_values(self) -> None:
        self._assert_section("curve_values", _curve_values())

    def test_alias_corpus(self) -> None:
        self._assert_section("aliases", _alias_corpus())

    def test_mathjs_bridge(self) -> None:
        self._assert_section("mathjs_bridge", _mathjs_bridge())

    def test_known_counts(self) -> None:
        # Sanity checks on the golden itself, so an empty or truncated file cannot pass
        tables = self.golden["tables"]
        self.assertEqual(len(tables["allowed_functions"]), 72)
        self.assertEqual(len(self.golden["namespaces"]["x"]), 74)
        self.assertEqual(len(self.golden["namespaces"]["t"]), 37)
        self.assertGreater(len(self.golden["aliases"]), 1000)
        self.assertTrue(math.isclose(float(self.golden["namespaces"]["x"]["pi"][len("const:") :]), math.pi))


if __name__ == "__main__":
    with open(GOLDEN_PATH, "w", encoding="utf-8") as output:
        json.dump(collect(), output, indent=1, ensure_ascii=False, sort_keys=True)
        output.write("\n")
    print(f"Wrote {GOLDEN_PATH}", file=sys.stderr)
