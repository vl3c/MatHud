"""Tests for the expression function registry (static/client/expression_functions.py).

The derived tables must equal the goldens recorded from the hand-written tables they
replace; the rows must be consistent with each other, with the callables' signatures, with
the evaluate_expression tool description and with the SymPy parser on the server.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import subprocess
import sys
import unittest
from importlib.util import find_spec
from typing import Any, Dict, Optional, Set, Tuple

from server_tests import python_path_setup
from server_tests import client_renderer  # noqa: F401  (browser stub, so MathUtils resolves)
import expression_functions as ef
from server_tests.test_expression_function_tables import GOLDEN_PATH, _label

# Names the server's SymPy parser accepts that are neither rows nor word aliases (sgn is a
# call alias of sign on the client)
SYMPY_ONLY_NAMES = {"coth", "sech", "csch", "sgn"}


# Aliases are matched as whole tokens; the client suite runs the same cases through Brython's re
ALIAS_CASES = {
    "2ln(x) + sine(x)": "2log(x) + sin(x)",
    "1e3ln(x)": "1e3log(x)",
    "2.5e-3ln(x)": "2.5e-3log(x)",
    "θ2ln(x)": "θ2log(x)",
    "ln (x)": "log (x)",
    "hyperbolic sine(x) + 2square root(x)": "sinh(x) + 2sqrt(x)",
    "arcsine(x)*cosine(x)": "asin(x)*cos(x)",
    "std(x)+var(y)": "stdev(x)+variance(y)",
    "kiln": "kiln",
    "ln2": "ln2",
    "x2ln(x)": "x2ln(x)",
    "a1sine(x)": "a1sine(x)",
    "artifact(3)": "artifact(3)",
    "myvar(2)": "myvar(2)",
    "sine (x)": "sine (x)",
    "var": "var",
    "sgn(x)": "sign(x)",
    "nthRoot(8, 3)": "nthroot(8, 3)",
    "xsgn(x)": "xsgn(x)",
}


def _golden() -> Dict[str, Any]:
    with open(GOLDEN_PATH, encoding="utf-8") as golden_file:
        return dict(json.load(golden_file))


def _signature_arity(function: Any) -> Optional[Tuple[int, Optional[int]]]:
    """(required, maximum) positional arguments of function, or None without a signature."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return None
    positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    required = sum(1 for p in positional if p.default is p.empty)
    variadic = any(p.kind == p.VAR_POSITIONAL for p in parameters)
    return required, None if variadic else len(positional)


def _evaluate_expression_description() -> str:
    from static.functions_definitions import FUNCTIONS

    for definition in FUNCTIONS:
        if definition["function"]["name"] == "evaluate_expression":
            return str(definition["function"]["description"])
    raise AssertionError("evaluate_expression is not defined")


class TestDerivedTables(unittest.TestCase):
    """The registry reproduces the tables recorded from the code before it existed."""

    golden: Dict[str, Any] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.golden = _golden()

    def test_validator_tables(self) -> None:
        tables = self.golden["tables"]
        self.assertEqual(sorted(ef.ALLOWED_FUNCTIONS), tables["allowed_functions"])
        self.assertEqual(sorted(node.__name__ for node in ef.ALLOWED_NODES), tables["allowed_nodes"])
        self.assertEqual(sorted(ef.SUPERSCRIPT_NAMES), tables["superscript_functions"])
        self.assertEqual(ef.INVERSE_NAMES, tables["inverse_functions"])
        self.assertEqual(ef.CALL_ALIASES, tables["call_aliases"])
        self.assertEqual(ef.WORD_ALIASES, tables["word_aliases"])

    def test_python_only_names(self) -> None:
        self.assertEqual(sorted(ef.PYTHON_ONLY_NAMES), self.golden["tables"]["python_only_functions"])

    def test_mathjs_renames(self) -> None:
        self.assertEqual(
            ef.MATHJS_RENAMES,
            {"arrangements": "permutations", "stdev": "std", "trunc": "fix", "nthroot": "nthRoot"},
        )

    def test_namespaces(self) -> None:
        namespaces = self.golden["namespaces"]
        python_namespace = ef.build_python_namespace()
        self.assertEqual({key: _label(key, value) for key, value in python_namespace.items()}, namespaces["x"])
        curve_namespace = {**ef.build_curve_namespace(), "t": 0}
        self.assertEqual({key: _label(key, value) for key, value in curve_namespace.items()}, namespaces["t"])

    def test_namespaces_are_new_dicts(self) -> None:
        self.assertIsNot(ef.build_python_namespace(), ef.build_python_namespace())
        self.assertIsNot(ef.build_curve_namespace(), ef.build_curve_namespace())


class TestRowIntegrity(unittest.TestCase):
    def test_names_are_unique_identifiers(self) -> None:
        names = [row.name for row in ef.FUNCTIONS]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assertTrue(name.isidentifier(), name)

    def test_aliases_are_not_names_and_are_unique(self) -> None:
        names = {row.name for row in ef.FUNCTIONS}
        aliases = [alias for row in ef.FUNCTIONS for alias in row.aliases + row.word_aliases]
        self.assertEqual(len(aliases), len(set(aliases)))
        self.assertEqual(names & set(aliases), set())
        self.assertNotIn("(", "".join(aliases))
        for row in ef.FUNCTIONS:
            for alias in row.word_aliases:
                self.assertTrue(alias.isidentifier(), alias)

    def test_inverse_targets_are_rows(self) -> None:
        for row in ef.FUNCTIONS:
            if row.inverse is not None:
                self.assertIn(row.inverse, ef.BY_NAME, row.name)
                self.assertTrue(row.superscript, row.name)

    def test_python_only_rows_have_no_mathjs_name(self) -> None:
        for row in ef.FUNCTIONS:
            if row.python_only:
                self.assertIsNone(row.mathjs, row.name)

    def test_curve_rows_are_in_the_python_namespace(self) -> None:
        self.assertLessEqual(set(ef.build_curve_namespace()), set(ef.build_python_namespace()))
        self.assertLessEqual(ef.CURVE_NAMES, set(ef.BY_NAME))

    def test_math_utils_names_resolve_to_callables(self) -> None:
        from utils.math_utils import MathUtils

        for row in ef.FUNCTIONS:
            if isinstance(row.python, str):
                with self.subTest(name=row.name):
                    self.assertTrue(callable(getattr(MathUtils, row.python)))

    def test_arity_fits_the_signature(self) -> None:
        namespace = ef.build_python_namespace()
        checked = 0
        for row in ef.FUNCTIONS:
            minimum, maximum = row.arity
            self.assertTrue(maximum is None or minimum <= maximum, row.name)
            signature = _signature_arity(namespace[row.name])
            if signature is None:
                continue  # builtins such as max and math.log have no signature
            checked += 1
            with self.subTest(name=row.name):
                required, accepted = signature
                self.assertGreaterEqual(minimum, required)
                if accepted is not None:
                    self.assertIsNotNone(maximum)
                    self.assertLessEqual(maximum, accepted)
        self.assertGreater(checked, 60)

    def test_rename_targets_are_not_sources(self) -> None:
        # Otherwise renames would chain (a -> b -> c) depending on their order
        self.assertEqual(set(ef.MATHJS_RENAMES) & set(ef.MATHJS_RENAMES.values()), set())


class TestAliases(unittest.TestCase):
    def test_aliases_are_matched_as_whole_tokens(self) -> None:
        for text, expected in ALIAS_CASES.items():
            with self.subTest(text=text):
                self.assertEqual(ef.replace_aliases(text), expected)

    def test_hints_cover_every_alias(self) -> None:
        # The pre-check must not skip text an alias could change
        for alias in list(ef.CALL_ALIASES) + list(ef.WORD_ALIASES):
            text = alias + "(" if alias in ef.CALL_ALIASES else alias
            with self.subTest(alias=alias):
                self.assertTrue(any(hint in text for hint in ef._ALIAS_HINTS))
                self.assertNotEqual(ef.replace_aliases(text), text)


class TestAliasesEvaluate(unittest.TestCase):
    """Aliased input gives the same values as the canonical names (sampled at x = 0.7)."""

    CASES = {
        "2πln(x)": "2*pi*log(x)",
        "ln²(x)": "log(x)**2",
        "2ln(x) + sine(x)": "2*log(x) + sin(x)",
        "hyperbolic sine(x) + arccosine(x)": "sinh(x) + acos(x)",
        "1e3ln(x)": "1000*log(x)",
    }

    def test_aliased_functions_evaluate_like_their_names(self) -> None:
        from expression_validator import ExpressionValidator

        for text, canonical in self.CASES.items():
            with self.subTest(text=text):
                self.assertAlmostEqual(
                    ExpressionValidator.parse_function_string(text)(0.7),
                    ExpressionValidator.parse_function_string(canonical)(0.7),
                )

    def test_a_name_that_only_starts_with_an_alias_is_not_a_function(self) -> None:
        # ln2(x) and ln10(x) used to become log2(x) and log10(x) by substring replacement
        from expression_validator import ExpressionValidator

        for text in ("ln2(x)", "ln10(x)"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    ExpressionValidator.parse_function_string(text)


class TestMathJsRenames(unittest.TestCase):
    def test_renames_are_matched_as_whole_tokens(self) -> None:
        cases = {
            "trunc(x) + xtrunc(x)": "fix(x) + xtrunc(x)",
            "2trunc(x)": "2fix(x)",
            "arrangements(5, 2)/stdev([1, 2])": "permutations(5, 2)/std([1, 2])",
            "xstdev(1) + stdev_2(1)": "xstdev(1) + stdev_2(1)",
            "trunc": "trunc",
            "sin(x)": "sin(x)",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(ef.rename_for_mathjs(text), expected)

    def test_percent_becomes_the_mod_operator(self) -> None:
        # math.js reads 7 % -3 as (7%) - 3 = -2.93; mod is the remainder in every engine
        cases = {
            "7 % -3": "7 mod -3",
            "x%2": "x mod 2",
            "trunc(x) % 3": "fix(x) mod 3",
            "2^7 % 5": "2^7 mod 5",
            "7 mod -3": "7 mod -3",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(ef.rename_for_mathjs(text), expected)
                self.assertEqual(ef.rename_for_mathjs(expected), expected)  # idempotent


class TestRemainderOperator(unittest.TestCase):
    """% is Python's floored remainder, with the precedence of * and /; never a percentage."""

    def test_percent_validates_and_evaluates_as_the_remainder(self) -> None:
        from expression_validator import ExpressionValidator

        cases = {"-7 % 3": 2, "x % 3": 1.5, "7 % -3": -2, "2^7 % 5": 3, "2*7 % 3": 2, "-x % 2": 0.5}
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(ExpressionValidator.parse_function_string(text)(7.5 if "x" in text else 0.0), expected)
        self.assertEqual(ExpressionValidator.fix_math_expression("x % 2", True), "x % 2")
        self.assertEqual(ExpressionValidator.fix_math_expression("x % 2", False), "x % 2")

    def test_a_zero_divisor_raises_in_python_but_mod_keeps_x(self) -> None:
        # A plot drops the sample of x % 0; mod(x, 0) is x, as in math.js (one rule is item 11's)
        from expression_validator import ExpressionValidator

        with self.assertRaises(ZeroDivisionError):
            ExpressionValidator.parse_function_string("x % 0")(3.0)
        self.assertEqual(ExpressionValidator.parse_function_string("mod(x, 0)")(3.0), 3.0)

    def test_postfix_percent_is_not_an_expression(self) -> None:
        from expression_validator import ExpressionValidator

        for text in ("50%", "x*(1 + 5%)"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    ExpressionValidator.validate_expression_tree(ExpressionValidator.fix_math_expression(text, True))

    def test_raw_text_sites_lose_postfix_percent(self) -> None:
        # The numeric solver passes raw text to math.js; 5% is no longer 0.05 there
        from numeric_solver.expression_utils import equation_to_residual

        self.assertEqual(equation_to_residual("x*(1 + 5%) = 105"), "(x*(1 + 5 mod )) - (105)")
        self.assertEqual(equation_to_residual("x % 4 = 1"), "(x mod 4) - (1)")


class TestCalculatorFunctions(unittest.TestCase):
    """The Python callables of the calculator rows follow math.js 14.5.2 (plots use them)."""

    @staticmethod
    def _call(name: str, *args: Any) -> Any:
        return ef.build_python_namespace()[name](*args)

    def test_mod_is_floored_and_keeps_x_for_a_zero_divisor(self) -> None:
        cases = {(17, 5): 2, (-7, 3): 2, (7, -3): -2, (-7, -3): -1, (5.5, 2): 1.5, (7, 0): 7, (-2.5, 0): -2.5}
        for (x, y), expected in cases.items():
            with self.subTest(x=x, y=y):
                self.assertEqual(self._call("mod", x, y), expected)
        self.assertIsInstance(self._call("mod", 17, 5), int)  # exact for integers
        with self.assertRaises(TypeError):
            self._call("mod", 3j, 2)

    def test_nthroot_is_the_real_root(self) -> None:
        cases = {(-8, 3): -2.0, (32, 5): 2.0, (16, -2): 0.25, (16, 2): 4.0, (-32, 5): -2.0, (-8, -3): -0.5, (0, 3): 0.0}
        for (x, n), expected in cases.items():
            with self.subTest(x=x, n=n):
                self.assertAlmostEqual(self._call("nthroot", x, n), expected)
        for x, n in ((-8, 2), (-27, 1.5), (5, 0), (-8, 0)):
            with self.subTest(x=x, n=n):
                with self.assertRaises(ValueError):
                    self._call("nthroot", x, n)

    def test_cbrt_atan2_gamma_erf(self) -> None:
        self.assertAlmostEqual(self._call("cbrt", -27), -3.0)
        self.assertAlmostEqual(self._call("atan2", 1, -1), 3 * math.pi / 4)
        self.assertAlmostEqual(self._call("atan2", -1, -1), -3 * math.pi / 4)
        self.assertEqual(self._call("atan2", 0, 0), 0.0)
        self.assertAlmostEqual(self._call("gamma", 5), 24.0)
        self.assertAlmostEqual(self._call("gamma", 0.5), math.sqrt(math.pi))
        self.assertAlmostEqual(self._call("erf", 0.5), 0.5204998778130465)
        for pole in (0, -1, -2):
            with self.subTest(pole=pole):
                with self.assertRaises(ValueError):  # math.js gives Infinity; the plot drops the sample
                    self._call("gamma", pole)

    def test_sign(self) -> None:
        self.assertEqual([self._call("sign", value) for value in (2.5, -0.1, 0, -0.0)], [1.0, -1.0, 0.0, 0.0])
        self.assertTrue(math.isnan(self._call("sign", math.nan)))
        self.assertAlmostEqual(self._call("sign", 3 + 4j), 0.6 + 0.8j)
        self.assertEqual(self._call("sign", 0j), 0.0)

    def test_complex_parts_of_complex_and_real_values(self) -> None:
        z = 3 + 4j
        self.assertEqual((self._call("re", z), self._call("im", z)), (3.0, 4.0))
        self.assertEqual(self._call("conj", z), 3 - 4j)
        self.assertAlmostEqual(self._call("arg", 1 + 1j), math.pi / 4)
        self.assertEqual((self._call("re", -2.5), self._call("im", -2.5), self._call("conj", -2.5)), (-2.5, 0.0, -2.5))
        self.assertAlmostEqual(self._call("arg", -1), math.pi)
        self.assertEqual(self._call("arg", 2), 0.0)

    def test_aliases_evaluate_like_their_names(self) -> None:
        from expression_validator import ExpressionValidator

        cases = {
            "sgn(x - 1)": "sign(x - 1)",
            "nthRoot(x, 3)": "nthroot(x, 3)",
            "real(x) + imag(x) + conjugate(x)": "re(x) + im(x) + conj(x)",
        }
        for text, canonical in cases.items():
            with self.subTest(text=text):
                self.assertEqual(
                    ExpressionValidator.parse_function_string(text)(0.7),
                    ExpressionValidator.parse_function_string(canonical)(0.7),
                )


class TestPopulationSpread(unittest.TestCase):
    """pstdev and pvariance divide by n, with the statistics tool's own routine."""

    DATA = [2, 4, 4, 4, 5, 5, 7, 9]

    def test_values(self) -> None:
        from utils.math_utils import MathUtils

        self.assertEqual(MathUtils.pstdev(self.DATA), 2.0)
        self.assertEqual(MathUtils.pvariance(self.DATA), 4.0)
        self.assertEqual(MathUtils.pstdev([5]), 0.0)
        self.assertAlmostEqual(MathUtils.stdev(self.DATA) - MathUtils.pstdev(self.DATA), 0.13808993529939517)

    def test_bit_identical_to_the_tool(self) -> None:
        import random

        from utils.math_utils import MathUtils
        from utils.statistics.descriptive import compute_descriptive_statistics

        rng = random.Random(5)
        for trial in range(100):
            data = [rng.uniform(-1e6, 1e6) for _ in range(rng.randint(1, 30))]
            tool = compute_descriptive_statistics(data)
            rng.shuffle(data)  # the tool sorts its copy; the figures must not depend on order
            with self.subTest(trial=trial):
                self.assertEqual(MathUtils.pstdev(data), tool["population_standard_deviation"])
                self.assertEqual(MathUtils.pvariance(data), tool["population_variance"])

    def test_errors_name_the_function(self) -> None:
        from utils.math_utils import MathUtils

        with self.assertRaisesRegex(ValueError, "pstdev's list must not be empty"):
            MathUtils.pstdev([])
        with self.assertRaisesRegex(TypeError, "pvariance's list must be a list"):
            MathUtils.pvariance(5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            MathUtils.pstdev([1, 2], [3])  # type: ignore[call-arg]

    def test_python_path_evaluates_them(self) -> None:
        from expression_validator import ExpressionValidator

        evaluate = ExpressionValidator.evaluate_expression
        self.assertEqual(evaluate("pstdev([2, 4, 4, 4, 5, 5, 7, 9])"), 2.0)
        self.assertEqual(evaluate("2*pvariance([1, 3])"), 2.0)
        self.assertAlmostEqual(
            evaluate("stdev([2, 4, 4, 4, 5, 5, 7, 9]) - pstdev([2, 4, 4, 4, 5, 5, 7, 9])"), 0.13808993529939517
        )


class TestNormalizationArgument(unittest.TestCase):
    """A string second argument of stdev or variance is refused: math.js's "biased" divides by n + 1."""

    def test_refused_with_a_pointer_to_the_population_functions(self) -> None:
        from expression_validator import ExpressionValidator

        for text in ('stdev([1, 2, 3], "biased")', "variance([1, 2], 'uncorrected')", 'std([1, 2], "unbiased")'):
            with self.subTest(text=text):
                python_text = ExpressionValidator.fix_math_expression(text, python_compatible=True)
                with self.assertRaisesRegex(ValueError, "use pstdev or pvariance"):
                    ExpressionValidator.validate_expression_tree(python_text)

    def test_other_calls_are_untouched(self) -> None:
        from expression_validator import ExpressionValidator

        for text in ("stdev([1, 2, 3])", "variance([1, 2])", "summation('n', 'n', 0, 3)", "pstdev([1, 2])"):
            with self.subTest(text=text):
                ExpressionValidator.validate_expression_tree(text)


class TestPythonOnlyRouting(unittest.TestCase):
    def test_python_only_functions_are_found_by_call(self) -> None:
        cases = {
            "is_prime(7)": True,
            "2 + randint (1, 2)": True,
            "summation('n^2', 'n', 0, 3)": True,
            "sin(next_prime(4))": True,
            "pstdev([1, 2])": True,
            "2*pvariance([1, 2]) + stdev([1, 2])": True,
            "stdev([1, 2])": False,
            "2*product": False,
            "cross_product(a, b)": False,
            "summation + 1": False,
            "sin(x)": False,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(ef.calls_python_only(text), expected)


class TestReservedNames(unittest.TestCase):
    """Names a variable may not take: functions, word aliases, constants, i, j, keywords."""

    def test_functions_constants_and_imaginary_units_are_reserved(self) -> None:
        for name in ("ln", "i", "j", "e", "pi", "inf", "Infinity", "sin", "mean", "product", "factorial"):
            with self.subTest(name=name):
                self.assertIn(name, ef.RESERVED_NAMES)
        self.assertEqual(ef.IMAGINARY_UNITS, frozenset(("i", "j")))

    def test_python_keywords_are_reserved(self) -> None:
        # fix_math_expression("2in + x", True) gives "2*in + x", a syntax error
        for name in ("in", "is", "or", "if", "as", "and", "not", "lambda", "None", "True"):
            with self.subTest(name=name):
                self.assertIn(name, ef.RESERVED_NAMES)

    def test_call_only_aliases_are_not_reserved(self) -> None:
        # They name a function only when "(" follows, so "power" or "var" stays a usable name
        for name in ("power", "binary", "var", "fact", "rand", "lim", "std", "sine", "derivative"):
            with self.subTest(name=name):
                self.assertNotIn(name, ef.RESERVED_NAMES)
                self.assertIn(name, ef.CALL_ALIASES)

    def test_every_function_and_constant_is_reserved(self) -> None:
        self.assertLessEqual(set(ef.BY_NAME), ef.RESERVED_NAMES)
        self.assertLessEqual(set(ef.CONSTANTS) | set(ef.WORD_ALIASES), ef.RESERVED_NAMES)

    def test_ordinary_variable_names_are_free(self) -> None:
        for name in ("x", "y", "t", "k", "n", "a", "θ", "r", "xy", "type", "match"):
            with self.subTest(name=name):
                self.assertNotIn(name, ef.RESERVED_NAMES)


class TestBrowserFreeImport(unittest.TestCase):
    def test_module_imports_without_browser(self) -> None:
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); import expression_functions as ef; "
            "assert 'browser' not in sys.modules; assert 'utils.math_utils' not in sys.modules; "
            "print(len(ef.FUNCTIONS))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, python_path_setup.SITE_PACKAGES_PATH],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(len(ef.FUNCTIONS)))


class TestToolDescription(unittest.TestCase):
    """The hand-written evaluate_expression description names exactly the described rows."""

    def test_described_rows_are_in_the_description(self) -> None:
        identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", _evaluate_expression_description()))
        missing = sorted(row.name for row in ef.FUNCTIONS if row.described and row.name not in identifiers)
        self.assertEqual(missing, [], "add these to the evaluate_expression description or set described=False")

    def test_description_names_only_registry_functions(self) -> None:
        description = _evaluate_expression_description()
        function_list = description.split("functions: ", 1)[1].split(". ", 1)[0]
        # Drop the notes in parentheses: "next_prime (smallest prime >= n, so next_prime(7) = 7)"
        function_list = re.sub(r" \((?:[^()]|\([^()]*\))*\)", "", function_list)
        listed = {item.strip() for item in function_list.split(", ")}
        called = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\(", description))
        self.assertGreater(len(listed), 40)
        self.assertEqual(sorted((listed | called) - set(ef.BY_NAME)), [])
        undescribed = {row.name for row in ef.FUNCTIONS if not row.described}
        self.assertEqual(sorted((listed | called) & undescribed), [], "described=False rows the description names")


@unittest.skipUnless(find_spec("sympy"), "SymPy is not installed")
class TestSympyTable(unittest.TestCase):
    """The SymPy parser (static/sympy_cas.py) and the registry agree on names and arities."""

    table: Dict[str, Any] = {}

    @classmethod
    def setUpClass(cls) -> None:
        from static import sympy_cas

        cls.table = sympy_cas._function_table()

    def test_sympy_rows_are_in_the_table(self) -> None:
        expected = {row.name for row in ef.FUNCTIONS if row.sympy} | {"ln"}
        self.assertEqual(sorted(expected - set(self.table)), [])

    def test_table_names_are_rows_aliases_or_sympy_only(self) -> None:
        known: Set[str] = set(ef.BY_NAME) | {alias for row in ef.FUNCTIONS for alias in row.word_aliases}
        self.assertEqual(sorted(set(self.table) - known - SYMPY_ONLY_NAMES), [])
        self.assertEqual(sorted(SYMPY_ONLY_NAMES - set(self.table)), [])
        for name in set(self.table) & set(ef.BY_NAME):
            self.assertTrue(ef.BY_NAME[name].sympy, f"{name} is in the SymPy table; set sympy=True")

    def test_sympy_arities_fit_the_rows(self) -> None:
        for row in ef.FUNCTIONS:
            if not row.sympy:
                continue
            minimum, maximum = row.arity
            for count in self.table[row.name][1]:
                with self.subTest(name=row.name, count=count):
                    self.assertGreaterEqual(count, minimum)
                    self.assertTrue(maximum is None or count <= maximum)

    def test_constants_are_known_to_sympy(self) -> None:
        from static import sympy_cas

        self.assertLessEqual(set(ef.CONSTANTS) | {"i"}, set(sympy_cas._CONSTANTS))


class TestNerdamerRenames(unittest.TestCase):
    def test_renamed_functions_target_known_names(self) -> None:
        from utils.cas_text import _RENAMED_FUNCTIONS

        for source, target in _RENAMED_FUNCTIONS.items():
            with self.subTest(source=source):
                self.assertTrue(target in ef.BY_NAME or target in SYMPY_ONLY_NAMES, target)


if __name__ == "__main__":
    unittest.main()
