"""Tests for the Unicode math notation accepted by ExpressionValidator.

Users type symbols such as π, x², × and ≤ in the chat and models copy them into tool
arguments, so fix_math_expression and normalize_unicode_math rewrite them as ASCII math.
The validator is plain Python, so these run under CPython; evaluation uses a small local
namespace because the full one needs the browser-only MathUtils.
"""

from __future__ import annotations

import ast
import math
import unittest
from typing import Dict, List, Tuple

from server_tests import python_path_setup  # noqa: F401
from expression_validator import ExpressionValidator

# Input -> (python_compatible=True, python_compatible=False)
FIX_CASES: Dict[str, Tuple[str, str]] = {
    # Multiplication, division and minus signs
    "3×4": ("3*4", "3*4"),
    "a⋅b": ("a*b", "a*b"),
    "a·b": ("a*b", "a*b"),
    "a∙b": ("a*b", "a*b"),
    "a•b": ("a*b", "a*b"),
    "a∗b": ("a*b", "a*b"),
    "6÷2": ("6/2", "6/2"),
    "6∕2": ("6/2", "6/2"),
    "5 − 2": ("5 - 2", "5 - 2"),
    "5 – 2": ("5 - 2", "5 - 2"),
    "2x−1": ("2*x-1", "2*x-1"),
    # Comparisons
    "x ≤ 3": ("x <= 3", "x <= 3"),
    "x ≥ 3": ("x >= 3", "x >= 3"),
    "x ≠ 3": ("x != 3", "x != 3"),
    "x ≠ 3!": ("x != factorial(3)", "x != factorial(3)"),
    # Infinity
    "∞": ("inf", "Infinity"),
    "-∞": ("-inf", "-Infinity"),
    "1/∞": ("1/inf", "1/Infinity"),
    "2∞": ("2*inf", "2*Infinity"),
    # Superscript powers
    "x²": ("x**2", "x^2"),
    "x² + 3x": ("x**2 + 3*x", "x^2 + 3*x"),
    "x²³": ("x**23", "x^23"),
    "x⁻¹": ("x**(-1)", "x^(-1)"),
    "x⁻²³": ("x**(-23)", "x^(-23)"),
    "x⁺²": ("x**2", "x^2"),
    "x¹⁰": ("x**10", "x^10"),
    "(x+1)²": ("(x+1)**2", "(x+1)^2"),
    "2²": ("2**2", "2^2"),
    "10⁻³": ("10**(-3)", "10^(-3)"),
    "x²y": ("x**2*y", "x^2*y"),
    "x²(x+1)": ("x**2*(x+1)", "x^2*(x+1)"),
    "x³π": ("x**3*pi", "x^3*pi"),
    # Superscripts on function names
    "sin²(x)": ("sin(x)**2", "sin(x)^2"),
    "2sin²(x)": ("2*sin(x)**2", "2*sin(x)^2"),
    "cos³(2x)": ("cos(2*x)**3", "cos(2*x)^3"),
    "sin²(cos²(x))": ("sin(cos(x)**2)**2", "sin(cos(x)^2)^2"),
    "sin²(x) + cos²(x)": ("sin(x)**2 + cos(x)**2", "sin(x)^2 + cos(x)^2"),
    "ln²(x)": ("log(x)**2", "log(x)^2"),
    "sin⁻¹(x)": ("asin(x)", "asin(x)"),
    "cos⁻¹(0.5)": ("acos(0.5)", "acos(0.5)"),
    "tan⁻¹(1)": ("atan(1)", "atan(1)"),
    "sinh⁻¹(x)": ("asinh(x)", "asinh(x)"),
    "log⁻¹(x)": ("log(x)**(-1)", "log(x)^(-1)"),
    "sin⁻¹(x/(1+x))": ("asin(x/(1+x))", "asin(x/(1+x))"),
    # Constants
    "π": ("pi", "pi"),
    "2π": ("2*pi", "2*pi"),
    "2πr": ("2*pi*r", "2*pi*r"),
    "πr^2": ("pi*r**2", "pi*r^2"),
    "πr²": ("pi*r**2", "pi*r^2"),
    "π(x+1)": ("pi*(x+1)", "pi*(x+1)"),
    "(x+1)π": ("(x+1)*pi", "(x+1)*pi"),
    "π2": ("pi*2", "pi*2"),
    "sin(π/2)": ("sin(pi/2)", "sin(pi/2)"),
    "π√2": ("pi*sqrt(2)", "pi*sqrt(2)"),
    "√π": ("sqrt(pi)", "sqrt(pi)"),
    "ℯ": ("e", "e"),
    "2ℯ": ("2*e", "2*e"),
    "ℯ^x": ("e**x", "e^x"),
    "ί": ("1j", "i"),
    "2.5ί": ("2.5j", "2.5i"),
    "-ί": ("-1j", "-i"),
    "2ί": ("2j", "2i"),
    "3+4ί": ("3+4j", "3+4i"),
    "ίx": ("1j*x", "i*x"),
    "xί": ("x*1j", "x*i"),
    # Greek letters stay as variable names, each a single-letter factor
    "θ": ("θ", "θ"),
    "2θ": ("2*θ", "2*θ"),
    "rθ": ("r*θ", "r*θ"),
    "αβ": ("α*β", "α*β"),
    "sin(θ)": ("sin(θ)", "sin(θ)"),
    "θ²": ("θ**2", "θ^2"),
    "θ1 + θ_0": ("θ1 + θ_0", "θ1 + θ_0"),
    "√θ": ("sqrt(θ)", "sqrt(θ)"),
    "λ": ("λ", "λ"),
    # Δ and δ followed by ASCII letters or digits name one quantity; other letters are factors
    "Δx": ("Δx", "Δx"),
    "δt": ("δt", "δt"),
    "Δx1": ("Δx1", "Δx1"),
    "2Δx": ("2*Δx", "2*Δx"),
    "Δx/Δt": ("Δx/Δt", "Δx/Δt"),
    "ΔxΔt": ("Δx*Δt", "Δx*Δt"),
    "Δx²": ("Δx**2", "Δx^2"),
    "Δ(x+1)": ("Δ*(x+1)", "Δ*(x+1)"),
    "ωt": ("ω*t", "ω*t"),
    "sin(ωt)": ("sin(ω*t)", "sin(ω*t)"),
    "2πθ": ("2*pi*θ", "2*pi*θ"),
    # Variant forms and look-alikes
    "µ": ("μ", "μ"),
    "ϕ + ϵ + ϑ + ϱ + ϰ": ("φ + ε + θ + ρ + κ", "φ + ε + θ + ρ + κ"),
    # Degree sign look-alikes
    "sin(30˚)": (f"sin({30 * math.pi / 180})", f"sin({30 * math.pi / 180})"),
    "sin(30º)": (f"sin({30 * math.pi / 180})", f"sin({30 * math.pi / 180})"),
    # Unicode spaces and invisible characters
    "x\u00a0+\u00a01": ("x + 1", "x + 1"),
    "x\u2009+\u202f1": ("x + 1", "x + 1"),
    "x\u2003+\u30001": ("x + 1", "x + 1"),
    "x\u200b+1": ("x+1", "x+1"),
    "\ufeffx+\u200d1\u2060": ("x+1", "x+1"),
    "2\u00adx": ("2*x", "2*x"),
}

# Plain ASCII inputs the normalisation must leave alone (and whose fixed form must not change)
ASCII_SAMPLES = [
    "x**2",
    "x^2 + 3x",
    "2x",
    "2pi",
    "pir",
    "sin(pi/4)^2",
    "e^x",
    "2sin(x)",
    "|x|",
    "2j",
    "3.14i",
    "1e-5",
    "2.5E+3",
    "atan2(1,1)",
    "log10(100)",
    "10!/(3!*(10-3)!)",
    "sin(30deg)",
    "cos(45 degrees)",
    "x <= 3",
    "x != 3",
    "3!=6",
    "inf",
    "Infinity",
    "limit(sin(x)/x, x, 0)",
    "derive(x**2, x)",
    "solve(x^2 - 4 = 0, x)",
    "hyperbolic sine(0.5)",
    "",
    "   ",
]

# Expected values of fix_math_expression(..., python_compatible=True) evaluated in Python
EVALUATION_CASES: List[Tuple[str, Dict[str, float], float]] = [
    ("2πr", {"r": 1}, 2 * math.pi),
    ("πr²", {"r": 2}, 4 * math.pi),
    ("x²", {"x": 3}, 9),
    ("x² + 3x", {"x": 3}, 18),
    ("x²³", {"x": 1.1}, 1.1**23),
    ("x⁻¹", {"x": 4}, 0.25),
    ("(x+1)²", {"x": 2}, 9),
    ("3×4", {}, 12),
    ("6÷2", {}, 3),
    ("5 − 2", {}, 3),
    ("2⋅3·4", {}, 24),
    ("sin²(x) + cos²(x)", {"x": 0.7}, 1),
    ("sin⁻¹(1)", {}, math.pi / 2),
    ("2ℯ", {}, 2 * math.e),
    ("1/∞", {}, 0),
    ("2θ + α", {"θ": 1.5, "α": 1}, 4),
    ("Δx/Δt", {"Δx": 3, "Δt": 2}, 1.5),
    ("µ", {"μ": 7}, 7),
    ("√π", {}, math.sqrt(math.pi)),
]


def _python_namespace(variables: Dict[str, float]) -> Dict[str, object]:
    namespace: Dict[str, object] = {
        "pi": math.pi,
        "e": math.e,
        "inf": math.inf,
        "sin": math.sin,
        "cos": math.cos,
        "asin": math.asin,
        "sqrt": math.sqrt,
    }
    namespace.update(variables)
    return namespace


class TestFixMathExpressionUnicode(unittest.TestCase):
    def test_unicode_notation_is_rewritten(self) -> None:
        for expression, (python_expected, js_expected) in FIX_CASES.items():
            with self.subTest(expression=expression):
                self.assertEqual(
                    ExpressionValidator.fix_math_expression(expression, python_compatible=True), python_expected
                )
                self.assertEqual(
                    ExpressionValidator.fix_math_expression(expression, python_compatible=False), js_expected
                )

    def test_pi_is_separated_from_neighbouring_names(self) -> None:
        # Regression: "2πr" used to become "2*pir" and "πr^2" "pir**2"
        self.assertEqual(ExpressionValidator.fix_math_expression("2πr", python_compatible=True), "2*pi*r")
        self.assertEqual(ExpressionValidator.fix_math_expression("πr^2", python_compatible=True), "pi*r**2")

    def test_unicode_results_pass_validation(self) -> None:
        for expression in ("2πr", "x²⁺¹", "sin⁻¹(x)", "3×4÷2", "2θ + α", "ℯ^x − 1", "∞", "√π", "5ί"):
            with self.subTest(expression=expression):
                fixed = ExpressionValidator.fix_math_expression(expression, python_compatible=True)
                ExpressionValidator.validate_expression_tree(fixed)

    def test_comparisons_fail_validation_with_a_readable_expression(self) -> None:
        with self.assertRaises(ValueError) as context:
            ExpressionValidator.validate_expression_tree(ExpressionValidator.fix_math_expression("x ≤ 3", True))
        self.assertIn("x <= 3", str(context.exception))

    def test_fixed_expressions_evaluate(self) -> None:
        for expression, variables, expected in EVALUATION_CASES:
            with self.subTest(expression=expression):
                fixed = ExpressionValidator.fix_math_expression(expression, python_compatible=True)
                ExpressionValidator.validate_expression_tree(fixed)
                code = compile(ast.parse(fixed, mode="eval"), "<test>", mode="eval")
                self.assertAlmostEqual(eval(code, _python_namespace(variables)), expected)

    def test_imaginary_iota_evaluates_as_complex(self) -> None:
        fixed = ExpressionValidator.fix_math_expression("3+4ί", python_compatible=True)
        self.assertEqual(eval(fixed, {}), complex(3, 4))

    def test_bare_imaginary_iota_evaluates_as_complex(self) -> None:
        # Regression: a lone ί became the name j (NameError) in Python mode
        for expression, variables, expected in (("ί", {}, 1j), ("ί*ί", {}, -1), ("2 + ίx", {"x": 3}, 2 + 3j)):
            with self.subTest(expression=expression):
                fixed = ExpressionValidator.fix_math_expression(expression, python_compatible=True)
                ExpressionValidator.validate_expression_tree(fixed)
                self.assertEqual(eval(fixed, dict(variables)), expected)

    def test_ascii_input_is_unchanged(self) -> None:
        for expression in ASCII_SAMPLES:
            for python_compatible in (True, False):
                with self.subTest(expression=expression, python_compatible=python_compatible):
                    self.assertEqual(
                        ExpressionValidator._normalize_unicode_notation(expression, python_compatible), expression
                    )
                    self.assertEqual(
                        ExpressionValidator.normalize_unicode_math(expression, python_compatible), expression
                    )

    def test_ascii_fixes_are_unchanged(self) -> None:
        # Spot checks of existing behaviour for plain ASCII input
        cases = {
            "x^2 + 3x": ("x**2 + 3*x", "x^2 + 3*x"),
            "2pi": ("2*pi", "2*pi"),
            "pir": ("pir", "pir"),
            "3.14i": ("3.14j", "3.14i"),
            "1e-5": ("1e-5", "1e-5"),
            "inf": ("inf", "inf"),
        }
        for expression, (python_expected, js_expected) in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(
                    ExpressionValidator.fix_math_expression(expression, python_compatible=True), python_expected
                )
                self.assertEqual(
                    ExpressionValidator.fix_math_expression(expression, python_compatible=False), js_expected
                )


class TestNormalizeUnicodeMath(unittest.TestCase):
    """normalize_unicode_math prepares raw expressions for nerdamer and math.js."""

    def test_raw_expressions_for_math_engines(self) -> None:
        cases = {
            "x³ − 2x": "x^3 - 2x",
            "2πr": "2*pi*r",
            "x²y": "x^2*y",
            "sin²(x) + cos²(x)": "sin(x)^2 + cos(x)^2",
            "sin⁻¹(x)": "asin(x)",
            "x ≤ 3": "x <= 3",
            "x ≠ 3": "x != 3",
            "∞": "Infinity",
            "-∞": "-Infinity",
            "2ί": "2i",
            "√(x+1)": "sqrt(x+1)",
            "√x": "sqrt(x)",
            "3×4÷2": "3*4/2",
            "θ² + ϕ": "θ^2 + φ",
            "x\u00a0=\u20093": "x = 3",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(ExpressionValidator.normalize_unicode_math(expression), expected)

    def test_every_superscript_digit(self) -> None:
        self.assertEqual(ExpressionValidator._superscript_exponent("⁰¹²³⁴⁵⁶⁷⁸⁹"), "0123456789")
        self.assertEqual(ExpressionValidator._superscript_exponent("⁻⁴²"), "(-42)")

    def test_variable_names_use_the_character_table_only(self) -> None:
        # Names in a scope or a variable argument must match the normalised expression
        # without being split into factors or having constants spelled out
        cases = {"ϕ": "φ", "µ": "μ", "ϑ_1": "θ_1", "Δx": "Δx", "αβ": "αβ", "x": "x", "π": "π"}
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(ExpressionValidator.normalize_unicode_name(name), expected)

    def test_python_compatible_infinity(self) -> None:
        self.assertEqual(ExpressionValidator.normalize_unicode_math("-∞", python_compatible=True), "-inf")

    def test_normalisation_is_idempotent(self) -> None:
        for expression in FIX_CASES:
            with self.subTest(expression=expression):
                once = ExpressionValidator.normalize_unicode_math(expression)
                self.assertEqual(ExpressionValidator.normalize_unicode_math(once), once)


class TestPythonEvaluationNamespace(unittest.TestCase):
    """The real plotting namespaces (MathUtils is importable behind the browser stub)."""

    @classmethod
    def setUpClass(cls) -> None:
        from server_tests import client_renderer  # noqa: F401  (installs the browser stub)

    def test_math_js_infinity_round_trips(self) -> None:
        # Function and ParametricFunction store the math.js form and parse it again later
        stored = ExpressionValidator.fix_math_expression("min(x, ∞)")
        self.assertEqual(stored, "min(x, Infinity)")
        self.assertEqual(ExpressionValidator.parse_function_string(stored)(3), 3.0)
        stored_t = ExpressionValidator.fix_math_expression("max(t, −∞)")
        self.assertEqual(stored_t, "max(t, -Infinity)")
        self.assertEqual(ExpressionValidator.parse_parametric_expression(stored_t)(2), 2.0)

    def test_reciprocal_and_inverse_hyperbolic_functions_plot(self) -> None:
        # Values match math.js 14.5.2, which evaluates the same names outside plotting
        cases = {
            "sinh⁻¹(x)": (1, math.asinh(1)),
            "cosh⁻¹(x)": (2, 1.3169578969248166),
            "tanh⁻¹(x)": (0.5, 0.5493061443340548),
            "sec⁻¹(x)": (-2, 2.0943951023931957),
            "csc⁻¹(x)": (-2, -0.5235987755982989),
            "cot⁻¹(x)": (-1, -0.7853981633974483),
            "acot(x)": (0, math.pi / 2),
            "sec(x)": (1, 1.8508157176809255),
            "csc(x)": (1, 1.1883951057781212),
            "cot(x)": (1, 0.6420926159343306),
            "sec²(x) - tan²(x)": (0.4, 1),
        }
        for expression, (x, expected) in cases.items():
            with self.subTest(expression=expression):
                self.assertAlmostEqual(ExpressionValidator.parse_function_string(expression)(x), expected)
                t_expression = expression.replace("x", "t")
                self.assertAlmostEqual(ExpressionValidator.parse_parametric_expression(t_expression)(x), expected)


class TestNumericSolverVariableDetection(unittest.TestCase):
    """detect_variables must find the Greek names normalize_unicode_math keeps."""

    def test_greek_variables_are_detected(self) -> None:
        from server_tests import client_renderer  # noqa: F401  (installs the browser stub)
        from numeric_solver.expression_utils import detect_variables

        self.assertEqual(detect_variables(["θ^2 = 2"]), ["θ"])
        self.assertEqual(detect_variables(["2*pi*r + α*β = x"]), ["r", "x", "α", "β"])
        self.assertEqual(detect_variables(["Δx*2 = δt1 + π"]), ["Δx", "δt1"])
        normalized = ExpressionValidator.normalize_unicode_math("2Δx + ωt = 1")
        self.assertEqual(detect_variables([normalized]), ["t", "Δx", "ω"])

    def test_ascii_detection_is_unchanged(self) -> None:
        from server_tests import client_renderer  # noqa: F401
        from numeric_solver.expression_utils import detect_variables

        self.assertEqual(detect_variables(["sin(x) + y = 1"]), ["x", "y"])
        self.assertEqual(detect_variables(["log(a) + exp(b) = 0", "x + pi = 0"]), ["a", "b", "x"])


if __name__ == "__main__":
    unittest.main()
