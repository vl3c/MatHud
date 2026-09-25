"""
MatHud Mathematical Expression Validation and Evaluation System

Secure mathematical expression parser and evaluator that validates and processes user input
for function plotting and mathematical computation. Provides safe expression evaluation
with comprehensive mathematical function support and syntax validation.

Key Features:
    - AST-based expression validation for security
    - Safe mathematical function evaluation
    - Expression syntax correction and normalization
    - Mathematical notation conversion (degrees, symbols, operators)
    - Comprehensive mathematical function library support
    - Python and Math.js compatibility modes

Security Features:
    - Restricted execution environment using AST parsing
    - Whitelist-based function and operator validation
    - Prevention of dangerous operations (imports, assignments, etc.)
    - Input sanitization and validation
    - Safe evaluation with controlled variable scope

Mathematical Support:
    - Standard arithmetic operations (+, -, *, /, ^, **)
    - Trigonometric functions (sin, cos, tan, sec, csc, cot and their inverses asin ... acot)
    - Hyperbolic functions (sinh, cosh, tanh, asinh, acosh, atanh)
    - Logarithmic functions (log, log10, log2, ln)
    - Advanced functions (sqrt, exp, abs, factorial)
    - Statistical functions (mean, median, mode, variance, stdev)
    - Mathematical constants (pi, e)
    - Calculus operations (derivative, integral, limit)
    - Algebraic operations (simplify, expand, factor, solve)

Expression Processing:
    - Automatic syntax correction and normalization
    - Mathematical notation conversion (sqrt, degrees, pi, factorial)
    - Unicode math notation (× ÷ − ≤ ≥ ≠, x², sin⁻¹(x), π ℯ ∞, Greek letters, Unicode spaces)
    - Implicit multiplication insertion (2x -> 2*x)
    - Power operator conversion (^ <-> **)
    - Function name standardization

Dependencies:
    - ast: Abstract syntax tree parsing for security
    - math: Standard mathematical functions
    - random: Random number generation
    - re: Regular expression pattern matching
    - utils.math_utils: Advanced mathematical operations
"""

from __future__ import annotations

import ast
import math
import random
import re
from typing import Any, Callable, Dict, Optional, Set, Type, cast


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


# Functions both Python evaluation namespaces (x and parametric t) share beyond the basics
_TRIGONOMETRIC_EXTRAS: Dict[str, Callable[[float], float]] = {
    "sec": _sec,
    "csc": _csc,
    "cot": _cot,
    "asec": _asec,
    "acsc": _acsc,
    "acot": _acot,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
}


# The ExpressionValidator class is used to validate and evaluate mathematical expressions
class ExpressionValidator(ast.NodeVisitor):
    """
    Secure mathematical expression validator and evaluator using AST parsing.

    Validates mathematical expressions against a whitelist of allowed operations
    and functions, then provides safe evaluation capabilities. Uses abstract
    syntax tree (AST) parsing to ensure security by preventing dangerous
    operations like imports, assignments, and arbitrary code execution.

    Security Model:
        - AST-based validation prevents code injection
        - Whitelist approach for allowed functions and operations
        - Controlled variable scope during evaluation
        - Prevention of dangerous Python operations

    Mathematical Capabilities:
        - Full arithmetic operation support
        - Comprehensive mathematical function library
        - Advanced mathematical operations (calculus, algebra)
        - Statistical analysis functions
        - Mathematical constant access

    Attributes:
        ALLOWED_NODES (set): Whitelist of permitted AST node types
        ALLOWED_FUNCTIONS (set): Whitelist of permitted mathematical functions
    """

    ALLOWED_NODES: Set[Type[ast.AST]] = {
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,  # Exponentiation operator (**)
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Name,
        ast.BinOp,
        ast.UnaryOp,
        ast.Expression,
        ast.Call,
        ast.List,  # List literals (e.g., [1, 2, 3])
    }
    ALLOWED_FUNCTIONS: Set[str] = {
        "sin",
        "cos",
        "tan",
        "sqrt",
        "log",
        "log10",
        "log2",
        "factorial",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "sec",
        "csc",
        "cot",
        "asec",
        "acsc",
        "acot",
        "asinh",
        "acosh",
        "atanh",
        "exp",
        "abs",
        "pi",
        "e",
        "pow",
        "det",
        "bin",
        "arrangements",
        "permutations",
        "combinations",
        "round",
        "ceil",
        "floor",
        "trunc",
        "max",
        "min",
        "sum",
        "limit",
        "derive",
        "integrate",
        "simplify",
        "expand",
        "factor",
        "solve",
        "gcd",
        "lcm",
        "is_prime",
        "prime_factors",
        "mod_pow",
        "mod_inverse",
        "next_prime",
        "prev_prime",
        "totient",
        "divisors",
        "mean",
        "median",
        "mode",
        "stdev",
        "variance",
        "random",
        "randint",
        "summation",
        "product",
        "arithmetic_sum",
        "geometric_sum",
        "geometric_sum_infinite",
        "ratio_test",
        "root_test",
        "p_series_test",
    }
    # Identifiers or number literals (with optional exponent), matched whole for implicit multiplication.
    # Note: "[-+]" rather than "[+-]" -- Brython's re fails to match "e-5" with the latter.
    _IMPLICIT_MULTIPLICATION_TOKEN = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
    # Evaluation namespace (built lazily once) and compiled code objects keyed by function string
    _functions: Optional[Dict[str, Any]] = None
    _compiled_cache: Dict[str, Any] = {}
    _COMPILED_CACHE_LIMIT = 256

    # ----- Unicode math notation (typed by users or copied into tool arguments by models) -----
    _NON_ASCII = re.compile(r"[^\x00-\x7f]")
    # Replaced one-for-one before any other processing.
    _UNICODE_REPLACEMENTS: Dict[str, str] = {
        # Multiplication signs
        "×": "*",  # U+00D7 multiplication sign
        "⋅": "*",  # U+22C5 dot operator
        "·": "*",  # U+00B7 middle dot
        "∙": "*",  # U+2219 bullet operator
        "•": "*",  # U+2022 bullet
        "∗": "*",  # U+2217 asterisk operator
        # Division signs
        "÷": "/",  # U+00F7 division sign
        "∕": "/",  # U+2215 division slash
        # Minus signs
        "−": "-",  # U+2212 minus sign
        "–": "-",  # U+2013 en dash
        # Comparisons ("≠" becomes "!=" only after factorials are handled, see fix_math_expression)
        "≤": "<=",  # U+2264
        "≥": ">=",  # U+2265
        # Degree sign look-alikes
        "˚": "°",  # U+02DA ring above
        "º": "°",  # U+00BA masculine ordinal indicator
        # Greek look-alikes and variant forms. Python folds these into the plain letter
        # (identifiers are NFKC-normalised) but math.js and nerdamer would not.
        "µ": "μ",  # U+00B5 micro sign
        "ϕ": "φ",  # U+03D5 phi symbol
        "ϵ": "ε",  # U+03F5 lunate epsilon
        "ϑ": "θ",  # U+03D1 theta symbol
        "ϱ": "ρ",  # U+03F1 rho symbol
        "ϰ": "κ",  # U+03F0 kappa symbol
        # Spaces become plain spaces
        "\u00a0": " ",  # no-break space
        "\u1680": " ",  # ogham space mark
        "\u2000": " ",
        "\u2001": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",  # thin space
        "\u200a": " ",  # hair space
        "\u202f": " ",  # narrow no-break space
        "\u205f": " ",  # medium mathematical space
        "\u3000": " ",  # ideographic space
        # Invisible characters are dropped
        "\u00ad": "",  # soft hyphen
        "\u200b": "",  # zero-width space
        "\u200c": "",  # zero-width non-joiner
        "\u200d": "",  # zero-width joiner
        "\u2060": "",  # word joiner
        "\ufeff": "",  # byte order mark / zero-width no-break space
    }
    _SUPERSCRIPT_DIGITS: Dict[str, str] = {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
    }
    # A superscript exponent: optional sign (⁻ U+207B, ⁺ U+207A) and a run of superscript digits
    _SUPERSCRIPT_POWER = re.compile("[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
    # A function name raised to a superscript power right before its argument list: sin²(x), sin⁻¹(x)
    _FUNCTION_SUPERSCRIPT = re.compile("(?<![A-Za-z_])([A-Za-z][A-Za-z0-9]*)([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)\\(")
    # Functions whose superscript applies to the result: sin²(x) = sin(x)^2
    _SUPERSCRIPT_FUNCTIONS: Set[str] = {
        "sin",
        "cos",
        "tan",
        "sec",
        "csc",
        "cot",
        "sinh",
        "cosh",
        "tanh",
        "asin",
        "acos",
        "atan",
        "log",
        "ln",
        "log10",
        "log2",
        "exp",
        "sqrt",
        "abs",
    }
    # ...except a ⁻¹ on these, which names the inverse function: sin⁻¹(x) = asin(x)
    _INVERSE_FUNCTIONS: Dict[str, str] = {
        "sin": "asin",
        "cos": "acos",
        "tan": "atan",
        "sec": "asec",
        "csc": "acsc",
        "cot": "acot",
        "sinh": "asinh",
        "cosh": "acosh",
        "tanh": "atanh",
    }
    _PI_SIGN = "π"  # U+03C0, rewritten to the constant name "pi"
    _SCRIPT_E = "ℯ"  # U+212F, rewritten to the constant name "e"
    _IMAGINARY_IOTA = "ί"  # U+03AF, GeoGebra's imaginary unit, rewritten to i (or j when python_compatible)
    _INFINITY_SIGN = "∞"  # U+221E, rewritten to inf (Python) or Infinity (math.js and nerdamer)
    _CONSTANT_SIGNS = _PI_SIGN + _SCRIPT_E + _INFINITY_SIGN + _IMAGINARY_IOTA
    _NOT_EQUAL_SIGN = "≠"  # U+2260
    # Δ and δ followed by ASCII letters or digits form one name (Δx, δt), not a product
    _DIFFERENCE_LETTERS = "Δδ"

    def _is_allowed_node_type(self, node: ast.AST) -> bool:
        """
        Check if the AST node type is in the allowed nodes whitelist.

        Args:
            node: AST node to validate

        Returns:
            bool: True if node type is allowed, False otherwise
        """
        return any(isinstance(node, node_type) for node_type in self.ALLOWED_NODES)

    def visit(self, node: ast.AST) -> Any:
        """
        Visit an AST node and validate it against security constraints.

        Args:
            node: AST node to visit

        Raises:
            ValueError: If node type is not in the allowed whitelist

        Returns:
            Result of visiting the node
        """
        if not self._is_allowed_node_type(node):
            raise ValueError(f"Disallowed node type: {type(node).__name__}")
        return super().visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """
        Visit name nodes and ensure they're only used for loading (reading) values.

        Args:
            node: AST Name node representing a variable or function name

        Raises:
            ValueError: If name is used for assignment or deletion
        """
        # Only allow Load context (reading), not Store or Del
        if not isinstance(node.ctx, ast.Load):
            raise ValueError(f"Name '{node.id}' cannot be used for assignment or deletion")

    def visit_List(self, node: ast.List) -> None:
        """
        Visit list literal nodes and validate their elements.

        Args:
            node: AST List node representing a list literal
        """
        # Visit all elements in the list
        for elt in node.elts:
            self.visit(elt)

    def visit_Call(self, node: ast.Call) -> None:
        """
        Visit function call nodes and validate against allowed functions.

        Args:
            node: AST Call node representing a function call

        Raises:
            ValueError: If function is not in the allowed functions whitelist
        """
        if isinstance(node.func, ast.Name) and node.func.id in self.ALLOWED_FUNCTIONS:
            # Visit the children of the ast.Call node manually
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword)
        else:
            raise ValueError(f"Disallowed function: {ast.dump(node)}")

    def visit_Import(self, node: ast.Import) -> None:
        raise ValueError("Import statements are not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise ValueError("Import statements are not allowed")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        raise ValueError("Lambda expressions are not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        raise ValueError("Attribute access is not allowed")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        raise ValueError("Subscripting is not allowed")

    def visit_Assign(self, node: ast.Assign) -> None:
        raise ValueError("Assignment is not allowed")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        raise ValueError("Augmented assignment is not allowed")

    # Note: ast.Exec was removed in Python 3.8+, so we skip defining visit_Exec
    # Exec statements would still be caught by the general visit method checking allowed nodes

    def visit_Global(self, node: ast.Global) -> None:
        raise ValueError("Global statements are not allowed")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise ValueError("Nonlocal statements are not allowed")

    @staticmethod
    def validate_expression_tree(expression: str) -> None:
        """
        Validate a mathematical expression using AST parsing.

        Parses the expression into an abstract syntax tree and validates
        all nodes against security constraints to ensure safe evaluation.

        Args:
            expression (str): Mathematical expression to validate

        Raises:
            ValueError: If expression contains disallowed operations or syntax errors
        """
        try:
            # Parse the expression into an abstract syntax tree
            tree = ast.parse(expression, mode="eval")
            validator = ExpressionValidator()
            validator.visit(tree)

        except SyntaxError as e:
            ExpressionValidator._handle_syntax_error(expression, e)
        except Exception as e:
            ExpressionValidator._handle_validation_error(expression, e)

    @staticmethod
    def _handle_syntax_error(expression: str, error: SyntaxError) -> None:
        """Handle syntax errors during expression validation"""
        print(f"Syntax error in expression: {expression}")
        raise ValueError(f"Syntax error in expression '{expression}': {str(error)}") from error

    @staticmethod
    def _handle_validation_error(expression: str, error: Exception) -> None:
        """Handle general validation errors during expression validation"""
        print(f"Invalid expression: {expression}")
        raise ValueError(f"Error validating expression '{expression}': {str(error)}") from error

    @staticmethod
    def evaluate_expression(expression: str, x: float = 0) -> float:
        """
        Safely evaluate a mathematical expression with controlled variable scope.

        Args:
            expression (str): Mathematical expression to evaluate
            x (float, optional): Value for variable x. Defaults to 0.

        Returns:
            float: Result of expression evaluation
        """
        variables_and_functions = ExpressionValidator._get_variables_and_functions(x)
        # Parse the expression into an abstract syntax tree
        tree = ast.parse(expression, mode="eval")
        # Evaluate the expression using the abstract syntax tree and the variables dictionary
        result = eval(compile(tree, "<string>", mode="eval"), variables_and_functions)
        return cast(float, result)

    @staticmethod
    def _get_variables_and_functions(x: float) -> Dict[str, Any]:
        """Create a dictionary with variables and functions for expression evaluation"""
        if ExpressionValidator._functions is None:
            ExpressionValidator._functions = ExpressionValidator._build_functions()
        variables_and_functions = dict(ExpressionValidator._functions)
        variables_and_functions["x"] = x
        return variables_and_functions

    @staticmethod
    def _build_functions() -> Dict[str, Any]:
        """Build the dictionary of functions and constants available to expressions (built once)"""
        from utils.math_utils import MathUtils

        return {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": MathUtils.sqrt,  # Square root function
            "log": math.log,  # Natural logarithm (base e)
            "log10": math.log10,  # Logarithm base 10
            "log2": math.log2,  # Logarithm base 2
            "factorial": math.factorial,  # Factorial function
            "asin": math.asin,  # Arcsine function
            "acos": math.acos,  # Arccosine function
            "atan": math.atan,  # Arctangent function
            "sinh": math.sinh,  # Hyperbolic sine function
            "cosh": math.cosh,  # Hyperbolic cosine function
            "tanh": math.tanh,  # Hyperbolic tangent function
            **_TRIGONOMETRIC_EXTRAS,  # sec, csc, cot, their inverses and the inverse hyperbolic functions
            "exp": math.exp,  # Exponential function
            "abs": abs,  # Absolute value function
            "pi": math.pi,  # The constant pi
            "e": math.e,  # The constant e
            "inf": math.inf,  # Infinity, written ∞
            "Infinity": math.inf,  # The math.js spelling of ∞, kept in stored function strings
            "pow": MathUtils.pow,  # Power function
            "bin": bin,  # Binary representation of an integer
            "det": MathUtils.det,  # Determinant of a matrix
            "arrangements": MathUtils.arrangements,  # Arrangements aka permutations nPk
            "permutations": MathUtils.permutations,  # Permutations
            "combinations": MathUtils.combinations,  # Combinations
            "limit": MathUtils.limit,  # Limit of a function
            "derive": MathUtils.derivative,  # Derivative of a function
            "integrate": MathUtils.integral,  # Indefinite integral of a function
            "simplify": MathUtils.simplify,  # Simplify an expression
            "expand": MathUtils.expand,  # Expand an expression
            "factor": MathUtils.factor,  # Factor an expression
            "solve": MathUtils.solve,  # Solve an equation
            "random": MathUtils.random,  # Generate a random number
            "round": MathUtils.round,  # Round a number
            "gcd": MathUtils.gcd,  # Greatest common divisor
            "lcm": MathUtils.lcm,  # Least common multiple
            "is_prime": MathUtils.is_prime,  # Check if number is prime
            "prime_factors": MathUtils.prime_factors,  # Prime factorization with multiplicity
            "mod_pow": MathUtils.mod_pow,  # Modular exponentiation
            "mod_inverse": MathUtils.mod_inverse,  # Modular multiplicative inverse
            "next_prime": MathUtils.next_prime,  # Find smallest prime >= n
            "prev_prime": MathUtils.prev_prime,  # Find largest prime <= n
            "totient": MathUtils.totient,  # Euler's totient function
            "divisors": MathUtils.divisors,  # All positive divisors
            "mean": MathUtils.mean,  # Mean of a list of numbers
            "median": MathUtils.median,  # Median of a list of numbers
            "mode": MathUtils.mode,  # Mode of a list of numbers
            "stdev": MathUtils.stdev,  # Standard deviation of a list of numbers
            "variance": MathUtils.variance,  # Variance of a list of numbers
            "ceil": math.ceil,  # Round up to the nearest integer
            "floor": math.floor,  # Round down to the nearest integer
            "trunc": math.trunc,  # Truncate to an integer
            "max": max,  # Maximum of a list of numbers
            "min": min,  # Minimum of a list of numbers
            "sum": sum,  # Sum of a list of numbers
            "randint": lambda a, b: random.randint(a, b),  # Random integer between a and b
            "summation": MathUtils.summation,
            "product": MathUtils.product,
            "arithmetic_sum": MathUtils.arithmetic_sum,
            "geometric_sum": MathUtils.geometric_sum,
            "geometric_sum_infinite": MathUtils.geometric_sum_infinite,
            "ratio_test": MathUtils.ratio_test,
            "root_test": MathUtils.root_test,
            "p_series_test": MathUtils.p_series_test,
        }

    @staticmethod
    def fix_math_expression(expression: str, python_compatible: bool = False) -> str:
        """
        Automatically correct and normalize mathematical expression syntax.

        Converts mathematical notation to proper syntax, handles special symbols,
        inserts implicit multiplication operators, and normalizes function names.

        Args:
            expression (str): Mathematical expression to fix
            python_compatible (bool): Whether to use Python-compatible syntax

        Returns:
            str: Corrected and normalized expression
        """
        expression = ExpressionValidator._normalize_unicode_notation(expression, python_compatible)
        expression = ExpressionValidator._convert_degrees(expression)
        expression = ExpressionValidator._handle_special_symbols(expression, python_compatible)
        expression = ExpressionValidator._replace_function_names(expression)
        expression = ExpressionValidator._handle_power_and_imaginary(expression, python_compatible)
        expression = ExpressionValidator._insert_multiplication_operators(expression, python_compatible)
        # Only now, so the "!" of "!=" is not read as a factorial
        return expression.replace(ExpressionValidator._NOT_EQUAL_SIGN, "!=")

    @staticmethod
    def normalize_unicode_math(expression: str, python_compatible: bool = False) -> str:
        """
        Rewrite Unicode math notation as the ASCII syntax the math engines parse.

        Covers the operator signs (× ÷ − ≤ ≥ ≠), superscript powers (x², x⁻¹, sin²(x),
        sin⁻¹(x)), the constants π, ℯ, ∞ and ί, Greek letter variants and Unicode spaces.
        Greek letters stay as they are: Python, math.js and nerdamer all accept them as
        variable names. ASCII input is returned unchanged.

        fix_math_expression already does this; call it directly only for expressions that
        go to nerdamer or math.js without passing through fix_math_expression.

        Args:
            expression (str): Expression that may contain Unicode math notation
            python_compatible (bool): Whether ∞ should become Python's inf (else Infinity)

        Returns:
            str: The expression in ASCII math syntax (Greek letters excepted)
        """
        expression = ExpressionValidator._normalize_unicode_notation(expression, python_compatible)
        expression = ExpressionValidator._convert_square_roots(expression)
        return expression.replace(ExpressionValidator._NOT_EQUAL_SIGN, "!=")

    @staticmethod
    def normalize_unicode_name(name: str) -> str:
        """
        Rewrite a variable name with the one-for-one character table only (ϕ -> φ, µ -> μ).

        Unlike normalize_unicode_math this never splits the name into factors or spells out
        constants, so it suits variable names passed next to an expression: the names in an
        evaluation scope or the variable of a derivative, integral, limit or solve.

        Args:
            name (str): Variable name that may contain Greek variant forms or Unicode spaces

        Returns:
            str: The name with the same characters the normalised expression uses
        """
        if not isinstance(name, str) or not ExpressionValidator._NON_ASCII.search(name):
            return name
        replacements = ExpressionValidator._UNICODE_REPLACEMENTS
        return "".join(replacements.get(char, char) for char in name)

    @staticmethod
    def _normalize_unicode_notation(expression: str, python_compatible: bool) -> str:
        """Rewrite Unicode math notation except "≠" (see normalize_unicode_math)."""
        if not ExpressionValidator._NON_ASCII.search(expression):
            return expression
        for symbol, replacement in ExpressionValidator._UNICODE_REPLACEMENTS.items():
            expression = expression.replace(symbol, replacement)
        expression = ExpressionValidator._convert_function_superscripts(expression)
        expression = ExpressionValidator._SUPERSCRIPT_POWER.sub(ExpressionValidator._superscript_power, expression)
        return ExpressionValidator._replace_unicode_symbols(expression, python_compatible)

    @staticmethod
    def _superscript_power(match: re.Match[str]) -> str:
        """Return "^exponent" for a superscript run, with a "*" when an operand follows: x²y -> x^2*y."""
        power = "^" + ExpressionValidator._superscript_exponent(match.group(0))
        following = match.string[match.end() : match.end() + 1]
        return power + "*" if ExpressionValidator._starts_operand(following) else power

    @staticmethod
    def _starts_operand(char: str) -> bool:
        """True when char can start a factor: an ASCII letter or digit, "(", "√", π, ℯ, ∞, ί or a Greek letter."""
        if char == "":
            return False
        is_ascii_alnum = "0" <= char <= "9" or "a" <= char <= "z" or "A" <= char <= "Z"
        return is_ascii_alnum or char in "(√" or ExpressionValidator._is_unicode_symbol(char)

    @staticmethod
    def _is_unicode_symbol(char: str) -> bool:
        """True for the single-letter tokens π, ℯ, ∞, ί and the Greek letters."""
        return char != "" and (
            char in ExpressionValidator._CONSTANT_SIGNS or ExpressionValidator._is_greek_letter(char)
        )

    @staticmethod
    def _superscript_exponent(superscript: str) -> str:
        """Return the ASCII exponent of a superscript run: "²³" -> "23", "⁻¹" -> "(-1)"."""
        digits = "".join(ExpressionValidator._SUPERSCRIPT_DIGITS.get(char, "") for char in superscript)
        return f"(-{digits})" if superscript.startswith("⁻") else digits

    @staticmethod
    def _convert_function_superscripts(expression: str) -> str:
        """Move a function's superscript after its call: sin²(x) -> sin(x)^2, sin⁻¹(x) -> asin(x)."""
        search_start = 0
        while True:
            match = ExpressionValidator._FUNCTION_SUPERSCRIPT.search(expression, search_start)
            if match is None:
                return expression
            name = match.group(1)
            open_index = match.end() - 1
            close_index = ExpressionValidator._find_closing_parenthesis(expression, open_index)
            if name not in ExpressionValidator._SUPERSCRIPT_FUNCTIONS or close_index < 0:
                search_start = match.end()  # not a function call: the superscript stays a plain power
                continue
            arguments = expression[open_index : close_index + 1]
            exponent = ExpressionValidator._superscript_exponent(match.group(2))
            if exponent == "(-1)" and name in ExpressionValidator._INVERSE_FUNCTIONS:
                replacement = ExpressionValidator._INVERSE_FUNCTIONS[name] + arguments
            else:
                replacement = f"{name}{arguments}^{exponent}"
            expression = expression[: match.start()] + replacement + expression[close_index + 1 :]
            search_start = match.start() + 1  # nested calls such as sin²(cos²(x)) are handled next

    @staticmethod
    def _find_closing_parenthesis(expression: str, open_index: int) -> int:
        """Return the index of the ")" matching the "(" at open_index, or -1 when unbalanced."""
        depth = 0
        for index in range(open_index, len(expression)):
            if expression[index] == "(":
                depth += 1
            elif expression[index] == ")":
                depth -= 1
                if depth == 0:
                    return index
        return -1

    @staticmethod
    def _is_greek_letter(char: str) -> bool:
        """True for the Greek letters Α-Ω and α-ω, which are kept as variable names."""
        return "Α" <= char <= "Ω" or "α" <= char <= "ω"

    @staticmethod
    def _continues_greek_name(letter: str, following: str) -> bool:
        """True when following extends the name the Greek letter starts: θ1, θ_0, Δx, δt."""
        if "0" <= following <= "9" or following == "_":
            return True
        is_ascii_letter = "a" <= following <= "z" or "A" <= following <= "Z"
        return is_ascii_letter and letter in ExpressionValidator._DIFFERENCE_LETTERS

    @staticmethod
    def _replace_unicode_symbols(expression: str, python_compatible: bool) -> str:
        """Spell out π, ℯ, ∞ and ί and give them and Greek letters explicit multiplication.

        Each of these characters is a single-letter token, so "2πr" is 2*pi*r and "αβ" is α*β.
        A Greek letter may still start a name with digits or underscores (θ1, θ_0), and Δ or δ
        followed by ASCII letters or digits is one name, the usual notation for a change or a
        small quantity (Δx, δt, Δx1), whereas other letters are factors ("ωt" is ω*t).
        """
        constants = {
            ExpressionValidator._PI_SIGN: "pi",
            ExpressionValidator._SCRIPT_E: "e",
            ExpressionValidator._INFINITY_SIGN: "inf" if python_compatible else "Infinity",
            ExpressionValidator._IMAGINARY_IOTA: "j" if python_compatible else "i",
        }
        parts = []
        previous = ""  # last character emitted
        for index, char in enumerate(expression):
            if not ExpressionValidator._is_unicode_symbol(char):
                parts.append(char)
                previous = char
                continue
            follows_number = "0" <= previous <= "9" or previous == "."
            imaginary_literal = char == ExpressionValidator._IMAGINARY_IOTA and follows_number  # 2ί -> 2i
            ends_operand = previous == ")" or (previous not in "(√" and ExpressionValidator._starts_operand(previous))
            if ends_operand and not imaginary_literal:
                parts.append("*")
            token = constants.get(char, char)
            parts.append(token)
            following = expression[index + 1 : index + 2]
            continues_name = char not in constants and ExpressionValidator._continues_greek_name(char, following)
            if ExpressionValidator._starts_operand(following) and not continues_name:
                parts.append("*")
                previous = "*"
            else:
                previous = token[-1]
        return "".join(parts)

    @staticmethod
    def _convert_degrees(expression: str) -> str:
        """Convert degree symbols and text to radians"""
        expression = expression.replace("°", " deg")
        expression = expression.replace("degrees", " deg")
        expression = expression.replace("degree", " deg")
        expression = re.sub(
            r"(\d+(?:\.\d+)?)\s*deg", lambda match: str(float(match.group(1)) * math.pi / 180), expression
        )
        return expression

    @staticmethod
    def _handle_special_symbols(expression: str, python_compatible: bool) -> str:
        """Handle square roots, absolute values, and factorials"""
        expression = ExpressionValidator._convert_square_roots(expression)

        # Replace | | with the Python equivalent if needed
        if python_compatible:
            expression = re.sub(r"\|(.*?)\|", r"abs(\1)", expression)

        # Handle factorials with balanced operand extraction
        expression = ExpressionValidator._replace_factorials(expression)
        return expression

    @staticmethod
    def _convert_square_roots(expression: str) -> str:
        """Replace √(...) and √x with sqrt() calls"""
        expression = re.sub(r"√\((.*?)\)", r"sqrt(\1)", expression)
        expression = re.sub(r"√([0-9a-zA-Z_Α-Ωα-ω]+)", r"sqrt(\1)", expression)
        return expression

    @staticmethod
    def _replace_factorials(expression: str) -> str:
        """Replace factorial shorthand (n!) with factorial() calls using balanced parsing."""
        if "!" not in expression:
            return expression

        def is_token_char(char: str) -> bool:
            return char.isalnum() or char in ["_", "."]

        matching_pairs = {")": "(", "]": "[", "}": "{"}

        index = expression.find("!")
        while index != -1:
            left = index - 1
            while left >= 0 and expression[left].isspace():
                left -= 1

            if left < 0:
                break

            start = left

            if expression[left] in matching_pairs:
                closing = expression[left]
                opening = matching_pairs[closing]
                depth = 1
                cursor = left - 1
                while cursor >= 0 and depth > 0:
                    char = expression[cursor]
                    if char == closing:
                        depth += 1
                    elif char == opening:
                        depth -= 1
                        if depth == 0:
                            break
                    cursor -= 1
                if depth != 0:
                    break
                start = cursor
                func_cursor = start - 1
                while func_cursor >= 0 and expression[func_cursor].isspace():
                    func_cursor -= 1
                while func_cursor >= 0 and is_token_char(expression[func_cursor]):
                    func_cursor -= 1
                start = func_cursor + 1
            else:
                while start >= 0 and is_token_char(expression[start]):
                    start -= 1
                start += 1

            operand = expression[start:index].strip()
            if not operand:
                break

            replacement = f"factorial({operand})"
            expression = expression[:start] + replacement + expression[index + 1 :]
            index = expression.find("!", start + len(replacement))

        return expression

    @staticmethod
    def _replace_function_names(expression: str) -> str:
        """Replace common mathematical function names with their Python equivalents"""
        replacements = ExpressionValidator._get_function_replacements()
        # Longest names first so "sine(" doesn't clobber "arcsine(", "cosine(" or "hyperbolic sine("
        for old, new in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            expression = expression.replace(old, new)
        return expression

    @staticmethod
    def _get_function_replacements() -> Dict[str, str]:
        """Get a dictionary of function name replacements"""
        return {
            "π": "pi",  # Using the variable from the dictionary
            "ln": "log",  # Python's math.log is ln by default
            "absolute(": "abs(",
            "power(": "pow(",
            "binary(": "bin(",
            "logarithm(": "log(",
            "logarithm10(": "log10(",
            "logarithm2(": "log2(",
            "square root(": "sqrt(",
            "sine(": "sin(",
            "cosine(": "cos(",
            "tangent(": "tan(",
            "arcsine(": "asin(",
            "arccosine(": "acos(",
            "arctangent(": "atan(",
            "hyperbolic sine(": "sinh(",
            "hyperbolic cosine(": "cosh(",
            "hyperbolic tangent(": "tanh(",
            "exponential(": "exp(",
            "determinant(": "det(",
            "std(": "stdev(",
            "var(": "variance(",
            "lim(": "limit(",
            "fact(": "factorial(",
            "rand(": "random(",
            "integral(": "integrate(",
            "derivative(": "derive(",
            "derivate(": "derive(",
        }

    @staticmethod
    def _handle_power_and_imaginary(expression: str, python_compatible: bool) -> str:
        """Handle power operators and imaginary numbers based on compatibility mode"""
        # Replace the power symbol with '**' if specified
        if python_compatible:
            expression = expression.replace("^", "**")
        else:
            expression = expression.replace("**", "^")

        # Replace 'i' with 'j' only in contexts likely to represent the imaginary unit
        imaginary_unit = "j" if python_compatible else "i"
        opposite_unit = "i" if python_compatible else "j"

        # Assuming it's used in the form of numbers like '2i' or standalone 'i'
        expression = re.sub(rf"(?<=\d){opposite_unit}\b", f"{imaginary_unit}", expression)  # For numbers like '2i'
        expression = re.sub(rf"\b{opposite_unit}\b", f"{imaginary_unit}", expression)  # For standalone 'i'

        return expression

    @staticmethod
    def _insert_multiplication_operators(expression: str, python_compatible: bool) -> str:
        """Insert multiplication operators where implicit multiplication is used"""
        imaginary_unit = "j" if python_compatible else "i"

        # Step 1: Protect "log" followed by any number from being altered
        expression = re.sub(r"log(\d+)", r"log[\1]", expression)

        # Step 2: Insert '*' between a number and a variable, function name, or parenthesis,
        # excluding 'i' or 'j' immediately after a number. Identifiers (atan2, x1) and number
        # literals with exponents (1e-5, 2.5E+3) are matched whole so they are never split.
        def insert_operator(match: re.Match[str]) -> str:
            token = match.group(0)
            next_char = expression[match.end() : match.end() + 1]
            is_number = token[0].isdigit() or token[0] == "."
            if is_number and next_char != imaginary_unit and re.match(r"[a-zA-Z_\(]", next_char):
                return token + "*"
            return token

        expression = re.sub(ExpressionValidator._IMPLICIT_MULTIPLICATION_TOKEN, insert_operator, expression)

        # Step 3: Revert "log" followed by any number back to its original form
        expression = re.sub(r"log\[(\d+)\]", r"log\1", expression)

        return expression

    @staticmethod
    def _parse_with_mathjs(function_string: str) -> Callable[[float], Any]:
        """Parse a function string using mathjs (slower but more powerful)"""
        from utils.math_utils import MathUtils

        return lambda x: MathUtils.evaluate(function_string, {"x": x})

    @staticmethod
    def _compile_function_string(function_string: str) -> Any:
        """Fix, validate and compile a function string, caching the code object per string"""
        cache = ExpressionValidator._compiled_cache
        compiled_code = cache.get(function_string)
        if compiled_code is None:
            fixed_string = ExpressionValidator.fix_math_expression(function_string, python_compatible=True)
            ExpressionValidator.validate_expression_tree(fixed_string)
            tree = ast.parse(fixed_string, mode="eval")
            compiled_code = compile(tree, "<string>", mode="eval")
            if len(cache) >= ExpressionValidator._COMPILED_CACHE_LIMIT:
                cache.clear()
            cache[function_string] = compiled_code
        return compiled_code

    @staticmethod
    def _parse_with_python(function_string: str) -> Callable[[float], float]:
        """Parse a function string using Python's built-in evaluation (faster)"""
        compiled_code = ExpressionValidator._compile_function_string(function_string)
        # One namespace per parsed function, reused across calls; only 'x' changes (hot plotting path)
        variables = ExpressionValidator._get_variables_and_functions(0)

        def evaluator(x: float) -> float:
            variables["x"] = x
            return float(eval(compiled_code, variables))

        return evaluator

    @staticmethod
    def parse_function_string(function_string: str, use_mathjs: bool = False) -> Callable[[float], Any]:
        """
        Parse a function string into a callable function object.

        Args:
            function_string (str): Mathematical function expression
            use_mathjs (bool): Whether to use Math.js parsing (slower but more powerful)

        Returns:
            callable: Function that can be called with x value
        """
        if use_mathjs:
            return ExpressionValidator._parse_with_mathjs(function_string)
        else:
            return ExpressionValidator._parse_with_python(function_string)

    @staticmethod
    def _get_variables_and_functions_parametric(t: float) -> Dict[str, Any]:
        """Create a dictionary with variables and functions for parametric expression evaluation.

        Similar to _get_variables_and_functions but uses 't' as the parameter variable
        instead of 'x', for parametric curves like x(t), y(t).
        """
        from utils.math_utils import MathUtils

        return {
            "t": t,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": MathUtils.sqrt,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "factorial": math.factorial,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            **_TRIGONOMETRIC_EXTRAS,
            "exp": math.exp,
            "abs": abs,
            "pi": math.pi,
            "e": math.e,
            "inf": math.inf,
            "Infinity": math.inf,
            "pow": MathUtils.pow,
            "ceil": math.ceil,
            "floor": math.floor,
            "trunc": math.trunc,
            "max": max,
            "min": min,
            "round": MathUtils.round,
        }

    @staticmethod
    def _parse_parametric_with_python(expression_string: str) -> Callable[[float], float]:
        """Parse a parametric expression string using Python's built-in evaluation.

        Uses 't' as the parameter variable instead of 'x'.
        """
        compiled_code = ExpressionValidator._compile_function_string(expression_string)
        # One namespace per parsed expression, reused across calls; only 't' changes
        variables = ExpressionValidator._get_variables_and_functions_parametric(0)

        def evaluator(t: float) -> float:
            variables["t"] = t
            return float(eval(compiled_code, variables))

        return evaluator

    @staticmethod
    def parse_parametric_expression(expression_string: str) -> Callable[[float], float]:
        """
        Parse a parametric expression string into a callable function object.

        The expression should use 't' as the parameter variable.
        Examples: "cos(t)", "t*sin(t)", "3*t + 1"

        Args:
            expression_string (str): Mathematical expression using 't' as parameter

        Returns:
            callable: Function that can be called with t value and returns a float
        """
        return ExpressionValidator._parse_parametric_with_python(expression_string)
