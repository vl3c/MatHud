import unittest
import math
from expression_validator import ExpressionValidator


class TestExpressionValidator(unittest.TestCase):
    def test_validate_expression_tree_valid(self):
        valid_expressions = [
            "x**2",
            "sin(x) + cos(x)",
            "sqrt(x) * log(x)",
            "exp(x) / tan(x)",
            "log10(x) + log2(x)",
            "factorial(5) - asin(0.5)",
            "acos(0.5) + atan(1)",
            "sinh(x) - cosh(x)",
            "tanh(x) + exp(1)",
            "abs(-x) + pi",
            "e * pow(2, 3)",
            "det([[1, 2], [3, 4]])",
            "bin(10)",
            "round(x) + ceil(x)",
            "floor(x) - trunc(x)",
            "max(x, y) * min(x, y)",
            "sum([1, 2, 3, x])",
            "arrangements(6, 3)",
            "permutations(5, 2)",
            "permutations(5)",
            "combinations(6, 3)",
            "limit(sin(x)/x, x, 0)",
            "derive(x**2, x)",
            "integrate(x**2, x)",
            "simplify(sin(x)**2 + cos(x)**2)",
            "expand((x + 1)**2)",
            "factor(x**2 - 1)",
            "solve(x**2 - 4, x)",
            "gcd(100, 80)",
            "lcm(5, 10)",
            "mean([1, 2, 3, x])",
            "median([1, 2, 3, x])",
            "mode([1, 1, 2, 3, x])",
            "stdev([1, 2, 3, x])",
            "variance([1, 2, 3, x])",
            "random()",
            "randint(1, 10)"
        ]
        for expr in valid_expressions:
            with self.subTest(expr=expr):
                try:
                    ExpressionValidator.validate_expression_tree(expr)
                except ValueError as e:
                    self.fail(f"Unexpected exception for '{expr}': {e}")

    def test_validate_expression_tree_invalid(self):
        invalid_expressions = [
            "", " ", "\t", "\n",  # Empty or whitespace-only strings
            "import os",  # Import statements
            "from os import system",  # From...import statements
            "__import__('os')",  # Disallowed function call
            "__import__('os').system('ls')",  # Attribute access and disallowed function call
            "os.system('ls')",  # Attribute access and disallowed function call
            "[1, 2, 3][0]",  # Subscripting
            "x = 1",  # Assignment
            "x += 1",  # Augmented assignment
            "exec('print(1)')",  # Exec statements
            "global x",  # Global statements
            "nonlocal x",  # Nonlocal statements
            "lambda x: x + 2",  # Lambda expressions
            "tan2(x)",
            "fact(5)",
            "asinn(0.5)",
            "acoss(0.5)",
            "atann(1)",
            "sinhh(x)",
            "cossh(x)",
            "tanhh(x)",
            "expp(1)"
        ]
        for expr in invalid_expressions:
            with self.subTest(expr=expr):
                with self.assertRaises(ValueError):
                    ExpressionValidator.validate_expression_tree(expr)

    def test_evaluate_expression(self):
        x = 2  # Define x for use in expressions
        expressions = {
            "sin(pi/2)": math.sin(math.pi / 2),
            "sqrt(16)": 4,
            "sqrt(-4)": '2i',
            "log(e)": 1.0,
            "cos(pi)": math.cos(math.pi),
            "tan(pi/4)": math.tan(math.pi / 4),
            "log10(100)": 2.0,
            "log2(16)": 4.0,
            "factorial(5)": 120.0,
            "(2+3)!": 120.0,
            "10!/(3!*(10-3)!)": 120.0,
            "(3!)!": math.factorial(math.factorial(3)),
            "asin(1)": math.pi / 2,
            "acos(0)": math.pi / 2,
            "atan(1)": math.pi / 4,
            "sinh(0)": 0.0,
            "cosh(0)": 1.0,
            "tanh(0)": 0.0,
            "exp(1)": math.e,
            "abs(-5)": 5.0,
            "pow(2, 3)": 8,
            "bin(10)": '0b1010',
            "det([[1, 2], [3, 4]])": -2.0,
            "arrangements(6, 3)": math.perm(6, 3),
            "permutations(5, 2)": math.perm(5, 2),
            "permutations(5)": math.factorial(5),
            "combinations(6, 3)": math.comb(6, 3),
            "x": x
        }
        for expr, expected in expressions.items():
            with self.subTest(expr=expr):
                evaluation_expr = expr
                if '!' in expr:
                    evaluation_expr = ExpressionValidator.fix_math_expression(expr, python_compatible=True)
                result = ExpressionValidator.evaluate_expression(evaluation_expr, x=x)
                if isinstance(expected, float):
                    self.assertAlmostEqual(result, expected)
                else:
                    self.assertEqual(result, expected)

    def test_degree_to_radian_conversion(self):
        expressions_and_expected = {
            "sin(30deg)": f"sin({30 * math.pi / 180})",  # Should convert 30 degrees to radians
            "cos(45deg) + sin(90deg)": f"cos({45 * math.pi / 180}) + sin({90 * math.pi / 180})",  # Converts both 45 and 90 degrees to radians
            "tan(60deg)": f"tan({60 * math.pi / 180})",  # Converts 60 degrees to radians
            "sin(30°)": f"sin({30 * math.pi / 180})",  # Should convert 30 degrees to radians
            "cos(45 degrees) + sin(90 degree)": f"cos({45 * math.pi / 180}) + sin({90 * math.pi / 180})",  # Converts both 45 and 90 degrees to radians
        }
        for expr, expected in expressions_and_expected.items():
            with self.subTest(expr=expr):
                fixed_expr = ExpressionValidator.fix_math_expression(expr, python_compatible=False)
                self.assertAlmostEqual(eval(fixed_expr, {"sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi}), \
                                       eval(expected, {"sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi}))

    def test_fix_math_expression_python_compatibility(self):
        expressions_and_fixes = {
            "sin(pi/4)^2": "sin(pi/4)**2",
            "2x": "2*x",
            "e^x": "e**x",
            "2sin(x)": "2*sin(x)",
            "|x|": "abs(x)",
            "√x": "sqrt(x)",
            "2j": "2j",
            "3.14i": "3.14j",
        }
        for expr, expected_fix in expressions_and_fixes.items():
            with self.subTest(expr=expr):
                fixed_expr = ExpressionValidator.fix_math_expression(expr, python_compatible=True)
                self.assertEqual(fixed_expr, expected_fix)

    def test_fix_math_expression_js_compatibility(self):
        expressions_and_expected = {
            # Checks for reverting Python-compatible transformations
            "sqrt(x)**2": "sqrt(x)^2",
            "x**2": "x^2",  # Power operator replaced with '^'
            "|x|": "|x|",
            "3.14j": "3.14i",  # Complex number notation reverted to 'i'
            "2i": "2i",
            "e**x": "e^x",  # Exponential notation
        }
        for expr, expected in expressions_and_expected.items():
            with self.subTest(expr=expr):
                fixed_expr = ExpressionValidator.fix_math_expression(expr, python_compatible=False)
                self.assertEqual(fixed_expr, expected)

    def test_parse_function_string_returns_number(self):
        function_elements = [
            "sin(pi/4)",
            "cos(π/3)",
            "sqrt(16)",
            "tan(5)",
            "log(e)",
            "log10(100)",
            "log2(8)",
            "factorial(3)",
            "asin(0.5)",
            "acos(0.5)",
            "atan(1)",
            "sinh(1)",
            "cosh(1)",
            "tanh(0)",
            "exp(1)",
            "abs(-pi)",
            "pow(2, 3)",
            "det([[1, 2], [3, 4]])"
        ]
        for element in function_elements:
            with self.subTest(element=element):
                f = ExpressionValidator.parse_function_string(element)
                # Test with a range of values
                for x in range(-10, 11):
                    result = f(x)
                    self.assertIsInstance(result, (int, float), f"Result of expression '{element}' for x={x} is not a number: {result}")

    def test_parse_function_string(self):
        expressions = {
            "x^2": lambda x: x**2,
            "sin(x)": math.sin,
            "log(x)": math.log,
            "log10(x)": math.log10,
            "sqrt(x)": math.sqrt,
            "exp(x)": math.exp,
            "tan(x)": math.tan,
            "sin(pi/4) + cos(π/3) - tan(sqrt(16)) * log(e) / log10(100) + log2(8) * factorial(3) + asin(0.5) - acos(0.5) + atan(1) + sinh(1) - cosh(1) + tanh(0) + exp(1) - abs(-pi) + pow(2, 3)^2": lambda x: 82.09880526150872
        }
        for use_mathjs in [False, True]:
            for expr, expected_func in expressions.items():
                with self.subTest(expr=expr, use_mathjs=use_mathjs):
                    func = ExpressionValidator.parse_function_string(expr, use_mathjs=use_mathjs)
                    test_values = [0.1, 1, math.pi]
                    for value in test_values:
                        expected_result = expected_func(value)
                        result = func(value)
                        print(f"Expected: {expected_result}, Result: {result}")
                        self.assertAlmostEqual(result, expected_result, msg=f"Failed on expr: {expr} with x={value}")

    def test_fix_math_expression_factorials(self):
        cases = [
            ("10!/(3!*(10-3)!)", "factorial(10)/(factorial(3)*factorial((10-3)))"),
            ("(3!)!", "factorial((factorial(3)))"),
            ("((2+3)!)!", "factorial((factorial((2+3))))"),
            ("sin(x!) + (cos(y)!)", "sin(factorial(x))+(factorial(cos(y)))"),
            ("((1+2)*(3+4))!", "factorial(((1+2)*(3+4)))")
        ]

        for expression, expected in cases:
            for python_compatible in (True, False):
                fixed_expression = ExpressionValidator.fix_math_expression(expression, python_compatible=python_compatible)
                self.assertNotIn('!', fixed_expression)
                self.assertEqual(fixed_expression.replace(' ', ''), expected)

