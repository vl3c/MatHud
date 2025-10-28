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
    - Trigonometric functions (sin, cos, tan, asin, acos, atan)
    - Hyperbolic functions (sinh, cosh, tanh)
    - Logarithmic functions (log, log10, log2, ln)
    - Advanced functions (sqrt, exp, abs, factorial)
    - Statistical functions (mean, median, mode, variance, stdev)
    - Mathematical constants (pi, e)
    - Calculus operations (derivative, integral, limit)
    - Algebraic operations (simplify, expand, factor, solve)

Expression Processing:
    - Automatic syntax correction and normalization
    - Mathematical notation conversion (√, °, π, !)
    - Implicit multiplication insertion (2x → 2*x)
    - Power operator conversion (^ ↔ **)
    - Function name standardization

Dependencies:
    - ast: Abstract syntax tree parsing for security
    - math: Standard mathematical functions
    - random: Random number generation
    - re: Regular expression pattern matching
    - utils.math_utils: Advanced mathematical operations
"""

import ast
import math
import random
import re


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
    ALLOWED_NODES = {
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Name,
        ast.BinOp,
        ast.UnaryOp,
        ast.Expression,
        ast.Call
    }
    ALLOWED_FUNCTIONS = {
        'sin',
        'cos',
        'tan', 
        'sqrt',
        'log',
        'log10',
        'log2',
        'factorial',
        'asin',
        'acos',
        'atan',
        'sinh',
        'cosh',
        'tanh',
        'exp',
        'abs',
        'pi',
        'e',
        'pow',
        'det',
        'bin',
        'arrangements',
        'permutations',
        'combinations',
        'round',
        'ceil',
        'floor',
        'trunc',
        'max',
        'min',
        'sum',
        'limit',
        'derive',
        'integrate',
        'simplify',
        'expand',
        'factor',
        'solve',
        'gcd',
        'lcm',
        'mean',
        'median',
        'mode',
        'stdev',
        'variance',
        'random',
        'randint'
    }

    def _is_allowed_node_type(self, node):
        """
        Check if the AST node type is in the allowed nodes whitelist.
        
        Args:
            node: AST node to validate
            
        Returns:
            bool: True if node type is allowed, False otherwise
        """
        return isinstance(node, self.ALLOWED_NODES)

    def visit(self, node):
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

    def visit_Call(self, node):
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

    def visit_Import(self, node):
        raise ValueError("Import statements are not allowed")
    
    def visit_ImportFrom(self, node):
        raise ValueError("Import statements are not allowed")

    def visit_Lambda(self, node):
        raise ValueError("Lambda expressions are not allowed")
    
    def visit_Attribute(self, node):
        raise ValueError("Attribute access is not allowed")

    def visit_Subscript(self, node):
        raise ValueError("Subscripting is not allowed")

    def visit_Assign(self, node):
        raise ValueError("Assignment is not allowed")

    def visit_AugAssign(self, node):
        raise ValueError("Augmented assignment is not allowed")

    def visit_Exec(self, node):
        raise ValueError("Exec statements are not allowed")

    def visit_Global(self, node):
        raise ValueError("Global statements are not allowed")

    def visit_Nonlocal(self, node):
        raise ValueError("Nonlocal statements are not allowed")
    
    @staticmethod
    def validate_expression_tree(expression):
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
            tree = ast.parse(expression, mode='eval')
            validator = ExpressionValidator()
            validator.visit(tree)

        except SyntaxError as e:
            ExpressionValidator._handle_syntax_error(expression, e)
        except Exception as e:
            ExpressionValidator._handle_validation_error(expression, e)

    @staticmethod
    def _handle_syntax_error(expression, error):
        """Handle syntax errors during expression validation"""
        print(f"Syntax error in expression: {expression}")
        raise ValueError(f"Syntax error in expression '{expression}': {str(error)}") from error

    @staticmethod
    def _handle_validation_error(expression, error):
        """Handle general validation errors during expression validation"""
        print(f"Invalid expression: {expression}")
        raise ValueError(f"Error validating expression '{expression}': {str(error)}") from error

    @staticmethod
    def evaluate_expression(expression, x=0):
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
        tree = ast.parse(expression, mode='eval')
        # Evaluate the expression using the abstract syntax tree and the variables dictionary
        result = eval(compile(tree, '<string>', mode='eval'), variables_and_functions)
        return result

    @staticmethod
    def _get_variables_and_functions(x):
        """Create a dictionary with variables and functions for expression evaluation"""
        from utils.math_utils import MathUtils
        return {
            'x': x,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': MathUtils.sqrt, # Square root function
            'log': math.log,   # Natural logarithm (base e)
            'log10': math.log10,  # Logarithm base 10
            'log2': math.log2,  # Logarithm base 2
            'factorial': math.factorial,  # Factorial function
            'asin': math.asin,  # Arcsine function
            'acos': math.acos,  # Arccosine function
            'atan': math.atan,  # Arctangent function
            'sinh': math.sinh, # Hyperbolic sine function
            'cosh': math.cosh, # Hyperbolic cosine function
            'tanh': math.tanh, # Hyperbolic tangent function
            'exp': math.exp, # Exponential function
            'abs': abs, # Absolute value function
            'pi': math.pi, # The constant pi
            'e': math.e, # The constant e
            'pow': MathUtils.pow, # Power function
            'bin': bin, # Binary representation of an integer
            'det': MathUtils.det, # Determinant of a matrix
            'arrangements': MathUtils.arrangements, # Arrangements aka permutations nPk
            'permutations': MathUtils.permutations, # Permutations
            'combinations': MathUtils.combinations, # Combinations
            'limit': MathUtils.limit, # Limit of a function
            'derive': MathUtils.derivative, # Derivative of a function
            'integrate': MathUtils.integral, # Indefinite integral of a function
            'simplify': MathUtils.simplify, # Simplify an expression
            'expand': MathUtils.expand, # Expand an expression
            'factor': MathUtils.factor, # Factor an expression
            'solve': MathUtils.solve, # Solve an equation
            'random': MathUtils.random, # Generate a random number
            'round': MathUtils.round, # Round a number
            'gcd': MathUtils.gcd, # Greatest common divisor
            'lcm': MathUtils.lcm, # Least common multiple
            'mean': MathUtils.mean, # Mean of a list of numbers
            'median': MathUtils.median, # Median of a list of numbers
            'mode': MathUtils.mode, # Mode of a list of numbers
            'stdev': MathUtils.stdev, # Standard deviation of a list of numbers
            'variance': MathUtils.variance, # Variance of a list of numbers
            'ceil': math.ceil, # Round up to the nearest integer
            'floor': math.floor, # Round down to the nearest integer
            'trunc': math.trunc, # Truncate to an integer
            'max': max, # Maximum of a list of numbers
            'min': min, # Minimum of a list of numbers
            'sum': sum, # Sum of a list of numbers
            'randint': lambda a, b: random.randint(a, b) # Random integer between a and b
        }

    @staticmethod
    def fix_math_expression(expression, python_compatible=False):
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
        expression = ExpressionValidator._convert_degrees(expression)
        expression = ExpressionValidator._handle_special_symbols(expression, python_compatible)
        expression = ExpressionValidator._replace_function_names(expression)
        expression = ExpressionValidator._handle_power_and_imaginary(expression, python_compatible)
        expression = ExpressionValidator._insert_multiplication_operators(expression, python_compatible)
        return expression

    @staticmethod
    def _convert_degrees(expression):
        """Convert degree symbols and text to radians"""
        expression = expression.replace('°', ' deg')
        expression = expression.replace('degrees', ' deg')
        expression = expression.replace('degree', ' deg')
        expression = re.sub(r'(\d+)\s*deg', lambda match: str(float(match.group(1)) * math.pi / 180), expression)
        return expression

    @staticmethod
    def _handle_special_symbols(expression, python_compatible):
        """Handle square roots, absolute values, and factorials"""
        # Handle square roots
        expression = re.sub(r'√\((.*?)\)', r'sqrt(\1)', expression)
        expression = re.sub(r'√([0-9a-zA-Z_]+)', r'sqrt(\1)', expression)

        # Replace | | with the Python equivalent if needed
        if python_compatible:
            expression = re.sub(r'\|(.*?)\|', r'abs(\1)', expression)

        # Handle factorials with balanced operand extraction
        expression = ExpressionValidator._replace_factorials(expression)
        return expression

    @staticmethod
    def _replace_factorials(expression):
        """Replace factorial shorthand (n!) with factorial() calls using balanced parsing."""
        if '!' not in expression:
            return expression

        def is_token_char(char):
            return char.isalnum() or char in ['_', '.']

        matching_pairs = {')': '(', ']': '[', '}': '{'}

        index = expression.find('!')
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
            expression = expression[:start] + replacement + expression[index + 1:]
            index = expression.find('!', start + len(replacement))

        return expression

    @staticmethod
    def _replace_function_names(expression):
        """Replace common mathematical function names with their Python equivalents"""
        replacements = ExpressionValidator._get_function_replacements()
        for old, new in replacements.items():
            expression = expression.replace(old, new)
        return expression

    @staticmethod
    def _get_function_replacements():
        """Get a dictionary of function name replacements"""
        return {
            'π': 'pi',  # Using the variable from the dictionary
            'ln': 'log',  # Python's math.log is ln by default
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
    def _handle_power_and_imaginary(expression, python_compatible):
        """Handle power operators and imaginary numbers based on compatibility mode"""
        # Replace the power symbol with '**' if specified
        if python_compatible:
            expression = expression.replace('^', '**')
        else:
            expression = expression.replace('**', '^') 

        # Replace 'i' with 'j' only in contexts likely to represent the imaginary unit
        imaginary_unit = 'j' if python_compatible else 'i'
        opposite_unit = 'i' if python_compatible else 'j'

        # Assuming it's used in the form of numbers like '2i' or standalone 'i'
        expression = re.sub(rf'(?<=\d){opposite_unit}\b', f'{imaginary_unit}', expression)  # For numbers like '2i'
        expression = re.sub(rf'\b{opposite_unit}\b', f'{imaginary_unit}', expression)  # For standalone 'i'
        
        return expression

    @staticmethod
    def _insert_multiplication_operators(expression, python_compatible):
        """Insert multiplication operators where implicit multiplication is used"""
        imaginary_unit = 'j' if python_compatible else 'i'
        
        # Step 1: Protect "log" followed by any number from being altered
        expression = re.sub(r'log(\d+)', r'log[\1]', expression)
        
        # Step 2: Insert '*' between a number and a variable, function name, or parenthesis, 
        # excluding 'i' or 'j' immediately after a number
        expression = re.sub(rf'(\d)(?!{imaginary_unit})([a-zA-Z_\(])', r'\1*\2', expression)
        
        # Step 3: Revert "log" followed by any number back to its original form
        expression = re.sub(r'log\[(\d+)\]', r'log\1', expression)
        
        return expression

    @staticmethod
    def _parse_with_mathjs(function_string):
        """Parse a function string using mathjs (slower but more powerful)"""
        from utils.math_utils import MathUtils
        return lambda x: MathUtils.evaluate(function_string, {'x': x})

    @staticmethod
    def _parse_with_python(function_string):
        """Parse a function string using Python's built-in evaluation (faster)"""
        function_string = ExpressionValidator.fix_math_expression(function_string, python_compatible=True)
        ExpressionValidator.validate_expression_tree(function_string)
        return lambda x: ExpressionValidator.evaluate_expression(function_string, x)

    @staticmethod
    def parse_function_string(function_string, use_mathjs=False):
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