from constants import default_color, default_point_size
from drawables.drawable import Drawable
from expression_validator import ExpressionValidator


class Function(Drawable):
    def __init__(self, function_string, canvas, name, step=default_point_size, color=default_color, left_bound=None, right_bound=None, vertical_asymptotes=None, horizontal_asymptotes=None, point_discontinuities=None):
        self.step = step
        self.left_bound = left_bound
        self.right_bound = right_bound
        try:
            self.function_string = ExpressionValidator.fix_math_expression(function_string)
            self.function = ExpressionValidator.parse_function_string(function_string)
            # Set asymptotes and discontinuities if provided, otherwise calculate them
            if vertical_asymptotes is not None:
                self.vertical_asymptotes = vertical_asymptotes
            if horizontal_asymptotes is not None:
                self.horizontal_asymptotes = horizontal_asymptotes
            if point_discontinuities is not None:
                self.point_discontinuities = point_discontinuities
            if vertical_asymptotes is None and horizontal_asymptotes is None and point_discontinuities is None:
                self._calculate_asymptotes_and_discontinuities()
        except Exception as e:
            raise ValueError(f"Failed to parse function string '{function_string}': {str(e)}")
        super().__init__(name=name, color=color, canvas=canvas)

    @Drawable.canvas.setter
    def canvas(self, value):
        self._canvas = value

    @canvas.getter
    def canvas(self):
        return self._canvas
    
    def get_class_name(self):
        return 'Function'

    def get_state(self):
        function_string = self.function_string
        state = {
            "name": self.name, 
            "args": {
                "function_string": function_string,
                "left_bound": self.left_bound, 
                "right_bound": self.right_bound
            }
        }
        
        # Only include asymptotes and discontinuities lists that have values
        if hasattr(self, 'vertical_asymptotes') and self.vertical_asymptotes:
            state["args"]["vertical_asymptotes"] = self.vertical_asymptotes
        if hasattr(self, 'horizontal_asymptotes') and self.horizontal_asymptotes:
            state["args"]["horizontal_asymptotes"] = self.horizontal_asymptotes
        if hasattr(self, 'point_discontinuities') and self.point_discontinuities:
            state["args"]["point_discontinuities"] = self.point_discontinuities
            
        return state

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        new_function = Function(
            function_string=self.function_string,
            canvas=self.canvas,
            name=self.name,
            step=self.step,
            color=self.color,
            left_bound=self.left_bound,
            right_bound=self.right_bound,
            vertical_asymptotes=self.vertical_asymptotes.copy() if hasattr(self, 'vertical_asymptotes') else None,
            horizontal_asymptotes=self.horizontal_asymptotes.copy() if hasattr(self, 'horizontal_asymptotes') else None,
            point_discontinuities=self.point_discontinuities.copy() if hasattr(self, 'point_discontinuities') else None
        )
        memo[id(self)] = new_function
        return new_function

    def translate(self, x_offset, y_offset):
        if x_offset == 0 and y_offset == 0:
            return

        # Translate bounds if they exist
        if self.left_bound is not None:
            self.left_bound += x_offset
        if self.right_bound is not None:
            self.right_bound += x_offset

        try:
            # First handle horizontal translation by replacing x with (x - x_offset)
            if x_offset != 0:
                import re
                # Use all allowed functions from ExpressionValidator
                protected_funcs = sorted(ExpressionValidator.ALLOWED_FUNCTIONS, key=len, reverse=True)
                
                # Create a regex pattern that matches standalone x while protecting function names
                func_pattern = '|'.join(map(re.escape, protected_funcs))
                # Use word boundaries to match standalone 'x'
                pattern = rf'\b(x)\b|({func_pattern})'
                
                def replace_match(match):
                    if match.group(1):  # If it's a standalone 'x'
                        return f'(x - {x_offset})'
                    elif match.group(2):  # If it's a function name
                        return match.group(2)  # Return the function name unchanged
                    return match.group(0)
                    
                new_function_string = re.sub(pattern, replace_match, self.function_string)
            else:
                new_function_string = self.function_string

            # Then handle vertical translation by adding y_offset
            if y_offset != 0:
                new_function_string = f"({new_function_string}) + {y_offset}"

            # Update function string and parse new function
            self.function_string = ExpressionValidator.fix_math_expression(new_function_string)
            self.function = ExpressionValidator.parse_function_string(new_function_string)
            
        except Exception as e:
            print(f"Warning: Could not translate function: {str(e)}")
            # If translation fails, revert bounds
            if self.left_bound is not None:
                self.left_bound -= x_offset
            if self.right_bound is not None:
                self.right_bound -= x_offset

    def rotate(self, angle):
        pass 

    def _calculate_asymptotes_and_discontinuities(self):
        """Calculate vertical and horizontal asymptotes and point discontinuities of the function"""
        from utils.math_utils import MathUtils
        
        # Calculate asymptotes and discontinuities using MathUtil
        self.vertical_asymptotes, self.horizontal_asymptotes, self.point_discontinuities = MathUtils.calculate_asymptotes_and_discontinuities(
            self.function_string,
            self.left_bound,
            self.right_bound
        )

    def has_point_discontinuity_between_x(self, x1, x2):
        """Check if there is a point discontinuity between x1 and x2"""
        return (hasattr(self, 'point_discontinuities') and any(x1 < x < x2 for x in self.point_discontinuities))
    
    def has_vertical_asymptote_between_x(self, x1, x2):
        """Check if there is a vertical asymptote between x1 and x2"""
        return (hasattr(self, 'vertical_asymptotes') and any(x1 <= x < x2 for x in self.vertical_asymptotes))

    def get_vertical_asymptote_between_x(self, x1, x2):
        """Get the x value of a vertical asymptote between x1 and x2, if any exists"""
        if hasattr(self, 'vertical_asymptotes'):
            for x in self.vertical_asymptotes:
                if x1 <= x < x2:
                    return x
        return None
