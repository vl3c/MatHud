class MockCanvas:
    def __init__(self, width, height, draw_enabled=True):
        self.width = width
        self.height = height
        self.points = []
        self.segments = []
        self.circles = []
        self.rectangles = []
        self.triangles = []
        self.ellipses = []
        self.functions = []
        self.vectors = []
        self.computations = []

    def get_drawables(self):
        """Get all drawable objects on the canvas."""
        return (
            self.points +
            self.segments +
            self.circles +
            self.rectangles +
            self.triangles +
            self.ellipses +
            self.functions +
            self.vectors
        )

    def get_drawables_by_class_name(self, class_name):
        """Get drawables of a specific class."""
        class_map = {
            "Point": self.points,
            "Segment": self.segments,
            "Circle": self.circles,
            "Rectangle": self.rectangles,
            "Triangle": self.triangles,
            "Ellipse": self.ellipses,
            "Function": self.functions,
            "Vector": self.vectors
        }
        return class_map.get(class_name, [])

    def get_canvas_state(self):
        """Return the current state of the canvas."""
        return {
            "Points": self.points,
            "Segments": self.segments,
            "Circles": self.circles,
            "Rectangles": self.rectangles,
            "Triangles": self.triangles,
            "Ellipses": self.ellipses,
            "Functions": self.functions,
            "Vectors": self.vectors,
            "computations": self.computations
        }

    def clear(self):
        """Clear all objects from the canvas."""
        self.points = []
        self.segments = []
        self.circles = []
        self.rectangles = []
        self.triangles = []
        self.ellipses = []
        self.functions = []
        self.vectors = []
        self.computations = []

    def create_point(self, x, y, name=""):
        """Create a point on the canvas."""
        point = {"x": x, "y": y, "name": name}
        self.points.append(point)
        return point

    def create_segment(self, x1, y1, x2, y2, name=""):
        """Create a segment on the canvas."""
        segment = {
            "point1": {"x": x1, "y": y1},
            "point2": {"x": x2, "y": y2},
            "name": name
        }
        self.segments.append(segment)
        return segment

    def create_circle(self, x, y, radius, name=""):
        """Create a circle on the canvas."""
        circle = {
            "center": {"x": x, "y": y},
            "radius": radius,
            "name": name
        }
        self.circles.append(circle)
        return circle

    def create_rectangle(self, x1, y1, x2, y2, name=""):
        """Create a rectangle on the canvas."""
        rectangle = {
            "point1": {"x": x1, "y": y1},
            "point3": {"x": x2, "y": y2},
            "name": name
        }
        self.rectangles.append(rectangle)
        return rectangle

    def create_triangle(self, x1, y1, x2, y2, x3, y3, name=""):
        """Create a triangle on the canvas."""
        triangle = {
            "point1": {"x": x1, "y": y1},
            "point2": {"x": x2, "y": y2},
            "point3": {"x": x3, "y": y3},
            "name": name
        }
        self.triangles.append(triangle)
        return triangle

    def create_ellipse(self, x, y, radius_x, radius_y, rotation_angle=0, name=""):
        """Create an ellipse on the canvas."""
        ellipse = {
            "center": {"x": x, "y": y},
            "radius_x": radius_x,
            "radius_y": radius_y,
            "rotation_angle": rotation_angle,
            "name": name
        }
        self.ellipses.append(ellipse)
        return ellipse

    def draw_function(self, function_string, name="", left_bound=None, right_bound=None):
        """Add a function to the canvas."""
        function = {
            "function_string": function_string,
            "name": name,
            "left_bound": left_bound,
            "right_bound": right_bound
        }
        self.functions.append(function)
        return function

    def create_vector(self, x1, y1, x2, y2, name=""):
        """Create a vector on the canvas."""
        vector = {
            "origin": {"x": x1, "y": y1},
            "tip": {"x": x2, "y": y2},
            "name": name
        }
        self.vectors.append(vector)
        return vector

    def add_computation(self, expression, result):
        """Add a computation to the canvas."""
        computation = {
            "expression": expression,
            "result": result
        }
        self.computations.append(computation)
        return computation 