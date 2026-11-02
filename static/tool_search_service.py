"""
MatHud Tool Search Service

Provides tool discovery via fast local keyword/category matching (default)
or AI-powered semantic matching.  The search mode is selected by the
``TOOL_SEARCH_MODE`` environment variable:

* ``local``  -- keyword + category index, no API call
* ``api``   -- original GPT-based semantic search
* ``hybrid`` (default) -- local first, fall back to API when confidence is low

Dependencies:
    - static.ai_model: AI model configuration
    - static.functions_definitions: Tool definitions to search through
    - openai: OpenAI API client (only needed for ``api`` / ``hybrid`` modes)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

from openai import OpenAI

from static.ai_model import PROVIDER_LOCAL_AGENT, PROVIDER_OPENAI, AIModel
from static.env_config import get_api_key
from static.functions_definitions import FUNCTIONS, FunctionDefinition

_logger = logging.getLogger("mathud")

# Tools to exclude from search results (meta-tools that shouldn't be recommended)
EXCLUDED_FROM_SEARCH = frozenset({"search_tools"})

# Model for API tool search when no model is given, and in place of models that
# cannot run it: the search request sends reasoning_effort "none", which
# GPT-6 Astra rejects.
API_SEARCH_FALLBACK_MODEL = "gpt-6-luna"
_MODELS_REJECTING_NO_REASONING = frozenset({"gpt-6-astra"})

# Destructive tools (delete_*/clear_*/reset_*) rank below every other match unless
# the query contains one of these words.
DESTRUCTIVE_TOOL_PREFIXES = ("delete_", "clear_", "reset_")
DESTRUCTIVE_QUERY_WORDS = frozenset(
    {"delete", "remove", "erase", "clear", "reset", "wipe", "rid", "destroy", "clean", "fresh"}
)

# Inflected destructive verbs ("deletes", "deleting", "erased", "clearing", ...) are
# folded to their base form before scoring so they count like the plain verb.
_DESTRUCTIVE_INFLECTION_RE = re.compile(
    r"^(?:(delet|remov|eras|wip)(?:es|ed|ing)"
    r"|(clear|reset|destroy)(?:s|ed|ing|ting)"
    r"|(drop)(?:s|ped|ping)?)$"
)
# "drop" means delete, except in "drop a perpendicular/altitude".
_DROP_CONSTRUCTION_WORDS = frozenset({"perpendicular", "altitude", "height", "normal", "foot"})

# run_tests is only offered when the query mentions tests.
TEST_QUERY_WORDS = frozenset({"test", "tests", "testing"})

# A function applied to a single number, e.g. "f(3)" or "g(-2.5)".
_FUNCTION_AT_POINT_RE = re.compile(r"\b[a-z]\w*\(\s*-?\d+(?:\.\d+)?\s*\)")
# A one-letter function, a defined one rather than sqrt(12), applied to a constant:
# numbers, operators, parentheses, pi, e and sqrt, e.g. "f(3)", "f(1/3)", "f(pi/4)",
# "f(sqrt(2))", but not "f(x)". Only read in exact queries: elsewhere "a(1/2)x" or
# "part b(1/2)" is rarely a function value. The bounded repeats keep a long query linear.
_NAMED_FUNCTION_AT_POINT_RE = re.compile(
    r"\b[a-z]\((?=[^)]{0,40}(?:\d|\bpi\b|\be\b))(?:[\d.+\-*/^\s()]|\bpi\b|\be\b|\bsqrt\b){1,40}\)"
)
# Measurements a tool takes (calculate_area): an "exact" one still goes to that tool first.
# No tool measures a distance or a length, and create_angle draws an angle in decimal
# degrees, so an exact distance, length or angle goes to simplify.
_MEASUREMENT_WORDS = frozenset({"area"})

# "graph" means a graph of vertices and edges (generate_graph) rather than the plot of
# a function when the query names that structure (_GRAPH_STRUCTURE_WORDS). Two weaker
# signals count only when nothing ties "graph" to another tool (_OTHER_GRAPH_WORDS or
# function notation): vertices and edges named in geometric terms ("points ...
# segments", "connect"), and a bare request such as "create a graph" or "build graph G1".
# Graph-analysis words only cancel the function-plot penalty; analyze_graph has its own boosts.
_GRAPH_STRUCTURE_WORDS = frozenset(
    {"vertex", "vertices", "edge", "edges", "node", "nodes", "directed", "undirected", "weighted", "network"}
)
_GRAPH_ANALYSIS_WORDS = frozenset(
    {"shortest", "path", "bfs", "dfs", "mst", "topological", "spanning", "adjacency", "degree", "dag", "tree"}
)
# Words that mark a correlation question as graph theory. The unambiguous ones count alone;
# the everyday ones ("tree height", "path length", "triangle vertices", "a graph" meaning a
# plot) only in pairs, and none of them when the query gives number lists to correlate.
_GRAPH_THEORY_ONLY_WORDS = frozenset(
    {"bfs", "dfs", "mst", "topological", "spanning", "adjacency", "dag", "dags", "undirected"}
)
_GRAPH_THEORY_AMBIGUOUS_WORDS = frozenset(
    {"graph", "graphs", "edge", "edges", "node", "nodes", "network", "networks", "tree", "trees"}
    | {"vertex", "vertices", "path", "paths", "shortest", "weighted", "degree", "directed"}
)
_NUMBER_LIST_RE = re.compile(r"\[\s*[-+.]?\d")
_GRAPH_VERTEX_WORDS = frozenset({"point", "points"})
_GRAPH_EDGE_WORDS = frozenset({"segment", "segments", "connect", "connected", "connecting", "connects"})
# Words that make "graph" another tool's object: plots and charts, curves and functions,
# shapes, calculus, statistics and the coordinate system.
_OTHER_GRAPH_WORDS = frozenset(
    {"bar", "bars", "chart", "histogram", "distribution", "normal", "gaussian", "bell", "probability"}
    | {"binomial", "poisson"}
    | {"scatter", "scatterplot", "box", "boxplot", "whisker", "whiskers", "bins"}
    | {"data", "regression", "line", "lines", "linear", "trend"}
    | {"parametric", "polar", "cartesian", "coordinate", "coordinates", "axes", "axis", "grid"}
    | {"piecewise", "inequality", "inequalities", "function", "functions", "curve", "curves", "equation"}
    | {"sequence", "sequences", "cobweb", "orbit", "recursive", "recursion", "recurrence", "fibonacci"}
    | {"equations", "plot", "plotted", "parabola", "polynomial", "quadratic", "squared", "cubed"}
    | {"sin", "cos", "tan", "exp", "ln", "log", "sqrt", "abs", "absolute", "value"}
    | {"circle", "circles", "ellipse", "ellipses", "triangle", "polygon", "arc", "angle"}
    | {"derivative", "integral", "tangent", "slope", "area"}
)
# Sequence and iteration words (_boost_sequence_tools): terms of a sequence go to plot_sequence,
# an iteration's orbit to plot_cobweb.
_SEQUENCE_WORDS = frozenset(
    {"sequence", "sequences", "recursive", "recursion", "recurrence", "recursively", "fibonacci", "recursions"}
)
_COBWEB_WORDS = frozenset({"cobweb", "cobwebs", "orbit", "orbits", "staircase"})
_ITERATION_WORDS = frozenset({"iterate", "iterates", "iteration", "iterations", "iterating", "iterated"})
# "5 iterations of Newton's method": a root finder's iterations, not an orbit to draw
_ROOT_METHOD_WORDS = frozenset({"newton", "newtons", "raphson", "bisection", "secant"})
# A sum or series of terms is a computation, not a plot of the terms (nor an orbit)
_SEQUENCE_SUM_WORDS = frozenset({"sum", "sums", "summation", "series", "sigma"})
# A limit or convergence test of a sequence is a computation; an iteration's convergence is a cobweb
_SEQUENCE_LIMIT_WORDS = frozenset(
    {"limit", "limits", "converge", "convergent", "converges", "diverge", "divergent", "diverges"}
    | {"test", "nth", "formula"}
)
_STABILITY_WORDS = frozenset({"stable", "unstable", "stability", "attracting", "repelling", "attract", "repel"})
# Geometric objects and transformations ("a sequence of circles", "rotate orbit by 45 degrees",
# where orbit names an ellipse)
_SEQUENCE_OBJECT_WORDS = frozenset(
    {"point", "points", "circle", "circles", "segment", "segments", "triangle", "triangles", "ellipse"}
    | {"polygon", "polygons", "square", "squares", "rectangle", "rectangles", "vector", "vectors", "angle"}
    | {"angles", "label", "labels", "arc", "arcs", "line", "lines", "ellipses", "named"}
    | {"rotate", "reflect", "translate", "rotations", "transformations", "transformation", "degrees"}
)
# Sequences of actions, steps, transformations or graph traversals are not number sequences
_NOT_NUMBER_SEQUENCE_WORDS = _SEQUENCE_OBJECT_WORDS | frozenset(
    {"undo", "redo", "actions", "action", "steps"}
    | {"traversal", "topological", "vertices", "vertex", "nodes", "edges", "bfs", "dfs", "path", "dag"}
)
_SEQUENCE_REQUEST_EXCLUDED_WORDS = frozenset(
    {"delete", "remove", "erase", "clear", "zoom", "save", "load", "workspace", "workspaces"}
)
# "the first 10 terms of 2^n": terms to list and plot
_FIRST_TERMS_RE = re.compile(r"\bfirst\s+\d+\s+terms\b")
# "the 10th term of the arithmetic sequence": a value to compute
_ORDINAL_TERM_RE = re.compile(r"\b\d+(?:st|nd|rd|th)\s+term")
# "iterate over the vertices": a graph traversal, not an iteration of a function
_NOT_ITERATION_WORDS = (_SEQUENCE_OBJECT_WORDS - {"point", "points"}) | frozenset(
    {"vertices", "vertex", "nodes", "edges", "graph", "tree", "dag", "bfs", "dfs", "traversal", "list", "workspaces"}
)
# u(n), a(n), u(n-1), x(n+1), a_n, u_{n-1}: a term of a sequence
_SEQUENCE_TERM_RE = re.compile(r"(?<![a-z])[a-z]\s*\(\s*n\s*(?:[-+]\s*\d\s*)?\)|\b[a-z]_\{?n\b")
# u(n-1), a(n-2): a previous term of a recursion
_PREVIOUS_TERM_RE = re.compile(r"(?<![a-z])[a-z]\s*\(\s*n\s*-\s*[12]\s*\)")
# x(n+1) = ..., x_{n+1}: a first-order iteration, drawn as a cobweb
_NEXT_TERM_RE = re.compile(r"(?<![a-z])[a-z]\s*\(\s*n\s*\+\s*1\s*\)|\b[a-z]_[({]?n\s*\+\s*1")
_PLOT_NOUN_WORDS = frozenset({"diagram", "diagrams", "plot", "plots", "chart", "picture"})
# "plot the convergent sequence 1/n": a plot request, whatever else it asks about the sequence
_SEQUENCE_PLOT_VERBS = frozenset({"plot", "plots", "plotting", "graph", "draw", "sketch", "visualize", "visualise"})
# Data-plot words: histograms, box plots and scatter plots; fit words send a scatter request
# to fit_regression.
_HISTOGRAM_WORDS = frozenset({"histogram", "histograms", "bin", "bins", "binned", "binning"})
_BOX_PLOT_WORDS = frozenset({"boxplot", "boxplots", "whisker", "whiskers"})
_SCATTER_WORDS = frozenset({"scatter", "scatterplot", "scatterplots"})
_FIT_WORDS = frozenset({"fit", "fitting", "fitted", "regression", "trend", "best"})
# Words that ask for descriptive statistics ("the standard deviation of the data and a histogram").
_DESCRIPTIVE_STATISTICS_WORDS = frozenset(
    {"descriptive", "stats", "statistics", "mean", "median", "average", "stdev", "std", "sd", "deviation"}
    | {"quartile", "quartiles", "iqr", "interquartile"}
)
# Function notation such as "f(x)", "y = ...", "x^2", "x > 3", "1/x".
_FUNCTION_NOTATION_RE = re.compile(r"\b[a-z]\(\s*[a-z]\s*\)|[=^<>/+*]")
# "create a graph", "build graph G1", "make a new graph called net": nothing but the request.
_BARE_GRAPH_REQUEST_RE = re.compile(
    r"^\s*(?:please\s+)?(?:create|build|make|generate|construct)\s+(?:(?:a|an|the|new)\s+)*graph"
    r"(?:\s+(?:named|called))?(?:\s+[a-z]\w*)?\s*[.!]?\s*$"
)
# Graph structure written out between point-like names (one capital letter, optional digits
# and primes) in the query as typed: an edge "A->B" between two different vertices (not
# "x -> 2x" or "f: R -> R"), or a list of two or more "A-B" pairs.
# Named graph families ("bipartite graph", "complete graph", "star graph") count too.
_VERTEX_NAME = r"[A-Z]\d*'*"
_EDGE_LIST_RE = re.compile(
    rf"\b({_VERTEX_NAME})\s*->\s*(?!\1(?![\w']))(?:{_VERTEX_NAME})(?![\w'])"
    rf"|\b{_VERTEX_NAME}\s*-\s*{_VERTEX_NAME}\s*,\s*{_VERTEX_NAME}\s*-\s*{_VERTEX_NAME}(?![\w'])"
)
_GRAPH_FAMILY_RE = re.compile(r"\b(?:bipartite|complete|star|cycle|wheel|petersen|tree)\s+graphs?\b")
# Function names and notation that make "graph" a function plot (draw_function).
# "y = ..." or "f(x) = ..." is a function even when the query also mentions points
# ("graph y = sin(x) and mark the points where it crosses zero").
_FUNCTION_DEFINITION_RE = re.compile(r"(?:^|[^\w(])(?:y|[a-z]\(\s*[a-z]\s*\))\s*=")
# Distribution words for compute_distribution and plot_distribution (_boost_distribution_tools).
_DISTRIBUTION_NAME_WORDS = frozenset(
    {"binomial", "poisson", "chi", "chisquared", "student", "students", "gaussian"}
    | {"normalcdf", "normalpdf", "invnorm", "binompdf", "binomcdf", "poissonpdf", "poissoncdf"}
    | {"tcdf", "tpdf", "invt", "chicdf", "chi2cdf", "fcdf", "geometpdf", "geometcdf", "invbinom"}
)
_AMBIGUOUS_DISTRIBUTION_WORDS = frozenset({"normal", "exponential", "uniform", "geometric"})
_DISTRIBUTION_CONTEXT_WORDS = frozenset(
    {"distribution", "distributions", "distributed", "random", "variable", "probability", "probabilities"}
    | {"pdf", "cdf", "pmf", "density", "quantile", "quantiles", "freedom", "trials", "success", "successes"}
)
_PROBABILITY_WORDS = frozenset(
    {"probability", "probabilities", "pdf", "cdf", "pmf", "quantile", "quantiles", "critical", "invnorm"}
    | {"normalcdf", "normalpdf", "binompdf", "binomcdf", "poissonpdf", "poissoncdf", "tcdf", "tpdf", "invt"}
    | {"chicdf", "chi2cdf", "fcdf", "geometpdf", "geometcdf", "invbinom", "likely", "chance", "odds"}
)
_PICTURE_WORDS = frozenset(
    {"plot", "draw", "graph", "sketch", "shade", "shaded", "shading", "show", "display", "visualize"}
    | {"visualise", "picture", "curve", "bars", "histogram", "create", "make", "put", "canvas"}
)
# "the normal to f at x = 2", "the normal of the parabola": a normal line, not a normal distribution.
_NORMAL_LINE_RE = re.compile(
    r"\bnormals?\s+(?:to\b|at\b|(?:of|for)\s+(?:[a-z]\b|the\s+(?:graph|curve|parabola|circle|ellipse|function)))"
)
_FUNCTION_NAME_WORDS = frozenset({"sin", "cos", "exp", "ln", "log", "squared", "cubed"})
_FUNCTION_EQUATION_RE = re.compile(r"\b[a-z]\(\s*[a-z]\s*\)|[=^]")
_PIECEWISE_RE = re.compile(r"for\s+x\s*[<>]|when\s+x\s*[<>]|x\s*>=|x\s*<=")
# An inequality sign: <, >, <=, >=, ≤, ≥, but not the arrow of "x->2"
_INEQUALITY_SIGN_RE = re.compile(r"[<≤≥]|(?<!-)>")
# Words that ask for the solution set of an inequality, and words that ask for something else,
# which an inequality only qualifies: drawing it, or another operation on x in a range ("the
# area under x^2 for 0 < x < 2", "the derivative of f(x) = x^2 when x > 0")
_INEQUALITY_SOLVE_WORDS = frozenset({"solve", "find", "which", "when"})
_INEQUALITY_OTHER_WORDS = frozenset(
    {"shade", "shaded", "shading", "graph", "plot", "draw", "sketch", "region", "piecewise", "rules"}
    | {"area", "integral", "integrate", "integration", "derivative", "derive", "differentiate"}
    | {"maximum", "minimum", "max", "min", "extrema", "extremum", "limit", "probability", "probabilities"}
    | {"normal", "distribution", "distributions", "distance", "distances"}
    | {"mean", "median", "mode", "average", "deviation", "variance", "statistics", "regression"}
    | {"angle", "angles", "length", "lengths", "segment", "segments", "perimeter", "triangle", "triangles"}
    | {"point", "points", "polygon", "polygons"}
)
# "P(X > 2)": a probability, whose condition is not an inequality to solve
_PROBABILITY_CALL_RE = re.compile(r"(?<![a-z0-9_])p\s*\(")
# "AB > BC": a relation between two named objects (lengths), not an inequality in x
_NAMES_RELATION_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9]*\s*(?:<=|>=|<|>|≤|≥)\s*[A-Z][A-Z0-9]*(?![A-Za-z0-9_])")
# "when x < 0 what is |x|": a value asked under a condition on x
# Polar curves r(θ): their names, nouns that make "polar" a curve, the "r =" / "r(θ) =" / "r^2 ="
# that starts one, and words that make "r = ..., θ = ..." a point or a conversion instead
_POLAR_CURVE_NAMES = frozenset({"cardioid", "cardioids", "limacon", "limaçon", "lemniscate", "lemniscates"})
_POLAR_CURVE_NOUNS = frozenset(
    {"curve", "curves", "plot", "plots", "equation", "function", "functions", "rose", "spiral"}
)
_POLAR_CURVE_RE = re.compile(r"(?:^|[^a-z0-9_])r\s*(?:\(\s*(?:θ|theta)\s*\))?\s*(?:\^\s*2\s*)?=(?!=)")
# r = f(θ) written out: θ right after a number, an operator, "(" or a trig function, or alone
# ("r = 2sin(3θ)", "r = 1 + cos θ", "r = θ"); not "r = 5 at angle θ = 30°"
_POLAR_EQUATION_RE = re.compile(
    r"(?:^|[^a-z0-9_])r\s*(?:\(\s*(?:θ|theta)\s*\))?\s*=(?!=)\s*"
    r"(?:(?:θ|theta)\b|[^,;=]*?(?:[0-9*+\-/^(]|\b(?:sin|cos|tan|sec|csc|cot))\s*(?:θ|theta)\b)"
)
_POLAR_NOT_CURVE_WORDS = frozenset({"point", "points", "convert", "conversion", "cartesian", "rectangular"})
# Shapes whose radius or angle a query may give as r and θ
_POLAR_SHAPE_WORDS = frozenset(
    {"vector", "vectors", "arc", "arcs", "segment", "segments", "magnitude", "radius", "sector", "angle", "angles"}
)
# Editing an existing polar curve, and drawing one
_POLAR_EDIT_WORDS = frozenset(
    {"change", "update", "edit", "set", "recolor", "recolour", "extend", "range", "make", "color", "colour"}
    | {"red", "blue", "green", "orange", "purple", "black", "yellow", "pink", "gray", "grey", "brown"}
)
# Verbs that only edit: "update two cardioids", "change the polar plot"
_POLAR_UPDATE_VERBS = frozenset({"update", "change", "edit", "modify", "recolor", "recolour", "extend"})
_POLAR_DRAW_WORDS = frozenset({"draw", "plot", "graph", "sketch", "create", "add", "show", "want"})
# The article right before the curve word (a few adjectives may come between) says whether
# the curve is new: "make a red cardioid at the origin", "one more rose"; or already drawn:
# "make the cardioid a different color"
_POLAR_CURVE_WORD = r"(?:cardioids?|lima[cç]ons?|lemniscates?|roses?|spirals?|polar|curves?)\b"
# A count reads as new curves only right before the curve word ("two cardioids", "3 red roses"),
# not as a range end ("set theta_max to 6 for the lemniscate")
_NEW_POLAR_CURVE_RE = re.compile(
    r"\b(?:a|an|another|more|new|second)\s+(?:[\w-]+\s+){0,3}?" + _POLAR_CURVE_WORD
    + r"|\b(?:two|three|four|five|[0-9]+)\s+(?:(?!for\b|on\b|to\b|from\b|in\b|pi\b)[\w-]+\s+){0,2}?" + _POLAR_CURVE_WORD
)
_EXISTING_POLAR_CURVE_RE = re.compile(
    r"\b(?:the|this|that|these|those|all|both|my|existing)\s+(?:[\w-]+\s+){0,2}?" + _POLAR_CURVE_WORD
    + r"|\b\w+_polar\b"
)
# "the polar curve", "polar functions": a curve, even next to the polar grid
_POLAR_CURVE_PHRASE_RE = re.compile(r"\bpolar\s+(?:curves?|functions?|equations?|plots?)\b")
# x(t) = ... or x = ... t ...: a curve given by x and y in t
_PARAMETRIC_RE = re.compile(r"\bx\s*\(\s*t\s*\)|\bx\s*=[^,;]*\bt\b")
_PARAMETRIC_Y_RE = re.compile(r"\by\s*\(\s*t\s*\)|\by\s*=[^,;]*\bt\b")
_VALUE_QUESTION_WORDS = frozenset({"what", "whats"})


def _names_new_polar_curve(raw_lower: str) -> bool:
    """Whether the query asks for a new curve ("make a cardioid"), not one already drawn."""
    new = _NEW_POLAR_CURVE_RE.search(raw_lower)
    if new is None:
        return False
    existing = _EXISTING_POLAR_CURVE_RE.search(raw_lower)
    # "make the cardioid a red curve" names the drawn one first
    return existing is None or new.start() < existing.start()


# "f(x) = x for x < 0 ...", "the function is -x when x < 0": a function defined by cases
_DEFINES_FUNCTION_WORDS = frozenset({"function", "define", "defined"})
# Shapes, charts and other drawables a query can "graph" with an equation or coordinates
# ("graph the circle x^2 + y^2 = 4", "graph the point A = (2, 3)"): not a function plot.
# "line" is left out: "graph the line y = 2x + 1" is a function.
_NOT_A_FUNCTION_WORDS = frozenset(
    {"point", "points", "vector", "vectors", "segment", "segments", "ray", "triangle", "triangles"}
    | {"polygon", "polygons", "rectangle", "square", "quadrilateral", "pentagon", "hexagon"}
    | {"circle", "circles", "ellipse", "ellipses", "arc", "arcs", "angle", "angles", "label", "labels"}
    | {"bar", "bars", "chart", "histogram", "distribution", "scatter", "data", "regression"}
    | {"scatterplot", "box", "boxplot", "whisker", "whiskers", "bins"}
    | {"parametric", "polar", "edge", "edges", "node", "nodes"}
    | {"sequence", "sequences", "cobweb", "cobwebs", "orbit", "orbits", "recursion", "recurrence", "recursive"}
)
# "connect A and B", "join the midpoints": a segment. The verb needs points to join, and
# never boosts when the query is about graphs, functions or forming a shape.
_CONNECT_WORDS = frozenset({"connect", "connects", "connecting", "join", "joins", "joining"})
_CONNECT_OBJECT_WORDS = frozenset(
    {"point", "points", "midpoint", "midpoints", "endpoint", "endpoints", "segment", "segments", "line", "lines"}
)
_CONNECT_EXCLUDED_WORDS = frozenset(
    {"graph", "graphs", "network", "node", "nodes", "edge", "edges", "vertex", "vertices"}
    | {"function", "functions", "curve", "curves", "form", "forming", "forms", "make", "making"}
    | {"triangle", "polygon", "quadrilateral", "square", "rectangle", "pentagon", "hexagon", "shape"}
    # another drawable named as the link: "connect P and Q with a vector", "join A and B with an arc"
    | {"vector", "vectors", "arc", "arcs", "circle", "circles", "ellipse", "ray", "rays", "angle", "polygons"}
)
# The step checker: "check my steps", "which step of this derivation is wrong", "is x^2 - 1
# equivalent to (x-1)(x+1)?", "did I make a mistake going from (a+b)^2 to a^2 + b^2?".
# Shapes, constructions, graphs, units and fractions keep their own tools ("is triangle ABC
# equivalent to DEF?", "follow these steps to construct ...", "the equivalent of 5 miles in
# km", "equivalent fractions of 1/2").
_DERIVATION_WORDS = frozenset({"derivation", "derivations"})
_STEP_WORDS = frozenset({"step", "steps", "algebra", "simplification"})
# Said only of a derivation to check: "my steps", "my working", "which step is wrong" (not
# "the plot is not working")
_OWN_STEPS_RE = re.compile(
    r"\b(?:my|your)\s+(?:steps?|working|workings|algebra|simplification|derivation)\b"
    r"|\bworkings\b|\b(?:the|of)\s+working\b|\bwhich\s+step\b"
)
# Another operation named in the query: "plot the function step by step and check" plots,
# "compute the mean step by step and verify" computes
_OTHER_OPERATION_WORDS = frozenset(
    {"plot", "plots", "plotting", "draw", "drawing", "sketch", "shade", "shading", "zoom", "undo", "redo"}
    | {"evaluate", "evaluating", "compute", "computing", "calculate", "calculating", "solve", "solving"}
    | {"integral", "integrals", "integrate", "integrating", "integration", "derivative", "derivatives"}
    | {"differentiate", "differentiating", "limit", "limits", "expand", "factor", "convert", "tangent"}
    | {"mean", "median", "mode", "average", "statistics", "regression", "distribution", "area"}
)
_STEP_CHECK_WORDS = frozenset(
    {"check", "checking", "verify", "verifying", "wrong", "mistake", "mistakes", "error", "errors"}
    | {"correct", "valid", "follow", "follows", "right"}
)
_EQUIVALENCE_WORDS = frozenset({"equivalent", "equivalence"})
_MISTAKE_WORDS = frozenset({"mistake", "mistakes", "wrong", "error"})
_FROM_TO_RE = re.compile(r"\bfrom\b.+\bto\b")
_MATH_NOTATION_RE = re.compile(r"[\^*/(]")
_NOT_STEP_WORDS = frozenset(
    {"triangle", "triangles", "polygon", "polygons", "rectangle", "square", "quadrilateral", "circle", "circles"}
    | {"segment", "segments", "line", "lines", "vector", "vectors", "angle", "angles", "point", "points"}
    | {"construct", "construction", "constructions", "congruent", "similar", "shape", "shapes"}
    | {"graph", "graphs", "vertex", "vertices", "edge", "edges", "node", "nodes"}
    | {"unit", "units", "miles", "km", "kilometers", "meters", "inches", "feet", "celsius", "fahrenheit"}
    | {"fraction", "fractions"}
)
# Partial fractions: "partial fractions of 2/(x^2-1)", "decompose 1/(x(x+1)) into fractions".
_PARTIAL_FRACTION_WORDS = frozenset({"partial", "decompose", "decomposition", "decomposing"})
# Taylor polynomials: "Taylor polynomial of e^x at 0", "Maclaurin series of sin(x)", "the
# power series of 1/(1-x)", "a polynomial approximation of cos(x) near 0"; "does the series
# of 1/n^2 converge?" is a convergence test.
# "write 2/(x^2-1) as a sum of simpler fractions"
_SIMPLER_FRACTIONS_RE = re.compile(r"\bsum of (?:simpler |partial )?fractions\b")
_TAYLOR_WORDS = frozenset({"taylor", "maclaurin"})
_TAYLOR_PHRASE_RE = re.compile(
    r"\bpower series\b|\bseries expansion\b|\b(?:expand|write|express)\b.*\b(?:in|into|as) a series\b"
    r"|\bapproximating polynomial\b|\bpolynomial approximation\b|\b(?:quadratic|cubic) approximation\b"
    r"|\bapproximat\w*\b.*\b(?:with|by) a polynomial\b"
    r"|\blineari[sz](?:ation|e)\b|\blinear approximation\b.*\b(?:near|at|around)\b"
)
# d/dx, d/dt (the tokens split it into d and dx)
_D_BY_D_RE = re.compile(r"\bd/d[a-z]\b")
_CONVERGENCE_WORDS = frozenset({"converge", "converges", "convergent", "convergence", "diverge", "diverges"})
# Closed forms of sums come from simplify(summation(...)): "a closed form for the sum of k^2
# from 1 to n", "the sum of 1/k^2 from 1 to infinity", "does the series of 1/n^2 converge";
# not a sum of numbers ("add up 1 to 100", "the sum of the data") and not a bare "sigma",
# which is also a normal distribution's ("sigma 1, shade from 0 to infinity").
_SUM_WORDS = frozenset({"sum", "sums", "summation", "series"})
_SUM_PHRASE_RE = re.compile(
    r"\bclosed[- ]form\b|\bformula\b|\bto (?:n|infinity)\b|\binfinite\b|\bgeometric\b|\bsigma notation\b"
)
_NOT_SUM_WORDS = frozenset(
    {"data", "dataset", "values", "list", "mean", "average", "statistics", "taylor", "maclaurin"}
)
# Point names such as A, B or A' in the query as typed; "connect A and B" needs two of
# them (the pronoun "I" is not one).
_POINT_NAME_RE = re.compile(r"\b[A-Z]'*(?![\w'])")

# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------
CACHE_TTL = 300  # seconds
CACHE_MAX_SIZE = 100

_search_cache: Dict[str, Tuple[float, List[FunctionDefinition]]] = {}


def _cache_get(key: str) -> Optional[List[FunctionDefinition]]:
    entry = _search_cache.get(key)
    if entry is None:
        return None
    ts, results = entry
    if time.monotonic() - ts > CACHE_TTL:
        _search_cache.pop(key, None)
        return None
    return results


def _cache_put(key: str, results: List[FunctionDefinition]) -> None:
    # Evict oldest if at capacity
    if len(_search_cache) >= CACHE_MAX_SIZE and key not in _search_cache:
        oldest_key = min(_search_cache, key=lambda k: _search_cache[k][0])
        _search_cache.pop(oldest_key, None)
    _search_cache[key] = (time.monotonic(), results)


def configured_search_mode() -> str:
    """The ``TOOL_SEARCH_MODE`` the service uses: ``local``, ``api`` or ``hybrid`` (the default).

    Any other value searches locally, as ``search_tools`` treats it.
    """
    mode = os.getenv("TOOL_SEARCH_MODE", "hybrid").strip().lower()
    return mode if mode in ("api", "hybrid") else "local"


def clear_search_cache() -> None:
    """Clear the search result cache."""
    _search_cache.clear()


# ---------------------------------------------------------------------------
# Tool category registry
# ---------------------------------------------------------------------------

class CategoryEntry(TypedDict):
    tools: List[str]
    keywords: List[str]


TOOL_CATEGORIES: Dict[str, CategoryEntry] = {
    "geometry_create": {
        "tools": [
            "create_point", "create_segment", "create_vector",
            "create_polygon", "create_circle", "create_circle_arc",
            "create_ellipse", "create_label", "create_angle",
        ],
        "keywords": [
            "create", "draw", "make", "add", "place", "put",
            "construct", "point", "segment", "vector", "polygon",
            "triangle", "rectangle", "circle", "ellipse", "arc",
            "label", "angle", "line",
        ],
    },
    "geometry_delete": {
        "tools": [
            "delete_point", "delete_segment", "delete_vector",
            "delete_polygon", "delete_circle", "delete_circle_arc",
            "delete_ellipse", "delete_label", "delete_angle",
        ],
        "keywords": [
            "delete", "remove", "erase", "destroy", "rid",
        ],
    },
    "geometry_update": {
        "tools": [
            "update_point", "update_segment", "update_vector",
            "update_polygon", "update_circle", "update_circle_arc",
            "update_ellipse", "update_label", "update_angle",
        ],
        "keywords": [
            "update", "change", "modify", "edit", "rename", "resize", "set",
            "recolor", "reposition",
        ],
    },
    "constructions": {
        "tools": [
            "construct_midpoint", "construct_perpendicular_bisector",
            "construct_perpendicular_from_point", "construct_angle_bisector",
            "construct_parallel_line", "construct_circumcircle",
            "construct_incircle",
        ],
        "keywords": [
            "midpoint", "bisector", "perpendicular", "parallel",
            "circumcircle", "incircle", "construct", "bisect",
            "inscribed", "circumscribed",
        ],
    },
    "functions_plots": {
        "tools": [
            "draw_function", "delete_function", "update_function",
            "draw_piecewise_function", "delete_piecewise_function",
            "update_piecewise_function",
            "draw_parametric_function", "delete_parametric_function",
            "update_parametric_function",
            "draw_polar_function", "delete_polar_function",
            "update_polar_function",
            "draw_tangent_line", "draw_normal_line",
            "find_function_features",
        ],
        "keywords": [
            "function", "plot", "curve", "equation", "parametric",
            "piecewise", "tangent", "normal", "y=", "f(x)",
            "parabola", "sine", "cosine", "exponential", "logarithm",
            "polynomial", "lissajous", "spiral",
            "cardioid", "limacon", "lemniscate", "rose",
            "extrema", "extremum", "maxima", "minima", "zeros",
        ],
    },
    "math": {
        "tools": [
            "evaluate_expression", "evaluate_linear_algebra_expression",
            "convert", "limit", "derive", "integrate", "numeric_integrate",
            "simplify", "expand", "factor", "solve",
            "solve_system_of_equations", "solve_numeric", "check_steps",
            "taylor_polynomial",
        ],
        "keywords": [
            "calculate", "evaluate", "solve", "derivative", "integral",
            "integrate", "simplify", "factor", "expand", "limit",
            "differentiate", "calculus", "algebra", "expression",
            "equation", "compute", "math", "formula", "linear",
            "matrix", "determinant", "eigenvalue", "inverse",
            "numeric", "numerical", "system", "simultaneous",
            "find", "root", "roots", "reduce", "multiply",
            "taylor", "maclaurin",
        ],
    },
    "statistics": {
        "tools": [
            "plot_distribution", "plot_bars", "plot_histogram", "plot_box",
            "plot_scatter", "delete_plot",
            "fit_regression", "compute_descriptive_statistics", "compute_distribution",
        ],
        "keywords": [
            "statistics", "distribution", "bar", "chart", "mean",
            "median", "regression", "histogram", "normal", "gaussian",
            "standard", "deviation", "stdev", "quartile", "percentile",
            "probability", "bell", "discrete", "continuous", "frequency",
            "binomial", "poisson", "quantile",
            "descriptive", "stats", "average", "variance",
            # The verb "correlate" is left to the intent boost, which skips graph theory and matrices
            "correlation", "covariance", "pearson",
            "bins", "boxplot", "whisker", "whiskers", "outlier", "outliers",
            "scatter", "scatterplot",
        ],
    },
    "sequences": {
        "tools": ["plot_sequence", "plot_cobweb", "delete_plot"],
        # Only the unambiguous words: "sequence", "iterate" and "orbit" also name sequences of
        # actions, graph traversals and objects, so _boost_sequence_tools weighs them instead.
        "keywords": ["cobweb", "cobwebs"],
    },
    "graph_theory": {
        "tools": [
            "generate_graph", "delete_graph", "analyze_graph",
        ],
        "keywords": [
            "graph", "tree", "dag", "vertex", "vertices", "edge", "nodes",
            "node", "shortest", "path", "bfs", "dfs", "mst",
            "topological", "sort", "spanning", "adjacency",
            "directed", "undirected", "weighted", "network",
        ],
    },
    "canvas": {
        "tools": [
            "zoom", "clear_canvas", "reset_canvas",
            "set_coordinate_system", "set_grid_visible",
            "get_current_canvas_state", "run_tests",
            "undo", "redo",
        ],
        "keywords": [
            "zoom", "canvas", "grid", "coordinate", "reset", "clear",
            "view", "axes", "viewport", "pan", "fit",
            "undo", "redo", "state", "polar", "cartesian",
            "wipe", "clean", "fresh", "oops", "back", "revert",
        ],
    },
    "workspace": {
        "tools": [
            "save_workspace", "load_workspace",
            "list_workspaces", "delete_workspace",
        ],
        "keywords": [
            "workspace", "save", "load", "export", "import", "project",
            "session", "open", "restore", "persist", "store",
            "inventory", "available",
        ],
    },
    "transforms": {
        "tools": [
            "translate_object", "rotate_object", "reflect_object",
            "scale_object", "shear_object",
        ],
        "keywords": [
            "translate", "rotate", "reflect", "mirror", "scale",
            "shear", "transform", "move", "shift", "flip",
            "enlarge", "shrink", "stretch", "turn", "spin",
            "slide", "twice", "double", "bigger", "smaller", "larger",
        ],
    },
    "areas": {
        "tools": [
            "create_colored_area", "create_region_colored_area",
            "delete_colored_area", "update_colored_area",
            "calculate_area",
        ],
        "keywords": [
            "area", "shade", "region", "color", "fill", "highlight",
            "between", "under", "above", "bounded",
        ],
    },
    "inspection": {
        "tools": ["inspect_relation", "find_intersections"],
        "keywords": [
            "inspect", "relation", "check", "verify", "collinear",
            "concurrent", "tangent", "congruent", "similar",
            "relationship", "distance", "measure", "intersection",
        ],
    },
    "coordinates": {
        "tools": [
            "set_coordinate_system", "convert_coordinates",
        ],
        "keywords": [
            "polar", "cartesian", "coordinate", "system", "convert",
            "cylindrical", "spherical",
        ],
    },
}

# ---------------------------------------------------------------------------
# Inverted indices — built once at module load time
# ---------------------------------------------------------------------------

# token -> list of tool names whose *name* contains the token
_TOOL_NAME_INDEX: Dict[str, List[str]] = defaultdict(list)

# token -> list of tool names whose *description* contains the token
_TOOL_DESC_INDEX: Dict[str, List[str]] = defaultdict(list)

# keyword -> list of category names
_CATEGORY_KEYWORD_INDEX: Dict[str, List[str]] = defaultdict(list)

# tool_name -> FunctionDefinition (fast lookup)
_TOOL_BY_NAME: Dict[str, FunctionDefinition] = {}

# Set of all searchable tool names
_ALL_TOOL_NAMES: frozenset[str] = frozenset()


def _build_indices() -> None:
    """Populate inverted indices from FUNCTIONS and TOOL_CATEGORIES."""
    global _ALL_TOOL_NAMES

    names: List[str] = []
    for tool in FUNCTIONS:
        func = tool.get("function", {})
        name = func.get("name", "")
        if not name:
            continue
        # Always register in the name lookup (used by get_tool_by_name)
        _TOOL_BY_NAME[name] = tool

        # Skip meta-tools for search indices
        if name in EXCLUDED_FROM_SEARCH:
            continue

        names.append(name)

        # Index name tokens
        name_tokens = name.lower().replace("_", " ").split()
        for token in name_tokens:
            if token and len(token) > 1:
                _TOOL_NAME_INDEX[token].append(name)

        # Index description tokens
        description = func.get("description", "")
        desc_tokens = set(re.findall(r"[a-z0-9]+", description.lower()))
        for token in desc_tokens:
            if len(token) > 1:
                _TOOL_DESC_INDEX[token].append(name)

    _ALL_TOOL_NAMES = frozenset(names)

    # Build category keyword index
    for cat_name, cat_entry in TOOL_CATEGORIES.items():
        for keyword in cat_entry["keywords"]:
            _CATEGORY_KEYWORD_INDEX[keyword.lower()].append(cat_name)


_build_indices()

# Confidence threshold for hybrid mode
CONFIDENCE_THRESHOLD = 6.0

# Every loaded tool's schema is sent with each later request, so a search returns at
# most MAX_SEARCH_RESULTS tools, and local search leaves out matches scoring below
# RELATIVE_SCORE_CUTOFF of the leading match (generic verbs like "create" or "draw" give
# every create_*/draw_* tool a weak score that would otherwise fill the list).
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 6
RELATIVE_SCORE_CUTOFF = 0.3

# Action-verb to tool-name prefix mapping
_ACTION_VERB_MAP: Dict[str, str] = {
    "create": "create_",
    "draw": "draw_",
    "make": "create_",
    "add": "create_",
    "delete": "delete_",
    "remove": "delete_",
    "erase": "delete_",
    "update": "update_",
    "change": "update_",
    "modify": "update_",
    "edit": "update_",
    "construct": "construct_",
    "plot": "draw_",
    "slide": "translate_",
}


class ToolSearchService:
    """Service for tool discovery via local keyword matching or AI-powered search.

    The search mode is controlled by the ``TOOL_SEARCH_MODE`` environment variable.
    """

    # System prompt for tool selection (used in API mode)
    TOOL_SELECTOR_PROMPT = """You are a tool selector. Given a user's description of what they want to accomplish, select the most relevant tools from the list below. Return ONLY a JSON array of tool names, ordered by relevance (most relevant first).

Available tools:
{tool_descriptions}

User query: "{query}"

Return a JSON array of up to {max_results} tool names. Example: ["create_circle", "create_point"]"""
    _STOPWORDS = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "how",
            "i",
            "if",
            "in",
            "is",
            "it",
            "me",
            "of",
            "on",
            "or",
            "please",
            "the",
            "to",
            "up",
            "with",
            "you",
        }
    )

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        default_model: Optional[AIModel] = None,
    ) -> None:
        """Initialize the tool search service.

        Args:
            client: Optional OpenAI-compatible client. If not provided, creates
                one only when API mode is actually needed.
            default_model: Optional default model to use for API-based search.
        """
        self._client = client
        self._client_initialized = client is not None
        # Without a supplied client the service builds its own for api.openai.com.
        self._client_supplied = client is not None
        self.default_model = default_model
        self.last_error: Optional[str] = None
        self._last_local_top_score: float = 0.0

    @property
    def client(self) -> OpenAI:
        """Lazily initialize the OpenAI client on first access."""
        if not self._client_initialized:
            api_key = self._initialize_api_key()
            self._client = OpenAI(api_key=api_key)
            self._client_initialized = True
        assert self._client is not None
        return self._client

    @client.setter
    def client(self, value: OpenAI) -> None:
        self._client = value
        self._client_initialized = True
        self._client_supplied = True

    @staticmethod
    def _initialize_api_key() -> str:
        """Initialize the OpenAI API key from environment or .env file."""
        return get_api_key("OPENAI_API_KEY")

    @staticmethod
    def get_all_tools() -> List[FunctionDefinition]:
        """Get all available tool definitions."""
        return list(FUNCTIONS)

    @staticmethod
    def build_tool_descriptions(exclude_meta_tools: bool = True) -> str:
        """Build a compact string of all tool names and descriptions.

        Args:
            exclude_meta_tools: If True, excludes meta-tools like search_tools.

        Returns:
            Formatted string with tool names and descriptions.
        """
        lines: List[str] = []
        for tool in FUNCTIONS:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            # Skip meta-tools to avoid recursive suggestions
            if exclude_meta_tools and name in EXCLUDED_FROM_SEARCH:
                continue
            description = func.get("description", "No description")
            # Truncate long descriptions for the prompt
            if len(description) > 150:
                description = description[:147] + "..."
            lines.append(f"- {name}: {description}")
        return "\n".join(lines)

    @staticmethod
    def get_tool_by_name(name: str) -> Optional[FunctionDefinition]:
        """Get a tool definition by its name."""
        return _TOOL_BY_NAME.get(name)

    # ------------------------------------------------------------------
    # Public search entry point
    # ------------------------------------------------------------------

    def search_tools(
        self,
        query: str,
        model: Optional[AIModel] = None,
        max_results: int = DEFAULT_SEARCH_RESULTS,
    ) -> List[FunctionDefinition]:
        """Search for tools matching a query description.

        Dispatches to local or API search based on ``TOOL_SEARCH_MODE`` env var.

        Args:
            query: Description of what the user wants to accomplish.
            model: AI model to use for matching (API mode only).
            max_results: Maximum number of tools to return (1 to MAX_SEARCH_RESULTS).

        Returns:
            List of matching tool definitions, ordered by relevance.
        """
        if not query or not query.strip():
            _logger.warning("Tool search called with empty query")
            return []
        self.last_error = None

        # Clamp max_results to valid range
        max_results = max(1, min(MAX_SEARCH_RESULTS, max_results))

        mode = configured_search_mode()

        # Check cache
        cache_key = f"{mode}:{query.lower().strip()}:{max_results}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        if mode == "api":
            results = self._search_tools_api(query, model, max_results)
        elif mode == "hybrid":
            results = self.search_tools_local(query, max_results)
            if not results or self._last_local_top_score < CONFIDENCE_THRESHOLD:
                results = self._search_tools_api(query, model, max_results)
        else:  # "local" (default)
            results = self.search_tools_local(query, max_results)

        _cache_put(cache_key, results)
        return results

    # ------------------------------------------------------------------
    # Local search
    # ------------------------------------------------------------------

    def search_tools_local(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[FunctionDefinition]:
        """Search for tools using fast local keyword/category matching.

        No API call is made. Scoring uses:
        1. Category keyword boost (+5.0)
        2. Inverted index name match (+3.0)
        3. Inverted index description match (+1.0)
        4. Exact tool name match (+8.0)
        5. Action-verb alignment (+2.0)
        6. Intent boosts for confusion clusters

        Args:
            query: Description of what the user wants to accomplish.
            max_results: Maximum number of tools to return.

        Returns:
            List of matching tool definitions, ordered by score.
        """
        query_tokens = self._fold_destructive_verbs(self._tokenize(query))
        if not query_tokens:
            return []

        scores: Dict[str, float] = defaultdict(float)

        # 1. Exact tool name match
        for token in query_tokens:
            if token in _ALL_TOOL_NAMES:
                scores[token] += 8.0

        # Also check underscore-joined bigrams/trigrams for multi-word tool names
        for i in range(len(query_tokens)):
            for j in range(i + 1, min(i + 4, len(query_tokens) + 1)):
                candidate = "_".join(query_tokens[i:j])
                if candidate in _ALL_TOOL_NAMES:
                    scores[candidate] += 8.0

        # 2. Category keyword boost
        matched_categories: set[str] = set()
        for token in query_tokens:
            cats = _CATEGORY_KEYWORD_INDEX.get(token, [])
            matched_categories.update(cats)

        for cat_name in matched_categories:
            cat_entry = TOOL_CATEGORIES[cat_name]
            for tool_name in cat_entry["tools"]:
                if tool_name in _ALL_TOOL_NAMES:
                    scores[tool_name] += 5.0

        # 3. Inverted index name match
        for token in query_tokens:
            for tool_name in _TOOL_NAME_INDEX.get(token, []):
                scores[tool_name] += 3.0

        # 4. Inverted index description match
        for token in query_tokens:
            for tool_name in _TOOL_DESC_INDEX.get(token, []):
                scores[tool_name] += 1.0

        # 5. Action-verb alignment
        for token in query_tokens:
            prefix = _ACTION_VERB_MAP.get(token)
            if prefix:
                for tool_name in _ALL_TOOL_NAMES:
                    if tool_name.startswith(prefix):
                        scores[tool_name] += 2.0

        # 6. Intent boosts (same as existing _tool_score confusion boosts)
        self._apply_intent_boosts(query_tokens, scores, raw_query=query)

        # 7. Only offer run_tests when the query is about tests
        token_set = set(query_tokens)
        if not token_set & TEST_QUERY_WORDS:
            scores.pop("run_tests", None)

        # Sort by score descending, then alphabetically for ties; destructive tools go
        # after every other match unless the query asks to delete/clear/reset something.
        demote_destructive = not token_set & DESTRUCTIVE_QUERY_WORDS
        ranked = sorted(
            scores.items(),
            key=lambda item: (demote_destructive and item[0].startswith(DESTRUCTIVE_TOOL_PREFIXES), -item[1], item[0]),
        )

        # Return top results as full FunctionDefinition objects
        # Relative to the leading match, so a demoted destructive tool's high score
        # does not cut the matches ranked ahead of it
        min_score = RELATIVE_SCORE_CUTOFF * ranked[0][1] if ranked else 0.0
        results: List[FunctionDefinition] = []
        for tool_name, score in ranked:
            if score <= 0:
                break
            if score < min_score:
                continue  # Demoted destructive tools follow lower scores, so keep scanning
            tool = _TOOL_BY_NAME.get(tool_name)
            if tool is not None:
                results.append(tool)
                if len(results) >= max_results:
                    break

        # Store top score for hybrid-mode confidence check (the demotion only reorders,
        # so the best raw score is the confidence signal)
        self._last_local_top_score = max(scores.values()) if scores else 0.0

        _logger.info(
            f"Local tool search for '{query}' found {len(results)} tools"
        )
        return results

    @staticmethod
    def _fold_destructive_verbs(tokens: List[str]) -> List[str]:
        """Map inflected destructive verbs to their base form ("deleting" -> "delete")."""
        token_set = set(tokens)
        folded: List[str] = []
        for token in tokens:
            match = _DESTRUCTIVE_INFLECTION_RE.match(token)
            if match is None:
                folded.append(token)
            elif match.group(1):
                folded.append(match.group(1) + "e")
            elif match.group(2):
                folded.append(match.group(2))
            elif token_set & _DROP_CONSTRUCTION_WORDS:
                folded.append(token)
            else:
                folded.append("delete")
        return folded

    @staticmethod
    def _may_be_function_plot(token_set: set[str], raw_lower: str) -> bool:
        """False when the query names a shape, point or chart, unless it defines y or f(x)."""
        named = token_set & _NOT_A_FUNCTION_WORDS
        if not named:
            return True
        return named <= {"point", "points"} and _FUNCTION_DEFINITION_RE.search(raw_lower) is not None

    @staticmethod
    def _describes_graph_structure(token_set: set[str], raw_query: str) -> bool:
        """Whether a query that says "graph" means vertices and edges, not a function plot."""
        raw_lower = raw_query.lower()
        if token_set & _GRAPH_STRUCTURE_WORDS or _GRAPH_FAMILY_RE.search(raw_lower):
            return True
        if token_set & _OTHER_GRAPH_WORDS:
            return False
        if _EDGE_LIST_RE.search(raw_query):  # before the notation check: "->" contains ">"
            return True
        if _FUNCTION_NOTATION_RE.search(raw_lower):
            return False
        if token_set & _GRAPH_VERTEX_WORDS and token_set & _GRAPH_EDGE_WORDS:
            return True
        return _BARE_GRAPH_REQUEST_RE.match(raw_lower) is not None

    @staticmethod
    def _names_graph_theory(token_set: set[str], raw_query: str) -> bool:
        """Whether a query uses graph-theory words (one unambiguous word, or two everyday ones)
        and gives no number lists, which make it a data question."""
        if _NUMBER_LIST_RE.search(raw_query):
            return False
        return bool(token_set & _GRAPH_THEORY_ONLY_WORDS) or len(token_set & _GRAPH_THEORY_AMBIGUOUS_WORDS) >= 2

    @staticmethod
    def _names_polar_curve(token_set: Set[str], raw_lower: str) -> bool:
        """Whether a query is about a polar curve r(θ), not a polar point, grid or conversion.

        "Plot r = 1 + cos(θ)", "the cardioid", "polar curve f_polar", "the rose curve" are;
        "the point with r = 2 and θ = π/3", "convert (r=5, theta=0.9) to cartesian", "switch
        to a polar grid and plot y = sin(x)", "plot (2, π/3) in polar coordinates" and "a
        vector with magnitude r = 5 at θ = 30°" are not.
        """
        if token_set & _POLAR_NOT_CURVE_WORDS:
            return False
        if token_set & {"parametric", "parametrically"} or _PARAMETRIC_RE.search(raw_lower):
            # "a cardioid using x(t) = ..., y(t) = ...": a parametric curve
            return False
        equation = bool(_POLAR_CURVE_RE.search(raw_lower)) and ("θ" in raw_lower or "theta" in token_set)
        if _POLAR_EQUATION_RE.search(raw_lower):
            # "plot r = 2sin(3θ) and measure the angle between two petals": the curve is given
            return True
        if token_set & _POLAR_SHAPE_WORDS:
            return False
        # A named curve is a curve on any grid: "plot a cardioid on a polar grid"
        names_curve = token_set & _POLAR_CURVE_NAMES or "limaçon" in raw_lower
        if names_curve or any(token.endswith("_polar") for token in token_set):
            return True
        if "rose" in token_set and token_set & {"curve", "petal", "petals", "polar"}:
            return True
        if _POLAR_CURVE_PHRASE_RE.search(raw_lower):
            return True
        if token_set & {"grid", "coordinates"} and not equation:
            # The polar grid, or a point in polar coordinates, unless an r(θ) is given
            return False
        if "polar" in token_set and token_set & _POLAR_CURVE_NOUNS:
            return True
        return equation

    @staticmethod
    def _asks_for_histogram(token_set: set[str]) -> bool:
        """Whether a query asks for a histogram of data, not a bar chart of grouped values
        ("a histogram-like bar plot from these grouped values" is a bar chart)."""
        if not token_set & _HISTOGRAM_WORDS:
            return False
        return not (token_set & {"bar", "bars"} and token_set & {"grouped", "like", "categories", "category"})

    @staticmethod
    def _asks_for_box_plot(token_set: set[str]) -> bool:
        """Whether a query asks for a box plot: "boxplot", whiskers, or "box" with plot or chart
        and no function words (a bare "box" is a rectangle)."""
        if token_set & _BOX_PLOT_WORDS:
            return True
        if "box" not in token_set or not token_set & {"plot", "plots", "chart", "diagram"}:
            return False
        # "plot the function inside the box": a function plot, not a box plot
        return not token_set & {"function", "functions", "curve", "graph", "equation"}

    @staticmethod
    def _penalize(scores: Dict[str, float], tools: Tuple[str, ...], amount: float) -> None:
        """Lower the tools' scores, dropping any that reach 0 (ranking stops at the first score <= 0)."""
        for tool in tools:
            if tool not in scores:
                continue
            scores[tool] -= amount
            if scores[tool] <= 0:
                del scores[tool]

    @staticmethod
    def _boost_sequence_tools(token_set: Set[str], raw_lower: str, scores: Dict[str, float]) -> None:
        """plot_cobweb for an iteration's orbit, plot_sequence for the terms of a sequence.

        Sums, series, limits and convergence tests of sequences stay with evaluate_expression,
        simplify and limit (both tools are lowered, which also cancels the name match of
        "sequence"); sequences of actions, transformations, graph traversals or objects are not
        number sequences; deleting, zooming to or saving a plot is left to those tools.
        """
        if token_set & _SEQUENCE_REQUEST_EXCLUDED_WORDS:
            if token_set & DESTRUCTIVE_QUERY_WORDS:
                # "remove the cobweb diagram", "delete the sequence": delete_plot, not a new plot
                ToolSearchService._penalize(scores, ("plot_sequence", "plot_cobweb"), 6.0)
                named = token_set & (_SEQUENCE_WORDS | _COBWEB_WORDS)
                if named and not token_set & (_NOT_NUMBER_SEQUENCE_WORDS - {"named"}):
                    scores["delete_plot"] += 8.0
            return
        partial_sums = "partial sum" in raw_lower
        time_series = "time series" in raw_lower
        sum_words = token_set & _SEQUENCE_SUM_WORDS and not partial_sums and not time_series
        web = bool(token_set & _COBWEB_WORDS) or ("web" in token_set and bool(token_set & _PLOT_NOUN_WORDS))
        if sum_words and not web:
            ToolSearchService._penalize(scores, ("plot_sequence", "plot_cobweb"), 8.0)
            return
        fixed_point = "fixed point" in raw_lower.replace("-", " ")
        iterates = bool(token_set & _ITERATION_WORDS) or "logistic map" in raw_lower
        if iterates and token_set & _ROOT_METHOD_WORDS and not fixed_point:
            iterates = False
        stability = fixed_point and bool(token_set & _STABILITY_WORDS)
        next_term = _NEXT_TERM_RE.search(raw_lower) is not None
        if (web or next_term or stability or iterates) and not token_set & _NOT_ITERATION_WORDS:
            # A named web or cobweb outranks the curve words it is drawn with ("a web plot for the logistic map")
            scores["plot_cobweb"] += 12.0 if web else 10.0
            return
        terms_named = bool(
            token_set & _SEQUENCE_WORDS
            or _SEQUENCE_TERM_RE.search(raw_lower)
            or _FIRST_TERMS_RE.search(raw_lower)
            or partial_sums
        )
        plotted = bool(token_set & _SEQUENCE_PLOT_VERBS) and terms_named
        if (token_set & _SEQUENCE_LIMIT_WORDS or _ORDINAL_TERM_RE.search(raw_lower)) and not plotted:
            ToolSearchService._penalize(scores, ("plot_sequence", "plot_cobweb"), 8.0)
            return
        if token_set & _NOT_NUMBER_SEQUENCE_WORDS:
            ToolSearchService._penalize(scores, ("plot_sequence",), 3.0)
            return
        if _PREVIOUS_TERM_RE.search(raw_lower):
            # u(n-1) only appears in a recursion; it outranks the "u(n) =" read as a function
            scores["plot_sequence"] += 14.0
        elif terms_named:
            scores["plot_sequence"] += 10.0

    @staticmethod
    def _boost_distribution_tools(token_set: set[str], scores: Dict[str, float], normal_line: bool) -> None:
        """compute_distribution for probabilities and quantiles, plot_distribution for pictures.

        A distribution is named outright (binomial, poisson, chi, student, calculator names
        such as normalcdf), or by an everyday word (normal, exponential, uniform, geometric)
        next to a distribution word; exponential growth, geometric sequences and uniform
        scaling stay with their own tools.
        """
        named = bool(token_set & _DISTRIBUTION_NAME_WORDS) or bool(
            token_set & _AMBIGUOUS_DISTRIBUTION_WORDS and token_set & _DISTRIBUTION_CONTEXT_WORDS
        )
        if "normal" in token_set and not normal_line:
            named = True
        asks_probability = bool(token_set & _PROBABILITY_WORDS)
        # "critical value of the t distribution": the one-letter name is lost, its context is not.
        named = named or (asks_probability and bool(token_set & _DISTRIBUTION_CONTEXT_WORDS - _PROBABILITY_WORDS))
        if not (named or asks_probability):
            return
        if token_set & _PICTURE_WORDS:
            if named:
                scores["plot_distribution"] += 6.0
            return
        # Calculator names (invNorm, binomcdf) outrank evaluate_expression's call-syntax boost.
        scores["compute_distribution"] += (10.0 if token_set & _DISTRIBUTION_NAME_WORDS else 8.0) if named else 3.0

    @staticmethod
    def _apply_intent_boosts(
        query_tokens: List[str],
        scores: Dict[str, float],
        raw_query: str = "",
    ) -> None:
        """Apply intent-based score boosts for known confusion clusters."""
        token_set = set(query_tokens)
        raw_lower = raw_query.lower() if raw_query else " ".join(query_tokens)

        # -- Transforms --
        if token_set & {"move", "shift", "translate", "slide"}:
            scores["translate_object"] += 6.0
        if token_set & {"turn", "spin", "rotate"}:
            scores["rotate_object"] += 6.0
        if token_set & {"reflect", "mirror", "flip"}:
            scores["reflect_object"] += 4.0
        if token_set & {"twice", "double", "bigger", "larger", "smaller", "big"}:
            scores["scale_object"] += 8.0

        # -- Areas / shading --
        if token_set & {"shade", "shading", "color", "fill", "highlight"}:
            scores["create_colored_area"] += 5.0
            scores["create_region_colored_area"] += 4.0
        if token_set & {"area"}:
            scores["calculate_area"] += 2.0
            scores["create_colored_area"] += 3.0
            scores["create_region_colored_area"] += 2.0
        if token_set & {"region", "between"}:
            scores["create_region_colored_area"] += 5.0
            scores["create_colored_area"] += 3.0

        # -- Statistics --
        # "the normal to f at x=2", "tangent and normal", "normal to y=x^2": a normal line, unless
        # distribution words say otherwise.
        normal_line = "normal" in token_set and bool(
            token_set & {"line", "perpendicular"}
            or (
                not token_set & _DISTRIBUTION_CONTEXT_WORDS
                and (
                    _FUNCTION_DEFINITION_RE.search(raw_lower)
                    or _NORMAL_LINE_RE.search(raw_lower)
                    or "tangent" in token_set
                )
            )
        )
        if token_set & {"distribution", "gaussian", "bell"}:
            scores["plot_distribution"] += 6.0
        if token_set & {"normal", "normally"} and not normal_line:
            scores["plot_distribution"] += 4.0
        ToolSearchService._boost_distribution_tools(token_set, scores, normal_line)
        ToolSearchService._boost_sequence_tools(token_set, raw_lower, scores)
        histogram = ToolSearchService._asks_for_histogram(token_set)
        box_plot = ToolSearchService._asks_for_box_plot(token_set)
        if token_set & {"bar", "bars"}:
            scores["plot_bars"] += 6.0
        if histogram:
            # "a bar chart of the bin counts": the bar chart that was named wins
            scores["plot_histogram"] += 3.0 if token_set & {"bar", "bars"} else 8.0
        if box_plot:
            scores["plot_box"] += 10.0
        # A scatter plot asked for its fit or trend line is a regression ("best-fit line for this scatter data")
        if token_set & _SCATTER_WORDS and not token_set & _FIT_WORDS:
            scores["plot_scatter"] += 8.0
        # "plot y = 2x + 1 on the scatter plot": the function is drawn over the data
        if token_set & _SCATTER_WORDS and (_FUNCTION_EQUATION_RE.search(raw_lower) or token_set & _FUNCTION_NAME_WORDS):
            scores["draw_function"] += 4.0
        if token_set & {"regression", "fit", "fitting", "trend"}:
            scores["fit_regression"] += 6.0
        if token_set & _DESCRIPTIVE_STATISTICS_WORDS:
            scores["compute_descriptive_statistics"] += 7.0
        if {"five", "summary"} <= token_set and not box_plot:
            # "the five-number summary of 4, 8, 15": statistics, not an expression to evaluate
            scores["compute_descriptive_statistics"] += 8.0
        # A fit that is asked for its correlation belongs to fit_regression, whose linear fit reports r;
        # a covariance matrix is linear algebra, and correlating vertices, edges or paths is a graph
        # analysis. A lone "graph" or "tree" is a plot or data ("graph the correlation of x and y",
        # "correlation between tree age and height").
        if token_set & {"correlation", "correlate", "correlated", "covariance", "pearson"} and not (
            token_set & {"fit", "fitting", "regression", "matrix", "matrices"} or ToolSearchService._names_graph_theory(token_set, raw_lower)
        ):
            scores["compute_descriptive_statistics"] += 6.0

        # -- Linear algebra --
        if token_set & {"determinant", "eigenvalue", "matrix", "matrices"}:
            scores["evaluate_linear_algebra_expression"] += 6.0
        if "multiply" in token_set and token_set & {"matrix", "matrices"}:
            scores["evaluate_linear_algebra_expression"] += 6.0

        # -- Undo/redo --
        if "undo" in token_set:
            scores["undo"] += 8.0
        if "redo" in token_set:
            scores["redo"] += 8.0
        if token_set & {"oops", "revert"}:
            scores["undo"] += 8.0
        if "back" in token_set and "go" in token_set:
            scores["undo"] += 8.0

        # -- Tangent/normal lines --
        if "tangent" in token_set:
            scores["draw_tangent_line"] += 6.0
        if normal_line:
            scores["draw_normal_line"] += 6.0

        # -- Roots, extrema and intersections of plotted functions --
        _curve_words = {"function", "functions", "curve", "curves", "graph", "graphs", "plotted", "plot", "local"}
        if token_set & {"extrema", "extremum", "maxima", "minima", "turning"}:
            scores["find_function_features"] += 8.0
        if token_set & {"inflection", "inflections", "concavity", "concave", "convex"}:
            scores["find_function_features"] += 10.0
        if token_set & {"maximum", "minimum", "maximums", "minimums"} and not (
            token_set & {"spanning", "tree", "path", "weight", "weighted", "flow"}
        ):
            scores["find_function_features"] += 12.0 if "local" in token_set else 8.0
        _shape_words = {"segment", "segments", "line", "lines", "circle", "circles", "ellipse", "polygon", "vector",
                        "polygons", "triangle", "triangles", "rectangle", "rectangles", "square",
                        "quadrilateral", "pentagon", "hexagon"}
        if token_set & {"intersect", "intersection", "intersections", "intersects", "meet", "cross"} and (
            token_set & _curve_words or not token_set & _shape_words
        ):
            scores["find_function_features"] += 8.0
        _on_canvas_words = {"plotted", "curve", "graph", "place", "mark"}
        if token_set & {"roots", "root", "zeros", "zero"} and token_set & _on_canvas_words:
            scores["find_function_features"] += 8.0

        # -- Intersections of segments, circles, ellipses and curves --
        _meet_words = {"intersect", "intersection", "intersections", "intersects", "meet", "meets", "cross", "crosses"}
        _object_words = _shape_words | {"vectors", "ellipses", "arc", "arcs", "parametric"}
        if token_set & _meet_words:
            scores["find_intersections"] += 12.0 if token_set & _object_words else 3.0

        # -- Constructions --
        if "inscribed" in token_set and token_set & {"circle"}:
            scores["construct_incircle"] += 10.0
        if "circumscribed" in token_set and token_set & {"circle"}:
            scores["construct_circumcircle"] += 10.0
        if token_set & {"perpendicular"}:
            # Question form ("are X and Y perpendicular?") → inspect, not construct
            is_question = (
                raw_lower.endswith("?")
                or "each other" in raw_lower
                or raw_lower.lstrip().startswith("are ")
                or raw_lower.lstrip().startswith("is ")
                or token_set & {"check", "verify", "whether"}
            )
            if is_question:
                scores["inspect_relation"] += 10.0
            else:
                scores["construct_perpendicular_bisector"] += 3.0
                scores["construct_perpendicular_from_point"] += 3.0
        if token_set & {"midpoint", "middle"}:
            scores["construct_midpoint"] += 6.0

        # -- Delete with casual language --
        if "rid" in token_set:
            # "get rid of" is a common idiom for delete
            for tool_name in _ALL_TOOL_NAMES:
                if tool_name.startswith("delete_"):
                    scores[tool_name] += 4.0
                # Penalize create_ tools when intent is clearly delete
                if tool_name.startswith("create_"):
                    scores[tool_name] -= 6.0

        # -- Solve family --
        # "Solve x^2 - 5x + 6 < 0", "for which x is |x - 1| >= 2?": the solution set of an
        # inequality, which solve gives. Shading or plotting one is left to the drawing tools,
        # an area, integral, derivative, extremum, ... over a range to their own tools, and a
        # function defined by cases ("f(x) = x for x < 0 and x^2 when x >= 0") to
        # draw_piecewise_function.
        condition_rule = _PIECEWISE_RE.search(raw_lower) is not None
        defines_function = bool(_FUNCTION_DEFINITION_RE.search(raw_lower) or token_set & _DEFINES_FUNCTION_WORDS)
        solves_inequality = (
            bool(_INEQUALITY_SIGN_RE.search(raw_lower) or token_set & {"inequality", "inequalities"})
            and bool(token_set & _INEQUALITY_SOLVE_WORDS)
            and not token_set & _INEQUALITY_OTHER_WORDS
            and not _PROBABILITY_CALL_RE.search(raw_lower)
            and not _NAMES_RELATION_RE.search(raw_query)
            and not (condition_rule and (defines_function or token_set & _VALUE_QUESTION_WORDS))
        )
        if solves_inequality:
            scores["solve"] += 6.0
        if token_set & {"solve", "find"} and token_set & {"system", "simultaneous", "equations"}:
            scores["solve_system_of_equations"] += 8.0
        elif token_set & {"system", "simultaneous"}:
            scores["solve_system_of_equations"] += 6.0
        # Detect multiple equations: "... and ..." pattern with = signs
        _eq_count = raw_lower.count("=")
        if _eq_count >= 2 and token_set & {"solve", "find"}:
            scores["solve_system_of_equations"] += 6.0
        elif _eq_count >= 2 and " and " in raw_lower:
            scores["solve_system_of_equations"] += 5.0
        if token_set & {"solve", "find"} and token_set & {"numeric", "numerical", "numerically", "approximate", "root"}:
            scores["solve_numeric"] += 6.0
        elif token_set & {"numeric", "numerical", "numerically", "approximate"}:
            scores["solve_numeric"] += 4.0
        if token_set & {"solve"}:
            scores["solve"] += 4.0
        if token_set & {"roots"}:
            scores["solve"] += 6.0
        # Word-problem patterns (cost, spent, buy, how many)
        if token_set & {"cost", "spent", "buy", "price", "total"} and token_set & {"how", "many", "each"}:
            scores["solve_system_of_equations"] += 8.0
            scores["solve"] += 6.0
        elif "how" in token_set and "many" in token_set:
            scores["solve"] += 4.0
            scores["solve_system_of_equations"] += 3.0
        # "find x" without system/numeric context -> solve
        if "find" in token_set and not (token_set & {"system", "simultaneous", "numeric", "numerical", "numerically",
                                                      "approximate", "shortest", "path", "bfs", "dfs"}):
            scores["solve"] += 3.0

        # -- Calculus --
        if token_set & {"derivative", "differentiate", "diff"} or _D_BY_D_RE.search(raw_lower) is not None:
            scores["derive"] += 6.0
        if token_set & {"integral", "integrate", "integration"}:
            scores["integrate"] += 6.0
            scores["numeric_integrate"] += 2.0
        if token_set & {"simplify", "reduce"}:
            scores["simplify"] += 6.0
        # An exact fraction or radical comes from simplify (evaluate_expression gives decimals).
        # A function's value at a point or an area ("f(3) as a fraction", "the exact area of
        # the triangle") keeps its own tools first, with simplify behind them.
        wants_exact = bool(token_set & {"exact", "fraction", "fractions", "radical"})
        named_at_point = wants_exact and _NAMED_FUNCTION_AT_POINT_RE.search(raw_lower) is not None
        at_point = named_at_point or _FUNCTION_AT_POINT_RE.search(raw_lower) is not None
        exact_alongside = wants_exact and (named_at_point or bool(token_set & _MEASUREMENT_WORDS))
        if wants_exact:
            scores["simplify"] += 3.0 if exact_alongside else 6.0
        only_exact = wants_exact and not exact_alongside
        if token_set & {"mod", "modulo", "remainder"}:
            scores["evaluate_expression"] += 6.0
        if token_set & {"expand", "multiply"} and not (token_set & {"matrix", "matrices"}):
            scores["expand"] += 6.0
        if token_set & {"factor", "factorize", "factorise", "factored"}:
            scores["factor"] += 6.0
        # Partial fractions come from expand(expr, x), as on calculators; "fractions" alone
        # still means an exact answer from simplify
        if (
            token_set & {"fraction", "fractions"} and token_set & _PARTIAL_FRACTION_WORDS
        ) or _SIMPLER_FRACTIONS_RE.search(raw_lower) is not None:
            scores["expand"] += 10.0
            scores["simplify"] -= 6.0
        # Taylor and Maclaurin polynomials; a series to test for convergence is not one
        if token_set & _TAYLOR_WORDS or (
            _TAYLOR_PHRASE_RE.search(raw_lower) is not None and not token_set & _CONVERGENCE_WORDS
        ):
            scores["taylor_polynomial"] += 10.0
            scores["expand"] -= 20.0  # "Taylor expand ln(1+x) at 0", "expand e^x in a power series"
        sum_named = bool(token_set & _SUM_WORDS) or "∑" in raw_lower or "sigma notation" in raw_lower
        if (
            sum_named
            and (_SUM_PHRASE_RE.search(raw_lower) is not None or token_set & _CONVERGENCE_WORDS or "∞" in raw_lower)
            and not token_set & (_NOT_SUM_WORDS | _NOT_STEP_WORDS)  # "a series of circles"
        ):
            scores["simplify"] += 8.0
        if token_set & {"limit", "lim"}:
            scores["limit"] += 6.0
        if token_set & {"compute", "calculate", "evaluate"} and not (
            token_set & {"area", "statistics", "descriptive", "stats", "mean", "median"}
        ):
            scores["evaluate_expression"] += 3.0
        # Values of a function at points: "f(3)", "value of f at 2", "table of values"; not an
        # exact value ("the exact value of sqrt(12)"), which simplify gives
        # "u(1) = 1" in "the recurrence u(n) = n*u(n-1), u(1) = 1" is a first term, not a value to find
        if not only_exact and at_point and not (token_set & _SEQUENCE_WORDS or _PREVIOUS_TERM_RE.search(raw_lower)):
            scores["evaluate_expression"] += 12.0
        elif not only_exact and token_set & {"value", "values", "table", "plug", "substitute"}:
            scores["evaluate_expression"] += 8.0

        # -- Step checker --
        if not token_set & _NOT_STEP_WORDS and (
            token_set & _DERIVATION_WORDS
            or _OWN_STEPS_RE.search(raw_lower) is not None
            or (
                not token_set & _OTHER_OPERATION_WORDS
                and (
                    (token_set & _STEP_WORDS and token_set & _STEP_CHECK_WORDS)
                    or token_set & _EQUIVALENCE_WORDS
                    or (
                        token_set & _MISTAKE_WORDS
                        and _FROM_TO_RE.search(raw_lower) is not None
                        and _MATH_NOTATION_RE.search(raw_lower) is not None
                    )
                )
            )
        ):
            scores["check_steps"] += 10.0
        elif token_set & _NOT_STEP_WORDS and token_set & _EQUIVALENCE_WORDS:
            scores["check_steps"] -= 10.0  # the equivalent of 5 miles in km, equivalent fractions

        polar_curve = ToolSearchService._names_polar_curve(token_set, raw_lower)

        # -- Convert (unit vs coordinate) --
        if token_set & ({"convert", "change"} | _EQUIVALENCE_WORDS) and token_set & {
            "unit", "units", "temperature", "celsius", "fahrenheit",
            "miles", "km", "kilometers", "meters", "inches", "feet",
        }:
            scores["convert"] += 8.0
        if token_set & {"convert", "change"} and token_set & {
            "polar", "cartesian", "coordinate", "coordinates",
            "rectangular",
        } and not polar_curve:
            scores["convert_coordinates"] += 8.0

        # -- Polar curves r(θ) --
        if polar_curve:
            if token_set & {"delete", "remove", "erase"}:
                scores["delete_polar_function"] += 8.0
            elif token_set & _POLAR_UPDATE_VERBS and not _POLAR_CURVE_RE.search(raw_lower):
                # "update two cardioids", "change the polar plot to red": the verb says edit
                scores["update_polar_function"] += 8.0
                if token_set & _POLAR_DRAW_WORDS:
                    scores["draw_polar_function"] += 4.0
            elif (token_set & _POLAR_EDIT_WORDS) and not (
                token_set & _POLAR_DRAW_WORDS
                or _POLAR_CURVE_RE.search(raw_lower)
                or _names_new_polar_curve(raw_lower)
            ):
                # "make the cardioid blue", "set θ_max of f_polar to 4π"; not "draw it in red"
                # or "make a cardioid" (a new curve)
                scores["update_polar_function"] += 8.0
            else:
                scores["draw_polar_function"] += 8.0
        elif "polar" in token_set and token_set & {"point", "points", "grid", "coordinate", "coordinates"}:
            # "the point with polar coordinates r = 2, θ = π/3", "plot (2, π/3) in polar
            # coordinates", "a polar grid": a point or the grid, not a curve
            scores["draw_polar_function"] -= 6.0
            if not token_set & _POLAR_EDIT_WORDS:
                scores["update_polar_function"] -= 6.0
        parametric_words = token_set & {"parametric", "parametrically"}
        if not polar_curve and (parametric_words or _PARAMETRIC_RE.search(raw_lower)):
            # "plot a cardioid using x(t) = ..., y(t) = ...": the curve is named, but given
            # parametrically
            scores["draw_polar_function"] -= 6.0

        # -- Functions/plotting --
        if parametric_words:
            scores["draw_parametric_function"] += 6.0
        # Detect parametric-like patterns: x=f(t), y=g(t)  or  x(t)=..., y(t)=...
        if ("x=" in raw_lower and "y=" in raw_lower and
                any(fn in raw_lower for fn in ("cos(t)", "sin(t)", "t*", "(t)"))):
            scores["draw_parametric_function"] += 8.0
        elif _PARAMETRIC_RE.search(raw_lower) and _PARAMETRIC_Y_RE.search(raw_lower):
            # The same with spaces: "x(t) = 3*cos(t), y(t) = 2*sin(t)", "x = t^2, y = t"
            scores["draw_parametric_function"] += 8.0
        if token_set & {"piecewise", "rules"}:
            scores["draw_piecewise_function"] += 6.0
        # Detect piecewise patterns: "for x<..." or "when x>..." with multiple conditions
        # "for which x is x >= 2 and x^2 < 9" is an inequality to solve, not a piecewise rule
        piecewise_pattern = condition_rule and not solves_inequality
        if piecewise_pattern:
            scores["draw_piecewise_function"] += 6.0
        # "graph f(x)", "graph y = x^2", "graph sin(x)", "graph x squared" mean draw_function, not
        # graph theory. Notation is read from the raw query (the tokenizer drops "(", "=" and
        # one-letter words); piecewise definitions are left to draw_piecewise_function, and
        # shapes or charts given by an equation or coordinates to their own tools.
        if (
            "graph" in token_set
            and not (piecewise_pattern or token_set & {"piecewise", "rules"})
            and ToolSearchService._may_be_function_plot(token_set, raw_lower)
            and (token_set & _FUNCTION_NAME_WORDS or _FUNCTION_EQUATION_RE.search(raw_lower))
        ):
            scores["draw_function"] += 6.0
        graph_structure_query = (
            "graph" in token_set and ToolSearchService._describes_graph_structure(token_set, raw_query or raw_lower)
        ) or bool(token_set & {"node", "nodes"} and token_set & {"edge", "edges"})
        # "graph f(x)=..." pattern: "graph" + words suggesting a function equation
        if "graph" in token_set and not (graph_structure_query or token_set & _GRAPH_ANALYSIS_WORDS):
            # Boost function plotting when "graph" appears without graph-theory context
            scores["draw_function"] += 4.0
            scores["draw_piecewise_function"] += 2.0
            # Penalize graph-theory tools to avoid confusion
            scores["generate_graph"] -= 3.0
            scores["analyze_graph"] -= 3.0
            scores["delete_graph"] -= 3.0
        if token_set & {"curve", "plot"} and not (token_set & {"bar", "bars", "distribution"}):
            scores["draw_function"] += 3.0

        # -- Graph theory (only with graph-theory context) --
        if graph_structure_query:
            scores["generate_graph"] += 8.0
            scores["analyze_graph"] += 4.0
        elif (
            token_set & _CONNECT_WORDS
            and not token_set & _CONNECT_EXCLUDED_WORDS
            and (token_set & _CONNECT_OBJECT_WORDS or len(set(_POINT_NAME_RE.findall(raw_query)) - {"I"}) >= 2)
        ):
            # "connect A and B", "connect the points with segments": a segment
            scores["create_segment"] += 6.0
        if token_set & {"shortest", "bfs", "dfs", "mst", "topological", "spanning", "degree"}:
            scores["analyze_graph"] += 8.0
        if "graph" in token_set and token_set & {"analysis", "analyze", "statistics", "degree"}:
            scores["analyze_graph"] += 6.0

        # -- Inspection --
        if token_set & {"inspect", "relation", "relationship"}:
            scores["inspect_relation"] += 6.0

        # -- Workspace operations --
        if token_set & {"workspace", "project", "session"}:
            for name in ("save_workspace", "load_workspace", "list_workspaces", "delete_workspace"):
                scores[name] += 4.0
        if token_set & {"save", "persist", "store"}:
            scores["save_workspace"] += 5.0
        if token_set & {"load", "open", "restore"}:
            scores["load_workspace"] += 5.0
        if token_set & {"list", "inventory", "available", "names"}:
            scores["list_workspaces"] += 5.0

        # -- Canvas operations --
        if token_set & {"zoom", "viewport", "pan", "reframe", "narrow", "window"}:
            scores["zoom"] += 6.0
        if "reset" in token_set and token_set & {"canvas", "zoom", "view", "default"}:
            scores["reset_canvas"] += 6.0
        if token_set & {"wipe", "clean", "fresh"}:
            scores["clear_canvas"] += 8.0
            scores["reset_canvas"] += 4.0
        if token_set & {"grid"}:
            scores["set_grid_visible"] += 6.0

        # -- Labels --
        if token_set & {"text", "annotation", "note", "label", "annotate"}:
            scores["create_label"] += 5.0
        if token_set & {"delete"} and token_set & {"plot"}:
            scores["delete_plot"] += 6.0

        # -- Polygons --
        if token_set & {"triangle", "quadrilateral", "pentagon", "hexagon", "rectangle", "square"}:
            scores["create_polygon"] += 4.0
            scores["delete_polygon"] += 4.0
            scores["update_polygon"] += 4.0

        # -- Coordinate system --
        if token_set & {"polar", "cartesian"} and token_set & {"system", "mode", "switch"}:
            scores["set_coordinate_system"] += 6.0

        # Simplify's +3 alone falls under RELATIVE_SCORE_CUTOFF of a function value's score
        # and behind the other area tools, so it scores one point under the leading tool
        if exact_alongside:
            leading = max(score for name, score in scores.items() if not name.startswith(DESTRUCTIVE_TOOL_PREFIXES))
            scores["simplify"] = max(scores["simplify"], leading - 1.0)

    # ------------------------------------------------------------------
    # API-based search (original implementation)
    # ------------------------------------------------------------------

    def _search_tools_api(
        self,
        query: str,
        model: Optional[AIModel] = None,
        max_results: int = 10,
    ) -> List[FunctionDefinition]:
        """Search for tools using the AI API (original implementation).

        Args:
            query: Description of what the user wants to accomplish.
            model: AI model to use for matching. Defaults to the instance
                default model, then to API_SEARCH_FALLBACK_MODEL.
            max_results: Maximum number of tools to return (1-20).

        Returns:
            List of matching tool definitions, ordered by relevance.
        """
        model = self._resolve_search_model(model)

        # Build the prompt
        tool_descriptions = self.build_tool_descriptions()
        prompt = self.TOOL_SELECTOR_PROMPT.format(
            tool_descriptions=tool_descriptions,
            query=query,
            max_results=max_results,
        )

        try:
            # Call the AI model
            request_kwargs = self._build_search_request(model, prompt)
            response = self.client.chat.completions.create(**request_kwargs)

            # Extract the response content
            content = response.choices[0].message.content
            if not content:
                _logger.warning("Tool search returned empty response")

            # Parse the JSON array of tool names
            tool_names = self._parse_tool_names(content) if content else []
            if not tool_names:
                # Fallback for non-compliant model outputs
                tool_names = self._fallback_tool_names(query, max_results)

            # Look up full definitions for each tool name (excluding meta-tools)
            matched_tools: List[FunctionDefinition] = []
            for name in tool_names:
                if name in EXCLUDED_FROM_SEARCH:
                    continue
                tool = self.get_tool_by_name(name)
                if tool is not None:
                    matched_tools.append(tool)
                    if len(matched_tools) >= max_results:
                        break

            _logger.info(f"Tool search for '{query}' found {len(matched_tools)} tools")
            return matched_tools

        except Exception as e:
            self.last_error = str(e)
            _logger.error(f"Tool search failed: {e}")
            fallback_names = self._fallback_tool_names(query, max_results)
            fallback_tools: List[FunctionDefinition] = []
            for name in fallback_names:
                tool = self.get_tool_by_name(name)
                if tool is not None:
                    fallback_tools.append(tool)
            return fallback_tools

    def _resolve_search_model(self, model: Optional[AIModel]) -> AIModel:
        """Pick the model for an API search: the given one, else the instance
        default, else the fallback. The fallback also replaces any model that
        rejects reasoning effort "none", and any non-OpenAI model when the
        service built its own OpenAI client (the active provider, e.g.
        Anthropic, had no OpenAI-compatible client to pass in)."""
        if model is None:
            model = self.default_model
        if model is None or model.id in _MODELS_REJECTING_NO_REASONING:
            return AIModel.from_identifier(API_SEARCH_FALLBACK_MODEL)
        if not self._client_supplied and model.provider != PROVIDER_OPENAI:
            return AIModel.from_identifier(API_SEARCH_FALLBACK_MODEL)
        return model

    @staticmethod
    def _build_search_request(model: AIModel, prompt: str) -> Dict[str, Any]:
        """Build the Chat Completions keyword arguments for an API search."""
        request_kwargs: Dict[str, Any] = {
            "model": model.id,
            "messages": [{"role": "user", "content": prompt}],
        }
        if model.is_reasoning_model and model.provider == PROVIDER_OPENAI:
            # OpenAI reasoning models on Chat Completions reject temperature and
            # max_tokens; a plain name lookup needs no reasoning.
            request_kwargs["reasoning_effort"] = "none"
            request_kwargs["max_completion_tokens"] = 500
        else:
            request_kwargs["temperature"] = 0.0  # Deterministic for consistent results
            request_kwargs["max_tokens"] = 500
        if model.provider == PROVIDER_LOCAL_AGENT:
            # Same reasoning effort as LocalAgent's chat requests; without one a
            # reasoning model can spend the whole 500 tokens thinking.
            from static.providers.local import get_configured_reasoning_effort

            effort = get_configured_reasoning_effort()
            if effort is not None:
                request_kwargs["extra_body"] = {"chat_template_kwargs": {"reasoning_effort": effort}}
        return request_kwargs

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        # "limaçon" is one word, as "limacon"
        tokens = [t for t in re.findall(r"[a-z0-9_]+", text.lower().replace("ç", "c")) if t]
        return [t for t in tokens if t not in cls._STOPWORDS and len(t) > 1]

    @classmethod
    def _tool_score(cls, query_tokens: List[str], tool_name: str, description: str) -> float:
        if not query_tokens:
            return 0.0
        name_text = tool_name.lower().replace("_", " ")
        desc_text = description.lower()
        score = 0.0
        for token in query_tokens:
            if token == tool_name.lower():
                score += 8.0
            if token in name_text:
                score += 3.0
            if token in desc_text:
                score += 1.0
        # Intent boosts for common confusion clusters.
        if any(t in query_tokens for t in ("move", "shift", "translate")) and tool_name == "translate_object":
            score += 4.0
        if any(t in query_tokens for t in ("area", "shade", "region")) and tool_name in {
            "calculate_area",
            "create_colored_area",
            "create_region_colored_area",
        }:
            score += 2.0
        if any(t in query_tokens for t in ("distribution", "gaussian", "normal")) and tool_name == "plot_distribution":
            score += 4.0
        if (
            any(t in query_tokens for t in ("determinant", "eigenvalue", "matrix", "vector"))
            and tool_name == "evaluate_linear_algebra_expression"
        ):
            score += 4.0
        if any(t in query_tokens for t in ("undo", "redo", "history")) and tool_name in {"undo", "redo"}:
            score += 4.0
        return score

    @classmethod
    def _fallback_tool_names(cls, query: str, max_results: int) -> List[str]:
        query_tokens = cls._tokenize(query)
        scored: List[tuple[float, str]] = []
        for tool in FUNCTIONS:
            func = tool.get("function", {})
            name = func.get("name", "")
            if not name or name in EXCLUDED_FROM_SEARCH:
                continue
            description = func.get("description", "")
            score = cls._tool_score(query_tokens, name, description)
            if score > 0:
                scored.append((score, name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, name in scored[:max_results]]

    @staticmethod
    def _extract_list_from_parsed(parsed: Any) -> List[str]:
        """Extract a list of strings from a parsed JSON value.

        Handles both direct arrays and objects with 'tools' key.
        """
        # Direct array
        if isinstance(parsed, list):
            return [str(item) for item in parsed if isinstance(item, str)]
        # Object with 'tools' key (e.g., {"tools": ["create_circle", ...]})
        if isinstance(parsed, dict):
            tools = parsed.get("tools")
            if isinstance(tools, list):
                return [str(item) for item in tools if isinstance(item, str)]
        return []

    @staticmethod
    def _parse_tool_names(content: str) -> List[str]:
        """Parse tool names from AI response.

        Handles various response formats including:
        - JSON arrays: ["tool1", "tool2"]
        - JSON objects: {"tools": ["tool1", "tool2"]}
        - Markdown code blocks with JSON
        """
        content = content.strip()

        # Try to extract JSON from markdown code blocks
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("[") or part.startswith("{"):
                    content = part
                    break

        # Try to parse as JSON (array or object)
        try:
            parsed = json.loads(content)
            result = ToolSearchService._extract_list_from_parsed(parsed)
            if result:
                return result
        except json.JSONDecodeError:
            pass

        # Fallback: try to find JSON array in the content
        start_idx = content.find("[")
        end_idx = content.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                json_str = content[start_idx : end_idx + 1]
                parsed = json.loads(json_str)
                result = ToolSearchService._extract_list_from_parsed(parsed)
                if result:
                    return result
            except json.JSONDecodeError:
                pass

        # Fallback: try to find JSON object in the content
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                json_str = content[start_idx : end_idx + 1]
                parsed = json.loads(json_str)
                result = ToolSearchService._extract_list_from_parsed(parsed)
                if result:
                    return result
            except json.JSONDecodeError:
                pass

        _logger.warning(f"Could not parse tool names from: {content[:100]}")
        return []

    def search_tools_formatted(
        self,
        query: str,
        model: Optional[AIModel] = None,
        max_results: int = DEFAULT_SEARCH_RESULTS,
    ) -> Dict[str, Any]:
        """Search for tools and return formatted result for AI consumption.

        The result names the tools rather than repeating their schemas: the server
        loads the found tools into the request's tool list, which already carries
        each schema, while this result stays in the conversation history.

        Args:
            query: Description of what the user wants to accomplish.
            model: AI model to use for matching.
            max_results: Maximum number of tools to return.

        Returns:
            Dict with the found tool names under 'tools', plus 'count' and 'query'.
        """
        tools = self.search_tools(query, model, max_results)
        return {
            "tools": [tool["function"]["name"] for tool in tools],
            "count": len(tools),
            "query": query,
        }
