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
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from openai import OpenAI

from static.ai_model import AIModel
from static.functions_definitions import FUNCTIONS, FunctionDefinition

_logger = logging.getLogger("mathud")

# Tools to exclude from search results (meta-tools that shouldn't be recommended)
EXCLUDED_FROM_SEARCH = frozenset({"search_tools"})

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
            "draw_tangent_line", "draw_normal_line",
        ],
        "keywords": [
            "function", "plot", "curve", "equation", "parametric",
            "piecewise", "tangent", "normal", "y=", "f(x)",
            "parabola", "sine", "cosine", "exponential", "logarithm",
            "polynomial", "lissajous", "spiral",
        ],
    },
    "math": {
        "tools": [
            "evaluate_expression", "evaluate_linear_algebra_expression",
            "convert", "limit", "derive", "integrate", "numeric_integrate",
            "simplify", "expand", "factor", "solve",
            "solve_system_of_equations", "solve_numeric",
        ],
        "keywords": [
            "calculate", "evaluate", "solve", "derivative", "integral",
            "integrate", "simplify", "factor", "expand", "limit",
            "differentiate", "calculus", "algebra", "expression",
            "equation", "compute", "math", "formula", "linear",
            "matrix", "determinant", "eigenvalue", "inverse",
            "numeric", "numerical", "system", "simultaneous",
            "find", "root", "roots", "reduce", "multiply",
        ],
    },
    "statistics": {
        "tools": [
            "plot_distribution", "plot_bars", "delete_plot",
            "fit_regression", "compute_descriptive_statistics",
        ],
        "keywords": [
            "statistics", "distribution", "bar", "chart", "mean",
            "median", "regression", "histogram", "normal", "gaussian",
            "standard", "deviation", "stdev", "quartile", "percentile",
            "probability", "bell", "discrete", "continuous", "frequency",
            "descriptive", "stats", "average", "variance",
        ],
    },
    "graph_theory": {
        "tools": [
            "generate_graph", "delete_graph", "analyze_graph",
        ],
        "keywords": [
            "graph", "tree", "dag", "vertex", "vertices", "edge",
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
        "tools": ["inspect_relation"],
        "keywords": [
            "inspect", "relation", "check", "verify", "collinear",
            "concurrent", "tangent", "congruent", "similar",
            "relationship", "distance", "measure",
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

    @staticmethod
    def _initialize_api_key() -> str:
        """Initialize the OpenAI API key from environment or .env file."""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key

        # Load from project .env, then parent .env (API keys may live outside repo)
        load_dotenv()
        parent_env = os.path.join(os.path.dirname(os.getcwd()), ".env")
        if os.path.exists(parent_env):
            load_dotenv(parent_env)
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or .env file")

        return api_key

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
        max_results: int = 10,
    ) -> List[FunctionDefinition]:
        """Search for tools matching a query description.

        Dispatches to local or API search based on ``TOOL_SEARCH_MODE`` env var.

        Args:
            query: Description of what the user wants to accomplish.
            model: AI model to use for matching (API mode only).
            max_results: Maximum number of tools to return (1-20).

        Returns:
            List of matching tool definitions, ordered by relevance.
        """
        if not query or not query.strip():
            _logger.warning("Tool search called with empty query")
            return []
        self.last_error = None

        # Clamp max_results to valid range
        max_results = max(1, min(20, max_results))

        mode = os.getenv("TOOL_SEARCH_MODE", "hybrid").strip().lower()

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
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = defaultdict(float)

        # 1. Exact tool name match
        query_lower = query.lower().strip()
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

        # Sort by score descending, then alphabetically for ties
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

        # Return top results as full FunctionDefinition objects
        results: List[FunctionDefinition] = []
        for tool_name, score in ranked:
            if score <= 0:
                break
            tool = _TOOL_BY_NAME.get(tool_name)
            if tool is not None:
                results.append(tool)
                if len(results) >= max_results:
                    break

        # Store top score for hybrid-mode confidence check
        self._last_local_top_score = ranked[0][1] if ranked else 0.0

        _logger.info(
            f"Local tool search for '{query}' found {len(results)} tools"
        )
        return results

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
        if token_set & {"distribution", "gaussian", "bell"}:
            scores["plot_distribution"] += 6.0
        if "normal" in token_set and not (token_set & {"line", "perpendicular"}):
            scores["plot_distribution"] += 4.0
        if token_set & {"bar", "bars"}:
            scores["plot_bars"] += 6.0
        if token_set & {"regression", "fit", "fitting"}:
            scores["fit_regression"] += 6.0
        if token_set & {"descriptive", "stats", "statistics", "mean", "median", "stdev", "quartile", "average"}:
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
        if "normal" in token_set and token_set & {"line", "perpendicular"}:
            scores["draw_normal_line"] += 6.0

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
        if token_set & {"derivative", "differentiate", "diff", "d/dx"}:
            scores["derive"] += 6.0
        if token_set & {"integral", "integrate", "integration"}:
            scores["integrate"] += 6.0
            scores["numeric_integrate"] += 2.0
        if token_set & {"simplify", "reduce"}:
            scores["simplify"] += 6.0
        if token_set & {"expand", "multiply"} and not (token_set & {"matrix", "matrices"}):
            scores["expand"] += 6.0
        if token_set & {"factor", "factorize", "factorise", "factored"}:
            scores["factor"] += 6.0
        if token_set & {"limit", "lim"}:
            scores["limit"] += 6.0
        if token_set & {"compute", "calculate", "evaluate"} and not (
            token_set & {"area", "statistics", "descriptive", "stats", "mean", "median"}
        ):
            scores["evaluate_expression"] += 3.0

        # -- Convert (unit vs coordinate) --
        if token_set & {"convert", "change"} and token_set & {
            "unit", "units", "temperature", "celsius", "fahrenheit",
            "miles", "km", "kilometers", "meters", "inches", "feet",
        }:
            scores["convert"] += 8.0
        if token_set & {"convert", "change"} and token_set & {
            "polar", "cartesian", "coordinate", "coordinates",
            "rectangular",
        }:
            scores["convert_coordinates"] += 8.0

        # -- Functions/plotting --
        if token_set & {"parametric"}:
            scores["draw_parametric_function"] += 6.0
        # Detect parametric-like patterns: x=f(t), y=g(t)  or  x(t)=..., y(t)=...
        if ("x=" in raw_lower and "y=" in raw_lower and
                any(fn in raw_lower for fn in ("cos(t)", "sin(t)", "t*", "(t)"))):
            scores["draw_parametric_function"] += 8.0
        if token_set & {"piecewise", "rules"}:
            scores["draw_piecewise_function"] += 6.0
        # Detect piecewise patterns: "for x<..." or "when x>..." with multiple conditions
        if re.search(r"for\s+x\s*[<>]|when\s+x\s*[<>]|x\s*>=|x\s*<=", raw_lower):
            scores["draw_piecewise_function"] += 6.0
        # "graph f(x)" / "graph y=" means draw_function, not graph theory
        if "graph" in token_set and token_set & {"f(x)", "y=", "sin", "cos", "exp", "ln", "log"}:
            scores["draw_function"] += 6.0
        # "graph f(x)=..." pattern: "graph" + words suggesting a function equation
        if "graph" in token_set and not (
            token_set & {"vertex", "vertices", "edge", "edges", "node", "nodes",
                         "directed", "undirected", "weighted", "shortest", "path",
                         "bfs", "dfs", "mst", "topological", "spanning", "adjacency",
                         "degree", "dag", "tree", "network"}
        ):
            # Boost function plotting when "graph" appears without graph-theory context
            scores["draw_function"] += 4.0
            scores["draw_piecewise_function"] += 2.0
            # Penalize graph-theory tools to avoid confusion
            scores["generate_graph"] -= 3.0
            scores["analyze_graph"] -= 3.0
            scores["delete_graph"] -= 3.0
        if token_set & {"curve", "plot"} and not (token_set & {"bar", "bars", "distribution"}):
            scores["draw_function"] += 3.0

        # -- Graph theory (only with explicit graph-theory context) --
        if "graph" in token_set and token_set & {
            "vertex", "vertices", "edge", "edges", "node", "nodes",
            "directed", "undirected", "weighted", "network",
        }:
            scores["generate_graph"] += 8.0
            scores["analyze_graph"] += 4.0
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
            model: AI model to use for matching. Defaults to gpt-4.1-mini.
            max_results: Maximum number of tools to return (1-20).

        Returns:
            List of matching tool definitions, ordered by relevance.
        """
        # Use provided model, instance default, or fallback to gpt-4.1-mini.
        if model is None:
            model = self.default_model or AIModel.from_identifier("gpt-5-nano")

        # Build the prompt
        tool_descriptions = self.build_tool_descriptions()
        prompt = self.TOOL_SELECTOR_PROMPT.format(
            tool_descriptions=tool_descriptions,
            query=query,
            max_results=max_results,
        )

        try:
            # Call the AI model
            request_kwargs: Dict[str, Any] = {
                "model": model.id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,  # Deterministic for consistent results
            }
            # OpenAI reasoning models in Chat Completions reject max_tokens
            # and require max_completion_tokens.
            if model.is_reasoning_model and model.provider == "openai":
                request_kwargs["max_completion_tokens"] = 500
            else:
                request_kwargs["max_tokens"] = 500

            response = self.client.chat.completions.create(**request_kwargs)

            # Extract the response content
            content = response.choices[0].message.content
            if not content:
                _logger.warning("Tool search returned empty response")
                return []

            # Parse the JSON array of tool names
            tool_names = self._parse_tool_names(content)
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

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        tokens = [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t]
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
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """Search for tools and return formatted result for AI consumption.

        Args:
            query: Description of what the user wants to accomplish.
            model: AI model to use for matching.
            max_results: Maximum number of tools to return.

        Returns:
            Dict with 'tools' list and 'count' for the AI to use.
        """
        tools = self.search_tools(query, model, max_results)
        return {
            "tools": tools,
            "count": len(tools),
            "query": query,
        }
