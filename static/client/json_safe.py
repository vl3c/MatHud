"""
Conversion of tool results to plain JSON data.

Tool results travel back to the model as JSON (``AIInterface`` posts them with
``json.dumps``). Brython's ``json.dumps`` rejects tuples and sets, and when it raises
part-way through a value it leaves that value's containers on its internal
circular-reference stack: every later ``json.dumps`` of the same objects then fails
with "Circular reference detected", even though nothing is circular. So a result is
converted to plain data (dicts with string keys, lists, strings, numbers, booleans and
None) before anything serializes it.

Brython's ``json.dumps`` also differs from CPython's on dict keys: it rejects ``None``
and float keys, and writes ``True`` as ``"True"`` (CPython writes ``"true"``). Only
string keys serialize the same way in both, so ``to_json_safe`` turns every key into
a string, as CPython would write it.

``to_json_safe`` never raises and never calls ``json.dumps``. Anything JSON cannot
express becomes a string saying what it was. ``is_json_plain`` checks, without
serializing, whether a value can be dumped as it is; ``ensure_json_safe`` converts
only values that cannot.
"""

from __future__ import annotations

from typing import Any, List, Set

# Deeper nesting than this is cut off; tool results are a few levels deep.
MAX_JSON_SAFE_DEPTH = 64

CIRCULAR_REFERENCE_TEXT = "<circular reference>"
TOO_DEEP_TEXT = "<nested too deeply>"


def to_json_safe(value: Any) -> Any:
    """Return a copy of ``value`` that ``json.dumps`` can serialize, in Brython and CPython.

    - tuples become lists; sets and frozensets become lists, sorted when possible;
    - dict keys become strings the way CPython's ``json.dumps`` writes them (``1`` -> ``"1"``,
      ``True`` -> ``"true"``, ``None`` -> ``"null"``; Brython's rejects ``None`` and float keys);
    - NaN and infinite floats become ``"NaN"``, ``"Infinity"`` and ``"-Infinity"``;
    - a drawable becomes ``"<Class> '<name>'"``; any other object becomes ``str(obj)``;
    - a container that contains itself becomes ``"<circular reference>"``.
    """
    return _convert(value, set(), 0)


def is_json_plain(value: Any) -> bool:
    """True when ``json.dumps`` accepts ``value`` as it is, in Brython and CPython.

    Plain data is None, booleans, numbers (NaN and infinities included), strings, lists
    and dicts with string, int or bool keys, nested without cycles. Tuples, sets, other
    objects and ``None`` or float keys make a value not plain. Never calls ``json.dumps``.
    """
    return _is_plain(value, set(), 0)


def ensure_json_safe(value: Any) -> Any:
    """``value`` itself when it is plain JSON data, else its ``to_json_safe`` copy."""
    return value if is_json_plain(value) else to_json_safe(value)


def _is_plain(value: Any, active: Set[int], depth: int) -> bool:
    if value is None or isinstance(value, (bool, str, int, float)):
        return True
    if not isinstance(value, (dict, list)) or depth >= MAX_JSON_SAFE_DEPTH:
        return False
    marker = id(value)
    if marker in active:
        return False
    active.add(marker)
    try:
        if isinstance(value, dict):
            return all(isinstance(k, (str, int)) and _is_plain(v, active, depth + 1) for k, v in value.items())
        return all(_is_plain(item, active, depth + 1) for item in value)
    finally:
        active.discard(marker)


def _convert(value: Any, active: Set[int], depth: int) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return _convert_float(value)
    if not isinstance(value, (dict, list, tuple, set, frozenset)):
        return _describe_object(value)
    if depth >= MAX_JSON_SAFE_DEPTH:
        return TOO_DEEP_TEXT
    marker = id(value)
    if marker in active:
        return CIRCULAR_REFERENCE_TEXT
    active.add(marker)
    try:
        if isinstance(value, dict):
            return {_convert_key(k): _convert(v, active, depth + 1) for k, v in value.items()}
        items: List[Any] = _ordered_items(value) if isinstance(value, (set, frozenset)) else list(value)
        return [_convert(item, active, depth + 1) for item in items]
    finally:
        active.discard(marker)


def _convert_float(value: float) -> Any:
    if value != value:
        return "NaN"
    if value == float("inf"):
        return "Infinity"
    if value == float("-inf"):
        return "-Infinity"
    return value


def _convert_key(key: Any) -> str:
    """Write a dict key as ``json.dumps`` would; keys JSON cannot express use ``str``."""
    if isinstance(key, str):
        return key
    if key is True:
        return "true"
    if key is False:
        return "false"
    if key is None:
        return "null"
    if isinstance(key, float):
        converted = _convert_float(key)
        return converted if isinstance(converted, str) else str(key)
    return str(key)


def _ordered_items(values: Any) -> List[Any]:
    """Set members in sorted order when they can be sorted, else in iteration order."""
    items = list(values)
    try:
        return sorted(items)
    except Exception:
        return items


def _describe_object(value: Any) -> str:
    """A truthful string for an object JSON cannot express."""
    try:
        class_name = value.get_class_name() if hasattr(value, "get_class_name") else None
        name = getattr(value, "name", None)
        if isinstance(class_name, str) and isinstance(name, str):
            return f"{class_name} '{name}'"
    except Exception:
        pass
    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}>"
