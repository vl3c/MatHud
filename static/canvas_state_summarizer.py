"""
Canvas state summarization utilities for AI-facing debugging and comparison.

This module keeps workspace/persistence state untouched and provides a pruned
view intended for prompt-quality inspection.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Tuple, cast


_DRAWABLE_FIELD_PRUNE_RULES: Dict[str, set[str]] = {
    "Segments": {"_p1_coords", "_p2_coords"},
    "Vectors": {"_origin_coords", "_tip_coords"},
    "Circles": {"circle_formula"},
    "Ellipses": {"ellipse_formula"},
    # Plot internals useful for rendering/persistence, but noisy for AI context.
    "ContinuousPlots": {"function_name", "fill_area_name"},
    "DiscretePlots": {"rectangle_names"},
    "BarsPlots": {"rectangle_names"},
}

_ALWAYS_PRUNE_PREFIXES: Tuple[str, ...] = ("_",)
_ESTIMATED_TOKEN_RATIO: int = 4  # rough estimate: ~4 chars/token

# Keys that should always stay present when found at object level.
_IDENTITY_KEYS: set[str] = {"name", "args"}


def summarize_canvas_state(full_state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a pruned copy of canvas state for AI-facing readability."""
    if not isinstance(full_state, dict):
        return {}

    summary: Dict[str, Any] = copy.deepcopy(full_state)

    for key, value in list(summary.items()):
        if isinstance(value, list):
            if _is_drawable_bucket(value):
                summary[key] = _summarize_drawable_bucket(key, value)
            else:
                summary[key] = _strip_empty_values(value)
        elif isinstance(value, dict):
            summary[key] = _strip_empty_values(value)

    return cast(Dict[str, Any], _canonicalize(summary))


def compare_canvas_states(full_state: Dict[str, Any]) -> Dict[str, Any]:
    """Return full+summary states with deterministic size metrics."""
    canonical_full = _canonicalize(copy.deepcopy(full_state if isinstance(full_state, dict) else {}))
    summary = summarize_canvas_state(canonical_full)

    full_json = json.dumps(canonical_full, separators=(",", ":"), sort_keys=True)
    summary_json = json.dumps(summary, separators=(",", ":"), sort_keys=True)

    full_bytes = len(full_json.encode("utf-8"))
    summary_bytes = len(summary_json.encode("utf-8"))
    reduction_pct = 0.0
    if full_bytes > 0:
        reduction_pct = round((1.0 - (summary_bytes / float(full_bytes))) * 100.0, 2)

    return {
        "full": canonical_full,
        "summary": summary,
        "metrics": {
            "full_bytes": full_bytes,
            "summary_bytes": summary_bytes,
            "full_estimated_tokens": _estimate_tokens(full_json),
            "summary_estimated_tokens": _estimate_tokens(summary_json),
            "reduction_pct": reduction_pct,
        },
    }


def _summarize_drawable_bucket(bucket_name: str, drawables: List[Any]) -> List[Any]:
    result: List[Any] = []
    prune_fields = _DRAWABLE_FIELD_PRUNE_RULES.get(bucket_name, set())

    for item in drawables:
        if not isinstance(item, dict):
            result.append(item)
            continue

        pruned = _prune_drawable_object(item, prune_fields)
        result.append(pruned)

    return _stable_sort_drawables(result)


def _prune_drawable_object(obj: Dict[str, Any], prune_fields: set[str]) -> Dict[str, Any]:
    pruned: Dict[str, Any] = {}

    for key, value in obj.items():
        if key in _IDENTITY_KEYS:
            pruned[key] = _strip_empty_values(value)
            continue
        if key in prune_fields:
            continue
        if any(key.startswith(prefix) for prefix in _ALWAYS_PRUNE_PREFIXES):
            continue
        pruned[key] = _strip_empty_values(value)

    # Keep identity keys if they existed on input.
    for key in _IDENTITY_KEYS:
        if key in obj and key not in pruned:
            pruned[key] = _strip_empty_values(obj[key])

    _prune_default_label_metadata(pruned)
    return cast(Dict[str, Any], _strip_empty_values(pruned))


def _prune_default_label_metadata(obj: Dict[str, Any]) -> None:
    args = obj.get("args")
    if not isinstance(args, dict):
        return
    label = args.get("label")
    if not isinstance(label, dict):
        return

    text = str(label.get("text", "") or "")
    visible = bool(label.get("visible", False))
    if not text and not visible:
        args.pop("label", None)


def _strip_empty_values(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in value.items():
            cleaned_v = _strip_empty_values(v)
            if _is_empty_value(cleaned_v):
                continue
            cleaned[k] = cleaned_v
        return cleaned
    if isinstance(value, list):
        cleaned_list: List[Any] = []
        for item in value:
            cleaned_item = _strip_empty_values(item)
            if _is_empty_value(cleaned_item):
                continue
            cleaned_list.append(cleaned_item)
        return cleaned_list
    return value


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict, str)):
        return len(value) == 0
    return False


def _is_drawable_bucket(items: List[Any]) -> bool:
    if not items:
        return False

    # Classify as drawable bucket only when sampled entries look like drawables.
    # This avoids accidental classification of unrelated list-valued metadata.
    sample = items[:3]
    return all(isinstance(entry, dict) and ("name" in entry or "args" in entry) for entry in sample)


def _stable_sort_drawables(items: List[Any]) -> List[Any]:
    keyed: List[Tuple[str, str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name", ""))
            args = item.get("args", {})
            args_key = ""
            try:
                args_key = json.dumps(args, sort_keys=True, separators=(",", ":"))
            except Exception:
                args_key = str(args)
            keyed.append((name, args_key, item))
        else:
            keyed.append(("", str(item), item))
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [item for _, _, item in keyed]


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _canonicalize(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        if value and all(isinstance(v, dict) for v in value):
            maybe_sorted = _stable_sort_drawables(value)
            return [_canonicalize(v) for v in maybe_sorted]
        return [_canonicalize(v) for v in value]
    return value


def _estimate_tokens(serialized_json: str) -> int:
    if not serialized_json:
        return 0
    return max(1, len(serialized_json) // _ESTIMATED_TOKEN_RATIO)
