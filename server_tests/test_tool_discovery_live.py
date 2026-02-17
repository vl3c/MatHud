from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from static.ai_model import AIModel
from static.functions_definitions import FUNCTIONS
from static.openai_api_base import ESSENTIAL_TOOLS
from static.providers import create_provider_instance, discover_providers
from static.tool_search_service import ToolSearchService

pytestmark = pytest.mark.live_tool_discovery

DATASET_PATH = Path("server_tests/data/tool_discovery_cases.yaml")
CSV_FIELDS = [
    "case_id",
    "category",
    "cluster",
    "query",
    "expected_any",
    "top1",
    "top3",
    "top5",
    "ranked",
    "status",
    "error",
]


def _load_dataset() -> Dict[str, Any]:
    raw = DATASET_PATH.read_text(encoding="utf-8")
    # The file uses JSON-compatible YAML so we can parse without PyYAML.
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Dataset root must be an object")
    return parsed


def _tool_name_set() -> set[str]:
    return {f.get("function", {}).get("name", "") for f in FUNCTIONS if f.get("function", {}).get("name")}


def _tool_hash(tool_names: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(tool_names)).encode("utf-8")).hexdigest()


def _is_blocked_error(error_text: Optional[str]) -> bool:
    if not error_text:
        return False
    text = error_text.lower()
    markers = (
        "invalid_prompt",
        "flagged as potentially violating",
        "safety",
        "content_policy",
    )
    return any(marker in text for marker in markers)


def _is_infra_error(error_text: Optional[str]) -> bool:
    if not error_text:
        return False
    text = error_text.lower()
    markers = (
        "rate_limit_error",
        "rate limit exceeded",
        "too low to access",
        "credit balance is too low",
        "model_not_found",
        "does not exist or you do not have access",
        "provider unavailable",
        "maximum context length",
        "requested about",
    )
    return any(marker in text for marker in markers)


def _search_ranked_names_with_model(
    service: ToolSearchService,
    model: Optional[AIModel],
    query: str,
    max_results: int,
    provider_instance: Optional[Any],
) -> Tuple[List[str], Optional[str]]:
    """Run search against the appropriate provider and return ranked tool names."""
    if model is None or model.provider == "openai":
        ranked_tools = service.search_tools(query=query, model=model, max_results=max_results)
        ranked = [t.get("function", {}).get("name", "") for t in ranked_tools if isinstance(t, dict)]
        ranked = [name for name in ranked if name]
        return ranked, service.last_error

    if provider_instance is None:
        provider_name = model.provider if model is not None else "unknown"
        return [], f"{provider_name} provider unavailable for tool discovery benchmark"

    tool_descriptions = service.build_tool_descriptions()
    prompt = service.TOOL_SELECTOR_PROMPT.format(
        tool_descriptions=tool_descriptions,
        query=query,
        max_results=max_results,
    )
    try:
        if hasattr(provider_instance, "reset_conversation"):
            provider_instance.reset_conversation()
        choice = provider_instance.create_chat_completion(prompt)
        message_obj = getattr(choice, "message", None)
        content = getattr(message_obj, "content", "") if message_obj is not None else ""
        if not isinstance(content, str):
            content = str(content)
        tool_names = service._parse_tool_names(content)

        ranked: List[str] = []
        for name in tool_names:
            if name in ESSENTIAL_TOOLS:
                continue
            tool = service.get_tool_by_name(name)
            if tool is None:
                continue
            ranked.append(name)
            if len(ranked) >= max_results:
                break
        return ranked, None
    except Exception as exc:  # pragma: no cover - exercised in live mode
        return [], str(exc)


def _rank_of_expected(ranked: List[str], expected_any: List[str]) -> Optional[int]:
    for idx, tool in enumerate(ranked, start=1):
        if tool in expected_any:
            return idx
    return None


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _resolve_model() -> Optional[AIModel]:
    model_id = os.getenv("TOOL_DISCOVERY_MODEL", "").strip()
    if not model_id:
        return None
    return AIModel.from_identifier(model_id)


def _to_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _to_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def _load_existing_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    case_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_id = row.get("case_id", "")
            if case_id:
                case_ids.add(case_id)
    return case_ids


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TOOL_DISCOVERY", "").strip() != "1",
    reason="Set RUN_LIVE_TOOL_DISCOVERY=1 to run paid/on-demand tool discovery benchmark.",
)
def test_live_tool_discovery_benchmark() -> None:
    dataset = _load_dataset()
    metadata = dataset.get("metadata", {})
    cases = dataset.get("cases", [])
    if not isinstance(metadata, dict) or not isinstance(cases, list):
        raise AssertionError("Dataset must include object metadata and list cases")

    all_tools = _tool_name_set()
    expected_count = int(metadata.get("expected_tool_count", len(all_tools)))
    expected_hash = str(metadata.get("expected_tool_hash", ""))
    actual_hash = _tool_hash(all_tools)

    assert len(all_tools) == expected_count, (
        f"Tool count mismatch; refresh dataset. expected={expected_count}, actual={len(all_tools)}"
    )
    assert actual_hash == expected_hash, (
        f"Tool hash mismatch; refresh dataset. expected={expected_hash}, actual={actual_hash}"
    )

    model = _resolve_model()
    service = ToolSearchService(default_model=model)
    provider_instance: Optional[Any] = None
    if model is not None and model.provider in {"anthropic", "openrouter", "ollama"}:
        discover_providers()
        provider_instance = create_provider_instance(
            model.provider,
            model=model,
            max_tokens=500,
        )

    tool_limit = _to_int_env("TOOL_DISCOVERY_LIMIT", 0)
    max_results = _to_int_env("TOOL_DISCOVERY_MAX_RESULTS", 10)
    csv_path_raw = os.getenv("TOOL_DISCOVERY_CSV", "").strip()
    csv_path = Path(csv_path_raw) if csv_path_raw else None
    resume = os.getenv("TOOL_DISCOVERY_RESUME", "").strip() == "1"

    confusion_clusters = metadata.get("confusion_clusters", {})
    if not isinstance(confusion_clusters, dict):
        confusion_clusters = {}

    thresholds = metadata.get("thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}

    top1_min = _to_float_env("TOOL_DISCOVERY_TOP1_MIN", float(thresholds.get("top1_min", 0.82)))
    top3_min = _to_float_env("TOOL_DISCOVERY_TOP3_MIN", float(thresholds.get("top3_min", 0.93)))
    confusion_hard_miss_max = _to_float_env(
        "TOOL_DISCOVERY_CONFUSION_HARD_MISS_MAX",
        float(thresholds.get("confusion_hard_miss_max", 0.05)),
    )
    blocked_max = _to_float_env("TOOL_DISCOVERY_BLOCKED_MAX", 0.20)
    infra_blocked_max = _to_float_env("TOOL_DISCOVERY_INFRA_BLOCKED_MAX", 0.40)
    pause_seconds = _to_float_env(
        "TOOL_DISCOVERY_PAUSE_SECONDS",
        2.5 if (model and model.provider in {"anthropic", "openrouter"}) else 0.0,
    )

    rows: List[Dict[str, Any]] = []

    positive_total = 0
    positive_evaluated = 0
    top1_hits = 0
    top3_hits = 0
    mrr_sum = 0.0

    negative_total = 0
    negative_pass = 0

    blocked_total = 0
    infra_blocked_total = 0

    confusion_total = 0
    confusion_top1_wrong_pair = 0
    confusion_hard_miss = 0

    failed_case_ids: List[str] = []

    selected_cases: List[Dict[str, Any]] = [c for c in cases if isinstance(c, dict)]
    if csv_path is not None and resume:
        completed_case_ids = _load_existing_case_ids(csv_path)
        if completed_case_ids:
            selected_cases = [c for c in selected_cases if str(c.get("id", "")) not in completed_case_ids]
    if tool_limit > 0:
        selected_cases = selected_cases[:tool_limit]

    for case_index, case in enumerate(selected_cases):
        if case_index > 0 and pause_seconds > 0:
            time.sleep(pause_seconds)
        case_id = str(case.get("id", "unknown"))
        query = str(case.get("query", "")).strip()
        expected_any = [str(x) for x in case.get("expected_any", []) if isinstance(x, str)]
        forbidden = {str(x) for x in case.get("forbidden", []) if isinstance(x, str)}
        category = str(case.get("category", "uncategorized"))
        cluster = str(case.get("cluster", ""))

        for tool_name in expected_any:
            assert tool_name in all_tools, f"Unknown expected tool '{tool_name}' in {case_id}"

        ranked, search_error = _search_ranked_names_with_model(
            service=service,
            model=model,
            query=query,
            max_results=max_results,
            provider_instance=provider_instance,
        )

        blocked = _is_blocked_error(search_error)
        infra_blocked = _is_infra_error(search_error)
        if blocked:
            blocked_total += 1
        if infra_blocked:
            infra_blocked_total += 1

        top1 = ranked[0] if ranked else ""
        top3 = ranked[:3]
        top5 = ranked[:5]

        if forbidden and any(name in forbidden for name in ranked):
            failed_case_ids.append(case_id)

        if not expected_any:
            negative_total += 1
            case_pass = (not ranked) and (not blocked)
            if case_pass:
                negative_pass += 1
            if infra_blocked:
                status = "infra_blocked"
            else:
                status = "pass" if case_pass else ("blocked" if blocked else "fail")
            if status == "fail":
                failed_case_ids.append(case_id)
        else:
            positive_total += 1
            if infra_blocked:
                status = "infra_blocked"
            elif blocked:
                status = "blocked"
            else:
                positive_evaluated += 1
                rank = _rank_of_expected(ranked, expected_any)
                is_top1 = rank == 1
                is_top3 = rank is not None and rank <= 3
                if is_top1:
                    top1_hits += 1
                if is_top3:
                    top3_hits += 1
                if rank is not None:
                    mrr_sum += 1.0 / rank

                status = "pass" if rank is not None else "fail"
                if status == "fail":
                    failed_case_ids.append(case_id)

                cluster_tools = confusion_clusters.get(cluster, [])
                if isinstance(cluster_tools, list) and cluster_tools:
                    cluster_set = {str(x) for x in cluster_tools if isinstance(x, str)}
                    confusion_total += 1
                    if top1 and (top1 not in expected_any) and (top1 in cluster_set):
                        confusion_top1_wrong_pair += 1
                    if not any(name in expected_any for name in top5):
                        confusion_hard_miss += 1

        row = {
            "case_id": case_id,
            "category": category,
            "cluster": cluster,
            "query": query,
            "expected_any": "|".join(expected_any),
            "top1": top1,
            "top3": "|".join(top3),
            "top5": "|".join(top5),
            "ranked": "|".join(ranked),
            "status": status,
            "error": search_error or "",
        }
        rows.append(row)
        if csv_path is not None:
            _append_csv_row(csv_path, row)

    top1_rate = _safe_rate(top1_hits, positive_evaluated)
    top3_rate = _safe_rate(top3_hits, positive_evaluated)
    mrr = (mrr_sum / positive_evaluated) if positive_evaluated else 0.0
    blocked_rate = _safe_rate(blocked_total, len(selected_cases))
    infra_blocked_rate = _safe_rate(infra_blocked_total, len(selected_cases))
    negative_rate = _safe_rate(negative_pass, negative_total)
    confusion_top1_wrong_rate = _safe_rate(confusion_top1_wrong_pair, confusion_total)
    confusion_hard_miss_rate = _safe_rate(confusion_hard_miss, confusion_total)

    summary = (
        f"Tool discovery benchmark: cases={len(selected_cases)}, "
        f"positive={positive_total}, evaluated={positive_evaluated}, blocked={blocked_total}, infra_blocked={infra_blocked_total}, "
        f"top1={top1_rate:.3f}, top3={top3_rate:.3f}, mrr={mrr:.3f}, "
        f"negative_pass={negative_rate:.3f}, "
        f"confused_top1_wrong_pair={confusion_top1_wrong_rate:.3f}, "
        f"confusion_hard_miss={confusion_hard_miss_rate:.3f}"
    )
    print(summary)

    # Keep end-of-run rewrite only when no incremental output was configured.
    if csv_path is None:
        _write_csv(Path("/tmp/tool_discovery_results.csv"), rows)

    assert positive_evaluated > 0, (
        "No positive cases were evaluated (all may have been blocked). Inspect TOOL_DISCOVERY_CSV output for details."
    )
    assert blocked_rate <= blocked_max, (
        f"Blocked rate too high: {blocked_rate:.3f} > {blocked_max:.3f}. "
        "Review flagged prompts in CSV and reword cases."
    )
    assert infra_blocked_rate <= infra_blocked_max, (
        f"Infrastructure-blocked rate too high: {infra_blocked_rate:.3f} > {infra_blocked_max:.3f}. "
        "Investigate provider limits/credits/model access."
    )
    assert top1_rate >= top1_min, (
        f"Top-1 accuracy below threshold: {top1_rate:.3f} < {top1_min:.3f}. "
        f"Sample failing cases: {failed_case_ids[:12]}"
    )
    assert top3_rate >= top3_min, (
        f"Top-3 accuracy below threshold: {top3_rate:.3f} < {top3_min:.3f}. "
        f"Sample failing cases: {failed_case_ids[:12]}"
    )
    assert confusion_hard_miss_rate <= confusion_hard_miss_max, (
        f"Confusion hard-miss rate too high: {confusion_hard_miss_rate:.3f} > {confusion_hard_miss_max:.3f}"
    )
