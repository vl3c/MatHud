#!/usr/bin/env python3
"""Compare local vs API tool search accuracy and latency.

Runs the benchmark dataset against both search modes and outputs a
side-by-side comparison table with disagreement analysis.

Usage::

    # Local only (no API key needed)
    python scripts/compare_search_modes.py --modes local

    # Full comparison (needs API key)
    python scripts/compare_search_modes.py --modes local,api

    # Save disagreements to CSV
    python scripts/compare_search_modes.py --modes local,api --disagreements /tmp/disagreements.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from static.tool_search_service import ToolSearchService, clear_search_cache

DATASET_PATH = Path("server_tests/data/tool_discovery_cases.yaml")


def _load_dataset() -> Dict[str, Any]:
    raw = DATASET_PATH.read_text(encoding="utf-8")
    parsed: Dict[str, Any] = json.loads(raw)
    return parsed


def _get_tool_name(tool: Dict[str, Any]) -> str:
    name: str = tool.get("function", {}).get("name", "")
    return name


def _run_mode(
    mode: str,
    cases: List[Dict[str, Any]],
    max_results: int = 10,
) -> Dict[str, Any]:
    """Run benchmark for a single search mode."""
    os.environ["TOOL_SEARCH_MODE"] = mode
    clear_search_cache()

    service = ToolSearchService.__new__(ToolSearchService)
    service._client = None
    service._client_initialized = False
    service.default_model = None
    service.last_error = None

    # For API mode, initialize the client properly
    if mode == "api":
        try:
            service = ToolSearchService()
        except ValueError:
            print(f"  [SKIP] API mode requires OPENAI_API_KEY")
            return {"skipped": True}

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    evaluated = 0
    latencies: List[float] = []
    case_results: List[Dict[str, Any]] = []

    for case in cases:
        expected_any = [str(x) for x in case.get("expected_any", []) if isinstance(x, str)]
        if not expected_any:
            continue

        query = str(case.get("query", "")).strip()
        evaluated += 1

        start = time.perf_counter()
        if mode == "local":
            results = service.search_tools_local(query, max_results)
        else:
            results = service.search_tools(query, max_results=max_results)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        ranked = [_get_tool_name(t) for t in results]
        expected_set = set(expected_any)

        top1 = ranked[0] if ranked else ""
        is_top1 = top1 in expected_set
        is_top3 = bool(expected_set & set(ranked[:3]))
        is_top5 = bool(expected_set & set(ranked[:5]))

        if is_top1:
            top1_hits += 1
        if is_top3:
            top3_hits += 1
        if is_top5:
            top5_hits += 1

        case_results.append({
            "id": case.get("id", "?"),
            "query": query,
            "expected": expected_any,
            "top1": top1,
            "top3": ranked[:3],
            "is_top1": is_top1,
            "is_top3": is_top3,
            "is_top5": is_top5,
        })

        # Rate-limit API calls
        if mode == "api":
            time.sleep(0.5)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1) if latencies else 0
    p99 = latencies[p99_idx] if latencies else 0

    return {
        "skipped": False,
        "evaluated": evaluated,
        "top1_hits": top1_hits,
        "top3_hits": top3_hits,
        "top5_hits": top5_hits,
        "top1_rate": top1_hits / evaluated if evaluated else 0,
        "top3_rate": top3_hits / evaluated if evaluated else 0,
        "top5_rate": top5_hits / evaluated if evaluated else 0,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50,
        "p99_latency_ms": p99,
        "case_results": case_results,
    }


def _find_disagreements(
    local_results: Dict[str, Any],
    api_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Find cases where local and API disagree."""
    disagreements: List[Dict[str, Any]] = []
    local_cases = local_results.get("case_results", [])
    api_cases = api_results.get("case_results", [])

    api_by_id = {c["id"]: c for c in api_cases}

    for lc in local_cases:
        ac = api_by_id.get(lc["id"])
        if ac is None:
            continue

        if lc["is_top1"] != ac["is_top1"]:
            disagreements.append({
                "id": lc["id"],
                "query": lc["query"],
                "expected": lc["expected"],
                "local_top1": lc["top1"],
                "api_top1": ac["top1"],
                "local_correct": lc["is_top1"],
                "api_correct": ac["is_top1"],
                "winner": "local" if lc["is_top1"] else "api",
            })

    return disagreements


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare tool search modes")
    parser.add_argument(
        "--modes",
        default="local",
        help="Comma-separated search modes to compare (local,api)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Max results per search (default: 10)",
    )
    parser.add_argument(
        "--disagreements",
        default="",
        help="Path to write disagreements CSV",
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        print("No modes specified")
        return 1

    dataset = _load_dataset()
    cases = dataset.get("cases", [])

    print(f"Dataset: {len(cases)} cases")
    print()

    all_results: Dict[str, Dict[str, Any]] = {}

    for mode in modes:
        print(f"Running mode: {mode}")
        result = _run_mode(mode, cases, args.max_results)
        all_results[mode] = result

        if result.get("skipped"):
            print(f"  Skipped (missing credentials)")
            continue

        print(f"  Evaluated: {result['evaluated']}")
        print(f"  Top-1: {result['top1_hits']}/{result['evaluated']} = {result['top1_rate']:.3f}")
        print(f"  Top-3: {result['top3_hits']}/{result['evaluated']} = {result['top3_rate']:.3f}")
        print(f"  Top-5: {result['top5_hits']}/{result['evaluated']} = {result['top5_rate']:.3f}")
        print(f"  Latency: avg={result['avg_latency_ms']:.1f}ms, "
              f"p50={result['p50_latency_ms']:.1f}ms, p99={result['p99_latency_ms']:.1f}ms")
        print()

    # Side-by-side comparison
    if len(all_results) >= 2:
        active = {k: v for k, v in all_results.items() if not v.get("skipped")}
        if len(active) >= 2:
            print("=" * 60)
            print("Side-by-Side Comparison")
            print("=" * 60)
            header = f"{'Metric':<20}"
            for mode in active:
                header += f" {mode:>15}"
            print(header)
            print("-" * 60)

            for metric in ["top1_rate", "top3_rate", "top5_rate", "avg_latency_ms", "p50_latency_ms", "p99_latency_ms"]:
                row = f"{metric:<20}"
                for mode in active:
                    val = active[mode].get(metric, 0)
                    if "rate" in metric:
                        row += f" {val:>14.3f}"
                    else:
                        row += f" {val:>13.1f}ms"
                print(row)
            print()

    # Disagreement analysis
    if "local" in all_results and "api" in all_results:
        local_r = all_results["local"]
        api_r = all_results["api"]
        if not local_r.get("skipped") and not api_r.get("skipped"):
            disagreements = _find_disagreements(local_r, api_r)
            local_wins = [d for d in disagreements if d["winner"] == "local"]
            api_wins = [d for d in disagreements if d["winner"] == "api"]

            print(f"Disagreements: {len(disagreements)} total")
            print(f"  Local wins: {len(local_wins)}")
            print(f"  API wins:   {len(api_wins)}")

            if api_wins:
                print(f"\nCases where API is right but local is wrong (tuning opportunities):")
                for d in api_wins[:10]:
                    print(f"  {d['id']}: {d['query']!r}")
                    print(f"    expected={d['expected']}, local={d['local_top1']!r}, api={d['api_top1']!r}")

            if local_wins:
                print(f"\nCases where local is right but API is wrong (local advantages):")
                for d in local_wins[:10]:
                    print(f"  {d['id']}: {d['query']!r}")
                    print(f"    expected={d['expected']}, local={d['local_top1']!r}, api={d['api_top1']!r}")

            # Write disagreements CSV
            if args.disagreements and disagreements:
                dis_path = Path(args.disagreements)
                dis_path.parent.mkdir(parents=True, exist_ok=True)
                with dis_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "id", "query", "expected", "local_top1", "api_top1",
                        "local_correct", "api_correct", "winner",
                    ])
                    writer.writeheader()
                    for d in disagreements:
                        csv_row: Dict[str, Any] = dict(d)
                        csv_row["expected"] = "|".join(csv_row["expected"])
                        writer.writerow(csv_row)
                print(f"\nDisagreements written to: {dis_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
