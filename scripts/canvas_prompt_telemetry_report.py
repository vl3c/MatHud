#!/usr/bin/env python3
"""
Build summary/CSV reports from `canvas_prompt_telemetry` log entries.

Usage examples:
  python scripts/canvas_prompt_telemetry_report.py
  python scripts/canvas_prompt_telemetry_report.py --log-file logs/mathud_session_26_02_11.log
  python scripts/canvas_prompt_telemetry_report.py --mode hybrid --csv-out /tmp/telemetry.csv
  python scripts/canvas_prompt_telemetry_report.py --json-out /tmp/telemetry_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


LOG_MARKER = "canvas_prompt_telemetry "


@dataclass
class Report:
    rows: List[Dict[str, Any]]
    grouped: Dict[str, Dict[str, Any]]
    totals: Dict[str, Any]


def _default_log_file() -> Optional[Path]:
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None
    candidates = sorted(logs_dir.glob("mathud_session_*.log"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def _read_rows(log_file: Path, modes: Optional[set[str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        marker_idx = line.find(LOG_MARKER)
        if marker_idx < 0:
            continue
        payload_text = line[marker_idx + len(LOG_MARKER):].strip()
        if not payload_text:
            continue
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        mode = str(payload.get("mode", "")).lower()
        if modes and mode not in modes:
            continue
        rows.append(payload)
    return rows


def _avg(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return round(mean(vals), 2)


def _build_report(rows: List[Dict[str, Any]]) -> Report:
    grouped_rows: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mode = str(row.get("mode", "unknown"))
        prompt_kind = str(row.get("prompt_kind", "unknown"))
        includes = str(row.get("includes_full_state", "null"))
        grouped_rows[(mode, prompt_kind, includes)].append(row)

    grouped: Dict[str, Dict[str, Any]] = {}
    for (mode, prompt_kind, includes), items in grouped_rows.items():
        key = f"{mode}|{prompt_kind}|includes_full_state={includes}"
        grouped[key] = {
            "count": len(items),
            "avg_input_tokens": _avg(float(i.get("input_estimated_tokens", 0)) for i in items),
            "avg_output_tokens": _avg(float(i.get("output_payload_estimated_tokens", 0)) for i in items),
            "avg_reduction_pct": _avg(float(i.get("reduction_pct", 0.0)) for i in items),
            "avg_normalize_ms": _avg(float(i.get("normalize_elapsed_ms", 0.0)) for i in items),
        }

    totals = {
        "count": len(rows),
        "avg_input_tokens": _avg(float(i.get("input_estimated_tokens", 0)) for i in rows),
        "avg_output_tokens": _avg(float(i.get("output_payload_estimated_tokens", 0)) for i in rows),
        "avg_reduction_pct": _avg(float(i.get("reduction_pct", 0.0)) for i in rows),
        "avg_normalize_ms": _avg(float(i.get("normalize_elapsed_ms", 0.0)) for i in rows),
    }

    return Report(rows=rows, grouped=grouped, totals=totals)


def _write_csv(rows: List[Dict[str, Any]], csv_out: Path) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "prompt_kind",
        "includes_full_state",
        "input_bytes",
        "normalized_prompt_bytes",
        "output_payload_bytes",
        "input_estimated_tokens",
        "normalized_prompt_estimated_tokens",
        "output_payload_estimated_tokens",
        "reduction_pct",
        "normalize_elapsed_ms",
    ]
    with csv_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _print_report(report: Report, log_file: Path) -> None:
    print(f"log_file: {log_file}")
    print(f"rows: {report.totals['count']}")
    print(
        "totals:"
        f" avg_input_tokens={report.totals['avg_input_tokens']},"
        f" avg_output_tokens={report.totals['avg_output_tokens']},"
        f" avg_reduction_pct={report.totals['avg_reduction_pct']}%,"
        f" avg_normalize_ms={report.totals['avg_normalize_ms']}"
    )
    if not report.grouped:
        return
    print("groups:")
    for key, group in sorted(report.grouped.items()):
        print(
            f"  - {key}:"
            f" count={group['count']},"
            f" avg_input_tokens={group['avg_input_tokens']},"
            f" avg_output_tokens={group['avg_output_tokens']},"
            f" avg_reduction_pct={group['avg_reduction_pct']}%,"
            f" avg_normalize_ms={group['avg_normalize_ms']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize canvas_prompt_telemetry logs.")
    parser.add_argument("--log-file", type=Path, help="Path to a MatHud log file.")
    parser.add_argument(
        "--mode",
        action="append",
        help="Filter by mode (off, hybrid, summary_only). Can be supplied multiple times.",
    )
    parser.add_argument("--csv-out", type=Path, help="Optional CSV output path.")
    parser.add_argument("--json-out", type=Path, help="Optional JSON summary output path.")
    args = parser.parse_args()

    log_file = args.log_file or _default_log_file()
    if log_file is None:
        print("No log file found. Pass --log-file explicitly.")
        return 1
    if not log_file.exists():
        print(f"Log file does not exist: {log_file}")
        return 1

    modes = {m.strip().lower() for m in (args.mode or []) if m and m.strip()}
    rows = _read_rows(log_file, modes if modes else None)
    report = _build_report(rows)

    _print_report(report, log_file)

    if args.csv_out:
        _write_csv(report.rows, args.csv_out)
        print(f"csv_out: {args.csv_out}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "totals": report.totals,
            "groups": report.grouped,
            "rows": report.rows,
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
