#!/usr/bin/env python3
"""Run the live tool discovery benchmark once and emit CSV + console summary."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live tool discovery benchmark")
    parser.add_argument("--model", default="", help="Optional model id override for search (e.g. gpt-4.1-mini)")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for number of cases")
    parser.add_argument(
        "--csv",
        default="/tmp/tool_discovery_results.csv",
        help="Path to CSV output",
    )
    parser.add_argument("--top1-min", type=float, default=None, help="Override top-1 threshold")
    parser.add_argument("--top3-min", type=float, default=None, help="Override top-3 threshold")
    parser.add_argument(
        "--confusion-hard-miss-max",
        type=float,
        default=None,
        help="Override confusion hard-miss threshold",
    )
    parser.add_argument(
        "--infra-blocked-max",
        type=float,
        default=None,
        help="Override maximum allowed infra-blocked rate.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=None,
        help="Pause between benchmark requests (helps avoid provider TPM limits).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Relax thresholds for quick wiring/proxy checks (not quality gating).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing CSV by skipping case_ids already recorded.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    csv_path = Path(args.csv)
    env["RUN_LIVE_TOOL_DISCOVERY"] = "1"
    env["TOOL_DISCOVERY_CSV"] = str(csv_path)

    if args.model:
        env["TOOL_DISCOVERY_MODEL"] = args.model
    if args.limit > 0:
        env["TOOL_DISCOVERY_LIMIT"] = str(args.limit)
    if args.top1_min is not None:
        env["TOOL_DISCOVERY_TOP1_MIN"] = str(args.top1_min)
    if args.top3_min is not None:
        env["TOOL_DISCOVERY_TOP3_MIN"] = str(args.top3_min)
    if args.confusion_hard_miss_max is not None:
        env["TOOL_DISCOVERY_CONFUSION_HARD_MISS_MAX"] = str(args.confusion_hard_miss_max)
    if args.infra_blocked_max is not None:
        env["TOOL_DISCOVERY_INFRA_BLOCKED_MAX"] = str(args.infra_blocked_max)
    if args.pause_seconds is not None:
        env["TOOL_DISCOVERY_PAUSE_SECONDS"] = str(args.pause_seconds)
    if args.smoke:
        env["TOOL_DISCOVERY_TOP1_MIN"] = "0.0"
        env["TOOL_DISCOVERY_TOP3_MIN"] = "0.0"
        env["TOOL_DISCOVERY_CONFUSION_HARD_MISS_MAX"] = "1.0"
        env["TOOL_DISCOVERY_BLOCKED_MAX"] = "1.0"
        env["TOOL_DISCOVERY_INFRA_BLOCKED_MAX"] = "1.0"
    if args.resume:
        env["TOOL_DISCOVERY_RESUME"] = "1"
    elif csv_path.exists():
        csv_path.unlink()

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "live_tool_discovery",
        "server_tests/test_tool_discovery_live.py",
        "-q",
        "-s",
    ]

    print("Running:", " ".join(cmd))
    print("CSV output:", env["TOOL_DISCOVERY_CSV"])
    result = subprocess.run(cmd, env=env)

    csv_path = Path(env["TOOL_DISCOVERY_CSV"])
    if csv_path.exists():
        summary_cmd = [
            sys.executable,
            "scripts/summarize_tool_discovery_results.py",
            str(csv_path),
        ]
        print("Summary:", " ".join(summary_cmd))
        subprocess.run(summary_cmd, env=env)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
