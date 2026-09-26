"""Scenario run reports: results.jsonl as the run goes, results.json and summary.md at the end.

Also ``regrade``: re-run every check against the stored states of a results file
with the current checker and scenario files, with no browser.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TextIO

from cli.scenarios.checks import CheckResult
from cli.scenarios.grade import ScenarioGrader, StepRecordData
from cli.scenarios.model import Catalogue, Scenario

CHECK_STATUSES = ("pass", "fail", "error", "xfail", "xpass", "warn", "skip")
# Scenario outcomes, worst first.
SCENARIO_STATUSES = ("error", "fail", "xpass", "xfail", "pass", "skipped")


@dataclass
class ScenarioOutcome:
    """Everything recorded for one scenario."""

    scenario: Scenario
    steps: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0
    infra_error: Optional[str] = None
    skipped_reason: Optional[str] = None
    attempts: int = 1

    def results(self) -> list[dict[str, Any]]:
        return [result for step in self.steps for result in step.get("results", [])]

    def counts(self) -> dict[str, int]:
        counts = {status: 0 for status in CHECK_STATUSES}
        for result in self.results():
            counts[result["status"]] = counts.get(result["status"], 0) + 1
        return counts

    @property
    def status(self) -> str:
        if self.skipped_reason:
            return "skipped"
        if self.infra_error:
            return "error"
        counts = self.counts()
        if counts["fail"] or counts["error"]:
            return "fail"
        if counts["xpass"]:
            return "xpass"
        if counts["xfail"]:
            return "xfail"
        return "pass"

    @property
    def unexpected(self) -> bool:
        return self.status in ("fail", "error")

    def xpassed_bugs(self) -> list[str]:
        return sorted({str(r.get("known")) for r in self.results() if r["status"] == "xpass"})

    def to_dict(self) -> dict[str, Any]:
        scenario = self.scenario
        return {
            "id": scenario.id,
            "title": scenario.title,
            "area": scenario.area,
            "smoke": scenario.smoke,
            "tags": scenario.tags,
            "known": scenario.known,
            "status": self.status,
            "counts": self.counts(),
            "duration_s": round(self.duration_s, 3),
            "attempts": self.attempts,
            "infra_error": self.infra_error,
            "skipped_reason": self.skipped_reason,
            "steps": self.steps,
        }


class ResultSink:
    """Appends step records to results.jsonl as they finish, and writes the final reports."""

    def __init__(self, out_dir: Path, config: dict[str, Any]) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.failures_dir = out_dir / "failures"
        self.config = config
        self.started = time.time()
        self.outcomes: list[ScenarioOutcome] = []
        self._jsonl: Optional[TextIO] = open(out_dir / "results.jsonl", "a", encoding="utf-8")

    def record_step(self, scenario_id: str, record: dict[str, Any]) -> None:
        if self._jsonl is None:
            return
        line = dict(record)
        line["scenario"] = scenario_id
        self._jsonl.write(json.dumps(line, default=str) + "\n")
        self._jsonl.flush()

    def add(self, outcome: ScenarioOutcome) -> None:
        self.outcomes.append(outcome)

    def close(self, catalogue: Catalogue, interrupted: bool = False) -> dict[str, Any]:
        """Write results.json and summary.md; returns the summary numbers."""
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None
        duration = time.time() - self.started
        summary = summarize(self.outcomes)
        summary["duration_s"] = round(duration, 1)
        summary["interrupted"] = interrupted
        payload = {
            "config": self.config,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started)),
            "duration_s": round(duration, 1),
            "interrupted": interrupted,
            "summary": summary,
            "scenarios": [outcome.to_dict() for outcome in self.outcomes],
        }
        (self.out_dir / "results.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        (self.out_dir / "summary.md").write_text(
            render_summary(self.outcomes, summary, catalogue, self.config, self.out_dir), encoding="utf-8"
        )
        return summary


def summarize(outcomes: list[ScenarioOutcome]) -> dict[str, Any]:
    """Counts of scenario outcomes and check statuses, plus the lists a reader needs."""
    scenario_counts = {status: 0 for status in SCENARIO_STATUSES}
    check_counts = {status: 0 for status in CHECK_STATUSES}
    for outcome in outcomes:
        scenario_counts[outcome.status] += 1
        for status, count in outcome.counts().items():
            check_counts[status] = check_counts.get(status, 0) + count
    return {
        "scenarios": len(outcomes),
        "scenario_counts": scenario_counts,
        "check_counts": check_counts,
        "unexpected": [o.scenario.id for o in outcomes if o.unexpected],
        "xpass": {o.scenario.id: o.xpassed_bugs() for o in outcomes if o.xpassed_bugs()},
        "exit_code": 1 if any(o.unexpected for o in outcomes) else 0,
    }


def _row(cells: list[Any]) -> str:
    return "| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |"


def render_summary(
    outcomes: list[ScenarioOutcome],
    summary: dict[str, Any],
    catalogue: Catalogue,
    config: dict[str, Any],
    out_dir: Path,
) -> str:
    """The Markdown summary: totals, per-area table, scenarios, failures, xpasses and known bugs."""
    lines = [f"# Scenario run ({config.get('mode', 'replay')})", ""]
    if summary.get("interrupted"):
        lines += ["**Interrupted**: partial results.", ""]
    sc, cc = summary["scenario_counts"], summary["check_counts"]
    lines += [
        f"- Scenarios: {summary['scenarios']} ({sc['pass']} pass, {sc['xfail']} xfail, {sc['xpass']} xpass, "
        f"{sc['fail']} fail, {sc['error']} error, {sc['skipped']} skipped)",
        f"- Checks: {cc['pass']} pass, {cc['xfail']} xfail, {cc['xpass']} xpass, {cc['fail']} fail, "
        f"{cc['error']} error, {cc['warn']} warn, {cc['skip']} skip",
        f"- Run time: {summary.get('duration_s', 0)} s",
        f"- Result: {'unexpected failures' if summary['exit_code'] else 'no unexpected failures'}",
        "",
    ]
    smoke = [o for o in outcomes if o.scenario.smoke]
    if smoke and len(smoke) != len(outcomes):
        smoke_bad = [o.scenario.id for o in smoke if o.unexpected]
        lines += [f"- Smoke subset: {len(smoke)} scenarios, unexpected failures: {', '.join(smoke_bad) or 'none'}", ""]
    lines += ["## By area", "", _row(["Area", "Scenarios", "pass", "xfail", "xpass", "fail", "error", "skipped"])]
    lines.append(_row(["---"] * 8))
    areas: dict[str, list[ScenarioOutcome]] = {}
    for outcome in outcomes:
        areas.setdefault(outcome.scenario.area, []).append(outcome)
    for area, items in areas.items():
        statuses = [o.status for o in items]
        lines.append(
            _row(
                [area, len(items)] + [statuses.count(s) for s in ("pass", "xfail", "xpass", "fail", "error", "skipped")]
            )
        )
    lines += [
        "",
        "## Scenarios",
        "",
        _row(["Scenario", "Status", "Checks (pass/xfail/xpass/fail)", "Known", "Time (s)"]),
    ]
    lines.append(_row(["---"] * 5))
    for outcome in outcomes:
        c = outcome.counts()
        title = f"{outcome.scenario.id} {outcome.scenario.title}" + (" (smoke)" if outcome.scenario.smoke else "")
        lines.append(
            _row(
                [
                    title,
                    outcome.status,
                    f"{c['pass']}/{c['xfail']}/{c['xpass']}/{c['fail'] + c['error']}",
                    ", ".join(outcome.scenario.known) or "-",
                    f"{outcome.duration_s:.1f}",
                ]
            )
        )
    failures = _failure_lines(outcomes, out_dir)
    if failures:
        lines += ["", "## Unexpected failures", ""] + failures
    if summary["xpass"]:
        lines += ["", "## Unexpected passes (fixed?)", ""]
        for scenario_id, bugs in summary["xpass"].items():
            lines.append(
                f"- {scenario_id}: fixed? {', '.join(bugs)} (remove the known marks when the fix is confirmed)"
            )
    known_counts: dict[str, int] = {}
    for outcome in outcomes:
        for result in outcome.results():
            if result["status"] == "xfail":
                known_counts[str(result.get("known"))] = known_counts.get(str(result.get("known")), 0) + 1
    if known_counts:
        lines += ["", "## Known bugs hit (xfail)", "", _row(["Bug", "Expected failures", "Title"]), _row(["---"] * 3)]
        for bug in sorted(known_counts, key=lambda k: int(k[1:]) if k[1:].isdigit() else 0):
            lines.append(_row([bug, known_counts[bug], catalogue.bugs.get(bug, "")]))
    lines.append("")
    return "\n".join(lines)


def _failure_lines(outcomes: list[ScenarioOutcome], out_dir: Path) -> list[str]:
    lines: list[str] = []
    for outcome in outcomes:
        if outcome.infra_error:
            lines.append(f"- **{outcome.scenario.id}**: infrastructure error: {outcome.infra_error}")
        for step in outcome.steps:
            bad = [r for r in step.get("results", []) if r["status"] in ("fail", "error")]
            if not bad and not step.get("error"):
                continue
            artifacts = step.get("artifacts") or {}
            links = ", ".join(
                f"[{kind}]({Path(path).relative_to(out_dir).as_posix()})" for kind, path in artifacts.items()
            )
            header = f"- **{outcome.scenario.id} / {step.get('step')}**" + (f" ({links})" if links else "")
            lines.append(header)
            if step.get("error"):
                lines.append(f"  - step error: {step['error']}")
            for result in bad:
                detail = result.get("message", "")
                lines.append(f"  - {result['id']} `{result['name']}` [{result['status']}]: {detail}")
    return lines


# ----------------------------------------------------------------------
# Regrade
# ----------------------------------------------------------------------


def regrade(results_path: Path, catalogue: Catalogue) -> tuple[dict[str, Any], Path]:
    """Re-run every check on the stored states of ``results_path``.

    Writes ``results_regraded.json`` and ``summary_regraded.md`` next to it.

    Returns:
        (summary, path of the regraded results).
    """
    data = json.loads(results_path.read_text(encoding="utf-8"))
    by_id = {scenario.id: scenario for scenario in catalogue.scenarios}
    config = dict(data.get("config") or {})
    config["regraded_from"] = str(results_path)
    mode = str(config.get("mode", "replay"))
    outcomes: list[ScenarioOutcome] = []
    for stored in data.get("scenarios", []):
        scenario = by_id.get(stored.get("id"))
        if scenario is None:
            continue
        outcome = ScenarioOutcome(
            scenario,
            duration_s=float(stored.get("duration_s") or 0.0),
            infra_error=stored.get("infra_error"),
            skipped_reason=stored.get("skipped_reason"),
            attempts=int(stored.get("attempts") or 1),
        )
        if not outcome.skipped_reason:
            outcome.steps = regrade_steps(scenario, stored.get("steps", []), catalogue.waivers_for(scenario), mode)
        outcomes.append(outcome)
    summary = summarize(outcomes)
    out_dir = results_path.parent
    payload = {"config": config, "summary": summary, "scenarios": [o.to_dict() for o in outcomes]}
    target = out_dir / "results_regraded.json"
    target.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    (out_dir / "summary_regraded.md").write_text(
        render_summary(outcomes, summary, catalogue, config, out_dir), encoding="utf-8"
    )
    return summary, target


def regrade_steps(
    scenario: Scenario, records: list[dict[str, Any]], waivers: dict[str, str], mode: str
) -> list[dict[str, Any]]:
    """Grade stored step records again, in order, with the current scenario definition."""
    grader = ScenarioGrader(scenario, waivers, mode)
    steps_by_id = {step.id: step for step in scenario.steps}
    regraded: list[dict[str, Any]] = []
    for record in records:
        record = dict(record)
        step_id = str(record.get("step"))
        data = StepRecordData.from_record(record)
        if step_id == "start":
            grader.start(data)
            record["results"] = []
        elif step_id == "setup" or step_id in steps_by_id:
            results = grader.grade(step_id, steps_by_id.get(step_id), data)
            record["results"] = [result.to_dict() for result in results]
        else:
            record["results"] = [
                CheckResult(f"{step_id}.regrade", "check", "regrade", False, error=True,
                            message=f"step {step_id} is no longer in scenario {scenario.id}").to_dict()
            ]  # fmt: skip
        regraded.append(record)
    return regraded
