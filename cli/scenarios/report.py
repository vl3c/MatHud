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
from cli.scenarios.classify import (
    CLASSES,
    DEFAULT_MAX_INFRA_RATE,
    FAILING_CLASSES,
    annotate_steps,
    apply_batch_verdicts,
    infra_rate,
    retrace_invariant_failures,
    step_classes,
)
from cli.scenarios.grade import ScenarioGrader, StepRecordData
from cli.scenarios.model import Catalogue, Scenario

CHECK_STATUSES = ("pass", "fail", "error", "xfail", "xpass", "warn", "skip", "unrecorded")
# Scenario outcomes, worst first. "waived" means the only expected failures are
# waived invariants; "xfail" means at least one check marked known failed.
SCENARIO_STATUSES = ("error", "fail", "xpass", "xfail", "waived", "pass", "skipped")


@dataclass
class ScenarioOutcome:
    """Everything recorded for one scenario."""

    scenario: Scenario
    steps: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0
    infra_error: Optional[str] = None
    skipped_reason: Optional[str] = None
    attempts: int = 1
    # Live and retrace runs: who answered, which repeat, and the retrace comparison.
    provider: Optional[str] = None
    model: Optional[str] = None
    repeat: int = 1
    retrace: Optional[dict[str, Any]] = None

    @property
    def label(self) -> str:
        """The scenario id, with the model and repeat when the run has them."""
        if self.model is None:
            return self.scenario.id
        return f"{self.scenario.id} [{self.model} #{self.repeat}]"

    def classes(self) -> set[str]:
        """Failure classes of this run (section 4.7); ``infra`` when it could not finish."""
        found = step_classes(self.steps)
        if self.infra_error:
            found.add("infra")
        if self.retrace:
            if self.retrace.get("reproduced") is False:
                found.add("nondeterministic")
            if self.retrace.get("invariant_failures"):
                # Retrace judges every batch on its own canvas: an invariant it breaks is an app bug.
                found.add("app")
        return found

    def annotate(self, mode: str) -> None:
        """Classify every failing result (sets ``class`` on the step records)."""
        annotate_steps(self.steps, mode, self.retrace)

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
        if any(r["status"] == "xfail" and r.get("kind") != "invariant" for r in self.results()):
            return "xfail"
        if counts["xfail"]:
            return "waived"
        return "pass"

    @property
    def unexpected(self) -> bool:
        return self.status in ("fail", "error")

    def app_failure(self) -> bool:
        """Live and retrace runs fail on app bugs and irreproducible canvases, not on model mistakes."""
        return bool(self.classes() & FAILING_CLASSES)

    def used_waivers(self) -> set[tuple[str, str]]:
        """``(invariant, bug)`` pairs whose waiver turned a failure into xfail here."""
        return {
            (str(r["name"]), str(r.get("known")))
            for r in self.results()
            if r.get("kind") == "invariant" and r["status"] == "xfail"
        }

    def xpassed_bugs(self) -> list[str]:
        return sorted({str(r.get("known")) for r in self.results() if r["status"] == "xpass"})

    def to_dict(self) -> dict[str, Any]:
        scenario = self.scenario
        data: dict[str, Any] = {
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
            "classes": sorted(self.classes()),
            "steps": self.steps,
        }
        if self.model is not None:
            data.update(provider=self.provider, model=self.model, repeat=self.repeat, retrace=self.retrace)
        return data


class ResultSink:
    """Appends step records to results.jsonl as they finish, and writes the final reports."""

    def __init__(self, out_dir: Path, config: dict[str, Any]) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.failures_dir = out_dir / "failures"
        self.config = config
        self.started = time.time()
        self.outcomes: list[ScenarioOutcome] = []
        # A new run starts a new file: never append to an earlier run's records.
        self._jsonl: Optional[TextIO] = open(out_dir / "results.jsonl", "w", encoding="utf-8")

    def record_step(self, scenario_id: str, record: dict[str, Any]) -> None:
        if self._jsonl is None:
            return
        line = dict(record)
        line["scenario"] = scenario_id
        self._jsonl.write(json.dumps(line, default=str) + "\n")
        self._jsonl.flush()

    def discard_attempt(self, scenario_id: str, attempt: int, reason: str) -> None:
        """Mark the records of an abandoned attempt in results.jsonl (results.json keeps the final attempt only)."""
        self.record_step(scenario_id, {"kind": "attempt_discarded", "attempt": attempt, "reason": reason})

    def add(self, outcome: ScenarioOutcome) -> None:
        self.outcomes.append(outcome)

    def close(self, catalogue: Catalogue, interrupted: bool = False) -> dict[str, Any]:
        """Write results.json and summary.md; returns the summary numbers."""
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None
        duration = time.time() - self.started
        mode = str(self.config.get("mode", "replay"))
        for outcome in self.outcomes:
            outcome.annotate(mode)
        summary = summarize(
            self.outcomes, catalogue.invariant_waivers, mode, self.config.get("max_infra_rate", DEFAULT_MAX_INFRA_RATE)
        )
        summary["duration_s"] = round(duration, 1)
        summary["interrupted"] = interrupted
        if self.config.get("stopped"):
            summary["stopped"] = self.config["stopped"]
            summary["exit_code"] = 1
        if self.config.get("attached_desktop"):
            summary["attached_desktop"] = {
                "url": self.config.get("desktop_app_url"),
                "pins_applied": False,
                "unpinned_settings": self.config.get("unpinned_settings") or [],
                "fit_view": bool(self.config.get("fit_view")),
            }
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


def summarize(
    outcomes: list[ScenarioOutcome],
    global_waivers: Optional[dict[str, str]] = None,
    mode: str = "replay",
    max_infra_rate: Optional[float] = DEFAULT_MAX_INFRA_RATE,
) -> dict[str, Any]:
    """Counts of scenario outcomes and check statuses, plus the lists a reader needs.

    In replay any ``fail`` or ``error`` is unexpected. Live and retrace runs
    measure a model, so only ``app`` and ``nondeterministic`` failures are
    unexpected there; ``classes`` counts the runs in each failure class and
    ``models`` holds the per-model figures. A live run also fails when more
    than ``max_infra_rate`` of its runs (or all of them) had an infrastructure
    failure: an app regression that breaks every turn must not pass as infra.
    """
    scenario_counts = {status: 0 for status in SCENARIO_STATUSES}
    check_counts = {status: 0 for status in CHECK_STATUSES}
    class_counts = {klass: 0 for klass in CLASSES}
    for outcome in outcomes:
        scenario_counts[outcome.status] += 1
        for status, count in outcome.counts().items():
            check_counts[status] = check_counts.get(status, 0) + count
        for klass in outcome.classes():
            class_counts[klass] = class_counts.get(klass, 0) + 1
    if mode == "replay":
        unexpected = [o.label for o in outcomes if o.unexpected]
    else:
        unexpected = [o.label for o in outcomes if o.app_failure()]
    summary: dict[str, Any] = {
        "scenarios": len(outcomes),
        "scenario_counts": scenario_counts,
        "check_counts": check_counts,
        "classes": class_counts,
        "unexpected": unexpected,
        "xpass": {o.label: o.xpassed_bugs() for o in outcomes if o.xpassed_bugs()},
        "unused_waivers": unused_waivers(outcomes, global_waivers or {}),
        "unrecorded": [o.label for o in outcomes if o.counts().get("unrecorded")],
        "exit_code": 1 if unexpected else 0,
    }
    if mode == "live":
        rate = infra_rate([o.classes() for o in outcomes if not o.skipped_reason])
        summary["infra_rate"] = rate
        limit = DEFAULT_MAX_INFRA_RATE if max_infra_rate is None else float(max_infra_rate)
        if rate is not None and (rate > limit or rate == 1.0):
            summary["infra_rate_exceeded"] = limit
            summary["exit_code"] = 1
    if mode == "live" and any(o.model is not None for o in outcomes):
        summary["models"] = model_summaries(outcomes)
    return summary


def _mean(values: list[Any]) -> Optional[float]:
    numbers = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(sum(numbers) / len(numbers), 3) if numbers else None


def _rate(passed: int, total: int) -> Optional[float]:
    return round(passed / total, 4) if total else None


def model_summaries(outcomes: list[ScenarioOutcome]) -> dict[str, dict[str, Any]]:
    """Per model: scenario statuses, outcome and invariant pass rates, classes and turn figures.

    Pass rates leave out runs with an infrastructure failure (like the
    benchmark's request errors); turn figures average over every live turn.
    """
    by_model: dict[str, list[ScenarioOutcome]] = {}
    for outcome in outcomes:
        by_model.setdefault(str(outcome.model), []).append(outcome)
    summaries: dict[str, dict[str, Any]] = {}
    for model, items in by_model.items():
        graded = [o for o in items if "infra" not in o.classes() and not o.skipped_reason]
        outcome_results = [r for o in graded for r in o.results() if r.get("kind") != "invariant"]
        invariant_results = [r for o in graded for r in o.results() if r.get("kind") == "invariant"]
        outcome_graded = [r for r in outcome_results if r["status"] not in ("skip", "unrecorded")]
        turns = [step for o in items for step in o.steps if step.get("turn")]
        metrics = [step["turn"].get("metrics") or {} for step in turns]
        signals = [step.get("signals") or {} for step in turns]
        classes = {klass: sum(1 for o in items if klass in o.classes()) for klass in CLASSES}
        summaries[model] = {
            "provider": items[0].provider,
            "runs": len(items),
            "statuses": {status: sum(1 for o in items if o.status == status) for status in SCENARIO_STATUSES},
            "scenario_pass_rate": _rate(sum(1 for o in graded if o.status in ("pass", "waived", "xfail")), len(graded)),
            "outcome_pass_rate": _rate(
                sum(1 for r in outcome_graded if r["status"] in ("pass", "xpass")), len(outcome_graded)
            ),
            "invariant_pass_rate": _rate(
                sum(1 for r in invariant_results if r["status"] in ("pass", "warn")), len(invariant_results)
            ),
            "classes": classes,
            "turns": len(turns),
            "turn_outcomes": _counts(str((step["turn"] or {}).get("outcome")) for step in turns),
            "mean_wall_time_s": _mean([step["turn"].get("wall_time_s") for step in turns]),
            "mean_time_to_first_token_s": _mean([m.get("time_to_first_token_s") for m in metrics]),
            "mean_requests": _mean([m.get("requests") for m in metrics]),
            "requests_sent": sum(int(step["turn"].get("requests_sent") or 0) for step in turns),
            "mean_prompt_tokens": _mean([m.get("prompt_tokens") for m in metrics]),
            "mean_completion_tokens": _mean([m.get("completion_tokens") for m in metrics]),
            "mean_tokens_per_s": _mean([m.get("output_tokens_per_s") for m in metrics]),
            "executed_calls": sum(int(s.get("executed_calls") or 0) for s in signals),
            "reference_calls": sum(int(s.get("reference_calls") or 0) for s in signals),
            "tool_errors": sum(int(s.get("tool_errors") or 0) for s in signals),
            "recovered_errors": sum(int(s.get("recovered_errors") or 0) for s in signals),
            "dropped_calls": sum(int(s.get("dropped_calls") or 0) for s in signals),
        }
    return summaries


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def unused_waivers(outcomes: list[ScenarioOutcome], global_waivers: dict[str, str]) -> list[str]:
    """Waivers that excused nothing in this run: like xpass, a hint that the bug may be fixed.

    A global waiver counts as used if it excused a failure in any scenario that ran;
    a scenario waiver, if it excused one in that scenario. Skipped scenarios and
    scenarios with infrastructure errors do not count.
    """
    graded = [o for o in outcomes if not o.skipped_reason and not o.infra_error]
    if not graded:
        return []
    used: set[tuple[str, str]] = set()
    for outcome in graded:
        used |= outcome.used_waivers()
    unused = [
        f"{invariant}:{bug} (global)"
        for invariant, bug in sorted(global_waivers.items())
        if (invariant, bug) not in used
    ]
    for outcome in graded:
        for invariant, bug in sorted(outcome.scenario.invariant_waivers.items()):
            if (invariant, bug) not in outcome.used_waivers():
                unused.append(f"{invariant}:{bug} ({outcome.scenario.id})")
    return unused


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
    mode = str(config.get("mode", "replay"))
    lines = [f"# Scenario run ({mode})", ""]
    if summary.get("interrupted"):
        lines += ["**Interrupted**: partial results.", ""]
    if summary.get("stopped"):
        lines += [f"**Stopped early**: {summary['stopped']}", ""]
    if summary.get("infra_rate_exceeded") is not None:
        lines += [
            f"**Too many infrastructure failures**: {_pct(summary.get('infra_rate'))} of the runs "
            f"(limit {_pct(summary['infra_rate_exceeded'])}); the run fails.",
            "",
        ]
    sc, cc = summary["scenario_counts"], summary["check_counts"]
    runs = "Runs" if mode != "replay" else "Scenarios"
    lines += [
        f"- {runs}: {summary['scenarios']} ({sc['pass']} pass, {sc['xfail']} xfail, {sc['waived']} waived, "
        f"{sc['xpass']} xpass, {sc['fail']} fail, {sc['error']} error, {sc['skipped']} skipped)",
        f"- Checks: {cc['pass']} pass, {cc['xfail']} xfail, {cc['xpass']} xpass, {cc['fail']} fail, "
        f"{cc['error']} error, {cc['warn']} warn, {cc['skip']} skip, {cc['unrecorded']} unrecorded",
        f"- Run time: {summary.get('duration_s', 0)} s",
        f"- Result: {'unexpected failures' if summary['exit_code'] else 'no unexpected failures'}",
    ]
    if config.get("attached_desktop"):
        lines.append(
            f"- Attached to the desktop app at {config.get('desktop_app_url')} (automation port "
            f"{config.get('automation_port')}); settings not pinned, its .env applies: "
            + ", ".join(config.get("unpinned_settings") or [])
        )
        if config.get("fit_view"):
            lines.append(
                "- View fitted to the drawings after each graded step (display only; view-sensitive "
                "scenarios are not fitted; a live model sees the fitted view)"
            )
    if mode != "replay":
        classes = summary.get("classes") or {}
        lines.append("- Failure classes (runs): " + ", ".join(f"{k} {v}" for k, v in classes.items()))
        lines += _config_lines(config)
    lines.append("")
    if summary.get("models"):
        lines += _model_lines(summary["models"])
        lines += _repeat_lines(outcomes)
    if mode == "retrace":
        lines += _retrace_lines(outcomes)
    smoke = [o for o in outcomes if o.scenario.smoke]
    if smoke and len(smoke) != len(outcomes):
        smoke_bad = [o.scenario.id for o in smoke if o.unexpected]
        lines += [f"- Smoke subset: {len(smoke)} scenarios, unexpected failures: {', '.join(smoke_bad) or 'none'}", ""]
    area_statuses = ("pass", "xfail", "waived", "xpass", "fail", "error", "skipped")
    lines += ["## By area", "", _row(["Area", "Scenarios", *area_statuses])]
    lines.append(_row(["---"] * (2 + len(area_statuses))))
    areas: dict[str, list[ScenarioOutcome]] = {}
    for outcome in outcomes:
        areas.setdefault(outcome.scenario.area, []).append(outcome)
    for area, items in areas.items():
        statuses = [o.status for o in items]
        lines.append(_row([area, len(items)] + [statuses.count(s) for s in area_statuses]))
    lines += [
        "",
        "## Scenarios",
        "",
        _row(["Scenario", "Status", "Checks (pass/xfail/xpass/fail)", "Known", "Time (s)"]),
    ]
    lines.append(_row(["---"] * 5))
    for outcome in outcomes:
        c = outcome.counts()
        title = f"{outcome.label} {outcome.scenario.title}" + (" (smoke)" if outcome.scenario.smoke else "")
        status = outcome.status
        if mode != "replay" and outcome.classes():
            status += f" ({', '.join(sorted(outcome.classes()))})"
        lines.append(
            _row(
                [
                    title,
                    status,
                    f"{c['pass']}/{c['xfail']}/{c['xpass']}/{c['fail'] + c['error']}",
                    ", ".join(outcome.scenario.known) or "-",
                    f"{outcome.duration_s:.1f}",
                ]
            )
        )
    failures = _failure_lines(outcomes, out_dir)
    if failures:
        title = "Unexpected failures" if mode == "replay" else "Failures (with class)"
        lines += ["", f"## {title}", ""] + failures
    if summary["xpass"]:
        lines += ["", "## Unexpected passes (fixed?)", ""]
        for scenario_id, bugs in summary["xpass"].items():
            lines.append(
                f"- {scenario_id}: fixed? {', '.join(bugs)} (remove the known marks when the fix is confirmed)"
            )
    if summary.get("unused_waivers"):
        lines += ["", "## Unused waivers (fixed?)", ""]
        lines += [f"- waiver {entry} excused nothing in this run" for entry in summary["unused_waivers"]]
    if summary.get("unrecorded"):
        lines += ["", "## Not evaluated (data not recorded)", ""]
        lines += [
            f"- {scenario_id}: a check needs a function sample this run did not record; rerun the replay"
            for scenario_id in summary["unrecorded"]
        ]
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


# Run settings shown at the top of a live or retrace summary.
_CONFIG_KEYS = (
    "provider",
    "models",
    "repeats",
    "local_reasoning_effort",
    "server_env",
    "turn_timeout_s",
    "turn_max_requests",
    "max_requests",
    "retraced_from",
)


def _config_lines(config: dict[str, Any]) -> list[str]:
    lines = []
    for key in _CONFIG_KEYS:
        value = config.get(key)
        if value is None or value == {} or value == []:
            continue
        if isinstance(value, dict):
            value = ", ".join(f"{k}={v}" for k, v in value.items())
        elif isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        lines.append(f"- {key}: {value}")
    return lines


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _pct(value: Any) -> str:
    return "-" if value is None else f"{100 * float(value):.0f}%"


def _model_lines(models: dict[str, dict[str, Any]]) -> list[str]:
    """The per-model table: pass rates, failure classes, turn costs and call efficiency."""
    header = [
        "Model",
        "Runs",
        "Scenario pass",
        "Outcome checks",
        "Invariants",
        "app/model/nondet/known/infra",
        "Mean turn (s)",
        "Mean first token (s)",
        "Mean requests",
        "Mean prompt/completion tok",
        "Calls (executed/reference)",
        "Tool errors (recovered)",
        "Dropped calls",
    ]
    lines = ["## By model", "", _row(header), _row(["---"] * len(header))]
    for model, data in models.items():
        classes = data.get("classes") or {}
        lines.append(
            _row(
                [
                    model,
                    data.get("runs"),
                    _pct(data.get("scenario_pass_rate")),
                    _pct(data.get("outcome_pass_rate")),
                    _pct(data.get("invariant_pass_rate")),
                    "/".join(str(classes.get(k, 0)) for k in CLASSES),
                    _fmt(data.get("mean_wall_time_s")),
                    _fmt(data.get("mean_time_to_first_token_s")),
                    _fmt(data.get("mean_requests")),
                    f"{_fmt(data.get('mean_prompt_tokens'))}/{_fmt(data.get('mean_completion_tokens'))}",
                    f"{data.get('executed_calls')}/{data.get('reference_calls')}",
                    f"{data.get('tool_errors')} ({data.get('recovered_errors')})",
                    data.get("dropped_calls"),
                ]
            )
        )
    outcomes = "; ".join(
        f"{model}: " + ", ".join(f"{k} {v}" for k, v in (data.get("turn_outcomes") or {}).items())
        for model, data in models.items()
    )
    lines += ["", f"Pass rates leave out runs with an infrastructure failure. Turn outcomes: {outcomes}", ""]
    return lines


def _retrace_lines(outcomes: list[ScenarioOutcome]) -> list[str]:
    """Each live run next to its retrace: whether the canvas was reproduced, and the live run's classes."""
    header = ["Live run", "Live status", "Live classes", "Retrace", "First difference"]
    lines = ["## Live runs retraced", "", _row(header), _row(["---"] * len(header))]
    verdicts = {True: "reproduced", False: "not reproduced", None: "not compared"}
    for outcome in outcomes:
        retrace = outcome.retrace or {}
        lines.append(
            _row(
                [
                    outcome.label,
                    retrace.get("live_status", "-"),
                    ", ".join(retrace.get("live_classes") or []) or "-",
                    verdicts.get(retrace.get("reproduced"), "-"),
                    retrace.get("first_difference") or retrace.get("error") or "-",
                ]
            )
        )
    lines.append("")
    return lines


def _repeat_lines(outcomes: list[ScenarioOutcome]) -> list[str]:
    """Passes per scenario and model, when a scenario ran more than once or with several models."""
    table: dict[str, dict[str, list[ScenarioOutcome]]] = {}
    for outcome in outcomes:
        table.setdefault(outcome.scenario.id, {}).setdefault(str(outcome.model), []).append(outcome)
    models = sorted({str(o.model) for o in outcomes})
    if len(models) == 1 and all(len(runs) == 1 for row in table.values() for runs in row.values()):
        return []
    lines = ["## Passes per scenario", "", _row(["Scenario", *models]), _row(["---"] * (1 + len(models)))]
    for scenario_id, row in table.items():
        cells = []
        for model in models:
            runs = row.get(model, [])
            passed = sum(1 for o in runs if o.status in ("pass", "waived", "xfail"))
            cells.append(f"{passed}/{len(runs)}" if runs else "-")
        lines.append(_row([scenario_id, *cells]))
    lines.append("")
    return lines


def _failure_lines(outcomes: list[ScenarioOutcome], out_dir: Path) -> list[str]:
    lines: list[str] = []
    for outcome in outcomes:
        if outcome.infra_error:
            lines.append(f"- **{outcome.label}**: infrastructure error: {outcome.infra_error}")
        for step in outcome.steps:
            bad = [r for r in step.get("results", []) if r["status"] in ("fail", "error")]
            if not bad and not step.get("error"):
                continue
            artifacts = step.get("artifacts") or {}
            links = ", ".join(f"[{kind}]({artifact_link(path, out_dir)})" for kind, path in artifacts.items())
            header = f"- **{outcome.label} / {step.get('step')}**" + (f" ({links})" if links else "")
            turn = step.get("turn")
            if turn:
                header += f" (turn {turn.get('outcome')})"
            lines.append(header)
            if step.get("error"):
                lines.append(f"  - step error: {step['error']}")
            for result in bad:
                detail = result.get("message", "")
                klass = f" {result['class']}" if result.get("class") else ""
                lines.append(f"  - {result['id']} `{result['name']}` [{result['status']}{klass}]: {detail}")
        if outcome.retrace and not outcome.retrace.get("reproduced"):
            first = outcome.retrace.get("first_difference")
            found = (outcome.retrace.get("differences") or {}).get(first) or []
            detail = "; ".join(found[:5])
            lines.append(f"- **{outcome.label}**: the retrace differs from the live canvas at {first}: {detail}")
    return lines


def artifact_link(path: str, out_dir: Path) -> str:
    """A link to an artifact from summary.md in ``out_dir``.

    Runs store artifact paths relative to their output directory; older or
    hand-made results may hold absolute paths, which are made relative when
    possible and kept as they are otherwise.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(out_dir.resolve()).as_posix()
    except (ValueError, OSError):
        return candidate.as_posix()


# ----------------------------------------------------------------------
# Regrade
# ----------------------------------------------------------------------


def regrade(
    results_path: Path,
    catalogue: Catalogue,
    scenario_ids: Optional[set[str]] = None,
    max_infra_rate: Optional[float] = None,
) -> tuple[dict[str, Any], Path]:
    """Re-run every check on the stored states of ``results_path``.

    ``scenario_ids`` limits the regrade to those scenarios (the command's
    ``--ids``, ``--tags`` and ``--smoke`` filters). ``max_infra_rate`` replaces
    the limit the run stored (the command's explicit ``--max-infra-rate``). Writes
    ``results_regraded.json`` and ``summary_regraded.md`` next to the results.
    A check that needs a function sample the run did not record is reported as
    ``unrecorded``.

    Returns:
        (summary, path of the regraded results).
    """
    data = json.loads(results_path.read_text(encoding="utf-8"))
    by_id = {scenario.id: scenario for scenario in catalogue.scenarios}
    config = dict(data.get("config") or {})
    config["regraded_from"] = str(results_path)
    if max_infra_rate is not None:
        config["max_infra_rate"] = max_infra_rate
    mode = str(config.get("mode", "replay"))
    outcomes: list[ScenarioOutcome] = []
    for stored in data.get("scenarios", []):
        scenario = by_id.get(stored.get("id"))
        if scenario is None or (scenario_ids is not None and scenario.id not in scenario_ids):
            continue
        outcome = ScenarioOutcome(
            scenario,
            duration_s=float(stored.get("duration_s") or 0.0),
            infra_error=stored.get("infra_error"),
            skipped_reason=stored.get("skipped_reason"),
            attempts=int(stored.get("attempts") or 1),
            provider=stored.get("provider"),
            model=stored.get("model"),
            repeat=int(stored.get("repeat") or 1),
            retrace=stored.get("retrace"),
        )
        if not outcome.skipped_reason:
            waivers = catalogue.waivers_for(scenario)
            outcome.steps = regrade_steps(scenario, stored.get("steps", []), waivers, mode)
            outcome.retrace = regrade_retrace(scenario, outcome, waivers, mode)
        outcome.annotate(mode)
        outcomes.append(outcome)
    summary = summarize(
        outcomes, catalogue.invariant_waivers, mode, config.get("max_infra_rate", DEFAULT_MAX_INFRA_RATE)
    )
    out_dir = results_path.parent
    payload = {"config": config, "summary": summary, "scenarios": [o.to_dict() for o in outcomes]}
    target = out_dir / "results_regraded.json"
    target.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    (out_dir / "summary_regraded.md").write_text(
        render_summary(outcomes, summary, catalogue, config, out_dir), encoding="utf-8"
    )
    return summary, target


def regrade_retrace(
    scenario: Scenario, outcome: ScenarioOutcome, waivers: dict[str, str], mode: str
) -> Optional[dict[str, Any]]:
    """The stored retrace of a run, with its verdicts from the current checker.

    A live run's retrace steps are graded again and their per-batch I4 and I5
    replace the live turn-level ones again; a retrace run's own invariant failures
    are its regraded steps'. A stored retrace without its steps (an older run)
    keeps its canvas comparison, but its invariant verdicts are dropped as stale.
    """
    retrace = dict(outcome.retrace) if outcome.retrace else None
    if retrace is None:
        return None
    if mode == "retrace":
        retrace["invariant_failures"] = retrace_invariant_failures(outcome.steps)
    elif retrace.get("steps"):
        retrace["steps"] = regrade_steps(scenario, retrace["steps"], waivers, "retrace")
        retrace["invariant_failures"] = retrace_invariant_failures(retrace["steps"])
        apply_batch_verdicts(outcome.steps, retrace)
    elif "invariant_failures" in retrace:
        retrace.pop("invariant_failures")
        retrace["stale"] = "the retrace's steps were not stored, so its invariant verdicts could not be regraded"
    return retrace


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
            fitted = record.get("fitted_view")
            if isinstance(fitted, dict):
                # Attach mode fitted the view after this step: the next step started from it.
                grader.rebase_view(StepRecordData(state=fitted.get("state") or {}, inspection=fitted.get("inspection")))
        else:
            record["results"] = [
                CheckResult(f"{step_id}.regrade", "check", "regrade", False, error=True,
                            message=f"step {step_id} is no longer in scenario {scenario.id}").to_dict()
            ]  # fmt: skip
        regraded.append(record)
    return regraded
