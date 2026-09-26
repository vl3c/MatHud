"""``python -m cli.main test scenarios``: the scenario-testing command."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import click

from cli.config import DEFAULT_PORT, PROJECT_ROOT
from cli.scenarios.model import Catalogue, Scenario, ScenarioError, load_catalogue
from cli.scenarios.report import ResultSink, regrade
from cli.scenarios.runner import DEFAULT_STEP_TIMEOUT_S, BrowserSession, ReplayOptions, ReplayRunner
from cli.server import ServerManager
from static.config import WORKSPACES_DIR_ENV

DEFAULT_OUT_ROOT = PROJECT_ROOT / "logs" / "scenario_runs"


def _split(values: tuple[str, ...]) -> list[str]:
    return [item.strip() for value in values for item in value.split(",") if item.strip()]


def _load(scenarios_dir: Optional[str]) -> Catalogue:
    try:
        return load_catalogue(Path(scenarios_dir) if scenarios_dir else None)
    except ScenarioError as exc:
        click.echo(click.style(f"{len(exc.problems)} problem(s) in the scenario files:", fg="red"), err=True)
        for problem in exc.problems:
            click.echo(f"  - {problem}", err=True)
        raise SystemExit(2)


def _plan(scenarios: list[Scenario], skipped: dict[str, str]) -> dict[str, Any]:
    return {
        "scenarios": len(scenarios),
        "turns": sum(s.turns for s in scenarios),
        "calls": sum(len(s.setup_calls) + sum(len(step.calls) for step in s.steps) for s in scenarios),
        "ids": [s.id for s in scenarios],
        "skipped": skipped,
    }


@click.command("scenarios")
@click.option("--mode", type=click.Choice(["replay", "live"]), default="replay", show_default=True, help="Run mode")
@click.option("--smoke", is_flag=True, help="Only the smoke subset")
@click.option("--tags", multiple=True, help="Only scenarios with one of these tags (repeatable or comma-separated)")
@click.option(
    "--ids",
    "--only",
    "ids",
    multiple=True,
    help="Only these scenario ids or areas, e.g. GEO-04 or CV (repeatable or comma-separated)",
)
@click.option("--port", "-p", default=DEFAULT_PORT, type=int, help=f"Server port (default: {DEFAULT_PORT})")
@click.option("--start-server", is_flag=True, help="Start a server for the run (workspaces go to a temp dir)")
@click.option(
    "--out", "out_dir", type=click.Path(file_okay=False), help="Output directory (default: logs/scenario_runs/<time>)"
)
@click.option("--json", "as_json", is_flag=True, help="Print the summary as JSON")
@click.option(
    "--regrade",
    "regrade_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Re-grade a results.json, no browser",
)
@click.option("--dry-run", is_flag=True, help="Validate the scenarios and print the plan")
@click.option(
    "--step-timeout", default=DEFAULT_STEP_TIMEOUT_S, type=float, show_default=True, help="Seconds per browser call"
)
@click.option("--no-headless", is_flag=True, help="Show the browser window")
@click.option(
    "--allow-workspace-writes",
    is_flag=True,
    help="With --port: run workspace scenarios against that server's workspace directory",
)
@click.option("--known-artifacts", is_flag=True, help="Also save state and screenshot for expected failures")
@click.option(
    "--scenarios-dir", type=click.Path(exists=True, file_okay=False), help="Scenario directory (default: scenarios/)"
)
def scenarios_cmd(
    mode: str,
    smoke: bool,
    tags: tuple[str, ...],
    ids: tuple[str, ...],
    port: int,
    start_server: bool,
    out_dir: Optional[str],
    as_json: bool,
    regrade_path: Optional[str],
    dry_run: bool,
    step_timeout: float,
    no_headless: bool,
    allow_workspace_writes: bool,
    known_artifacts: bool,
    scenarios_dir: Optional[str],
) -> None:
    """Run the agentic scenario tests (see documentation/development/agentic_scenario_testing.md).

    Replay mode runs every scenario's reference tool calls in the real app with no
    model. Known bugs (K<n>) are expected failures; the command exits non-zero only
    on unexpected failures.
    """
    if mode == "live":
        click.echo(click.style("--mode live is not implemented yet; use --mode replay.", fg="red"), err=True)
        raise SystemExit(2)
    catalogue = _load(scenarios_dir)

    if regrade_path:
        filtered = smoke or bool(_split(tags)) or bool(_split(ids))
        wanted = {s.id for s in catalogue.select(smoke=smoke, tags=_split(tags), ids=_split(ids))} if filtered else None
        summary, target = regrade(Path(regrade_path), catalogue, wanted)
        _print_summary(summary, as_json, Path(target).parent, regraded=True)
        raise SystemExit(summary["exit_code"])

    chosen = catalogue.select(smoke=smoke, tags=_split(tags), ids=_split(ids))
    if not chosen:
        click.echo(click.style("No scenarios match the filters.", fg="red"), err=True)
        raise SystemExit(2)
    skipped = {} if start_server or allow_workspace_writes else _workspace_skips(chosen)

    if dry_run:
        plan = _plan(chosen, skipped)
        if as_json:
            click.echo(json.dumps(plan, indent=2))
        else:
            click.echo(
                f"{plan['scenarios']} scenarios, {plan['turns']} turns, {plan['calls']} reference calls (all valid)"
            )
            for scenario in chosen:
                flag = f"  [skipped: {skipped[scenario.id]}]" if scenario.id in skipped else ""
                smoke_mark = " (smoke)" if scenario.smoke else ""
                click.echo(f"  {scenario.id}{smoke_mark} {scenario.title}{flag}")
        return

    exit_code = _run_replay(
        catalogue,
        chosen,
        allow_workspace_writes=allow_workspace_writes,
        port=port,
        start_server=start_server,
        out_dir=Path(out_dir) if out_dir else DEFAULT_OUT_ROOT / time.strftime("%Y%m%d-%H%M%S"),
        as_json=as_json,
        options=ReplayOptions(step_timeout_s=step_timeout, known_artifacts=known_artifacts),
        headless=not no_headless,
        config_extra={"smoke": smoke, "tags": _split(tags), "ids": _split(ids)},
    )
    raise SystemExit(exit_code)


def _workspace_skips(scenarios: list[Scenario]) -> dict[str, str]:
    """Scenarios that save or load workspaces, which must not touch another server's workspace directory."""
    reason = "uses workspaces: run with --start-server or --allow-workspace-writes"
    return {scenario.id: reason for scenario in scenarios if scenario.uses_workspaces}


def _run_replay(
    catalogue: Catalogue,
    chosen: list[Scenario],
    *,
    allow_workspace_writes: bool,
    port: int,
    start_server: bool,
    out_dir: Path,
    as_json: bool,
    options: ReplayOptions,
    headless: bool,
    config_extra: dict[str, Any],
) -> int:
    from cli.browser import BrowserAutomation

    manager = ServerManager(port=port)
    workspaces_tmp: Optional[str] = None
    server_started = False
    if not manager.is_server_running():
        if not start_server:
            click.echo(click.style(f"Server is not running on port {port}. Use --start-server.", fg="red"), err=True)
            return 2
        workspaces_tmp = tempfile.mkdtemp(prefix="mathud-scenario-workspaces-")
        ok, message = manager.start(
            wait=True,
            auto_increment_port=True,
            max_port_tries=25,
            extra_env={WORKSPACES_DIR_ENV: workspaces_tmp},
        )
        if not ok:
            click.echo(click.style(f"Failed to start server: {message}", fg="red"), err=True)
            shutil.rmtree(workspaces_tmp, ignore_errors=True)
            return 2
        server_started = True
        port = manager.port
    elif start_server:
        click.echo(click.style(f"A server is already running on port {port}; using it.", fg="yellow"), err=True)
    # Workspace scenarios run only against a server this command started (temp workspace dir) or when allowed.
    skipped = {} if server_started or allow_workspace_writes else _workspace_skips(chosen)

    config: dict[str, Any] = {
        "mode": "replay",
        "port": port,
        "server_started": server_started,
        "workspaces_dir": workspaces_tmp,
        "step_timeout_s": options.step_timeout_s,
    }
    config.update(config_extra)
    sink = ResultSink(out_dir, config)
    session = BrowserSession(lambda: BrowserAutomation(port=port, headless=headless), options.step_timeout_s)
    runner = ReplayRunner(catalogue, session, sink, options, log=lambda line: click.echo(line, err=as_json))
    interrupted = False
    click.echo(f"Replaying {len(chosen)} scenario(s) on port {port}; output in {out_dir}", err=as_json)
    try:
        runner.run(chosen, skipped)
    except KeyboardInterrupt:
        interrupted = True
        click.echo(click.style("Interrupted; writing partial results.", fg="yellow"), err=True)
    finally:
        session.kill()
        summary = sink.close(catalogue, interrupted=interrupted)
        if server_started:
            manager.stop()
        if workspaces_tmp:
            shutil.rmtree(workspaces_tmp, ignore_errors=True)
    _print_summary(summary, as_json, out_dir)
    return 130 if interrupted else int(summary["exit_code"])


def _print_summary(summary: dict[str, Any], as_json: bool, out_dir: Path, regraded: bool = False) -> None:
    if as_json:
        payload = dict(summary)
        payload["out_dir"] = str(out_dir)
        click.echo(json.dumps(payload, indent=2))
        return
    sc, cc = summary["scenario_counts"], summary["check_counts"]
    title = "Regraded" if regraded else "Scenario results"
    click.echo()
    click.echo(click.style(f"=== {title} ===", bold=True))
    click.echo(
        f"Scenarios: {summary['scenarios']} - {sc['pass']} pass, {sc['xfail']} xfail, {sc['waived']} waived, "
        f"{sc['xpass']} xpass, "
        f"{sc['fail']} fail, {sc['error']} error, {sc['skipped']} skipped"
    )
    click.echo(
        f"Checks: {cc['pass']} pass, {cc['xfail']} xfail, {cc['xpass']} xpass, {cc['fail']} fail, "
        f"{cc['error']} error, {cc['warn']} warn, {cc['skip']} skip, {cc['unrecorded']} unrecorded"
    )
    if "duration_s" in summary:
        click.echo(f"Run time: {summary['duration_s']} s")
    for scenario_id, bugs in summary.get("xpass", {}).items():
        click.echo(click.style(f"  fixed? {scenario_id}: {', '.join(bugs)}", fg="yellow"))
    for entry in summary.get("unused_waivers", []):
        click.echo(click.style(f"  waiver {entry} unused in this run (fixed?)", fg="yellow"))
    if summary.get("unrecorded"):
        click.echo(click.style(f"  not evaluated (data not recorded): {', '.join(summary['unrecorded'])}", fg="yellow"))
    if summary["unexpected"]:
        click.echo(click.style(f"Unexpected failures: {', '.join(summary['unexpected'])}", fg="red"))
    else:
        click.echo(click.style("No unexpected failures.", fg="green"))
    click.echo(f"Reports: {out_dir}")
