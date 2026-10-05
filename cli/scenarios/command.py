"""``python -m cli.main test scenarios``: the scenario-testing command."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

import click

from cli.config import DEFAULT_PORT, PROJECT_ROOT
from cli.scenarios.classify import annotate_steps, step_classes
from cli.scenarios.live import (
    DEFAULT_TURN_MAX_REQUESTS,
    DEFAULT_TURN_TIMEOUT_S,
    LiveOptions,
    LiveRunner,
    RequestBudget,
    RetraceRunner,
    planned_requests,
    retrace_summary,
)
from cli.scenarios.live_config import (
    CANVAS_FORMATS,
    DEFAULT_LOCAL_REASONING_EFFORT,
    DEFAULT_MAX_REQUESTS,
    LOCAL_REASONING_EFFORT_CHOICES,
    PROVIDERS,
    TOOL_EXPOSURES,
    TOOL_SEARCH_MODES,
    GuardError,
    LiveSettings,
    check_models,
    check_request_cap,
    live_plan,
    recorded_env,
    server_env,
)
from cli.scenarios.model import Catalogue, Scenario, ScenarioError, load_catalogue
from cli.scenarios.report import ResultSink, regrade
from cli.scenarios.runner import DEFAULT_STEP_TIMEOUT_S, BrowserSession, ReplayOptions, ReplayRunner
from cli.server import ServerManager
from static.config import WORKSPACES_DIR_ENV

DEFAULT_OUT_ROOT = PROJECT_ROOT / "logs" / "scenario_runs"
# Ports a live or retrace run tries after --port when a server already answers there.
OWN_SERVER_PORT_TRIES = 25


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


def _fail(message: str, code: int = 2) -> int:
    click.echo(click.style(message, fg="red"), err=True)
    return code


@click.command("scenarios")
@click.argument("results_path", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--mode",
    type=click.Choice(["replay", "live", "retrace"]),
    default="replay",
    show_default=True,
    help="replay: reference calls, no model; live: prompts to a model; retrace RESULTS: a live run's calls again",
)
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
@click.option("--dry-run", is_flag=True, help="Validate the scenarios and print the plan (live: requests and cost)")
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
@click.option(
    "--provider", type=click.Choice(PROVIDERS), default="local", show_default=True, help="Live: model provider"
)
@click.option(
    "--models",
    multiple=True,
    help="Live: model ids (repeatable or comma-separated). Local default: every model llama-server serves; "
    "OpenRouter: required",
)
@click.option("--repeats", default=1, type=click.IntRange(min=1), show_default=True, help="Live: runs per scenario")
@click.option(
    "--max-requests",
    default=DEFAULT_MAX_REQUESTS,
    type=click.IntRange(min=1),
    show_default=True,
    help="OpenRouter: refuse a run that may send more model requests, and stop at this many",
)
@click.option(
    "--local-reasoning-effort",
    type=click.Choice(LOCAL_REASONING_EFFORT_CHOICES),
    default=DEFAULT_LOCAL_REASONING_EFFORT,
    show_default=True,
    help="Live, local: MATHUD_LOCAL_REASONING_EFFORT (default sends none)",
)
@click.option("--tool-exposure", type=click.Choice(TOOL_EXPOSURES), default="search", show_default=True, help="Live")
@click.option("--canvas-format", type=click.Choice(CANVAS_FORMATS), default="text", show_default=True, help="Live")
@click.option(
    "--canvas-budget", type=int, default=None, help="Live: MATHUD_CANVAS_BUDGET_TOKENS (default: provider default)"
)
@click.option(
    "--tool-search-mode",
    type=click.Choice(TOOL_SEARCH_MODES),
    default=None,
    help="Live: TOOL_SEARCH_MODE (default: hybrid for local, local for OpenRouter)",
)
@click.option(
    "--turn-timeout",
    default=DEFAULT_TURN_TIMEOUT_S,
    type=float,
    show_default=True,
    help="Live: seconds per turn unless the scenario sets limits.timeout_s",
)
@click.option(
    "--turn-max-requests",
    default=DEFAULT_TURN_MAX_REQUESTS,
    type=click.IntRange(min=1),
    show_default=True,
    help="Live: model requests per turn unless the scenario sets limits.max_requests",
)
@click.option("--no-retrace", is_flag=True, help="Live: do not retrace failing scenarios")
def scenarios_cmd(
    results_path: Optional[str],
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
    provider: str,
    models: tuple[str, ...],
    repeats: int,
    max_requests: int,
    local_reasoning_effort: str,
    tool_exposure: str,
    canvas_format: str,
    canvas_budget: Optional[int],
    tool_search_mode: Optional[str],
    turn_timeout: float,
    turn_max_requests: int,
    no_retrace: bool,
) -> None:
    """Run the agentic scenario tests (see documentation/development/agentic_scenario_testing.md).

    Replay mode runs every scenario's reference tool calls in the real app with no
    model. Known bugs (K<n>) are expected failures; the command exits non-zero only
    on unexpected failures. Live mode sends the prompts to a model (LocalAgent by
    default; OpenRouter only with --provider openrouter and its request cap) and
    exits non-zero only on app failures (class app or nondeterministic), not on
    model mistakes. Retrace mode re-executes a live run's calls with no model.
    """
    catalogue = _load(scenarios_dir)
    if results_path and mode != "retrace":
        raise SystemExit(_fail("A results file argument is only taken by --mode retrace."))

    filtered = smoke or bool(_split(tags)) or bool(_split(ids))
    if regrade_path:
        wanted = {s.id for s in catalogue.select(smoke=smoke, tags=_split(tags), ids=_split(ids))} if filtered else None
        summary, target = regrade(Path(regrade_path), catalogue, wanted)
        _print_summary(summary, as_json, Path(target).parent, regraded=True)
        raise SystemExit(summary["exit_code"])

    stamp = time.strftime("%Y%m%d-%H%M%S")
    options = ReplayOptions(step_timeout_s=step_timeout, known_artifacts=known_artifacts)
    if mode == "retrace":
        if not results_path:
            raise SystemExit(_fail("--mode retrace needs a live run's results.json: --mode retrace RESULTS"))
        wanted = {s.id for s in catalogue.select(smoke=smoke, tags=_split(tags), ids=_split(ids))} if filtered else None
        source = Path(results_path)
        raise SystemExit(
            _run_retrace(
                catalogue,
                source,
                wanted,
                port=port,
                out_dir=Path(out_dir) if out_dir else source.parent / f"retrace-{stamp}",
                as_json=as_json,
                options=options,
                headless=not no_headless,
            )
        )

    chosen = catalogue.select(smoke=smoke, tags=_split(tags), ids=_split(ids))
    if not chosen:
        raise SystemExit(_fail("No scenarios match the filters."))
    run_out = Path(out_dir) if out_dir else DEFAULT_OUT_ROOT / stamp
    filters = {"smoke": smoke, "tags": _split(tags), "ids": _split(ids)}

    if mode == "live":
        settings = LiveSettings(
            provider=provider,
            tool_exposure=tool_exposure,
            canvas_format=canvas_format,
            canvas_budget=canvas_budget,
            tool_search_mode=tool_search_mode,
            local_reasoning_effort=local_reasoning_effort,
        )
        live_options = LiveOptions(
            turn_timeout_s=turn_timeout, turn_max_requests=turn_max_requests, retrace_failures=not no_retrace
        )
        # A live run never retries a scenario: a retry would send its prompts again.
        options.retries = 0
        raise SystemExit(
            _run_live(
                catalogue,
                chosen,
                settings=settings,
                models=_split(models),
                repeats=repeats,
                max_requests=max_requests,
                live_options=live_options,
                port=port,
                out_dir=run_out,
                as_json=as_json,
                options=options,
                headless=not no_headless,
                dry_run=dry_run,
                config_extra=filters,
            )
        )

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
        out_dir=run_out,
        as_json=as_json,
        options=options,
        headless=not no_headless,
        config_extra=filters,
    )
    raise SystemExit(exit_code)


def _workspace_skips(scenarios: list[Scenario]) -> dict[str, str]:
    """Scenarios that save or load workspaces, which must not touch another server's workspace directory."""
    reason = "uses workspaces: run with --start-server or --allow-workspace-writes"
    return {scenario.id: reason for scenario in scenarios if scenario.uses_workspaces}


def _start_server(manager: ServerManager, extra_env: dict[str, str]) -> Optional[str]:
    """Start ``manager``'s server with ``extra_env``; returns an error message, or None."""
    ok, message = manager.start(wait=True, auto_increment_port=True, max_port_tries=25, extra_env=extra_env)
    return None if ok else message


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
            return _fail(f"Server is not running on port {port}. Use --start-server.")
        workspaces_tmp = tempfile.mkdtemp(prefix="mathud-scenario-workspaces-")
        error = _start_server(manager, {WORKSPACES_DIR_ENV: workspaces_tmp})
        if error:
            shutil.rmtree(workspaces_tmp, ignore_errors=True)
            return _fail(f"Failed to start server: {error}")
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
    click.echo(f"Replaying {len(chosen)} scenario(s) on port {port}; output in {out_dir}", err=as_json)
    return _drive(
        lambda: runner.run(chosen, skipped), session, sink, catalogue, manager if server_started else None,
        workspaces_tmp, as_json, out_dir,
    )  # fmt: skip


def _drive(
    run: Callable[[], Any],
    session: BrowserSession,
    sink: ResultSink,
    catalogue: Catalogue,
    manager: Optional[ServerManager],
    workspaces_tmp: Optional[str],
    as_json: bool,
    out_dir: Path,
) -> int:
    """Run, then always write the reports and clean up (also on Ctrl+C); returns the exit code."""
    interrupted = False
    try:
        run()
    except KeyboardInterrupt:
        interrupted = True
        click.echo(click.style("Interrupted; writing partial results.", fg="yellow"), err=True)
    finally:
        session.kill()
        summary = sink.close(catalogue, interrupted=interrupted)
        if manager is not None:
            manager.stop()
        if workspaces_tmp:
            shutil.rmtree(workspaces_tmp, ignore_errors=True)
    _print_summary(summary, as_json, out_dir)
    return 130 if interrupted else int(summary["exit_code"])


# ----------------------------------------------------------------------
# Live
# ----------------------------------------------------------------------


def _start_own_server(port: int, extra_env: dict[str, str]) -> tuple[Optional[ServerManager], str]:
    """Start a server for a live or retrace run on ``port`` or the next port no server answers on.

    The run pins its server's environment, so it never uses a server it did not start.
    Returns (manager, "") or (None, error message).
    """
    for candidate in range(port, port + OWN_SERVER_PORT_TRIES):
        manager = ServerManager(port=candidate)
        if manager.is_server_running():
            continue
        error = _start_server(manager, extra_env)
        return (None, error) if error else (manager, "")
    return None, f"no free port from {port} to {port + OWN_SERVER_PORT_TRIES - 1}"


def _available_models(base_url: str) -> Any:
    import requests

    response = requests.get(f"{base_url}/api/available_models", timeout=60)
    response.raise_for_status()
    return response.json()


def _conversation_resetter(base_url: str) -> Callable[[], None]:
    """POST /new_conversation and wait for it, so a scenario never starts on the previous one's history."""
    import requests

    def reset() -> None:
        response = requests.post(f"{base_url}/new_conversation", timeout=30)
        response.raise_for_status()

    return reset


def _print_live_plan(plan: dict[str, Any], as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(plan, indent=2))
        return
    click.echo(
        f"Live plan ({plan['provider']}): {plan['scenarios']} scenarios, {plan['turns']} turns "
        f"({plan['repeats']} repeat(s), models: {', '.join(plan['models'])})"
    )
    cap = plan.get("max_requests")
    click.echo(f"Requests: at most {plan['planned_requests']}" + (f" (cap {cap})" if cap is not None else ""))
    if not plan["within_cap"]:
        click.echo(click.style("A live run would abort: the plan exceeds --max-requests.", fg="yellow"))
    for model, cost in (plan.get("estimated_cost_usd") or {}).items():
        shown = "no price listed" if cost is None else f"at most about ${cost:.4f}"
        click.echo(
            f"Estimated cost for {model}: {shown} ({plan['estimated_prompt_tokens_per_request']} prompt and "
            f"{plan['estimated_completion_tokens_per_request']} completion tokens per request, prices as of "
            f"{plan['prices_as_of']}; every turn assumed to use its whole request cap)"
        )


def _run_live(
    catalogue: Catalogue,
    chosen: list[Scenario],
    *,
    settings: LiveSettings,
    models: list[str],
    repeats: int,
    max_requests: int,
    live_options: LiveOptions,
    port: int,
    out_dir: Path,
    as_json: bool,
    options: ReplayOptions,
    headless: bool,
    dry_run: bool,
    config_extra: dict[str, Any],
) -> int:
    try:
        settings.validate()
    except GuardError as exc:
        return _fail(str(exc))
    openrouter = settings.provider == "openrouter"
    cap = max_requests if openrouter else None
    if openrouter and not models:
        return _fail("--provider openrouter needs --models (no default model is ever chosen for a paid provider)")
    planned = planned_requests(chosen, max(len(models), 1), repeats, live_options)
    if dry_run:
        plan = live_plan(
            scenarios=len(chosen),
            turns=sum(s.turns for s in chosen),
            models=models or ["<every model llama-server serves>"],
            repeats=repeats,
            planned=planned,
            settings=settings,
            max_requests=cap,
        )
        _print_live_plan(plan, as_json)
        return 0
    if openrouter:
        try:
            check_request_cap(planned, max_requests)
        except GuardError as exc:
            return _fail(f"Aborting: {exc}")

    from cli.browser import BrowserAutomation

    workspaces_tmp = tempfile.mkdtemp(prefix="mathud-scenario-workspaces-")
    env = server_env(settings, workspaces_tmp)
    manager, error = _start_own_server(port, env)
    if manager is None:
        shutil.rmtree(workspaces_tmp, ignore_errors=True)
        return _fail(f"Failed to start server: {error}")
    try:
        run_models = check_models(_available_models(manager.base_url), settings.provider, models)
    except Exception as exc:  # GuardError, or the server did not answer
        manager.stop()
        shutil.rmtree(workspaces_tmp, ignore_errors=True)
        return _fail(f"Aborting before the first message: {exc}")

    config: dict[str, Any] = {
        "mode": "live",
        "provider": settings.provider,
        "models": run_models,
        "repeats": repeats,
        "port": manager.port,
        "server_started": True,
        "workspaces_dir": workspaces_tmp,
        "server_env": recorded_env(env),
        "local_reasoning_effort": settings.local_reasoning_effort if settings.provider == "local" else None,
        "turn_timeout_s": live_options.turn_timeout_s,
        "turn_max_requests": live_options.turn_max_requests,
        "max_requests": cap,
        "planned_requests": planned_requests(chosen, len(run_models), repeats, live_options),
        "retrace_failures": live_options.retrace_failures,
        "step_timeout_s": options.step_timeout_s,
    }
    config.update(config_extra)
    sink = ResultSink(out_dir, config)
    session = BrowserSession(lambda: BrowserAutomation(port=manager.port, headless=headless), options.step_timeout_s)
    runner = LiveRunner(
        catalogue,
        session,
        sink,
        options,
        live_options,
        provider=settings.provider,
        reset_conversation=_conversation_resetter(manager.base_url),
        budget=RequestBudget(cap),
        log=lambda line: click.echo(line, err=as_json),
    )
    click.echo(
        f"Live run ({settings.provider}: {', '.join(run_models)}), {len(chosen)} scenario(s) x {repeats} on port "
        f"{manager.port}; output in {out_dir}",
        err=as_json,
    )

    def run() -> None:
        try:
            runner.run_live(chosen, run_models, repeats)
        finally:
            config["requests_sent"] = runner.budget.sent
            if runner.stopped:
                config["stopped"] = runner.stopped

    return _drive(run, session, sink, catalogue, manager, workspaces_tmp, as_json, out_dir)


# ----------------------------------------------------------------------
# Retrace
# ----------------------------------------------------------------------


def _run_retrace(
    catalogue: Catalogue,
    source: Path,
    wanted: Optional[set[str]],
    *,
    port: int,
    out_dir: Path,
    as_json: bool,
    options: ReplayOptions,
    headless: bool,
) -> int:
    """Retrace every stored live run on a server of its own (a clean workspace directory)."""
    from cli.browser import BrowserAutomation

    data = json.loads(source.read_text(encoding="utf-8"))
    source_config = data.get("config") or {}
    if source_config.get("mode") != "live":
        return _fail(f"{source} is not a live run's results (mode {source_config.get('mode')!r})")
    by_id = {scenario.id: scenario for scenario in catalogue.scenarios}
    stored = [
        item
        for item in data.get("scenarios", [])
        if item.get("id") in by_id
        and (wanted is None or item.get("id") in wanted)
        and not item.get("skipped_reason")
        and item.get("steps")
    ]
    if not stored:
        return _fail("No live runs to retrace match the filters.")

    workspaces_tmp = tempfile.mkdtemp(prefix="mathud-scenario-workspaces-")
    # No model is involved, but the server still gets no provider keys.
    manager, error = _start_own_server(
        port,
        {WORKSPACES_DIR_ENV: workspaces_tmp, "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "", "OPENROUTER_API_KEY": ""},
    )
    if manager is None:
        shutil.rmtree(workspaces_tmp, ignore_errors=True)
        return _fail(f"Failed to start server: {error}")
    config: dict[str, Any] = {
        "mode": "retrace",
        "retraced_from": str(source),
        "provider": source_config.get("provider"),
        "models": source_config.get("models"),
        "repeats": source_config.get("repeats"),
        "port": manager.port,
        "server_started": True,
        "workspaces_dir": workspaces_tmp,
        "step_timeout_s": options.step_timeout_s,
    }
    sink = ResultSink(out_dir, config)
    session = BrowserSession(lambda: BrowserAutomation(port=manager.port, headless=headless), options.step_timeout_s)
    runner = RetraceRunner(catalogue, session, sink, options, log=lambda line: click.echo(line, err=as_json))
    click.echo(f"Retracing {len(stored)} live run(s) from {source} on port {manager.port}; output in {out_dir}")

    def run() -> None:
        for number, item in enumerate(stored, start=1):
            outcome = runner.retrace_outcome(
                by_id[item["id"]], item["steps"], item.get("provider"), item.get("model"), int(item.get("repeat") or 1)
            )
            comparison = retrace_summary(item["steps"], outcome)
            live_steps = annotate_steps(copy.deepcopy(item["steps"]), "live", comparison)
            live_classes = step_classes(live_steps) | ({"infra"} if item.get("infra_error") else set())
            comparison.update(live_status=item.get("status"), live_classes=sorted(live_classes))
            outcome.retrace = comparison
            sink.add(outcome)
            verdict = {True: "reproduced", False: "NOT reproduced", None: "not compared"}[comparison.get("reproduced")]
            click.echo(
                f"[{number}/{len(stored)}] {outcome.label} live {item.get('status')}, retrace {verdict}", err=as_json
            )

    return _drive(run, session, sink, catalogue, manager, workspaces_tmp, as_json, out_dir)


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
    for model, data in (summary.get("models") or {}).items():
        classes = ", ".join(f"{k} {v}" for k, v in (data.get("classes") or {}).items() if v)
        click.echo(
            f"  {model}: scenario pass rate {_percent(data.get('scenario_pass_rate'))}, outcome checks "
            f"{_percent(data.get('outcome_pass_rate'))}, invariants {_percent(data.get('invariant_pass_rate'))}, "
            f"mean turn {data.get('mean_wall_time_s')} s, {data.get('requests_sent')} requests"
            + (f"; classes: {classes}" if classes else "")
        )
    if summary.get("stopped"):
        click.echo(click.style(f"Stopped early: {summary['stopped']}", fg="red"))
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


def _percent(value: Any) -> str:
    return "-" if value is None else f"{100 * float(value):.0f}%"
