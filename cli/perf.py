"""Performance benchmark command: ``python -m cli.main test perf``.

Runs ``window.startMatHudPerfBenchmark`` (static/client/perf_benchmark.py) in headless
Chrome a few times, keeps the median of each timing, and optionally saves the result
or compares it with an earlier run. Timings depend on the machine, so compare runs
made on the same machine: record a baseline on main, then compare a branch with it.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from selenium.common.exceptions import TimeoutException

from cli.browser import BrowserAutomation
from cli.config import DEFAULT_PORT, PROJECT_ROOT
from cli.server import ServerManager

# Seconds one benchmark run may take, and between polls for its result.
BENCHMARK_TIMEOUT = 900
POLL_INTERVAL = 2
# Polls that may fail in a row for a reason other than a busy page before the run is abandoned.
MAX_POLL_ERRORS = 3
# Metrics taken from one event per run (the first frame, creating a few drawables), so they
# vary by about 25% between runs: marked when slower, but never fail --fail-on-slower.
NOISY_METRIC_PARTS = (".build.", ".first_draw")


def is_noisy(key: str) -> bool:
    return any(part in key for part in NOISY_METRIC_PARTS)


def flatten_run(run: Dict[str, Any]) -> Dict[str, float]:
    """One benchmark result as ``{metric key: average ms}``.

    Keys look like ``functions.draw``, ``functions.plan_build``, ``geometry.build.circles``
    (milliseconds to create one circle) and ``expr.function_eval``.
    """
    flat: Dict[str, float] = {}
    for scene_name, scene in (run.get("scenes") or {}).items():
        for operation, value in (scene.get("metrics") or {}).items():
            flat[f"{scene_name}.{operation}"] = float(value)
        for phase, value in (scene.get("phases") or {}).items():
            if value is not None:
                flat[f"{scene_name}.{phase}"] = float(value)
        for kind, value in (scene.get("build_ms_per_drawable") or {}).items():
            flat[f"{scene_name}.build.{kind}"] = float(value)
    for name, timing in (run.get("expressions") or {}).items():
        flat[f"expr.{name}"] = float(timing["avg_ms"])
    return flat


def median_metrics(runs: List[Dict[str, Any]]) -> Dict[str, float]:
    """The median of each metric over several runs (a metric missing from a run is skipped there)."""
    values: Dict[str, List[float]] = {}
    for run in runs:
        for key, value in flatten_run(run).items():
            values.setdefault(key, []).append(value)
    return {key: statistics.median(samples) for key, samples in sorted(values.items())}


def compare_metrics(
    current: Dict[str, float], baseline: Dict[str, float]
) -> List[Tuple[str, Optional[float], Optional[float], Optional[float]]]:
    """Rows of (key, baseline ms, current ms, current/baseline) over the union of keys."""
    rows: List[Tuple[str, Optional[float], Optional[float], Optional[float]]] = []
    for key in sorted(set(current) | set(baseline)):
        before = baseline.get(key)
        after = current.get(key)
        ratio = after / before if before and after is not None else None
        rows.append((key, before, after, ratio))
    return rows


def is_slower(before: Optional[float], after: Optional[float], flag_ratio: float, min_delta_ms: float) -> bool:
    """True when ``after`` exceeds ``before`` by more than ``flag_ratio`` and by more than ``min_delta_ms``.

    The absolute floor keeps sub-millisecond metrics from flagging on timer noise.
    """
    if before is None or after is None or before <= 0:
        return False
    return after / before > flag_ratio and after - before > min_delta_ms


def _git_commit() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "describe", "--always", "--dirty"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _run_once(browser: BrowserAutomation, options: Dict[str, Any]) -> Dict[str, Any]:
    started = json.loads(
        browser.execute_js("return window.startMatHudPerfBenchmark(arguments[0])", json.dumps(options)) or "{}"
    )
    if started.get("status") != "started":
        return {"status": "error", "error": f"Could not start the benchmark: {started}"}
    deadline = time.time() + BENCHMARK_TIMEOUT
    errors = 0
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        try:
            result: Dict[str, Any] = json.loads(
                browser.execute_js("return window.getMatHudPerfBenchmarkResult()", timeout=120) or "{}"
            )
        except TimeoutException:
            continue  # the page is busy running the benchmark; try again
        except Exception as exc:
            errors += 1
            if errors >= MAX_POLL_ERRORS:
                return {"status": "error", "error": f"Lost the browser while polling: {exc}"}
            continue
        errors = 0
        if result.get("status") != "running":
            return result
    return {"status": "error", "error": f"The benchmark did not finish within {BENCHMARK_TIMEOUT} s"}


def _run_in_browser(
    port: int, start_server: bool, repeats: int, options: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Optional[str], bool]:
    """The runs, an error message or None, and whether an already running server was used."""
    manager = ServerManager(port=port)
    started = False
    if not manager.is_server_running():
        if not start_server:
            return [], f"Server is not running on port {port}. Start it or use --start-server.", False
        click.echo("Starting server...")
        success, message = manager.start(wait=True, auto_increment_port=True, max_port_tries=25)
        if not success:
            return [], f"Failed to start server: {message}", False
        started = True
        port = manager.port
    else:
        click.echo(
            f"Using the server already running on port {port}; it serves the checkout that started it, "
            "which may not be this one."
        )
    try:
        with BrowserAutomation(port=port, headless=True) as browser:
            if not browser.navigate_to_app() or not browser.wait_for_app_ready():
                return [], "The application did not load", not started
            runs: List[Dict[str, Any]] = []
            for index in range(repeats):
                click.echo(f"Benchmark run {index + 1}/{repeats}...")
                run = _run_once(browser, options)
                if run.get("status") != "done":
                    detail = "\n" + run["traceback"] if run.get("traceback") else ""
                    return runs, str(run.get("error", run)) + detail, not started
                runs.append(run)
            return runs, None, not started
    finally:
        if started:
            manager.stop()


def _echo_table(
    current: Dict[str, float], baseline: Optional[Dict[str, float]], flag_ratio: float, min_delta_ms: float
) -> List[str]:
    """Print the metrics (and the comparison); return the keys marked slower (see ``is_slower``).

    Noisy metrics (see ``NOISY_METRIC_PARTS``) are marked in yellow and left out of the result.
    """
    slower: List[str] = []
    if baseline is None:
        for key, value in current.items():
            click.echo(f"{key:<40} {value:>10.3f} ms")
        return slower
    click.echo(f"{'metric':<40} {'baseline':>10} {'current':>10} {'ratio':>7}")
    for key, before, after, ratio in compare_metrics(current, baseline):
        before_text = f"{before:.3f}" if before is not None else "-"
        after_text = f"{after:.3f}" if after is not None else "-"
        ratio_text = f"{ratio:.2f}" if ratio is not None else "-"
        line = f"{key:<40} {before_text:>10} {after_text:>10} {ratio_text:>7}"
        if is_slower(before, after, flag_ratio, min_delta_ms):
            if is_noisy(key):
                line = click.style(line + "  (noisy)", fg="yellow")
            else:
                slower.append(key)
                line = click.style(line, fg="red")
        click.echo(line)
    return slower


@click.command("perf")
@click.option("--port", "-p", default=DEFAULT_PORT, type=int, help=f"Server port (default: {DEFAULT_PORT})")
@click.option("--start-server", is_flag=True, help="Start the server if it is not running")
@click.option("--scene", "scenes", multiple=True, help="Scene to run (repeatable; default: all)")
@click.option("--iterations", default=3, type=int, help="Draws per operation in each scene (default: 3)")
@click.option("--scale", default=1, type=int, help="Multiplier for the expression loops (default: 1)")
@click.option("--repeats", default=3, type=int, help="Benchmark runs; the median is kept (default: 3)")
@click.option("--output", "-o", type=click.Path(), help="Save the medians and raw runs as JSON")
@click.option("--compare", "compare_path", type=click.Path(exists=True), help="Compare with a saved JSON result")
@click.option("--flag-ratio", default=1.25, type=float, help="Mark metrics slower than this ratio (default: 1.25)")
@click.option(
    "--min-delta-ms", default=1.0, type=float, help="Ignore slowdowns smaller than this many ms (default: 1.0)"
)
@click.option(
    "--fail-on-slower", is_flag=True, help="Exit with status 1 when a metric other than a noisy one is marked slower"
)
def perf_cmd(
    port: int,
    start_server: bool,
    scenes: Tuple[str, ...],
    iterations: int,
    scale: int,
    repeats: int,
    output: Optional[str],
    compare_path: Optional[str],
    flag_ratio: float,
    min_delta_ms: float,
    fail_on_slower: bool,
) -> None:
    """Time rendering and expression evaluation in headless Chrome."""
    options: Dict[str, Any] = {"iterations": iterations, "scale": scale}
    if scenes:
        options["scenes"] = list(scenes)
    runs, error, reused_server = _run_in_browser(port, start_server, max(1, repeats), options)
    if error:
        click.echo(click.style(f"Error: {error}", fg="red"), err=True)
        raise SystemExit(1)

    medians = median_metrics(runs)
    baseline = None
    if compare_path:
        saved = json.loads(Path(compare_path).read_text(encoding="utf-8"))
        baseline = saved.get("medians_ms", {})
        if saved.get("options") != options:
            click.echo(
                click.style(
                    f"Warning: the baseline ran with {saved.get('options')}, this run with {options}", fg="yellow"
                )
            )
    slower = _echo_table(medians, baseline, flag_ratio, min_delta_ms)

    if output:
        payload = {
            "commit": _git_commit(),
            "reused_server": reused_server,
            "options": options,
            "repeats": len(runs),
            "medians_ms": medians,
            "runs": runs,
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        click.echo(f"Saved {output}")
    if slower:
        click.echo(click.style(f"{len(slower)} metric(s) slower than {flag_ratio:.2f}x the baseline", fg="red"))
        if fail_on_slower:
            raise SystemExit(1)
