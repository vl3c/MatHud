"""Per-turn AI response metrics for the model workbench.

A user turn can span several model requests: the first answers the user's
message and each following one answers the results of the tool calls the model
made. The server attaches a per-request ``metrics`` record to every final
stream event (see ``static/response_metrics.py``). ``TurnMetricsCollector``
gathers those records together with the tool results the client executed and
produces one JSON-serializable summary per turn, kept in a small history for
the benchmark harness (``window.getMatHudLastTurnMetrics()``).

``format_metrics_footer`` and ``format_metrics_details`` turn a summary into
the muted footer and tooltip shown under the final assistant message.

Pure Python (no ``browser`` import), so it runs in the Brython test runner and
under CPython alike.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

TURN_METRICS_SCHEMA_VERSION = 1
MAX_TURN_HISTORY = 50
MAX_TOOL_ERROR_MESSAGES = 10
_USER_MESSAGE_PREVIEW_CHARS = 500
_TOOL_ERROR_PREVIEW_CHARS = 200

_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens", "total_tokens")

_RATE_SOURCE_LABELS = {
    "server_timings": "llama-server timings",
    "usage": "provider usage",
    "estimated": "estimated from text",
}


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _sum_optional(values: List[Any]) -> Optional[int]:
    """Sum the numeric values; None when no value was reported."""
    numbers = [_number(value) for value in values]
    reported = [number for number in numbers if number is not None]
    if not reported:
        return None
    return int(sum(reported))


def _is_error_result(result: Any) -> bool:
    return isinstance(result, str) and result.startswith("Error")


def short_model_name(model: Any) -> str:
    """Model label for the footer: file paths reduced to the bare name, without ``.gguf``."""
    name = str(model or "").replace("\\", "/").rstrip("/")
    name = name.rsplit("/", 1)[-1]
    if name.lower().endswith(".gguf"):
        name = name[: -len(".gguf")]
    return name


def format_seconds(seconds: Any) -> str:
    """``0.84 -> '0.8 s'``, ``12.3 -> '12 s'``, ``75 -> '1 m 15 s'``."""
    value = _number(seconds)
    if value is None:
        return "?"
    if value < 10:
        return f"{value:.1f} s"
    if value < 60:
        return f"{int(round(value))} s"
    minutes = int(value // 60)
    rest = int(round(value - minutes * 60))
    if rest == 60:
        minutes, rest = minutes + 1, 0
    return f"{minutes} m {rest} s"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def aggregate_turn(
    requests: List[Dict[str, Any]],
    tool_results: List[Dict[str, Any]],
    wall_time_s: Optional[float],
    outcome: str,
) -> Dict[str, Any]:
    """Build the turn summary from its per-request metrics and executed tool results.

    Args:
        requests: Per-request ``metrics`` records, in request order.
        tool_results: Traced tool calls the client executed (``function_name``,
            ``result``, ``is_error``).
        wall_time_s: Client-measured time from sending the message to the final answer.
        outcome: How the turn ended (``stop``, ``error``, ``stopped``, ``timeout``).
    """
    last = requests[-1] if requests else {}
    first = requests[0] if requests else {}
    tool_errors = [call for call in tool_results if call.get("is_error") or _is_error_result(call.get("result"))]
    summary: Dict[str, Any] = {
        "schema": TURN_METRICS_SCHEMA_VERSION,
        "provider": last.get("provider"),
        "model": last.get("model"),
        "outcome": outcome,
        "wall_time_s": None if wall_time_s is None else round(wall_time_s, 3),
        "model_time_s": _round_optional(_sum_float([r.get("total_latency_s") for r in requests])),
        "time_to_first_token_s": first.get("time_to_first_token_s"),
        "requests": len(requests),
        "tool_calls": sum(int(_number(r.get("tool_calls")) or 0) for r in requests),
        "tool_executions": len(tool_results),
        "tool_errors": len(tool_errors),
        "tool_error_messages": [_tool_error_message(call) for call in tool_errors[:MAX_TOOL_ERROR_MESSAGES]],
        "output_tokens": _sum_optional([r.get("output_tokens") for r in requests]),
        "output_tokens_estimated": any(bool(r.get("output_tokens_estimated")) for r in requests),
        "output_tokens_per_s": _aggregate_rate(requests),
        "per_request": requests,
    }
    for field in _TOKEN_FIELDS:
        summary[field] = _sum_optional([r.get(field) for r in requests])
    return summary


def _sum_float(values: List[Any]) -> Optional[float]:
    numbers = [_number(value) for value in values]
    reported = [number for number in numbers if number is not None]
    return sum(reported) if reported else None


def _round_optional(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 3)


def _aggregate_rate(requests: List[Dict[str, Any]]) -> Optional[float]:
    """Tokens/s over the whole turn: total output tokens over total generation time."""
    tokens = 0.0
    seconds = 0.0
    for request in requests:
        count = _number(request.get("output_tokens"))
        rate = _number(request.get("output_tokens_per_s"))
        if count is None or rate is None or rate <= 0:
            continue
        tokens += count
        seconds += count / rate
    if seconds <= 0:
        return None
    return round(tokens / seconds, 2)


def _tool_error_message(call: Dict[str, Any]) -> str:
    name = str(call.get("function_name") or "tool")
    result = str(call.get("result") or "Error")
    if len(result) > _TOOL_ERROR_PREVIEW_CHARS:
        result = result[:_TOOL_ERROR_PREVIEW_CHARS] + "..."
    return f"{name}: {result}"


def format_metrics_footer(turn: Optional[Dict[str, Any]]) -> str:
    """One-line footer, e.g. ``qwen3.8-27b · 4.2 s · first token 0.8 s · 38 tok/s · 2 requests · 3 tool calls``."""
    if not turn or not turn.get("requests"):
        return ""
    parts: List[str] = []
    model = short_model_name(turn.get("model"))
    if model:
        parts.append(model)
    wall = turn.get("wall_time_s")
    parts.append(format_seconds(wall if wall is not None else turn.get("model_time_s")))
    ttft = turn.get("time_to_first_token_s")
    if ttft is not None:
        parts.append(f"first token {format_seconds(ttft)}")
    rate = _number(turn.get("output_tokens_per_s"))
    if rate is not None:
        prefix = "~" if turn.get("output_tokens_estimated") else ""
        parts.append(f"{prefix}{int(round(rate))} tok/s")
    requests = int(turn.get("requests") or 0)
    if requests > 1:
        parts.append(_plural(requests, "request"))
    tool_calls = int(turn.get("tool_calls") or 0)
    if tool_calls:
        parts.append(_plural(tool_calls, "tool call"))
    tool_errors = int(turn.get("tool_errors") or 0)
    if tool_errors:
        parts.append(_plural(tool_errors, "tool error"))
    return " · ".join(parts)


def format_metrics_details(turn: Optional[Dict[str, Any]]) -> str:
    """Multi-line tooltip text with token counts, the per-request breakdown and tool errors."""
    if not turn or not turn.get("requests"):
        return ""
    lines = [
        f"Provider: {turn.get('provider') or '?'} · model: {turn.get('model') or '?'}",
        _timing_line(turn),
        _tokens_line(turn),
        _rate_line(turn),
        f"Requests: {turn.get('requests')} · tool calls: {turn.get('tool_calls')} · "
        f"tool errors: {turn.get('tool_errors')}",
    ]
    for index, request in enumerate(turn.get("per_request") or [], start=1):
        lines.append(_request_line(index, request))
    messages = turn.get("tool_error_messages") or []
    if messages:
        lines.append("Tool errors:")
        lines.extend(f"- {message}" for message in messages)
    return "\n".join(line for line in lines if line)


def _timing_line(turn: Dict[str, Any]) -> str:
    parts = [f"Wall time {format_seconds(turn.get('wall_time_s'))}"]
    if turn.get("model_time_s") is not None:
        parts.append(f"model time {format_seconds(turn.get('model_time_s'))}")
    if turn.get("time_to_first_token_s") is not None:
        parts.append(f"first token {format_seconds(turn.get('time_to_first_token_s'))}")
    return " · ".join(parts)


def _tokens_line(turn: Dict[str, Any]) -> str:
    parts: List[str] = []
    if turn.get("prompt_tokens") is not None:
        cached = turn.get("cached_tokens")
        suffix = f" (cached {cached})" if cached else ""
        parts.append(f"prompt {turn.get('prompt_tokens')}{suffix}")
    if turn.get("completion_tokens") is not None:
        parts.append(f"completion {turn.get('completion_tokens')}")
    elif turn.get("output_tokens") is not None:
        approx = "~" if turn.get("output_tokens_estimated") else ""
        parts.append(f"output {approx}{turn.get('output_tokens')}")
    if turn.get("reasoning_tokens"):
        parts.append(f"reasoning {turn.get('reasoning_tokens')}")
    if turn.get("total_tokens") is not None:
        parts.append(f"total {turn.get('total_tokens')}")
    return "Tokens: " + " · ".join(parts) if parts else "Tokens: not reported"


def _rate_line(turn: Dict[str, Any]) -> str:
    rate = _number(turn.get("output_tokens_per_s"))
    if rate is None:
        return ""
    sources = {r.get("tokens_per_s_source") for r in turn.get("per_request") or [] if r.get("tokens_per_s_source")}
    labels = [_RATE_SOURCE_LABELS.get(str(source), str(source)) for source in sorted(sources)]
    source_text = f" ({', '.join(labels)})" if labels else ""
    return f"Output speed: {rate:g} tok/s{source_text}"


def _request_line(index: int, request: Dict[str, Any]) -> str:
    parts = [format_seconds(request.get("total_latency_s"))]
    if request.get("time_to_first_token_s") is not None:
        parts.append(f"first token {format_seconds(request.get('time_to_first_token_s'))}")
    if request.get("prompt_tokens") is not None:
        parts.append(f"prompt {request.get('prompt_tokens')}")
    if request.get("output_tokens") is not None:
        approx = "~" if request.get("output_tokens_estimated") else ""
        parts.append(f"output {approx}{request.get('output_tokens')}")
    rate = _number(request.get("output_tokens_per_s"))
    if rate is not None:
        parts.append(f"{rate:g} tok/s")
    tool_calls = int(_number(request.get("tool_calls")) or 0)
    if tool_calls:
        parts.append(_plural(tool_calls, "tool call"))
    parts.append(f"finish {request.get('finish_reason') or '?'}")
    if request.get("error"):
        parts.append(f"error: {request.get('error')}")
    return f"Request {index}: " + ", ".join(parts)


class TurnMetricsCollector:
    """Aggregates per-request metrics into per-turn summaries with a bounded history.

    Args:
        clock_ms: Millisecond clock used for the client-side wall time
            (``window.performance.now`` in the browser).
        max_history: Number of completed turns kept.
    """

    def __init__(self, clock_ms: Callable[[], float], max_history: int = MAX_TURN_HISTORY) -> None:
        self._clock_ms = clock_ms
        self._max_history = max_history
        self._history: List[Dict[str, Any]] = []
        self._next_turn_id = 1
        self._active = False
        self._started_ms: Optional[float] = None
        self._user_message: Optional[str] = None
        self._requests: List[Dict[str, Any]] = []
        self._tool_results: List[Dict[str, Any]] = []

    @property
    def is_active(self) -> bool:
        """True while a turn has started and not finished."""
        return self._active

    def start_turn(self, user_message: Optional[str] = None) -> None:
        """Begin a new turn (discarding an unfinished one)."""
        self._active = True
        self._started_ms = self._clock_ms()
        self._user_message = user_message
        self._requests = []
        self._tool_results = []

    def record_request(self, metrics: Any) -> None:
        """Add one request's ``metrics`` from a final stream event.

        Ignored when absent or when no turn is active (e.g. a late event after the user stopped).
        """
        if self._active and isinstance(metrics, dict):
            self._requests.append(dict(metrics))

    def record_tool_results(self, traced_calls: Any) -> None:
        """Add the tool calls the client executed for this turn."""
        if not self._active or not isinstance(traced_calls, list):
            return
        for call in traced_calls:
            if isinstance(call, dict):
                self._tool_results.append(
                    {
                        "function_name": call.get("function_name"),
                        "result": call.get("result"),
                        "is_error": bool(call.get("is_error")),
                    }
                )

    def finish_turn(self, outcome: str = "stop") -> Optional[Dict[str, Any]]:
        """Close the active turn, store its summary in the history and return it."""
        if not self._active:
            return None
        wall_time_s: Optional[float] = None
        if self._started_ms is not None:
            wall_time_s = max(self._clock_ms() - self._started_ms, 0.0) / 1000.0
        summary = aggregate_turn(self._requests, self._tool_results, wall_time_s, outcome)
        summary["turn_id"] = self._next_turn_id
        summary["user_message"] = self._preview(self._user_message)
        self._next_turn_id += 1
        self._active = False
        self._history.append(summary)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        return summary

    def last_turn(self) -> Optional[Dict[str, Any]]:
        """The most recently completed turn, or None."""
        return self._history[-1] if self._history else None

    def history(self) -> List[Dict[str, Any]]:
        """Completed turns, oldest first."""
        return list(self._history)

    def clear(self) -> None:
        """Forget the history (turn ids keep increasing)."""
        self._history = []

    @staticmethod
    def _preview(text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        return text if len(text) <= _USER_MESSAGE_PREVIEW_CHARS else text[:_USER_MESSAGE_PREVIEW_CHARS] + "..."
