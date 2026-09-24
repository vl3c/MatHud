"""
Per-request response metrics for the model workbench.

Every provider path (Chat Completions, Responses, Anthropic, OpenRouter,
LocalAgent) creates a ``ResponseMetricsTracker`` when it sends a request to the
model, marks the first streamed output, feeds it whatever usage figures the
provider reports and calls ``finish`` to get a JSON-serializable
``ResponseMetrics`` record. The record is attached to the request's final
stream event (``metrics``), returned by ``/send_message`` and logged as a
``response_metrics {...}`` JSON line, so a benchmark can compare models by
speed and token use.

Only measurements are added here; nothing that reaches the model changes.
Metrics are a side channel: the recorders and ``finish`` never raise, and
non-finite or non-numeric figures are dropped, so a bad usage report cannot turn
a good answer into an error.

Field sources:

- ``time_to_first_token_s``: first streamed content or tool-call delta.
- ``output_tokens_per_s``: llama-server ``timings.predicted_per_second`` when
  present, otherwise completion tokens divided by the generation window (first
  streamed output, reasoning included, to the end of the request), otherwise an
  estimate from the streamed text (``output_tokens_estimated`` is then True).
  When the output arrived in a single chunk or the window is shorter than
  ``MIN_GENERATION_WINDOW_S`` (a buffered reply or tool call), the window says
  nothing about generation speed, so the whole request is used instead and
  ``tokens_per_s_source`` gets a ``_request`` suffix (``usage_request``,
  ``estimated_request``). The same happens when the usage reports reasoning
  tokens but no reasoning was streamed (hidden thinking happened before the
  first streamed output).
- ``prompt_tokens`` / ``completion_tokens`` / ``cached_tokens`` /
  ``reasoning_tokens``: the provider's usage report, when it sends one.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TypedDict

from static.token_estimation import estimate_tokens_from_text

_logger = logging.getLogger("mathud")

METRICS_SCHEMA_VERSION = 1

# Asks a Chat Completions stream for a trailing usage chunk (OpenAI, OpenRouter, llama-server).
STREAM_USAGE_OPTIONS: Dict[str, bool] = {"include_usage": True}

# Shorter generation windows are delivery bursts, not generation; the rate then uses the whole request.
MIN_GENERATION_WINDOW_S = 0.25

# llama-server ``timings`` keys worth keeping (llama.cpp tools/server).
LLAMA_TIMING_KEYS = (
    "cache_n",
    "prompt_n",
    "prompt_ms",
    "prompt_per_token_ms",
    "prompt_per_second",
    "predicted_n",
    "predicted_ms",
    "predicted_per_token_ms",
    "predicted_per_second",
    "draft_n",
    "draft_n_accepted",
)


class ResponseMetrics(TypedDict, total=False):
    """One model request's measurements (JSON-serializable)."""

    schema: int
    provider: str
    model: str
    api: str
    streamed: bool
    request_started_at: float
    time_to_first_token_s: Optional[float]
    total_latency_s: float
    output_tokens: Optional[int]
    output_tokens_estimated: bool
    output_tokens_per_s: Optional[float]
    tokens_per_s_source: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    cached_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    total_tokens: Optional[int]
    prompt_tokens_per_s: Optional[float]
    server_timings: Dict[str, float]
    tool_calls: int
    finish_reason: str
    error: str
    # Added by the routes: "user_message" (first request of a turn) or "tool_results".
    request_kind: str


def _read(source: Any, name: str) -> Any:
    """Read ``name`` from a dict, an attribute or a pydantic model's extra fields (None on failure)."""
    if source is None:
        return None
    try:
        if isinstance(source, Mapping):
            return source.get(name)
        value = getattr(source, name, None)
        if value is not None:
            return value
        extra = getattr(source, "model_extra", None)
        if isinstance(extra, Mapping):
            return extra.get(name)
    except Exception:
        return None
    return None


def _as_float(value: Any) -> Optional[float]:
    """A finite number as float; None for NaN, infinity, booleans and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> Optional[int]:
    """A finite number as int; None for NaN, infinity, booleans and non-numbers."""
    number = _as_float(value)
    return None if number is None else int(number)


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(value, digits)


def usage_from_chat_completions(usage: Any) -> Dict[str, Optional[int]]:
    """Token counts from a Chat Completions ``usage`` object (OpenAI, OpenRouter, llama-server)."""
    prompt_details = _read(usage, "prompt_tokens_details")
    completion_details = _read(usage, "completion_tokens_details")
    return {
        "prompt_tokens": _as_int(_read(usage, "prompt_tokens")),
        "completion_tokens": _as_int(_read(usage, "completion_tokens")),
        "total_tokens": _as_int(_read(usage, "total_tokens")),
        "cached_tokens": _as_int(_read(prompt_details, "cached_tokens")),
        "reasoning_tokens": _as_int(_read(completion_details, "reasoning_tokens")),
    }


def usage_from_responses(usage: Any) -> Dict[str, Optional[int]]:
    """Token counts from a Responses API ``usage`` object."""
    input_details = _read(usage, "input_tokens_details")
    output_details = _read(usage, "output_tokens_details")
    return {
        "prompt_tokens": _as_int(_read(usage, "input_tokens")),
        "completion_tokens": _as_int(_read(usage, "output_tokens")),
        "total_tokens": _as_int(_read(usage, "total_tokens")),
        "cached_tokens": _as_int(_read(input_details, "cached_tokens")),
        "reasoning_tokens": _as_int(_read(output_details, "reasoning_tokens")),
    }


def usage_from_anthropic(usage: Any) -> Dict[str, Optional[int]]:
    """Token counts from an Anthropic ``usage`` object.

    Anthropic reports uncached input separately from cache reads and cache
    writes, so the prompt total is their sum and ``cached_tokens`` the reads.
    """
    input_tokens = _as_int(_read(usage, "input_tokens"))
    cache_read = _as_int(_read(usage, "cache_read_input_tokens"))
    cache_write = _as_int(_read(usage, "cache_creation_input_tokens"))
    prompt_tokens: Optional[int] = None
    if input_tokens is not None:
        prompt_tokens = input_tokens + (cache_read or 0) + (cache_write or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": _as_int(_read(usage, "output_tokens")),
        "cached_tokens": cache_read,
    }


def llama_timings_from(source: Any) -> Optional[Dict[str, float]]:
    """The llama-server ``timings`` block of a chunk or response, if it has one."""
    timings = _read(source, "timings")
    if timings is None:
        return None
    result: Dict[str, float] = {}
    for key in LLAMA_TIMING_KEYS:
        value = _as_float(_read(timings, key))
        if value is not None:
            result[key] = value
    return result or None


def reasoning_text_from_delta(delta: Any) -> str:
    """Reasoning text of a Chat Completions stream delta, or "".

    OpenRouter streams it as ``delta.reasoning`` and llama-server as
    ``delta.reasoning_content``; the openai SDK keeps both as extra fields.
    """
    for name in ("reasoning", "reasoning_content"):
        text = _read(delta, name)
        if isinstance(text, str) and text:
            return text
    return ""


def rejects_stream_usage_option(exc: BaseException) -> bool:
    """True when a request error says the server does not accept ``stream_options``."""
    if getattr(exc, "status_code", None) not in (400, 422):
        return False
    text = f"{exc} {getattr(exc, 'body', '')}"
    return "stream_options" in text or "include_usage" in text


def create_stream_requesting_usage(create: Callable[..., Any], request_usage: bool, **kwargs: Any) -> Tuple[Any, bool]:
    """Start a streamed Chat Completions request that asks for a trailing usage chunk.

    ``create`` is ``client.chat.completions.create`` and ``kwargs`` its other
    arguments. A strict OpenAI-compatible server may reject ``stream_options``
    with a 400; the request is then sent again without it. Returns the stream
    and whether later requests should keep asking for usage, which the caller
    passes back as ``request_usage``.
    """
    if not request_usage:
        return create(**kwargs), False
    try:
        return create(stream_options=dict(STREAM_USAGE_OPTIONS), **kwargs), True
    except Exception as exc:
        if not rejects_stream_usage_option(exc):
            raise
        _logger.warning(f"Server rejected stream_options; streaming without usage reports: {exc}")
        return create(**kwargs), False


def record_chat_completions_usage(source: Any, tracker: "ResponseMetricsTracker") -> bool:
    """Feed a Chat Completions chunk's or response's usage and llama-server timings to ``tracker``.

    Returns True when ``source`` carried a usage report.
    """
    usage = _read(source, "usage")
    try:
        if usage is not None:
            tracker.record_usage(usage_from_chat_completions(usage))
        tracker.record_server_timings(llama_timings_from(source))
    except Exception as exc:
        _logger.debug(f"Ignoring an unreadable usage report: {exc}")
    return usage is not None


class ResponseMetricsTracker:
    """Collects the measurements of one model request.

    Args:
        provider: Provider key (``openai``, ``anthropic``, ``openrouter``, ``local_agent``).
        model: Model identifier sent to the provider.
        api: Wire API used (``chat_completions``, ``responses``, ``anthropic_messages``).
        streamed: False for the non-streaming ``/send_message`` path.
        clock: Monotonic clock in seconds (injectable for tests).
        wall_clock: Epoch clock in seconds for ``request_started_at``.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api: str,
        streamed: bool = True,
        clock: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api = api
        self._streamed = streamed
        self._clock = clock
        self._started_at_wall = wall_clock()
        self._started = clock()
        self._first_token: Optional[float] = None
        self._first_output: Optional[float] = None
        self._output_marks = 0
        self._reasoning_seen = False
        self._usage: Dict[str, Optional[int]] = {}
        self._server_timings: Optional[Dict[str, float]] = None
        self._output_text_parts: List[str] = []

    def mark_output(self, kind: str = "content") -> None:
        """Record streamed output; ``content`` and ``tool_call`` count toward the first token."""
        now = self._clock()
        self._output_marks += 1
        if kind == "reasoning":
            self._reasoning_seen = True
        if self._first_output is None:
            self._first_output = now
        if kind != "reasoning" and self._first_token is None:
            self._first_token = now

    def add_output_text(self, text: str) -> None:
        """Keep generated text (content or tool arguments) for the token estimate fallback."""
        if text and isinstance(text, str):
            self._output_text_parts.append(text)

    def record_usage(self, usage: Mapping[str, Optional[int]]) -> None:
        """Merge provider-reported token counts; later non-empty values win (bad values are ignored)."""
        if not isinstance(usage, Mapping):
            return
        for key, value in usage.items():
            count = _as_int(value)
            if count is not None:
                self._usage[str(key)] = count

    def record_server_timings(self, timings: Optional[Mapping[str, float]]) -> None:
        """Keep a llama-server ``timings`` block (the last one wins; non-finite values are dropped)."""
        if not isinstance(timings, Mapping):
            return
        numeric = {str(key): _as_float(value) for key, value in timings.items()}
        finite = {key: value for key, value in numeric.items() if value is not None}
        if finite:
            self._server_timings = finite

    def finish(self, finish_reason: Optional[str], tool_calls: int, error: Optional[str] = None) -> ResponseMetrics:
        """Close the request and return its metrics record.

        Never raises: if building the record fails, a minimal record whose
        ``error`` says so is returned instead.
        """
        try:
            return self._build_record(finish_reason, tool_calls, error)
        except Exception as exc:
            _logger.warning(f"Response metrics failed: {exc}")
            failure = f"metrics unavailable: {exc}"
            return {
                "schema": METRICS_SCHEMA_VERSION,
                "provider": self._provider,
                "model": self._model,
                "api": self._api,
                "streamed": self._streamed,
                "finish_reason": str(finish_reason or "stop"),
                "error": f"{error}; {failure}" if error else failure,
            }

    def _build_record(self, finish_reason: Optional[str], tool_calls: int, error: Optional[str]) -> ResponseMetrics:
        total = max(self._clock() - self._started, 0.0)
        ttft = None if self._first_token is None else self._first_token - self._started
        metrics: ResponseMetrics = {
            "schema": METRICS_SCHEMA_VERSION,
            "provider": self._provider,
            "model": self._model,
            "api": self._api,
            "streamed": self._streamed,
            "request_started_at": round(self._started_at_wall, 3),
            "time_to_first_token_s": _round(ttft),
            "total_latency_s": round(total, 3),
            "prompt_tokens": self._usage.get("prompt_tokens"),
            "completion_tokens": self._usage.get("completion_tokens"),
            "cached_tokens": self._usage.get("cached_tokens"),
            "reasoning_tokens": self._usage.get("reasoning_tokens"),
            "total_tokens": self._total_tokens(),
            "tool_calls": _as_int(tool_calls) or 0,
            "finish_reason": finish_reason or "stop",
        }
        self._add_throughput(metrics, total)
        if self._server_timings:
            metrics["server_timings"] = self._server_timings
            metrics["prompt_tokens_per_s"] = _round(self._server_timings.get("prompt_per_second"), 2)
            if metrics["cached_tokens"] is None:
                metrics["cached_tokens"] = _as_int(self._server_timings.get("cache_n"))
        if error:
            metrics["error"] = error
        return metrics

    def _total_tokens(self) -> Optional[int]:
        total = self._usage.get("total_tokens")
        if total is not None:
            return total
        prompt = self._usage.get("prompt_tokens")
        completion = self._usage.get("completion_tokens")
        if prompt is None or completion is None:
            return None
        return prompt + completion

    def _add_throughput(self, metrics: ResponseMetrics, total: float) -> None:
        """Fill output_tokens, the estimated flag and tokens/s with the best available source."""
        completion = self._usage.get("completion_tokens")
        timings = self._server_timings or {}
        predicted_n = _as_int(timings.get("predicted_n"))
        output_tokens = completion if completion is not None else predicted_n
        estimated = False
        if output_tokens is None:
            output_tokens = estimate_tokens_from_text("".join(self._output_text_parts))
            estimated = True
        metrics["output_tokens"] = output_tokens
        metrics["output_tokens_estimated"] = estimated

        server_rate = _as_float(timings.get("predicted_per_second"))
        if server_rate is not None and server_rate > 0:
            metrics["output_tokens_per_s"] = round(server_rate, 2)
            metrics["tokens_per_s_source"] = "server_timings"
            return
        window, whole_request = self._generation_window(total)
        if output_tokens and window > 0:
            metrics["output_tokens_per_s"] = round(output_tokens / window, 2)
            source = "estimated" if estimated else "usage"
            metrics["tokens_per_s_source"] = f"{source}_request" if whole_request else source
            return
        metrics["output_tokens_per_s"] = None
        metrics["tokens_per_s_source"] = None

    def _generation_window(self, total: float) -> Tuple[float, bool]:
        """Seconds spent generating and whether that is the whole request.

        The window runs from the first streamed output to the end. It falls back
        to the whole request when nothing was streamed, when the output came in
        one chunk, when it is shorter than ``MIN_GENERATION_WINDOW_S`` or when
        the usage counts reasoning that was never streamed: the tokens were then
        generated before the first streamed output.
        """
        if self._first_output is None or self._output_marks < 2:
            return total, True
        if (self._usage.get("reasoning_tokens") or 0) > 0 and not self._reasoning_seen:
            return total, True
        window = total - (self._first_output - self._started)
        if window < MIN_GENERATION_WINDOW_S:
            return total, True
        return window, False


def tool_call_argument_text(tool_calls: List[Dict[str, Any]]) -> str:
    """Serialized tool names and arguments, for estimating generated tokens."""
    parts: List[str] = []
    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        if isinstance(function, dict):
            parts.append(str(function.get("name") or ""))
            arguments = function.get("arguments")
            try:
                parts.append(arguments if isinstance(arguments, str) else json.dumps(arguments, default=str))
            except (TypeError, ValueError):
                continue  # e.g. circular arguments; the estimate just misses them
    return "".join(parts)
