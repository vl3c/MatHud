"""Live-run settings and spend guards: the pinned server environment, model checks and the plan.

Pure functions, so the guards are tested without a server, a browser or a model:

- ``server_env`` pins every setting that changes what the model sees, so a local
  ``.env`` changes nothing but ``LOCAL_AGENT_BASE_URL`` and the OpenRouter key.
  It also blanks the provider keys the run does not use: python-dotenv never
  overrides a variable that is already set, so the server cannot reach a paid
  provider by falling back to it.
- ``check_models`` is the spend guard before the first message: an id the
  server has not registered under the run's provider falls back to the OpenAI
  provider (``routes.py``), so it aborts the run.
- ``live_plan`` gives the planned turns and requests, and an OpenRouter cost estimate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from static.config import WORKSPACES_DIR_ENV
from static.model_prices import ESTIMATED_COMPLETION_TOKENS, PRICES_AS_OF, cost_usd
from static.providers.local import DEFAULT_REASONING_EFFORT, REASONING_EFFORT_CHOICES, REASONING_EFFORT_ENV
from static.providers.openrouter_api import MAX_RETRIES_ENV

PROVIDERS = ("local", "openrouter")
TOOL_EXPOSURES = ("search", "full")
CANVAS_FORMATS = ("text", "min_json", "json")
TOOL_SEARCH_MODES = ("local", "api", "hybrid")
# The app's default search mode for local runs. OpenRouter runs search locally by
# default: an api or hybrid search sends model requests the turn metrics do not
# count, so the request cap could not see them.
DEFAULT_TOOL_SEARCH_MODE = {"local": "hybrid", "openrouter": "local"}
DEFAULT_MAX_REQUESTS = 250
LOCAL_REASONING_EFFORT_CHOICES = REASONING_EFFORT_CHOICES
DEFAULT_LOCAL_REASONING_EFFORT = DEFAULT_REASONING_EFFORT

# Provider groups of /api/available_models each provider's models must appear under.
REGISTRATION_GROUPS = {"local": ("local_agent",), "openrouter": ("openrouter_paid", "openrouter_free")}
PROVIDER_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")
_KEEP_KEYS = {"local": (), "openrouter": ("OPENROUTER_API_KEY",)}
# Canvas summary settings the canvas benchmark also pins; empty means the app's default.
_DEFAULTED_ENV = ("AI_CANVAS_SUMMARY_MODE", "AI_CANVAS_HYBRID_FULL_MAX_BYTES", "AI_CANVAS_SUMMARY_TELEMETRY")
# Rough prompt size per request beyond the system prompt, tools and canvas: the
# user message, earlier turns, tool results and [canvas changes].
HISTORY_ALLOWANCE_TOKENS = 2000
# Tools a search_tools call typically loads into the request (search exposure).
SEARCH_LOADED_TOOLS = 10
# The cloud providers' canvas budget when none is pinned (and the size counted for "unlimited").
DEFAULT_CLOUD_CANVAS_BUDGET = 4000


class GuardError(Exception):
    """A spend guard refused the run before anything was sent."""


@dataclass
class LiveSettings:
    """What a live run pins on its server and records in its config."""

    provider: str = "local"
    tool_exposure: str = "search"
    canvas_format: str = "text"
    canvas_budget: Optional[int] = None
    tool_search_mode: Optional[str] = None
    local_reasoning_effort: str = DEFAULT_LOCAL_REASONING_EFFORT

    @property
    def search_mode(self) -> str:
        return self.tool_search_mode or DEFAULT_TOOL_SEARCH_MODE[self.provider]

    def validate(self) -> None:
        """Refuse combinations a guard cannot police."""
        if self.provider not in PROVIDERS:
            raise GuardError(f"unknown provider {self.provider!r}")
        if self.provider == "openrouter" and self.search_mode != "local":
            raise GuardError(
                "--tool-search-mode api or hybrid sends tool-search requests the request cap cannot count; "
                "OpenRouter runs use local search"
            )
        if self.local_reasoning_effort not in LOCAL_REASONING_EFFORT_CHOICES:
            raise GuardError(f"unknown reasoning effort {self.local_reasoning_effort!r}")


def server_env(settings: LiveSettings, workspaces_dir: str) -> dict[str, str]:
    """The environment a live run's server starts with (see the module docstring)."""
    from static.openai_api_base import CANVAS_BUDGET_ENV, CANVAS_FORMAT_ENV, TOOL_EXPOSURE_ENV

    env = {
        TOOL_EXPOSURE_ENV: settings.tool_exposure,
        CANVAS_FORMAT_ENV: settings.canvas_format,
        CANVAS_BUDGET_ENV: "" if settings.canvas_budget is None else str(settings.canvas_budget),
        "TOOL_SEARCH_MODE": settings.search_mode,
        WORKSPACES_DIR_ENV: workspaces_dir,
        "REQUIRE_AUTH": "false",
        REASONING_EFFORT_ENV: settings.local_reasoning_effort if settings.provider == "local" else "",
        MAX_RETRIES_ENV: "0",
    }
    env.update({name: "" for name in _DEFAULTED_ENV})
    env.update({key: "" for key in PROVIDER_KEYS if key not in _KEEP_KEYS[settings.provider]})
    return env


def recorded_env(env: dict[str, str]) -> dict[str, str]:
    """``env`` as the run config records it: blanked keys listed by name, never a value."""
    return {name: ("<blank>" if name in PROVIDER_KEYS else value) for name, value in env.items()}


def registered_models(available: Any, provider: str) -> list[str]:
    """Model ids ``/api/available_models`` lists for ``provider``."""
    ids: list[str] = []
    if not isinstance(available, dict):
        return ids
    for group in REGISTRATION_GROUPS[provider]:
        for entry in available.get(group) or []:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                ids.append(entry["id"])
    return ids


def check_models(available: Any, provider: str, requested: list[str]) -> list[str]:
    """The models to run: ``requested``, or for a local run every served model when none is given.

    Raises GuardError when a requested id is not registered under the provider
    (it would fall back to a paid provider), or when there is nothing to run.
    """
    registered = registered_models(available, provider)
    groups = " or ".join(REGISTRATION_GROUPS[provider])
    if not requested:
        if provider != "local":
            raise GuardError("--provider openrouter needs --models")
        if not registered:
            raise GuardError("the server lists no local_agent model: is llama-server running and serving a model?")
        return registered
    missing = [model for model in requested if model not in registered]
    if missing:
        hint = (
            "is OPENROUTER_API_KEY set for the server, and is the id an OpenRouter model the app knows?"
            if provider == "openrouter"
            else "is llama-server serving it?"
        )
        raise GuardError(
            f"not registered under {groups} in /api/available_models: {', '.join(missing)} "
            f"(an unregistered id falls back to a paid provider; {hint}). Registered: {', '.join(registered) or 'none'}"
        )
    return list(dict.fromkeys(requested))


def check_request_cap(planned: int, max_requests: int) -> None:
    """Refuse a run whose planned requests exceed the cap, before anything is sent."""
    if planned > max_requests:
        raise GuardError(
            f"the run may send {planned} requests (turns x per-turn request cap x models x repeats), "
            f"more than --max-requests {max_requests}; nothing was sent"
        )


def estimated_prompt_tokens(settings: LiveSettings) -> int:
    """A rough prompt size per request: system prompt, tool schemas, canvas budget and history."""
    from static.functions_definitions import FUNCTIONS
    from static.openai_api_base import SEARCH_MODE_TOOLS, OpenAIAPIBase, build_developer_message
    from static.token_estimation import estimate_tokens_from_text

    system = build_developer_message(settings.canvas_format)
    if settings.tool_exposure == "search":
        system += " " + OpenAIAPIBase.SEARCH_MODE_MSG
        tools = list(SEARCH_MODE_TOOLS)
        # Upper end: the largest tools are the ones a search may load.
        sizes = sorted((estimate_tokens_from_text(json.dumps(tool)) for tool in FUNCTIONS), reverse=True)
        loaded = sum(sizes[:SEARCH_LOADED_TOOLS])
    else:
        tools, loaded = list(FUNCTIONS), 0
    tool_tokens = estimate_tokens_from_text(json.dumps(tools)) + loaded
    budget = settings.canvas_budget or DEFAULT_CLOUD_CANVAS_BUDGET
    return int(estimate_tokens_from_text(system) + tool_tokens + budget + HISTORY_ALLOWANCE_TOKENS)


def live_plan(
    *,
    scenarios: int,
    turns: int,
    models: list[str],
    repeats: int,
    planned: int,
    settings: LiveSettings,
    max_requests: Optional[int],
) -> dict[str, Any]:
    """The dry-run plan: turns, the request ceiling and, for OpenRouter, a cost ceiling per model."""
    plan: dict[str, Any] = {
        "provider": settings.provider,
        "models": models,
        "scenarios": scenarios,
        "turns": turns * repeats * len(models),
        "repeats": repeats,
        "planned_requests": planned,
        "max_requests": max_requests,
        "within_cap": max_requests is None or planned <= max_requests,
    }
    if settings.provider == "openrouter":
        prompt = estimated_prompt_tokens(settings)
        per_model = planned // max(len(models), 1)
        costs = {
            model: cost_usd(model, prompt * per_model, ESTIMATED_COMPLETION_TOKENS * per_model) for model in models
        }
        plan.update(
            estimated_prompt_tokens_per_request=prompt,
            estimated_completion_tokens_per_request=ESTIMATED_COMPLETION_TOKENS,
            estimated_cost_usd={m: (None if c is None else round(c, 4)) for m, c in costs.items()},
            prices_as_of=PRICES_AS_OF,
        )
    return plan
