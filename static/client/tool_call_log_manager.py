"""Tool call log manager for the AI interface.

Manages the collapsible tool-call log dropdown that appears in the chat
when the AI executes tool calls. Tracks entries, builds DOM elements,
and updates the summary as tool calls accumulate.

Extracted from ``AIInterface`` to reduce god-class complexity while
preserving the identical public behaviour.
"""

from __future__ import annotations

from typing import Any

from browser import html

from result_processor import ResultProcessor


class ToolCallLogManager:
    """Manages the tool-call log dropdown UI and its backing state.

    Attributes:
        entries: Accumulated tool-call entry dicts for the current turn.
        element: The ``<details>`` DOM element (or ``None`` before first use).
        summary: The ``<summary>`` DOM element inside *element*.
        content: The container ``<div>`` holding individual entry rows.
    """

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.element: Any | None = None   # <details> element
        self.summary: Any | None = None   # <summary> element
        self.content: Any | None = None   # content container div

    # ── State management ────────────────────────────────────────

    def reset(self) -> None:
        """Reset all tool call log state for a new turn."""
        self.entries = []
        self.element = None
        self.summary = None
        self.content = None

    # ── Formatting helpers ──────────────────────────────────────

    def format_args_display(self, args: dict[str, Any]) -> str:
        """Format a tool call's arguments dict for compact display.

        Filters out the ``canvas`` key, truncates individual values to 30
        characters and the total string to 80 characters.
        """
        parts: list[str] = []
        for k, v in args.items():
            if k == "canvas":
                continue
            v_str = str(v)
            if len(v_str) > 30:
                v_str = v_str[:27] + "..."
            parts.append(f"{k}: {v_str}")
        result = ", ".join(parts)
        if len(result) > 80:
            result = result[:77] + "..."
        return result

    # ── DOM element creation ────────────────────────────────────

    def create_entry_element(self, entry: dict[str, Any]) -> Any:
        """Build the DOM element for a single tool call log entry."""
        div = html.DIV(Class="tool-call-entry")

        is_error = entry.get("is_error", False)
        status_class = "tool-call-status error" if is_error else "tool-call-status success"
        status_char = "\u2717" if is_error else "\u2713"
        status_span = html.SPAN(status_char, Class=status_class)
        div <= status_span

        name_span = html.SPAN(entry.get("name", ""), Class="tool-call-name")
        div <= name_span

        short_args = entry.get("args_display", "")
        full_args = entry.get("args_full", short_args)
        args_span = html.SPAN(f"({short_args})", Class="tool-call-args")
        div <= args_span

        # Show error message or result
        result_display = entry.get("result_display", "")
        result_full = entry.get("result_full", result_display)
        result_span: Any = None

        if is_error:
            error_msg = entry.get("error_message", "")
            if error_msg:
                err_span = html.SPAN(f" \u2192 {error_msg}", Class="tool-call-error-msg")
                div <= err_span
        elif result_display:
            result_span = html.SPAN(f" \u2192 {result_display}", Class="tool-call-result")
            div <= result_span

        # Click to toggle between truncated and full view
        def _toggle_expand(event: Any) -> None:
            try:
                if div.classList.contains("expanded"):
                    div.classList.remove("expanded")
                    args_span.text = f"({short_args})"
                    if result_span is not None and result_display:
                        result_span.text = f" \u2192 {result_display}"
                else:
                    div.classList.add("expanded")
                    args_span.text = f"({full_args})"
                    if result_span is not None and result_full:
                        result_span.text = f" \u2192 {result_full}"
            except Exception:
                pass

        div.bind("click", _toggle_expand)

        return div

    # ── Ensure / create the log dropdown ────────────────────────

    def ensure_element(
        self,
        stream_container: Any | None,
        stream_content: Any | None,
    ) -> None:
        """Create the tool-call-log ``<details>`` element if it doesn't exist yet.

        Args:
            stream_container: The outer message container DOM node. If ``None``
                the caller must create the stream message element first.
            stream_content: The chat-content ``<div>`` inside *stream_container*.
        """
        if self.element is not None:
            return

        details = html.DETAILS(Class="tool-call-log-dropdown")
        summary = html.SUMMARY("Using tools...", Class="tool-call-log-summary")
        content_div = html.DIV(Class="tool-call-log-content")
        details <= summary
        details <= content_div

        # Insert before the content element so it appears after reasoning but before text
        if stream_container is not None and stream_content is not None:
            try:
                stream_container.insertBefore(details, stream_content)
            except Exception:
                stream_container <= details
        elif stream_container is not None:
            stream_container <= details

        self.element = details
        self.summary = summary
        self.content = content_div

    # ── Adding entries ──────────────────────────────────────────

    def add_entries(self, tool_calls: list[dict[str, Any]], call_results: dict[str, Any]) -> None:
        """Record tool call entries and update the dropdown UI.

        .. note::

            The caller must ensure that ``ensure_element`` has been called
            (or the element already exists) before invoking this method.
            This method calls ``ensure_element`` itself as a convenience,
            but passes ``None`` containers — so the log dropdown will only
            be created if the caller has previously set up the element.

        Args:
            tool_calls: Raw tool call dicts from the AI response.
            call_results: Dict mapping result keys to their outcomes.
        """
        # ensure_element is a no-op when self.element is already set
        # When called from AIInterface, ensure_element is called beforehand
        # with the proper container references.
        if self.element is None:
            # Defensive: do nothing if the element was never created.
            # The caller (AIInterface) is responsible for calling
            # ensure_element with the right containers first.
            pass

        for call in tool_calls:
            function_name: str = call.get("function_name", "")
            args: dict[str, Any] = call.get("arguments", {})
            args_display = self.format_args_display(args)

            result_key = ResultProcessor._generate_result_key(function_name, args)

            # Special handling for evaluate_expression which uses expression as key
            if function_name == "evaluate_expression" and "expression" in args:
                expr = str(args.get("expression", "")).replace(" ", "")
                variables = args.get("variables")
                if variables and isinstance(variables, dict):
                    vars_str = ", ".join(f"{k}:{v}" for k, v in variables.items())
                    expr_key = f"{expr} for {vars_str}"
                else:
                    expr_key = expr
                result_value = call_results.get(expr_key, call_results.get(result_key, ""))
            else:
                result_value = call_results.get(result_key, call_results.get(function_name, ""))
            is_error = isinstance(result_value, str) and result_value.startswith("Error:")
            error_message = result_value if is_error else ""

            # Full untruncated args for the expanded view
            args_full = ", ".join(f"{k}: {v}" for k, v in args.items() if k != "canvas")

            # Format result for display (truncate if too long)
            result_display = ""
            if not is_error and result_value:
                result_str = str(result_value)
                if len(result_str) > 100:
                    result_display = result_str[:97] + "..."
                else:
                    result_display = result_str

            entry: dict[str, Any] = {
                "name": function_name,
                "args_display": args_display,
                "args_full": args_full,
                "is_error": is_error,
                "error_message": error_message,
                "result_display": result_display,
                "result_full": str(result_value) if result_value else "",
            }
            self.entries.append(entry)

            entry_el = self.create_entry_element(entry)
            if self.content is not None:
                self.content <= entry_el

        # Update summary with running count
        count = len(self.entries)
        if self.summary is not None:
            self.summary.text = f"Using tools... ({count} so far)"

    # ── Finalization ────────────────────────────────────────────

    def finalize(self) -> None:
        """Update the tool call log summary to its final state."""
        if not self.entries:
            return

        count = len(self.entries)
        error_count = sum(1 for e in self.entries if e.get("is_error"))

        label = f"Used {count} tool" if count == 1 else f"Used {count} tools"
        if error_count:
            label += f" ({error_count} failed)"

        if self.summary is not None:
            self.summary.text = label

        # Ensure collapsed — removeAttribute is reliable for boolean HTML attributes
        if self.element is not None:
            try:
                self.element.removeAttribute("open")
            except Exception:
                pass
