# MatHud Agent Direction

## AI-First Product Philosophy
- MatHud is an AI-augmented application where users communicate intent through chat.
- Agents should execute workflows end-to-end via tool calls and update the HUD canvas as the visual output.
- Do not assume gesture-driven workflows are primary; treat them as optional support paths.

## Implementation Priorities
- Prioritize deterministic execution and state integrity (atomic actions, rollback, reliable undo/redo).
- Prefer explicit, explainable action reporting (what was resolved, what changed, and why).
- Keep canvas behavior and documentation aligned with conversational, AI-led usage.
