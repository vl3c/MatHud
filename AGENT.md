# MatHud Agent Direction

## Vision
- MatHud is the prototype and foundation of a mathematical heads-up display: an AI that models and explains real situations by voice and with drawings over what the user sees, on smart glasses or in VR, in 3D as well as 2D.
- Scope: practical math grown from the scientific graphing calculator tradition, plus what real-world problems need; advanced math for its own sake stays out.
- Desktop and web now, Android next, glasses and VR later; the code may be rewritten in another stack, so keep the tool contract, math-only models, canvas text state, checked math and scenario catalogue clean and stack-independent.

## AI-First Product Philosophy
- MatHud is an AI-augmented application where users communicate intent through chat.
- Agents should execute workflows end-to-end via tool calls and update the HUD canvas as the visual output.
- Do not assume gesture-driven workflows are primary; treat them as optional support paths.

## Implementation Priorities
- Prioritize deterministic execution and state integrity (atomic actions, rollback, reliable undo/redo).
- Prefer explicit, explainable action reporting (what was resolved, what changed, and why).
- Keep canvas behavior and documentation aligned with conversational, AI-led usage.

# Project Rules

The following rules MUST be followed:

- Do not add Co-Authored-By, signature lines, or AI attribution to commits, PRs, or code comments
