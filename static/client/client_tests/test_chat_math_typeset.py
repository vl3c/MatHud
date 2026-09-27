"""Chat math is typeset once per formula, however long the chat grows.

Re-typesetting the whole chat after every message once nested a hidden copy of each
earlier formula (MathJax's MathML input read the ``<math>`` inside ``mjx-assistive-mml``),
so ``ChatUIManager.render_math`` typesets only newly added messages and the page reads
TeX only.
These tests drive the chat through live messages, streamed replies and a restored
transcript with real MathJax and check that the number of ``mjx-container`` elements
equals the number of formulas.

The harness is synchronous, so ``window.MatHudMathSafety`` is replaced by a stand-in
that typesets with ``MathJax.typeset`` and then runs the real sanitiser, recording
each root it was given. It also forwards ``sanitize``, which the page's MathJax render
action calls on every formula, so every container must come out marked as sanitised.
Only bundled TeX is used: formulas needing an autoloaded extension cannot be typeset
synchronously.
"""

from __future__ import annotations

import unittest
from typing import Any, List

from browser import document, html, window

from chat_ui_manager import ChatUIManager
from message_menu_manager import MessageMenuManager
from tool_call_log_manager import ToolCallLogManager

# Each string holds exactly one formula (bundled TeX, nothing to autoload).
INLINE = "\\(x^2 + 1\\)"
DISPLAY = "$$\\frac{1}{2}$$"
ROOT = "\\(\\sqrt{y}\\)"


class _SyncMathSafety:
    """Typesets synchronously, then sanitises with the real sanitiser.

    ``as_js()`` exposes it as a plain JS object: the page's MathJax render action calls
    ``window.MatHudMathSafety.sanitize`` from JavaScript, which cannot call methods of a
    Python instance. Errors raised while typesetting are recorded, since
    ``ChatUIManager.render_math`` swallows them.
    """

    def __init__(self, mathjax: Any, safety: Any) -> None:
        self._mathjax = mathjax
        self._safety = safety
        self.roots: List[Any] = []
        self.errors: List[str] = []
        self.render_action_calls = 0

    def as_js(self) -> Any:
        stub = window.Object.new()
        stub.sanitize = self.sanitize
        stub.typesetAndSanitize = self.typesetAndSanitize
        return stub

    def sanitize(self, root: Any, skip_sanitized: bool = False) -> int:
        """Called by the page's MathJax render action on each formula."""
        self.render_action_calls += 1
        return int(self._safety.sanitize(root, skip_sanitized))

    def typesetAndSanitize(self, root: Any) -> None:
        self.roots.append(root)
        try:
            self._mathjax.typeset([root])
        except Exception as error:
            self.errors.append(str(error))
            raise
        self._safety.sanitize(root, True)


class _PageChatUI(ChatUIManager):
    """ChatUIManager drawing into its own element on the page (MathJax measures it)."""

    def __init__(self, history: Any) -> None:
        super().__init__(message_menu=MessageMenuManager(), tool_call_log=ToolCallLogManager())
        self.history = history

    def _chat_history_element(self) -> Any:
        return self.history


class TestChatMathTypeset(unittest.TestCase):
    def setUp(self) -> None:
        self.mathjax = getattr(window, "MathJax", None)
        if self.mathjax is None or not hasattr(self.mathjax, "typeset"):
            self.fail("MathJax is not loaded")
        self.original_safety = getattr(window, "MatHudMathSafety", None)
        if self.original_safety is None:
            self.fail("static/math_output_sanitizer.js is not loaded")
        self.safety = _SyncMathSafety(self.mathjax, self.original_safety)
        self.history = html.DIV(style={"position": "absolute", "left": "-10000px", "width": "600px"})
        window.MatHudMathSafety = self.safety.as_js()
        # Registered right after the swap, so the stand-in never outlives the test.
        self.addCleanup(self._restore_page)
        document.body <= self.history
        self.ui = _PageChatUI(self.history)

    def _restore_page(self) -> None:
        window.MatHudMathSafety = self.original_safety
        if self.history.parentNode is not None:
            self.mathjax.typesetClear([self.history])
            self.history.remove()

    def _assert_typeset_once(self, formulas: int) -> None:
        containers = self.history.select("mjx-container")
        self.assertEqual(len(containers), formulas, "a formula was typeset more than once (or not at all)")
        self.assertEqual(self.history.select("mjx-assistive-mml mjx-container"), [])
        self.assertEqual(self.safety.errors, [])
        # The render action sanitised each new formula (the stand-in's own pass would mark them too).
        self.assertGreaterEqual(self.safety.render_action_calls, formulas)
        unsanitized = [c for c in containers if not c.hasAttribute("data-mathud-sanitized")]
        self.assertEqual(unsanitized, [], "a formula was not sanitised")

    def _assert_roots_disjoint(self) -> None:
        roots = self.safety.roots
        for i, first in enumerate(roots):
            for second in roots[i + 1 :]:
                self.assertFalse(first.contains(second) or second.contains(first), "a typeset root was typeset again")

    def test_live_messages_typeset_each_formula_once(self) -> None:
        formulas = 0
        for i in range(4):
            self.ui.print_user_message(f"step {i}: {INLINE}")
            formulas += 1
            self._assert_typeset_once(formulas)
            self.ui.print_ai_message(f"Result {ROOT} and {DISPLAY}")
            formulas += 2
            self._assert_typeset_once(formulas)
            self.ui.print_system_message(f"note {INLINE}")
            formulas += 1
            self._assert_typeset_once(formulas)
        self._assert_roots_disjoint()
        self.assertNotIn(self.history, self.safety.roots)

    def test_streamed_reply_is_typeset_once_when_it_finishes(self) -> None:
        self.ui.print_user_message(f"area of {ROOT}?")
        for token in ("The area is ", "\\(\\pi", " r^2\\)", " and ", DISPLAY):
            self.ui.on_stream_token(token)
        self._assert_typeset_once(1)  # streaming text is not typeset yet
        self.ui.print_system_message(f"meanwhile {INLINE}")
        self._assert_typeset_once(2)  # the unfinished stream is left alone
        self.ui.finalize_stream()
        self._assert_typeset_once(4)
        self.ui.print_ai_message(f"Next {INLINE}")
        self._assert_typeset_once(5)
        self._assert_roots_disjoint()

    def test_streamed_reply_with_reasoning_is_typeset_once(self) -> None:
        self.ui.print_user_message(INLINE)
        self.ui.on_stream_reasoning("Thinking about it")
        self.ui.on_stream_token(f"So {ROOT} equals {DISPLAY}")
        self.ui.finalize_stream()
        self._assert_typeset_once(3)
        self.ui.print_user_message(INLINE)
        self._assert_typeset_once(4)
        self._assert_roots_disjoint()

    def test_streamed_reply_with_tool_log_is_typeset_once(self) -> None:
        self.ui.print_user_message(INLINE)
        self.ui.ensure_stream_element()
        self.ui._tool_call_log.ensure_element(self.ui.stream_container, self.ui.stream_content)
        self.ui.on_stream_token(f"Drew {ROOT}")
        self.ui.finalize_stream()
        self._assert_typeset_once(2)
        self.ui.print_ai_message(DISPLAY)
        self._assert_typeset_once(3)
        self._assert_roots_disjoint()

    def test_restored_chat_typesets_every_message_once(self) -> None:
        self.ui.print_user_message(INLINE)
        self.ui.print_ai_message(DISPLAY)
        self._assert_typeset_once(2)
        restored = {
            "messages": [
                {"role": "user", "text": f"solve {INLINE}"},
                {"role": "assistant", "text": f"Roots {ROOT} and {DISPLAY}", "tools": [{"name": "solve", "args": ""}]},
                {"role": "user", "text": "thanks"},
                {"role": "assistant", "text": f"Also {INLINE}"},
            ],
            "truncated": 2,
        }
        self.assertEqual(self.ui.restore_transcript(restored), 4)
        self._assert_typeset_once(4)  # the earlier chat is gone; each restored formula typeset once
        self.ui.print_user_message(ROOT)
        self._assert_typeset_once(5)
