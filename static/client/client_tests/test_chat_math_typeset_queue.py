"""Chat typesets run one after another, so a formula waiting on a TeX extension keeps it.

MathJax's ``typesetPromise`` is not queued. When a formula needs an extension that is
not loaded yet (``\\ce`` needs mhchem, ``\\bra`` needs braket), MathJax loads it and then
renders again whatever the latest typeset asked for. The chat typesets each message on
its own, so a second message typeset during the load left the first formula as an
undefined command (``\\ce`` in red). ``window.MatHudMathSafety.typesetAndSanitize``
chains each typeset on the previous one.

The harness runs tests synchronously, so no promise settles during a test. The
typesets run in ``prepare_async``, which the runner awaits before the suite
(``window.startMatHudTests`` and the Run Tests button); the tests check what it recorded.
"""

from __future__ import annotations

import unittest
from typing import Any, ClassVar, Dict, List, Optional

from browser import aio, document, window

from chat_ui_manager import ChatUIManager
from message_menu_manager import MessageMenuManager
from tool_call_log_manager import ToolCallLogManager

# (extension, formula needing it, a piece of its correct MathML)
AUTOLOADED = (
    ("mhchem", "\\ce{H2O}", "<msub>"),
    ("braket", "\\bra{\\psi}", "&#x27E8;"),
)
BUNDLED = ("\\frac{a}{b}", "<mfrac>")
TIMEOUT_MS = 10000


def _holder(tex: str) -> Any:
    holder = document.createElement("div")
    holder.style.position = "absolute"
    holder.style.left = "-10000px"
    holder.innerHTML = f"\\({tex}\\)"
    document.body.appendChild(holder)
    return holder


class TestChatMathTypesetQueue(unittest.TestCase):
    # Set by prepare_async and cleared after the tests; None means it did not run
    # (a synchronous-only run: the AI tool or the slash command).
    outcome: ClassVar[Optional[Dict[str, Any]]] = None
    prepared_in_page: ClassVar[int] = 0

    @classmethod
    async def prepare_async(cls) -> None:
        first_in_page = cls.prepared_in_page == 0
        cls.prepared_in_page += 1
        try:
            cls.outcome = await cls._typeset_back_to_back(first_in_page)
        except Exception as exc:
            cls.outcome = {"error": f"preparation failed: {exc!r}"}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.outcome = None

    @classmethod
    async def _typeset_back_to_back(cls, first_in_page: bool) -> Dict[str, Any]:
        mathjax = getattr(window, "MathJax", None)
        safety = getattr(window, "MatHudMathSafety", None)
        if mathjax is None or not hasattr(mathjax, "typesetPromise"):
            return {"error": "MathJax is not loaded"}
        if safety is None:
            return {"error": "static/math_output_sanitizer.js is not loaded"}
        loaded = mathjax._.input.tex
        pending = [entry for entry in AUTOLOADED if not hasattr(loaded, entry[0])]
        if not pending:
            if first_in_page:
                # Preloaded (templates/index.html) or used by the page before the tests.
                return {"error": "mhchem and braket are loaded before the tests: pick autoloaded extensions"}
            return {"skip": "every candidate TeX extension was loaded by an earlier run (reload the page)"}
        extension, tex, expected = pending[0]
        # The third is taken off the page before its turn, as when the chat is cleared.
        holders = [_holder(tex), _holder(BUNDLED[0]), _holder(BUNDLED[0])]
        settled: List[int] = []
        counts: List[int] = [-1, -1, -1]
        rejected: List[str] = []

        def on_settled(count: Any, index: int) -> None:
            counts[index] = int(count)
            settled.append(index)

        try:
            # Back to back: the others start while the first waits for the extension.
            for index, holder in enumerate(holders):
                safety.typesetAndSanitize(holder).then(
                    lambda count, index=index: on_settled(count, index),
                    lambda error: rejected.append(str(error)),
                )
            holders[2].remove()
            started = window.performance.now()
            while len(settled) + len(rejected) < len(holders) and window.performance.now() - started < TIMEOUT_MS:
                await aio.sleep(0.05)
            return {
                "extension": extension,
                "expected": [expected, BUNDLED[1]],
                "settled": settled,
                "rejected": rejected,
                "mathml": [cls._mathml(mathjax, holder) for holder in holders[:2]],
                "sanitized": [
                    len(holder.querySelectorAll("mjx-container:not([data-mathud-sanitized])")) == 0
                    for holder in holders[:2]
                ],
                "removed_count": counts[2],
                "removed_items": len(cls._mathml(mathjax, holders[2])),
                "removed_containers": len(holders[2].querySelectorAll("mjx-container")),
            }
        finally:
            mathjax.typesetClear(holders)
            for holder in holders:
                holder.remove()

    @staticmethod
    def _mathml(mathjax: Any, holder: Any) -> List[str]:
        items = mathjax.startup.document.getMathItemsWithin([holder])
        return [str(mathjax.startup.toMML(item.root)) for item in items]

    def setUp(self) -> None:
        outcome = type(self).outcome
        if outcome is None:
            self.skipTest("needs the async preparation (window.startMatHudTests or the Run Tests button)")
        if "skip" in outcome:
            self.skipTest(outcome["skip"])
        if "error" in outcome:
            self.fail(outcome["error"])
        self.result = outcome

    def test_typesets_finish_in_order_without_rejecting(self) -> None:
        self.assertEqual(self.result["rejected"], [])
        self.assertEqual(self.result["settled"], [0, 1, 2], "a typeset did not finish, or finished out of order")

    def test_each_formula_renders_without_an_undefined_command(self) -> None:
        for mathml, expected in zip(self.result["mathml"], self.result["expected"]):
            self.assertEqual(len(mathml), 1, "the formula was not typeset exactly once")
            # An undefined command is shown as its name, backslash included.
            self.assertNotIn("\\", mathml[0], f"{self.result['extension']}: {mathml[0]}")
            self.assertNotIn("merror", mathml[0])
            self.assertIn(expected, mathml[0])

    def test_every_formula_is_sanitised(self) -> None:
        self.assertEqual(self.result["sanitized"], [True, True])

    def test_a_root_removed_while_queued_is_not_typeset(self) -> None:
        # MathJax would keep records of its math that no typesetClear of the chat reaches.
        self.assertEqual(self.result["removed_count"], 0)
        self.assertEqual(self.result["removed_items"], 0)
        self.assertEqual(self.result["removed_containers"], 0)


class _FakeHistory:
    def __init__(self, scroll_top: float, scroll_height: float, client_height: float = 100) -> None:
        self.scrollTop = scroll_top
        self.scrollHeight = scroll_height
        self.clientHeight = client_height


class _FakeChatUI(ChatUIManager):
    def __init__(self, history: _FakeHistory) -> None:
        super().__init__(message_menu=MessageMenuManager(), tool_call_log=ToolCallLogManager())
        self.history = history

    def _chat_history_element(self) -> Any:
        return self.history


class TestChatScrollAfterTypeset(unittest.TestCase):
    """The scroll once a queued typeset finishes (``_scroll_to_bottom_unless_scrolled_up``)."""

    def _scroll(self, history: _FakeHistory, top: float) -> float:
        _FakeChatUI(history)._scroll_to_bottom_unless_scrolled_up(top)
        return history.scrollTop

    def test_scrolls_to_the_end_when_the_message_grew(self) -> None:
        self.assertEqual(self._scroll(_FakeHistory(900, 1300), top=900), 1300)

    def test_scrolls_when_the_chat_was_scrolled_further_down_since(self) -> None:
        self.assertEqual(self._scroll(_FakeHistory(1000, 1300), top=900), 1300)

    def test_keeps_the_position_when_the_user_scrolled_up(self) -> None:
        self.assertEqual(self._scroll(_FakeHistory(400, 1300), top=900), 400)

    def test_scrolls_when_the_chat_got_shorter_and_is_at_the_end(self) -> None:
        self.assertEqual(self._scroll(_FakeHistory(600, 700), top=900), 700)
