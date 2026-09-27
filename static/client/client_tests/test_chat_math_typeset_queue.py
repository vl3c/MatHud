"""Chat typesets run one after another, so a formula waiting on a TeX extension keeps it.

MathJax's ``typesetPromise`` is not queued. When a formula needs an extension that is
not loaded yet (``\\ce`` needs mhchem, ``\\bra`` needs braket), MathJax loads it and then
renders again whatever the latest typeset asked for. The chat typesets each message on
its own, so a second message typeset during the load left the first formula as an
undefined command (``\\ce`` in red). ``window.MatHudMathSafety.typesetAndSanitize``
chains each typeset on the previous one.

The harness runs tests synchronously, so no promise settles during a test. The two
typesets run in ``prepare_async``, which the runner awaits before the suite
(``window.startMatHudTests`` and the Run Tests button); the tests check what it recorded.
"""

from __future__ import annotations

import unittest
from typing import Any, ClassVar, Dict, List, Optional

from browser import aio, document, window

# (extension, formula needing it, a piece of its correct MathML)
AUTOLOADED = (
    ("mhchem", "\\ce{H2O}", "<msub>"),
    ("braket", "\\bra{\\psi}", "&#x27E8;"),
)
BUNDLED = ("\\frac{a}{b}", "<mfrac>")
TIMEOUT_S = 10.0


def _holder(tex: str) -> Any:
    holder = document.createElement("div")
    holder.style.position = "absolute"
    holder.style.left = "-10000px"
    holder.innerHTML = f"\\({tex}\\)"
    document.body.appendChild(holder)
    return holder


class TestChatMathTypesetQueue(unittest.TestCase):
    # Set by prepare_async; None means it never ran (a synchronous-only run).
    outcome: ClassVar[Optional[Dict[str, Any]]] = None

    @classmethod
    async def prepare_async(cls) -> None:
        cls.outcome = await cls._typeset_back_to_back()

    @classmethod
    async def _typeset_back_to_back(cls) -> Dict[str, Any]:
        mathjax = getattr(window, "MathJax", None)
        safety = getattr(window, "MatHudMathSafety", None)
        if mathjax is None or not hasattr(mathjax, "typesetPromise"):
            return {"error": "MathJax is not loaded"}
        if safety is None:
            return {"error": "static/math_output_sanitizer.js is not loaded"}
        loaded = mathjax._.input.tex
        pending = [entry for entry in AUTOLOADED if not hasattr(loaded, entry[0])]
        if not pending:
            return {"skip": "every candidate TeX extension is already loaded (reload the page)"}
        extension, tex, expected = pending[0]
        holders = [_holder(tex), _holder(BUNDLED[0])]
        settled: List[int] = []
        rejected: List[str] = []
        try:
            # Back to back: the second starts while the first waits for the extension.
            for index, holder in enumerate(holders):
                safety.typesetAndSanitize(holder).then(
                    lambda _count, index=index: settled.append(index),
                    lambda error: rejected.append(str(error)),
                )
            waited = 0.0
            while len(settled) + len(rejected) < len(holders) and waited < TIMEOUT_S:
                await aio.sleep(0.05)
                waited += 0.05
            return {
                "extension": extension,
                "expected": [expected, BUNDLED[1]],
                "settled": settled,
                "rejected": rejected,
                "mathml": [cls._mathml(mathjax, holder) for holder in holders],
                "sanitized": [
                    len(holder.querySelectorAll("mjx-container:not([data-mathud-sanitized])")) == 0
                    for holder in holders
                ],
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
        self.assertEqual(self.result["settled"], [0, 1], "a typeset did not finish, or finished out of order")

    def test_each_formula_renders_without_an_undefined_command(self) -> None:
        for mathml, expected in zip(self.result["mathml"], self.result["expected"]):
            self.assertEqual(len(mathml), 1, "the formula was not typeset exactly once")
            # An undefined command is shown as its name, backslash included.
            self.assertNotIn("\\", mathml[0], f"{self.result['extension']}: {mathml[0]}")
            self.assertNotIn("merror", mathml[0])
            self.assertIn(expected, mathml[0])

    def test_every_formula_is_sanitised(self) -> None:
        self.assertEqual(self.result["sanitized"], [True, True])
