"""Chat text must never be parsed as HTML or turned into script links.

Brython's ``html.SPAN(str)`` parses its string argument as HTML, so user messages,
tool names, arguments and results are set with ``.text``. These tests restore a
chat carrying HTML payloads (as a tampered workspace file could) and render the
same payloads live, then check that no element was created from them. They also
check that MathJax does not load its html extension, whose ``\\href`` would make
``javascript:`` links out of chat math.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from browser import window

from tool_call_log_manager import ToolCallLogManager
from .simple_mock import get_class_attr
from .test_chat_persistence import _DetachedChatUI

IMG_PAYLOAD = '<img src=x onerror="window.__xss=1">'
BOLD_PAYLOAD = "<b>x</b>"
PAYLOADS = (BOLD_PAYLOAD, IMG_PAYLOAD)


def _children_with_class(parent: Any, class_name: str) -> List[Any]:
    return [child for child in parent.children if class_name in get_class_attr(child).split()]


def _xss_flag() -> Any:
    return getattr(window, "__xss", None)


class TestChatHtmlSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = _DetachedChatUI()

    def _assert_literal_text(self, element: Any, expected: str) -> None:
        self.assertEqual(len(element.children), 0, f"markup was parsed: {element.innerHTML}")
        self.assertEqual(element.text, expected)

    def _restored_chat(self, payload: str) -> Dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "text": f"hello {payload}"},
                {
                    "role": "assistant",
                    "text": "Done.",
                    "tools": [{"name": f"tool{payload}", "args": f"x: {payload}", "error": True}],
                },
            ],
            "truncated": 0,
        }

    def test_restored_user_text_and_tool_entries_stay_text(self) -> None:
        for payload in PAYLOADS:
            with self.subTest(payload=payload):
                self.ui.restore_transcript(self._restored_chat(payload))
                user_el, ai_el = _children_with_class(self.ui.history, "chat-message")

                self._assert_literal_text(_children_with_class(user_el, "chat-content")[0], f"hello {payload}")

                log = _children_with_class(ai_el, "tool-call-log-dropdown")[0]
                content = _children_with_class(log, "tool-call-log-content")[0]
                entry = _children_with_class(content, "tool-call-entry")[0]
                self._assert_literal_text(_children_with_class(entry, "tool-call-name")[0], f"tool{payload}")
                self._assert_literal_text(_children_with_class(entry, "tool-call-args")[0], f"(x: {payload})")
        self.assertIsNone(_xss_flag())

    def test_live_user_message_stays_text(self) -> None:
        for payload in PAYLOADS:
            with self.subTest(payload=payload):
                element = self.ui.create_message_element("User", payload)
                self._assert_literal_text(_children_with_class(element, "chat-content")[0], payload)
        self.assertIsNone(_xss_flag())

    def test_live_tool_entry_name_args_error_and_result_stay_text(self) -> None:
        log = ToolCallLogManager()
        for payload in PAYLOADS:
            with self.subTest(payload=payload):
                error_entry = log.create_entry_element(
                    {
                        "name": payload,
                        "args_display": payload,
                        "is_error": True,
                        "error_message": f"Error: {payload}",
                    }
                )
                self._assert_literal_text(_children_with_class(error_entry, "tool-call-name")[0], payload)
                self._assert_literal_text(_children_with_class(error_entry, "tool-call-args")[0], f"({payload})")
                self._assert_literal_text(
                    _children_with_class(error_entry, "tool-call-error-msg")[0], f" → Error: {payload}"
                )

                result_entry = log.create_entry_element({"name": "t", "is_error": False, "result_display": payload})
                self._assert_literal_text(_children_with_class(result_entry, "tool-call-result")[0], f" → {payload}")
        self.assertIsNone(_xss_flag())


class TestMathJaxHrefDisabled(unittest.TestCase):
    """``\\href{javascript:...}`` must not typeset to a link; ordinary math still renders."""

    def _mathjax(self) -> Any:
        mathjax = getattr(window, "MathJax", None)
        if mathjax is None or not hasattr(mathjax, "tex2chtml"):
            self.skipTest("MathJax is not ready")
        return mathjax

    def _typeset(self, tex: str) -> Any:
        try:
            return self._mathjax().tex2chtml(tex)
        except Exception as exc:
            # A synchronous typeset throws "retry" when it needs to load an extension first.
            if "retry" in str(exc).lower():
                self.skipTest(f"MathJax needed to load an extension for {tex!r}")
            raise

    def test_html_extension_is_not_autoloaded(self) -> None:
        autoload = self._mathjax().config.tex.autoload
        self.assertEqual(len(autoload.html), 0)

    def test_href_does_not_create_a_link(self) -> None:
        # No skip on "retry" here: a retry would mean MathJax wants to load the html extension.
        node = self._mathjax().tex2chtml("\\href{javascript:window.__xss=1}{\\text{click}}")
        self.assertIsNone(node.querySelector("a"))
        self.assertIsNone(node.querySelector("[href]"))

    def test_ordinary_math_still_renders(self) -> None:
        for tex in (
            "\\frac{1}{2}",
            "\\sqrt{x^2+1}",
            "\\begin{pmatrix}1 & 2 \\\\ 3 & 4\\end{pmatrix}",
            "\\text{area} = \\pi r^2",
        ):
            with self.subTest(tex=tex):
                node = self._typeset(tex)
                self.assertIsNone(node.querySelector("mjx-merror"))
                self.assertIsNone(node.querySelector("[data-mjx-error]"))
